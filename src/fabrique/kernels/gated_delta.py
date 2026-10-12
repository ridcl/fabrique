"""Fused chunk recurrence of the gated delta rule, as Pallas (Triton) kernels.

The chunked gated delta rule (``qwen3_5.model.chunk_gated_delta_rule``) splits
into a parallel part -- per-chunk matrices built with batched matmuls -- and a
*sequential* part, the state recurrence over chunks::

    v_new[c] = U[c] - W[c] @ S[c]                        # [C, Dv]
    S[c + 1] = a[c] * S[c] + kd[c].T @ v_new[c]          # [Dk, Dv]

Under ``lax.scan`` that recurrence costs ~16 tiny kernel launches per chunk,
per layer, forward and backward: ~113k launches per training step at 8k tokens,
the single largest cost in the step.  Here each Pallas program keeps its slice
of ``S`` on-chip and loops over all chunks inside one kernel; the backward pass
is a second kernel running the adjoint recurrence in reverse.  Only the truly
sequential work lives in the kernels; the rest of the backward (dW, dkd, da)
is batched matmuls over all chunks once the state-gradient sequence is known.

Work is split over (batch * head, value-column block): every quantity in the
recurrence is separable over the value dimension, so a program owns columns
``S[:, j*bv:(j+1)*bv]``.  Measured on an RTX 3090 for one Qwen3.5-0.8B layer at
8k tokens (16 heads, Dk = Dv = 128): 0.93 ms forward, 2.9 ms forward+backward,
against ~20 ms for the lax.scan recurrence.  block_v=32, num_warps=4 was the
fastest of {16, 32, 64} x {2, 4, 8}.

Shapes (B = batch * heads, N = num_chunks * C)::

    u      [B, N, Dv]        w, kd  [B, N, Dk]      a [B, num_chunks]
    s0     [B, Dk, Dv]
    ->  states [B, num_chunks, Dk, Dv]  (state *before* each chunk)
        v_new  [B, N, Dv]
        final  [B, Dk, Dv]

Numerics follow the lax.scan path: float32 storage, matmuls in TF32 (the GPU
default for float32 dots, which is also what the scan path gets).

Under data parallelism call it through ``fabrique.kernels.sharding
.per_batch_shard`` (``chunk_gated_delta_rule`` does): XLA cannot partition a
Pallas kernel, and unwrapped it all-gathers the batch onto every device.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

F32 = jnp.float32

# TF32 matmuls, like the lax.scan path gets by default.  Tests flip this off to
# check the kernels against the reference in exact float32.
ALLOW_TF32: bool = True


def _dot(a, b, **kw):
    return pl.dot(a, b, allow_tf32=ALLOW_TF32, **kw)


def _fwd_kernel(u_ref, w_ref, kd_ref, a_ref, s0_ref, st_ref, vn_ref, sf_ref, *, C, nc):
    def body(c, s):
        sl = pl.dslice(c * C, C)
        st_ref[c] = s
        vn = u_ref[sl, :] - _dot(w_ref[sl, :], s)  # [C, bv]
        vn_ref[sl, :] = vn
        return a_ref[c] * s + _dot(kd_ref[sl, :], vn, trans_a=True)  # [Dk, bv]

    sf_ref[...] = lax.fori_loop(0, nc, body, s0_ref[...].astype(F32))


def _bwd_kernel(
    w_ref, kd_ref, a_ref, dvn_ref, dst_ref, dsf_ref,  # inputs
    du_ref, gs_ref, ds0_ref,  # outputs
    *, C, nc,
):  # fmt: skip
    """Only the sequential part of the backward: dv_new and dL/dS[c + 1].

    Everything else (dW, dkd, da) is a batched matmul over all chunks once the
    G sequence is known, done in JAX on tensor cores -- cheaper than per-block
    partial outputs written from here and reduced afterwards.
    """

    def body(i, g):  # g = dL/dS[c + 1]
        c = nc - 1 - i
        sl = pl.dslice(c * C, C)
        gs_ref[c] = g
        dvn = dvn_ref[sl, :] + _dot(kd_ref[sl, :], g)  # [C, bv]
        du_ref[sl, :] = dvn
        return a_ref[c] * g - _dot(w_ref[sl, :], dvn, trans_a=True) + dst_ref[c]

    ds0_ref[...] = lax.fori_loop(0, nc, body, dsf_ref[...].astype(F32))


def _check(C, dk, dv, bv):
    for name, n in (("chunk_size", C), ("key dim", dk), ("value block", bv)):
        if n < 16 or n & (n - 1):
            raise ValueError(f"{name} must be a power of 2 >= 16 for tl.dot, got {n}")
    if dv % bv:
        raise ValueError(f"value dim {dv} must be a multiple of block_v {bv}")


def _forward(u, w, kd, a, s0, *, chunk_size, block_v, num_warps):
    B, N, dv = u.shape
    dk = w.shape[-1]
    C, bv = chunk_size, block_v
    nc = N // C
    _check(C, dk, dv, bv)
    kernel = functools.partial(_fwd_kernel, C=C, nc=nc)
    col = lambda i, j: (i, 0, j)  # noqa: E731
    return pl.pallas_call(
        kernel,
        grid=(B, dv // bv),
        in_specs=[
            pl.BlockSpec((None, N, bv), col),  # u
            pl.BlockSpec((None, N, dk), lambda i, j: (i, 0, 0)),  # w
            pl.BlockSpec((None, N, dk), lambda i, j: (i, 0, 0)),  # kd
            pl.BlockSpec((None, nc), lambda i, j: (i, 0)),  # a
            pl.BlockSpec((None, dk, bv), col),  # s0
        ],
        out_specs=[
            pl.BlockSpec((None, nc, dk, bv), lambda i, j: (i, 0, 0, j)),  # states
            pl.BlockSpec((None, N, bv), col),  # v_new
            pl.BlockSpec((None, dk, bv), col),  # final
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, nc, dk, dv), F32),
            jax.ShapeDtypeStruct((B, N, dv), F32),
            jax.ShapeDtypeStruct((B, dk, dv), F32),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        name="gated_delta_fwd",
    )(u, w, kd, a, s0)


def _backward(w, kd, a, states, vn, dvn, dst, dsf, *, chunk_size, block_v, num_warps):
    B, N, dk = w.shape
    dv = vn.shape[-1]
    C, bv = chunk_size, block_v
    nc = N // C
    kernel = functools.partial(_bwd_kernel, C=C, nc=nc)
    col = lambda i, j: (i, 0, j)  # noqa: E731
    full = lambda i, j: (i, 0, 0)  # noqa: E731
    per_chunk = lambda i, j: (i, 0, 0, j)  # noqa: E731
    du, gs, ds0 = pl.pallas_call(
        kernel,
        grid=(B, dv // bv),
        in_specs=[
            pl.BlockSpec((None, N, dk), full),  # w
            pl.BlockSpec((None, N, dk), full),  # kd
            pl.BlockSpec((None, nc), lambda i, j: (i, 0)),  # a
            pl.BlockSpec((None, N, bv), col),  # d v_new (from the outputs)
            pl.BlockSpec((None, nc, dk, bv), per_chunk),  # d states
            pl.BlockSpec((None, dk, bv), col),  # d final
        ],
        out_specs=[
            pl.BlockSpec((None, N, bv), col),  # du = total d v_new
            pl.BlockSpec((None, nc, dk, bv), per_chunk),  # G[c] = dL/dS[c + 1]
            pl.BlockSpec((None, dk, bv), col),  # ds0
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, N, dv), F32),
            jax.ShapeDtypeStruct((B, nc, dk, dv), F32),
            jax.ShapeDtypeStruct((B, dk, dv), F32),
        ],
        compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=1),
        name="gated_delta_bwd",
    )(w, kd, a, dvn, dst, dsf)
    # The parallel remainder, batched over all chunks:
    #   dkd[c] = v_new[c] @ G[c].T     dW[c] = -dv_new[c] @ S[c].T
    #   da[c]  = sum(S[c] * G[c])
    chunks = lambda t: t.reshape(B, nc, C, t.shape[-1])  # noqa: E731
    hp = jax.lax.Precision.DEFAULT if ALLOW_TF32 else jax.lax.Precision.HIGHEST
    dkd = jnp.einsum("bcie,bcke->bcik", chunks(vn), gs, precision=hp).reshape(B, N, dk)
    dw = -jnp.einsum("bcie,bcke->bcik", chunks(du), states, precision=hp).reshape(
        B, N, dk
    )
    da = jnp.einsum("bcke,bcke->bc", states, gs, precision=hp)
    return du, dw, dkd, da, ds0


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def recurrence(u, w, kd, a, s0, chunk_size=64, block_v=32, num_warps=4):
    """Fused gated-delta chunk recurrence; see the module docstring for shapes."""
    return _forward(
        u, w, kd, a, s0, chunk_size=chunk_size, block_v=block_v, num_warps=num_warps
    )


def _recurrence_fwd(u, w, kd, a, s0, chunk_size, block_v, num_warps):
    out = _forward(
        u, w, kd, a, s0, chunk_size=chunk_size, block_v=block_v, num_warps=num_warps
    )
    states, vn, _ = out
    return out, (w, kd, a, states, vn)


def _recurrence_bwd(chunk_size, block_v, num_warps, res, cts):
    w, kd, a, states, vn = res
    dst, dvn, dsf = cts  # custom_vjp materialises zero cotangents for us
    du, dw, dkd, da, ds0 = _backward(
        w, kd, a, states, vn, dvn.astype(F32), dst.astype(F32), dsf.astype(F32),
        chunk_size=chunk_size, block_v=block_v, num_warps=num_warps,
    )  # fmt: skip
    return du, dw, dkd, da, ds0


recurrence.defvjp(_recurrence_fwd, _recurrence_bwd)


def recurrence_reference(u, w, kd, a, s0, chunk_size=64):
    """The same recurrence as a lax.scan -- the CPU path and the test oracle."""
    B, N, dv = u.shape
    nc = N // chunk_size
    chunks = lambda t: t.reshape(B, nc, chunk_size, t.shape[-1]).swapaxes(0, 1)  # noqa: E731

    def step(s, xs):
        u_c, w_c, kd_c, a_c = xs
        vn = u_c - w_c @ s
        return a_c[:, None, None] * s + jnp.swapaxes(kd_c, -1, -2) @ vn, (s, vn)

    final, (states, vn) = lax.scan(
        step, s0.astype(F32), (chunks(u), chunks(w), chunks(kd), a.T)
    )
    return states.swapaxes(0, 1), vn.swapaxes(0, 1).reshape(B, N, dv), final
