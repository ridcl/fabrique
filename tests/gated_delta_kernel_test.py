"""Fused gated-delta recurrence kernel (fabrique.kernels.gated_delta).

Checked in exact float32 against the lax.scan reference, forward and backward,
so a wrong adjoint shows up as ~1e-2 rather than hiding under TF32's ~3e-4.
GPU-only: Pallas' Triton backend does not run on CPU.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from fabrique.kernels import gated_delta as G
from fabrique.models.qwen3_5 import model as M

pytestmark = pytest.mark.skipif(
    M._select_recurrence(None, 64, 128, 128) != "pallas", reason="needs an sm80+ GPU"
)


@pytest.fixture
def exact_float32():
    old = G.ALLOW_TF32
    G.ALLOW_TF32 = False
    with jax.default_matmul_precision("highest"):
        yield
    G.ALLOW_TF32 = old


def _rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.maximum(jnp.abs(b).max(), 1e-30))


def _problem(batch, num_chunks, chunk, dk, dv, seed=0):
    n = num_chunks * chunk
    ks = jax.random.split(jax.random.key(seed), 8)
    args = (
        jax.random.normal(ks[0], (batch, n, dv)) * 0.5,  # u
        jax.random.normal(ks[1], (batch, n, dk)) * 0.05,  # w
        jax.random.normal(ks[2], (batch, n, dk)) * 0.05,  # kd
        jax.nn.sigmoid(jax.random.normal(ks[3], (batch, num_chunks)) + 3),  # decay
        jax.random.normal(ks[4], (batch, dk, dv)) * 0.1,  # nonzero initial state
    )
    cotangents = (
        jax.random.normal(ks[5], (batch, num_chunks, dk, dv)),
        jax.random.normal(ks[6], (batch, n, dv)),
        jax.random.normal(ks[7], (batch, dk, dv)),
    )
    return args, cotangents


@pytest.mark.parametrize(
    "batch,num_chunks,chunk,dk,dv",
    [
        (2, 1, 64, 128, 128),
        (3, 7, 64, 128, 128),
        (2, 4, 128, 128, 128),
        (1, 5, 64, 64, 64),
    ],
)
def test_matches_reference_exactly(exact_float32, batch, num_chunks, chunk, dk, dv):
    args, cts = _problem(batch, num_chunks, chunk, dk, dv)

    def loss(fn):
        return lambda *x: sum(jnp.sum(o * c) for o, c in zip(fn(*x), cts, strict=True))

    kernel = lambda *x: G.recurrence(*x, chunk)  # noqa: E731
    ref = lambda *x: G.recurrence_reference(*x, chunk_size=chunk)  # noqa: E731
    for got, want in zip(jax.jit(kernel)(*args), jax.jit(ref)(*args), strict=True):
        assert _rel(got, want) < 1e-5
    g_got = jax.jit(jax.grad(loss(kernel), argnums=range(5)))(*args)
    g_ref = jax.jit(jax.grad(loss(ref), argnums=range(5)))(*args)
    for name, got, want in zip(
        ("u", "w", "kd", "decay", "s0"), g_got, g_ref, strict=True
    ):
        assert _rel(got, want) < 1e-5, f"d{name}"


def test_chunked_rule_kernel_matches_scan():
    """The integrated path: same outputs, state and input gradients as lax.scan."""
    ks = jax.random.split(jax.random.key(1), 6)
    L, H, D = 512, 4, 128
    q, k, v = (jax.random.normal(ks[i], (2, L, H, D), jnp.bfloat16) for i in range(3))
    g = -jax.nn.softplus(jax.random.normal(ks[3], (2, L, H)))
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (2, L, H)))
    w = jax.random.normal(ks[5], (2, L, H, D))

    def run(impl):
        def loss(q, k, v):
            o, s = M.chunk_gated_delta_rule(q, k, v, g, beta, recurrence_impl=impl)
            return jnp.sum(o.astype(jnp.float32) * w) + jnp.sum(s), (o, s)

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
            q, k, v
        )

    (_, (o_k, s_k)), g_k = run("pallas")
    (_, (o_s, s_s)), g_s = run("scan")
    f32 = lambda t: t.astype(jnp.float32)  # noqa: E731
    assert _rel(f32(o_k), f32(o_s)) < 1e-2  # bf16 outputs
    assert _rel(s_k, s_s) < 1e-3  # float32 state, TF32 matmuls on both sides
    for got, want in zip(g_k, g_s, strict=True):
        assert _rel(f32(got), f32(want)) < 2e-2


def test_program_size_is_independent_of_length():
    """The kernel path must not unroll anything per chunk or token either."""

    def size(L):
        q = jnp.zeros((1, L, 4, 128), jnp.bfloat16)
        z = jnp.zeros((1, L, 4))

        def f(q):
            out = M.chunk_gated_delta_rule(q, q, q, z, z, recurrence_impl="pallas")[0]
            return jnp.sum(out.astype(jnp.float32))

        return _count_eqns(jax.make_jaxpr(jax.grad(f))(q).jaxpr)

    assert size(1024) == size(8192)


def _count_eqns(jaxpr) -> int:
    """Equations including nested jaxprs (scan / custom_vjp / pjit bodies)."""
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for p in eqn.params.values():
            for sub in p if isinstance(p, (list, tuple)) else (p,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    n += _count_eqns(inner)
    return n


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 GPUs")
def test_kernel_is_sharded_not_gathered_under_a_mesh():
    """Without shard_map XLA all-gathers the inputs and each GPU does all of it."""
    mesh = jax.make_mesh(
        (2,),
        ("data",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=jax.devices()[:2],
    )
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P("data")))  # noqa: E731
    ks = jax.random.split(jax.random.key(2), 5)
    q, k, v = (
        put(jax.random.normal(ks[i], (2, 256, 4, 128), jnp.bfloat16)) for i in range(3)
    )
    g = put(-jax.nn.softplus(jax.random.normal(ks[3], (2, 256, 4))))
    beta = put(jax.nn.sigmoid(jax.random.normal(ks[4], (2, 256, 4))))

    def loss(q, k, v, impl):
        out = M.chunk_gated_delta_rule(q, k, v, g, beta, recurrence_impl=impl)[0]
        return jnp.sum(out.astype(jnp.float32))

    with jax.set_mesh(mesh):
        fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)), static_argnums=3)
        hlo = fn.lower(q, k, v, "pallas").compile().as_text() or ""
        got = fn(q, k, v, "pallas")
        want = fn(q, k, v, "scan")
    assert not re.search(r"all-gather", hlo)
    for a, b in zip(got, want, strict=True):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        assert np.abs(a - b).max() / np.abs(b).max() < 2e-2
