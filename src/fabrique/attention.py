"""Device-aware scaled dot-product attention.

One entry point, :func:`dot_product_attention`, that picks the fastest kernel
the current device can run and falls back gracefully:

==========  ===============================================  =================
backend     when                                             memory
==========  ===============================================  =================
``cudnn``   GPU, bf16/fp16, head_dim <= 128 (<= 256 on        O(seq)
            Hopper+), cuDNN actually loads
``pallas``  GPU sm80+, bf16/fp16, causal, head_dim <= 256,    O(seq)
            when cuDNN cannot (e.g. head_dim 256 on Ampere)
``xla``     everything else (CPU, float32 parity runs, ...)  O(seq^2)
==========  ===============================================  =================

``jax.nn.dot_product_attention`` alone does not get you there: its default is
the XLA kernel, which materialises the full logits (2 GiB per tensor for
Qwen3.5 at 8k tokens), and JAX/XLA's cuDNN path rejects head_dim > 128 below
Hopper -- both in JAX's wrapper and again in XLA's ``cuda_dnn.cc``.  JAX's own
Pallas/Triton flash attention does run head_dim 256 on Ampere, given tiles
small enough for its shared memory.

Data parallelism: XLA cannot partition the Pallas kernel, so under a
multi-device ``jit`` it all-gathers q/k/v and every device computes attention
for the *whole* batch (measured: 3.1x slower on 2 GPUs).  When a mesh is set
with ``jax.set_mesh(mesh)``, the Pallas call is wrapped in ``shard_map`` so each
device processes only its own batch shard.  Set the mesh for multi-device
training.  cuDNN needs no such wrapper: JAX ships partitioning rules for it.
"""

from __future__ import annotations

import functools
import logging
from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

BACKENDS = ("cudnn", "pallas", "xla")


# ---------------------------------------------------------------------------
# Device capabilities
# ---------------------------------------------------------------------------


@functools.cache
def _compute_capability() -> tuple[int, int] | None:
    try:
        d = jax.devices()[0]
    except RuntimeError:
        return None
    if d.platform != "gpu" or not hasattr(d, "compute_capability"):
        return None
    major, minor = (int(x) for x in str(d.compute_capability).split("."))
    return major, minor


@functools.cache
def cudnn_attention_available() -> bool:
    """Whether jax.nn.dot_product_attention's cuDNN backend actually works here.

    Probed rather than inferred: a cuDNN built for a different CUDA major
    version installs over the expected one and then fails every shape check at
    runtime, which is invisible from the package list.
    """
    if _compute_capability() is None:
        return False
    probe = jnp.zeros((1, 8, 2, 8), jnp.bfloat16)
    try:
        jax.block_until_ready(
            jax.jit(
                lambda q, k, v: jax.nn.dot_product_attention(
                    q, k, v, implementation="cudnn"
                )
            )(probe, probe, probe)
        )
    except Exception:
        return False
    return True


def cudnn_max_head_dim() -> int:
    """JAX's own limit (fused_attention_stablehlo.check_is_flash_attention)."""
    cc = _compute_capability()
    return 256 if cc is not None and cc >= (9, 0) else 128


def _pallas_block_sizes(head_dim: int):
    """Tiles that fit the GPU's shared memory.

    Measured on an RTX 3090 (sm86, ~99 KiB/block) at head_dim 256: 64x32 runs,
    64x64 requests 104 KiB and fails.  A100 (sm80, 164 KiB) and Hopper (228
    KiB) take the larger tile -- an assumption, not yet measured.  JAX's default
    (128x128) is tuned for those big-shared-memory parts.
    """
    from jax.experimental.pallas.ops.gpu import attention as pa

    cc = _compute_capability() or (0, 0)
    small_smem = cc in ((8, 6), (8, 7), (8, 9))  # consumer Ampere / Ada
    block_k = 32 if (small_smem and head_dim > 128) else 64
    return pa.BlockSizes(
        block_q=64,
        block_k=block_k,
        block_q_dkv=32,
        block_kv_dkv=32,
        block_q_dq=32,
        block_kv_dq=32,
    )


def select_backend(
    dtype: jnp.dtype, head_dim: int, *, causal: bool, implementation: str | None = None
) -> str:
    """The backend :func:`dot_product_attention` will use for these inputs."""
    if implementation is not None:
        if implementation not in BACKENDS:
            raise ValueError(f"implementation must be one of {BACKENDS}")
        return implementation
    low_precision = jnp.dtype(dtype) in (
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float16),
    )
    if not low_precision or head_dim % 8:
        return "xla"
    if head_dim <= cudnn_max_head_dim() and cudnn_attention_available():
        return "cudnn"
    cc = _compute_capability()
    if causal and cc is not None and cc >= (8, 0) and head_dim <= 256:
        return "pallas"
    return "xla"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _dense_mask(seq_q: int, seq_k: int, causal: bool, padding_mask):
    """[B or 1, 1, T, S] boolean mask for the cuDNN / XLA kernels, or None."""
    mask = None
    if causal:
        mask = (jnp.arange(seq_k)[None, :] <= jnp.arange(seq_q)[:, None])[None, None]
    if padding_mask is not None:
        pad = padding_mask.astype(jnp.bool_)[:, None, None, :]  # keys only
        mask = pad if mask is None else mask & pad
    return mask


def _pallas_attention(q, k, v, padding_mask, scale):
    from jax.experimental.pallas.ops.gpu import attention as pa

    _, seq_len, num_heads, head_dim = q.shape
    rep = num_heads // k.shape[2]
    if rep > 1:  # GQA -> MHA; the kernel wants equal head counts
        k, v = jnp.repeat(k, rep, axis=2), jnp.repeat(v, rep, axis=2)
    bs = _pallas_block_sizes(head_dim)
    # The kernel needs seq_len to be a multiple of its tiles.  Padding at the
    # end is exact under causal masking: no real query sees a later position.
    blk = max(bs.block_q, bs.block_k)
    pad = (-seq_len) % blk
    seg = None if padding_mask is None else padding_mask.astype(jnp.int32)
    if pad:
        widths = ((0, 0), (0, pad), (0, 0), (0, 0))
        q, k, v = (jnp.pad(t, widths) for t in (q, k, v))
        if seg is not None:
            seg = jnp.pad(seg, ((0, 0), (0, pad)))
    out = pa.mha(
        q,
        k,
        v,
        seg,  # tokens attend only within their segment: pads (0) vs real (1)
        sm_scale=scale,
        causal=True,
        block_sizes=bs,
        num_warps=4,
    )
    return out[:, :seq_len]


def _shard_over_mesh(fn, n_args: int):
    """Run ``fn`` per batch shard when a mesh is set (see module docstring)."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return fn
    batch = P(mesh.axis_names)
    return jax.shard_map(
        fn,
        in_specs=(batch,) * n_args,
        out_specs=batch,
        # The Pallas op does not annotate how its outputs vary over mesh axes.
        check_vma=False,
    )


def dot_product_attention(
    q: jaxtyping.Array,  # [B, T, N, H]
    k: jaxtyping.Array,  # [B, S, K, H], N % K == 0
    v: jaxtyping.Array,  # [B, S, K, H]
    *,
    causal: bool,
    padding_mask: jaxtyping.Array | None = None,  # [B, S], True = real token
    scale: float | None = None,
    implementation: str | None = None,
) -> jaxtyping.Array:
    """Attention with causal + key-padding masking, on the best kernel available.

    ``padding_mask`` masks *keys*; padded *queries* produce garbage rows, which
    callers ignore (their loss/outputs are masked anyway).
    """
    head_dim = q.shape[-1]
    scale = head_dim**-0.5 if scale is None else scale
    backend = select_backend(
        q.dtype, head_dim, causal=causal, implementation=implementation
    )

    if backend == "pallas":
        if padding_mask is None:
            fn = _shard_over_mesh(
                lambda q, k, v: _pallas_attention(q, k, v, None, scale), 3
            )
            return fn(q, k, v)
        fn = _shard_over_mesh(
            lambda q, k, v, m: _pallas_attention(q, k, v, m, scale), 4
        )
        return fn(q, k, v, padding_mask)

    impl: Literal["cudnn", "xla"] = "cudnn" if backend == "cudnn" else "xla"
    if padding_mask is None:
        # is_causal rather than a dense mask: no T x S tensor for the kernel to read.
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, is_causal=causal, implementation=impl
        )
    mask = _dense_mask(q.shape[1], k.shape[1], causal, padding_mask)
    return jax.nn.dot_product_attention(
        q, k, v, mask=mask, scale=scale, implementation=impl
    )
