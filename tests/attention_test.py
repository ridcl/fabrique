"""fabrique.attention: every backend must match the XLA reference.

GPU-only cases are skipped on CPU.  The reference is the XLA kernel in float32
with explicit masks, so a failing backend shows up as a numerical mismatch.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from fabrique import attention as A

GPU = A._compute_capability() is not None
needs_gpu = pytest.mark.skipif(not GPU, reason="GPU backend")


def _inputs(batch, seq, heads, kv_heads, head_dim, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (batch, seq, heads, head_dim), jnp.bfloat16)
    k = jax.random.normal(ks[1], (batch, seq, kv_heads, head_dim), jnp.bfloat16)
    v = jax.random.normal(ks[2], (batch, seq, kv_heads, head_dim), jnp.bfloat16)
    w = jax.random.normal(ks[3], (batch, seq, heads, head_dim), jnp.float32)
    return q, k, v, w


def _reference(q, k, v, padding_mask):
    """float32 XLA attention with an explicit causal + key-padding mask."""
    rep = q.shape[2] // k.shape[2]
    k, v = jnp.repeat(k, rep, 2), jnp.repeat(v, rep, 2)
    q, k, v = (t.astype(jnp.float32) for t in (q, k, v))
    s = jnp.einsum("btnh,bsnh->bnts", q, k) * q.shape[-1] ** -0.5
    L = q.shape[1]
    mask = (jnp.arange(L)[None, :] <= jnp.arange(L)[:, None])[None, None]
    if padding_mask is not None:
        mask = mask & padding_mask[:, None, None, :]
    p = jax.nn.softmax(jnp.where(mask, s, -1e30), axis=-1)
    return jnp.einsum("bnts,bsnh->btnh", p, v)


def _value_and_grads(fn, q, k, v, w, valid):
    """Loss over *valid query rows only*: padded queries are garbage by contract."""

    def loss(q, k, v):
        o = fn(q, k, v).astype(jnp.float32)
        return jnp.sum(o * w * valid[:, :, None, None])

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(q, k, v)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max())


CASES = {
    # name: (batch, seq, heads, kv_heads, head_dim, padding)
    "gqa_hd256": (2, 256, 8, 2, 256, None),
    "hd128": (2, 256, 4, 4, 128, None),
    "right_pad": (2, 256, 8, 2, 256, "right"),
    "left_pad": (2, 256, 8, 2, 256, "left"),
    "odd_length": (1, 100, 8, 2, 256, None),  # not a multiple of the tiles
}


def _padding(kind, batch, seq):
    if kind is None:
        return None
    n_real = np.array([seq, seq * 5 // 8])[:batch]
    idx = np.arange(seq)[None, :]
    m = idx < n_real[:, None] if kind == "right" else idx >= (seq - n_real)[:, None]
    return jnp.asarray(m)


@needs_gpu
@pytest.mark.parametrize("backend", ["pallas", "cudnn"])
@pytest.mark.parametrize("case", list(CASES))
def test_backend_matches_reference(backend, case):
    batch, seq, heads, kv_heads, head_dim, pad = CASES[case]
    if backend == "cudnn" and (
        head_dim > A.cudnn_max_head_dim() or not A.cudnn_attention_available()
    ):
        pytest.skip(f"cuDNN does not support head_dim {head_dim} on this GPU")
    q, k, v, w = _inputs(batch, seq, heads, kv_heads, head_dim)
    pm = _padding(pad, batch, seq)
    valid = jnp.ones((batch, seq), bool) if pm is None else pm

    def fn(q, k, v):
        return A.dot_product_attention(
            q, k, v, causal=True, padding_mask=pm, implementation=backend
        )

    # Outputs compared on valid query rows; not via the summed loss, whose
    # signed terms cancel and turn bf16 rounding into a misleading 1% "error".
    rows = np.asarray(valid)
    out = np.asarray(fn(q, k, v), np.float32)[rows]
    ref_out = np.asarray(_reference(q, k, v, pm), np.float32)[rows]
    assert np.abs(out - ref_out).max() < 2e-2  # bf16 outputs of O(1)

    _, g_got = _value_and_grads(fn, q, k, v, w, valid)
    _, g_ref = _value_and_grads(
        lambda q, k, v: _reference(q, k, v, pm), q, k, v, w, valid
    )
    for got, ref in zip(g_got, g_ref, strict=True):
        assert _rel(got, ref) < 2e-2


def test_selection_rules():
    assert A.select_backend(jnp.float32, 256, causal=True) == "xla"  # parity runs
    assert A.select_backend(jnp.bfloat16, 100, causal=True) == "xla"  # head_dim % 8
    assert (
        A.select_backend(jnp.bfloat16, 256, causal=True, implementation="xla") == "xla"
    )
    if not GPU:
        assert A.select_backend(jnp.bfloat16, 256, causal=True) == "xla"
        return
    if A.cudnn_attention_available():
        assert A.select_backend(jnp.bfloat16, 128, causal=True) == "cudnn"
    if A.cudnn_max_head_dim() < 256:  # pre-Hopper: the case this module exists for
        assert A.select_backend(jnp.bfloat16, 256, causal=True) == "pallas"
        assert A.select_backend(jnp.bfloat16, 256, causal=False) == "xla"


@pytest.mark.skipif(not GPU or jax.device_count() < 2, reason="needs 2 GPUs")
def test_pallas_is_sharded_not_gathered_under_a_mesh():
    """Without shard_map XLA all-gathers q/k/v and every GPU does all the work."""
    n = 2
    mesh = jax.make_mesh(
        (n,),
        ("data",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=jax.devices()[:n],
    )
    q, k, v, w = _inputs(n, 256, 8, 2, 256)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P("data")))  # noqa: E731
    q, k, v, w = map(put, (q, k, v, w))

    def loss(q, k, v):
        o = A.dot_product_attention(q, k, v, causal=True, implementation="pallas")
        return jnp.sum(o.astype(jnp.float32) * w)

    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile()
        grads = compiled(q, k, v)
    assert not re.search(r"all-gather", compiled.as_text() or "")
    ref = jax.jit(
        jax.grad(
            lambda q, k, v: jnp.sum(_reference(q, k, v, None) * w), argnums=(0, 1, 2)
        )
    )(q, k, v)
    for got, r in zip(grads, ref, strict=True):
        assert _rel(got, r) < 2e-2
