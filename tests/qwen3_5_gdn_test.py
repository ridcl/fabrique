"""Gated DeltaNet correctness, differentiability and compile-stability.

Three properties, none of which needs a checkpoint:

1. the chunked (prefill) and recurrent (decode) kernels agree with each other
   and with a literal per-token delta-rule loop;
2. gradients flow through both, are finite and non-zero, and match between the
   two formulations;
3. the emitted jaxpr has a **fixed size regardless of sequence length** -- i.e.
   the recurrence is a ``lax.scan``, not an unrolled per-token chain.  (3) is
   the property that breaks first if someone "simplifies" the scan away, and it
   is invisible in correctness tests.

    pytest tests/qwen3_5_gdn_test.py
    python tests/qwen3_5_gdn_test.py     # verbose
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.extend  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fabrique.models.qwen3_5 import model as M  # noqa: E402


def _inputs(batch=2, seq=64, heads=4, k_dim=16, v_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)  # noqa: E731
    a = jnp.asarray(rng.standard_normal((batch, seq, heads)), jnp.float32)
    return (
        f(batch, seq, heads, k_dim),
        f(batch, seq, heads, k_dim),
        f(batch, seq, heads, v_dim),
        -jnp.exp(jnp.asarray(rng.uniform(-3, 0, (heads,)))) * jax.nn.softplus(a),
        jax.nn.sigmoid(
            jnp.asarray(rng.standard_normal((batch, seq, heads)), jnp.float32)
        ),
    )


def _count_eqns(jaxpr, depth=0) -> int:
    """Total equations, recursing into sub-jaxprs (scan bodies included)."""
    n = len(jaxpr.eqns)
    for eq in jaxpr.eqns:
        for v in eq.params.values():
            if isinstance(v, jax.extend.core.ClosedJaxpr):
                n += _count_eqns(v.jaxpr, depth + 1)
            elif isinstance(v, jax.extend.core.Jaxpr):
                n += _count_eqns(v, depth + 1)
            elif isinstance(v, tuple):
                for x in v:
                    if isinstance(x, jax.extend.core.ClosedJaxpr):
                        n += _count_eqns(x.jaxpr, depth + 1)
    return n


def test_chunked_matches_recurrent():
    for seq, chunk in [(8, 8), (64, 64), (96, 64), (192, 64)]:
        args = _inputs(seq=seq)
        chunked, s1 = M.chunk_gated_delta_rule(*args, chunk_size=chunk)
        recurrent, s2 = M.recurrent_gated_delta_rule(*args)
        scale = max(float(jnp.abs(recurrent).max()), 1e-6)
        rel = float(jnp.abs(chunked - recurrent).max()) / scale
        assert rel < 1e-4, f"seq={seq} chunk={chunk}: rel err {rel:.2e}"
        state_rel = float(jnp.abs(s1 - s2).max()) / max(float(jnp.abs(s2).max()), 1e-6)
        assert state_rel < 1e-4, f"seq={seq}: final state rel err {state_rel:.2e}"


def test_matches_literal_delta_rule():
    """Ground truth: the delta rule written out as a Python loop."""
    batch, seq, heads, dim = 1, 8, 1, 4
    rng = np.random.default_rng(1)
    q, k, v = (
        jnp.asarray(rng.standard_normal((batch, seq, heads, dim)), jnp.float32)
        for _ in range(3)
    )
    g = jnp.zeros((batch, seq, heads), jnp.float32)  # no decay
    beta = jnp.ones((batch, seq, heads), jnp.float32)  # full overwrite

    qn = np.asarray(M.l2norm(jnp.swapaxes(q, 1, 2))) / dim**0.5
    kn = np.asarray(M.l2norm(jnp.swapaxes(k, 1, 2)))
    vv = np.asarray(jnp.swapaxes(v, 1, 2))
    state, want = np.zeros((dim, dim), np.float32), []
    for t in range(seq):
        k_t, v_t, q_t = kn[0, 0, t], vv[0, 0, t], qn[0, 0, t]
        state = state + np.outer(k_t, v_t - state.T @ k_t)
        want.append(state.T @ q_t)
    want = np.stack(want)

    for fn, kw in (
        (M.chunk_gated_delta_rule, {"chunk_size": 8}),
        (M.recurrent_gated_delta_rule, {}),
    ):
        got = np.asarray(fn(q, k, v, g, beta, **kw)[0])[0, :, 0, :]
        assert np.abs(got - want).max() < 1e-5, (
            f"{fn.__name__}: {np.abs(got - want).max():.2e}"
        )


def test_gradients_flow_and_agree():
    args = _inputs(seq=96)

    def loss(fn, **kw):
        def f(*a):
            out, _ = fn(*a, **kw)
            return jnp.sum(
                out * jnp.cos(jnp.arange(out.size, dtype=out.dtype).reshape(out.shape))
            )

        return f

    gc = jax.jit(
        jax.grad(loss(M.chunk_gated_delta_rule, chunk_size=64), argnums=(0, 1, 2, 3, 4))
    )(*args)
    gr = jax.jit(jax.grad(loss(M.recurrent_gated_delta_rule), argnums=(0, 1, 2, 3, 4)))(
        *args
    )
    for name, a, b in zip(["q", "k", "v", "g", "beta"], gc, gr, strict=True):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        assert np.isfinite(a).all(), f"grad wrt {name} has non-finite values"
        assert np.abs(a).max() > 0, f"grad wrt {name} is all zeros"
        rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)
        assert rel < 1e-3, f"grad wrt {name}: chunked vs recurrent rel {rel:.2e}"


def test_no_unrolling_over_sequence_length():
    """Jaxpr size must not grow with sequence length.

    A per-token unrolled recurrence makes the graph grow linearly in the
    sequence length, taking compile time with it.  Because the sequential
    dependency is expressed as a single ``lax.scan``, the graph is a fixed size:
    a 16x longer sequence compiles to exactly the same number of equations.

    The single-chunk case (seq == chunk_size) is a slightly different -- and
    marginally larger -- graph, so it is checked separately rather than lumped
    in; what matters is that nothing grows.
    """
    chunk = 64

    def sizes(seq):
        args = _inputs(seq=seq)
        fwd = jax.make_jaxpr(lambda *a: M.chunk_gated_delta_rule(*a, chunk_size=chunk))(
            *args
        )
        bwd = jax.make_jaxpr(
            jax.grad(
                lambda *a: M.chunk_gated_delta_rule(*a, chunk_size=chunk)[0].sum(),
                argnums=(0, 1, 2, 3, 4),
            )
        )(*args)
        n_scan = sum(1 for eq in fwd.jaxpr.eqns if str(eq.primitive) == "scan")
        return _count_eqns(fwd.jaxpr), _count_eqns(bwd.jaxpr), n_scan

    # Graph shape depends on two *static* predicates -- whether the sequence
    # needs padding, and whether it is a single chunk -- and on nothing else.
    # Within each group, a 16x change in length must not move a single equation.
    groups = {
        "exact multiples, >=2 chunks": (128, 512, 1024, 2048),
        "needs padding, >=2 chunks": (96, 160, 544, 1056),
    }
    for label, lengths in groups.items():
        measured = {seq: sizes(seq) for seq in lengths}
        assert len({v[0] for v in measured.values()}) == 1, (
            f"{label}: forward jaxpr grows with seq len: {measured}"
        )
        assert len({v[1] for v in measured.values()}) == 1, (
            f"{label}: backward jaxpr grows with seq len: {measured}"
        )
        assert all(v[2] == 1 for v in measured.values()), (
            f"{label}: expected exactly one scan primitive: {measured}"
        )

    # Single chunk is its own (slightly larger) shape, but still one scan.
    single = sizes(chunk)
    assert single[2] == 1, f"single-chunk case lost its scan: {single}"


def test_model_forward_backward():
    """End-to-end on a tiny random model: finite loss, non-zero grads."""
    from flax import nnx

    cfg = M.ModelConfig(
        num_layers=4,
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10_000_000,
        norm_eps=1e-6,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        mrope_section=(2, 1, 1),
        param_dtype=jnp.float32,
        gdn_chunk_size=8,
    )
    model = M.Qwen3_5(cfg, rngs=nnx.Rngs(0))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 128, (2, 24)), jnp.int32)

    def loss_fn(m):
        logits, _ = m(tokens)
        return jnp.mean(
            jnp.square(logits[:, :-1].astype(jnp.float32))
        )  # any smooth scalar

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    assert jnp.isfinite(loss), loss
    flat = jax.tree.leaves(nnx.to_pure_dict(grads))
    assert flat, "no gradient leaves"
    assert all(bool(jnp.isfinite(g).all()) for g in flat), "non-finite gradients"
    assert any(float(jnp.abs(g).max()) > 0 for g in flat), "all gradients are zero"


if __name__ == "__main__":
    for fn in (
        test_chunked_matches_recurrent,
        test_matches_literal_delta_rule,
        test_gradients_flow_and_agree,
        test_no_unrolling_over_sequence_length,
        test_model_forward_backward,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}\n      {exc}")
