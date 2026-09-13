"""Layer-by-layer consistency test: JAX Qwen3.5 vs HuggingFace.

Unlike the sibling qwen3vl test, this does not import torch.  The project's JAX
environment pins transformers 4.57, which has no ``qwen3_5``, so the reference
is produced separately by ``tests/qwen3_5_dump_hf_reference.py`` in an isolated
environment and compared here from an .npz.  That also keeps the check fast
(no second copy of the model) and reproducible.

    python -m fabrique.models.qwen3_5.consistency_test \
        --model Qwen/Qwen3.5-0.8B --ref /path/to/qwen3_5_ref.npz

Exits 0 when every layer and the logits agree within tolerance.
"""

import argparse
import json
import sys

import jax.numpy as jnp
import numpy as np

from fabrique.models.qwen3_5.loading import load_model, resolve_model_dir


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs diff scaled by the reference's own magnitude.

    Absolute diffs are misleading on transformer hidden states: a handful of
    "massive activation" dims reach into the thousands, so a large absolute
    error there can still be a tiny relative one.
    """
    scale = max(float(np.abs(b).max()), 1e-6)
    return float(np.abs(a - b).max() / scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument(
        "--ref", required=True, help=".npz from qwen3_5_dump_hf_reference.py"
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tol", type=float, default=2e-3, help="max relative error")
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=False)
    print(
        f"reference: {ref['model']} dtype={ref['dtype']} "
        f"transformers={ref['transformers_version']} layers={int(ref['n_layers'])}"
    )
    hf_cfg = json.loads(str(ref["text_config"]))

    model_dir = resolve_model_dir(args.model)
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    config, model = load_model(model_dir, dtype=dtype)

    # Guard against the trap hit in the qwen3vl port: a requested dtype that
    # does not actually reach the modules makes every diff below meaningless.
    assert config.param_dtype == dtype, (config.param_dtype, dtype)

    for key, attr in [
        ("num_hidden_layers", "num_layers"),
        ("hidden_size", "embed_dim"),
        ("num_attention_heads", "num_heads"),
        ("head_dim", "head_dim"),
        ("linear_num_value_heads", "linear_num_value_heads"),
    ]:
        assert hf_cfg[key] == getattr(config, attr), (
            f"config mismatch {key}: HF {hf_cfg[key]} vs JAX {getattr(config, attr)}"
        )

    input_ids = jnp.asarray(ref["input_ids"], dtype=jnp.int32)
    logits, hidden = model(input_ids, output_hidden_states=True)
    assert hidden is not None, "output_hidden_states=True must return them"
    logits = np.asarray(logits, dtype=np.float32)

    n_layers = int(ref["n_layers"])
    assert len(hidden) == n_layers, (len(hidden), n_layers)

    print(f"\n{'layer':>6} {'type':<18}{'rel err':>12}{'max abs':>12}{'ref scale':>12}")
    worst, worst_at = 0.0, None
    for i in range(n_layers):
        got = np.asarray(hidden[i], dtype=np.float32)
        want = ref[f"hidden_{i}"]
        rel = _rel_err(got, want)
        if rel > worst:
            worst, worst_at = rel, i
        print(
            f"{i:>6} {config.layer_types[i]:<18}{rel:>12.3e}"
            f"{np.abs(got - want).max():>12.3e}{np.abs(want).max():>12.3e}"
        )

    logit_rel = _rel_err(logits, ref["logits"])
    top1_jax = logits.argmax(-1)
    top1_hf = ref["logits"].argmax(-1)
    agree = float((top1_jax == top1_hf).mean())

    print(f"\n  worst layer        {worst:.3e} (layer {worst_at})")
    print(f"  logits rel err     {logit_rel:.3e}")
    print(
        f"  top-1 agreement    {agree:.4%}  ({int((top1_jax == top1_hf).sum())}"
        f"/{top1_jax.size} positions)"
    )

    ok = worst <= args.tol and logit_rel <= args.tol and agree == 1.0
    print(f"\n{'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
