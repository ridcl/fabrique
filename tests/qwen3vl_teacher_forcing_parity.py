"""Score the JAX model against vLLM's outputs with teacher forcing.

Comparing two model variants by running full evals is wasteful: each document
needs 1000-2000 sequential decode steps, and the resulting metric is dominated by
wherever the first token error happens (everything after is drift).

Instead, feed vLLM's known-good completion through the JAX model in a *single*
forward pass and count how often JAX's argmax agrees with vLLM's next token.
That is one prefill per document instead of thousands of decode steps, and it
measures per-position agreement directly, so it is both far cheaper and more
sensitive than an end-to-end metric.

    python tests/qwen3vl_teacher_forcing_parity.py
    FABRIQUE_DEEPSTACK_AT_FIRST_LAYERS=1 python tests/qwen3vl_teacher_forcing_parity.py
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))

import doc_vqa_eval as E  # noqa: E402

from fabrique.models.qwen3vl import model as model_lib  # noqa: E402
from fabrique.models.qwen3vl.loading import load_model  # noqa: E402
from fabrique.models.qwen3vl.utils import encode_messages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=E.MODEL_ID)
    ap.add_argument("--config", default=E.BASE_CONFIG)
    ap.add_argument("--refs", default="output/doc_vqa_eval/records_vllm.json")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=E.RANDOM_SEED)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    args = ap.parse_args()

    with open(args.refs) as fh:
        refs = {r["source"]: r for r in json.load(fh)}
    print(f"{len(refs)} vLLM reference outputs from {args.refs}")

    rows = E.load_rows(args.n, args.seed)
    samples = []
    for row in rows:
        if len(samples) >= args.n:
            break
        s = E.build_sample(row, ignore_case=False)
        if s is not None and s.source in refs:
            samples.append(s)
    print(f"{len(samples)} documents matched to references")

    mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
    config = getattr(model_lib.ModelConfig, args.config)()
    processor, model = load_model(args.model, mesh=mesh, config=config)

    from fabrique.models.qwen3vl.vision import attention_impl_kwargs

    print(f"attention impl: {attention_impl_kwargs(config.param_dtype) or 'xla'}\n")

    total_hits = total_toks = 0
    print(f"{'document':<34}{'tokens':>8}{'top1 agree':>12}{'mean NLL':>10}")
    for s in samples:
        query_list = "\n".join(f"- {q}" for q in s.queries)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in s.images]
                + [{"type": "text", "text": f"{E.PROMPT_HEADER}\n{query_list}"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": refs[s.source]["output"]}],
            },
        ]
        batch = encode_messages(
            processor,
            [conversation],
            loss_roles={"assistant"},
            vcfg=config.vision_config,
            max_seq_len=args.max_seq_len,
            padding=True,
            pad_to_multiple_of=128,
            truncation=True,
        )
        logits, _ = model(
            jnp.asarray(batch.input_tokens),
            jnp.asarray(batch.positions),
            jnp.asarray(batch.pixel_values, dtype=jnp.bfloat16),
            batch.vision_grid,
            None,  # no cache: single teacher-forced forward pass
            jnp.asarray(batch.input_mask).astype(jnp.bool_),
        )
        logits = logits[:, :-1, :].astype(jnp.float32)
        targets = jnp.asarray(batch.input_tokens)[:, 1:]
        mask = jnp.asarray(batch.completion_mask)[:, 1:].astype(bool)

        hits = int(((jnp.argmax(logits, -1) == targets) & mask).sum())
        n_tok = int(mask.sum())
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        tgt_lp = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
        nll = float(-(tgt_lp * mask).sum() / max(n_tok, 1))

        total_hits += hits
        total_toks += n_tok
        print(
            f"{s.source[-32:]:<34}{n_tok:>8}{hits / max(n_tok, 1):>11.1%}{nll:>10.4f}"
        )

    print(f"\n{'TOTAL':<34}{total_toks:>8}{total_hits / max(total_toks, 1):>11.1%}")


if __name__ == "__main__":
    main()
