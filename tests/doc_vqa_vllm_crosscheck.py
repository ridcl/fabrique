"""Cross-check the fabrique JAX inference path against vLLM on identical inputs.

``doc_vqa_eval.py`` scored ridcl/paperwerk-vqa at value_f1 0.19 / iou_matched
0.03 with 80% of outputs malformed — implausibly bad for a model that reportedly
worked when served with vLLM.  This script decides whether that is an inference
bug in the JAX path or the model's real behaviour, by sending byte-identical
prompts and images to a vLLM OpenAI-compatible endpoint and diffing the results.

Both sides must see the same pixels and the same text, so this reuses
``doc_vqa_eval`` for sampling, resizing, prompting and scoring — one source of
truth.  Sampling is greedy (``temperature=0``) on both sides: under a correct
implementation greedy decoding is deterministic, so the outputs should be
identical or diverge only far into the sequence.  A short common prefix is
strong evidence of a numerical bug.

Start the server first (in a container with the GPUs):

    docker run --rm --gpus all --shm-size=16g -p 8000:8000 \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      paperwerk-serve:latest \\
      /venv/bin/vllm serve ridcl/paperwerk-vqa \\
        --tensor-parallel-size 2 \\
        --dtype bfloat16 \\
        --max-model-len 16384 \\
        --gpu-memory-utilization 0.85 \\
        --enforce-eager \\
        --max-num-seqs 4 \\
        --limit-mm-per-prompt '{"image": 2}'

Then:

    python experiments/doc_vqa_vllm_crosscheck.py --n 5 --seed 42

Notes on the server flags, relative to the Gemma 4 command:
  * ``--limit-mm-per-prompt`` must allow at least ``--max-pages`` images or
    multi-page documents are rejected.  Older vLLM wants ``image=2`` instead of
    the JSON form.
  * Mount the HF cache so the 8 GB checkpoint is not downloaded again.
  * The tool-calling flags are irrelevant here (this model emits plain JSON).
  * ``--max-model-len 16384`` is plenty: worst case in budget is ~8k tokens.
"""

import argparse
import base64
import io
import json
import os
import sys

import requests

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))
import doc_vqa_eval as E  # noqa: E402


def _data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(sample: E.Sample) -> list[dict]:
    """The OpenAI-format twin of ``doc_vqa_eval.build_prompt``.

    vLLM applies the same chat_template.jinja server-side, so passing the
    structured message is equivalent to the string the JAX path builds.
    """
    query_list = "\n".join(f"- {q}" for q in sample.queries)
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _data_url(img)}}
        for img in sample.images
    ]
    content.append({"type": "text", "text": f"{E.PROMPT_HEADER}\n{query_list}"})
    return [{"role": "user", "content": content}]


def generate(base_url: str, model: str, sample: E.Sample, max_tokens: int) -> str:
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": build_messages(sample),
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy, to match the JAX sampler
        },
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=E.MODEL_ID)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=E.RANDOM_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=E.EVAL_MAX_NEW_TOKENS)
    ap.add_argument("--ignore-case", action="store_true")
    ap.add_argument(
        "--jax-records",
        default="output/doc_vqa_eval/records2.json",
        help="dump written by doc_vqa_eval.py --dump, for the diff",
    )
    ap.add_argument("--out", default="output/doc_vqa_eval/records_vllm.json")
    args = ap.parse_args()

    try:
        served = requests.get(f"{args.base_url}/models", timeout=10).json()
        print(f"server up; serving: {[m['id'] for m in served.get('data', [])]}")
    except requests.RequestException as exc:
        sys.exit(
            f"cannot reach vLLM at {args.base_url}: {exc}\n"
            "start the server first (see the module docstring)."
        )

    # Identical sample selection to doc_vqa_eval: same seed, same budget filter.
    rows = E.load_rows(args.n, args.seed)
    samples: list[E.Sample] = []
    for row in rows:
        if len(samples) >= args.n:
            break
        s = E.build_sample(row, args.ignore_case)
        if s is not None:
            samples.append(s)
    print(f"{len(samples)} documents selected (seed={args.seed})")

    jax_by_source = {}
    if os.path.exists(args.jax_records):
        with open(args.jax_records) as fh:
            jax_by_source = {r["source"]: r for r in json.load(fh)}
        print(f"loaded {len(jax_by_source)} JAX records from {args.jax_records}")
    else:
        print(f"no JAX records at {args.jax_records}; running vLLM-only")

    acc = E.Accumulator()
    records = []
    for i, sample in enumerate(samples):
        output = generate(args.base_url, args.model, sample, args.max_new_tokens)
        preds, status = E.parse_output(output, args.ignore_case)
        stats = acc.add(preds, sample.gt, status)

        jrec = jax_by_source.get(sample.source)
        if jrec is None:
            diff = ""
        elif jrec["output"] == output:
            diff = "  [IDENTICAL to JAX]"
        else:
            cp = common_prefix(jrec["output"], output)
            diff = (
                f"  [DIVERGES from JAX at char {cp}; "
                f"jax {len(jrec['output'])} chars/{jrec['status']}, "
                f"vllm {len(output)} chars/{status}]"
            )

        print(
            f"[{i + 1}/{len(samples)}] gt={stats['n_gt']} pred={stats['n_pred']} "
            f"tp={stats['tp']} f1={stats['f1']:.3f} iou={stats['iou']:.3f} "
            f"({status}){diff}"
        )
        records.append(
            {"source": sample.source, "status": status, "output": output, **stats}
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(records, fh, indent=2)

    vllm_metrics = acc.summary()

    jax_metrics = None
    if jax_by_source:
        jacc = E.Accumulator()
        by_source = {s.source: s for s in samples}
        for src, rec in jax_by_source.items():
            if src not in by_source:
                continue
            preds, status = E.parse_output(rec["output"], args.ignore_case)
            jacc.add(preds, by_source[src].gt, status)
        jax_metrics = jacc.summary()

    print("\n" + "=" * 62)
    print(f"{'metric':<16}{'JAX (fabrique)':>16}{'vLLM (torch)':>16}{'delta':>14}")
    print("=" * 62)
    for key in vllm_metrics:
        v = vllm_metrics[key]
        if jax_metrics is None:
            print(
                f"  {key:<14}{'-':>16}{v:>16.4f}"
                if isinstance(v, float)
                else f"  {key:<14}{'-':>16}{v:>16}"
            )
            continue
        j = jax_metrics[key]
        if isinstance(v, float):
            print(f"  {key:<14}{j:>16.4f}{v:>16.4f}{v - j:>+14.4f}")
        else:
            print(f"  {key:<14}{j:>16}{v:>16}{'':>14}")
    print("=" * 62)
    print(
        "\nIf the outputs are identical or diverge only late, the JAX path is "
        "sound and the model is genuinely this weak.\nIf they diverge early, "
        "the JAX inference path has a bug and the eval numbers are void."
    )


if __name__ == "__main__":
    main()
