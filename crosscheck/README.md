# crosscheck

One-off scripts that verify fabrique's JAX models against **another framework**
(HuggingFace/PyTorch, or vLLM over HTTP). They are kept in the repo because they
are worth re-running whenever a model changes, but they are deliberately *not*
part of the `fabrique` package:

* they import `torch`/`torchvision`, which are **not** installed in the default
  environment, so having them under `src/` made every type-checker run and IDE
  session report unresolved imports;
* nothing in the library imports them — they are entry points, run by hand;
* library users have no reason to see them.

No `__init__.py`, no `test_`/`_test` suffixes (pytest collects both `test_*.py`
and `*_test.py`; these scripts must never be collected, since the default
environment cannot import them). Real pytest suites live in `tests/`.

Run from the repo root — `fabrique` is installed editable, so plain
`from fabrique... import ...` resolves without `sys.path` games.

## Scripts

| script | compares | needs |
|---|---|---|
| `qwen3_5_consistency.py` | JAX Qwen3.5 vs a golden `.npz`, layer by layer | numpy only |
| `qwen3_5_dump_hf_reference.py` | *produces* that `.npz` from HuggingFace | separate venv, see below |
| `qwen3vl_consistency.py` | JAX Qwen3-VL vs HuggingFace, layer by layer | `--group crosscheck` |
| `qwen3vl_vision_parity.py` | the vision tower alone, stage by stage | `--group crosscheck` |
| `qwen3vl_teacher_forcing_parity.py` | argmax agreement + NLL vs a vLLM completion | running vLLM |
| `doc_vqa_vllm_crosscheck.py` | end-to-end DocVQA answers vs vLLM | running vLLM |

## Qwen3.5: a two-environment check

`qwen3_5_consistency.py` does **not** import torch. The project's JAX
environment pins transformers 4.57, which has no `qwen3_5`, and installing a
newer transformers + CUDA torch into it has broken that environment before. So
the reference is dumped separately and compared from an `.npz`:

```bash
# 1. produce the golden file (isolated venv, CPU, ~minutes)
/data/hfref-venv/bin/python crosscheck/qwen3_5_dump_hf_reference.py \
    --model Qwen/Qwen3.5-0.8B --dtype float32 \
    --out /data/consistency-tests/qwen3_5_0_8b_fp32.npz

# 2. compare, from the normal JAX environment
python crosscheck/qwen3_5_consistency.py \
    --model Qwen/Qwen3.5-0.8B --dtype float32 \
    --ref /data/consistency-tests/qwen3_5_0_8b_fp32.npz
```

Both paths are under `/data` on purpose: it is the only mount that survives a
devcontainer rebuild. Golden files written to `/tmp` have been lost to a rebuild
before, and regenerating one costs a fresh venv plus a full fp32 CPU forward
pass.

Recreate the reference venv if it is missing. **The CPU torch wheel matters**:
the CUDA wheel pulls `nvidia-cudnn-cu13`, whose `libcudnn.so.9` overwrites the
cu12 copy JAX needs, silently disabling the cuDNN flash-attention kernel.

```bash
uv venv /data/hfref-venv --python 3.12
uv pip install --python /data/hfref-venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    "transformers>=5.17" torch accelerate numpy
```

## Qwen3-VL: same environment

These do import torch, so they need the `crosscheck` dependency group — the
`build-dev-torch` Docker stage, or:

```bash
uv sync --active --inexact --group crosscheck
```

`--inexact` is required: a plain `uv sync` would prune the JAX environment.

## Known blind spot

`qwen3vl_consistency.py` injects fabrique's *own* attention mask into both
frameworks, so it structurally cannot catch mask bugs — it passed throughout the
period when the causal mask was wrong (image attention was bidirectional).
For localising that class of bug, prefer `qwen3vl_teacher_forcing_parity.py`.
