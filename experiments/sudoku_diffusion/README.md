# Sudoku Diffusion

A small, dense, **DiffusionGemma-like** model that learns to solve Sudoku by
*denoising* a board, re-implemented from scratch in JAX / Flax NNX.

It mirrors the architecture of Google's
[DiffusionGemma](https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/)
(HF [`modeling_diffusion_gemma.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py))
but is deliberately stripped down for the toy task.

## What this is (and isn't)

| DiffusionGemma | This re-implementation |
| --- | --- |
| PyTorch | JAX / Flax NNX |
| 26B Mixture-of-Experts | Dense (single GeGLU MLP per layer) |
| Pretrained, 256k SentencePiece vocab | From scratch, 12-token digit vocab |
| Encoder–decoder + vision tower | Decoder-only canvas, no vision |
| KV cache for AR-style decoding | Full bidirectional forward passes |

What it **keeps** from the Gemma / DiffusionGemma family — the parts that define
the architecture:

- **Bidirectional attention** over the whole board (diffusion denoises every
  cell at once, so there is no causal mask).
- **Sandwich RMSNorm** (norm before *and* after each attention / FFN sub-block),
  Gemma `(1 + weight)` scaling.
- **QK-norm**, **RoPE** (with optional sliding-window layers).
- **GeGLU** feed-forward with tanh-approx GELU.
- **Embedding scaling** by `sqrt(embed_dim)` and **final-logit softcapping**.
- **Self-conditioning**: the model can be fed its own soft predictions from the
  previous denoising step.

## Files

- `tokenizer.py` — `SudokuTokenizer`: ten digit tokens (`0`–`9`) plus `<pad>`
  and `<mask>`. A board is serialized row-by-row, space/newline separated;
  separators are formatting only and never become tokens.
- `model.py` — `ModelConfig` (presets `sudoku_tiny` ≈ 7M, `sudoku_small` ≈ 45M
  ≈ 90 MB bf16 [default], `sudoku_base` ≈ 163M) and `SudokuDiffusion`.
- `generator.py` — synthetic data: `random_solution` (constructive, solver-free
  valid grids), `make_puzzle`, `solved_board_batches` (endless training stream).
- `trainer.py` — `DiffusionConfig`, `SudokuDiffusionTrainer`, and the
  uniform-state `corrupt` / `diffusion_loss` primitives.
- `sampler.py` — `SamplerConfig`, `solve` (entropy-bounded denoising loop),
  `solve_iter` (yields the board after every step), and `solve_accuracy`.
- `visualize.py` — `visualize_solving`: animate the denoising in the terminal,
  one frame per step with a controllable `delay`. Givens get a **blue**
  background; blanks are coloured **green** (correct) or **red** (wrong) against
  the known solution, so you can watch cells flip and settle.
- `checkpoint.py` — `save_model` / `load_model`: persist a trained denoiser
  (config JSON + orbax weights) and reload it standalone.
- `train.py` — runnable end-to-end demo (generate → train → save → solve →
  report), ending with an animated solve of one held-out puzzle.
- `test_sudoku_diffusion.py`, `test_diffusion_pipeline.py` — smoke tests.

## Usage

```python
from flax import nnx
import jax.numpy as jnp
from experiments.sudoku_diffusion import ModelConfig, SudokuDiffusion, SudokuTokenizer

tok = SudokuTokenizer()
cfg = ModelConfig.sudoku_small(vocab_size=tok.vocab_size)
model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))

ids = jnp.array(tok.encode(board_text))[None]   # [1, 81]
logits = model(ids)                             # [1, 81, vocab]
```

Run the tests:

```bash
JAX_PLATFORMS=cpu uv run python -m pytest experiments/sudoku_diffusion/ -q
```

## How the diffusion works

Like DiffusionGemma — and unlike MaskGIT / LLaDA — this is **uniform-state**
(multinomial) discrete diffusion, *not* masked / absorbing-state diffusion. The
difference is the whole point:

- **Masked diffusion** corrupts a cell into a special `<mask>` placeholder. The
  model always *knows* which cells are noise, and a committed cell can never be
  revised.
- **Uniform-state diffusion** corrupts a cell into a *random digit*. Noise is
  indistinguishable from signal, so the model must decide both *whether* each
  cell is wrong and *what* it should be — and because corrupted cells are
  ordinary tokens, the sampler is free to overwrite an earlier guess later.

The denoiser is **time-agnostic**: it never receives `t` (neither does
DiffusionGemma's network). The noise level only drives corruption at training
time and the temperature schedule at sampling time.

1. **Forward (noising), `trainer.corrupt`:** sample `t ~ U(0,1)`; replace each
   cell independently with probability `t` by a uniform random digit.
2. **Training, `trainer.diffusion_loss`:** the model predicts the original board
   from the corrupted one; cross-entropy is taken over the *corrupted* cells.
   Self-conditioning (DiffusionGemma-style) feeds a detached first pass back in
   on a random fraction of steps.
3. **Sampling / solving, `sampler.solve`:** start from a fully random canvas
   (givens clamped, blanks random); each step predicts all cells, commits the
   most-confident ones (entropy-bounded), **renoises** the rest with fresh random
   digits, and feeds its logits back via self-conditioning. An annealed
   temperature shifts from exploration to commitment as the noise level falls.

### Run the demo

```bash
JAX_PLATFORMS=cpu uv run python -m experiments.sudoku_diffusion.train
```

This generates solved boards, trains the tiny model, and periodically solves
held-out puzzles, printing cell/board accuracy and an example. The defaults
favour a quick, watchable run; raise `MAX_STEPS` (and/or use
`ModelConfig.sudoku_small`) for real solving accuracy.
