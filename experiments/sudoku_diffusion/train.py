"""End-to-end demo: train ``SudokuDiffusion`` and watch it learn to solve Sudoku.

This wires together the three pieces of the experiment:

* ``generator.py`` -- an endless stream of synthetic, fully-solved boards.
* ``trainer.py``   -- uniform-state (DiffusionGemma-style) diffusion training.
* ``sampler.py``   -- the entropy-bounded denoising loop that solves puzzles.

The model is trained purely on *solved* boards: the diffusion forward process
corrupts random cells to random digits and the network learns to undo that.
Periodically we hold out fresh puzzles (solved boards with cells removed), solve
them with the denoiser, and report cell-/board-level accuracy plus one example.

Run (CPU is fine for the tiny default model)::

    JAX_PLATFORMS=cpu uv run python -m experiments.sudoku_diffusion.train

The defaults below favour a quick, watchable demo over a strong solver -- bump
``MAX_STEPS`` / use ``ModelConfig.sudoku_small`` and more clues for real accuracy.
"""

from __future__ import annotations

import logging

import jax
import numpy as np
from flax import nnx

from experiments.sudoku_diffusion.generator import (
    make_puzzle,
    random_solution,
    solved_board_batches,
)
from experiments.sudoku_diffusion.model import (
    ModelConfig,
    SudokuDiffusion,
    count_params,
)
from experiments.sudoku_diffusion.checkpoint import save_model
from experiments.sudoku_diffusion.sampler import SamplerConfig, solve, solve_accuracy
from experiments.sudoku_diffusion.tokenizer import SudokuTokenizer
from experiments.sudoku_diffusion.trainer import DiffusionConfig, SudokuDiffusionTrainer
from experiments.sudoku_diffusion.visualize import visualize_solving

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Experiment knobs ------------------------------------------------------
SEED = 0
BATCH_SIZE = 256
MAX_STEPS = 20_000
LOG_EVERY = 100
EVAL_EVERY = 1_000
NUM_EVAL_PUZZLES = 64
NUM_CLUES = 15  # easier puzzles (more givens) -> visible progress sooner
MODEL_OUTPUT_DIR = "/data/sudoku_diffusion_15c"


def make_eval_set(rng: np.random.Generator):
    """Return (puzzles, solutions) as int32 arrays [N, L]."""
    sols = [random_solution(rng) for _ in range(NUM_EVAL_PUZZLES)]
    puzzles = np.stack(
        [make_puzzle(s, rng, num_clues=NUM_CLUES).reshape(-1) for s in sols]
    ).astype(np.int32)
    solutions = np.stack([s.reshape(-1) for s in sols]).astype(np.int32)
    return puzzles, solutions


def main() -> None:
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(SEED))
    log.info("Model ready -- %.1fM params", count_params(model) / 1e6)

    trainer = SudokuDiffusionTrainer(
        model,
        DiffusionConfig(
            max_steps=MAX_STEPS,
            decay_steps=MAX_STEPS,
            log_every=LOG_EVERY,
            eval_every=EVAL_EVERY,
        ),
    )

    data_rng = np.random.default_rng(SEED)
    eval_rng = np.random.default_rng(SEED + 1)
    eval_puzzles, eval_solutions = make_eval_set(eval_rng)
    sampler_cfg = SamplerConfig()
    sample_key = jax.random.key(SEED + 2)

    def eval_fn(model: SudokuDiffusion, step: int) -> None:
        # Derive the per-eval key from the step (no mutable closure state).
        k = jax.random.fold_in(sample_key, step)
        filled = solve(model, eval_puzzles, k, sampler_cfg)
        metrics = solve_accuracy(filled, eval_solutions, eval_puzzles)
        log.info(
            "step %d  eval: blank-cell acc=%.3f  boards solved=%.3f",
            step,
            metrics["cell_accuracy"],
            metrics["board_solved"],
        )
        log.info(
            "example (puzzle | model solution | truth):\n%s",
            _side_by_side(tok, eval_puzzles[0], filled[0], eval_solutions[0]),
        )

    data = solved_board_batches(BATCH_SIZE, data_rng, tokenizer=tok)
    log.info("Training for %d steps (batch=%d)…", MAX_STEPS, BATCH_SIZE)
    trainer.train(data, key=jax.random.key(SEED), eval_fn=eval_fn)

    saved = save_model(model, MODEL_OUTPUT_DIR)
    log.info("Saved trained model -> %s", saved)

    log.info("Animating the denoising of one held-out puzzle")
    log.info("(blue = clue, green = correct guess, red = wrong guess):")

    visualize_solving(
        model,
        eval_puzzles[0],
        eval_solutions[0],
        jax.random.fold_in(sample_key, MAX_STEPS),
        config=sampler_cfg,
        delay=0.15,
    )


def _side_by_side(
    tok: SudokuTokenizer, puzzle: np.ndarray, solved: np.ndarray, truth: np.ndarray
) -> str:
    """Render three boards next to each other for a readable eval printout."""
    p = tok.decode(puzzle, as_grid=True).split("\n")
    s = tok.decode(solved, as_grid=True).split("\n")
    t = tok.decode(truth, as_grid=True).split("\n")
    return "\n".join(f"{pr}   {sr}   {tr}" for pr, sr, tr in zip(p, s, t))


if __name__ == "__main__" and "__file__" in globals():
    main()
