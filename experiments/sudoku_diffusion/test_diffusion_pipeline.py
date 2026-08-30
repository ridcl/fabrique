"""Smoke tests for the Sudoku generator, uniform-state trainer, and sampler.

Run with::

    JAX_PLATFORMS=cpu uv run python -m pytest \
        experiments/sudoku_diffusion/test_diffusion_pipeline.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from experiments.sudoku_diffusion.generator import (
    is_valid_solution,
    make_puzzle,
    random_solution,
    solved_board_batches,
)
from experiments.sudoku_diffusion.model import ModelConfig, SudokuDiffusion
from experiments.sudoku_diffusion.sampler import (
    SamplerConfig,
    solve,
    solve_accuracy,
    solve_iter,
)
from experiments.sudoku_diffusion.tokenizer import SudokuTokenizer
from experiments.sudoku_diffusion.trainer import (
    DiffusionConfig,
    SudokuDiffusionTrainer,
    corrupt,
    diffusion_loss,
)
from experiments.sudoku_diffusion.visualize import render_board, visualize_solving

# --- generator -------------------------------------------------------------


def test_random_solution_is_valid():
    rng = np.random.default_rng(0)
    for _ in range(20):
        board = random_solution(rng)
        assert board.shape == (9, 9)
        assert board.min() == 1 and board.max() == 9
        assert is_valid_solution(board)


def test_make_puzzle_keeps_clue_count_and_subset():
    rng = np.random.default_rng(1)
    sol = random_solution(rng)
    puzzle = make_puzzle(sol, rng, num_clues=30)
    assert int((puzzle != 0).sum()) == 30
    # Every remaining (non-zero) clue matches the solution.
    clues = puzzle != 0
    np.testing.assert_array_equal(puzzle[clues], sol[clues])


def test_solved_board_batches_shapes_and_values():
    rng = np.random.default_rng(2)
    gen = solved_board_batches(4, rng)
    batch = next(gen)
    assert batch.shape == (4, 81)
    assert batch.min() >= 1 and batch.max() <= 9


# --- corruption / loss -----------------------------------------------------


def test_corrupt_only_uses_digits_and_corruption_grows_with_t():
    boards = jnp.ones((256, 81), dtype=jnp.int32) * 5
    # t = 0 -> nothing corrupted; t = 1 -> everything corrupted.
    xt0, c0 = corrupt(
        jax.random.key(0), boards, jnp.zeros(256), noise_lo=1, noise_hi=10
    )
    xt1, c1 = corrupt(jax.random.key(0), boards, jnp.ones(256), noise_lo=1, noise_hi=10)
    assert not bool(c0.any())
    assert bool(c1.all())
    np.testing.assert_array_equal(np.asarray(xt0), np.asarray(boards))
    # Replacements are real digits in [1, 9], never a <pad>/<mask> id.
    assert int(np.asarray(xt1).min()) >= 1 and int(np.asarray(xt1).max()) <= 9


def test_diffusion_loss_finite():
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    boards = jnp.asarray(next(solved_board_batches(8, rng)))
    loss = diffusion_loss(
        model, boards, jax.random.key(0), noise_lo=1, noise_hi=10, self_cond_prob=0.5
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_trainer_overfits_single_batch():
    """A few steps on one repeated batch must drive the loss down."""
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    trainer = SudokuDiffusionTrainer(
        model, DiffusionConfig(learning_rate=1e-3, warmup_steps=0, max_steps=40)
    )
    rng = np.random.default_rng(0)
    batch = jnp.asarray(next(solved_board_batches(16, rng)))

    key = jax.random.key(0)
    first = float(trainer.train_step(batch, key))
    for _ in range(40):
        key, k = jax.random.split(key)
        last = float(trainer.train_step(batch, k))
    assert last < first


# --- sampler ---------------------------------------------------------------


def test_solve_preserves_givens_and_outputs_digits():
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))

    rng = np.random.default_rng(3)
    sols = [random_solution(rng) for _ in range(4)]
    puzzles = np.stack(
        [make_puzzle(s, rng, num_clues=40).reshape(-1) for s in sols]
    ).astype(np.int32)

    filled = solve(model, puzzles, jax.random.key(0), SamplerConfig(num_steps=8))
    assert filled.shape == puzzles.shape
    # Output cells are all real digits 1..9 (never 0/pad/mask).
    assert filled.min() >= 1 and filled.max() <= 9
    # Givens are untouched.
    clues = puzzles != 0
    np.testing.assert_array_equal(filled[clues], puzzles[clues])


def test_solve_accuracy_perfect_when_filled_equals_solution():
    rng = np.random.default_rng(4)
    sol = random_solution(rng).reshape(1, -1)
    puzzle = make_puzzle(sol.reshape(9, 9), rng, num_clues=30).reshape(1, -1)
    m = solve_accuracy(sol, sol, puzzle)
    assert m["cell_accuracy"] == 1.0
    assert m["board_solved"] == 1.0


def test_solve_iter_yields_one_board_per_step_plus_initial():
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(3)
    puzzle = make_puzzle(random_solution(rng), rng, num_clues=40).reshape(1, -1)

    boards = list(
        solve_iter(model, puzzle, jax.random.key(0), SamplerConfig(num_steps=6))
    )
    assert len(boards) == 7  # initial + 6 steps
    # Givens are clamped in every intermediate canvas.
    clues = puzzle[0] != 0
    for b in boards:
        np.testing.assert_array_equal(b[0][clues], puzzle[0][clues])
    # Draining the generator agrees with solve().
    final = solve(model, puzzle, jax.random.key(0), SamplerConfig(num_steps=6))
    np.testing.assert_array_equal(final, boards[-1])


# --- visualization ---------------------------------------------------------


def test_render_board_colors_givens_correct_and_incorrect():
    solution = (np.arange(81) % 9 + 1).astype(np.int32)
    given_mask = np.zeros(81, dtype=bool)
    given_mask[0] = True  # cell 0 is a clue -> blue
    board = solution.copy()
    board[1] = solution[1]  # correct guess -> green
    board[2] = (solution[2] % 9) + 1  # wrong guess -> red

    colored = render_board(board, given_mask=given_mask, solution=solution, color=True)
    assert "\033[44m" in colored  # blue (given)
    assert "\033[42m" in colored  # green (correct)
    assert "\033[41m" in colored  # red (incorrect)

    # color=False is plain text with no escape codes.
    plain = render_board(board, given_mask=given_mask, solution=solution, color=False)
    assert "\033[" not in plain


def test_save_load_round_trip(tmp_path):
    from experiments.sudoku_diffusion.checkpoint import load_model, save_model

    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))

    rng = np.random.default_rng(8)
    boards = jnp.asarray(next(solved_board_batches(2, rng)))
    before = model(boards)

    path = save_model(model, str(tmp_path / "ckpt"))
    restored = load_model(path)

    # Config survives the round trip and predictions match exactly.
    assert restored.config == cfg
    np.testing.assert_array_equal(np.asarray(restored(boards)), np.asarray(before))


def test_visualize_solving_runs_without_tty():
    import io

    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    rng = np.random.default_rng(7)
    sol = random_solution(rng).reshape(-1)
    puzzle = make_puzzle(sol.reshape(9, 9), rng, num_clues=40).reshape(-1)

    buf = io.StringIO()  # not a TTY -> color auto-disabled, no escape codes
    final = visualize_solving(
        model,
        puzzle,
        sol,
        jax.random.key(0),
        config=SamplerConfig(num_steps=4),
        delay=0.0,
        stream=buf,
    )
    assert final.shape == puzzle.shape
    out = buf.getvalue()
    assert "step" in out and "\033[" not in out
