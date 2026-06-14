"""Smoke tests for the Sudoku diffusion model and tokenizer.

Run with::

    JAX_PLATFORMS=cpu uv run python -m pytest experiments/sudoku_diffusion/test_sudoku_diffusion.py -q
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from experiments.sudoku_diffusion.model import (
    ModelConfig,
    SudokuDiffusion,
    count_params,
)
from experiments.sudoku_diffusion.tokenizer import SudokuTokenizer

BOARD = (
    "5 3 0 0 7 0 0 0 0\n"
    "6 0 0 1 9 5 0 0 0\n"
    "0 9 8 0 0 0 0 6 0\n"
    "8 0 0 0 6 0 0 0 3\n"
    "4 0 0 8 0 3 0 0 1\n"
    "7 0 0 0 2 0 0 0 6\n"
    "0 6 0 0 0 0 2 8 0\n"
    "0 0 0 4 1 9 0 0 5\n"
    "0 0 0 0 8 0 0 7 9"
)


def test_tokenizer_roundtrip():
    tok = SudokuTokenizer()
    ids = tok.encode(BOARD)
    assert len(ids) == 81
    assert all(0 <= i <= 9 for i in ids)
    # Decoding back as a grid reproduces the original text exactly.
    assert tok.decode(ids, as_grid=True) == BOARD


def test_tokenizer_specials():
    tok = SudokuTokenizer()
    assert tok.vocab_size == 12
    assert tok.pad_id == 10
    assert tok.mask_id == 11
    assert tok.digit_ids == list(range(10))
    # Mask renders as '?', pad is dropped.
    text = tok.decode([tok.mask_id, 5, tok.pad_id], as_grid=False)
    assert text == "? 5"


def test_grid_helpers():
    tok = SudokuTokenizer()
    grid = tok.decode_grid(tok.encode(BOARD))
    assert grid.shape == (9, 9)
    assert grid[0, 0] == 5 and grid[0, 2] == 0
    np.testing.assert_array_equal(tok.encode_grid(grid), np.array(tok.encode(BOARD)))


@pytest.mark.parametrize("self_cond", [False, True])
def test_forward_shapes(self_cond):
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    cfg.use_self_conditioning = self_cond
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))

    b, l = 2, tok.num_cells
    rng = jax.random.PRNGKey(1)
    tokens = jax.random.randint(rng, (b, l), 0, tok.vocab_size)
    logits = model(tokens)
    assert logits.shape == (b, l, tok.vocab_size)
    assert jnp.isfinite(logits).all()


def test_self_conditioning_changes_output():
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    b, l = 1, tok.num_cells
    tokens = jnp.full((b, l), tok.mask_id, dtype=jnp.int32)

    logits0 = model(tokens)
    logits1 = model(tokens, self_cond_logits=logits0)
    # Feeding a non-zero self-conditioning signal must change the prediction.
    assert not jnp.allclose(logits0, logits1)


def test_padding_mask_isolates_padding():
    """Predictions over real tokens must not depend on padded positions."""
    tok = SudokuTokenizer()
    cfg = ModelConfig.sudoku_tiny(vocab_size=tok.vocab_size)
    cfg.use_self_conditioning = False
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))

    real = 6
    total = 10
    tokens_a = jnp.concatenate(
        [jnp.arange(real, dtype=jnp.int32), jnp.full((total - real,), tok.pad_id)]
    )[None]
    tokens_b = tokens_a.at[:, real:].set(jnp.int32(3))  # different padding content
    mask = (jnp.arange(total) < real)[None]

    out_a = model(tokens_a, padding_mask=mask)
    out_b = model(tokens_b, padding_mask=mask)
    np.testing.assert_allclose(out_a[:, :real], out_b[:, :real], rtol=1e-5, atol=1e-5)


def test_param_count_small_is_about_100mb():
    cfg = ModelConfig.sudoku_small()
    model = SudokuDiffusion(cfg, rngs=nnx.Rngs(0))
    n = count_params(model)
    # ~45M params -> ~90 MB at 2 bytes (bf16).
    assert 30_000_000 < n < 70_000_000
    assert 60 < n * 2 / 1e6 < 140  # MB in bf16
