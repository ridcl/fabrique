"""Uniform-state diffusion sampler -- solving a Sudoku by iterative denoising.

This re-implements DiffusionGemma's sampling loop (``_sampler.py``:
``DiffusionSampler`` / ``SampleFromPredictions`` / ``AnnealingTemperatureShaper``)
for a single, fixed-size Sudoku canvas, dropping the autoregressive block / KV
cache machinery that the toy task does not need.

The loop, per denoising step (current noise level falls ``1 -> 0`` linearly):

1. **Predict.** One bidirectional forward pass gives per-cell logits.
2. **Temperature.** Logits are divided by a temperature that anneals from
   ``max_temperature`` (fully noisy) down to ``min_temperature`` (clean), so
   early steps explore and late steps commit.
3. **Accept by confidence.** Cells are ranked by prediction entropy (ascending)
   and accepted greedily while their cumulative entropy stays under
   ``entropy_bound`` -- i.e. only the most-confident cells are committed.
4. **Renoise the rest.** Every *non*-accepted cell is overwritten with a fresh
   uniform random digit.  This is the hallmark of uniform-state diffusion: an
   earlier guess is never frozen; the model may revise it on a later step.
5. **Self-condition.** The (temperature-scaled) logits are fed back into the
   next step via the model's self-conditioning path.

For *solving* (as opposed to unconditional generation) the puzzle's givens are
clamped: those cells are always "accepted" and never renoised, so the model only
ever fills the blanks.

The loop here is a plain Python ``for`` over steps calling a JIT-compiled single
step -- clearer than a ``lax.while_loop`` and plenty fast for evaluation.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from experiments.sudoku_diffusion.model import SudokuDiffusion


@dataclasses.dataclass
class SamplerConfig:
    """Hyperparameters for the uniform-state denoising sampler.

    Attributes:
      num_steps: Number of denoising iterations.
      entropy_bound: Per-step confidence budget; lower commits fewer cells per
        step (more conservative, usually higher quality).
      max_temperature: Sampling temperature at full noise.
      min_temperature: Sampling temperature when fully denoised.
      temp_exponent: Shapes the anneal; 1.0 is linear in the noise level.
      noise_lo, noise_hi: Half-open digit-id range ``[lo, hi)`` used for the
        random initial canvas and for renoising; must match training.
    """

    num_steps: int = 64
    entropy_bound: float = 0.1
    max_temperature: float = 0.8
    min_temperature: float = 0.4
    temp_exponent: float = 1.0
    noise_lo: int = 1
    noise_hi: int = 10


def _make_step_fn(config: SamplerConfig):
    """Build a JIT-compiled single denoising step (DiffusionGemma-style)."""
    lo, hi = int(config.noise_lo), int(config.noise_hi)
    entropy_bound = float(config.entropy_bound)
    max_t, min_t = float(config.max_temperature), float(config.min_temperature)
    exponent = float(config.temp_exponent)

    @nnx.jit
    def step(
        model: SudokuDiffusion,
        canvas: jax.Array,  # [B, L] int
        sc_logits: jax.Array | None,  # [B, L, V] float or None (cold start)
        given_mask: jax.Array,  # [B, L] bool, True = clamped clue
        given_vals: jax.Array,  # [B, L] int
        noise_proportion: jax.Array,  # [] float, current noise level in [0, 1]
        key: jax.Array,
    ):
        b = canvas.shape[0]
        logits = model(canvas, self_cond_logits=sc_logits)  # [B, L, V] f32

        # Annealed temperature (1 - (1 - np)^exp) maps noise 1->0 to frac 0->1,
        # i.e. temperature max_t -> min_t.
        frac = 1.0 - (1.0 - noise_proportion) ** exponent
        temperature = min_t + frac * (max_t - min_t)
        shaped_full = logits / temperature  # fed back for self-conditioning

        # Restrict the categorical / entropy computation to the digit ids.
        digit_logits = shaped_full[:, :, lo:hi]  # [B, L, nd]
        k_cat, k_noise = jax.random.split(key)
        sampled = jax.random.categorical(k_cat, digit_logits, axis=-1) + lo  # [B,L]

        log_probs = jax.nn.log_softmax(digit_logits, axis=-1)
        probs = jnp.exp(log_probs)
        safe_log = jnp.where(probs == 0, 0.0, log_probs)
        entropy = -jnp.sum(probs * safe_log, axis=-1)  # [B, L]

        # Clamp givens: zero entropy so they never consume the budget, and they
        # are force-accepted below.
        entropy = jnp.where(given_mask, 0.0, entropy)

        # Greedy entropy-bounded acceptance (accept lowest-entropy cells first).
        order = jnp.argsort(entropy, axis=-1)
        sorted_entropy = jnp.take_along_axis(entropy, order, axis=-1)
        accumulated = jnp.cumsum(sorted_entropy, axis=-1)
        accept_sorted = (accumulated - sorted_entropy) <= entropy_bound
        accept = (
            jnp.zeros_like(order, dtype=jnp.bool_)
            .at[jnp.arange(b)[:, None], order]
            .set(accept_sorted)
        )
        accept = accept | given_mask

        # Accepted cells keep the model's digit; the rest are renoised.
        random_digits = jax.random.randint(k_noise, canvas.shape, minval=lo, maxval=hi)
        out = jnp.where(accept, sampled, random_digits)
        out = jnp.where(given_mask, given_vals, out)  # clues stay fixed
        return out, shaped_full

    return step


def solve_iter(
    model: SudokuDiffusion,
    puzzles: np.ndarray,  # [B, L] int; 0 = blank to fill, 1..9 = givens
    key: jax.Array,
    config: SamplerConfig | None = None,
):
    """Yield the canvas ``[B, L]`` after each denoising step.

    The first item yielded is the initial (fully noised) canvas, followed by one
    canvas per step, so the generator produces ``config.num_steps + 1`` boards in
    total.  Useful for visualising the solving process; :func:`solve` just drains
    this and returns the last board.
    """
    config = config or SamplerConfig()
    puzzles = np.asarray(puzzles, dtype=np.int32)
    given_mask = jnp.asarray(puzzles != 0)
    given_vals = jnp.asarray(puzzles)

    step_fn = _make_step_fn(config)

    # Initial canvas: clues fixed, blanks = uniform random digits.
    key, k_init = jax.random.split(key)
    rand0 = jax.random.randint(
        k_init, puzzles.shape, minval=config.noise_lo, maxval=config.noise_hi
    )
    canvas = jnp.where(given_mask, given_vals, rand0)
    yield np.asarray(canvas)

    # Noise level falls linearly from 1 (fully noisy) to 0 (clean).
    noise_proportions = 1.0 - jnp.arange(config.num_steps + 1) / config.num_steps

    sc_logits = None  # cold start -> the model uses a zero self-conditioning signal
    for i in range(config.num_steps):
        key, k_step = jax.random.split(key)
        canvas, sc_logits = step_fn(
            model,
            canvas,
            sc_logits,
            given_mask,
            given_vals,
            noise_proportions[i],
            k_step,
        )
        yield np.asarray(canvas)


def solve(
    model: SudokuDiffusion,
    puzzles: np.ndarray,  # [B, L] int; 0 = blank to fill, 1..9 = givens
    key: jax.Array,
    config: SamplerConfig | None = None,
) -> np.ndarray:
    """Solve a batch of puzzles by uniform-state iterative denoising.

    Givens (non-zero cells) are clamped throughout; blanks start as random
    digits and are refined over ``config.num_steps`` steps.

    Returns ``[B, L]`` int array of filled boards (digit ids 1..9).
    """
    canvas = None
    for canvas in solve_iter(model, puzzles, key, config):
        pass
    return canvas


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def solve_accuracy(
    filled: np.ndarray, solutions: np.ndarray, puzzles: np.ndarray
) -> dict[str, float]:
    """Score solved boards against the known solutions.

    Returns cell-level accuracy over the originally-blank cells and the fraction
    of boards that came out exactly right.
    """
    filled = np.asarray(filled)
    solutions = np.asarray(solutions).reshape(filled.shape)
    blanks = np.asarray(puzzles).reshape(filled.shape) == 0
    correct_blanks = (filled == solutions) & blanks
    cell_acc = float(correct_blanks.sum()) / float(max(int(blanks.sum()), 1))
    board_solved = float(np.all(filled == solutions, axis=-1).mean())
    return {"cell_accuracy": cell_acc, "board_solved": board_solved}
