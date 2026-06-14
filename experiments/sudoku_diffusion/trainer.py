"""Uniform-state discrete-diffusion training for the Sudoku denoiser.

This follows the *DiffusionGemma* recipe (Google DeepMind's
``hackable_diffusion_adapter``) rather than masked / absorbing-state diffusion.
The distinction is the whole point of the model:

* **Masked diffusion** (MaskGIT, LLaDA, ...) corrupts a token by turning it into
  a special ``<mask>`` placeholder.  The model always *knows* which tokens are
  noise, and once it commits a token it can never revise it.
* **Uniform-state diffusion** (DiffusionGemma) corrupts a token by replacing it
  with a *random real token*.  Noise is therefore indistinguishable from signal:
  the model has to decide, for every cell, both *whether* it is wrong and *what*
  it should be -- and because corrupted cells are ordinary tokens, the sampler is
  free to overwrite an earlier guess on a later step.

Training one step (mirrors ``SFTDiffusion.__call__`` in the reference, minus the
prompt / canvas / KV-cache machinery we don't need for a single Sudoku board):

    t        ~ U(0, 1)                              # per-example noise level
    corrupt  : each cell, w.p. t, -> a uniform random digit
    xt       = corrupted board                      # no <mask> token involved
    logits   = model(xt[, self_cond])               # one bidirectional pass
    loss     = cross-entropy(logits, x0) over the corrupted cells only

The loss is restricted to corrupted cells because the un-corrupted cells already
hold their target value, so predicting them is trivial copying.

Self-conditioning (Chen et al. 2022), exactly as DiffusionGemma does it: on a
random fraction of steps we run a first, gradient-detached pass and feed its
logits back through the model's self-conditioning path; the rest of the time the
signal is empty.  The model thus learns both regimes it meets while sampling.

The denoiser is deliberately *time-agnostic* -- DiffusionGemma's network never
receives ``t`` either (see ``_transformer.call_with_self_conditioning``); the
noise level only drives the corruption here and the temperature schedule at
sampling time (see ``sampler.py``).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from experiments.sudoku_diffusion.model import SudokuDiffusion

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DiffusionConfig:
    """Hyperparameters for uniform-state Sudoku diffusion training.

    Attributes:
      learning_rate: Peak learning rate for the warmup-cosine schedule.
      warmup_steps: Linear warmup duration (steps).
      decay_steps: Total steps for cosine decay (>= max_steps).
      end_lr_factor: end_lr = learning_rate * end_lr_factor.
      weight_decay: AdamW weight decay.
      grad_clip: Global gradient-norm clip value.
      noise_lo, noise_hi: Half-open id range ``[lo, hi)`` of the "text"
        vocabulary the diffusion lives in -- the digits.  Corruption draws
        replacement tokens uniformly from this range.  Default ``[1, 10)`` is
        digits 1..9 (a solved board never contains a 0/empty, so we never inject
        one).
      self_cond_prob: Probability, per step, of training the self-conditioning
        pass.  Ignored if the model has self-conditioning disabled.
      log_every: Print a loss summary every this many steps.
      eval_every: Run eval_fn every this many steps (None = never).
      max_steps: Stop after this many gradient steps.
    """

    learning_rate: float = 3e-4
    warmup_steps: int = 200
    decay_steps: int = 20_000
    end_lr_factor: float = 0.1
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    noise_lo: int = 1
    noise_hi: int = 10
    self_cond_prob: float = 0.5
    log_every: int = 100
    eval_every: int | None = None
    max_steps: int = 20_000


# ---------------------------------------------------------------------------
# Uniform-state corruption (the forward / noising process)
# ---------------------------------------------------------------------------


def corrupt(
    key: jax.Array,
    x0: jax.Array,  # [B, L] int, clean board (digit ids)
    t: jax.Array,  # [B] float in [0, 1], per-example noise level
    *,
    noise_lo: int,
    noise_hi: int,
) -> tuple[jax.Array, jax.Array]:
    """Replace each cell, independently with probability ``t``, by a random digit.

    Returns ``(xt, is_corrupted)`` where ``xt`` is the noised board and
    ``is_corrupted`` is the boolean mask of cells that were replaced.  Note the
    replacement is an ordinary digit token -- there is no ``<mask>`` here.
    """
    k_mask, k_tok = jax.random.split(key)
    is_corrupted = jax.random.uniform(k_mask, x0.shape, dtype=jnp.float32) < t[:, None]
    random_digits = jax.random.randint(
        k_tok, x0.shape, minval=noise_lo, maxval=noise_hi
    )
    xt = jnp.where(is_corrupted, random_digits, x0.astype(jnp.int32))
    return xt, is_corrupted


# ---------------------------------------------------------------------------
# Loss (pure -- usable outside the trainer)
# ---------------------------------------------------------------------------


def diffusion_loss(
    model: SudokuDiffusion,
    boards: jax.Array,  # [B, L] int, fully-solved boards (digit ids)
    key: jax.Array,
    *,
    noise_lo: int,
    noise_hi: int,
    self_cond_prob: float = 0.5,
) -> jax.Array:
    """Uniform-state diffusion loss for one batch of solved boards.

    Pure (no side effects): usable for eval or a custom loop, not just the
    :class:`SudokuDiffusionTrainer`.
    """
    k_t, k_corr, k_sc = jax.random.split(key, 3)
    b, _ = boards.shape

    t = jax.random.uniform(k_t, (b,), dtype=jnp.float32)
    xt, is_corrupted = corrupt(k_corr, boards, t, noise_lo=noise_lo, noise_hi=noise_hi)

    # Self-conditioning, per batch (matches the empty-vs-fed regimes of sampling;
    # "no signal" is `self_cond_logits=None`, which the model maps to a zero
    # signal -- exactly the cold start a sampler sees on its first step).
    use_self_cond = model.config.use_self_conditioning and self_cond_prob > 0.0
    if use_self_cond:
        do_sc = jax.random.bernoulli(k_sc, self_cond_prob)

        def with_sc(_):
            first = jax.lax.stop_gradient(model(xt))
            return model(xt, self_cond_logits=first)

        def without_sc(_):
            return model(xt)

        logits = jax.lax.cond(do_sc, with_sc, without_sc, operand=None)
    else:
        logits = model(xt)

    # Cross-entropy to recover the clean board, averaged over the corrupted cells
    # of the whole batch (un-corrupted cells are excluded -- predicting them is
    # trivial copying and carries no learning signal).
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), boards.astype(jnp.int32)
    )  # [B, L]
    weight = is_corrupted.astype(jnp.float32)
    return jnp.sum(ce * weight) / jnp.maximum(jnp.sum(weight), 1.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SudokuDiffusionTrainer:
    """Uniform-state diffusion trainer for :class:`SudokuDiffusion`.

    Owns the optimiser and the JIT-compiled train/eval steps; mirrors the API of
    ``fabrique.trainers.flow.FlowMatchingTrainer``.

    Example::

        model = SudokuDiffusion(ModelConfig.sudoku_tiny(), rngs=nnx.Rngs(0))
        trainer = SudokuDiffusionTrainer(model, DiffusionConfig())
        trainer.train(board_batches, key=jax.random.key(0))
    """

    def __init__(self, model: SudokuDiffusion, config: DiffusionConfig):
        self.model = model
        self.config = config
        self.optimizer = nnx.Optimizer(model, self._make_tx(), wrt=nnx.Param)
        self._train_step_jit = self._make_train_step_jit()
        self._eval_step_jit = self._make_eval_step_jit()

    # ------------------------------------------------------------------ public

    def train_step(self, boards: jax.Array, key: jax.Array) -> jax.Array:
        """Run one gradient update; returns the scalar loss."""
        return self._train_step_jit(self.model, self.optimizer, boards, key)

    def eval_loss(self, boards: jax.Array, key: jax.Array) -> jax.Array:
        """Compute the loss without a parameter update."""
        return self._eval_step_jit(self.model, boards, key)

    def train(
        self,
        data: Iterable,
        key: jax.Array,
        eval_fn: Callable[[SudokuDiffusion, int], None] | None = None,
        start_step: int = 0,
    ) -> list[float]:
        """Run the training loop over (possibly infinite) ``data`` batches."""
        cfg = self.config
        losses: list[float] = []
        step = start_step
        window_start = time.perf_counter()

        for batch in data:
            if step >= cfg.max_steps:
                break

            key, subkey = jax.random.split(key)
            loss = self.train_step(jnp.asarray(batch), subkey)
            losses.append(float(loss))
            step += 1

            if step % cfg.log_every == 0:
                window = losses[-cfg.log_every :]
                avg = sum(window) / len(window)
                elapsed = time.perf_counter() - window_start
                ms = elapsed * 1000 / cfg.log_every
                print(
                    f"step {step:7d}/{cfg.max_steps}"
                    f"  loss={avg:.4f}  {ms:.0f} ms/step"
                )
                window_start = time.perf_counter()

            if cfg.eval_every and step % cfg.eval_every == 0 and eval_fn is not None:
                eval_fn(self.model, step)

        return losses

    # ---------------------------------------------------------------- internal

    def _make_tx(self) -> optax.GradientTransformation:
        cfg = self.config
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.decay_steps,
            end_value=cfg.learning_rate * cfg.end_lr_factor,
        )
        return optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(schedule, weight_decay=cfg.weight_decay),
        )

    def _make_train_step_jit(self):
        noise_lo, noise_hi = int(self.config.noise_lo), int(self.config.noise_hi)
        self_cond_prob = float(self.config.self_cond_prob)

        @nnx.jit
        def _step(model, optimizer, boards, key):
            def loss_fn(model):
                return diffusion_loss(
                    model,
                    boards,
                    key,
                    noise_lo=noise_lo,
                    noise_hi=noise_hi,
                    self_cond_prob=self_cond_prob,
                )

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        return _step

    def _make_eval_step_jit(self):
        noise_lo, noise_hi = int(self.config.noise_lo), int(self.config.noise_hi)
        self_cond_prob = float(self.config.self_cond_prob)

        @nnx.jit
        def _eval(model, boards, key):
            return diffusion_loss(
                model,
                boards,
                key,
                noise_lo=noise_lo,
                noise_hi=noise_hi,
                self_cond_prob=self_cond_prob,
            )

        return _eval
