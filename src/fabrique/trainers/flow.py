"""Flow-matching trainer for DiT pixel-space image generation.

Algorithm (conditional flow matching, Lipman et al. 2022):
    x_t     = (1 - t) * x_noise + t * x_data   # linear interpolant
    v_target = x_data - x_noise                  # constant velocity
    loss     = MSE(model(x_t, t), v_target)

Time distribution uses logit-normal sampling (Esser et al. 2024 / SD3):
    t = sigmoid(N(logit_mean, logit_std))
This concentrates training signal near t ≈ 0.5 (high-uncertainty midpoints)
rather than the easy near-data / near-noise extremes, improving convergence
compared to uniform t at a fixed compute budget.

Expected image format: float32 or bfloat16 tensors in [-1, 1],
shape [B, H, W, C].  Normalisation is the caller's responsibility.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from fabrique.models.dit.model import DiT, compute_rope

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FMConfig:
    """Hyperparameters for flow-matching training.

    Attributes:
      learning_rate: Peak learning rate for the cosine schedule.
      warmup_steps: Linear warmup duration (steps).
      decay_steps: Total steps for cosine decay (should equal or exceed
        max_steps).
      end_lr_factor: end_lr = learning_rate * end_lr_factor.
      weight_decay: AdamW weight decay.
      grad_clip: Global gradient norm clip value.
      logit_mean: Mean of the normal distribution before sigmoid for time
        sampling.  0.0 centres mass at t=0.5.
      logit_std: Standard deviation of that normal distribution.  1.0 gives
        a broad logit-normal; increase to further concentrate near t=0.5.
      log_every: Print a loss summary every this many steps.
      eval_every: Run eval_fn every this many steps (None = never).
      max_steps: Stop training after this many gradient steps.
    """

    learning_rate: float = 1e-4
    warmup_steps: int = 500
    decay_steps: int = 100_000
    end_lr_factor: float = 0.1
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    logit_mean: float = 0.0
    logit_std: float = 1.0
    log_every: int = 100
    eval_every: int | None = None
    max_steps: int = 100_000


# ---------------------------------------------------------------------------
# Loss (reusable outside the trainer)
# ---------------------------------------------------------------------------


def fm_loss(
    model: DiT,
    images: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    key: jax.Array,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> jax.Array:
    """Flow-matching loss for one batch.

    This function is pure (no side effects) and can be used independently
    of :class:`FlowMatchingTrainer` — e.g. for eval or custom training loops.

    Args:
      model: DiT model.
      images: Normalised images [B, H, W, C], float in [-1, 1].
      cos: RoPE cosines [N, head_dim] float32 (from compute_rope).
      sin: RoPE sines   [N, head_dim] float32 (from compute_rope).
      key: PRNG key consumed for noise and time sampling.
      logit_mean: Mean parameter for logit-normal time distribution.
      logit_std: Std parameter for logit-normal time distribution.

    Returns:
      Scalar MSE loss, float32.
    """
    key_noise, key_t = jax.random.split(key)
    B = images.shape[0]
    dtype = images.dtype

    x0 = jax.random.normal(key_noise, images.shape, dtype=dtype)

    # Logit-normal time: t = sigmoid(N(mean, std)), shape [B]
    t = jax.nn.sigmoid(
        jax.random.normal(key_t, (B,), dtype=jnp.float32) * logit_std + logit_mean
    )

    # Linear interpolant and constant-velocity target
    t_bc = t[:, None, None, None].astype(dtype)
    x_t = (1.0 - t_bc) * x0 + t_bc * images
    v_target = images - x0  # [B, H, W, C]

    v_pred = model(x_t, t, cos, sin)  # [B, H, W, C]

    # Accumulate MSE in float32 to avoid bfloat16 underflow in the squared term
    return jnp.mean((v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) ** 2)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FlowMatchingTrainer:
    """Flow-matching trainer for a :class:`~fabrique.models.dit.model.DiT` model.

    Manages the optimiser, precomputed RoPE tables, and the JIT-compiled
    train / eval steps.  The caller provides an iterator over image batches
    and a PRNG key.

    Example::

        from fabrique.models.dit import DiT, dit_small
        from fabrique.trainers.fm import FlowMatchingTrainer, FMConfig

        model = DiT(dit_small(), rngs=nnx.Rngs(0))
        trainer = FlowMatchingTrainer(
            model,
            FMConfig(learning_rate=1e-4, max_steps=50_000),
            image_shape=(64, 64, 3),
        )
        losses = trainer.train(image_batch_iter, key=jax.random.key(42))
    """

    def __init__(
        self,
        model: DiT,
        config: FMConfig,
        image_shape: tuple[int, int, int],
    ):
        """
        Args:
          model: DiT instance to train (modified in-place).
          config: Training hyperparameters.
          image_shape: (H, W, C) of the training images.  H and W must be
            divisible by model.config.patch_size.
        """
        self.model = model
        self.config = config
        self.image_shape = image_shape

        H, W, _ = image_shape
        P = model.config.patch_size
        self.cos, self.sin = compute_rope(
            H // P, W // P, model.config.hidden_size // model.config.num_heads
        )

        self.optimizer = nnx.Optimizer(model, self._make_tx(), wrt=nnx.Param)

        # Build the JIT-compiled step functions, closing over Python constants.
        self._train_step_jit = self._make_train_step_jit()
        self._eval_step_jit = self._make_eval_step_jit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(
        self,
        images: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Run one gradient update step.

        Args:
          images: Normalised images [B, H, W, C] in [-1, 1].
          key: PRNG key (consumed).

        Returns:
          Scalar loss (float32).
        """
        return self._train_step_jit(
            self.model, self.optimizer, images, self.cos, self.sin, key
        )

    def eval_loss(
        self,
        images: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Compute validation loss without a parameter update.

        Args:
          images: Normalised images [B, H, W, C] in [-1, 1].
          key: PRNG key (consumed).

        Returns:
          Scalar loss (float32).
        """
        return self._eval_step_jit(self.model, images, self.cos, self.sin, key)

    def train(
        self,
        data: Iterable,
        key: jax.Array,
        eval_fn: Callable[[DiT, int], None] | None = None,
        start_step: int = 0,
    ) -> list[float]:
        """Run the training loop.

        Args:
          data: Iterable that yields image batches [B, H, W, C].
            May be infinite; training stops after config.max_steps steps.
          key: PRNG key for noise and time sampling.
          eval_fn: Optional callback ``eval_fn(model, step)`` invoked every
            ``config.eval_every`` steps.
          start_step: Step to resume from (0 = fresh start).  The optimiser
            schedule uses this as its initial count, so LR resumes correctly.

        Returns:
          List of per-step training losses (float32 scalars).
        """
        cfg = self.config
        losses: list[float] = []
        step = start_step
        window_start = time.perf_counter()

        for batch in data:
            if step >= cfg.max_steps:
                break

            key, subkey = jax.random.split(key)
            images = jnp.asarray(batch)

            loss = self.train_step(images, subkey)
            losses.append(float(loss))
            step += 1

            if step % cfg.log_every == 0:
                window = losses[-cfg.log_every :]
                avg_loss = sum(window) / len(window)
                elapsed = time.perf_counter() - window_start
                ms_per_step = elapsed * 1000 / cfg.log_every
                print(
                    f"step {step:7d}/{cfg.max_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  {ms_per_step:.0f} ms/step"
                )
                window_start = time.perf_counter()

            if cfg.eval_every and step % cfg.eval_every == 0 and eval_fn is not None:
                eval_fn(self.model, step)

        return losses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        """Build @nnx.jit train step, closing over Python-scalar config values."""
        logit_mean = float(self.config.logit_mean)
        logit_std = float(self.config.logit_std)

        @nnx.jit
        def _step(
            model: DiT,
            optimizer: nnx.Optimizer,
            images: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            def loss_fn(model):
                return fm_loss(
                    model,
                    images,
                    cos,
                    sin,
                    key,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                )

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        return _step

    def _make_eval_step_jit(self):
        """Build @nnx.jit eval step (no gradient update)."""
        logit_mean = float(self.config.logit_mean)
        logit_std = float(self.config.logit_std)

        @nnx.jit
        def _eval(
            model: DiT,
            images: jax.Array,
            cos: jax.Array,
            sin: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            return fm_loss(
                model,
                images,
                cos,
                sin,
                key,
                logit_mean=logit_mean,
                logit_std=logit_std,
            )

        return _eval
