"""Flow-matching DiT experiment: training on ImageNet (32×32).

Trains a Diffusion Transformer with conditional flow matching on ImageNet
images centre-cropped and downscaled to IMAGE_SIZE × IMAGE_SIZE.

Algorithm: conditional flow matching (Lipman et al. 2022)
  x_t = (1-t)*noise + t*image,  target velocity = image - noise
  loss = MSE(model(x_t, t), velocity)

Time sampling: logit-normal (SD3 / Esser et al. 2024) for better coverage
of the intermediate noise levels that dominate perceptual quality.

Dataset:
  Default: ILSVRC/imagenet-1k (requires HuggingFace login).
  Set DATASET_ID to any HF image dataset, or point DATASET_DIR to a local
  directory tree of JPEG/PNG images (scanned recursively).

Model:
  Uses dit_qwen() config (depth=24, hidden=1024, heads=16) warm-started from
  Qwen/Qwen3-VL-4B-Instruct.  The checkpoint is downloaded automatically via
  huggingface_hub if not already cached.

Outputs (written to OUT_DIR):
  samples/step_NNNNNNNN.png   — 4×4 grid of generated images
  checkpoints/step_NNNNNNNN/  — orbax checkpoint of the model state

Usage:
  python experiments/flow_imagenet.py
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import random
import time
from collections.abc import Callable, Iterator

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from fabrique.models.dit.model import DiT, compute_rope, dit_qwen
from fabrique.trainers.flow import FlowMatchingTrainer, FMConfig
from fabrique.utils import show_hbm_usage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# --- Image / model ---
IMAGE_SIZE: int = 64  # spatial resolution fed to the model
PATCH_SIZE: int = 16  # matches Qwen3-VL native; yields (IS/PS)² = 16 tokens
IN_CHANNELS: int = 3
MODEL_ID: str = "Qwen/Qwen3-VL-4B-Instruct"

# --- Data ---
# HuggingFace dataset ID, or None to use DATASET_DIR.
# ILSVRC/imagenet-1k requires `huggingface-cli login`.
DATASET_ID: str | None = "ILSVRC/imagenet-1k"
# Fallback: local directory tree of images (scanned recursively for *.jpg/png)
DATASET_DIR: str | None = None
DATASET_SPLIT: str = "train"
SHUFFLE_BUFFER: int = 10_000

# --- Training ---
BATCH_SIZE: int = 32  # TODO: use fsdp to train on 2 GPUs in parallel
MAX_STEPS: int = 500_000
LEARNING_RATE: float = 1e-4
WARMUP_STEPS: int = 2_000
DECAY_STEPS: int = MAX_STEPS
END_LR_FACTOR: float = 0.1
WEIGHT_DECAY: float = 0.01
GRAD_CLIP: float = 1.0
LOGIT_MEAN: float = 0.0
LOGIT_STD: float = 1.0

# --- Logging / outputs ---
LOG_EVERY: int = 200
SAMPLE_EVERY: int = 1_000  # generate image grid
SAVE_EVERY: int = 1_000  # orbax checkpoint
NUM_SAMPLES: int = 16  # images per sample grid (must be a perfect square)
SAMPLER_STEPS: int = 50  # Euler steps for generation
OUT_DIR: str = "/data/flow/dit_imagenet32"


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(img: Image.Image, size: int) -> np.ndarray:
    """Centre-crop, resize, normalise a PIL image → float32 [H, W, C] in [-1, 1]."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 127.5 - 1.0


# ---------------------------------------------------------------------------
# Dataset / data iterators
# ---------------------------------------------------------------------------


def _hf_image_iter(dataset_id: str, split: str, shuffle_buffer: int):
    """Yield preprocessed images one at a time from a HuggingFace dataset."""
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(
        dataset_id, split=split, streaming=True, trust_remote_code=True
    )
    ds = ds.shuffle(seed=random.randint(0, 2**31), buffer_size=shuffle_buffer)
    for example in ds:
        # HF image datasets use either 'image' or 'img' as the column name
        pil = example.get("image") or example.get("img")
        if pil is None:
            continue
        yield preprocess_image(pil, IMAGE_SIZE)


def _local_image_iter(root: str, shuffle_buffer: int):
    """Yield preprocessed images scanned recursively from a local directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in pathlib.Path(root).rglob("*") if p.suffix.lower() in exts]
    random.shuffle(paths)
    log.info("Found %d images under %s", len(paths), root)
    while True:
        random.shuffle(paths)
        for p in paths:
            try:
                yield preprocess_image(Image.open(p), IMAGE_SIZE)
            except Exception:
                continue


def batch_iter(
    image_gen,
    batch_size: int,
    dtype=np.float32,
) -> Iterator[jax.Array]:
    """Collate individual images into batched JAX arrays [B, H, W, C]."""
    buf: list[np.ndarray] = []
    for img in image_gen:
        buf.append(img)
        if len(buf) == batch_size:
            yield jnp.array(np.stack(buf, axis=0), dtype=dtype)
            buf = []


def make_data_iter(dtype=jnp.bfloat16) -> Iterator[jax.Array]:
    """Return an (effectively infinite) iterator of image batches."""
    if DATASET_DIR is not None:
        log.info("Loading images from local directory: %s", DATASET_DIR)
        gen = _local_image_iter(DATASET_DIR, SHUFFLE_BUFFER)
    elif DATASET_ID is not None:
        log.info("Streaming dataset %s (split=%s)", DATASET_ID, DATASET_SPLIT)

        def _cycling_gen():
            while True:
                yield from _hf_image_iter(DATASET_ID, DATASET_SPLIT, SHUFFLE_BUFFER)

        gen = _cycling_gen()
    else:
        raise ValueError("Set either DATASET_ID or DATASET_DIR.")
    return batch_iter(gen, BATCH_SIZE, dtype=dtype)


# ---------------------------------------------------------------------------
# Euler sampler
# ---------------------------------------------------------------------------


def make_euler_sampler(
    model: DiT,
    cos: jax.Array,
    sin: jax.Array,
    num_steps: int,
    num_samples: int,
) -> "Callable[[jax.Array], jax.Array]":
    """Return a JIT-compiled Euler sampler.

    All shape parameters are closed over as Python constants so the traced
    shapes are always concrete.

    Args:
      model: DiT model.
      cos, sin: Precomputed RoPE tables.
      num_steps: Number of Euler integration steps.
      num_samples: Batch size for generation (fixed at compile time).

    Returns:
      Callable ``sample(key) → float32 [num_samples, H, W, C]`` in [0, 1].
    """
    # Close over Python ints so jax.random.normal receives a concrete shape.
    n = int(num_samples)
    h = int(IMAGE_SIZE)
    w = int(IMAGE_SIZE)
    c = int(IN_CHANNELS)
    dt = float(1.0 / num_steps)

    @nnx.jit
    def _sample(key: jax.Array) -> jax.Array:
        x = jax.random.normal(key, (n, h, w, c), dtype=jnp.bfloat16)

        def step_fn(i, x):
            t = jnp.full((n,), i * dt, dtype=jnp.float32)
            v = model(x, t, cos, sin)
            return (x + dt * v.astype(x.dtype)).astype(x.dtype)

        x = jax.lax.fori_loop(0, num_steps, step_fn, x)
        return jnp.clip((x.astype(jnp.float32) + 1.0) / 2.0, 0.0, 1.0)

    return _sample


# ---------------------------------------------------------------------------
# Utilities: image grid, checkpointing
# ---------------------------------------------------------------------------


def save_image_grid(images: np.ndarray, path: str) -> None:
    """Save [B, H, W, C] float32 array in [0,1] as a square PNG grid."""
    B, H, W, C = images.shape
    n = int(B**0.5)
    assert n * n == B, f"NUM_SAMPLES={B} must be a perfect square"
    grid = np.zeros((H * n, W * n, C), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, n)
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = (
            (img * 255).clip(0, 255).astype(np.uint8)
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)
    log.info("Saved sample grid → %s", path)


def find_latest_checkpoint(ckpt_root: str) -> tuple[str, int] | None:
    """Return (path, step) of the most recent checkpoint, or None."""
    if not os.path.isdir(ckpt_root):
        return None
    candidates = []
    for name in os.listdir(ckpt_root):
        if name.startswith("step_"):
            try:
                candidates.append((int(name[5:]), os.path.join(ckpt_root, name)))
            except ValueError:
                pass
    if not candidates:
        return None
    step, path = max(candidates)
    return path, step


def save_checkpoint(
    model: DiT, optimizer: nnx.Optimizer, step: int, ckpt_root: str
) -> None:
    """Save model weights and optimiser state via orbax."""
    import orbax.checkpoint as ocp

    ckpt_dir = os.path.join(ckpt_root, f"step_{step:08d}")
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        ckpt_dir,
        {"model": nnx.state(model), "opt": nnx.state(optimizer, nnx.OptState)},
        force=True,
    )
    log.info("Checkpoint saved → %s", ckpt_dir)


def restore_checkpoint(model: DiT, optimizer: nnx.Optimizer, ckpt_dir: str) -> None:
    """Restore model weights and optimiser state in-place."""
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(lambda: model)
    abstract_opt = nnx.eval_shape(lambda: optimizer)
    target = {
        "model": nnx.state(abstract_model),
        "opt": nnx.state(abstract_opt, nnx.OptState),
    }
    restored = checkpointer.restore(ckpt_dir, target=target)
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["opt"])
    log.info("Checkpoint restored ← %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id: str) -> str:
    """Return a local directory for *model_id*, downloading via HF Hub if needed."""
    if os.path.isdir(model_id):
        return model_id
    log.info("Downloading %s from HuggingFace Hub…", model_id)
    return huggingface_hub.snapshot_download(model_id)


def make_model(model_dir: str) -> DiT:
    """Warm-start a DiT from the Qwen3-VL vision encoder weights."""
    from fabrique.models.dit import params as dit_params

    config = dit_qwen()  # patch_size=16 and in_channels=3 are already the defaults
    log.info(
        "Loading DiT blocks from %s  (depth=%d hidden=%d heads=%d patch=%d)",
        model_dir,
        config.depth,
        config.hidden_size,
        config.num_heads,
        config.patch_size,
    )
    model = dit_params.load_from_qwen_vl(model_dir, config, dtype=jnp.bfloat16)
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    log.info("Model ready — %.1fM params total", n_params / 1e6)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_dir = os.path.join(OUT_DIR, "samples")
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")

    # ── Model ────────────────────────────────────────────────────────────────
    model_dir = resolve_model_dir(MODEL_ID)
    model = make_model(model_dir)
    show_hbm_usage()

    _cfg = model.config
    head_dim = _cfg.hidden_size // _cfg.num_heads
    patch_size = model.config.patch_size
    cos, sin = compute_rope(
        IMAGE_SIZE // patch_size, IMAGE_SIZE // patch_size, head_dim
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    fm_cfg = FMConfig(
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_lr_factor=END_LR_FACTOR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        logit_mean=LOGIT_MEAN,
        logit_std=LOGIT_STD,
        log_every=LOG_EVERY,
        eval_every=SAMPLE_EVERY,
        max_steps=MAX_STEPS,
    )
    trainer = FlowMatchingTrainer(
        model, fm_cfg, image_shape=(IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS)
    )

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    start_step = 0
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is not None:
        ckpt_path, start_step = latest
        log.info("Resuming from step %d (%s)", start_step, ckpt_path)
        restore_checkpoint(model, trainer.optimizer, ckpt_path)
    else:
        log.info("No checkpoint found — starting from scratch.")

    # Build sampler once (JIT compiles on first call)
    euler_sample = make_euler_sampler(model, cos, sin, SAMPLER_STEPS, NUM_SAMPLES)

    # ── eval_fn: sample + checkpoint ─────────────────────────────────────────
    sample_key = jax.random.key(999)

    def eval_fn(model: DiT, step: int) -> None:
        nonlocal sample_key
        sample_key, k = jax.random.split(sample_key)

        images = np.array(euler_sample(k))
        grid_path = os.path.join(sample_dir, f"step_{step:08d}.png")
        save_image_grid(images, grid_path)

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, trainer.optimizer, step, ckpt_dir)

    # ── Data iterator ────────────────────────────────────────────────────────
    log.info(
        "Building data pipeline (IMAGE_SIZE=%d, BATCH_SIZE=%d)…", IMAGE_SIZE, BATCH_SIZE
    )
    data = make_data_iter(dtype=jnp.bfloat16)

    # ── Train ────────────────────────────────────────────────────────────────
    log.info("Starting training for %d steps.", MAX_STEPS)
    log.info("Outputs → %s", OUT_DIR)
    key = jax.random.key(0)
    t0 = time.perf_counter()
    trainer.train(data, key=key, eval_fn=eval_fn, start_step=start_step)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f h.", elapsed / 3600)

    # Final checkpoint
    save_checkpoint(model, trainer.optimizer, MAX_STEPS, ckpt_dir)


if __name__ == "__main__" and "__file__" in globals():
    main()
