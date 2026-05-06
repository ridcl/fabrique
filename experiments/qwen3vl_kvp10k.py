"""KVP10K experiment: evaluate and fine-tune Qwen3-VL-4B-Instruct on KVP extraction.

Steps:
1. Load KVP10K dataset from Hyperspace (extractions/kvp10k-with-images/v1)
2. Evaluate base Qwen3-VL-4B-Instruct on test split
   - F1 of exact value match (per document, then macro-averaged)
   - Mean IoU of predicted vs. ground-truth bounding boxes (on exact-match hits)
3. Fine-tune on train split with LoRA
4. Re-evaluate on test split
"""

import io
import json
import logging
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import qwix
from flax import nnx
from PIL import Image, ImageDraw
from tqdm import tqdm
from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl import params as params_lib
from fabrique.models.qwen3vl.sampler import (
    Qwen3VLSampler,
    load_sampler,
    resolve_model_dir,
)
from fabrique.models.qwen3vl.utils import encode_messages, load_processor
from fabrique.models.qwen3vl.vision import VisionGridData
from fabrique.saving import save_qwen3vl_lora_merged
from tunix.rl import reshard as reshard_lib
from tunix.sft import metrics_logger, peft_trainer
from fabrique.utils import show_hbm_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATH = "/data/kvp10k-with-images.parquet"

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# OUTPUT_DIR = "/data/models/kvp10k-qwen3vl-4b"
OUTPUT_DIR = "/data/models/kvp10k-qwen3vl-4b-retrained"
# LORA_CKPT_DIR = "/tmp/kvp10k_lora_ckpts"
LORA_CKPT_DIR = "/data/cache/kvp10k_lora_ckpts_retrained"

BATCH_SIZE = 1
MAX_SEQ_LEN = 4096
# MAX_IMAGE_SIZE = 1280
MAX_IMAGE_SIZE = 896
EVAL_CACHE_SIZE = 4096
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_SAMPLES = 500

LORA_RANK = 16
LORA_ALPHA = float(2 * LORA_RANK)
_LORA_TARGETS = ".*q_proj|.*k_proj|.*gate_proj|.*up_proj|.*down_proj"
MAX_STEPS = 10_000
EVAL_EVERY_N_STEPS = 500

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATASET_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    # Accept "test" or "val"/"validation" as the held-out split
    test_df = df[~df["split"].isin(["train"])].reset_index(drop=True)
    logger.info("Train: %d rows, Test/Val: %d rows", len(train_df), len(test_df))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------


def _load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def _make_prompt(keys: list[str]) -> str:
    key_list = "\n".join(f"- {k}" for k in keys)
    return (
        "Extract the following key-value pairs from this document. "
        'Return a JSON list where each item has "key", "value", and "bbox" fields. '
        'The "box_2d" is the bounding box of the value text in the format '
        '{"box_2d": [x0, y0, x1, y1]} with coordinates in 0–1000 range.\n'
        f"Keys to extract:\n{key_list}"
    )


def _gt_bbox_to_qwen(bbox: list[float]) -> list[int]:
    """Ground-truth [x0,y0,x1,y1] (0–1 normalised) → Qwen [x0,y0,x1,y1] (0–1000)."""
    x0, y0, x1, y1 = bbox
    return [round(x0 * 1000), round(y0 * 1000), round(x1 * 1000), round(y1 * 1000)]


def _qwen_bbox_to_normalized(bbox: list[int]) -> list[float]:
    """Qwen [x0,y0,x1,y1] (0–1000) → [x0,y0,x1,y1] (0–1 normalised)."""
    x0, y0, x1, y1 = bbox
    return [x0 / 1000, y0 / 1000, x1 / 1000, y1 / 1000]


def _make_target(kvps: list[dict]) -> str:
    targets = [
        {
            "key": kvp["key"],
            "value": kvp["value"],
            "box_2d": _gt_bbox_to_qwen(kvp["bounding_box"]),
        }
        for kvp in kvps
    ]
    return json.dumps(targets)


def make_conversation(row: pd.Series) -> list[dict]:
    image = _load_image(row["image"])
    kvps = list(row["kvps"])
    keys = [kvp["key"] for kvp in kvps]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _make_prompt(keys)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": _make_target(kvps)}],
        },
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x0, y0, x1, y1] normalised bounding boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compute_metrics(
    predictions: list[str],
    ground_truths: list[list[dict]],
) -> dict[str, float]:
    """Compute macro-averaged F1 (exact value match) and mean IoU.

    F1 is computed per document over key-level precision/recall, then averaged.
    IoU is computed only for keys where the value matches exactly.

    Args:
        predictions: Raw model output strings (expected to be JSON).
        ground_truths: List of KVP lists; each KVP has {key, value, bounding_box}.
    """
    all_f1: list[float] = []
    all_iou: list[float] = []
    n_parse_errors = 0

    for pred_str, gt_kvps in zip(predictions, ground_truths):
        try:
            pred_list = json.loads(pred_str)
            if not isinstance(pred_list, list):
                pred_list = []
        except (json.JSONDecodeError, ValueError):
            pred_list = []
            n_parse_errors += 1

        pred_by_key = {
            str(p.get("key", "")).strip().lower(): p
            for p in pred_list
            if isinstance(p, dict)
        }

        tp = 0
        for gt in gt_kvps:
            key_norm = str(gt["key"]).strip().lower()
            pred = pred_by_key.get(key_norm)
            if pred is None:
                continue
            if str(pred.get("value", "")).strip() == str(gt["value"]).strip():
                tp += 1
                try:
                    pred_norm = _qwen_bbox_to_normalized(pred["box_2d"])
                    all_iou.append(_iou(pred_norm, list(gt["bounding_box"])))
                except (KeyError, TypeError, ValueError):
                    pass

        n_pred = len(pred_list)
        n_gt = len(gt_kvps)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        denom = precision + recall
        all_f1.append(2 * precision * recall / denom if denom > 0 else 0.0)

    if n_parse_errors:
        logger.warning("JSON parse errors: %d / %d", n_parse_errors, len(predictions))

    return {
        "f1_exact_match": float(np.mean(all_f1)) if all_f1 else 0.0,
        "iou": float(np.mean(all_iou)) if all_iou else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# def sample_from_model(lora_model, processor, prompts, images):

#     try:
#         sampler = Qwen3VLSampler(lora_model, processor, cache_size=EVAL_CACHE_SIZE)
#         return sampler(prompts=prompts, images=images, max_new_tokens=EVAL_MAX_NEW_TOKENS)
#     finally:
#         lora_model.config.remat_config = model_lib.RematConfig.BLOCK


def visualize_predictions(
    image: Image.Image,
    output: str,
    gt_kvps: list[dict],
    output_path: str,
) -> None:
    """Draw GT boxes (blue) and predicted boxes (red) on image and save."""
    try:
        items = json.loads(output)
        if not isinstance(items, list):
            items = []
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse model output for visualization")
        items = []

    W, H = image.size
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    for kvp in gt_kvps:
        bb = kvp.get("bounding_box")
        if bb is None or len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        draw.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], outline="blue", width=2)

    for item in items:
        bbox = item.get("box_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        px0, py0 = x0 / 1000 * W, y0 / 1000 * H
        px1, py1 = x1 / 1000 * W, y1 / 1000 * H
        draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
        draw.text((px0, max(0, py0 - 12)), item.get("key", ""), fill="red")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis.save(output_path)


def show_row(lora_model, processor, row):
    image = _load_image(row["image"])
    messages = make_conversation(row)
    del messages[-1]
    lora_model.config.remat_config = model_lib.RematConfig.NONE
    try:
        sampler = Qwen3VLSampler(lora_model, processor, cache_size=EVAL_CACHE_SIZE)
        prompt = sampler._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        output = sampler(
            prompts=[prompt],
            images=[image],
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
        )[0]
        print(output)
    finally:
        lora_model.config.remat_config = model_lib.RematConfig.BLOCK

    visualize_predictions(image, output, list(row["kvps"]), "output/out.jpeg")
    print("Saved to output/out.jpeg")


def evaluate(
    sampler,
    df: pd.DataFrame,
    output_dir: str | None = None,
    max_samples: int = EVAL_MAX_SAMPLES,
) -> dict[str, float]:
    df = df.head(max_samples)
    predictions: list[str] = []
    ground_truths: list[list[dict]] = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), desc="Evaluating", total=df.shape[0])
    ):
        image = _load_image(row["image"])
        messages = make_conversation(row)
        del messages[-1]
        prompt = sampler._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        output = sampler(
            prompts=[prompt],
            images=[image],
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
        )[0]
        predictions.append(output)
        ground_truths.append(list(row["kvps"]))
        if output_dir is not None:
            visualize_predictions(
                image,
                output,
                list(row["kvps"]),
                os.path.join(output_dir, f"row_{i:03}.jpeg"),
            )

    return compute_metrics(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def _gen_model_input_fn(batch) -> dict:
    return {
        "input_tokens": jnp.array(batch.input_tokens),
        "padding_mask": jnp.array(batch.input_mask).astype(jnp.bool_),
        "completion_mask": jnp.array(batch.completion_mask),
        "positions": jnp.array(batch.positions),
        "pixel_values": jnp.array(batch.pixel_values, dtype=jnp.bfloat16),
        "vision_grid": batch.vision_grid,
    }


def _loss_fn(
    model: model_lib.Qwen3VL,
    input_tokens: jax.Array,
    positions: jax.Array,
    pixel_values: jax.Array,
    vision_grid: VisionGridData,
    padding_mask: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
    logits, _ = model(
        input_tokens,
        positions,
        pixel_values,
        vision_grid,
        cache=None,
        padding_mask=padding_mask,
    )
    logits = logits[:, :-1, :]
    targets = input_tokens[:, 1:]
    mask = completion_mask[:, 1:].astype(jnp.float32)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets
    )
    return jnp.sum(token_loss * mask) / jnp.sum(mask)


class _DataLoader:
    """Iterator that encodes KVP10K rows as batched EncodedBatch objects."""

    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        vcfg: model_lib.VisionModelConfig,
        batch_size: int,
        max_seq_len: int,
        num_epochs: int = 1,
    ):
        self._df = df
        self._processor = processor
        self._vcfg = vcfg
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._num_epochs = num_epochs

    def __iter__(self):
        for epoch in range(self._num_epochs):
            df = self._df.sample(frac=1, random_state=RANDOM_SEED + epoch)
            buffer: list[list[dict]] = []
            for _, row in df.iterrows():
                buffer.append(make_conversation(row))
                if len(buffer) == self._batch_size:
                    yield encode_messages(
                        self._processor,
                        buffer,
                        loss_roles={"assistant"},
                        vcfg=self._vcfg,
                        max_seq_len=self._max_seq_len,
                        padding=True,
                        pad_to_multiple_of=1024,
                        truncation=True,
                    )
                    buffer = []


def _get_lora_model(
    base_model: model_lib.Qwen3VL,
    mesh: jax.sharding.Mesh,
) -> model_lib.Qwen3VL:
    lora_provider = qwix.LoraProvider(
        module_path=_LORA_TARGETS, rank=LORA_RANK, alpha=LORA_ALPHA
    )
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **base_model.get_model_input()
    )
    # Fix sharding metadata: LoRA A/B are rank-2 but inherit rank-3 specs from
    # Einsum weights — trim the extra axis so nnx.get_partition_spec is valid.
    for _, node in nnx.iter_graph(lora_model):
        if isinstance(node, nnx.Variable) and node.has_metadata("out_sharding"):
            sharding = node.get_metadata()["out_sharding"]
            if sharding and len(sharding) > len(node.shape):
                node.set_metadata("out_sharding", tuple(sharding[: len(node.shape)]))
    with mesh:
        graph_def, state = nnx.split(lora_model)
        default_memory_kind = jax.devices()[0].default_memory().kind
        dst_shardings = jax.tree_util.tree_map(
            lambda x: jax.sharding.NamedSharding(
                mesh, x, memory_kind=default_memory_kind
            ),
            nnx.get_partition_spec(state),
        )
        lora_model = nnx.merge(
            graph_def, reshard_lib.reshard_pytree(state, dst_shardings)
        )
    return lora_model


def train(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    os.makedirs(LORA_CKPT_DIR, exist_ok=True)

    config = model_lib.ModelConfig.qwen3vl_4b()
    config.remat_config = model_lib.RematConfig.BLOCK
    model_dir = resolve_model_dir(MODEL_ID)

    mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
    base_model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=mesh, dtype=jnp.bfloat16
    )
    show_hbm_usage()

    processor = load_processor(model_dir)
    lora_model = _get_lora_model(base_model, mesh=mesh)
    show_hbm_usage()

    # TEMP: Visualize before training
    eval_rows = [row for _, row in eval_df.iterrows()]
    show_row(lora_model, processor, eval_rows[8])

    num_epochs = math.ceil(MAX_STEPS / len(train_df))
    train_loader = _DataLoader(
        train_df,
        processor,
        config.vision_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_epochs=num_epochs,
    )
    eval_loader = _DataLoader(
        eval_df.head(20),
        processor,
        config.vision_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_epochs=1,
    )

    logging_opts = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/kvp10k_qwen3vl",
        flush_every_n_steps=EVAL_EVERY_N_STEPS,
    )
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        metrics_logging_options=logging_opts,
        checkpoint_root_directory=LORA_CKPT_DIR,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=2e-4,
                warmup_steps=50,
                decay_steps=MAX_STEPS,
                end_value=1e-5,
            ),
            weight_decay=0.01,
        ),
    )
    trainer = peft_trainer.PeftTrainer(
        lora_model, optimizer, training_config
    ).with_gen_model_input_fn(_gen_model_input_fn)
    trainer.loss_fn = _loss_fn
    trainer.eval_loss_fn = _loss_fn

    logger.info("Starting LoRA fine-tuning for %d steps", MAX_STEPS)
    with mesh:
        trainer.train(train_loader, eval_ds=eval_loader)

    logger.info("Saving merged LoRA model to %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_qwen3vl_lora_merged(
        local_model_path=model_dir,
        output_dir=OUTPUT_DIR,
        lora_model=lora_model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # jax.config.update('jax_explain_cache_misses', True)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    train_df, eval_df = load_splits()

    # --- Evaluate base model ---
    logger.info("Loading base model for evaluation…")
    sampler = load_sampler(MODEL_ID, cache_size=EVAL_CACHE_SIZE)
    logger.info(
        "Evaluating base model on %d test samples…", min(EVAL_MAX_SAMPLES, len(eval_df))
    )

    metrics_before = evaluate(sampler, eval_df, output_dir="output/base_model")
    logger.info(
        "Base model  — F1: %.4f  IoU: %.4f",
        metrics_before["f1_exact_match"],
        metrics_before["iou"],
    )
    del sampler  # free HBM before training

    # --- Fine-tune ---
    train(train_df, eval_df)

    # --- Evaluate fine-tuned model ---
    logger.info("Loading fine-tuned model for evaluation…")
    sampler_ft = load_sampler(OUTPUT_DIR, cache_size=EVAL_CACHE_SIZE)
    metrics_after = evaluate(sampler_ft, eval_df, output_dir="output/ft_model")
    logger.info(
        "Fine-tuned  — F1: %.4f  IoU: %.4f",
        metrics_after["f1_exact_match"],
        metrics_after["iou"],
    )

    print("\n=== Results ===")
    print(
        f"Base model:  F1={metrics_before['f1_exact_match']:.4f}  IoU={metrics_before['iou']:.4f}"
    )
    print(
        f"Fine-tuned:  F1={metrics_after['f1_exact_match']:.4f}  IoU={metrics_after['iou']:.4f}"
    )


if __name__ == "__main__" and "__file__" in globals():
    main()
