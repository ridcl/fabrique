"""Train the cross-attention box head from ``model_1_extra_layer`` on kvp10k.

Same data, split, metrics and baseline as ``train_1.py`` -- the pipeline is
imported from it rather than copied, so the two runs are directly comparable and
only the head differs.  Qwen3-VL stays frozen; only the head trains.

What is cached differs, though.  ``train_1`` stores one pooled vector per
example; this needs the whole image-token block, ~736 x 2560 in bfloat16 =
3.7 MB per example (~2.8 GB for 756 examples), so it is staged in host memory
and moved to the device per batch.  That is the cheap end of the
pseudo-token design space: reading K layers instead of the last one multiplies
it by K, and running pseudo-tokens through the backbone's own 36 layers would
mean caching full KV, ~116 MB per example.

Extra metrics: ``peak`` and ``cent`` are the fractions of examples whose peak
attention cell, and whose attention centroid, fall inside the true box.  They
isolate the attention from the box readout -- if they climb while IoU does not,
the pseudo-token is finding the field and the readout is losing it; if neither
moves, the frozen features do not support localisation and the next lever is
unfreezing, not more head.  Peak and centroid disagreeing is itself a result: a
diffuse map whose mean happens to land on the field is memorising a centroid,
not pointing at anything.

Run::

    python -m experiments.bbox_heads.train_1_extra_layer
    python -m experiments.bbox_heads.train_1_extra_layer --rows 96 --epochs 150
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image, ImageDraw
from transformers import AutoProcessor

from experiments.bbox_heads import model_1, model_1_extra_layer, train_1
from fabrique.models.qwen3vl.loading import load_model
from fabrique.models.qwen3vl.model import ModelConfig
from fabrique.models.qwen3vl.utils import encode_batch

# force=True: importing jax/tunix installs a root logging handler, which makes
# a plain basicConfig() a silent no-op and swallows every logger.info below.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction (frozen backbone)
# ---------------------------------------------------------------------------

_features_fn = nnx.jit(model_1_extra_layer.Qwen3VLBBoxAttn.features)


def probe_image_grid(processor, image: Image.Image, merge_size: int) -> tuple[int, int]:
    """Merged image-token grid ``(rows, cols)`` for a page of this size.

    The processor applies its own smart-resize, so the grid cannot be derived
    from the page dimensions by hand; ask it once.  Every page is resized to the
    same size, so the answer holds for the whole run -- ``encode_features``
    re-checks it against the actual token count on every batch.
    """
    out = processor.image_processor(images=[image], return_tensors=None)
    thw = np.asarray(out["image_grid_thw"], dtype=int).reshape(-1, 3)[0]
    return int(thw[1]) // merge_size, int(thw[2]) // merge_size


def encode_features(
    model: model_1_extra_layer.Qwen3VLBBoxAttn,
    processor,
    examples: list[train_1.Example],
    *,
    batch_size: int,
    max_seq_len: int,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(pooled [N, D] float32, image tokens [N, n, D] bfloat16)`` on the host.

    Image tokens stay in bfloat16: they are a bfloat16 backbone's output, and
    float32 would double a multi-gigabyte cache for no added information.

    The cache lives in host memory, written chunk by chunk into a preallocated
    array.  Accumulating device chunks and concatenating at the end costs twice
    the cache in peak device memory -- 2 x 2.1 GB on top of the 8 GB backbone,
    which OOMs a 24 GB card.  Batches are moved to the device as the loop needs
    them: 118 MB per step of 32, a few tens of seconds over a whole run.
    """
    vcfg = model.backbone.config.vision_config
    assert vcfg is not None
    embed_dim = model.backbone.config.embed_dim
    pooled_out = np.empty((len(examples), embed_dim), dtype=np.float32)
    image_out = np.empty(
        (len(examples), model.n_image_tokens, embed_dim), dtype=jnp.bfloat16
    )
    t_prep = t_fwd = 0.0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        t0 = time.perf_counter()
        batch = encode_batch(
            processor,
            [train_1.build_chat_prompt(processor, e) for e in chunk],
            [[e.image] for e in chunk],
            vcfg=vcfg,
            max_length=max_seq_len,
            padding="max_length",
        )
        t_prep += time.perf_counter() - t0

        # Truncation would drop image tokens without raising, and the gather in
        # the head has a fixed output width, so a changed count is silent
        # corruption rather than an error.
        n_image = (batch.input_tokens == vcfg.image_pad_id).sum(axis=1)
        if n_image.min() != model.n_image_tokens or n_image.max() != n_image.min():
            raise RuntimeError(
                f"expected {model.n_image_tokens} image tokens per example, got "
                f"{n_image.tolist()}; --max-seq-len is truncating the image"
            )

        t0 = time.perf_counter()
        pooled, image = _features_fn(
            model,
            jnp.asarray(batch.input_tokens),
            jnp.asarray(batch.positions),
            jnp.asarray(batch.pixel_values, dtype=jnp.bfloat16),
            batch.vision_grid,
            jnp.asarray(batch.input_mask).astype(jnp.bool_),
        )
        pooled_out[start : start + len(chunk)] = np.asarray(pooled, dtype=np.float32)
        image_out[start : start + len(chunk)] = np.asarray(image.astype(jnp.bfloat16))
        t_fwd += time.perf_counter() - t0

        done = min(start + batch_size, len(examples))
        if done % (batch_size * 20) == 0 or done == len(examples):
            logger.info(
                "encode   %s %d/%d  (prep %.0fs, forward %.0fs)",
                tag,
                done,
                len(examples),
                t_prep,
                t_fwd,
            )

    return pooled_out, image_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@nnx.jit
def train_step(
    head: model_1_extra_layer.BBoxAttnHead,
    optimizer: nnx.Optimizer,
    pooled: jax.Array,  # [B, D]
    image: jax.Array,  # [B, n, D]
    targets: jax.Array,  # [B, 4]
) -> jax.Array:
    def loss_fn(head):
        boxes, _attn = head(pooled, image)
        return model_1.bbox_loss(boxes, targets)

    loss, grads = nnx.value_and_grad(loss_fn)(head)
    optimizer.update(head, grads)
    return loss


@nnx.jit
def predict(
    head: model_1_extra_layer.BBoxAttnHead, pooled: jax.Array, image: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return head(pooled, image)


def _inside(xy: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (
        (xy[:, 0] >= targets[:, 0])
        & (xy[:, 0] <= targets[:, 2])
        & (xy[:, 1] >= targets[:, 1])
        & (xy[:, 1] <= targets[:, 3])
    )


def attention_hits(
    attn: jax.Array, centres: jax.Array, targets: jax.Array
) -> tuple[float, float]:
    """``(peak hit, centroid hit)`` -- attention localisation, two ways.

    The peak is the sharper question ("does one cell win, and is it the right
    one"), but the readout consumes the *centroid*, so a diffuse map that
    averages onto the field scores 0 on peak and 1 on centroid.  Reporting both
    separates "attention points at the field" from "attention is shaped so its
    mean lands there", which are very different things to build on.
    """
    grid = np.asarray(centres)
    tgt = np.asarray(targets)
    peak_xy = grid[np.asarray(jnp.argmax(attn, axis=-1))]
    centroid_xy = np.asarray(attn @ centres)
    return float(_inside(peak_xy, tgt).mean()), float(_inside(centroid_xy, tgt).mean())


def evaluate(
    head: model_1_extra_layer.BBoxAttnHead,
    pooled: np.ndarray,
    image: np.ndarray,
    targets: jax.Array,
    batch_size: int = 64,
) -> dict:
    """Metrics over a whole split, batched to bound device memory."""
    boxes, attns = [], []
    for start in range(0, pooled.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        box, attn = predict(head, jnp.asarray(pooled[sl]), jnp.asarray(image[sl]))
        boxes.append(box)
        attns.append(attn)
    raw = jnp.concatenate(boxes, axis=0)
    attn = jnp.concatenate(attns, axis=0)

    metrics = train_1.box_metrics(model_1.clip_boxes(raw), targets)
    metrics["loss"] = float(model_1.bbox_loss(raw, targets))
    metrics["attn_hit"], metrics["centroid_hit"] = attention_hits(
        attn, head.grid_centres(), targets
    )
    return metrics


def save_visualisations(
    examples: list[train_1.Example],
    pred: np.ndarray,
    attn: np.ndarray,
    grid_hw: tuple[int, int],
    out_dir: str,
    limit: int,
) -> int:
    """Attention heat map under the boxes: ground truth green, prediction red."""
    os.makedirs(out_dir, exist_ok=True)
    grid_h, grid_w = grid_hw
    n = min(limit, len(examples))
    for i in range(n):
        example = examples[i]
        base = example.image.convert("RGB").copy()
        width, height = base.size

        heat = attn[i].reshape(grid_h, grid_w)
        heat = heat / max(float(heat.max()), 1e-8)
        # 0.65 alpha at the peak: enough to read the attention, still legible
        # over the page text underneath.
        alpha = Image.fromarray((heat * 0.65 * 255).astype(np.uint8), mode="L")
        alpha = alpha.resize(base.size, Image.Resampling.BILINEAR)
        image = Image.composite(Image.new("RGB", base.size, (255, 0, 0)), base, alpha)

        draw = ImageDraw.Draw(image)
        for box, colour, label in (
            (example.box, (0, 160, 0), f"gt {example.query}"),
            (pred[i], (0, 0, 220), "pred"),
        ):
            xy = [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
            draw.rectangle(xy, outline=colour, width=3)
            draw.text((xy[0], max(0.0, xy[1] - 12)), label, fill=colour)
        image.save(os.path.join(out_dir, f"eval_{i:02d}.png"))
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=train_1.MODEL_ID)
    p.add_argument("--base-config", default=None)
    p.add_argument("--rows", type=int, default=96, help="documents to stream")
    p.add_argument("--max-per-doc", type=int, default=8)
    p.add_argument("--max-examples", type=int, default=768)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--page-width", type=int, default=train_1.PAGE_WIDTH)
    p.add_argument("--page-height", type=int, default=train_1.PAGE_HEIGHT)
    p.add_argument("--max-seq-len", type=int, default=None, help="default: probed")
    p.add_argument("--encode-batch", type=int, default=4, help="backbone batch")
    p.add_argument("--batch-size", type=int, default=32, help="head batch")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=512, help="attention width")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=10, help="epochs between logs")
    p.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="epochs between metrics.csv rows (evaluation is cheap; 1 is fine)",
    )
    p.add_argument("--out-dir", default="output/bbox_heads_extra_layer")
    p.add_argument(
        "--tb-dir",
        default="output/tb",
        help="shared across experiments so TensorBoard overlays all runs",
    )
    p.add_argument("--run-name", default=None, help="TensorBoard run name")
    p.add_argument("--vis", type=int, default=8, help="eval PNGs to write (0 = none)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    examples = train_1.load_examples(
        rows=args.rows,
        max_per_doc=args.max_per_doc,
        max_examples=args.max_examples,
        page_size=(args.page_width, args.page_height),
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("no usable examples; raise --rows")
    train_ex, eval_ex = train_1.split_by_document(examples, args.eval_frac, args.seed)
    if not train_ex or not eval_ex:
        raise RuntimeError("empty split; raise --rows or adjust --eval-frac")

    config = getattr(ModelConfig, args.base_config)() if args.base_config else None
    processor, backbone = load_model(args.model, dtype=jnp.bfloat16, config=config)
    config = backbone.config
    if config.vision_config is None:
        raise ValueError(f"{args.model} resolved to a text-only config")
    # load_model returns a concrete Qwen2VLProcessor; encode_batch annotates the
    # parameter as the AutoProcessor factory.  Same object at runtime.
    processor = cast(AutoProcessor, processor)

    grid_hw = probe_image_grid(
        processor, train_ex[0].image, config.vision_config.spatial_merge_size
    )
    n_image_tokens = grid_hw[0] * grid_hw[1]
    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = train_1.probe_seq_len(
            processor, train_ex[0], config.vision_config
        )
        max_seq_len += 32
    logger.info(
        "tokens   %d image on a %dx%d grid | seq len %d (padded)",
        n_image_tokens,
        grid_hw[0],
        grid_hw[1],
        max_seq_len,
    )
    cache_gb = len(examples) * n_image_tokens * config.embed_dim * 2 / 1e9
    logger.info("cache    %.1f GB of image-token features (bfloat16, host)", cache_gb)

    model = model_1_extra_layer.Qwen3VLBBoxAttn(
        backbone,
        n_image_tokens=n_image_tokens,
        grid_hw=grid_hw,
        rngs=nnx.Rngs(args.seed),
        width=args.width,
        num_heads=args.num_heads,
        head_hidden_dim=args.head_hidden,
    )

    encode_kwargs = dict(batch_size=args.encode_batch, max_seq_len=max_seq_len)
    train_pooled, train_img = encode_features(
        model, processor, train_ex, tag="train", **encode_kwargs
    )
    eval_pooled, eval_img = encode_features(
        model, processor, eval_ex, tag="eval", **encode_kwargs
    )
    train_y = jnp.asarray(np.stack([e.box for e in train_ex]))
    eval_y = jnp.asarray(np.stack([e.box for e in eval_ex]))

    head = model.head
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(head, nnx.Param)))
    logger.info("head     %d trainable params, backbone frozen", n_params)

    steps_per_epoch = max(1, len(train_ex) // args.batch_size)
    # Cosine decay: L1 has a constant gradient magnitude, so at a fixed lr the
    # head settles into a limit cycle around the target rather than landing.
    schedule = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch)
    optimizer = nnx.Optimizer(
        head,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        ),
        wrt=nnx.Param,
    )

    base = train_1.mean_box_baseline(train_y, eval_y)
    logger.info(
        "baseline mean-box eval iou %.4f  centre hit %.3f  centre dist %.4f",
        base["iou"],
        base["centre_hit"],
        base["centre_dist"],
    )

    run_name = args.run_name or f"attn-{time.strftime('%m%d-%H%M')}"
    metrics = train_1.MetricWriter(
        os.path.join(args.out_dir, "metrics.csv"), args.tb_dir, run_name
    )
    logger.info("tb       %s", metrics.tb_path)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_ex))
        losses = []
        for step in range(steps_per_epoch):
            idx = order[step * args.batch_size : (step + 1) * args.batch_size]
            losses.append(
                train_step(
                    head,
                    optimizer,
                    jnp.asarray(train_pooled[idx]),
                    jnp.asarray(train_img[idx]),
                    train_y[jnp.asarray(idx)],
                )
            )
        if epoch % args.metrics_every and epoch not in (1, args.epochs):
            continue
        train_loss = float(jnp.mean(jnp.stack(losses)))
        n_tr = min(len(train_ex), train_1.TRAIN_METRIC_EXAMPLES)
        tr = evaluate(head, train_pooled[:n_tr], train_img[:n_tr], train_y[:n_tr])
        ev = evaluate(head, eval_pooled, eval_img, eval_y)
        metrics.write(train_1.metric_row(epoch, train_loss, tr, ev))
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            logger.info(
                "epoch %3d  loss %.4f | train iou %.4f cent %.3f | "
                "eval iou %.4f hit %.3f peak %.3f cent %.3f dist %.4f",
                epoch,
                train_loss,
                tr["iou"],
                tr["centroid_hit"],
                ev["iou"],
                ev["centre_hit"],
                ev["attn_hit"],
                ev["centroid_hit"],
                ev["centre_dist"],
            )

    metrics.close()
    logger.info("wrote    %s", metrics.path)

    final = evaluate(head, eval_pooled, eval_img, eval_y)
    logger.info(
        "final    eval iou %.4f (baseline %.4f) acc@0.5 %.3f acc@0.75 %.3f",
        final["iou"],
        base["iou"],
        final["acc50"],
        final["acc75"],
    )
    logger.info(
        "final    centre hit %.3f (baseline %.3f)  centre dist %.4f (baseline %.4f)",
        final["centre_hit"],
        base["centre_hit"],
        final["centre_dist"],
        base["centre_dist"],
    )
    logger.info(
        "final    attention inside the true box: peak %.3f, centroid %.3f",
        final["attn_hit"],
        final["centroid_hit"],
    )

    if args.vis:
        boxes, attn = predict(
            head,
            jnp.asarray(eval_pooled[: args.vis]),
            jnp.asarray(eval_img[: args.vis]),
        )
        n = save_visualisations(
            eval_ex,
            np.asarray(model_1.clip_boxes(boxes)),
            np.asarray(attn, dtype=np.float32),
            grid_hw,
            args.out_dir,
            args.vis,
        )
        logger.info("wrote    %d PNGs to %s", n, args.out_dir)


if __name__ == "__main__":
    main()
