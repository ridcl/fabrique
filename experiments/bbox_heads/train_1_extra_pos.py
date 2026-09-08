"""Train the ``<|box_start|>`` readout from ``model_1_extra_pos`` on kvp10k.

Same data, document-level split, metrics and mean-box baseline as ``train_1``,
so the numbers are comparable to the three earlier models.  Two things differ.

**The backbone is in the loop.**  Gradients reach ``slot_delta`` through all 36
layers, so nothing can be precomputed: measured at 1.23 s/example against
0.36 s/example for a frozen forward done once.  That is 3.4x the cost, paid
every pass instead of once, so this script is step-based -- at ~4.9 s per step
of 4, an "epoch" is 30+ minutes and useless as a reporting unit.  Metrics go to
TensorBoard every ``--log-every`` steps so a run is watchable from the start.

**Evaluation is on a fixed subset.**  A full 400-example eval is 2.5 minutes of
forward passes; doing that every 100 steps would be a third of the run.

What a frozen run means here
----------------------------
``slot_delta`` starts at zero, so step 0 is exactly the frozen model read at its
own ``<|box_start|>`` position.  The question being asked is whether Qwen3-VL
already encodes the box where it would emit one -- not, as in the three earlier
probes, whether a foreign readout can learn to find it.

Run::

    python -m experiments.bbox_heads.train_1_extra_pos
    python -m experiments.bbox_heads.train_1_extra_pos --examples 4000 --epochs 3
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
from transformers import AutoProcessor

from experiments.bbox_heads import model_1, train_1
from experiments.bbox_heads import model_1_extra_pos as mep
from fabrique.models.qwen3vl.loading import load_model
from fabrique.models.qwen3vl.model import ModelConfig, RematConfig
from fabrique.models.qwen3vl.utils import encode_batch

# force=True: importing jax/tunix installs a root logging handler, which makes
# a plain basicConfig() a silent no-op and swallows every logger.info below.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def build_prompt(processor, example: train_1.Example) -> str:
    """Chat prompt, then the model's own grounding prefix.

    ``apply_chat_template(add_generation_prompt=True)`` ends at
    ``<|im_start|>assistant\\n``; appending the grounding markup puts
    ``<|box_start|>`` last, which is the position the readout reads.
    """
    return train_1.build_chat_prompt(processor, example) + mep.grounding_suffix(
        train_1._humanise(example.query)
    )


def encode(processor, examples: list[train_1.Example], vcfg, max_seq_len: int) -> tuple:
    """Host-side preprocessing for one batch -> device arrays.

    Runs per step rather than once: ``pixel_values`` alone is ~18 MB per page,
    so caching the encoded form of even 2000 examples would need 36 GB.  At
    ~0.02 s/example it is negligible against the 1.23 s/example step.
    """
    batch = encode_batch(
        processor,
        [build_prompt(processor, e) for e in examples],
        [[e.image] for e in examples],
        vcfg=vcfg,
        max_length=max_seq_len,
        padding="max_length",
    )
    mep.check_slot_tokens(batch.input_tokens)
    tokens, mask, positions = mep.ensure_even_length(
        batch.input_tokens, batch.input_mask, batch.positions
    )
    return (
        jnp.asarray(tokens),
        jnp.asarray(positions),
        jnp.asarray(batch.pixel_values, dtype=jnp.bfloat16),
        batch.vision_grid,
        jnp.asarray(mask).astype(jnp.bool_),
    )


@nnx.jit
def train_step(
    model: mep.SlotBoxModel,
    optimizer: nnx.Optimizer,
    args: tuple,
    targets: jax.Array,  # [B, 4]
) -> jax.Array:
    def loss_fn(m):
        return model_1.bbox_loss(m(*args), targets)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, mep.TRAINABLE))(
        model
    )
    optimizer.update(model, grads)
    return loss


@nnx.jit
def predict(model: mep.SlotBoxModel, args: tuple) -> jax.Array:
    return model(*args)


def evaluate(
    model: mep.SlotBoxModel,
    processor,
    examples: list[train_1.Example],
    vcfg,
    max_seq_len: int,
    batch_size: int,
) -> dict:
    """Forward-only metrics over ``examples``, dropping any partial tail batch."""
    boxes, kept = [], []
    for start in range(0, len(examples) - batch_size + 1, batch_size):
        chunk = examples[start : start + batch_size]
        boxes.append(predict(model, encode(processor, chunk, vcfg, max_seq_len)))
        kept += chunk
    raw = jnp.concatenate(boxes, axis=0)
    targets = jnp.asarray(np.stack([e.box for e in kept]))
    metrics = train_1.box_metrics(model_1.clip_boxes(raw), targets)
    metrics["loss"] = float(model_1.bbox_loss(raw, targets))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=train_1.MODEL_ID)
    p.add_argument("--rows", type=int, default=1400, help="documents to stream")
    p.add_argument("--max-per-doc", type=int, default=6)
    p.add_argument("--examples", type=int, default=2000)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--eval-examples", type=int, default=256, help="fixed eval subset")
    p.add_argument("--page-width", type=int, default=train_1.PAGE_WIDTH)
    p.add_argument("--page-height", type=int, default=train_1.PAGE_HEIGHT)
    p.add_argument("--max-seq-len", type=int, default=None, help="default: probed")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2, help="passes over the train split")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25, help="steps between log rows")
    p.add_argument("--eval-every", type=int, default=100, help="steps between evals")
    p.add_argument("--out-dir", default="output/bbox_heads_extra_pos")
    p.add_argument("--tb-dir", default="output/tb")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    examples = train_1.load_examples(
        rows=args.rows,
        max_per_doc=args.max_per_doc,
        max_examples=args.examples,
        page_size=(args.page_width, args.page_height),
        seed=args.seed,
    )
    train_ex, eval_ex = train_1.split_by_document(examples, args.eval_frac, args.seed)
    eval_subset = eval_ex[: args.eval_examples]

    # Remat is not optional here: without it the 36-layer backward pass does not
    # fit alongside an 8 GB backbone on a 24 GB card.
    config = ModelConfig.qwen3vl_4b()
    config.remat_config = RematConfig.BLOCK
    processor, backbone = load_model(args.model, dtype=jnp.bfloat16, config=config)
    config = backbone.config
    if config.vision_config is None:
        raise ValueError(f"{args.model} resolved to a text-only config")
    processor = cast(AutoProcessor, processor)
    vcfg = config.vision_config

    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        # The grounding suffix adds a handful of tokens on top of the plain
        # prompt; probe the real thing rather than guessing the margin.
        probe = encode_batch(
            processor,
            [build_prompt(processor, train_ex[0])],
            [[train_ex[0].image]],
            vcfg=vcfg,
            max_length=32768,
            padding=False,
        )
        max_seq_len = int(probe.input_tokens.shape[1]) + 32
    logger.info("seq len  %d (padded)", max_seq_len)

    model = mep.SlotBoxModel(backbone, rngs=nnx.Rngs(args.seed))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, mep.TRAINABLE)))
    logger.info("trainable %d params, backbone frozen", n_params)

    steps_per_epoch = len(train_ex) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        ),
        wrt=mep.TRAINABLE,
    )

    run_name = args.run_name or f"extra_pos-{time.strftime('%m%d-%H%M')}"
    # Declared up front: eval rows appear only every --eval-every steps, and a
    # CSV header cannot grow after the first row.
    eval_fields = ["iou", "acc50", "acc75", "centre_dist", "centre_hit", "loss"]
    metrics = train_1.MetricWriter(
        os.path.join(args.out_dir, "metrics.csv"),
        args.tb_dir,
        run_name,
        fieldnames=["epoch", "pass", "batch_loss"] + [f"eval_{k}" for k in eval_fields],
    )
    logger.info("tb       %s", metrics.tb_path)
    logger.info(
        "plan     %d train / %d eval ex, %d steps of %d (~%.0f min at 4.9 s/step)",
        len(train_ex),
        len(eval_ex),
        total_steps,
        args.batch_size,
        total_steps * 4.9 / 60,
    )

    base = train_1.mean_box_baseline(
        jnp.asarray(np.stack([e.box for e in train_ex])),
        jnp.asarray(np.stack([e.box for e in eval_subset])),
    )
    logger.info(
        "baseline mean-box eval iou %.4f  centre hit %.3f  centre dist %.4f",
        base["iou"],
        base["centre_hit"],
        base["centre_dist"],
    )

    rng = np.random.default_rng(args.seed)
    step = 0
    started = time.perf_counter()
    recent: list[jax.Array] = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_ex))
        for i in range(steps_per_epoch):
            batch = [
                train_ex[j]
                for j in order[i * args.batch_size : (i + 1) * args.batch_size]
            ]
            targets = jnp.asarray(np.stack([e.box for e in batch]))
            recent.append(
                train_step(
                    model,
                    optimizer,
                    encode(processor, batch, vcfg, max_seq_len),
                    targets,
                )
            )
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                batch_loss = float(jnp.mean(jnp.stack(recent)))
                recent = []
                # MetricWriter uses row["epoch"] as the x-axis for both CSV
                # and TensorBoard.  This run is step-based, so the step goes
                # there and the pass number rides along as a plain column.
                row = {"epoch": step, "pass": epoch + 1, "batch_loss": batch_loss}
                if step % args.eval_every == 0 or step == total_steps:
                    ev = evaluate(
                        model,
                        processor,
                        eval_subset,
                        vcfg,
                        max_seq_len,
                        args.batch_size,
                    )
                    row.update({f"eval_{k}": v for k, v in ev.items()})
                    logger.info(
                        "step %4d/%d  loss %.4f | eval iou %.4f hit %.3f dist %.4f "
                        "acc@0.5 %.3f  (%.0f min)",
                        step,
                        total_steps,
                        batch_loss,
                        ev["iou"],
                        ev["centre_hit"],
                        ev["centre_dist"],
                        ev["acc50"],
                        (time.perf_counter() - started) / 60,
                    )
                else:
                    logger.info(
                        "step %4d/%d  loss %.4f  (%.0f min)",
                        step,
                        total_steps,
                        batch_loss,
                        (time.perf_counter() - started) / 60,
                    )
                metrics.write(row)

    metrics.close()
    final = evaluate(model, processor, eval_ex, vcfg, max_seq_len, args.batch_size)
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
    logger.info("wrote    %s", metrics.path)


if __name__ == "__main__":
    main()
