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

Checkpointing saves only the trainable slice (LoRA + readout) and the optimizer
state -- the frozen backbone comes back from the HuggingFace cache.  ``last`` is
rewritten at every eval, ``best`` tracks the lowest eval loss, and ``--resume``
picks ``last`` unless pointed at a specific directory.  Note that the cosine
schedule is defined over ``--epochs``, so resuming with a different value gives
a different schedule than an uninterrupted run would have had.

Run::

    python -m experiments.bbox_heads.train_1_extra_pos
    python -m experiments.bbox_heads.train_1_extra_pos --examples 8000 --epochs 4
    python -m experiments.bbox_heads.train_1_extra_pos --resume output/.../ckpt

Defaults are LoRA rank 16 at batch 2, which is what fits in 24 GB.  Raising
--batch-size with LoRA on will OOM; raise --lora-rank only if you have more
memory than that.
"""

from __future__ import annotations

import argparse
import json
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


@nnx.jit(static_argnames=("trainable",))
def train_step(
    model: mep.SlotBoxModel,
    optimizer: nnx.Optimizer,
    args: tuple,
    targets: jax.Array,  # [B, 4]
    trainable=mep.TRAINABLE,
) -> tuple[jax.Array, jax.Array]:
    def loss_fn(m):
        return model_1.bbox_loss(m(*args), targets)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable))(
        model
    )
    # Global norm *before* clipping, so TensorBoard shows whether
    # clip_by_global_norm(1.0) is actually biting.  Early gradients on this
    # readout were measured at |max| 125, so it does.
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    optimizer.update(model, grads)
    return loss, grad_norm


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


_CKPT_TRAINABLE = "trainable"
_CKPT_OPT = "opt"
_CKPT_META = "meta.json"


def save_checkpoint(
    path: str,
    model: mep.SlotBoxModel,
    optimizer: nnx.Optimizer,
    trainable,
    meta: dict,
) -> None:
    """Write the trainable state, the optimizer state and ``meta`` to ``path``.

    Only the trainable slice is saved -- LoRA adapters plus the readout, ~109 MB
    in float32 -- not the 8 GB frozen backbone, which is reloaded from the
    HuggingFace cache instead.  The optimizer state is adamw's mu/nu over the
    same slice (~217 MB) and its own step counter, which is what makes the
    cosine schedule resume at the right point rather than restarting.
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        os.path.join(path, _CKPT_TRAINABLE), nnx.state(model, trainable), force=True
    )
    checkpointer.save(os.path.join(path, _CKPT_OPT), nnx.state(optimizer), force=True)
    checkpointer.wait_until_finished()
    with open(os.path.join(path, _CKPT_META), "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(
    path: str, model: mep.SlotBoxModel, optimizer: nnx.Optimizer, trainable
) -> dict:
    """Restore in place and return the saved metadata.

    ``path`` may be a checkpoint directory or a parent holding ``last/``; a
    parent resolves to ``last``, since resuming almost always means "carry on
    from where it stopped" rather than "go back to the best eval".
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "last")):
        path = os.path.join(path, "last")
    checkpointer = ocp.StandardCheckpointer()
    nnx.update(
        model,
        checkpointer.restore(
            os.path.join(path, _CKPT_TRAINABLE), target=nnx.state(model, trainable)
        ),
    )
    nnx.update(
        optimizer,
        checkpointer.restore(
            os.path.join(path, _CKPT_OPT), target=nnx.state(optimizer)
        ),
    )
    with open(os.path.join(path, _CKPT_META)) as f:
        return json.load(f)


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
    # 2, not 4: batch 4 plus LoRA OOMs a 24 GB card (remat gets peak no lower
    # than 20.4 GiB, then a 7.2 GiB allocation fails).  Batch 2 is also *faster*
    # per example -- 0.90 s vs 1.23 s -- because batch 4 was thrashing the
    # memory ceiling.  Frozen runs can afford 4.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2, help="passes over the train split")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--lora-rank", type=int, default=16, help="0 disables LoRA (frozen control)"
    )
    p.add_argument("--lora-alpha", type=float, default=None, help="default: 2 * rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25, help="steps between log rows")
    p.add_argument("--eval-every", type=int, default=100, help="steps between evals")
    p.add_argument("--out-dir", default="output/bbox_heads_extra_pos")
    p.add_argument("--tb-dir", default="output/tb")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--ckpt-dir", default=None, help="default: <out-dir>/ckpt; '' disables"
    )
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint directory to continue from (resolves <dir>/last)",
    )
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

    # LoRA wraps the backbone before the readout is attached, because qwix
    # traces the module it is handed and `get_model_input()` describes the
    # backbone's signature rather than SlotBoxModel's.
    if args.lora_rank:
        backbone = mep.add_lora(
            backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha or 2.0 * args.lora_rank,
            rngs=nnx.Rngs(args.seed),
        )
    trainable = mep.TRAINABLE_WITH_LORA if args.lora_rank else mep.TRAINABLE

    model = mep.SlotBoxModel(backbone, rngs=nnx.Rngs(args.seed))
    n_readout = sum(p.size for p in jax.tree.leaves(nnx.state(model, mep.TRAINABLE)))
    n_lora = (
        sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.LoRAParam)))
        if args.lora_rank
        else 0
    )
    logger.info(
        "trainable %d readout + %d LoRA (rank %d) = %d params",
        n_readout,
        n_lora,
        args.lora_rank,
        n_readout + n_lora,
    )

    steps_per_epoch = len(train_ex) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    schedule = optax.cosine_decay_schedule(args.lr, total_steps)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        ),
        wrt=trainable,
    )

    ckpt_dir = (
        args.ckpt_dir
        if args.ckpt_dir is not None
        else os.path.join(args.out_dir, "ckpt")
    )
    start_step, best_eval_loss = 0, float("inf")
    if args.resume:
        meta = load_checkpoint(args.resume, model, optimizer, trainable)
        start_step = int(meta.get("step", 0))
        best_eval_loss = float(meta.get("best_eval_loss", float("inf")))
        if meta.get("lora_rank") != args.lora_rank:
            raise ValueError(
                f"checkpoint has lora_rank={meta.get('lora_rank')}, run has "
                f"{args.lora_rank}; the trainable shapes would not match"
            )
        logger.info(
            "resume   step %d, best eval loss %.4f, from %s",
            start_step,
            best_eval_loss,
            args.resume,
        )

    tag = f"lora{args.lora_rank}" if args.lora_rank else "frozen"
    run_name = args.run_name or f"extra_pos-{tag}-{time.strftime('%m%d-%H%M')}"
    # Declared up front: eval rows appear only every --eval-every steps, and a
    # CSV header cannot grow after the first row.
    eval_fields = ["iou", "acc50", "acc75", "centre_dist", "centre_hit", "loss"]
    metrics = train_1.MetricWriter(
        os.path.join(args.out_dir, "metrics.csv"),
        args.tb_dir,
        run_name,
        fieldnames=["epoch", "pass", "batch_loss", "lr", "grad_norm"]
        + [f"eval_{k}" for k in eval_fields],
        append=bool(args.resume),
    )
    logger.info("tb       %s", metrics.tb_path)
    # 0.9 s/example measured with LoRA at batch 2, 1.23 s/example frozen at
    # batch 4.  Only ever an estimate -- the step logs carry the real elapsed.
    per_example = 0.9 if args.lora_rank else 1.23
    logger.info(
        "plan     %d train / %d eval ex, %d steps of %d (~%.0f min at %.2f s/example)",
        len(train_ex),
        len(eval_ex),
        total_steps,
        args.batch_size,
        total_steps * args.batch_size * per_example / 60,
        per_example,
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

    step = start_step
    started = time.perf_counter()
    recent: list[jax.Array] = []
    recent_norms: list[jax.Array] = []
    for epoch in range(start_step // steps_per_epoch, args.epochs):
        # Seeded per pass rather than from one running generator, so resuming at
        # pass k replays exactly the order pass k had the first time.
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train_ex))
        for i in range(step - epoch * steps_per_epoch, steps_per_epoch):
            batch = [
                train_ex[j]
                for j in order[i * args.batch_size : (i + 1) * args.batch_size]
            ]
            targets = jnp.asarray(np.stack([e.box for e in batch]))
            loss, grad_norm = train_step(
                model,
                optimizer,
                encode(processor, batch, vcfg, max_seq_len),
                targets,
                trainable=trainable,
            )
            recent.append(loss)
            recent_norms.append(grad_norm)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                batch_loss = float(jnp.mean(jnp.stack(recent)))
                grad_norm = float(jnp.mean(jnp.stack(recent_norms)))
                recent, recent_norms = [], []
                # MetricWriter uses row["epoch"] as the x-axis for both CSV
                # and TensorBoard.  This run is step-based, so the step goes
                # there and the pass number rides along as a plain column.
                row = {
                    "epoch": step,
                    "pass": epoch + 1,
                    "batch_loss": batch_loss,
                    # Reading the schedule at the loop's step is how a resumed
                    # run shows, in TensorBoard, that it continued the cosine
                    # decay instead of restarting it.
                    # asarray narrows optax's broad schedule return type, which
                    # nominally includes complex, to something float() accepts.
                    "lr": float(jnp.asarray(schedule(step), dtype=jnp.float32)),
                    "grad_norm": grad_norm,
                }
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

                if ckpt_dir and "eval_loss" in row:
                    meta = {
                        "step": step,
                        "pass": epoch + 1,
                        "lora_rank": args.lora_rank,
                        "eval_loss": row["eval_loss"],
                        "best_eval_loss": min(best_eval_loss, row["eval_loss"]),
                    }
                    save_checkpoint(
                        os.path.join(ckpt_dir, "last"),
                        model,
                        optimizer,
                        trainable,
                        meta,
                    )
                    if row["eval_loss"] < best_eval_loss:
                        best_eval_loss = row["eval_loss"]
                        save_checkpoint(
                            os.path.join(ckpt_dir, "best"),
                            model,
                            optimizer,
                            trainable,
                            {**meta, "best_eval_loss": best_eval_loss},
                        )
                    logger.info(
                        "ckpt     step %d -> %s/last%s",
                        step,
                        ckpt_dir,
                        " (+best)" if row["eval_loss"] == best_eval_loss else "",
                    )

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


if __name__ == "__main__" and "__file__" in globals():
    main()
