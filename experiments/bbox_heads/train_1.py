"""Train a bounding-box head on (page, query) -> box examples from vqa_kvp10k_synth.

Each dataset row holds several pages, several queries and one answer per
(query, page).  This script explodes rows into independent single-box examples
-- one page image, one short prompt (``find on page: total assets``), one target
box -- and trains ``model_1.BBoxHead`` on them.  Qwen3-VL stays frozen; only the
head's two Linear layers (plus its input LayerNorm) are trained, fully.

Because the backbone is frozen, its output for a given example never changes, so
features are computed once up front and the epoch loop then trains the head on
them.  That is a linear-probe setup: the moment you unfreeze the backbone or add
image augmentation, the ``encode_features`` call has to move inside the loop.

Two things to check before believing a number
---------------------------------------------
``mean-box baseline`` is the score of the single constant box that best fits the
training set.  A head that does not clearly beat it has learned the page prior
rather than the query.

Watch ``centre hit``, not just IoU.  The median target here is one text line --
0.13% of the page -- so being one line off scores IoU 0 exactly like being on
the wrong half of the page, and a run that is genuinely learning can show mean
IoU near zero for a long time while the centre metrics move.

The train/eval split is by *document*, not by example.  Splitting by example
would put other fields of the same page on both sides, which is close to
training on the eval set.

Run::

    python -m experiments.bbox_heads.train_1
    python -m experiments.bbox_heads.train_1 --rows 128 --epochs 100
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, cast

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from PIL import Image, ImageDraw
from tensorboardX import SummaryWriter
from transformers import AutoProcessor

from experiments.bbox_heads import model_1
from fabrique.models.qwen3vl.loading import load_model
from fabrique.models.qwen3vl.model import ModelConfig
from fabrique.models.qwen3vl.utils import encode_batch

# force=True: importing jax/tunix installs a root logging handler, which makes
# a plain basicConfig() a silent no-op and swallows every logger.info below.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "ridcl/vqa_kvp10k_synth"
DATASET_SPLIT = "vqa_kvp10k_synth"  # the dataset has no "train" split

# Every page is resized to exactly these dimensions.  Pages are A4 at 150 dpi
# (1242x1756) give or take a few percent, so this is a downscale plus <3% of
# aspect distortion -- and boxes are normalised, so the target distorts with the
# image.  A single fixed size is what keeps the patch count, and therefore every
# JIT-compiled shape, constant across the run.
PAGE_WIDTH, PAGE_HEIGHT = 736, 1024

PROMPT_TEMPLATE = "find on page: {query}"

# Documents are cached here after decoding and resizing.  `streaming=True` never
# writes to the HuggingFace cache -- it reads parquet over HTTP range requests --
# so without this every run re-downloads ~2.7 MB per document (3.8 GB for 1400)
# and re-decodes every page.  The cached form is much smaller than the source:
# pages are stored at the resized resolution, and the originals are up to 4 MB
# of PNG each.  Keyed on row count and page size only, so changing
# --max-per-doc, --seed or --max-examples still hits the cache.
DOC_CACHE_ROOT = os.path.expanduser("~/.cache/fabrique/vqa_kvp10k_synth")

# Train-split metrics are computed on this many examples rather than all of
# them.  Each example is 3.8 MB of cached features, so a full-split pass moves
# tens of GB over PCIe every epoch for a number that is already stable at n=2048.
TRAIN_METRIC_EXAMPLES = 2048


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """One page, one query, one box."""

    image: Image.Image  # resized page; shared by all examples on that page
    query: str  # raw dataset key, e.g. "student.last_name"
    prompt: str  # what the model actually sees
    value: str
    box: np.ndarray  # (4,) xyxy in [0, 1]
    doc: str
    page: int


def _humanise(query: str) -> str:
    """``student.last_name`` -> ``student last name``.

    The queries are snake_case field keys.  Spelling them out costs nothing and
    keeps the prompt inside the distribution the backbone was pretrained on.
    """
    return query.replace("_", " ").replace(".", " ").strip()


def _valid_box(raw) -> bool:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return False
    x0, y0, x1, y1 = (float(v) for v in raw)
    return all(np.isfinite(v) for v in raw) and x1 > x0 and y1 > y0


@dataclass
class Document:
    """One cached document: its eligible answers and the pages they point at."""

    doc: str
    answers: list[dict[str, Any]]
    pages: dict[int, Image.Image]


def _cache_dir(rows: int, page_size: tuple[int, int]) -> str:
    return os.path.join(DOC_CACHE_ROOT, f"{rows}rows-{page_size[0]}x{page_size[1]}")


def _read_cache(path: str) -> list[Document] | None:
    manifest = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest) as f:
        entries = json.load(f)
    return [
        Document(
            doc=e["doc"],
            answers=e["answers"],
            pages={
                int(page): Image.open(os.path.join(path, name)).convert("RGB")
                for page, name in e["pages"].items()
            },
        )
        for e in entries
    ]


def _write_cache(path: str, documents: list[Document]) -> None:
    # Built under a temporary name and renamed, so an interrupted run leaves no
    # half-written cache that a later run would trust.
    staging = f"{path}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    entries = []
    for i, document in enumerate(documents):
        names = {}
        for page, image in document.pages.items():
            name = f"doc{i:05d}_p{page}.png"
            image.save(os.path.join(staging, name), optimize=True)
            names[str(page)] = name
        entries.append(
            {"doc": document.doc, "answers": document.answers, "pages": names}
        )
    with open(os.path.join(staging, "manifest.json"), "w") as f:
        json.dump(entries, f)
    shutil.rmtree(path, ignore_errors=True)
    os.rename(staging, path)


def fetch_documents(rows: int, page_size: tuple[int, int]) -> list[Document]:
    """Stream ``rows`` documents, decode and resize their pages, cache, return.

    Everything that depends on sampling (``max_per_doc``, ``seed``) is left to
    the caller so that the cache survives changing them.
    """
    path = _cache_dir(rows, page_size)
    cached = _read_cache(path)
    if cached is not None:
        logger.info("data     %d documents from cache %s", len(cached), path)
        return cached

    ds = datasets.load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    documents: list[Document] = []
    started = time.perf_counter()
    for n_rows, row in enumerate(ds.take(rows), start=1):
        # Network-bound at ~6 MB/s, so this can run for 15 minutes on a large
        # --rows.  Without progress here the run looks hung: no output, idle
        # GPU, one core busy on PNG decode.
        if n_rows % 100 == 0:
            logger.info(
                "loading  %d/%d docs, %d cached, %.0fs elapsed",
                n_rows,
                rows,
                len(documents),
                time.perf_counter() - started,
            )
        counts = collections.Counter(a["query"] for a in row["answers"])
        answers: list[dict[str, Any]] = [
            {
                "query": str(a["query"]),
                "value": str(a["value"]),
                "bounding_box": [float(v) for v in a["bounding_box"]],
                "index": int(a["index"]),
            }
            for a in row["answers"]
            if counts[a["query"]] == 1
            and 0 <= int(a["index"]) < len(row["images"])
            and _valid_box(a["bounding_box"])
        ]
        if not answers:
            continue
        pages: dict[int, Image.Image] = {}
        for page in sorted({int(a["index"]) for a in answers}):
            image = Image.open(io.BytesIO(row["images"][page])).convert("RGB")
            pages[page] = image.resize(page_size, Image.Resampling.LANCZOS)
        documents.append(
            Document(
                doc=str(row.get("source", "")) or f"row{n_rows}",
                answers=answers,
                pages=pages,
            )
        )

    logger.info(
        "data     %d documents streamed in %.0fs, caching to %s",
        len(documents),
        time.perf_counter() - started,
        path,
    )
    _write_cache(path, documents)
    return documents


def load_examples(
    *,
    rows: int,
    max_per_doc: int,
    max_examples: int,
    page_size: tuple[int, int],
    seed: int,
) -> list[Example]:
    """Explode cached documents into independent single-box examples.

    Only queries with exactly one answer are eligible (enforced in
    ``fetch_documents``): a query like "names of the parties" has three boxes,
    and a single-box head has no defensible target for it.

    ``max_per_doc`` caps how many examples one document contributes.  Answer
    counts per document run from 8 to over 100, so without a cap a handful of
    documents would supply most of the training set.
    """
    documents = fetch_documents(rows, page_size)
    rng = random.Random(seed)

    examples: list[Example] = []
    for document in documents:
        answers = list(document.answers)
        rng.shuffle(answers)
        for answer in answers[:max_per_doc]:
            page = int(answer["index"])
            examples.append(
                Example(
                    # Pages are shared by reference, so a document with 6 fields
                    # on one page costs one bitmap, not six.
                    image=document.pages[page],
                    query=answer["query"],
                    prompt=PROMPT_TEMPLATE.format(query=_humanise(answer["query"])),
                    value=answer["value"],
                    box=np.asarray(answer["bounding_box"], dtype=np.float32),
                    doc=document.doc,
                    page=page,
                )
            )

    rng.shuffle(examples)
    if len(examples) > max_examples:
        examples = examples[:max_examples]
    logger.info("data     %d examples from %d documents", len(examples), len(documents))
    return examples


def split_by_document(
    examples: list[Example], eval_frac: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Hold out whole documents, so no page appears in both splits."""
    docs = sorted({e.doc for e in examples})
    random.Random(seed).shuffle(docs)
    n_eval = max(1, round(len(docs) * eval_frac))
    eval_docs = set(docs[:n_eval])
    train = [e for e in examples if e.doc not in eval_docs]
    evl = [e for e in examples if e.doc in eval_docs]
    logger.info(
        "split    train %d ex / %d docs | eval %d ex / %d docs",
        len(train),
        len(docs) - n_eval,
        len(evl),
        n_eval,
    )
    return train, evl


def build_chat_prompt(processor, example: Example) -> str:
    content = [
        {"type": "image", "image": example.image},
        {"type": "text", "text": example.prompt},
    ]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Feature extraction (frozen backbone)
# ---------------------------------------------------------------------------

_features_fn = nnx.jit(model_1.Qwen3VLBBox.features)


def encode_features(
    model: model_1.Qwen3VLBBox,
    processor,
    examples: list[Example],
    *,
    batch_size: int,
    max_seq_len: int,
    image_pad_id: int,
    tag: str,
) -> jax.Array:
    """Pooled backbone features for every example, ``[N, D]``.

    Sequences are padded to ``max_seq_len`` so that every batch compiles to the
    same shape; combined with the fixed page size that means exactly one trace
    of the 4B forward pass for the whole run.
    """
    vcfg = model.backbone.config.vision_config
    assert vcfg is not None
    chunks = []
    expected_image_tokens: int | None = None
    # Split the wall clock: `prep` is the HuggingFace image processor and chat
    # template on CPU, `fwd` is the backbone.  They are the same order of
    # magnitude here, so it is worth knowing which one a slow run is stuck in.
    t_prep = t_fwd = 0.0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        t0 = time.perf_counter()
        batch = encode_batch(
            processor,
            [build_chat_prompt(processor, e) for e in chunk],
            [[e.image] for e in chunk],
            vcfg=vcfg,
            max_length=max_seq_len,
            padding="max_length",
        )
        t_prep += time.perf_counter() - t0

        # Truncation would drop image tokens without raising, leaving the vision
        # embeddings silently unused (model.py scatters them by position).  Every
        # page is the same size, so the count must never move.
        n_image = (batch.input_tokens == image_pad_id).sum(axis=1)
        if expected_image_tokens is None:
            expected_image_tokens = int(n_image[0])
        if n_image.min() != expected_image_tokens or n_image.max() != n_image.min():
            raise RuntimeError(
                f"image tokens per example changed ({n_image.tolist()} vs "
                f"{expected_image_tokens}); --max-seq-len is truncating the image"
            )

        t0 = time.perf_counter()
        features = _features_fn(
            model,
            jnp.asarray(batch.input_tokens),
            jnp.asarray(batch.positions),
            jnp.asarray(batch.pixel_values, dtype=jnp.bfloat16),
            batch.vision_grid,
            jnp.asarray(batch.input_mask).astype(jnp.bool_),
        ).astype(jnp.float32)
        features.block_until_ready()
        t_fwd += time.perf_counter() - t0
        chunks.append(features)

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

    return jnp.concatenate(chunks, axis=0)


def probe_seq_len(processor, example: Example, vcfg) -> int:
    """Unpadded token length of one encoded example.

    Only the query text varies between examples and it is a handful of tokens,
    so this plus a small headroom is a safe fixed padding length.
    """
    batch = encode_batch(
        processor,
        [build_chat_prompt(processor, example)],
        [[example.image]],
        vcfg=vcfg,
        max_length=32768,
        padding=False,
    )
    return int(batch.input_tokens.shape[1])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@nnx.jit
def train_step(
    head: model_1.BBoxHead,
    optimizer: nnx.Optimizer,
    features: jax.Array,  # [B, D]
    targets: jax.Array,  # [B, 4]
) -> jax.Array:
    def loss_fn(head):
        return model_1.bbox_loss(head(features), targets)

    loss, grads = nnx.value_and_grad(loss_fn)(head)
    optimizer.update(head, grads)
    return loss


def _centres(boxes: jax.Array) -> jax.Array:
    return jnp.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=-1
    )


def box_metrics(pred: jax.Array, targets: jax.Array) -> dict:
    """IoU, IoU-threshold accuracies, and centre-based metrics.

    The centre metrics carry most of the signal on this dataset.  The median
    target is a single text line -- 0.13% of the page, ~1% of page height -- so
    a prediction one line off scores IoU 0 exactly like a prediction on the
    wrong half of the page, and mean IoU on its own cannot tell the two apart.
    ``centre_hit`` (predicted centre lands inside the true box) is the same
    thing LocateAnything scores as "pointing".
    """
    iou = np.asarray(model_1.box_iou(pred, targets))
    pc, tc = _centres(pred), _centres(targets)
    dist = np.asarray(jnp.linalg.norm(pc - tc, axis=-1))
    inside = np.asarray(
        (pc[:, 0] >= targets[:, 0])
        & (pc[:, 0] <= targets[:, 2])
        & (pc[:, 1] >= targets[:, 1])
        & (pc[:, 1] <= targets[:, 3])
    )
    return {
        "iou": float(iou.mean()),
        "acc50": float((iou >= 0.5).mean()),
        "acc75": float((iou >= 0.75).mean()),
        "centre_dist": float(dist.mean()),
        "centre_hit": float(inside.mean()),
    }


def evaluate(head: model_1.BBoxHead, features: jax.Array, targets: jax.Array) -> dict:
    """Metrics for a whole split, plus the loss."""
    raw = head(features)
    # Scored on clipped boxes (a box cannot extend past the page), but the loss
    # is reported on the raw output so it stays comparable to the training loss.
    metrics = box_metrics(model_1.clip_boxes(raw), targets)
    metrics["loss"] = float(model_1.bbox_loss(raw, targets))
    return metrics


def mean_box_baseline(train_targets: jax.Array, eval_targets: jax.Array) -> dict:
    """Scores of the constant box that best fits the training set."""
    const = jnp.mean(train_targets, axis=0, keepdims=True)
    return box_metrics(jnp.broadcast_to(const, eval_targets.shape), eval_targets)


def tb_tag(key: str) -> str:
    """``eval_iou`` -> ``iou/eval``.

    TensorBoard groups charts by the text before the slash, so putting the split
    last draws train and eval as two lines on one chart instead of two charts.
    """
    for split in ("train", "eval"):
        if key.startswith(f"{split}_"):
            return f"{key[len(split) + 1 :]}/{split}"
    if key == "batch_loss":
        return "loss/batch"
    return key


class MetricWriter:
    """One row per epoch, to CSV and to TensorBoard.

    The console log is sampled (``--log-every``) and dies with the process; both
    of these survive it.  Flushing per row means a run killed at epoch 90 still
    leaves 89 usable rows -- which happened repeatedly while developing these
    experiments.

    Each run gets its own subdirectory under ``tb_dir``, so pointing TensorBoard
    at the parent overlays every run done so far.
    """

    def __init__(
        self,
        csv_path: str,
        tb_dir: str,
        run_name: str,
        fieldnames: list[str] | None = None,
        append: bool = False,
    ):
        """``fieldnames`` declares the full schema up front.

        ``append`` keeps an existing CSV and adds to it instead of truncating,
        which is what a resumed run wants.  Without it a restart silently
        destroys the previous run's history -- which is exactly what happened to
        2400 steps of the first long LoRA run.

        A CSV header is fixed by the first row written, so a run that logs cheap
        rows often and expensive ones (evaluation) rarely has to declare both
        sets in advance; missing values are left blank.  Leave it None when
        every row has the same keys.
        """
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self.path = csv_path
        resuming = append and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        # Held open for the run and flushed per row, so a killed run still
        # leaves a readable file -- a context manager would close it here.
        self._file = open(csv_path, "a" if resuming else "w", newline="")  # noqa: SIM115
        self._wrote_header = resuming
        self._fieldnames = fieldnames
        self._writer: csv.DictWriter | None = None
        self.tb_path = os.path.join(tb_dir, run_name)
        self._tb = SummaryWriter(self.tb_path)

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames or list(row), restval=""
            )
            if not self._wrote_header:
                self._writer.writeheader()
                self._wrote_header = True
        self._writer.writerow(row)
        self._file.flush()

        epoch = int(row["epoch"])
        for key, value in row.items():
            if key != "epoch":
                self._tb.add_scalar(tb_tag(key), float(value), epoch)
        self._tb.flush()

    def close(self) -> None:
        self._file.close()
        self._tb.close()


def metric_row(epoch: int, batch_loss: float, train: dict, evl: dict) -> dict:
    """Flatten a pair of metric dicts into one CSV row.

    Three different losses, deliberately named apart: ``batch_loss`` is the mean
    minibatch loss *during* the epoch (what the console prints), while
    ``train_loss`` and ``eval_loss`` come from ``evaluate`` over the full splits
    *after* it.  Calling the first one ``train_loss`` collides with the
    ``train_`` prefix applied to the eval dict and silently drops it -- which is
    exactly what the first version of this function did.
    """
    row: dict = {"epoch": epoch, "batch_loss": batch_loss}
    for prefix, metrics in (("train_", train), ("eval_", evl)):
        for key, value in metrics.items():
            name = prefix + key
            if name in row:
                raise ValueError(f"metric name collision on {name!r}")
            row[name] = value
    return row


def save_visualisations(
    examples: list[Example], pred: np.ndarray, out_dir: str, limit: int
) -> int:
    """Ground truth in green, prediction in red, one PNG per example."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(limit, len(examples))
    for i in range(n):
        example = examples[i]
        image = example.image.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for box, colour, label in (
            (example.box, (0, 200, 0), f"gt {example.query}"),
            (pred[i], (220, 0, 0), "pred"),
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
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument(
        "--base-config",
        default=None,
        help="ModelConfig factory name, e.g. qwen3vl_4b (default: infer from --model)",
    )
    p.add_argument("--rows", type=int, default=64, help="documents to stream")
    p.add_argument("--max-per-doc", type=int, default=8)
    p.add_argument("--max-examples", type=int, default=1024)
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--page-width", type=int, default=PAGE_WIDTH)
    p.add_argument("--page-height", type=int, default=PAGE_HEIGHT)
    p.add_argument("--max-seq-len", type=int, default=None, help="default: probed")
    p.add_argument("--encode-batch", type=int, default=4, help="backbone batch")
    p.add_argument("--batch-size", type=int, default=32, help="head batch")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", choices=("last", "mean"), default="last")
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=5, help="epochs between logs")
    p.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="epochs between metrics.csv rows (evaluation is cheap; 1 is fine)",
    )
    p.add_argument("--out-dir", default="output/bbox_heads")
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

    examples = load_examples(
        rows=args.rows,
        max_per_doc=args.max_per_doc,
        max_examples=args.max_examples,
        page_size=(args.page_width, args.page_height),
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("no usable examples; raise --rows")
    train_ex, eval_ex = split_by_document(examples, args.eval_frac, args.seed)
    if not train_ex or not eval_ex:
        raise RuntimeError("empty split; raise --rows or adjust --eval-frac")
    logger.info("prompt   %r -> box %s", train_ex[0].prompt, train_ex[0].box.tolist())

    config = getattr(ModelConfig, args.base_config)() if args.base_config else None
    processor, backbone = load_model(args.model, dtype=jnp.bfloat16, config=config)
    config = backbone.config
    if config.vision_config is None:
        raise ValueError(f"{args.model} resolved to a text-only config")
    # load_model returns a concrete Qwen2VLProcessor; encode_batch annotates the
    # parameter as the AutoProcessor factory.  Same object at runtime.
    processor = cast(AutoProcessor, processor)

    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        max_seq_len = probe_seq_len(processor, train_ex[0], config.vision_config) + 32
    logger.info("seq len  %d (padded)", max_seq_len)

    model = model_1.Qwen3VLBBox(
        backbone,
        rngs=nnx.Rngs(args.seed),
        head_hidden_dim=args.head_hidden,
        pool=args.pool,
    )

    encode_kwargs = dict(
        batch_size=args.encode_batch,
        max_seq_len=max_seq_len,
        image_pad_id=config.vision_config.image_pad_id,
    )
    train_x = encode_features(model, processor, train_ex, tag="train", **encode_kwargs)
    eval_x = encode_features(model, processor, eval_ex, tag="eval", **encode_kwargs)
    train_y = jnp.asarray(np.stack([e.box for e in train_ex]))
    eval_y = jnp.asarray(np.stack([e.box for e in eval_ex]))

    # `model.head` is the same object, so training it in place also updates
    # `model`: afterwards `model(...)` is a working page+query -> box detector.
    head = model.head
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(head, nnx.Param)))
    logger.info("head     %d trainable params, backbone frozen", n_params)

    steps_per_epoch = max(1, len(train_ex) // args.batch_size)
    # Cosine decay matters more than usual: L1 has a constant gradient magnitude,
    # so at a fixed lr the head settles into a limit cycle around the target
    # rather than landing on it.
    schedule = optax.cosine_decay_schedule(args.lr, args.epochs * steps_per_epoch)
    optimizer = nnx.Optimizer(
        head,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        ),
        wrt=nnx.Param,
    )

    base = mean_box_baseline(train_y, eval_y)
    logger.info(
        "baseline mean-box eval iou %.4f  centre hit %.3f  centre dist %.4f",
        base["iou"],
        base["centre_hit"],
        base["centre_dist"],
    )

    run_name = args.run_name or f"pool_{args.pool}-{time.strftime('%m%d-%H%M')}"
    metrics = MetricWriter(
        os.path.join(args.out_dir, "metrics.csv"), args.tb_dir, run_name
    )
    logger.info("tb       %s", metrics.tb_path)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_ex))
        losses = []
        for step in range(steps_per_epoch):
            idx = jnp.asarray(
                order[step * args.batch_size : (step + 1) * args.batch_size]
            )
            losses.append(train_step(head, optimizer, train_x[idx], train_y[idx]))
        if epoch % args.metrics_every and epoch not in (1, args.epochs):
            continue
        train_loss = float(jnp.mean(jnp.stack(losses)))
        ev = evaluate(head, eval_x, eval_y)
        n_tr = min(len(train_ex), TRAIN_METRIC_EXAMPLES)
        tr = evaluate(head, train_x[:n_tr], train_y[:n_tr])
        metrics.write(metric_row(epoch, train_loss, tr, ev))
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            logger.info(
                "epoch %3d  loss %.4f | train iou %.4f hit %.3f | "
                "eval iou %.4f hit %.3f dist %.4f acc@0.5 %.3f",
                epoch,
                train_loss,
                tr["iou"],
                tr["centre_hit"],
                ev["iou"],
                ev["centre_hit"],
                ev["centre_dist"],
                ev["acc50"],
            )

    metrics.close()
    logger.info("wrote    %s", metrics.path)

    final = evaluate(head, eval_x, eval_y)
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

    if args.vis:
        pred = np.asarray(model_1.clip_boxes(head(eval_x)))
        n = save_visualisations(eval_ex, pred, args.out_dir, args.vis)
        logger.info("wrote    %d PNGs to %s", n, args.out_dir)


if __name__ == "__main__":
    main()
