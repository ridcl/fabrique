"""Evaluate a document-VQA model that predicts values *and* bounding boxes.

Target model:   https://huggingface.co/ridcl/paperwerk-vqa
Target dataset: https://huggingface.co/datasets/ridcl/vqa_kvp10k_synth

The model takes a (multi-page) document plus a list of free-form queries and
returns a JSON list of ``{query, value, box_2d, index}`` items, where ``box_2d``
is ``[x0, y0, x1, y1]`` on a 0..1000 scale and ``index`` is the page the value
was found on.  A single query may have *many* answers (e.g. "names of the
parties" -> three names), so scoring is set-based rather than a dict lookup.

    ⚠️  CONTAMINATION WARNING
    Every row of vqa_kvp10k_synth has ``split == "train"`` and the model was
    (per the user) trained on this dataset with a split that was not saved.
    Numbers produced here are therefore an *optimistic upper bound* — treat
    them as a smoke test / relative-comparison harness, not a generalisation
    estimate.  Each row carries a ``source`` like
    ``registration_form/kvp10k_17710e7c4393#00431``; grouping by that prefix is
    the obvious basis for a real holdout if one is ever constructed.

Metrics
-------
value_f1     Macro-averaged F1 over documents of exact value extraction.
             Predictions are matched to ground truth within a query by exact
             (whitespace-normalised) value string; page and box are ignored.
iou_matched  Mean IoU over value-matched pairs only.  Comparable to the metric
             in ``qwen3vl_kvp10k.py``, and biased upward: it only scores boxes
             for items the model already got right.
iou_all      Mean IoU over *all* ground-truth items, scoring a miss as 0.  This
             is the honest grounding number; prefer it when comparing models.
acc@0.5      Fraction of all ground-truth items localised with IoU >= 0.5
acc@0.75     ... and >= 0.75.
page_acc     Among value-matched pairs, fraction with the correct page index.
parse_errors Fraction of documents whose output was not parseable JSON.

A box on the wrong page scores IoU 0, so ``page_acc`` and the IoU metrics stay
independent of the value-matching step.

Usage
-----
    python experiments/doc_vqa_eval.py --n 100
    python experiments/doc_vqa_eval.py --n 20 --vis-dir output/doc_vqa_eval
"""

import argparse
import io
import json
import logging
import os
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from PIL import Image, ImageDraw

from fabrique.models.qwen3vl import model as model_lib
from fabrique.models.qwen3vl.sampler import load_sampler

# force=True: importing jax/tunix installs a root logging handler, which makes
# a plain basicConfig() a silent no-op -- the root level stays at WARNING and
# every logger.info below is discarded, so a long run looks like it has hung.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "ridcl/paperwerk-vqa"
# Fine-tuned repos cannot be resolved to an architecture by name, so state it.
BASE_CONFIG = "qwen3vl_4b"
DATASET_ID = "ridcl/vqa_kvp10k_synth"
DATASET_SHARDS = 7  # data/vqa_kvp10k_synth-0000X-of-00007.parquet

N_SAMPLES = 100
RANDOM_SEED = 42

# Sequence budget.  Qwen3-VL uses patch 16 with a 2x2 spatial merge, so one
# visual token covers a 32x32 px region: an 896 px page is ~28x28 = 784 tokens.
# Documents in this dataset go up to 32 pages and 348 queries, which will not
# fit any reasonable cache, so oversized documents are filtered out and the
# count is reported.
# Pages are natively 1242x1756 (A4 @ 150 dpi).  Downscaling to 896 px throws
# away ~74% of the pixels and small form text stops being legible, so keep the
# long edge at 1536 and pay for it by allowing fewer pages.
MAX_IMAGE_SIZE = 1536
MAX_PAGES = 2
MAX_QUERIES = 48
EVAL_CACHE_SIZE = 12288
EVAL_MAX_NEW_TOKENS = 4096

PROMPT_HEADER = "Extract values:"

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data loading
#
# The dataset is 13.6 GB across 7 parquet shards, but it is written with one
# row per row group (~2 MB each), so individual rows can be fetched over HTTP
# by range request.  Sampling is therefore uniform over all 5000 documents
# while downloading only the rows actually scored (~2 MB x n).
# ---------------------------------------------------------------------------


def _shard_paths() -> list[str]:
    return [
        f"datasets/{DATASET_ID}/data/vqa_kvp10k_synth-{i:05d}-of-{DATASET_SHARDS:05d}.parquet"
        for i in range(DATASET_SHARDS)
    ]


def load_rows(n: int, seed: int, full_download: bool = False) -> list[dict]:
    """Return ``n`` randomly sampled dataset rows as plain dicts.

    Reads only the row groups it needs, grouped by shard so each remote file is
    opened once.  ``full_download`` pulls the whole 13.6 GB instead, which is
    only worth it if you plan to iterate on the same sample repeatedly.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    rng = random.Random(seed)

    if full_download:
        import pandas as pd

        df = pd.concat(
            [pd.read_parquet(f"hf://{p}") for p in _shard_paths()], ignore_index=True
        )
        return [
            df.iloc[i].to_dict() for i in rng.sample(range(len(df)), min(n, len(df)))
        ]

    fs = HfFileSystem()
    index: list[tuple[str, int]] = []
    for path in _shard_paths():
        with fs.open(path, "rb") as fh:
            index += [(path, g) for g in range(pq.ParquetFile(fh).num_row_groups)]
    logger.info(
        "%s: %d documents indexed across %d shards",
        DATASET_ID,
        len(index),
        DATASET_SHARDS,
    )

    # Oversample: some documents get dropped by the page/query budget filter.
    rng.shuffle(index)
    wanted = index[: min(len(index), max(n * 2, n + 16))]

    by_shard: dict[str, list[int]] = {}
    for path, g in wanted:
        by_shard.setdefault(path, []).append(g)

    rows: list[dict] = []
    for path, groups in by_shard.items():
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for g in sorted(groups):
                rows += pf.read_row_group(g).to_pylist()
        logger.info(
            "%s: %d rows (%d total)", os.path.basename(path), len(groups), len(rows)
        )

    rng.shuffle(rows)
    return rows


def _decode_images(raw: Iterable[bytes]) -> list[Image.Image]:
    images = []
    for buf in raw:
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        if max(img.size) > MAX_IMAGE_SIZE:
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Normalised representation
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One (query, value, box, page) tuple, boxes normalised to 0..1."""

    query: str
    value: str
    bbox: tuple[float, float, float, float] | None
    page: int


@dataclass
class Sample:
    images: list[Image.Image]
    queries: list[str]
    gt: list[Item]
    source: str


def _norm_text(s: Any, ignore_case: bool) -> str:
    s = _WS.sub(" ", str(s)).strip()
    return s.casefold() if ignore_case else s


def _as_bbox(raw: Any, scale: float) -> tuple[float, float, float, float] | None:
    """Coerce a 4-element box to a normalised, corner-ordered tuple."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) / scale for v in raw)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def build_sample(row: dict, ignore_case: bool) -> Sample | None:
    """Convert a raw dataset row into a ``Sample``, or None if over budget."""
    images_raw = list(row["images"])
    queries = [str(q) for q in row["queries"]]
    if len(images_raw) > MAX_PAGES or len(queries) > MAX_QUERIES:
        return None
    gt = [
        Item(
            query=_norm_text(a["query"], ignore_case),
            value=_norm_text(a["value"], ignore_case),
            bbox=_as_bbox(a["bounding_box"], scale=1.0),
            page=int(a.get("index", 0) or 0),
        )
        for a in row["answers"]
    ]
    return Sample(
        images=_decode_images(images_raw),
        queries=queries,
        gt=gt,
        source=str(row.get("source", "")),
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def build_prompt(processor, sample: Sample) -> str:
    """Format the chat prompt, matching the model card's training format."""
    query_list = "\n".join(f"- {q}" for q in sample.queries)
    content: list[dict] = [{"type": "image", "image": img} for img in sample.images]
    content.append({"type": "text", "text": f"{PROMPT_HEADER}\n{query_list}"})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _iter_json_objects(text: str) -> Iterable[str]:
    """Yield top-level ``{...}`` substrings, respecting strings and escapes."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def parse_output(text: str, ignore_case: bool) -> tuple[list[Item], str]:
    """Parse the model's JSON list into items.

    Returns ``(items, status)`` where status is ``ok`` (the whole list parsed),
    ``salvaged`` (the list was truncated or partly malformed, so individually
    well-formed objects were recovered), or ``failed`` (nothing usable).

    Salvaging matters because the model routinely runs past the token budget or
    emits a broken object mid-list; scoring the items it *did* produce is more
    informative than zeroing the document.  Objects that are themselves invalid
    JSON are dropped — a downstream consumer could not use them either.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)

    raw: list | None = None
    status = "ok"
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                raw = parsed
        except (json.JSONDecodeError, ValueError):
            raw = None

    if raw is None:
        status = "salvaged"
        raw = []
        for chunk in _iter_json_objects(text):
            try:
                obj = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                raw.append(obj)
        if not raw:
            return [], "failed"

    items = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            page = int(entry.get("index", 0) or 0)
        except (TypeError, ValueError):
            page = 0
        items.append(
            Item(
                query=_norm_text(entry.get("query", ""), ignore_case),
                value=_norm_text(entry.get("value", ""), ignore_case),
                bbox=_as_bbox(entry.get("box_2d"), scale=1000.0),
                page=page,
            )
        )
    return items, status


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def iou(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_items(preds: list[Item], gts: list[Item]) -> list[tuple[int, int]]:
    """Match predictions to ground truth on (query, value) equality.

    Where several candidates tie (a query with repeated values), pairs are
    assigned greedily highest-IoU-first, which is charitable to the model and
    avoids a scipy dependency for the exact assignment.
    """
    cands = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p.query == g.query and p.value == g.value:
                score = iou(p.bbox, g.bbox) if p.page == g.page else 0.0
                cands.append((score, pi, gi))
    cands.sort(key=lambda t: -t[0])

    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for _, pi, gi in cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))
    return pairs


@dataclass
class Accumulator:
    f1s: list[float] = field(default_factory=list)
    iou_matched: list[float] = field(default_factory=list)
    iou_all: list[float] = field(default_factory=list)
    page_hits: list[float] = field(default_factory=list)
    n_docs: int = 0
    n_parse_errors: int = 0
    n_salvaged: int = 0

    def add(self, preds: list[Item], gts: list[Item], status: str = "ok") -> dict:
        self.n_docs += 1
        if status == "failed":
            self.n_parse_errors += 1
        elif status == "salvaged":
            self.n_salvaged += 1

        pairs = match_items(preds, gts)
        tp = len(pairs)
        precision = tp / len(preds) if preds else 0.0
        recall = tp / len(gts) if gts else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        self.f1s.append(f1)

        # A box on the wrong page counts as a miss.
        doc_ious = []
        for pi, gi in pairs:
            p, g = preds[pi], gts[gi]
            doc_ious.append(iou(p.bbox, g.bbox) if p.page == g.page else 0.0)
            self.page_hits.append(float(p.page == g.page))
        self.iou_matched += doc_ious
        # Unmatched ground truth scores 0.
        self.iou_all += doc_ious + [0.0] * (len(gts) - len(pairs))

        return {
            "f1": f1,
            "tp": tp,
            "n_pred": len(preds),
            "n_gt": len(gts),
            "iou": float(np.mean(doc_ious)) if doc_ious else 0.0,
        }

    def summary(self) -> dict[str, float]:
        def mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        allv = np.array(self.iou_all) if self.iou_all else np.zeros(0)
        return {
            "n_docs": self.n_docs,
            "value_f1": mean(self.f1s),
            "iou_matched": mean(self.iou_matched),
            "iou_all": mean(self.iou_all),
            "acc@0.5": float((allv >= 0.5).mean()) if allv.size else 0.0,
            "acc@0.75": float((allv >= 0.75).mean()) if allv.size else 0.0,
            "page_acc": mean(self.page_hits),
            "parse_errors": self.n_parse_errors / max(self.n_docs, 1),
            "salvaged": self.n_salvaged / max(self.n_docs, 1),
            "n_gt_items": len(self.iou_all),
        }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize(sample: Sample, preds: list[Item] | None, out_dir: str, tag: str) -> None:
    """Write one annotated image per page: GT in blue, predictions in red."""
    os.makedirs(out_dir, exist_ok=True)
    for page, img in enumerate(sample.images):
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        W, H = vis.size
        for items, colour, dy in ((sample.gt, "blue", -22), (preds or [], "red", -11)):
            for it in items:
                if it.bbox is None or it.page != page:
                    continue
                x0, y0, x1, y1 = it.bbox
                box = [x0 * W, y0 * H, x1 * W, y1 * H]
                draw.rectangle(box, outline=colour, width=2)
                draw.text((box[0], max(0, box[1] + dy)), it.query[:28], fill=colour)
        vis.save(os.path.join(out_dir, f"{tag}_p{page}.jpg"), quality=85)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    sampler,
    samples: list[Sample],
    vis_dir: str | None,
    ignore_case: bool,
    dump: str | None,
) -> dict[str, float]:
    acc = Accumulator()
    records = []

    for i, sample in enumerate(samples):
        prompt = build_prompt(sampler._processor, sample)
        try:
            output = sampler(
                prompts=[prompt],
                images=[sample.images],  # one list of pages for this prompt
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
            )[0]
        except ValueError as exc:  # sequence longer than the cache
            logger.warning("[%d/%d] skipped (%s)", i + 1, len(samples), exc)
            continue

        preds, status = parse_output(output, ignore_case)
        stats = acc.add(preds, sample.gt, status)
        logger.info(
            "[%d/%d] pages=%d queries=%d gt=%d pred=%d tp=%d f1=%.3f iou=%.3f%s",
            i + 1,
            len(samples),
            len(sample.images),
            len(sample.queries),
            stats["n_gt"],
            stats["n_pred"],
            stats["tp"],
            stats["f1"],
            stats["iou"],
            "" if status == "ok" else f"  [{status.upper()}]",
        )

        if vis_dir:
            visualize(sample, preds, vis_dir, f"{i:03d}")
        if dump:
            records.append(
                {"source": sample.source, "status": status, "output": output, **stats}
            )

    if dump:
        with open(dump, "w") as fh:
            json.dump(records, fh, indent=2)
        logger.info("wrote per-document records to %s", dump)

    return acc.summary()


def main() -> None:
    global MAX_IMAGE_SIZE, MAX_PAGES, MAX_QUERIES, EVAL_MAX_NEW_TOKENS

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n", type=int, default=N_SAMPLES, help="documents to score")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument(
        "--ignore-case",
        action="store_true",
        help="case-insensitive value matching (default: exact)",
    )
    ap.add_argument(
        "--full-download",
        action="store_true",
        help="pull all 13.6 GB up front instead of fetching rows on "
        "demand; only worth it when re-running many times",
    )
    ap.add_argument("--vis-dir", default=None, help="write annotated pages here")
    ap.add_argument("--dump", default=None, help="write per-document JSON records here")
    ap.add_argument("--cache-size", type=int, default=EVAL_CACHE_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    ap.add_argument(
        "--max-image-size",
        type=int,
        default=MAX_IMAGE_SIZE,
        help="long edge in px; pages are natively 1242x1756",
    )
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    ap.add_argument(
        "--base-config",
        default=BASE_CONFIG,
        choices=["qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"],
        help="architecture of the base model (fine-tuned repo names "
        "cannot be resolved automatically)",
    )
    args = ap.parse_args()

    MAX_IMAGE_SIZE = args.max_image_size
    MAX_PAGES = args.max_pages
    MAX_QUERIES = args.max_queries
    EVAL_MAX_NEW_TOKENS = args.max_new_tokens

    logger.warning(
        "Every row of %s is split='train' and the model was trained on this "
        "dataset -> these numbers are an optimistic upper bound, not a "
        "generalisation estimate.",
        DATASET_ID,
    )

    rows = load_rows(args.n, args.seed, full_download=args.full_download)
    samples, n_over_budget = [], 0
    for row in rows:
        if len(samples) >= args.n:
            break
        sample = build_sample(row, args.ignore_case)
        if sample is None:
            n_over_budget += 1
            continue
        samples.append(sample)
    logger.info(
        "%d documents selected (%d skipped: >%d pages or >%d queries)",
        len(samples),
        n_over_budget,
        MAX_PAGES,
        MAX_QUERIES,
    )

    mesh = jax.make_mesh((1, len(jax.devices())), ("fsdp", "tp"))
    config = getattr(model_lib.ModelConfig, args.base_config)()
    sampler = load_sampler(
        args.model, cache_size=args.cache_size, mesh=mesh, config=config
    )

    metrics = evaluate(sampler, samples, args.vis_dir, args.ignore_case, args.dump)

    print("\n" + "=" * 46)
    print(f"{args.model}  on  {DATASET_ID}")
    print("=" * 46)
    for key, value in metrics.items():
        print(
            f"  {key:<14} {value:.4f}"
            if isinstance(value, float)
            else f"  {key:<14} {value}"
        )
    print("=" * 46)


if __name__ == "__main__":
    main()
