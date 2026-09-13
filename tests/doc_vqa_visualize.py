"""Render document-VQA predictions over the page, coloured by match quality.

Works from a saved ``--dump`` file, so it needs no GPU and can run while an
evaluation is in flight.  Unlike a flat "GT blue / prediction red" overlay, boxes
are coloured by what actually went wrong, which is the thing you want to see:

    blue    ground truth
    green   matched value, IoU >= 0.5   (localised well)
    orange  matched value, IoU <  0.5   (right text, wrong place) + a line to
            the ground-truth box showing the displacement
    red     predicted value that matched no ground truth (spurious)

Two dumps can be drawn side by side (e.g. JAX vs vLLM) for the same page.

    python tests/doc_vqa_visualize.py --dump output/doc_vqa_eval/eval100.json --n 100
    python tests/doc_vqa_visualize.py --dump output/doc_vqa_eval/records6_mask.json \
        --compare output/doc_vqa_eval/records_vllm.json --n 5
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments"))

import doc_vqa_eval as E  # noqa: E402

GT_COLOR = "#1f77b4"
GOOD_COLOR = "#2ca02c"
POOR_COLOR = "#ff7f0e"
SPURIOUS_COLOR = "#d62728"


def _centre(box, w, h):
    return ((box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h)


def render_page(sample, preds, page, title, iou_good=0.5):
    """Draw one page with quality-coloured boxes; returns a PIL image."""
    vis = sample.images[page].convert("RGB")
    w, h = vis.size
    draw = ImageDraw.Draw(vis)

    pairs = E.match_items(preds, sample.gt)
    matched_pred = {pi: gi for pi, gi in pairs}

    for gt in sample.gt:
        if gt.bbox is None or gt.page != page:
            continue
        draw.rectangle(
            [gt.bbox[0] * w, gt.bbox[1] * h, gt.bbox[2] * w, gt.bbox[3] * h],
            outline=GT_COLOR,
            width=2,
        )

    n_good = n_poor = n_spurious = 0
    for pi, p in enumerate(preds):
        if p.bbox is None or p.page != page:
            continue
        gi = matched_pred.get(pi)
        if gi is None:
            colour, n_spurious = SPURIOUS_COLOR, n_spurious + 1
        else:
            gt = sample.gt[gi]
            iou = E.iou(p.bbox, gt.bbox) if p.page == gt.page else 0.0
            if iou >= iou_good:
                colour, n_good = GOOD_COLOR, n_good + 1
            else:
                colour, n_poor = POOR_COLOR, n_poor + 1
                if gt.bbox is not None and gt.page == page:
                    draw.line(
                        [_centre(p.bbox, w, h), _centre(gt.bbox, w, h)],
                        fill=POOR_COLOR,
                        width=2,
                    )
        box = [p.bbox[0] * w, p.bbox[1] * h, p.bbox[2] * w, p.bbox[3] * h]
        draw.rectangle(box, outline=colour, width=3)
        draw.text((box[0], max(0, box[1] - 11)), p.query[:26], fill=colour)

    banner = (
        f"{title}   good {n_good}  poor {n_poor}  spurious {n_spurious}  "
        f"gt {sum(1 for g in sample.gt if g.page == page)}"
    )
    draw.rectangle([0, 0, w, 18], fill="white")
    draw.text((4, 4), banner, fill="black")
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", required=True, help="records JSON from --dump")
    ap.add_argument("--compare", default=None, help="second dump, drawn alongside")
    ap.add_argument("--n", type=int, default=5, help="must match the eval's --n")
    ap.add_argument("--seed", type=int, default=E.RANDOM_SEED)
    ap.add_argument("--max-pages", type=int, default=E.MAX_PAGES)
    ap.add_argument("--out-dir", default="output/doc_vqa_vis")
    ap.add_argument("--limit", type=int, default=6, help="documents to render")
    args = ap.parse_args()

    E.MAX_PAGES = args.max_pages

    with open(args.dump) as fh:
        primary = {r["source"]: r for r in json.load(fh)}
    secondary = {}
    if args.compare:
        with open(args.compare) as fh:
            secondary = {r["source"]: r for r in json.load(fh)}

    rows = E.load_rows(args.n, args.seed)
    samples, seen = [], 0
    for row in rows:
        if seen >= args.n:
            break
        s = E.build_sample(row, ignore_case=False)
        if s is None:
            continue
        seen += 1
        if s.source in primary:
            samples.append(s)
    print(f"{len(samples)} documents have predictions in {args.dump}")

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for i, s in enumerate(samples[: args.limit]):
        panels = [("dump", primary[s.source])]
        if s.source in secondary:
            panels.append(("compare", secondary[s.source]))
        for page in range(len(s.images)):
            imgs = []
            for label, rec in panels:
                preds, status = E.parse_output(rec["output"], False)
                imgs.append(render_page(s, preds, page, f"{label} ({status})"))
            width = sum(im.width for im in imgs) + 8 * (len(imgs) - 1)
            canvas = Image.new("RGB", (width, max(im.height for im in imgs)), "white")
            x = 0
            for im in imgs:
                canvas.paste(im, (x, 0))
                x += im.width + 8
            path = os.path.join(
                args.out_dir, f"{i:02d}_{s.source.split('/')[0]}_p{page}.jpg"
            )
            canvas.save(path, quality=88)
            written.append(path)
    print(f"wrote {len(written)} image(s) to {args.out_dir}/")
    for p in written[:12]:
        print("  ", p)


if __name__ == "__main__":
    main()
