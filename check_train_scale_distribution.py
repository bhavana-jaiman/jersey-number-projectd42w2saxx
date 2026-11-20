"""
Measures the bbox-size (scale) distribution of digit instances in the real
training data (70_30_dataset/training/, flat folder with <stem>.jpg +
<stem>.txt YOLO-format label pairs), using the SAME bucket edges as the
original failure-mode analysis on the validation set, so the two can be
compared directly.

Why: the original testing-time failure analysis (inspect_small_scale_
failures_v3.py, run on the validation set) found accuracy collapses from
~0.95 at large digit pixel-sizes down to ~0.5-0.58 at small sizes
(roughly <=50px), and that this small-scale bucket held the MAJORITY of
validation samples (~70%). Before deciding whether the fix is scale
augmentation (train/val distribution mismatch) or a resolution/
architecture change (training already has plenty of small-scale examples
but still fails on them), we need to know: does TRAINING data have a
similar scale distribution to validation, or is training mostly
large-scale while validation is mostly small-scale?

bbox pixel size here = max(box_width_px, box_height_px) computed from the
normalized YOLO box (cx, cy, w, h) and the real image's pixel dimensions,
matching how the original validation-side analysis defined "digit scale".

Usage:
    python check_train_scale_distribution.py \
        --train_dir ~/workspace_bhavana/Datasets/70_30_dataset/training \
        --sample_size 20000
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

# Same bucket edges used in the original validation-side scale analysis
# (5-9, 7-11 type overlapping ranges were approximations from that plot;
# here we use clean non-overlapping buckets covering the same regimes:
# small <=50px, mid 50-84px, and larger bins above that, matching the
# "small-scale (<=84px)" cutoff used in inspect_small_scale_failures_v3.py)
BUCKET_EDGES = [0, 10, 17, 25, 35, 50, 65, 84, 110, 150, 220, 10_000]


def bucket_label(edges, i):
    lo, hi = edges[i], edges[i + 1]
    if hi >= 10_000:
        return f"{lo}+"
    return f"{lo}-{hi}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True,
                     help="flat folder containing <stem>.jpg + <stem>.txt pairs "
                          "(YOLO format), e.g. 70_30_dataset/training")
    ap.add_argument("--sample_size", type=int, default=20000,
                     help="randomly sample this many labeled digit instances "
                          "(full dataset has ~40k, sampling keeps this fast)")
    ap.add_argument("--seed", type=int, default=42)
    opt = ap.parse_args()

    random.seed(opt.seed)

    txt_files = glob.glob(os.path.join(opt.train_dir, "*.txt"))
    print(f"Found {len(txt_files)} label files in {opt.train_dir}")
    if not txt_files:
        print("ERROR: no .txt files found -- check --train_dir path")
        return

    # build full instance list: (jpg_path, cx, cy, w, h) all normalized 0-1
    instances = []
    for txt_path in txt_files:
        stem = os.path.splitext(txt_path)[0]
        jpg_path = stem + ".jpg"
        if not os.path.exists(jpg_path):
            continue
        try:
            with open(txt_path, "r", encoding="utf8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                continue
            instances.append((jpg_path, cx, cy, w, h))

    print(f"Total digit instances: {len(instances)}")
    random.shuffle(instances)
    sample = instances[: opt.sample_size]
    print(f"Sampling {len(sample)} instances for scale measurement")
    print()

    # cache image dims per jpg_path to avoid re-reading the same image
    # repeatedly when a file has multiple digit instances
    dim_cache = {}
    bucket_counts = [0] * (len(BUCKET_EDGES) - 1)
    n_skipped = 0

    for jpg_path, cx, cy, w, h in sample:
        if jpg_path not in dim_cache:
            img = cv2.imread(jpg_path)
            if img is None:
                dim_cache[jpg_path] = None
            else:
                dim_cache[jpg_path] = (img.shape[1], img.shape[0])  # W, H
        dims = dim_cache[jpg_path]
        if dims is None:
            n_skipped += 1
            continue
        W, H = dims
        box_w_px = w * W
        box_h_px = h * H
        scale_px = max(box_w_px, box_h_px)

        for i in range(len(BUCKET_EDGES) - 1):
            if BUCKET_EDGES[i] <= scale_px < BUCKET_EDGES[i + 1]:
                bucket_counts[i] += 1
                break

    total_measured = sum(bucket_counts)
    print(f"Measured {total_measured} instances ({n_skipped} skipped -- unreadable image)")
    print()
    print(f"{'bucket (px)':<14}{'count':>10}{'% of total':>14}")
    print("-" * 38)
    small_scale_total = 0
    for i, count in enumerate(bucket_counts):
        label = bucket_label(BUCKET_EDGES, i)
        pct = 100 * count / max(1, total_measured)
        print(f"{label:<14}{count:>10}{pct:>13.1f}%")
        if BUCKET_EDGES[i + 1] <= 84 or (BUCKET_EDGES[i] < 84 <= BUCKET_EDGES[i + 1]):
            small_scale_total += count

    print()
    small_pct = 100 * small_scale_total / max(1, total_measured)
    print(f"Small-scale (<=84px) share in TRAINING data: {small_scale_total} / {total_measured} "
          f"({small_pct:.1f}%)")
    print()
    print("Compare this to the validation-set finding: small-scale (<=84px) was")
    print("~70% of validation samples, with accuracy collapsing to ~0.5-0.58 in")
    print("that regime vs ~0.95 at larger scales.")
    print()
    print("If training's small-scale share is similar to validation's (~70%), the")
    print("model already sees plenty of small-scale examples and still fails on")
    print("them -- pointing toward a resolution/architecture limitation, not a")
    print("train/val distribution mismatch (scale augmentation would help less).")
    print()
    print("If training's small-scale share is much LOWER than validation's, that's")
    print("a real distribution mismatch -- scale augmentation / oversampling by")
    print("bbox size during training should help close the gap.")


if __name__ == "__main__":
    main()
