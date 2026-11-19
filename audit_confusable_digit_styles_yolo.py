"""
Visual audit of TRAINING data crops for the digit classes most involved in
the confusion pairs found by inspect_small_scale_failures_v3.py (run on the
VALIDATION set):

    7 <-> 1   (71 total confusions, both directions)
    6 -> 8    (20)
    3 -> 2    (19)
    9 -> 8    (18)
    2 -> 1    (18)
    5 -> 6    (18)

And the digits with the highest per-class error rate: 7 (23.5%), 4 (18.7%),
3 (18.4%), 5 (18.0%).

This version matches the REAL format of 70_30_dataset/training/: a FLAT
folder (no images/ or labels/ subfolders) where each image "<stem>.jpg" has
a matching "<stem>.txt" in YOLO format:

    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

one line per digit instance in that image (can be multiple lines/digits per
image), all box coordinates normalized 0-1 relative to that image's width/
height. class_id IS the digit (0-9).

For each target digit, this samples real instances, crops the YOLO box out
of the full image, and tiles them into a contact sheet for visual review --
no model inference involved, just "what does a labeled digit N actually
look like in this training data".

Usage:
    python audit_confusable_digit_styles_yolo.py \
        --train_dir ~/workspace_bhavana/Datasets/70_30_dataset/training \
        --out_dir confusable_digit_audit \
        --digits 1,7,6,8,3,2,9,5,4 \
        --n_per_digit 30
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True,
                     help="flat folder containing <stem>.jpg + <stem>.txt pairs "
                          "(YOLO format), e.g. 70_30_dataset/training")
    ap.add_argument("--out_dir", default="confusable_digit_audit")
    ap.add_argument("--digits", default="1,7,6,8,3,2,9,5,4",
                     help="comma-separated digit classes to audit")
    ap.add_argument("--n_per_digit", type=int, default=30,
                     help="how many example crops to sample per digit class")
    ap.add_argument("--seed", type=int, default=42)
    opt = ap.parse_args()

    random.seed(opt.seed)
    target_digits = set(int(d) for d in opt.digits.split(","))
    os.makedirs(opt.out_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(opt.train_dir, "*.txt")))
    print(f"Found {len(txt_files)} label .txt files in {opt.train_dir}")
    if not txt_files:
        print("ERROR: no .txt files found -- check --train_dir path")
        return

    # pool[digit] = list of (jpg_path, cx, cy, w, h)  (all normalized 0-1)
    pool = {d: [] for d in target_digits}
    n_parsed = 0
    n_bad_lines = 0

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
                n_bad_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                n_bad_lines += 1
                continue
            n_parsed += 1
            if cls in pool:
                pool[cls].append((jpg_path, cx, cy, w, h))

    print(f"Parsed {n_parsed} digit instances total ({n_bad_lines} malformed lines skipped)")
    print()
    for d in sorted(target_digits):
        print(f"  digit {d}: {len(pool[d])} labeled instances available")

    cell = 140
    label_h = 14
    cols = 10

    for d in sorted(target_digits):
        samples = pool[d]
        if not samples:
            print(f"  [SKIP] digit {d}: no samples found")
            continue
        random.shuffle(samples)
        samples = samples[: opt.n_per_digit]

        crops = []
        for jpg_path, cx, cy, w, h in samples:
            img = cv2.imread(jpg_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            box_w, box_h = w * W, h * H
            x1 = int((cx * W) - box_w / 2)
            y1 = int((cy * H) - box_h / 2)
            x2 = int((cx * W) + box_w / 2)
            y2 = int((cy * H) + box_h / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, (cell - 8, cell - 8 - label_h))
            crops.append(crop_resized)

        if not crops:
            print(f"  [SKIP] digit {d}: no valid crops after loading")
            continue

        rows = (len(crops) + cols - 1) // cols
        sheet = np.full((rows * cell, cols * cell, 3), 30, dtype=np.uint8)
        for i, c in enumerate(crops):
            r, col = divmod(i, cols)
            y0 = r * cell + label_h
            x0 = col * cell + 4
            ch, cw = c.shape[:2]
            sheet[y0:y0 + ch, x0:x0 + cw] = c

        cv2.putText(sheet, f"digit {d}  (n={len(crops)} of {len(pool[d])} available)",
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(opt.out_dir, f"digit_{d}_samples.jpg")
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved digit {d} contact sheet ({len(crops)} crops) -> {out_path}")

    print()
    print(f"Done. Review the contact sheets in {opt.out_dir}/ for each digit class,")
    print("especially comparing digit 7 vs 1, 6 vs 8, 3 vs 2, 9 vs 8, 5 vs 6 side by side")
    print("for a systematic font/style cause of the confusion (missing crossbar on 7,")
    print("closed loop on 6, flat-top 3, etc).")


if __name__ == "__main__":
    main()
