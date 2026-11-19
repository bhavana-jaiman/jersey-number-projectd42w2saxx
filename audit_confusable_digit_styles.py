"""
Visual audit of TRAINING data crops for the digit classes most involved in
the confusion pairs found by inspect_small_scale_failures_v3.py:

    7 <-> 1   (71 total confusions, both directions)
    6 -> 8    (20)
    3 -> 2    (19)
    9 -> 8    (18)
    2 -> 1    (18)
    5 -> 6    (18)

And the digits with the highest per-class error rate: 7 (23.5%), 4 (18.7%),
3 (18.4%), 5 (18.0%).

This does NOT run the model -- it just pulls real single-digit crops from
the TRAINING set for each of these digit classes and tiles them into a
contact sheet, so we can visually check for a systematic style cause (e.g.
a "7" rendered without a crossbar looking like "1", a "6" with a closed
loop looking like "8", a "3" with a flat top looking like "2") before
deciding how to augment or reweight the loss.

Crops are the body.rect regions from training labels, split into left/right
digit when gt has 2 digits (since the model has separate digit_1/digit_2
heads operating on crops of the whole number, not single-digit boxes --
adjust SPLIT_STRATEGY below if your training crops are already
single-digit).

Usage:
    python audit_confusable_digit_styles.py \
        --train_dataset ~/workspace_bhavana/Datasets/70_30_dataset_v3/training \
        --out_dir confusable_digit_audit \
        --digits 1,7,6,8,3,2,9,5,4 \
        --n_per_digit 30
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset", required=True,
                     help="folder with images/ and labels/ subfolders "
                          "(e.g. 70_30_dataset_v3/training)")
    ap.add_argument("--out_dir", default="confusable_digit_audit")
    ap.add_argument("--digits", default="1,7,6,8,3,2,9,5,4",
                     help="comma-separated digit classes to audit")
    ap.add_argument("--n_per_digit", type=int, default=30,
                     help="how many example crops to sample per digit class")
    ap.add_argument("--seed", type=int, default=42)
    opt = ap.parse_args()

    random.seed(opt.seed)
    target_digits = [int(d) for d in opt.digits.split(",")]
    os.makedirs(opt.out_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(opt.train_dataset, "labels", "*.json")))
    print(f"Found {len(label_files)} training label files")
    if not label_files:
        print(f"ERROR: no label files found under {opt.train_dataset}/labels/ -- check the path")
        return

    # collect per-digit sample pool: (image_path, body_rect, which_digit_pos, digit_value)
    pool = {d: [] for d in target_digits}

    for label_path in label_files:
        try:
            with open(label_path, "r", encoding="utf8") as f:
                data = json.load(f)
        except Exception:
            continue
        img_name = data.get("file_name")
        if not img_name:
            continue
        img_path = os.path.join(opt.train_dataset, "images", img_name)
        if not os.path.exists(img_path):
            continue

        for item in data.get("data", {}).get("jersey", []):
            digits_gt = [n["digits"] for n in item.get("numbers", [])][: len(item.get("jersey_number", []))]
            if len(digits_gt) == 0 or len(digits_gt) > 2:
                continue
            rect = item.get("body", {}).get("rect")
            if not rect or rect[2] <= 0 or rect[3] <= 0:
                continue
            for pos, dval in enumerate(digits_gt):
                if dval in pool:
                    pool[dval].append((img_path, rect, pos, len(digits_gt)))

    print()
    for d in target_digits:
        print(f"  digit {d}: {len(pool[d])} samples available in training labels")

    cell = 140
    label_h = 14
    cols = 10

    for d in target_digits:
        samples = pool[d]
        if not samples:
            print(f"  [SKIP] digit {d}: no samples found")
            continue
        random.shuffle(samples)
        samples = samples[: opt.n_per_digit]

        crops = []
        for img_path, rect, pos, n_digits in samples:
            img = cv2.imread(img_path)
            if img is None:
                continue
            x, y, w, h = rect
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            # if 2-digit number, split the crop in half horizontally to
            # isolate the digit at this position (approximate -- the real
            # per-digit box isn't stored separately in this label schema)
            if n_digits == 2:
                cw = crop.shape[1]
                if pos == 0:
                    crop = crop[:, : cw // 2]
                else:
                    crop = crop[:, cw // 2:]
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
    print()
    print("NOTE: for 2-digit numbers, the crop is split in half horizontally as an")
    print("approximation of the per-digit region, since the label schema stores one")
    print("body.rect for the whole number, not separate per-digit boxes. If digits")
    print("are not evenly split in the original image (common with proportional")
    print("fonts), some crops may be slightly off-center -- treat minor framing")
    print("issues in the contact sheets as an artifact of this approximation, not")
    print("necessarily a real crop problem.")


if __name__ == "__main__":
    main()
