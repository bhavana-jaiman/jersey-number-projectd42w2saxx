"""
make_body_validation.py
-----------------------
Builds a FIXED body-box-like validation set from full player images with YOLO
digit labels (e.g. 70_30_dataset/validation_excluded).

For every image it cuts ONE crop around the number, the same way the training
code makes its body-like crops (JerseyWithState_Dataset):
  * number height = 15-40% of the crop height   (like the client's body boxes)
  * number placed anywhere from upper to lower part (v_pos 0.15-0.80)
  * crop width = 0.75-1.10 x crop height, at least 1.15 x number width
  * crop never goes outside the real photo
The random choices are seeded per file name, so running the script again gives
exactly the same crops (the validation score is comparable between runs).

Output (same format as validation_final, read by JerseyNumber_ValidationDataset_V2):
  <out>/images/<name>.jpg   the crop at its original resolution (the loader does
                            RGB -> pad to square -> resize 96, like the test)
  <out>/labels/<name>.txt   one line of digits: "1 8", "6", or "10" (= no number)
3+ digit numbers are written too ("1 2 3"); the current validation loader skips them.

Run from New_arcitecture_2:
    python make_body_validation.py
    python make_body_validation.py --src ../Datasets/70_30_dataset/validation_excluded \
                                   --out ../Datasets/70_30_dataset/validation_body
"""
import argparse
import glob
import os
import random
from collections import Counter

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_yolo(label_path):
    """YOLO rows: cls cx cy w h (normalised). Returns (N, 5) array or None."""
    if not os.path.exists(label_path):
        return None
    data = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if data.size == 0 or data.shape[1] != 5:
        return None
    return data


def number_box_and_digits(label, W, H):
    """Same rules as JerseyWithState_Dataset.__getitem__:
       class 10            -> no number (state 0), box of that row
       2 boxes far apart   -> only the first box, 1 digit
       otherwise           -> union of all boxes, digits sorted left -> right
    Returns (x1, y1, x2, y2) in pixels and the digit list."""
    if int(label[0][0]) == 10:
        rows, digits = label[:1], [10]
    elif label.shape[0] == 2 and (abs(label[0][1] - label[1][1]) > 0.25 or
                                  abs(label[0][2] - label[1][2]) > 0.25):
        rows, digits = label[:1], [int(label[0][0])]
    else:
        rows = label[np.argsort(label[:, 1])]
        digits = [int(r[0]) for r in rows]
    x1 = min(r[1] - r[3] / 2 for r in rows) * W
    y1 = min(r[2] - r[4] / 2 for r in rows) * H
    x2 = max(r[1] + r[3] / 2 for r in rows) * W
    y2 = max(r[2] + r[4] / 2 for r in rows) * H
    return (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)), digits


def body_like_crop(box, W, H, rng, ratio_range, vpos_range):
    """Crop box (x_min, y_min, x_max, y_max) in pixels, number fully inside."""
    x1, y1, x2, y2 = box
    num_h = max(1.0, y2 - y1)
    num_w = max(1.0, x2 - x1)
    ratio = rng.uniform(*ratio_range)                     # number height / crop height
    v_pos = rng.uniform(*vpos_range)
    crop_h = num_h / ratio
    crop_w = max(num_w * 1.15, crop_h * rng.uniform(0.75, 1.10))
    y_min = y1 - (crop_h - num_h) * v_pos
    x_min = x1 - (crop_w - num_w) * rng.uniform(0.30, 0.70)
    y_max, x_max = y_min + crop_h, x_min + crop_w
    # stay inside the real photo
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(float(W), x_max), min(float(H), y_max)
    return int(x_min), int(y_min), int(round(x_max)), int(round(y_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../Datasets/70_30_dataset/validation_excluded",
                    help="folder with images/ and labels/ (YOLO)")
    ap.add_argument("--out", default="../Datasets/70_30_dataset/validation_body",
                    help="output folder (images/ and labels/ are created)")
    ap.add_argument("--ratio_min", type=float, default=0.15, help="smallest number height / crop height")
    ap.add_argument("--ratio_max", type=float, default=0.40, help="largest number height / crop height")
    ap.add_argument("--seed", type=int, default=42)
    opt = ap.parse_args()

    img_dir = os.path.join(opt.src, "images")
    lbl_dir = os.path.join(opt.src, "labels")
    out_img = os.path.join(opt.out, "images")
    out_lbl = os.path.join(opt.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    images = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                    if p.lower().endswith(IMG_EXTS))
    print("source images:", len(images))

    stats = Counter()
    ratios_real = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label = read_yolo(os.path.join(lbl_dir, name + ".txt"))
        if label is None:
            stats["skipped: missing/empty/non-YOLO label"] += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            stats["skipped: unreadable image"] += 1
            continue
        W, H = img.size

        box, digits = number_box_and_digits(label, W, H)
        rng = random.Random(f"{opt.seed}-{name}")          # same crop every run
        cx1, cy1, cx2, cy2 = body_like_crop(box, W, H, rng,
                                            (opt.ratio_min, opt.ratio_max), (0.15, 0.80))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            stats["skipped: crop too small"] += 1
            continue

        img.crop((cx1, cy1, cx2, cy2)).save(os.path.join(out_img, name + ".jpg"), quality=95)
        with open(os.path.join(out_lbl, name + ".txt"), "w") as f:
            f.write(" ".join(str(d) for d in digits) + "\n")

        ratios_real.append((box[3] - box[1]) / max(1, cy2 - cy1))
        if digits == [10]:
            stats["written: no number (state 0)"] += 1
        elif len(digits) <= 2:
            stats["written: 1-2 digits (state 1)"] += 1
        else:
            stats["written: 3+ digits (skipped by the current loader)"] += 1

    print("\nresult:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if ratios_real:
        r = np.array(ratios_real)
        print("\nreal number height / crop height after clipping at the photo edge")
        print("  percentiles 5/25/50/75/95:", np.percentile(r, [5, 25, 50, 75, 95]).round(2))
    print("\nsaved to:", opt.out)


if __name__ == "__main__":
    main()
