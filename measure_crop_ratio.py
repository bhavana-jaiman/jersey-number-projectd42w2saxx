"""
measure_crop_ratio.py

Compares how big the digits are INSIDE the model crop in training vs test,
and suggests margins relative to the digit height so test crops look like
training crops.

Training crop (JerseyWithState_Dataset): union of digit boxes + fixed pixel
margins left 23, top 30, right 30, bottom 33 (then random zoom-in).
Test --crop number: same fixed pixel margins around the JSON digit boxes.

Usage (from ~/workspace_bhavana/New_arcitecture):
    python measure_crop_ratio.py \
        --train ../Datasets/70_30_dataset/training \
        --test  ../Datasets/JerseyNumber_Validation_Dataset
"""
import os
import glob
import json
import argparse
import warnings

import numpy as np
from PIL import Image

MARGINS = (23, 30, 30, 33)          # left, top, right, bottom (training)


def union_from_yolo(lbl, w, h):
    x1 = min(l[1] - l[3] / 2 for l in lbl) * w
    x2 = max(l[1] + l[3] / 2 for l in lbl) * w
    y1 = min(l[2] - l[4] / 2 for l in lbl) * h
    y2 = max(l[2] + l[4] / 2 for l in lbl) * h
    return x2 - x1, y2 - y1


def collect_train(folder, limit):
    warnings.filterwarnings("ignore")
    heights, ratios = [], []
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))[:limit]
    for p in files:
        a = np.atleast_1d(np.loadtxt(p))
        if a.size == 0 or a.size % 5 != 0:
            continue
        lbl = a.reshape(-1, 5)
        if lbl[0][0] == 10 or len(lbl) > 2:
            continue
        img = os.path.splitext(p)[0] + ".jpg"
        if not os.path.exists(img):
            continue
        w, h = Image.open(img).size            # reads only the header
        uw, uh = union_from_yolo(lbl, w, h)
        if uh <= 1:
            continue
        heights.append(uh)
        ratios.append(uh / (uh + MARGINS[1] + MARGINS[3]))
    return np.array(heights), np.array(ratios)


def collect_test(folder):
    heights, ratios = [], []
    for p in sorted(glob.glob(os.path.join(folder, "labels", "*.json"))):
        with open(p, encoding="utf8") as f:
            data = json.load(f)
        for b in data["data"]["jersey"]:
            n = len(b["jersey_number"])
            if n == 0 or n > 2:
                continue
            rects = [[float(v) for v in b["numbers"][j]["rect"]] for j in range(n)]
            y1 = min(r[1] for r in rects)
            y2 = max(r[1] + r[3] for r in rects)
            uh = y2 - y1
            if uh <= 1:
                continue
            heights.append(uh)
            ratios.append(uh / (uh + MARGINS[1] + MARGINS[3]))
    return np.array(heights), np.array(ratios)


def show(name, hs, rs):
    if len(hs) == 0:
        print(f"{name}: no samples found")
        return
    p = lambda a, q: np.percentile(a, q)
    print(f"\n{name}  ({len(hs)} numbers with 1-2 digits)")
    print(f"  digit height (px)        : 10%={p(hs,10):6.1f}  median={p(hs,50):6.1f}  90%={p(hs,90):6.1f}")
    print(f"  digits / crop height     : 10%={p(rs,10):6.2f}  median={p(rs,50):6.2f}  90%={p(rs,90):6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="training folder (.jpg + .txt side by side)")
    ap.add_argument("--test", required=True, help="test folder with images/ and labels/ (json)")
    ap.add_argument("--train_limit", type=int, default=5000, help="how many training labels to read")
    opt = ap.parse_args()

    th, tr = collect_train(opt.train, opt.train_limit)
    sh, sr = collect_test(opt.test)
    show("TRAINING (fixed margins 23/30/30/33 px)", th, tr)
    show("TEST     (same fixed margins)", sh, sr)

    if len(th):
        # margins as a fraction of the digit-union height, typical for training
        rel = [float(np.median(m / th)) for m in MARGINS]
        print("\nSuggested relative margins (fraction of digit height), from training medians:")
        print("  --rel_margins {:.2f} {:.2f} {:.2f} {:.2f}".format(*rel))
        print("  -> with these, a test crop has the same digits/crop ratio as a typical training crop")


if __name__ == "__main__":
    main()
