"""
measure_client_bodies.py
Measures how the jersey number sits inside the client's body boxes (test set),
and how big the number is inside the training images.

Run from the workspace:
    python measure_client_bodies.py \
        --test ../workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset \
        --train /home/eng_bhavana/workspace_bhavana/Datasets/70_30_dataset/training
"""
import argparse
import glob
import json
import os

import numpy as np


def pct(name, v):
    v = np.array(v, dtype=float)
    if len(v) == 0:
        print(f"{name:32s} no data")
        return
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    print(f"{name:32s} p5={p5:.2f}  median={p50:.2f}  p95={p95:.2f}")


def measure_test(root):
    files = glob.glob(os.path.join(root, "labels", "*.json"))
    print(f"\n=== CLIENT TEST SET: {len(files)} json files ===")
    r_h, asp, cx, cy, h96 = [], [], [], [], []
    two = rev = n0 = n1 = n3 = 0
    for f in files:
        d = json.load(open(f))
        for j in d["data"]["jersey"]:
            bx, by, bw, bh = j["body"]["rect"]
            nums = j.get("numbers", [])
            if len(nums) == 0:
                n0 += 1
                continue
            if len(nums) == 1:
                n1 += 1
            elif len(nums) == 2:
                two += 1
                if nums[0]["rect"][0] > nums[1]["rect"][0]:
                    rev += 1
            else:
                n3 += 1
            if bw <= 0 or bh <= 0:
                continue
            rs = [n["rect"] for n in nums]
            x0 = min(r[0] for r in rs)
            y0 = min(r[1] for r in rs)
            x1 = max(r[0] + r[2] for r in rs)
            y1 = max(r[1] + r[3] for r in rs)
            r_h.append((y1 - y0) / bh)
            h96.append((y1 - y0) / bh * 96)
            asp.append(bw / bh)
            cx.append(((x0 + x1) / 2 - bx) / bw)
            cy.append(((y0 + y1) / 2 - by) / bh)
    print(f"bodies: no number={n0}  1 digit={n1}  2 digits={two}  3+ digits={n3}")
    print(f"2-digit bodies stored RIGHT-TO-LEFT in json: {rev} of {two}")
    pct("digit h / body h", r_h)
    pct("digit h in 96x96 (px)", h96)
    pct("body w / body h", asp)
    pct("number centre x in body (0-1)", cx)
    pct("number centre y in body (0-1)", cy)


def measure_train(root, max_files=5000):
    from PIL import Image
    labels = sorted(glob.glob(os.path.join(root, "*.txt")))[:max_files]
    print(f"\n=== TRAINING SET: {len(labels)} label files checked ===")
    r_h, r_w, img_asp, sizes = [], [], [], []
    for p in labels:
        img = None
        for ext in (".jpg", ".png", ".jpeg"):
            if os.path.exists(p[:-4] + ext):
                img = p[:-4] + ext
                break
        if img is None:
            continue
        try:
            lab = np.loadtxt(p, ndmin=2)
        except Exception:
            continue
        if lab.size == 0 or lab.shape[1] != 5 or int(lab[0][0]) == 10:
            continue
        W, H = Image.open(img).size
        y0 = min(l[2] - l[4] / 2 for l in lab)
        y1 = max(l[2] + l[4] / 2 for l in lab)
        x0 = min(l[1] - l[3] / 2 for l in lab)
        x1 = max(l[1] + l[3] / 2 for l in lab)
        r_h.append(y1 - y0)          # number height / image height
        r_w.append(x1 - x0)          # number width  / image width
        img_asp.append(W / H)
        sizes.append(H)
    pct("number h / IMAGE h", r_h)
    pct("number w / IMAGE w", r_w)
    pct("image w / image h", img_asp)
    pct("image height (px)", sizes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="client test folder (has images/ and labels/*.json)")
    ap.add_argument("--train", default=None, help="training folder (jpg + YOLO txt)")
    a = ap.parse_args()
    measure_test(a.test)
    if a.train:
        measure_train(a.train)
