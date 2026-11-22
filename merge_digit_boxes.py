#!/usr/bin/env python3
"""Merge per-digit YOLO boxes into ONE box per number (whole-number box).

Usage:
  python merge_digit_boxes.py --labels <labels_dir> --out <new_labels_dir> [options]

Options:
  --new-class N     class id written for the merged box (default 0)
  --pad F           extra padding as a fraction of merged box w/h (default 0.0, e.g. 0.05)
  --gap F           if >0, split into separate numbers when the horizontal gap between
                    neighbouring digits is > F * median digit height (default 0 = merge ALL
                    boxes in a file into one)
  --dry-run         only print stats, write nothing

Input : YOLO txt  ->  class xc yc w h   (normalized 0-1). Polygon lines are also accepted.
Output: same file names in --out. Original labels are never modified.
Empty label files are copied as empty files (background images stay background).
"""
import os, sys, argparse, statistics
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--new-class", type=int, default=0)
ap.add_argument("--pad", type=float, default=0.0)
ap.add_argument("--gap", type=float, default=0.0)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

src, dst = os.path.abspath(a.labels), os.path.abspath(a.out)
if src == dst:
    sys.exit("--out must be different from --labels (originals are never overwritten)")

def read_boxes(path):
    """return list of (x1,y1,x2,y2) in normalized coords"""
    boxes = []
    for line in open(path, errors="ignore"):
        t = line.split()
        if len(t) < 5: continue
        v = [float(x) for x in t[1:]]
        if len(v) in (4, 5):                       # xc yc w h (+conf)
            xc, yc, w, h = v[:4]
            boxes.append((xc - w/2, yc - h/2, xc + w/2, yc + h/2))
        elif len(v) >= 6 and len(v) % 2 == 0:      # polygon -> its bbox
            xs, ys = v[0::2], v[1::2]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def groups(boxes):
    if a.gap <= 0 or len(boxes) < 2: return [boxes]
    boxes = sorted(boxes, key=lambda b: b[0])
    mh = statistics.median(b[3] - b[1] for b in boxes)
    out, cur, right = [], [boxes[0]], boxes[0][2]
    for b in boxes[1:]:
        if b[0] - right > a.gap * mh: out.append(cur); cur = [b]
        else: cur.append(b)
        right = max(right, b[2])
    out.append(cur); return out

def merge(g):
    x1 = min(b[0] for b in g); y1 = min(b[1] for b in g)
    x2 = max(b[2] for b in g); y2 = max(b[3] for b in g)
    w, h = x2 - x1, y2 - y1
    x1 -= a.pad * w; x2 += a.pad * w; y1 -= a.pad * h; y2 += a.pad * h
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    return (a.new_class, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)

n_files = n_empty = 0
hist_in, hist_out = Counter(), Counter()
for d, _, files in os.walk(src):
    for f in sorted(files):
        if not f.endswith(".txt") or f.lower() in ("classes.txt", "labels.txt"): continue
        p = os.path.join(d, f)
        boxes = read_boxes(p)
        n_files += 1; hist_in[len(boxes)] += 1
        res = [merge(g) for g in groups(boxes)] if boxes else []
        if not boxes: n_empty += 1
        hist_out[len(res)] += 1
        if not a.dry_run:
            o = os.path.join(dst, os.path.relpath(p, src))
            os.makedirs(os.path.dirname(o), exist_ok=True)
            with open(o, "w") as fh:
                for c, xc, yc, w, h in res:
                    fh.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print(f"label files: {n_files} (empty: {n_empty})")
print("boxes per file BEFORE:", dict(sorted(hist_in.items())))
print("boxes per file AFTER :", dict(sorted(hist_out.items())))
print("dry run, nothing written" if a.dry_run else f"written to: {dst}")
