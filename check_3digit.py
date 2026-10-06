"""
Check for 3-digit (or longer) jersey numbers in a YOLO-format dataset.

Each .txt label line:  <digit_class> <x_center> <y_center> <width> <height>  (normalized 0-1)
Digits whose boxes are within DIST_THRESHOLD pixels of each other are grouped
into the same jersey number; farther apart = different jersey numbers.

Usage:
    python check_3digit.py --data ../Datasets/70_30_dataset/training
    python check_3digit.py --data ../Datasets/70_30_dataset/training --mode gap --out three_digit.csv
"""

import argparse
import csv
import math
import os
from collections import Counter

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(label_path):
    base = os.path.splitext(label_path)[0]
    for ext in IMG_EXTS:
        if os.path.exists(base + ext):
            return base + ext
    return None


def read_boxes(label_path, img_w, img_h):
    """Return list of (digit, x1, y1, x2, y2, cx, cy) in pixels."""
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
            cx, cy = xc * img_w, yc * img_h
            bw, bh = w * img_w, h * img_h
            boxes.append((cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, cx, cy))
    return boxes


def distance(a, b, mode):
    if mode == "center":
        # Euclidean distance between box centers
        return math.hypot(a[5] - b[5], a[6] - b[6])
    # "gap": shortest edge-to-edge distance between boxes (0 if overlapping)
    dx = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    dy = max(0.0, max(a[2], b[2]) - min(a[4], b[4]))
    return math.hypot(dx, dy)


def group_digits(boxes, threshold, mode):
    """Union-find clustering: boxes within threshold belong to the same jersey number."""
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if distance(boxes[i], boxes[j], mode) <= threshold:
                parent[root(i)] = root(j)

    groups = {}
    for i, b in enumerate(boxes):
        groups.setdefault(root(i), []).append(b)

    # Read each number left to right
    numbers = []
    for g in groups.values():
        g.sort(key=lambda b: b[5])
        numbers.append("".join(str(b[0]) for b in g))
    return numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with images and YOLO .txt labels")
    ap.add_argument("--threshold", type=float, default=25, help="Grouping distance in pixels")
    ap.add_argument("--mode", choices=["center", "gap"], default="center",
                    help="center = center-to-center distance, gap = edge-to-edge distance")
    ap.add_argument("--min-digits", type=int, default=3, help="Report numbers with at least this many digits")
    ap.add_argument("--out", default="three_digit_report.csv", help="CSV report path")
    args = ap.parse_args()

    label_files = sorted(f for f in os.listdir(args.data) if f.endswith(".txt"))
    length_counts = Counter()
    hits = []
    empty_labels = missing_images = 0

    for name in label_files:
        label_path = os.path.join(args.data, name)
        img_path = find_image(label_path)
        if img_path is None:
            missing_images += 1
            continue
        with Image.open(img_path) as im:  # reads header only, fast
            img_w, img_h = im.size

        boxes = read_boxes(label_path, img_w, img_h)
        if not boxes:
            empty_labels += 1  # empty jersey / no-number images
            continue

        for number in group_digits(boxes, args.threshold, args.mode):
            length_counts[len(number)] += 1
            if len(number) >= args.min_digits:
                hits.append((os.path.basename(img_path), number, img_w, img_h))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "jersey_number", "img_width", "img_height"])
        writer.writerows(hits)

    print(f"Label files scanned      : {len(label_files)}")
    print(f"Empty labels (no digits) : {empty_labels}")
    print(f"Labels without image     : {missing_images}")
    print(f"Grouping                 : {args.mode} distance <= {args.threshold}px")
    print("\nJersey numbers by digit count:")
    for n in sorted(length_counts):
        print(f"  {n}-digit : {length_counts[n]}")
    print(f"\n{args.min_digits}+ digit numbers found: {len(hits)} "
          f"in {len({h[0] for h in hits})} images")
    for img, num, *_ in hits[:20]:
        print(f"  {img}  ->  {num}")
    if len(hits) > 20:
        print(f"  ... and {len(hits) - 20} more")
    print(f"\nFull list saved to: {args.out}")


if __name__ == "__main__":
    main()
