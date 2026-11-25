"""
Draws YOLO boxes from 1768_digits_train (and, optionally, its widened
version from widen_yolo_boxes.py) onto sample images, so you can visually
check the margin before using the widened dataset for anything.

Two modes:

1. Single folder (just look at one dataset's boxes):
   python visualize_1768_boxes.py \
       --data_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train \
       --out_dir annotated_original --n 15

2. Compare original vs widened side-by-side-by-overlay (draws BOTH boxes
   on the same image: original in cyan, widened in yellow/orange -- so you
   see exactly how much bigger the widened box is):
   python visualize_1768_boxes.py \
       --data_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train \
       --compare_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train_widened \
       --out_dir annotated_compare --n 15

Images are matched by filename across --data_dir and --compare_dir (same
basename, box widths/heights will differ but should be the SAME images).

Samples are chosen deterministically (sorted, evenly spaced) by default so
re-running gives the same images; pass --random --seed N for a random
sample instead.
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw, ImageFont


def read_yolo_boxes(txt_path):
    """Returns list of (class_str, cx, cy, w, h) normalized."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            try:
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes


def to_pixel_rect(cx, cy, w, h, W, H):
    left = (cx - w / 2) * W
    top = (cy - h / 2) * H
    right = (cx + w / 2) * W
    bottom = (cy + h / 2) * H
    return left, top, right, bottom


def find_image(folder, name):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = os.path.join(folder, name + ext)
        if os.path.exists(cand):
            return cand
    return None


def pick_samples(names, n, use_random, seed):
    if use_random:
        random.seed(seed)
        return random.sample(names, min(n, len(names)))
    if len(names) <= n:
        return names
    step = len(names) / n
    return [names[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder of <name>.jpg + <name>.txt (YOLO) -- the 'original' boxes, drawn in cyan")
    ap.add_argument("--compare_dir", default=None, help="Optional second folder (e.g. the _widened output) with the SAME filenames -- its boxes are drawn in orange on top of the same image, for comparison")
    ap.add_argument("--out_dir", required=True, help="Where annotated images are saved")
    ap.add_argument("--n", type=int, default=15, help="Number of sample images to annotate (default: 15)")
    ap.add_argument("--random", action="store_true", help="Pick samples randomly instead of evenly spaced")
    ap.add_argument("--seed", type=int, default=42, help="Random seed if --random is set")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"ERROR: no .txt files found in {args.data_dir}")
    names = [os.path.splitext(os.path.basename(p))[0] for p in label_files]

    sample_names = pick_samples(names, args.n, args.random, args.seed)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    n_done = 0
    n_missing_image = 0
    n_missing_compare = 0

    for name in sample_names:
        img_path = find_image(args.data_dir, name)
        if img_path is None:
            n_missing_image += 1
            print(f"{name}: no image found in {args.data_dir}, skipped")
            continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)

        orig_boxes = read_yolo_boxes(os.path.join(args.data_dir, name + ".txt"))
        for cls, cx, cy, w, h in orig_boxes:
            l, t, r, b = to_pixel_rect(cx, cy, w, h, W, H)
            draw.rectangle([l, t, r, b], outline=(0, 230, 230), width=2)  # cyan = original
            draw.text((l, max(0, t - 12)), f"orig:{cls}", fill=(0, 230, 230), font=font)

        if args.compare_dir:
            cmp_txt = os.path.join(args.compare_dir, name + ".txt")
            if not os.path.exists(cmp_txt):
                n_missing_compare += 1
            else:
                widened_boxes = read_yolo_boxes(cmp_txt)
                for cls, cx, cy, w, h in widened_boxes:
                    l, t, r, b = to_pixel_rect(cx, cy, w, h, W, H)
                    draw.rectangle([l, t, r, b], outline=(255, 140, 0), width=2)  # orange = widened
                    draw.text((l, b + 2), f"wide:{cls}", fill=(255, 140, 0), font=font)

        out_path = os.path.join(args.out_dir, name + ".jpg")
        im.save(out_path)
        n_done += 1

    print(f"Done. Annotated {n_done} images -> {args.out_dir}")
    if args.compare_dir:
        print("  cyan box   = original (--data_dir)")
        print("  orange box = widened  (--compare_dir)")
    if n_missing_image:
        print(f"  NOTE: {n_missing_image} samples had no image file in --data_dir.")
    if args.compare_dir and n_missing_compare:
        print(f"  NOTE: {n_missing_compare} samples had no matching label in --compare_dir.")


if __name__ == "__main__":
    main()
