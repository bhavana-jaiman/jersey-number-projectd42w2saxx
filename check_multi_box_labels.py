"""
Checks every label file in a folder (YOLO-format, one box per line) and
reports how many files have exactly 1 box line vs. 2+ box lines vs. 0 (empty).

This answers: "are there any label files with more than one bounding box?"
before deciding whether multi-box handling is even needed.

Usage:
    python check_multi_box_labels.py --labels_dir validation_whole_labels_whole
"""
import argparse
import glob
import os


def count_box_lines(path):
    n = 0
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--show_multi", type=int, default=10, help="Print up to this many multi-box filenames (default 10)")
    args = ap.parse_args()

    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"ERROR: no .txt files found in {args.labels_dir}")

    n_zero = 0
    n_one = 0
    n_multi = 0
    multi_examples = []

    for lf in label_files:
        n = count_box_lines(lf)
        if n == 0:
            n_zero += 1
        elif n == 1:
            n_one += 1
        else:
            n_multi += 1
            if len(multi_examples) < args.show_multi:
                multi_examples.append((os.path.basename(lf), n))

    total = len(label_files)
    print(f"Checked {total} label files in {args.labels_dir}")
    print(f"  0 boxes (empty):     {n_zero}")
    print(f"  1 box (normal):      {n_one}")
    print(f"  2+ boxes (multi):    {n_multi}")

    if n_multi:
        print(f"\nSample of multi-box files (up to {args.show_multi}):")
        for name, n in multi_examples:
            print(f"  {name}: {n} boxes")
    else:
        print("\nNo multi-box files found -- every label file has at most 1 box.")


if __name__ == "__main__":
    main()
