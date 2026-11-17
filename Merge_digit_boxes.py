"""
Merge per-digit bounding boxes into ONE box per whole jersey number.

Background: label files currently have one line per digit, YOLO-normalized:
    <digit> <cx> <cy> <w> <h>
e.g. for a 2-digit number:
    0 0.136816 0.291819 0.089552 0.146520
    8 0.304726 0.291819 0.216418 0.146520

This script replaces that with ONE line per image:
    <merged_number> <cx> <cy> <w> <h>
where the box is the union of all digit boxes in that file, expanded by a
margin (default 8% of the merged box's own width/height on each side,
clipped to stay inside the image), and the digit order in <merged_number>
is determined by sorting boxes left-to-right by cx (not by line order in
the file, since that isn't guaranteed to be reading order).

Source layout handling (matches the two layouts already used in this
project's datasets):
  --mode train : source folder has images (.jpg/.jpeg/.png) and label
                 (.txt) files together, flat, same basename.
  --mode val   : source folder has images/ and labels/ subfolders.

Output layout mirrors the same convention at --dst:
  --mode train : <dst>/<basename>.jpg + <dst>/<basename>.txt (flat)
  --mode val   : <dst>/images/<basename>.jpg + <dst>/labels/<basename>.txt

Files are COPIED (images untouched, only the label file content changes).

Usage:
    python merge_digit_boxes.py --src <dataset_folder> --mode train \
        --dst <output_folder> --margin 0.08
    python merge_digit_boxes.py --src <dataset_folder> --mode val \
        --dst <output_folder> --margin 0.08
"""
import argparse
import shutil
from pathlib import Path


def find_image(img_stem_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_stem_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label_lines(text: str):
    """Returns list of (digit_str, cx, cy, w, h) for each non-empty line."""
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        digit = parts[0]
        cx, cy, w, h = (float(x) for x in parts[1:5])
        out.append((digit, cx, cy, w, h))
    return out


def merge_boxes(rows, margin_frac: float):
    """
    rows: list of (digit_str, cx, cy, w, h), normalized 0-1.
    Returns (merged_number_str, mcx, mcy, mw, mh) with margin applied and
    clipped to [0, 1].
    """
    # sort left-to-right by cx so the digit string reads correctly
    rows_sorted = sorted(rows, key=lambda r: r[1])
    number_str = "".join(r[0] for r in rows_sorted)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for _, cx, cy, w, h in rows_sorted:
        xmins.append(cx - w / 2)
        xmaxs.append(cx + w / 2)
        ymins.append(cy - h / 2)
        ymaxs.append(cy + h / 2)
    xmin, xmax = min(xmins), max(xmaxs)
    ymin, ymax = min(ymins), max(ymaxs)

    mw = xmax - xmin
    mh = ymax - ymin
    # margin as a fraction of the merged box's own size, each side
    xmin -= mw * margin_frac
    xmax += mw * margin_frac
    ymin -= mh * margin_frac
    ymax += mh * margin_frac

    # clip to image bounds
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(1.0, xmax)
    ymax = min(1.0, ymax)

    mcx = (xmin + xmax) / 2
    mcy = (ymin + ymax) / 2
    mw = xmax - xmin
    mh = ymax - ymin
    return number_str, mcx, mcy, mw, mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", required=True, choices=["train", "val"])
    ap.add_argument("--margin", type=float, default=0.08,
                     help="fraction of merged box's own w/h added as margin on each side")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()

    if args.mode == "train":
        img_src_dir = src
        label_src_dir = src
        img_dst_dir = dst
        label_dst_dir = dst
    else:
        img_src_dir = src / "images"
        label_src_dir = src / "labels"
        img_dst_dir = dst / "images"
        label_dst_dir = dst / "labels"

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skipped_no_image = n_skipped_empty = 0

    for label_path in sorted(label_src_dir.glob("*.txt")):
        stem = label_path.stem
        img_path = find_image(img_src_dir, stem)
        if img_path is None:
            n_skipped_no_image += 1
            continue

        rows = parse_label_lines(label_path.read_text(encoding="utf8"))
        if not rows:
            n_skipped_empty += 1
            continue

        number_str, mcx, mcy, mw, mh = merge_boxes(rows, args.margin)

        # copy image
        shutil.copy2(img_path, img_dst_dir / img_path.name)
        # write merged label
        out_label = label_dst_dir / f"{stem}.txt"
        out_label.write_text(f"{number_str} {mcx:.6f} {mcy:.6f} {mw:.6f} {mh:.6f}\n", encoding="utf8")
        n_ok += 1

    print(f"Merged {n_ok} samples.")
    if n_skipped_no_image:
        print(f"Skipped {n_skipped_no_image} label files with no matching image.")
    if n_skipped_empty:
        print(f"Skipped {n_skipped_empty} empty label files.")
    print(f"Output at: {dst}")


if __name__ == "__main__":
    main()
