"""
Builds a new validation_body folder by combining:
  - images from   final_dataset/validation_body_cropped_whole/images/
  - labels from   validation_body_cropped_whole_9603/labels/

Only MATCHED pairs (same basename present in both images/ and labels/) are
copied. An image with no matching label, or a label with no matching image,
is skipped and reported -- not copied -- so the output folder never has an
orphaned image or label.

Both sources are read-only; nothing in them is modified. The new folder is
created fresh at --out_dir (images/ and labels/ subfolders).

Usage:
    python build_validation_body.py \
        --images_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/final_dataset/validation_body_cropped_whole/images \
        --labels_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_body_cropped_whole_9603/labels \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_body

Dry run first (just counts/reports matches, copies nothing):
    python build_validation_body.py --images_dir ... --labels_dir ... --out_dir ... --dry_run

After copying (or in --dry_run), also prints a breakdown of how many label
files contain each exact label content (e.g. how many files say "2 3", how
many say "7", how many are empty), sorted by count descending.
"""
import argparse
import glob
import os
import shutil
from collections import Counter


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def read_label_content(path):
    """Returns the label file's content as a single normalized string,
    e.g. '2 3', '7', or '' for an empty/blank label file."""
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        text = f.read().strip()
    # collapse any internal whitespace/newlines to single spaces
    return " ".join(text.split())


def print_label_breakdown(label_paths, title):
    counts = Counter()
    for p in label_paths:
        content = read_label_content(p)
        key = content if content else "(empty)"
        counts[key] += 1

    print(f"\n{title} ({len(label_paths)} label files, {len(counts)} distinct labels):")
    for label, n in counts.most_common():
        print(f"  {label!r}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Source images folder (read-only)")
    ap.add_argument("--labels_dir", required=True, help="Source labels folder (read-only)")
    ap.add_argument("--out_dir", required=True, help="New folder to create; images/ and labels/ subfolders go here")
    ap.add_argument("--dry_run", action="store_true", help="Only report counts; copy nothing")
    args = ap.parse_args()

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(glob.glob(os.path.join(args.images_dir, f"*{ext}")))
    # map basename -> actual image path (keep first match if duplicate extensions)
    image_map = {}
    for p in image_files:
        base = os.path.splitext(os.path.basename(p))[0]
        image_map.setdefault(base, p)

    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    label_map = {os.path.splitext(os.path.basename(p))[0]: p for p in label_files}

    if not image_map:
        raise SystemExit(f"ERROR: no images found in {args.images_dir}")
    if not label_map:
        raise SystemExit(f"ERROR: no labels found in {args.labels_dir}")

    image_basenames = set(image_map.keys())
    label_basenames = set(label_map.keys())

    matched = sorted(image_basenames & label_basenames)
    images_without_label = sorted(image_basenames - label_basenames)
    labels_without_image = sorted(label_basenames - image_basenames)

    print(f"Images found: {len(image_basenames)}  (in {args.images_dir})")
    print(f"Labels found: {len(label_basenames)}  (in {args.labels_dir})")
    print(f"Matched pairs: {len(matched)}")
    print(f"Images with NO matching label: {len(images_without_label)}")
    if images_without_label:
        print("  sample:", images_without_label[:10])
    print(f"Labels with NO matching image: {len(labels_without_image)}")
    if labels_without_image:
        print("  sample:", labels_without_image[:10])

    if args.dry_run:
        matched_label_paths = [label_map[base] for base in matched]
        print_label_breakdown(matched_label_paths, "Label breakdown (matched pairs that WOULD be copied)")
        print("\n--dry_run: nothing copied. Re-run without --dry_run to actually build the folder.")
        return

    out_images = os.path.join(args.out_dir, "images")
    out_labels = os.path.join(args.out_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    n_copied = 0
    for base in matched:
        src_img = image_map[base]
        src_lbl = label_map[base]
        img_ext = os.path.splitext(src_img)[1]
        shutil.copy2(src_img, os.path.join(out_images, base + img_ext))
        shutil.copy2(src_lbl, os.path.join(out_labels, base + ".txt"))
        n_copied += 1

    print(f"\nCopied {n_copied} matched image+label pairs to:")
    print(f"  {out_images}")
    print(f"  {out_labels}")

    final_images = len(glob.glob(os.path.join(out_images, "*")))
    final_label_paths = sorted(glob.glob(os.path.join(out_labels, "*.txt")))
    final_labels = len(final_label_paths)
    print(f"\nFinal count in {args.out_dir}:")
    print(f"  images: {final_images}")
    print(f"  labels: {final_labels}")

    print_label_breakdown(final_label_paths, "Label breakdown (copied files in validation_body/labels)")


if __name__ == "__main__":
    main()
