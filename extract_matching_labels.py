"""
Pulls out only the label files from a big label pool (e.g. upper_labels/,
which holds labels for BOTH training and validation images combined, before
the 70:30 split) that correspond to images already present in your actual
validation image folder (70_30_dataset_v3/validation/images/).

This does NOT touch --labels_pool_dir or --images_dir -- both are read-only.
It only WRITES new copies into --out_dir.

For each image file in --images_dir, it looks for a same-basename .txt in
--labels_pool_dir and copies it (unchanged) into --out_dir. Images with no
matching label, and label files in the pool that don't correspond to any
validation image, are both reported (not errors -- just informational).

Usage (dry run -- just counts/reports, copies nothing):
    python extract_matching_labels.py \
        --images_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation/images \
        --labels_pool_dir ~/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying_V3/upper_labels \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_whole_labels_from_Ying \
        --dry_run

Then for real (actually copies matching files):
    python extract_matching_labels.py \
        --images_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation/images \
        --labels_pool_dir ~/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying_V3/upper_labels \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_whole_labels_from_Ying
"""
import argparse
import glob
import os
import shutil


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Your actual validation images folder, e.g. 70_30_dataset_v3/validation/images (read-only)")
    ap.add_argument("--labels_pool_dir", required=True, help="The big label pool to pull from, e.g. training_dataset_Ying_V3/upper_labels (read-only)")
    ap.add_argument("--out_dir", required=True, help="Where matching label files are copied to (new folder)")
    ap.add_argument("--dry_run", action="store_true", help="Only report counts; copy nothing")
    args = ap.parse_args()

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(glob.glob(os.path.join(args.images_dir, f"*{ext}")))
    image_basenames = sorted(set(os.path.splitext(os.path.basename(p))[0] for p in image_files))

    if not image_basenames:
        raise SystemExit(f"ERROR: no images found in {args.images_dir}")

    print(f"Found {len(image_basenames)} images in {args.images_dir}")
    print(f"Looking for matching labels in {args.labels_pool_dir}")
    print()

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    n_matched = 0
    n_missing = 0
    missing_sample = []

    for name in image_basenames:
        label_path = os.path.join(args.labels_pool_dir, name + ".txt")
        if os.path.exists(label_path):
            n_matched += 1
            if not args.dry_run:
                shutil.copy2(label_path, os.path.join(args.out_dir, name + ".txt"))
        else:
            n_missing += 1
            if len(missing_sample) < 10:
                missing_sample.append(name)

    print(f"Matched:  {n_matched} / {len(image_basenames)} validation images had a label in the pool")
    print(f"Missing:  {n_missing} validation images had NO matching label in the pool")
    if missing_sample:
        print("Sample of missing (up to 10):")
        for n in missing_sample:
            print(f"  {n}")

    if args.dry_run:
        print("\n--dry_run: nothing was copied. Re-run without --dry_run to copy the matched label files.")
    else:
        print(f"\nCopied {n_matched} label files to: {args.out_dir}")


if __name__ == "__main__":
    main()
