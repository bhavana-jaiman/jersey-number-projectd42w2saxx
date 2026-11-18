"""
Merge the three 3-digit flat-label sources into a single combined dataset
under 3_digit_validation/:

  syn3_dataset/                                  (593 images/labels)
  syn3_15_10_2026_digits/                        (891 images/labels)
  syn3_digit_2_11_2026/syn3_digit_2_11_2026/     (912 images/labels, nested)

Label format for all three: a plain text file containing just the 3-digit
ground truth (e.g. "781"), no bounding box / JSON.

Since filenames may repeat ACROSS sources (confirmed not overlapping in
content per user check, but to be safe this script still checks and would
warn, not silently overwrite), each source's files are copied with a
source-prefix added to the filename stem so nothing can collide:

  syn3_dataset/images/0001.jpg          -> 3_digit_validation/images/sA_0001.jpg
  syn3_15_10_2026_digits/images/0001.jpg -> 3_digit_validation/images/sB_0001.jpg
  syn3_digit_2_11_2026/.../0001.jpg      -> 3_digit_validation/images/sC_0001.jpg

(and same prefix applied to the matching label file, so image/label pairs
stay linked by stem).

Usage:
    python merge_3digit_validation.py
    # or override the base dir:
    python merge_3digit_validation.py --base ~/workspace_bhavana/Datasets/70_30_dataset_v3
"""
import argparse
import os
import shutil


SOURCES = [
    ("sA", "syn3_dataset"),
    ("sB", "syn3_15_10_2026_digits"),
    ("sC", "syn3_digit_2_11_2026/syn3_digit_2_11_2026"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser(
        "~/workspace_bhavana/Datasets/70_30_dataset_v3"))
    ap.add_argument("--dest_name", default="3_digit_validation")
    args = ap.parse_args()

    dest = os.path.join(args.base, args.dest_name)
    dest_images = os.path.join(dest, "images")
    dest_labels = os.path.join(dest, "labels")
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    total_copied = 0
    total_skipped_no_label = 0

    for prefix, rel_src in SOURCES:
        src = os.path.join(args.base, rel_src)
        src_images = os.path.join(src, "images")
        src_labels = os.path.join(src, "labels")

        if not os.path.isdir(src_images):
            print(f"[SKIP] {src_images} not found")
            continue

        image_files = sorted(os.listdir(src_images))
        n_copied = 0
        n_missing_label = 0

        for img_name in image_files:
            stem, ext = os.path.splitext(img_name)
            src_img_path = os.path.join(src_images, img_name)
            if not os.path.isfile(src_img_path):
                continue

            # find matching label file (try same stem, any extension)
            label_path = None
            for lext in (".txt", ".json", ""):
                candidate = os.path.join(src_labels, stem + lext)
                if os.path.isfile(candidate):
                    label_path = candidate
                    break
            if label_path is None:
                # fallback: label dir might use exact same filename as image
                candidate = os.path.join(src_labels, img_name)
                if os.path.isfile(candidate):
                    label_path = candidate

            if label_path is None:
                n_missing_label += 1
                continue

            label_ext = os.path.splitext(label_path)[1]
            new_stem = f"{prefix}_{stem}"
            dst_img_path = os.path.join(dest_images, new_stem + ext)
            dst_label_path = os.path.join(dest_labels, new_stem + label_ext)

            shutil.copy2(src_img_path, dst_img_path)
            shutil.copy2(label_path, dst_label_path)
            n_copied += 1

        print(f"[{prefix}] {rel_src}: copied {n_copied} pairs, "
              f"{n_missing_label} images with no matching label")
        total_copied += n_copied
        total_skipped_no_label += n_missing_label

    print()
    print(f"Done. Total pairs copied into {dest}: {total_copied}")
    if total_skipped_no_label:
        print(f"WARNING: {total_skipped_no_label} images had no matching label and were skipped.")

    final_images = len(os.listdir(dest_images))
    final_labels = len(os.listdir(dest_labels))
    print(f"\nFinal check -- images: {final_images}  labels: {final_labels}")
    if final_images != final_labels:
        print("WARNING: image/label counts do not match!")


if __name__ == "__main__":
    main()
