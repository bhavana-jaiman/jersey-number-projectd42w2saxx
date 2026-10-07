"""
Create a filtered copy of a jersey-number dataset, excluding images whose
label contains a 3-digit jersey number.

Expected input structure:
    input_dataset/
        images/   <name>.jpg (or .jpeg/.png/...)
        labels/   <name>.json

Output structure (same layout):
    output_dataset/
        images/
        labels/

Usage:
    python filter_3digit_jerseys.py --input JerseyNumber_Validation_Dataset --output JerseyNumber_Filtered
"""

import argparse
import json
import shutil
from pathlib import Path

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]


def num_digits(jersey_number):
    """Return digit count for jersey_number given as list [4, 8, 5], string '485' or int 485."""
    if isinstance(jersey_number, (list, tuple)):
        return len(jersey_number)
    return len(str(jersey_number).strip())


def has_3_digit_jersey(label):
    """True if any jersey in the label has a 3-digit jersey_number."""
    jerseys = label.get("data", {}).get("jersey", [])
    for jersey in jerseys:
        jn = jersey.get("jersey_number")
        if jn is not None and num_digits(jn) == 3:
            return True
    return False


def find_image(images_dir, stem):
    """Find the image with the same base name as the label, trying common extensions."""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Exclude images with 3-digit jersey numbers.")
    parser.add_argument("--input", required=True, help="Input dataset folder (contains images/ and labels/)")
    parser.add_argument("--output", required=True, help="Output dataset folder to create")
    args = parser.parse_args()

    in_images = Path(args.input) / "images"
    in_labels = Path(args.input) / "labels"
    out_images = Path(args.output) / "images"
    out_labels = Path(args.output) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept, excluded, missing_image, bad_label = 0, 0, 0, 0

    for label_path in sorted(in_labels.glob("*.json")):
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Could not read {label_path.name}: {e}")
            bad_label += 1
            continue

        image_path = find_image(in_images, label_path.stem)
        if image_path is None:
            print(f"[WARN] No image found for {label_path.name}")
            missing_image += 1
            continue

        if has_3_digit_jersey(label):
            excluded += 1
            continue

        shutil.copy2(image_path, out_images / image_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)
        kept += 1

    print("\nDone.")
    print(f"  Kept (copied):          {kept}")
    print(f"  Excluded (3-digit):     {excluded}")
    print(f"  Skipped, missing image: {missing_image}")
    print(f"  Skipped, bad label:     {bad_label}")
    print(f"  Output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
