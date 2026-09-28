#!/usr/bin/env python3

import shutil
import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    """Return all image files from a folder."""
    return sorted([
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def copy_dataset(source_dir, output_images, output_labels, dataset_name):
    """
    Copy images and corresponding labels from one validation dataset.
    """

    source_images = source_dir / "images"
    source_labels = source_dir / "labels"

    if not source_images.exists():
        print(f"WARNING: Images folder not found: {source_images}")
        return 0, 0

    if not source_labels.exists():
        print(f"WARNING: Labels folder not found: {source_labels}")
        return 0, 0

    images = get_images(source_images)

    copied_images = 0
    missing_labels = 0

    print("\n" + "=" * 60)
    print(f"PROCESSING {dataset_name.upper()}")
    print("=" * 60)

    print(f"Images found: {len(images)}")

    for image_path in images:

        label_path = source_labels / f"{image_path.stem}.txt"

        # Destination paths
        destination_image = output_images / image_path.name
        destination_label = output_labels / label_path.name

        # Check label
        if not label_path.exists():
            print(f"WARNING: Missing label for {image_path.name}")
            missing_labels += 1
            continue

        # Avoid overwriting duplicate filenames
        if destination_image.exists():
            print(
                f"WARNING: Duplicate image filename found: "
                f"{image_path.name}"
            )

            # Add dataset name to filename
            new_name = f"{dataset_name}_{image_path.name}"

            destination_image = output_images / new_name
            destination_label = (
                output_labels / f"{dataset_name}_{label_path.name}"
            )

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

        copied_images += 1

    print(f"Images copied : {copied_images}")
    print(f"Missing labels: {missing_labels}")

    return copied_images, missing_labels


def main():

    parser = argparse.ArgumentParser(
        description="Merge existing, real and synthetic validation datasets."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to validation_crop directory"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path for final validation dataset"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # ---------------------------------------------------------
    # Create final images and labels folders
    # ---------------------------------------------------------

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Process existing validation
    # ---------------------------------------------------------

    existing_images, existing_missing = copy_dataset(
        input_dir / "existing",
        output_images,
        output_labels,
        "existing"
    )

    # ---------------------------------------------------------
    # Process new real validation
    # ---------------------------------------------------------

    real_images, real_missing = copy_dataset(
        input_dir / "real",
        output_images,
        output_labels,
        "real"
    )

    # ---------------------------------------------------------
    # Process new synthetic validation
    # ---------------------------------------------------------

    synthetic_images, synthetic_missing = copy_dataset(
        input_dir / "synthetic",
        output_images,
        output_labels,
        "synthetic"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    total_images = (
        existing_images +
        real_images +
        synthetic_images
    )

    total_missing = (
        existing_missing +
        real_missing +
        synthetic_missing
    )

    print("\n")
    print("=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)

    print(f"Existing validation : {existing_images}")
    print(f"New real validation : {real_images}")
    print(f"New synthetic      : {synthetic_images}")
    print("-" * 60)
    print(f"FINAL VALIDATION   : {total_images}")
    print(f"Missing labels     : {total_missing}")

    print("\nOutput:")
    print(output_dir)

    print("\nFinal structure:")
    print(f"""
{output_dir}/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
""")

    if total_missing == 0:
        print("✓ Every image has a corresponding label.")
    else:
        print(
            f"⚠ {total_missing} images were skipped "
            f"because their labels were missing."
        )


if __name__ == "__main__":
    main()
