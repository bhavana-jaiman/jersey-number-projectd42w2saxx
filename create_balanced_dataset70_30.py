#!/usr/bin/env python3

import os
import shutil
import random
import argparse
from pathlib import Path


def get_image_files(folder):
    """Get all image files from a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    return sorted([
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])


def is_synthetic(image_path):
    """
    Synthetic images have '_' in their filename.
    Example:
        ABC123.jpg       -> Real
        ABC123_0.jpg     -> Synthetic
        ABC123_1.jpg     -> Synthetic
    """
    return "_" in image_path.stem


def copy_image_and_label(image_path, label_path, destination):
    """Copy image and corresponding label to destination."""
    shutil.copy2(image_path, destination / image_path.name)

    if label_path.exists():
        shutil.copy2(label_path, destination / label_path.name)
    else:
        print(f"WARNING: Label not found for {image_path.name}")


def main():

    parser = argparse.ArgumentParser(
        description="Create balanced train/validation split using additional validation images from training."
    )

    parser.add_argument(
        "--training_dir",
        required=True,
        help="Path to original training dataset"
    )

    parser.add_argument(
        "--validation_dir",
        required=True,
        help="Path to original validation dataset"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.30,
        help="Final validation ratio. Example: 0.30 for 70:30"
    )

    parser.add_argument(
        "--real_count",
        type=int,
        required=True,
        help="Number of REAL training images to move to additional validation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    training_dir = Path(args.training_dir)
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)

    # ---------------------------------------------------------
    # Create output folders
    # ---------------------------------------------------------

    training_output = output_dir / "training"

    validation_existing_images = (
        output_dir / "validation" / "existing" / "images"
    )
    validation_existing_labels = (
        output_dir / "validation" / "existing" / "labels"
    )

    validation_real_images = (
        output_dir / "validation" / "real" / "images"
    )
    validation_real_labels = (
        output_dir / "validation" / "real" / "labels"
    )

    validation_synthetic_images = (
        output_dir / "validation" / "synthetic" / "images"
    )
    validation_synthetic_labels = (
        output_dir / "validation" / "synthetic" / "labels"
    )

    all_dirs = [
        training_output,

        validation_existing_images,
        validation_existing_labels,

        validation_real_images,
        validation_real_labels,

        validation_synthetic_images,
        validation_synthetic_labels,
    ]

    for directory in all_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Read training images
    # ---------------------------------------------------------

    training_images = get_image_files(training_dir)

    print("\n========================================")
    print("DATASET INFORMATION")
    print("========================================")

    print(f"Training images found : {len(training_images)}")

    # Separate real and synthetic
    real_training = []
    synthetic_training = []

    for image in training_images:

        if is_synthetic(image):
            synthetic_training.append(image)
        else:
            real_training.append(image)

    print(f"Real training images  : {len(real_training)}")
    print(f"Synthetic images      : {len(synthetic_training)}")

    # ---------------------------------------------------------
    # Read original validation
    # ---------------------------------------------------------

    original_validation_images_dir = validation_dir / "images"
    original_validation_labels_dir = validation_dir / "labels"

    original_validation_images = get_image_files(
        original_validation_images_dir
    )

    print(f"Original validation   : {len(original_validation_images)}")

    # ---------------------------------------------------------
    # Calculate required validation size
    # ---------------------------------------------------------

    total_dataset = (
        len(training_images) +
        len(original_validation_images)
    )

    target_validation = round(total_dataset * args.val_ratio)

    additional_validation = (
        target_validation -
        len(original_validation_images)
    )

    synthetic_count = additional_validation - args.real_count

    print("\n========================================")
    print("SPLIT CALCULATION")
    print("========================================")

    print(f"Total dataset                 : {total_dataset}")
    print(f"Target validation ratio       : {args.val_ratio * 100:.2f}%")
    print(f"Target final validation       : {target_validation}")
    print(f"Existing validation           : {len(original_validation_images)}")
    print(f"Additional validation needed  : {additional_validation}")

    print(f"Real added to validation      : {args.real_count}")
    print(f"Synthetic added to validation : {synthetic_count}")

    if args.real_count > len(real_training):
        raise ValueError(
            f"Requested {args.real_count} real images, "
            f"but only {len(real_training)} real images are available."
        )

    if synthetic_count > len(synthetic_training):
        raise ValueError(
            f"Requested {synthetic_count} synthetic images, "
            f"but only {len(synthetic_training)} synthetic images are available."
        )

    if additional_validation <= 0:
        raise ValueError(
            "Calculated additional validation size is <= 0."
        )

    # ---------------------------------------------------------
    # Randomly select validation images FROM TRAINING ONLY
    # ---------------------------------------------------------

    random.shuffle(real_training)
    random.shuffle(synthetic_training)

    selected_real = real_training[:args.real_count]

    selected_synthetic = synthetic_training[:synthetic_count]

    selected_validation = set(
        selected_real + selected_synthetic
    )

    # ---------------------------------------------------------
    # Copy remaining training images
    #
    # IMPORTANT:
    # Images and labels are directly inside training/
    # NO images/labels subfolders.
    # ---------------------------------------------------------

    print("\n========================================")
    print("CREATING TRAINING DATA")
    print("========================================")

    training_count = 0

    for image_path in training_images:

        if image_path in selected_validation:
            continue

        label_path = training_dir / f"{image_path.stem}.txt"

        copy_image_and_label(
            image_path,
            label_path,
            training_output
        )

        training_count += 1

    print(f"Training images copied: {training_count}")

    # ---------------------------------------------------------
    # Copy original validation WITHOUT modification
    # ---------------------------------------------------------

    print("\n========================================")
    print("COPYING EXISTING VALIDATION")
    print("========================================")

    existing_count = 0

    for image_path in original_validation_images:

        label_path = (
            original_validation_labels_dir /
            f"{image_path.stem}.txt"
        )

        destination_image = (
            validation_existing_images /
            image_path.name
        )

        destination_label = (
            validation_existing_labels /
            label_path.name
        )

        shutil.copy2(image_path, destination_image)

        if label_path.exists():
            shutil.copy2(label_path, destination_label)
        else:
            print(
                f"WARNING: Label not found for {image_path.name}"
            )

        existing_count += 1

    print(f"Existing validation copied: {existing_count}")

    # ---------------------------------------------------------
    # Copy newly selected REAL validation
    # ---------------------------------------------------------

    print("\n========================================")
    print("COPYING NEW REAL VALIDATION")
    print("========================================")

    real_count = 0

    for image_path in selected_real:

        label_path = training_dir / f"{image_path.stem}.txt"

        copy_image_and_label(
            image_path,
            label_path,
            validation_real_images
        )

        # Move label into labels folder
        source_label = validation_real_images / label_path.name
        destination_label = validation_real_labels / label_path.name

        if source_label.exists():
            shutil.move(source_label, destination_label)

        real_count += 1

    print(f"New real validation: {real_count}")

    # ---------------------------------------------------------
    # Copy newly selected SYNTHETIC validation
    # ---------------------------------------------------------

    print("\n========================================")
    print("COPYING NEW SYNTHETIC VALIDATION")
    print("========================================")

    synthetic_count_actual = 0

    for image_path in selected_synthetic:

        label_path = training_dir / f"{image_path.stem}.txt"

        copy_image_and_label(
            image_path,
            label_path,
            validation_synthetic_images
        )

        # Move label into labels folder
        source_label = (
            validation_synthetic_images /
            label_path.name
        )

        destination_label = (
            validation_synthetic_labels /
            label_path.name
        )

        if source_label.exists():
            shutil.move(source_label, destination_label)

        synthetic_count_actual += 1

    print(
        f"New synthetic validation: "
        f"{synthetic_count_actual}"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    final_training = training_count

    final_validation = (
        existing_count +
        real_count +
        synthetic_count_actual
    )

    print("\n========================================")
    print("FINAL DATASET")
    print("========================================")

    print(f"Final training   : {final_training}")
    print(f"Final validation : {final_validation}")
    print(f"Total            : {final_training + final_validation}")

    print("\n========================================")
    print("OUTPUT STRUCTURE")
    print("========================================")

    print(f"""
{output_dir}/
│
├── training/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
│
└── validation/
    ├── existing/
    │   ├── images/
    │   └── labels/
    │
    ├── real/
    │   ├── images/
    │   └── labels/
    │
    └── synthetic/
        ├── images/
        └── labels/
""")

    print("Dataset split completed successfully!")


if __name__ == "__main__":
    main()
