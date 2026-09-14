import os
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Dataset 1
DATASET_1 = Path("datasets/validation_dataset_Ying")

# Dataset 2
DATASET_2 = Path("demo3/complex2D_v14_validation")

# NEW output directory
# Neither original dataset will be modified.
OUTPUT_DIR = Path("datasets/merged_validation_dataset")


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FIND IMAGES AND LABELS
# ============================================================

def find_images_and_labels(dataset_dir):
    """
    Finds images and labels regardless of whether the dataset is:

        dataset/
            images/
            labels/

    OR:

        dataset/
            image.jpg
            image.txt

    OR contains images/labels somewhere deeper.
    """

    image_files = []
    label_files = {}

    # Search recursively
    for file in dataset_dir.rglob("*"):

        if not file.is_file():
            continue

        suffix = file.suffix.lower()

        # Image
        if suffix in IMAGE_EXTENSIONS:
            image_files.append(file)

        # Label
        elif suffix == ".txt":
            label_files[file.stem] = file

    return image_files, label_files


# ============================================================
# COPY DATASET
# ============================================================

def copy_dataset(dataset_dir, prefix, output_images, output_labels):

    print()
    print("=" * 60)
    print(f"PROCESSING: {dataset_dir}")
    print("=" * 60)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset does not exist: {dataset_dir}")
        return 0, 0

    images, labels = find_images_and_labels(dataset_dir)

    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    copied_images = 0
    copied_labels = 0

    for index, image_path in enumerate(images, start=1):

        # ----------------------------------------------------
        # Create unique filename
        # ----------------------------------------------------

        new_stem = f"{prefix}_{index:06d}"

        new_image_name = new_stem + image_path.suffix.lower()
        new_image_path = output_images / new_image_name

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(image_path, new_image_path)
        copied_images += 1

        # ----------------------------------------------------
        # Find corresponding label
        # ----------------------------------------------------

        original_stem = image_path.stem

        label_path = labels.get(original_stem)

        if label_path is not None:

            new_label_path = output_labels / f"{new_stem}.txt"

            shutil.copy2(label_path, new_label_path)

            copied_labels += 1

        else:
            print(
                f"WARNING: No label found for image: "
                f"{image_path.name}"
            )

    print()
    print(f"Copied images : {copied_images}")
    print(f"Copied labels : {copied_labels}")

    return copied_images, copied_labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DATASET MERGING")
    print("=" * 60)

    print(f"Dataset 1 : {DATASET_1}")
    print(f"Dataset 2 : {DATASET_2}")
    print(f"Output    : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if OUTPUT_DIR.resolve() in [
        DATASET_1.resolve(),
        DATASET_2.resolve(),
    ]:
        raise RuntimeError(
            "ERROR: Output directory cannot be one of "
            "the original datasets!"
        )

    # --------------------------------------------------------
    # Create NEW output directories
    # --------------------------------------------------------

    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Copy Dataset 1
    # --------------------------------------------------------

    images_1, labels_1 = copy_dataset(
        DATASET_1,
        "ying",
        output_images,
        output_labels,
    )

    # --------------------------------------------------------
    # Copy Dataset 2
    # --------------------------------------------------------

    images_2, labels_2 = copy_dataset(
        DATASET_2,
        "complex2d",
        output_images,
        output_labels,
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print(f"Dataset 1 images : {images_1}")
    print(f"Dataset 1 labels : {labels_1}")

    print(f"Dataset 2 images : {images_2}")
    print(f"Dataset 2 labels : {labels_2}")

    print()
    print(f"TOTAL IMAGES     : {images_1 + images_2}")
    print(f"TOTAL LABELS     : {labels_1 + labels_2}")

    print()
    print("NEW DATASET:")
    print(OUTPUT_DIR)

    print()
    print("Original datasets were NOT modified.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
