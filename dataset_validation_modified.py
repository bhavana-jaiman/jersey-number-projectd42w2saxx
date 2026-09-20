import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/path/to/your/datasets")

VALIDATION_REAL = BASE_DIR / "validation_real"
VALIDATION_SYNTHETIC = BASE_DIR / "validation_synthetic"
VALIDATION_EXISTING = BASE_DIR / "validation_existing"

# New merged validation folder
OUTPUT_DIR = BASE_DIR / "validation"


# ============================================================
# DATASETS TO MERGE
# ============================================================

DATASETS = {
    "real": VALIDATION_REAL,
    "synthetic": VALIDATION_SYNTHETIC,
    "existing": VALIDATION_EXISTING,
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MERGE FUNCTION
# ============================================================

def merge_dataset(dataset_name, dataset_path):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 60)

    image_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"

    if not image_dir.exists():
        print(f"ERROR: Image folder not found: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Label folder not found: {label_dir}")
        return

    images = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS
    ]

    print(f"Images found: {len(images)}")

    copied = 0
    missing_labels = 0
    renamed = 0

    for image_path in images:

        # Corresponding label
        label_path = label_dir / f"{image_path.stem}.txt"

        # ----------------------------------------------------
        # Check label
        # ----------------------------------------------------

        if not label_path.exists():

            print(
                f"WARNING: Label missing for "
                f"{image_path.name}"
            )

            missing_labels += 1
            continue

        # ----------------------------------------------------
        # Destination names
        # ----------------------------------------------------

        image_name = image_path.name
        label_name = label_path.name

        destination_image = OUTPUT_IMAGE_DIR / image_name
        destination_label = OUTPUT_LABEL_DIR / label_name

        # ----------------------------------------------------
        # Handle duplicate filenames
        # ----------------------------------------------------

        if destination_image.exists():

            new_stem = f"{dataset_name}_{image_path.stem}"

            destination_image = (
                OUTPUT_IMAGE_DIR /
                f"{new_stem}{image_path.suffix}"
            )

            destination_label = (
                OUTPUT_LABEL_DIR /
                f"{new_stem}.txt"
            )

            renamed += 1

        # ----------------------------------------------------
        # Copy image and label
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        copied += 1

    print(f"Copied: {copied}")
    print(f"Missing labels: {missing_labels}")
    print(f"Renamed due to duplicate: {renamed}")


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

for dataset_name, dataset_path in DATASETS.items():

    merge_dataset(
        dataset_name,
        dataset_path
    )


# ============================================================
# FINAL COUNTS
# ============================================================

final_images = [
    f for f in OUTPUT_IMAGE_DIR.iterdir()
    if f.is_file() and f.suffix in IMAGE_EXTENSIONS
]

final_labels = list(
    OUTPUT_LABEL_DIR.glob("*.txt")
)


print("\n" + "=" * 60)
print("MERGING COMPLETED")
print("=" * 60)

print(f"Final images : {len(final_images)}")
print(f"Final labels : {len(final_labels)}")

print("\nOutput:")
print(OUTPUT_DIR)
