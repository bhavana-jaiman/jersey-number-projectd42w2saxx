import random
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

REAL_DIR = Path("/path/to/real")
SYNTHETIC_DIR = Path("/path/to/synthetic")

OUTPUT_DIR = Path("/path/to/output_dataset")

VAL_COUNT = 2000
RANDOM_SEED = 42


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".JPG", ".JPEG", ".PNG", ".BMP"
}


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS
    ]


# ============================================================
# COPY IMAGE + LABEL
# ============================================================

def copy_pair(image_path, label_path, destination):

    shutil.copy2(
        image_path,
        destination / image_path.name
    )

    shutil.copy2(
        label_path,
        destination / label_path.name
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    directories = [

        # Real validation
        OUTPUT_DIR / "validation_real" / "images",
        OUTPUT_DIR / "validation_real" / "labels",

        # Synthetic validation
        OUTPUT_DIR / "validation_synthetic" / "images",
        OUTPUT_DIR / "validation_synthetic" / "labels",

        # Training - IMAGE + LABEL IN SAME FOLDER
        OUTPUT_DIR / "training",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS REAL DATASET
# ============================================================

def process_real_dataset():

    print("\n" + "=" * 60)
    print("PROCESSING REAL DATASET")
    print("=" * 60)

    images = get_images(REAL_DIR)

    print(f"Total real images: {len(images)}")

    if len(images) < VAL_COUNT:

        raise ValueError(
            f"Real dataset has only {len(images)} images."
        )

    random.shuffle(images)

    val_images = images[:VAL_COUNT]
    train_images = images[VAL_COUNT:]

    print(f"Real validation: {len(val_images)}")
    print(f"Real training:   {len(train_images)}")

    # --------------------------------------------------------
    # REAL VALIDATION
    # --------------------------------------------------------

    val_image_dir = OUTPUT_DIR / "validation_real" / "images"
    val_label_dir = OUTPUT_DIR / "validation_real" / "labels"

    for image_path in val_images:

        label_path = REAL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():

            print(
                f"WARNING: Label missing for {image_path.name}"
            )

            continue

        shutil.copy2(
            image_path,
            val_image_dir / image_path.name
        )

        shutil.copy2(
            label_path,
            val_label_dir / label_path.name
        )

    # --------------------------------------------------------
    # REAL TRAINING
    # --------------------------------------------------------

    train_dir = OUTPUT_DIR / "training"

    for image_path in train_images:

        label_path = REAL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():

            print(
                f"WARNING: Label missing for {image_path.name}"
            )

            continue

        # BOTH IMAGE AND LABEL GO INTO SAME FOLDER
        copy_pair(
            image_path,
            label_path,
            train_dir
        )

    print("Real dataset completed.")


# ============================================================
# PROCESS SYNTHETIC DATASET
# ============================================================

def process_synthetic_dataset():

    print("\n" + "=" * 60)
    print("PROCESSING SYNTHETIC DATASET")
    print("=" * 60)

    synthetic_images_dir = SYNTHETIC_DIR / "images"
    synthetic_labels_dir = SYNTHETIC_DIR / "labels"

    images = get_images(synthetic_images_dir)

    print(f"Total synthetic images: {len(images)}")

    if len(images) < VAL_COUNT:

        raise ValueError(
            f"Synthetic dataset has only {len(images)} images."
        )

    random.shuffle(images)

    val_images = images[:VAL_COUNT]
    train_images = images[VAL_COUNT:]

    print(f"Synthetic validation: {len(val_images)}")
    print(f"Synthetic training:   {len(train_images)}")

    # --------------------------------------------------------
    # SYNTHETIC VALIDATION
    # --------------------------------------------------------

    val_image_dir = (
        OUTPUT_DIR /
        "validation_synthetic" /
        "images"
    )

    val_label_dir = (
        OUTPUT_DIR /
        "validation_synthetic" /
        "labels"
    )

    for image_path in val_images:

        label_path = (
            synthetic_labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: Label missing for {image_path.name}"
            )

            continue

        shutil.copy2(
            image_path,
            val_image_dir / image_path.name
        )

        shutil.copy2(
            label_path,
            val_label_dir / label_path.name
        )

    # --------------------------------------------------------
    # SYNTHETIC TRAINING
    # --------------------------------------------------------

    train_dir = OUTPUT_DIR / "training"

    for image_path in train_images:

        label_path = (
            synthetic_labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: Label missing for {image_path.name}"
            )

            continue

        # BOTH IMAGE AND LABEL GO INTO SAME FOLDER
        copy_pair(
            image_path,
            label_path,
            train_dir
        )

    print("Synthetic dataset completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("DATASET SPLITTING")
    print("=" * 60)

    create_directories()

    process_real_dataset()

    process_synthetic_dataset()

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

    print(f"""
Output:

{OUTPUT_DIR}/

├── validation_real/
│   ├── images/
│   └── labels/
│
├── validation_synthetic/
│   ├── images/
│   └── labels/
│
└── training/
    ├── image.jpg
    ├── image.txt
    ├── image.jpg
    ├── image.txt
    └── ...
""")
