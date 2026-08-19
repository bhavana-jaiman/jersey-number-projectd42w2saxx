from pathlib import Path
import shutil
import random
import sys

# ============================================================
# CONFIGURATION
# ============================================================

# This script should be run from the "datasets" directory.
BASE_DIR = Path(__file__).resolve().parent

# Existing datasets - THESE WILL NOT BE MODIFIED
TRAINING_DATASET = BASE_DIR / "training_dataset_Ying"
VALIDATION_DATASET = BASE_DIR / "validation_dataset_Ying"

# New dataset
OUTPUT_DATASET = BASE_DIR / "combined_split_dataset_Ying"

# Train/Validation split
TRAIN_RATIO = 0.90

# Fixed seed so that the same split can be reproduced
RANDOM_SEED = 42

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# CHECK EXISTING DATASET PATHS
# ============================================================

print("=" * 70)
print("COMBINED TRAINING + VALIDATION DATASET CREATION")
print("=" * 70)

print("\nChecking source directories...")

if not TRAINING_DATASET.exists():
    print(f"ERROR: Training dataset not found:")
    print(f"       {TRAINING_DATASET}")
    sys.exit(1)

if not VALIDATION_DATASET.exists():
    print(f"ERROR: Validation dataset not found:")
    print(f"       {VALIDATION_DATASET}")
    sys.exit(1)

print(f"Training dataset   : {TRAINING_DATASET}")
print(f"Validation dataset : {VALIDATION_DATASET}")
print(f"Output dataset     : {OUTPUT_DATASET}")


# ============================================================
# SAFETY CHECK
# ============================================================

# Do not overwrite an existing output dataset.
if OUTPUT_DATASET.exists():
    print("\nERROR: Output dataset already exists!")
    print(f"       {OUTPUT_DATASET}")
    print("\nFor safety, this script will NOT modify an existing output.")
    print("Delete/rename the output folder if you want to run the")
    print("script again.")
    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRAIN_OUTPUT = OUTPUT_DATASET / "train"
VAL_OUTPUT = OUTPUT_DATASET / "validation"

TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
VAL_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCTION TO FIND IMAGE + LABEL PAIRS
# ============================================================

def collect_pairs(dataset_path):
    """
    Finds image files and their corresponding .txt labels.

    Expected structure:

        dataset/
            image1.jpg
            image1.txt
            image2.jpg
            image2.txt

    Returns:
        list of tuples:
        (image_path, label_path)
    """

    pairs = []

    missing_labels = []
    total_images = 0

    for image_path in dataset_path.iterdir():

        # Ignore directories
        if not image_path.is_file():
            continue

        # Check if file is an image
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1

        # Corresponding label
        label_path = dataset_path / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    print(f"\nDataset: {dataset_path.name}")
    print(f"Images found          : {total_images}")
    print(f"Valid image-label pairs: {len(pairs)}")
    print(f"Missing labels        : {len(missing_labels)}")

    if missing_labels:
        print("\nWARNING: The following images do not have labels:")

        for filename in missing_labels[:20]:
            print(f"  {filename}")

        if len(missing_labels) > 20:
            print(
                f"  ... and {len(missing_labels) - 20} more"
            )

    return pairs


# ============================================================
# COLLECT TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: READING TRAINING DATASET")
print("=" * 70)

training_pairs = collect_pairs(TRAINING_DATASET)


# ============================================================
# COLLECT VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: READING VALIDATION DATASET")
print("=" * 70)

validation_pairs = collect_pairs(VALIDATION_DATASET)


# ============================================================
# CHECK TOTAL DATA
# ============================================================

total_pairs = len(training_pairs) + len(validation_pairs)

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Training pairs originally   : {len(training_pairs)}")
print(f"Validation pairs originally : {len(validation_pairs)}")
print(f"Total combined pairs        : {total_pairs}")


if total_pairs == 0:
    print("\nERROR: No valid image-label pairs were found.")
    sys.exit(1)


# ============================================================
# COMBINE BOTH DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: COMBINING DATASETS")
print("=" * 70)

all_pairs = training_pairs + validation_pairs

print(f"Combined pairs: {len(all_pairs)}")


# ============================================================
# SHUFFLE DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: RANDOM SHUFFLING")
print("=" * 70)

random.seed(RANDOM_SEED)

random.shuffle(all_pairs)

print(f"Random seed: {RANDOM_SEED}")
print("Dataset shuffled successfully.")


# ============================================================
# CALCULATE TRAIN / VALIDATION SPLIT
# ============================================================

train_count = int(len(all_pairs) * TRAIN_RATIO)

new_train_pairs = all_pairs[:train_count]
new_validation_pairs = all_pairs[train_count:]

print("\n" + "=" * 70)
print("STEP 5: NEW DATASET SPLIT")
print("=" * 70)

print(f"Total images       : {len(all_pairs)}")
print(f"Training images    : {len(new_train_pairs)}")
print(f"Validation images  : {len(new_validation_pairs)}")

print(
    f"Training percentage   : "
    f"{len(new_train_pairs) / len(all_pairs) * 100:.2f}%"
)

print(
    f"Validation percentage : "
    f"{len(new_validation_pairs) / len(all_pairs) * 100:.2f}%"
)


# ============================================================
# FUNCTION TO COPY IMAGE + LABEL TOGETHER
# ============================================================

def copy_pairs(pairs, destination):
    """
    Copies each image and its corresponding label
    into the same destination directory.
    """

    copied = 0

    for image_path, label_path in pairs:

        # Keep the original filename.
        destination_image = destination / image_path.name
        destination_label = destination / label_path.name

        # Safety check
        if destination_image.exists():
            raise RuntimeError(
                f"Duplicate image filename detected: "
                f"{image_path.name}"
            )

        if destination_label.exists():
            raise RuntimeError(
                f"Duplicate label filename detected: "
                f"{label_path.name}"
            )

        # Copy image
        shutil.copy2(
            image_path,
            destination_image
        )

        # Copy corresponding label
        shutil.copy2(
            label_path,
            destination_label
        )

        copied += 1

    return copied


# ============================================================
# COPY TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: CREATING NEW TRAINING DATASET")
print("=" * 70)

train_copied = copy_pairs(
    new_train_pairs,
    TRAIN_OUTPUT
)

print(f"Training image-label pairs copied: {train_copied}")


# ============================================================
# COPY VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: CREATING NEW VALIDATION DATASET")
print("=" * 70)

validation_copied = copy_pairs(
    new_validation_pairs,
    VAL_OUTPUT
)

print(
    f"Validation image-label pairs copied: "
    f"{validation_copied}"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: FINAL VERIFICATION")
print("=" * 70)


def verify_dataset(folder):
    """
    Verify that every image has a corresponding label
    and every label has a corresponding image.
    """

    images = [
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        f for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() == ".txt"
    ]

    image_stems = {f.stem for f in images}
    label_stems = {f.stem for f in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    return (
        len(images),
        len(labels),
        missing_labels,
        missing_images
    )


train_images, train_labels, train_missing_labels, train_missing_images = (
    verify_dataset(TRAIN_OUTPUT)
)

val_images, val_labels, val_missing_labels, val_missing_images = (
    verify_dataset(VAL_OUTPUT)
)


print("\nNEW TRAINING DATASET")
print("--------------------")
print(f"Images : {train_images}")
print(f"Labels : {train_labels}")

print("\nNEW VALIDATION DATASET")
print("----------------------")
print(f"Images : {val_images}")
print(f"Labels : {val_labels}")


# ============================================================
# CHECK FOR ERRORS
# ============================================================

errors = False

if train_missing_labels:
    print(
        f"\nERROR: {len(train_missing_labels)} "
        "training images have no label."
    )
    errors = True

if train_missing_images:
    print(
        f"\nERROR: {len(train_missing_images)} "
        "training labels have no image."
    )
    errors = True

if val_missing_labels:
    print(
        f"\nERROR: {len(val_missing_labels)} "
        "validation images have no label."
    )
    errors = True

if val_missing_images:
    print(
        f"\nERROR: {len(val_missing_images)} "
        "validation labels have no image."
    )
    errors = True


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if errors:
    print("DATASET CREATED WITH ERRORS - PLEASE CHECK ABOVE.")
else:
    print("DATASET CREATED SUCCESSFULLY!")
    print()
    print(f"New dataset location:")
    print(f"  {OUTPUT_DATASET}")
    print()
    print("Training:")
    print(f"  {TRAIN_OUTPUT}")
    print()
    print("Validation:")
    print(f"  {VAL_OUTPUT}")
    print()
    print("Original datasets were NOT modified.")

print("=" * 70)
