import os
import shutil
import random
import argparse


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create a new balanced train/validation dataset "
                "from training_dataset_Ying and validation_dataset_Ying."
)

parser.add_argument(
    "--training_dir",
    required=True,
    help="Path to training_dataset_Ying"
)

parser.add_argument(
    "--validation_dir",
    required=True,
    help="Path to validation_dataset_Ying"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Path for the new balanced dataset"
)

parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.23,
    help="Validation ratio. Example: 0.23, 0.25 or 0.30"
)

parser.add_argument(
    "--val_real",
    type=int,
    default=3000,
    help="Number of real images required in validation"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DIR = args.training_dir
VALIDATION_DIR = args.validation_dir
OUTPUT_DIR = args.output_dir

VAL_RATIO = args.val_ratio
VAL_REAL_COUNT = args.val_real
SEED = args.seed

random.seed(SEED)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
)


# ============================================================
# PATHS
# ============================================================

training_images_dir = os.path.join(
    TRAINING_DIR,
    "images"
)

training_labels_dir = os.path.join(
    TRAINING_DIR,
    "labels"
)

validation_images_dir = os.path.join(
    VALIDATION_DIR,
    "images"
)

validation_labels_dir = os.path.join(
    VALIDATION_DIR,
    "labels"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

output_training_images = os.path.join(
    OUTPUT_DIR,
    "training",
    "images"
)

output_training_labels = os.path.join(
    OUTPUT_DIR,
    "training",
    "labels"
)

output_validation_images = os.path.join(
    OUTPUT_DIR,
    "validation",
    "images"
)

output_validation_labels = os.path.join(
    OUTPUT_DIR,
    "validation",
    "labels"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [

    output_training_images,
    output_training_labels,
    output_validation_images,
    output_validation_labels

]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# GET IMAGE + LABEL PAIRS
# ============================================================

def get_image_label_pairs(
    images_dir,
    labels_dir
):

    pairs = []

    if not os.path.isdir(images_dir):

        raise FileNotFoundError(
            f"Images directory not found:\n{images_dir}"
        )

    if not os.path.isdir(labels_dir):

        raise FileNotFoundError(
            f"Labels directory not found:\n{labels_dir}"
        )

    for filename in os.listdir(images_dir):

        if not filename.endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        image_path = os.path.join(
            images_dir,
            filename
        )

        stem = os.path.splitext(
            filename
        )[0]

        label_path = os.path.join(
            labels_dir,
            stem + ".txt"
        )

        # ----------------------------------------------------
        # Only use images which have corresponding labels
        # ----------------------------------------------------

        if not os.path.exists(label_path):

            print(
                f"WARNING: Label missing for {filename}"
            )

            continue

        pairs.append(
            (
                image_path,
                label_path,
                filename
            )
        )

    return pairs


# ============================================================
# IDENTIFY REAL / SYNTHETIC
# ============================================================

def is_synthetic(filename):

    """
    Synthetic images are identified by '_'
    in the filename.

    Example:

        ABC123.jpg       -> REAL
        ABC123_0.jpg     -> SYNTHETIC
        ABC123_1.jpg     -> SYNTHETIC
    """

    stem = os.path.splitext(
        filename
    )[0]

    return "_" in stem


# ============================================================
# LOAD ALL DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)


training_pairs = get_image_label_pairs(
    training_images_dir,
    training_labels_dir
)

validation_pairs = get_image_label_pairs(
    validation_images_dir,
    validation_labels_dir
)


print(
    f"\nTraining dataset pairs   : "
    f"{len(training_pairs)}"
)

print(
    f"Validation dataset pairs : "
    f"{len(validation_pairs)}"
)


# ============================================================
# COMBINE BOTH DATASETS
# ============================================================

all_pairs = (
    training_pairs +
    validation_pairs
)


print(
    f"Total paired images      : "
    f"{len(all_pairs)}"
)


# ============================================================
# SEPARATE REAL / SYNTHETIC
# ============================================================

real_pairs = []
synthetic_pairs = []


for pair in all_pairs:

    image_path = pair[0]
    filename = pair[2]

    if is_synthetic(filename):

        synthetic_pairs.append(pair)

    else:

        real_pairs.append(pair)


# Shuffle independently
random.shuffle(real_pairs)
random.shuffle(synthetic_pairs)


print("\n" + "=" * 70)
print("REAL / SYNTHETIC DISTRIBUTION")
print("=" * 70)

print(
    f"Real images      : {len(real_pairs)}"
)

print(
    f"Synthetic images : {len(synthetic_pairs)}"
)


# ============================================================
# CALCULATE TARGET VALIDATION SIZE
# ============================================================

total_images = len(all_pairs)

target_validation = round(
    total_images * VAL_RATIO
)

target_training = (
    total_images -
    target_validation
)


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    f"Validation ratio : {VAL_RATIO * 100:.2f}%"
)

print(
    f"Training ratio   : {(1 - VAL_RATIO) * 100:.2f}%"
)

print(
    f"Total images     : {total_images}"
)

print(
    f"Training target  : {target_training}"
)

print(
    f"Validation target: {target_validation}"
)

print(
    f"Real validation  : {VAL_REAL_COUNT}"
)

print(
    f"Synthetic validation: "
    f"{target_validation - VAL_REAL_COUNT}"
)


# ============================================================
# CHECK AVAILABILITY
# ============================================================

if VAL_REAL_COUNT > len(real_pairs):

    raise ValueError(
        f"\nRequested {VAL_REAL_COUNT} real images "
        f"for validation, but only "
        f"{len(real_pairs)} real images are available."
    )


validation_synthetic_count = (
    target_validation -
    VAL_REAL_COUNT
)


if validation_synthetic_count > len(
    synthetic_pairs
):

    raise ValueError(
        f"\nRequested {validation_synthetic_count} "
        f"synthetic images for validation, but only "
        f"{len(synthetic_pairs)} are available."
    )


# ============================================================
# SELECT VALIDATION DATA
# ============================================================

validation_real = real_pairs[
    :VAL_REAL_COUNT
]

validation_synthetic = synthetic_pairs[
    :validation_synthetic_count
]


# Combine validation
validation_selected = (
    validation_real +
    validation_synthetic
)


# ============================================================
# REMAINING DATA = TRAINING
# ============================================================

validation_ids = {
    os.path.abspath(pair[0])
    for pair in validation_selected
}


training_selected = [

    pair

    for pair in all_pairs

    if os.path.abspath(pair[0])
    not in validation_ids

]


# Shuffle final datasets
random.shuffle(
    validation_selected
)

random.shuffle(
    training_selected
)


# ============================================================
# FINAL COUNTS
# ============================================================

training_real = [
    pair
    for pair in training_selected
    if not is_synthetic(pair[2])
]

training_synthetic = [
    pair
    for pair in training_selected
    if is_synthetic(pair[2])
]


validation_real_final = [
    pair
    for pair in validation_selected
    if not is_synthetic(pair[2])
]

validation_synthetic_final = [
    pair
    for pair in validation_selected
    if is_synthetic(pair[2])
]


print("\n" + "=" * 70)
print("FINAL DATASET DISTRIBUTION")
print("=" * 70)

print("\nTRAINING")
print(
    f"Total     : {len(training_selected)}"
)
print(
    f"Real      : {len(training_real)}"
)
print(
    f"Synthetic : {len(training_synthetic)}"
)

print("\nVALIDATION")
print(
    f"Total     : {len(validation_selected)}"
)
print(
    f"Real      : {len(validation_real_final)}"
)
print(
    f"Synthetic : {len(validation_synthetic_final)}"
)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_dataset(
    pairs,
    output_images,
    output_labels
):

    copied = 0

    for image_path, label_path, filename in pairs:

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = os.path.join(
            output_images,
            filename
        )

        shutil.copy2(
            image_path,
            destination_image
        )


        # ----------------------------------------------------
        # Copy label
        # ----------------------------------------------------

        label_filename = os.path.basename(
            label_path
        )

        destination_label = os.path.join(
            output_labels,
            label_filename
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        copied += 1

    return copied


# ============================================================
# COPY TRAINING
# ============================================================

print("\n" + "=" * 70)
print("COPYING TRAINING DATA")
print("=" * 70)

training_copied = copy_dataset(
    training_selected,
    output_training_images,
    output_training_labels
)

print(
    f"Training images copied: "
    f"{training_copied}"
)


# ============================================================
# COPY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("COPYING VALIDATION DATA")
print("=" * 70)

validation_copied = copy_dataset(
    validation_selected,
    output_validation_images,
    output_validation_labels
)

print(
    f"Validation images copied: "
    f"{validation_copied}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETED")
print("=" * 70)

print(
    f"""
Output:

{OUTPUT_DIR}/
│
├── training/
│   ├── images/
│   └── labels/
│
└── validation/
    ├── images/
    └── labels/
"""
)

print(
    f"Training   : {len(training_selected)}"
)

print(
    f"Validation : {len(validation_selected)}"
)

print(
    f"Total      : "
    f"{len(training_selected) + len(validation_selected)}"
)

print("\nValidation composition:")

print(
    f"  Real      : "
    f"{len(validation_real_final)}"
)

print(
    f"  Synthetic : "
    f"{len(validation_synthetic_final)}"
)

print("\nDone.")
