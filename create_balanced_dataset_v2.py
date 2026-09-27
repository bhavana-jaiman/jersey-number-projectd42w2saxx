import os
import shutil
import random
import argparse


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create balanced train/validation dataset"
)

parser.add_argument(
    "--training_dir",
    required=True,
    help="Training dataset containing images and labels directly"
)

parser.add_argument(
    "--validation_dir",
    required=True,
    help="Validation dataset containing images/ and labels/"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output dataset directory"
)

parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.23,
    help="Validation ratio, e.g. 0.23, 0.25 or 0.30"
)

parser.add_argument(
    "--val_real",
    type=int,
    default=3000,
    help="Number of real images in validation"
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

random.seed(args.seed)


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
# OUTPUT DIRECTORIES
# ============================================================

TRAIN_IMAGES = os.path.join(
    OUTPUT_DIR,
    "training",
    "images"
)

TRAIN_LABELS = os.path.join(
    OUTPUT_DIR,
    "training",
    "labels"
)

VAL_IMAGES = os.path.join(
    OUTPUT_DIR,
    "validation",
    "images"
)

VAL_LABELS = os.path.join(
    OUTPUT_DIR,
    "validation",
    "labels"
)


for directory in [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    VAL_IMAGES,
    VAL_LABELS
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# READ TRAINING DATASET
# ============================================================

def read_training_dataset():

    """
    Training structure:

    training_dataset_Ying/
        image1.jpg
        image1.txt
        image2.jpg
        image2.txt
        ...
    """

    pairs = []

    print("\nReading training dataset...")

    for filename in os.listdir(
        TRAINING_DIR
    ):

        if not filename.endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        image_path = os.path.join(
            TRAINING_DIR,
            filename
        )

        label_name = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_path = os.path.join(
            TRAINING_DIR,
            label_name
        )

        if not os.path.exists(
            label_path
        ):

            print(
                "WARNING: Missing label:",
                filename
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
# READ VALIDATION DATASET
# ============================================================

def read_validation_dataset():

    """
    Validation structure:

    validation_dataset_Ying/
        images/
            image1.jpg
            image2.jpg

        labels/
            image1.txt
            image2.txt
    """

    pairs = []

    images_dir = os.path.join(
        VALIDATION_DIR,
        "images"
    )

    labels_dir = os.path.join(
        VALIDATION_DIR,
        "labels"
    )


    if not os.path.isdir(
        images_dir
    ):

        raise FileNotFoundError(
            f"Validation images directory not found:\n"
            f"{images_dir}"
        )


    if not os.path.isdir(
        labels_dir
    ):

        raise FileNotFoundError(
            f"Validation labels directory not found:\n"
            f"{labels_dir}"
        )


    print("\nReading validation dataset...")


    for filename in os.listdir(
        images_dir
    ):

        if not filename.endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        image_path = os.path.join(
            images_dir,
            filename
        )

        label_name = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_path = os.path.join(
            labels_dir,
            label_name
        )


        if not os.path.exists(
            label_path
        ):

            print(
                "WARNING: Missing label:",
                filename
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
# IDENTIFY SYNTHETIC DATA
# ============================================================

def is_synthetic(filename):

    """
    Synthetic:

        ABC123_0.jpg
        ABC123_1.jpg

    Real:

        ABC123.jpg
    """

    filename_without_extension = os.path.splitext(
        filename
    )[0]

    return "_" in filename_without_extension


# ============================================================
# LOAD DATA
# ============================================================

training_pairs = read_training_dataset()

validation_pairs = read_validation_dataset()


print("\n" + "=" * 70)
print("DATASET COUNTS")
print("=" * 70)

print(
    f"Training pairs   : {len(training_pairs)}"
)

print(
    f"Validation pairs : {len(validation_pairs)}"
)


# ============================================================
# COMBINE BOTH DATASETS
# ============================================================

all_pairs = (
    training_pairs +
    validation_pairs
)


print(
    f"Total pairs      : {len(all_pairs)}"
)


# ============================================================
# SEPARATE REAL / SYNTHETIC
# ============================================================

real_pairs = []
synthetic_pairs = []


for pair in all_pairs:

    filename = pair[2]

    if is_synthetic(filename):

        synthetic_pairs.append(pair)

    else:

        real_pairs.append(pair)


# Shuffle
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
# CALCULATE VALIDATION SIZE
# ============================================================

total_images = len(all_pairs)

validation_count = round(
    total_images * VAL_RATIO
)

training_count = (
    total_images -
    validation_count
)


# ============================================================
# CALCULATE SYNTHETIC VALIDATION COUNT
# ============================================================

validation_synthetic_count = (
    validation_count -
    VAL_REAL_COUNT
)


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    f"Validation ratio      : "
    f"{VAL_RATIO * 100:.2f}%"
)

print(
    f"Training ratio        : "
    f"{(1 - VAL_RATIO) * 100:.2f}%"
)

print(
    f"Total images          : "
    f"{total_images}"
)

print(
    f"Training images       : "
    f"{training_count}"
)

print(
    f"Validation images     : "
    f"{validation_count}"
)

print(
    f"Real validation       : "
    f"{VAL_REAL_COUNT}"
)

print(
    f"Synthetic validation : "
    f"{validation_synthetic_count}"
)


# ============================================================
# CHECK
# ============================================================

if VAL_REAL_COUNT > len(real_pairs):

    raise ValueError(
        f"Only {len(real_pairs)} real images available, "
        f"but {VAL_REAL_COUNT} requested."
    )


if validation_synthetic_count > len(
    synthetic_pairs
):

    raise ValueError(
        f"Only {len(synthetic_pairs)} synthetic images "
        f"available, but "
        f"{validation_synthetic_count} requested."
    )


# ============================================================
# SELECT VALIDATION
# ============================================================

validation_real = real_pairs[
    :VAL_REAL_COUNT
]

validation_synthetic = synthetic_pairs[
    :validation_synthetic_count
]


validation_selected = (
    validation_real +
    validation_synthetic
)


# Shuffle validation
random.shuffle(
    validation_selected
)


# ============================================================
# IDENTIFY VALIDATION FILES
# ============================================================

validation_image_paths = {

    os.path.abspath(pair[0])

    for pair in validation_selected

}


# ============================================================
# REMAINING DATA = TRAINING
# ============================================================

training_selected = [

    pair

    for pair in all_pairs

    if os.path.abspath(pair[0])
    not in validation_image_paths

]


random.shuffle(
    training_selected
)


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

final_train_real = [
    pair
    for pair in training_selected
    if not is_synthetic(pair[2])
]

final_train_synthetic = [
    pair
    for pair in training_selected
    if is_synthetic(pair[2])
]

final_val_real = [
    pair
    for pair in validation_selected
    if not is_synthetic(pair[2])
]

final_val_synthetic = [
    pair
    for pair in validation_selected
    if is_synthetic(pair[2])
]


print("\n" + "=" * 70)
print("FINAL DISTRIBUTION")
print("=" * 70)

print("\nTRAINING")

print(
    f"Total     : {len(training_selected)}"
)

print(
    f"Real      : {len(final_train_real)}"
)

print(
    f"Synthetic : {len(final_train_synthetic)}"
)


print("\nVALIDATION")

print(
    f"Total     : {len(validation_selected)}"
)

print(
    f"Real      : {len(final_val_real)}"
)

print(
    f"Synthetic : {len(final_val_synthetic)}"
)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_pairs(
    pairs,
    image_output,
    label_output
):

    copied = 0

    for image_path, label_path, filename in pairs:

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = os.path.join(
            image_output,
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
            label_output,
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


train_copied = copy_pairs(
    training_selected,
    TRAIN_IMAGES,
    TRAIN_LABELS
)


print(
    f"Training pairs copied: "
    f"{train_copied}"
)


# ============================================================
# COPY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("COPYING VALIDATION DATA")
print("=" * 70)


val_copied = copy_pairs(
    validation_selected,
    VAL_IMAGES,
    VAL_LABELS
)


print(
    f"Validation pairs copied: "
    f"{val_copied}"
)


# ============================================================
# VERIFY
# ============================================================

actual_train_images = len([
    f
    for f in os.listdir(TRAIN_IMAGES)
    if f.endswith(IMAGE_EXTENSIONS)
])

actual_train_labels = len([
    f
    for f in os.listdir(TRAIN_LABELS)
    if f.endswith(".txt")
])

actual_val_images = len([
    f
    for f in os.listdir(VAL_IMAGES)
    if f.endswith(IMAGE_EXTENSIONS)
])

actual_val_labels = len([
    f
    for f in os.listdir(VAL_LABELS)
    if f.endswith(".txt")
])


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETED")
print("=" * 70)

print(
    f"""
Output:

{OUTPUT_DIR}/

├── training/
│   ├── images/
│   └── labels/
│
└── validation/
    ├── images/
    └── labels/
"""
)


print("Verification:")
print(
    f"Training images : {actual_train_images}"
)

print(
    f"Training labels : {actual_train_labels}"
)

print(
    f"Validation images : {actual_val_images}"
)

print(
    f"Validation labels : {actual_val_labels}"
)


if (
    actual_train_images ==
    actual_train_labels
):

    print(
        "✓ Training image-label count matches."
    )

else:

    print(
        "WARNING: Training image-label count mismatch!"
    )


if (
    actual_val_images ==
    actual_val_labels
):

    print(
        "✓ Validation image-label count matches."
    )

else:

    print(
        "WARNING: Validation image-label count mismatch!"
    )


print("\nDone.")
