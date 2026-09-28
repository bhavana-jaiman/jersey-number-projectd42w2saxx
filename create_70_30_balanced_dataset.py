import os
import shutil
import random
import argparse


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create 70:30 train/validation split by taking "
                "additional validation samples ONLY from training."
)

parser.add_argument(
    "--training_dir",
    required=True,
    help="Training dataset where images and labels are in the same folder"
)

parser.add_argument(
    "--validation_dir",
    required=True,
    help="Existing validation dataset containing images/ and labels/"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="New output dataset directory"
)

parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.30,
    help="Final validation ratio. Default = 0.30"
)

parser.add_argument(
    "--additional_real",
    type=int,
    default=3000,
    help="Number of REAL training images to add to validation"
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
ADDITIONAL_REAL = args.additional_real

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
# CHECK LABEL FORMAT
# ============================================================

def check_yolo_label(label_path):

    """
    Checks that label contains YOLO 5-value annotations:

        class x_center y_center width height

    Example:

        3 0.42 0.53 0.12 0.25
    """

    try:

        with open(
            label_path,
            "r"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line in lines:

            values = line.split()

            if len(values) != 5:

                return False

        return True

    except Exception:

        return False


# ============================================================
# READ TRAINING DATA
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

    print("\nReading TRAINING dataset...")

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

        stem = os.path.splitext(
            filename
        )[0]

        label_path = os.path.join(
            TRAINING_DIR,
            stem + ".txt"
        )

        if not os.path.exists(
            label_path
        ):

            print(
                f"WARNING: Missing label: {filename}"
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
# READ EXISTING VALIDATION DATA
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


    if not os.path.isdir(images_dir):

        raise FileNotFoundError(
            f"Validation images directory not found:\n"
            f"{images_dir}"
        )


    if not os.path.isdir(labels_dir):

        raise FileNotFoundError(
            f"Validation labels directory not found:\n"
            f"{labels_dir}"
        )


    print("\nReading EXISTING VALIDATION dataset...")


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

        stem = os.path.splitext(
            filename
        )[0]

        label_path = os.path.join(
            labels_dir,
            stem + ".txt"
        )


        if not os.path.exists(
            label_path
        ):

            print(
                f"WARNING: Missing label: {filename}"
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
# REAL / SYNTHETIC IDENTIFICATION
# ============================================================

def is_synthetic(filename):

    """
    Synthetic example:

        ABC123_0.jpg
        ABC123_1.jpg

    Real example:

        ABC123.jpg
    """

    stem = os.path.splitext(
        filename
    )[0]

    return "_" in stem


# ============================================================
# LOAD DATA
# ============================================================

training_pairs = read_training_dataset()

existing_validation_pairs = read_validation_dataset()


print("\n" + "=" * 70)
print("INPUT DATASET COUNTS")
print("=" * 70)

print(
    f"Training images/labels pairs   : "
    f"{len(training_pairs)}"
)

print(
    f"Existing validation pairs      : "
    f"{len(existing_validation_pairs)}"
)


# ============================================================
# CLASSIFY TRAINING DATA
# ============================================================

training_real = []

training_synthetic = []


for pair in training_pairs:

    filename = pair[2]

    if is_synthetic(filename):

        training_synthetic.append(pair)

    else:

        training_real.append(pair)


random.shuffle(training_real)
random.shuffle(training_synthetic)


print("\n" + "=" * 70)
print("TRAINING DATA COMPOSITION")
print("=" * 70)

print(
    f"Real      : {len(training_real)}"
)

print(
    f"Synthetic : {len(training_synthetic)}"
)


# ============================================================
# TOTAL DATASET
# ============================================================

total_images = (
    len(training_pairs)
    +
    len(existing_validation_pairs)
)


# ============================================================
# FINAL VALIDATION SIZE
# ============================================================

target_validation = round(
    total_images * VAL_RATIO
)


target_training = (
    total_images -
    target_validation
)


# ============================================================
# HOW MANY TRAINING IMAGES TO MOVE TO VALIDATION
# ============================================================

additional_validation_count = (
    target_validation
    -
    len(existing_validation_pairs)
)


additional_synthetic = (
    additional_validation_count
    -
    ADDITIONAL_REAL
)


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    f"Total dataset              : "
    f"{total_images}"
)

print(
    f"Target training            : "
    f"{target_training}"
)

print(
    f"Target validation          : "
    f"{target_validation}"
)

print(
    f"Existing validation        : "
    f"{len(existing_validation_pairs)}"
)

print(
    f"Additional validation     : "
    f"{additional_validation_count}"
)

print(
    f"Additional REAL            : "
    f"{ADDITIONAL_REAL}"
)

print(
    f"Additional SYNTHETIC       : "
    f"{additional_synthetic}"
)


# ============================================================
# SAFETY CHECKS
# ============================================================

if ADDITIONAL_REAL > len(training_real):

    raise ValueError(
        f"\nRequested {ADDITIONAL_REAL} real images, "
        f"but only {len(training_real)} real images "
        f"are available in training."
    )


if additional_synthetic > len(training_synthetic):

    raise ValueError(
        f"\nRequested {additional_synthetic} synthetic "
        f"images, but only {len(training_synthetic)} "
        f"synthetic images are available."
    )


if additional_validation_count < 0:

    raise ValueError(
        "\nExisting validation dataset is already "
        "larger than the requested validation size."
    )


# ============================================================
# SELECT ADDITIONAL VALIDATION DATA
# ============================================================

selected_real = training_real[
    :ADDITIONAL_REAL
]


selected_synthetic = training_synthetic[
    :additional_synthetic
]


additional_validation = (
    selected_real
    +
    selected_synthetic
)


random.shuffle(
    additional_validation
)


# ============================================================
# IDENTIFY SELECTED TRAINING IMAGES
# ============================================================

selected_training_paths = {

    os.path.abspath(pair[0])

    for pair in additional_validation

}


# ============================================================
# NEW TRAINING DATA
#
# ONLY REMAINING TRAINING DATA
#
# Existing validation is NEVER added to training.
# ============================================================

new_training_pairs = [

    pair

    for pair in training_pairs

    if os.path.abspath(pair[0])
    not in selected_training_paths

]


# ============================================================
# NEW VALIDATION DATA
#
# Existing validation + selected training images
# ============================================================

new_validation_pairs = (
    existing_validation_pairs
    +
    additional_validation
)


random.shuffle(
    new_training_pairs
)

random.shuffle(
    new_validation_pairs
)


# ============================================================
# FINAL COMPOSITION
# ============================================================

final_training_real = [
    pair
    for pair in new_training_pairs
    if not is_synthetic(pair[2])
]

final_training_synthetic = [
    pair
    for pair in new_training_pairs
    if is_synthetic(pair[2])
]

final_validation_real = [
    pair
    for pair in new_validation_pairs
    if not is_synthetic(pair[2])
]

final_validation_synthetic = [
    pair
    for pair in new_validation_pairs
    if is_synthetic(pair[2])
]


# ============================================================
# PRINT FINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET DISTRIBUTION")
print("=" * 70)

print("\nTRAINING")
print(
    f"Total     : {len(new_training_pairs)}"
)

print(
    f"Real      : {len(final_training_real)}"
)

print(
    f"Synthetic : {len(final_training_synthetic)}"
)


print("\nVALIDATION")
print(
    f"Total     : {len(new_validation_pairs)}"
)

print(
    f"Real      : {len(final_validation_real)}"
)

print(
    f"Synthetic : {len(final_validation_synthetic)}"
)


# ============================================================
# COPY DATA
# ============================================================

def copy_pairs(
    pairs,
    output_images,
    output_labels
):

    copied = 0

    for image_path, label_path, filename in pairs:

        destination_image = os.path.join(
            output_images,
            filename
        )

        destination_label = os.path.join(
            output_labels,
            os.path.basename(label_path)
        )


        # ----------------------------------------------------
        # Prevent accidental overwriting
        # ----------------------------------------------------

        if os.path.exists(
            destination_image
        ):

            raise FileExistsError(
                f"Duplicate image filename detected:\n"
                f"{filename}"
            )


        if os.path.exists(
            destination_label
        ):

            raise FileExistsError(
                f"Duplicate label filename detected:\n"
                f"{os.path.basename(label_path)}"
            )


        shutil.copy2(
            image_path,
            destination_image
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
    new_training_pairs,
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
    new_validation_pairs,
    VAL_IMAGES,
    VAL_LABELS
)


print(
    f"Validation pairs copied: "
    f"{val_copied}"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

train_image_count = len([
    f
    for f in os.listdir(TRAIN_IMAGES)
    if f.endswith(IMAGE_EXTENSIONS)
])

train_label_count = len([
    f
    for f in os.listdir(TRAIN_LABELS)
    if f.endswith(".txt")
])

val_image_count = len([
    f
    for f in os.listdir(VAL_IMAGES)
    if f.endswith(IMAGE_EXTENSIONS)
])

val_label_count = len([
    f
    for f in os.listdir(VAL_LABELS)
    if f.endswith(".txt")
])


print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print(
    f"Training images   : {train_image_count}"
)

print(
    f"Training labels   : {train_label_count}"
)

print(
    f"Validation images : {val_image_count}"
)

print(
    f"Validation labels : {val_label_count}"
)


if train_image_count == train_label_count:

    print(
        "✓ Training image-label count matches."
    )

else:

    print(
        "✗ Training image-label count mismatch!"
    )


if val_image_count == val_label_count:

    print(
        "✓ Validation image-label count matches."
    )

else:

    print(
        "✗ Validation image-label count mismatch!"
    )


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

print("Done.")
