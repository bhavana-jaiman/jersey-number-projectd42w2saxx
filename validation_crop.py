import os
import random
import shutil
import argparse

import cv2
import numpy as np


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--real_dir",
    required=True,
    help="Path to real dataset"
)

parser.add_argument(
    "--synthetic_dir",
    required=True,
    help="Path to synthetic dataset"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output validation directory"
)

parser.add_argument(
    "--val_count",
    type=int,
    default=2000,
    help="Number of validation images from each dataset"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--visualize",
    action="store_true",
    help="Save visualization of cropped validation images"
)

parser.add_argument(
    "--visualize_count",
    type=int,
    default=20
)

args = parser.parse_args()


REAL_DIR = args.real_dir
SYNTHETIC_DIR = args.synthetic_dir
OUTPUT_DIR = args.output_dir

VAL_COUNT = args.val_count
RANDOM_SEED = args.seed


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    os.path.join(
        OUTPUT_DIR,
        "validation_real",
        "images"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_DIR,
        "validation_real",
        "labels"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_DIR,
        "validation_synthetic",
        "images"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        OUTPUT_DIR,
        "validation_synthetic",
        "labels"
    ),
    exist_ok=True
)


# ============================================================
# PAD IMAGE TO SQUARE
# ============================================================

def pad_to_square(image, pad_value=0):

    h, w = image.shape[:2]

    if h == w:

        return image, [0, 0, 0, 0]

    if h > w:

        difference = h - w

        pad_left = difference // 2
        pad_right = difference - pad_left

        padded_image = cv2.copyMakeBorder(
            image,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        return padded_image, [
            0,
            pad_left,
            0,
            pad_right
        ]

    else:

        difference = w - h

        pad_top = difference // 2
        pad_bottom = difference - pad_top

        padded_image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )

        return padded_image, [
            pad_top,
            0,
            pad_bottom,
            0
        ]


# ============================================================
# READ LABEL
# ============================================================

def read_label(label_path):

    labels = []

    if not os.path.exists(label_path):

        return labels

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:

                values = [
                    float(x)
                    for x in parts[:5]
                ]

                labels.append(values)

            except ValueError:

                continue

    return labels


# ============================================================
# GET DIGIT NUMBER AND CROP BOX
# ============================================================

def calculate_crop(image, labels):

    """
    This follows the cropping logic from the training
    Dataset shown in the screenshots.
    """

    if len(labels) == 0:

        return None, None


    # --------------------------------------------------------
    # First annotation
    # --------------------------------------------------------

    label0 = labels[0]

    center_x = label0[1]
    center_y = label0[2]

    new_width = label0[3]
    new_height = label0[4]


    # --------------------------------------------------------
    # Get class/digit numbers
    # --------------------------------------------------------

    digit_number = [
        int(label[0])
        for label in labels[:2]
    ]


    # --------------------------------------------------------
    # Handle two annotations
    # --------------------------------------------------------

    if len(labels) > 1 and labels[0][0] != 10:

        # If boxes are far apart,
        # use only first digit

        if (
            abs(labels[0][1] - labels[1][1]) > 0.25
            or
            abs(labels[0][2] - labels[1][2]) > 0.25
        ):

            digit_number = []

            digit_number.append(
                int(labels[0][0])
            )

        else:

            # Combine two digit boxes

            center_x = (
                labels[0][1] +
                labels[1][1]
            ) / 2

            center_y = (
                labels[0][2] +
                labels[1][2]
            ) / 2

            new_width = (
                labels[0][3] +
                labels[1][3]
                +
                abs(
                    labels[0][1] -
                    labels[1][1]
                )
            )

            new_height = (
                labels[0][4] +
                labels[1][4]
            )

            # Sort digits from left to right

            if labels[0][1] > labels[1][1]:

                digit_number[0], digit_number[1] = (
                    digit_number[1],
                    digit_number[0]
                )


    # --------------------------------------------------------
    # Read image dimensions
    # --------------------------------------------------------

    h_factor, w_factor = image.shape[:2]


    # --------------------------------------------------------
    # Pad image to square
    # --------------------------------------------------------

    image, pad = pad_to_square(
        image,
        128
    )

    padded_h, padded_w = image.shape[:2]


    # --------------------------------------------------------
    # Convert normalized YOLO coordinates
    # to pixel coordinates
    # --------------------------------------------------------

    x1 = w_factor * (
        center_x - new_width / 2
    )

    y1 = h_factor * (
        center_y - new_height / 2
    )

    x2 = w_factor * (
        center_x + new_width / 2
    )

    y2 = h_factor * (
        center_y + new_height / 2
    )


    # --------------------------------------------------------
    # Adjust for padding
    # --------------------------------------------------------

    x1 += pad[1]
    y1 += pad[0]

    x2 += pad[1]
    y2 += pad[0]


    # --------------------------------------------------------
    # Convert crop coordinates back to normalized
    # coordinates relative to padded image
    # --------------------------------------------------------

    center_x_new = (
        ((x1 + x2) / 2)
        / padded_w
    )

    center_y_new = (
        ((y1 + y2) / 2)
        / padded_h
    )

    width_new = (
        (x2 - x1)
        / padded_w
    )

    height_new = (
        (y2 - y1)
        / padded_h
    )


    # --------------------------------------------------------
    # SAME EXPANSION AS TRAINING
    # --------------------------------------------------------

    if labels[0][0] != 10:

        width_new *= 3

        height_new *= 1.7


    # --------------------------------------------------------
    # Convert expanded normalized box
    # to pixels
    # --------------------------------------------------------

    x_min = (
        center_x_new * padded_w
        -
        width_new * padded_w / 2
    )

    y_min = (
        center_y_new * padded_h
        -
        height_new * padded_h / 2
    )

    x_max = (
        center_x_new * padded_w
        +
        width_new * padded_w / 2
    )

    y_max = (
        center_y_new * padded_h
        +
        height_new * padded_h / 2
    )


    # --------------------------------------------------------
    # SAME MARGINS AS TRAINING
    # --------------------------------------------------------

    x_min -= 23
    y_min -= 30

    x_max += 30
    y_max += 33


    # --------------------------------------------------------
    # Clip to image boundaries
    # --------------------------------------------------------

    x_min = max(
        0,
        int(x_min)
    )

    y_min = max(
        0,
        int(y_min)
    )

    x_max = min(
        padded_w,
        int(x_max)
    )

    y_max = min(
        padded_h,
        int(y_max)
    )


    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    cropped_image = image[
        y_min:y_max,
        x_min:x_max
    ]


    # --------------------------------------------------------
    # Calculate jersey number
    # --------------------------------------------------------

    jersey_number_length = len(
        digit_number
    )

    # No jersey number
    if labels[0][0] == 10:

        jersey_number = 100

    elif jersey_number_length == 1:

        jersey_number = digit_number[0]

    else:

        jersey_number = (
            digit_number[0] * 10
            +
            digit_number[1]
        )


    return cropped_image, jersey_number


# ============================================================
# CREATE 1D LABEL
# ============================================================

def create_1d_label(labels):

    """

    Example:

        3 x y w h
        7 x y w h

    becomes:

        3 7
    """

    digit_number = [
        int(label[0])
        for label in labels[:2]
    ]


    if len(labels) > 1 and labels[0][0] != 10:

        if (
            abs(labels[0][1] - labels[1][1]) <= 0.25
            and
            abs(labels[0][2] - labels[1][2]) <= 0.25
        ):

            # Sort left -> right

            if labels[0][1] > labels[1][1]:

                digit_number.reverse()


    return " ".join(
        str(x)
        for x in digit_number
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    label_path,
    output_image_dir,
    output_label_dir
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"ERROR reading: {image_path}"
        )

        return False, None


    labels = read_label(
        label_path
    )

    if len(labels) == 0:

        print(
            f"WARNING: No labels: {label_path}"
        )

        return False, None


    cropped_image, jersey_number = calculate_crop(
        image,
        labels
    )

    if cropped_image is None:
        return False, None


    if cropped_image.size == 0:

        print(
            f"WARNING: Empty crop: {image_path}"
        )

        return False, None


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    image_name = os.path.basename(
        image_path
    )

    output_image_path = os.path.join(
        output_image_dir,
        image_name
    )

    cv2.imwrite(
        output_image_path,
        cropped_image
    )


    # --------------------------------------------------------
    # Save 1D label
    # --------------------------------------------------------

    output_label_path = os.path.join(
        output_label_dir,
        os.path.splitext(image_name)[0] + ".txt"
    )

    new_label = create_1d_label(
        labels
    )

    with open(
        output_label_path,
        "w"
    ) as f:

        f.write(new_label)


    return True, jersey_number


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    image_dir,
    label_dir,
    output_image_dir,
    output_label_dir,
    visualize_count
):

    print("\n" + "=" * 70)

    print(
        f"PROCESSING {dataset_name.upper()}"
    )

    print("=" * 70)


    image_files = [

        f for f in os.listdir(image_dir)

        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]


    print(
        f"Total images: {len(image_files)}"
    )


    if len(image_files) < VAL_COUNT:

        raise ValueError(
            f"{dataset_name} has only "
            f"{len(image_files)} images."
        )


    # --------------------------------------------------------
    # Random selection
    # --------------------------------------------------------

    random.shuffle(
        image_files
    )

    selected_images = image_files[
        :VAL_COUNT
    ]


    print(
        f"Selected validation images: "
        f"{len(selected_images)}"
    )


    success_count = 0


    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, image_name in enumerate(
        selected_images
    ):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        label_path = os.path.join(
            label_dir,
            os.path.splitext(
                image_name
            )[0] + ".txt"
        )


        if not os.path.exists(
            label_path
        ):

            print(
                f"Missing label: {image_name}"
            )

            continue


        success, jersey_number = process_image(

            image_path,

            label_path,

            output_image_dir,

            output_label_dir

        )


        if success:

            success_count += 1


        # ----------------------------------------------------
        # Optional visualization
        # ----------------------------------------------------

        if (
            args.visualize
            and
            success_count <= visualize_count
        ):

            print(
                f"[{success_count}] "
                f"{image_name} -> "
                f"jersey number: "
                f"{jersey_number}"
            )


    print(
        f"\nSuccessfully processed: "
        f"{success_count}/{len(selected_images)}"
    )


# ============================================================
# PATHS
# ============================================================

real_output_images = os.path.join(
    OUTPUT_DIR,
    "validation_real",
    "images"
)

real_output_labels = os.path.join(
    OUTPUT_DIR,
    "validation_real",
    "labels"
)

synthetic_output_images = os.path.join(
    OUTPUT_DIR,
    "validation_synthetic",
    "images"
)

synthetic_output_labels = os.path.join(
    OUTPUT_DIR,
    "validation_synthetic",
    "labels"
)


# ============================================================
# REAL DATASET
# ============================================================

process_dataset(

    "real",

    REAL_DIR,

    REAL_DIR,

    real_output_images,

    real_output_labels,

    args.visualize_count

)


# ============================================================
# SYNTHETIC DATASET
# ============================================================

process_dataset(

    "synthetic",

    os.path.join(
        SYNTHETIC_DIR,
        "images"
    ),

    os.path.join(
        SYNTHETIC_DIR,
        "labels"
    ),

    synthetic_output_images,

    synthetic_output_labels,

    args.visualize_count

)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)

print("VALIDATION DATASET CREATION COMPLETED")

print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)
