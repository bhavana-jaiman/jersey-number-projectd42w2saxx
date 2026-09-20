import os
import random
import argparse

import cv2
import numpy as np


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Create cropped validation dataset using "
                "the same cropping logic as training."
)

parser.add_argument(
    "--real_dir",
    required=True,
    help="Path to validation_real directory"
)

parser.add_argument(
    "--synthetic_dir",
    required=True,
    help="Path to validation_synthetic directory"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output directory"
)

parser.add_argument(
    "--val_count",
    type=int,
    default=2000,
    help="Number of images to process from each dataset"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

parser.add_argument(
    "--visualize",
    action="store_true",
    help="Save sample cropped images for visualization"
)

parser.add_argument(
    "--visualize_count",
    type=int,
    default=20,
    help="Number of samples to save for visualization"
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

VAL_COUNT = args.val_count
RANDOM_SEED = args.seed

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

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
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    directories = [

        # Real
        os.path.join(
            args.output_dir,
            "validation_real",
            "images"
        ),

        os.path.join(
            args.output_dir,
            "validation_real",
            "labels"
        ),

        # Synthetic
        os.path.join(
            args.output_dir,
            "validation_synthetic",
            "images"
        ),

        os.path.join(
            args.output_dir,
            "validation_synthetic",
            "labels"
        ),
    ]

    # Visualization directories
    if args.visualize:

        directories.extend([

            os.path.join(
                args.output_dir,
                "visualization",
                "real"
            ),

            os.path.join(
                args.output_dir,
                "visualization",
                "synthetic"
            )
        ])

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# PAD IMAGE TO SQUARE
# ============================================================

def pad_to_square(image, pad_value=128):

    """
    Same concept as the training loader:

        image, pad = pad_to_square(image, 128)

    Returns:

        padded_image

        pad = [
            pad_top,
            pad_left,
            pad_bottom,
            pad_right
        ]
    """

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
# READ YOLO LABEL
# ============================================================

def read_yolo_label(label_path):

    """
    Reads labels in YOLO format:

        class x_center y_center width height

    Returns:

        [
            [class, x, y, w, h],
            ...
        ]
    """

    labels = []

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:

                print(
                    f"WARNING: Invalid YOLO label "
                    f"in {label_path}: {line}"
                )

                continue

            try:

                label = [
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ]

                labels.append(label)

            except ValueError:

                print(
                    f"WARNING: Cannot parse label "
                    f"in {label_path}: {line}"
                )

    return labels


# ============================================================
# CALCULATE CROP
# ============================================================

def calculate_crop(image, labels):

    """
    Implements the cropping logic from the training
    jersey_Dataset shown in the screenshots.
    """

    if len(labels) == 0:

        return None, None


    # --------------------------------------------------------
    # First label
    # --------------------------------------------------------

    label0 = labels[0]

    center_x = label0[1]
    center_y = label0[2]

    new_width = label0[3]
    new_height = label0[4]


    # --------------------------------------------------------
    # Get digit numbers
    # --------------------------------------------------------

    digit_number = [
        int(label[0])
        for label in labels[:2]
    ]


    # --------------------------------------------------------
    # Two-digit logic
    # --------------------------------------------------------

    if len(labels) > 1 and labels[0][0] != 10:

        # ----------------------------------------------------
        # If two boxes are far apart
        # use only first digit
        # ----------------------------------------------------

        if (
            abs(labels[0][1] - labels[1][1]) > 0.25
            or
            abs(labels[0][2] - labels[1][2]) > 0.25
        ):

            digit_number = [
                int(labels[0][0])
            ]

        # ----------------------------------------------------
        # Otherwise combine both boxes
        # ----------------------------------------------------

        else:

            center_x = (
                labels[0][1] +
                labels[1][1]
            ) / 2.0

            center_y = (
                labels[0][2] +
                labels[1][2]
            ) / 2.0

            new_width = (
                labels[0][3] +
                labels[1][3] +
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
    # Original image dimensions
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
    # YOLO normalized coordinates
    # -> original image pixel coordinates
    # --------------------------------------------------------

    x1 = w_factor * (
        center_x -
        new_width / 2
    )

    y1 = h_factor * (
        center_y -
        new_height / 2
    )

    x2 = w_factor * (
        center_x +
        new_width / 2
    )

    y2 = h_factor * (
        center_y +
        new_height / 2
    )


    # --------------------------------------------------------
    # Adjust coordinates for padding
    # --------------------------------------------------------

    x1 += pad[1]
    y1 += pad[0]

    x2 += pad[1]
    y2 += pad[0]


    # --------------------------------------------------------
    # Same coordinate normalization as training
    # --------------------------------------------------------

    center_x = (
        ((x1 + x2) / 2.0)
        / padded_w
    )

    center_y = (
        ((y1 + y2) / 2.0)
        / padded_h
    )

    new_width = (
        (x2 - x1)
        / padded_w
    )

    new_height = (
        (y2 - y1)
        / padded_h
    )


    # --------------------------------------------------------
    # SAME EXPANSION AS TRAINING
    # --------------------------------------------------------

    if labels[0][0] != 10:

        new_width *= 3

        new_height *= 1.7


    # --------------------------------------------------------
    # Convert expanded normalized coordinates
    # back to pixels
    # --------------------------------------------------------

    x_min = (
        center_x * padded_w
        -
        new_width * padded_w / 2.0
    )

    y_min = (
        center_y * padded_h
        -
        new_height * padded_h / 2.0
    )

    x_max = (
        center_x * padded_w
        +
        new_width * padded_w / 2.0
    )

    y_max = (
        center_y * padded_h
        +
        new_height * padded_h / 2.0
    )


    # --------------------------------------------------------
    # SAME MARGINS AS TRAINING
    #
    # x_min -= 23
    # y_min -= 30
    # x_max += 30
    # y_max += 33
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
    # Check crop validity
    # --------------------------------------------------------

    if x_max <= x_min:

        return None, None

    if y_max <= y_min:

        return None, None


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

    if labels[0][0] == 10:

        jersey_number = 100

    elif len(digit_number) == 1:

        jersey_number = digit_number[0]

    else:

        jersey_number = (
            digit_number[0] * 10
            +
            digit_number[1]
        )


    return cropped_image, digit_number


# ============================================================
# CREATE 1D LABEL
# ============================================================

def create_1d_label(
    labels,
    digit_number
):

    """
    Converts:

        3 x y w h
        7 x y w h

    into:

        3 7
    """

    if labels[0][0] == 10:

        return "10"

    return " ".join(
        str(x)
        for x in digit_number
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    label,
    output_path
):

    """
    Saves the cropped image with the 1D label
    written on top.
    """

    visualization = image.copy()

    # Add white area at top
    header_height = 40

    canvas = cv2.copyMakeBorder(
        visualization,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    cv2.putText(
        canvas,
        f"Label: {label}",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        output_path,
        canvas
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    dataset_dir,
    output_dir,
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {dataset_name.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Input structure
    #
    # dataset_dir/
    # ├── images/
    # └── labels/
    # --------------------------------------------------------

    image_dir = os.path.join(
        dataset_dir,
        "images"
    )

    label_dir = os.path.join(
        dataset_dir,
        "labels"
    )


    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not os.path.isdir(image_dir):

        raise FileNotFoundError(
            f"\nImages directory does not exist:\n"
            f"{image_dir}"
        )

    if not os.path.isdir(label_dir):

        raise FileNotFoundError(
            f"\nLabels directory does not exist:\n"
            f"{label_dir}"
        )


    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = [

        f

        for f in os.listdir(image_dir)

        if f.endswith(IMAGE_EXTENSIONS)
    ]


    print(
        f"Images directory:\n{image_dir}"
    )

    print(
        f"Labels directory:\n{label_dir}"
    )

    print(
        f"Total images found: "
        f"{len(image_files)}"
    )


    if len(image_files) == 0:

        raise ValueError(
            f"\nNo images found in:\n"
            f"{image_dir}"
        )


    # --------------------------------------------------------
    # Select validation images
    # --------------------------------------------------------

    if len(image_files) > VAL_COUNT:

        random.shuffle(
            image_files
        )

        selected_images = image_files[
            :VAL_COUNT
        ]

    else:

        selected_images = image_files


    print(
        f"Images selected: "
        f"{len(selected_images)}"
    )


    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_image_dir = os.path.join(
        output_dir,
        "images"
    )

    output_label_dir = os.path.join(
        output_dir,
        "labels"
    )

    os.makedirs(
        output_image_dir,
        exist_ok=True
    )

    os.makedirs(
        output_label_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Visualization path
    # --------------------------------------------------------

    if args.visualize:

        visualization_dir = os.path.join(
            args.output_dir,
            "visualization",
            dataset_name
        )

        os.makedirs(
            visualization_dir,
            exist_ok=True
        )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    processed = 0
    missing_labels = 0
    invalid_images = 0
    empty_labels = 0


    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_name in enumerate(
        selected_images
    ):

        image_path = os.path.join(
            image_dir,
            image_name
        )

        label_name = (
            os.path.splitext(image_name)[0]
            +
            ".txt"
        )

        label_path = os.path.join(
            label_dir,
            label_name
        )


        # ----------------------------------------------------
        # Check label
        # ----------------------------------------------------

        if not os.path.exists(
            label_path
        ):

            print(
                f"WARNING: Missing label: "
                f"{label_name}"
            )

            missing_labels += 1

            continue


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"WARNING: Cannot read image: "
                f"{image_name}"
            )

            invalid_images += 1

            continue


        # ----------------------------------------------------
        # Read YOLO labels
        # ----------------------------------------------------

        labels = read_yolo_label(
            label_path
        )

        if len(labels) == 0:

            print(
                f"WARNING: Empty/invalid label: "
                f"{label_name}"
            )

            empty_labels += 1

            continue


        # ----------------------------------------------------
        # Crop using training logic
        # ----------------------------------------------------

        cropped_image, digit_number = calculate_crop(
            image,
            labels
        )

        if cropped_image is None:

            print(
                f"WARNING: Invalid crop: "
                f"{image_name}"
            )

            invalid_images += 1

            continue


        # ----------------------------------------------------
        # Create 1D label
        # ----------------------------------------------------

        new_label = create_1d_label(
            labels,
            digit_number
        )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_image_path = os.path.join(
            output_image_dir,
            image_name
        )

        cv2.imwrite(
            output_image_path,
            cropped_image
        )


        # ----------------------------------------------------
        # Save 1D label
        # ----------------------------------------------------

        output_label_path = os.path.join(
            output_label_dir,
            os.path.splitext(image_name)[0]
            +
            ".txt"
        )

        with open(
            output_label_path,
            "w"
        ) as f:

            f.write(new_label)


        processed += 1


        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        if (
            args.visualize
            and
            processed <= args.visualize_count
        ):

            visualization_path = os.path.join(
                visualization_dir,
                image_name
            )

            save_visualization(
                cropped_image,
                new_label,
                visualization_path
            )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % 100 == 0:

            print(
                f"Processed: "
                f"{processed}/"
                f"{len(selected_images)}"
            )


    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"{dataset_name.upper()} SUMMARY"
    )

    print("-" * 70)

    print(
        f"Selected images : "
        f"{len(selected_images)}"
    )

    print(
        f"Processed        : "
        f"{processed}"
    )

    print(
        f"Missing labels   : "
        f"{missing_labels}"
    )

    print(
        f"Invalid images   : "
        f"{invalid_images}"
    )

    print(
        f"Empty labels     : "
        f"{empty_labels}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "VALIDATION DATASET CROPPING"
    )

    print("=" * 70)

    print(
        f"Validation count : {VAL_COUNT}"
    )

    print(
        f"Random seed      : {RANDOM_SEED}"
    )

    print(
        "Crop logic       : Same as training"
    )

    print(
        "Width expansion  : x3"
    )

    print(
        "Height expansion : x1.7"
    )

    print(
        "Margins          : "
        "left=23, top=30, right=30, bottom=33"
    )


    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()


    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    real_output = os.path.join(
        args.output_dir,
        "validation_real"
    )

    process_dataset(
        "real",
        args.real_dir,
        real_output
    )


    # --------------------------------------------------------
    # SYNTHETIC
    # --------------------------------------------------------

    synthetic_output = os.path.join(
        args.output_dir,
        "validation_synthetic"
    )

    process_dataset(
        "synthetic",
        args.synthetic_dir,
        synthetic_output
    )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "VALIDATION CROPPING COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nOutput directory:\n"
        f"{args.output_dir}"
    )

    print(
        "\nStructure:"
    )

    print(
        f"""
{args.output_dir}/
│
├── validation_real/
│   ├── images/
│   └── labels/
│
├── validation_synthetic/
│   ├── images/
│   └── labels/
│
└── visualization/
    ├── real/
    └── synthetic/
"""
    )


if __name__ == "__main__":

    main()
