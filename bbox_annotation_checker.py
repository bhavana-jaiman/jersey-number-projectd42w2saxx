#!/usr/bin/env python3

import os
import cv2
import argparse


# ============================================================
# YOLO BOUNDING-BOX VISUALIZER
# ============================================================
#
# Supports BOTH dataset structures:
#
# 1) Images and labels in subfolders:
#
#    dataset/
#       images/
#          001.jpg
#          002.jpg
#       labels/
#          001.txt
#          002.txt
#
# 2) Images and labels directly in the same folder:
#
#    dataset/
#       001.jpg
#       001.txt
#       002.jpg
#       002.txt
#
# Expected annotation format:
#
#    class_id x_center y_center width height
#
# Example:
#
#    4 0.821226 0.946736 0.040894 0.077151
#
# The class ID is displayed, but it is NOT used for calculating
# the bounding-box position.
# ============================================================


# You can add/change names if you know your class mapping.
# Unknown class IDs will automatically be shown as "class_X".
CLASS_NAMES = {
    # 0: "single_digit",
    # 1: "double_digit",
    # 4: "your_class_name",
}


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ============================================================
# FIND DATASET DIRECTORIES
# ============================================================

def find_dataset_dirs(dataset_path):
    """
    Automatically supports:

    dataset/
        images/
        labels/

    OR:

    dataset/
        image.jpg
        image.txt
    """

    image_subdir = os.path.join(dataset_path, "images")
    label_subdir = os.path.join(dataset_path, "labels")

    if os.path.isdir(image_subdir) and os.path.isdir(label_subdir):
        return image_subdir, label_subdir, "images_labels_subfolders"

    # Flat dataset:
    # images and labels are directly inside dataset_path
    return dataset_path, dataset_path, "flat_dataset"


# ============================================================
# DRAW YOLO BOXES
# ============================================================

def draw_yolo_boxes(image, label_path, image_name=""):
    """
    Draw all YOLO bounding boxes on an image.

    YOLO:
        class_id x_center y_center width height

    x_center, y_center, width, height are normalized [0, 1].
    """

    image_height, image_width = image.shape[:2]

    if not os.path.isfile(label_path):
        return image, 0, 0

    boxes_drawn = 0
    invalid_boxes = 0

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) < 5:
            print(
                f"WARNING: Invalid label format | "
                f"{label_path} | line {line_number}"
            )
            invalid_boxes += 1
            continue

        try:
            # Keep class ID for displaying the annotation.
            # It does NOT affect box position.
            class_id = int(float(values[0]))

            x_center = float(values[1])
            y_center = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

        except ValueError:
            print(
                f"WARNING: Non-numeric label | "
                f"{label_path} | line {line_number}"
            )
            invalid_boxes += 1
            continue

        # ----------------------------------------------------
        # IMPORTANT VALIDATION
        # ----------------------------------------------------
        # For standard YOLO format, all these values should
        # normally be between 0 and 1.
        #
        # We DO NOT silently convert bad values because doing
        # so can make a wrong annotation look correct.
        # ----------------------------------------------------

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 <= box_width <= 1.0
            and 0.0 <= box_height <= 1.0
        ):
            print(
                f"WARNING: Coordinates outside YOLO [0,1] range | "
                f"{image_name} | line {line_number}: {line}"
            )
            invalid_boxes += 1
            continue

        # ----------------------------------------------------
        # YOLO normalized coordinates -> pixel coordinates
        # ----------------------------------------------------

        px_center_x = x_center * image_width
        px_center_y = y_center * image_height

        px_width = box_width * image_width
        px_height = box_height * image_height

        x1 = int(round(px_center_x - px_width / 2))
        y1 = int(round(px_center_y - px_height / 2))

        x2 = int(round(px_center_x + px_width / 2))
        y2 = int(round(px_center_y + px_height / 2))

        # ----------------------------------------------------
        # Clip ONLY to image boundaries
        # ----------------------------------------------------

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))

        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        # Ignore zero/negative boxes
        if x2 <= x1 or y2 <= y1:
            print(
                f"WARNING: Invalid/zero-size bounding box | "
                f"{image_name} | line {line_number}: {line}"
            )
            invalid_boxes += 1
            continue

        # ----------------------------------------------------
        # DRAW BOX
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # CLASS TEXT
        # ----------------------------------------------------

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        label_text = f"{class_id}: {class_name}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )

        # Put the label ABOVE the box when possible.
        # If the box is at the top edge, put text INSIDE/BELOW
        # the top edge instead of creating a strange rectangle.
        if y1 - text_height - baseline - 6 >= 0:
            text_x = x1
            text_y = y1 - 5

            bg_x1 = x1
            bg_y1 = y1 - text_height - baseline - 8
            bg_x2 = x1 + text_width + 8
            bg_y2 = y1

        else:
            text_x = x1 + 3
            text_y = min(
                image_height - 3,
                y1 + text_height + 3
            )

            bg_x1 = x1
            bg_y1 = y1
            bg_x2 = min(
                image_width - 1,
                x1 + text_width + 8
            )
            bg_y2 = min(
                image_height - 1,
                y1 + text_height + baseline + 8
            )

        # Text background
        cv2.rectangle(
            image,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 255, 0),
            -1,
        )

        # Text
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        boxes_drawn += 1

    return image, boxes_drawn, invalid_boxes


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def visualize_dataset(dataset_path, output_path):

    image_dir, label_dir, structure = find_dataset_dirs(dataset_path)

    if not os.path.isdir(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return

    if structure == "images_labels_subfolders":
        print("Dataset structure: images/ + labels/")
    else:
        print("Dataset structure: images and labels directly in dataset folder")

    print(f"Dataset       : {dataset_path}")
    print(f"Image folder  : {image_dir}")
    print(f"Label folder  : {label_dir}")
    print(f"Output folder : {output_path}")
    print("-" * 70)

    os.makedirs(output_path, exist_ok=True)

    image_names = sorted(
        [
            filename
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ]
    )

    print(f"Images found  : {len(image_names)}")
    print()

    processed = 0
    missing_labels = 0
    unreadable_images = 0
    total_boxes = 0
    total_invalid_boxes = 0

    for image_name in image_names:

        image_path = os.path.join(image_dir, image_name)

        # Match label using EXACT image basename.
        # Example:
        #
        # 44_00039.jpg -> 44_00039.txt
        #
        base_name = os.path.splitext(image_name)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(label_dir, label_name)

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            unreadable_images += 1
            print(f"WARNING: Could not read image: {image_path}")
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Find matching label
        # ----------------------------------------------------

        if not os.path.isfile(label_path):
            missing_labels += 1
            print(f"WARNING: No matching label: {image_name}")
            continue

        # ----------------------------------------------------
        # Draw annotation
        # ----------------------------------------------------

        visualized_image, boxes, invalid = draw_yolo_boxes(
            image,
            label_path,
            image_name,
        )

        total_boxes += boxes
        total_invalid_boxes += invalid

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        output_image_path = os.path.join(
            output_path,
            image_name
        )

        success = cv2.imwrite(
            output_image_path,
            visualized_image
        )

        if not success:
            print(
                f"WARNING: Could not save: "
                f"{output_image_path}"
            )
            continue

        processed += 1

        # Print useful information for each image
        print(
            f"[{processed}/{len(image_names)}] "
            f"{image_name} | "
            f"size={image_width}x{image_height} | "
            f"boxes={boxes}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"Dataset              : {dataset_path}")
    print(f"Images found         : {len(image_names)}")
    print(f"Processed images     : {processed}")
    print(f"Missing labels       : {missing_labels}")
    print(f"Unreadable images    : {unreadable_images}")
    print(f"Bounding boxes drawn : {total_boxes}")
    print(f"Invalid boxes        : {total_invalid_boxes}")
    print(f"Output directory     : {output_path}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize YOLO bounding-box annotations. "
            "Supports both images/labels subfolders and "
            "flat datasets."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset to visualize"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="bbox_visualization",
        help="Directory where visualized images will be saved"
    )

    args = parser.parse_args()

    visualize_dataset(
        args.dataset,
        args.output
    )


if __name__ == "__main__":
    main()
