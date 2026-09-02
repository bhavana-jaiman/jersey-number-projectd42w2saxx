import os
import cv2
import argparse


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "single_digit",
    1: "double_digit",
    # Change these according to your dataset
}


# ---------------------------------------------------------
# DRAW YOLO BOUNDING BOXES
# ---------------------------------------------------------

def draw_yolo_boxes(image, label_path):
    """
    Read YOLO-format labels and draw bounding boxes.

    YOLO format:
        class_id x_center y_center width height

    All coordinates are normalized between 0 and 1.
    """

    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            print(f"WARNING: Invalid label format in {label_path}")
            continue

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # ---------------------------------------------
        # Convert normalized YOLO coordinates
        # to pixel coordinates
        # ---------------------------------------------

        x_center *= w
        y_center *= h
        box_width *= w
        box_height *= h

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Keep coordinates inside image
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # ---------------------------------------------
        # Draw bounding box
        # ---------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ---------------------------------------------
        # Class name
        # ---------------------------------------------

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        label = f"{class_id}: {class_name}"

        # Text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        # Background rectangle for text
        cv2.rectangle(
            image,
            (x1, max(0, y1 - text_h - baseline - 5)),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            image,
            label,
            (x1, max(text_h, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return image


# ---------------------------------------------------------
# PROCESS DATASET
# ---------------------------------------------------------

def visualize_dataset(dataset_path, output_path):

    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]

    images.sort()

    print("\n--------------------------------------")
    print(f"Dataset: {dataset_path}")
    print(f"Images found: {len(images)}")
    print("--------------------------------------")

    processed = 0
    missing_labels = 0

    for image_name in images:

        image_path = os.path.join(
            image_dir,
            image_name
        )

        # Corresponding label
        label_name = os.path.splitext(image_name)[0] + ".txt"

        label_path = os.path.join(
            label_dir,
            label_name
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        if not os.path.exists(label_path):
            missing_labels += 1
            print(f"No label: {image_name}")

        # Draw boxes
        image = draw_yolo_boxes(
            image,
            label_path
        )

        # Save
        output_image_path = os.path.join(
            output_path,
            image_name
        )

        cv2.imwrite(
            output_image_path,
            image
        )

        processed += 1

    print("\nFinished!")
    print(f"Processed images : {processed}")
    print(f"Missing labels   : {missing_labels}")
    print(f"Output directory : {output_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes"
    )

    parser.add_argument(
        "--train",
        type=str,
        default="training_dataset_Ying",
        help="Path to training dataset"
    )

    parser.add_argument(
        "--val",
        type=str,
        default="validation_dataset_Ying",
        help="Path to validation dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="bbox_visualization",
        help="Output directory"
    )

    args = parser.parse_args()

    # -----------------------------------------
    # Training dataset
    # -----------------------------------------

    train_output = os.path.join(
        args.output,
        "training"
    )

    visualize_dataset(
        args.train,
        train_output
    )

    # -----------------------------------------
    # Validation dataset
    # -----------------------------------------

    val_output = os.path.join(
        args.output,
        "validation"
    )

    visualize_dataset(
        args.val,
        val_output
    )


if __name__ == "__main__":
    main()
