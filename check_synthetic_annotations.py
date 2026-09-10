#!/usr/bin/env python3

import os
import cv2
import random
import argparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_labels(label_path):
    """
    Read YOLO labels.

    Format:
        class_id x_center y_center width height

    All coordinates are normalized [0, 1].
    """
    labels = []

    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"[WARNING] Invalid label format: "
                    f"{label_path}, line {line_no}: {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(
                    f"[WARNING] Cannot parse: "
                    f"{label_path}, line {line_no}: {line}"
                )
                continue

            labels.append((class_id, xc, yc, w, h))

    return labels


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    Convert normalized YOLO coordinates to pixel coordinates.
    """

    x_center = xc * img_w
    y_center = yc * img_h

    box_w = w * img_w
    box_h = h * img_h

    x1 = int(round(x_center - box_w / 2))
    y1 = int(round(y_center - box_h / 2))

    x2 = int(round(x_center + box_w / 2))
    y2 = int(round(y_center + box_h / 2))

    return x1, y1, x2, y2


def draw_annotations(image, labels):
    """
    Draw YOLO annotations on image.
    """

    output = image.copy()

    img_h, img_w = output.shape[:2]

    for idx, (class_id, xc, yc, w, h) in enumerate(labels):

        x1, y1, x2, y2 = yolo_to_xyxy(
            xc, yc, w, h,
            img_w,
            img_h
        )

        # Clip to image boundaries
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label text
        text = f"class={class_id}"

        text_x = x1
        text_y = max(20, y1 - 7)

        cv2.putText(
            output,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Center point
        cx = int(round(xc * img_w))
        cy = int(round(yc * img_h))

        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

    return output


def create_side_by_side(original, annotated):
    """
    Put original and annotated image side-by-side.
    """

    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]

    # Resize annotated to original height if necessary
    if h1 != h2:
        scale = h1 / h2
        annotated = cv2.resize(
            annotated,
            (int(w2 * scale), h1)
        )

    separator = 5

    separator_img = 255 * (
        cv2.UMat(
            h1,
            separator,
            cv2.CV_8UC3
        ).get()
    )

    combined = cv2.hconcat([
        original,
        separator_img,
        annotated
    ])

    return combined


def find_images(image_dir):
    """
    Recursively find images.
    """

    image_files = []

    for root, dirs, files in os.walk(image_dir):

        for filename in files:

            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:

                image_files.append(
                    os.path.join(root, filename)
                )

    return sorted(image_files)


def main():

    parser = argparse.ArgumentParser(
        description="Visualize YOLO annotations for synthetic jersey dataset."
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing synthetic images"
    )

    parser.add_argument(
        "--label-dir",
        default=None,
        help=(
            "Directory containing labels. "
            "If omitted, labels are searched next to each image."
        )
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images to visualize"
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Select random images"
    )

    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save visualization images"
    )

    args = parser.parse_args()

    image_files = find_images(args.image_dir)

    if len(image_files) == 0:
        print("[ERROR] No images found.")
        return

    print(f"\nFound {len(image_files)} images.")

    # Select images
    if args.random:

        count = min(args.num_images, len(image_files))

        selected_images = random.sample(
            image_files,
            count
        )

    else:

        selected_images = image_files[
            :args.num_images
        ]

    print(
        f"Visualizing {len(selected_images)} images...\n"
    )

    missing_labels = 0
    invalid_boxes = 0

    if args.save_dir:
        os.makedirs(
            args.save_dir,
            exist_ok=True
        )

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        image = cv2.imread(image_path)

        if image is None:
            print(
                f"[WARNING] Could not read image: "
                f"{image_path}"
            )
            continue

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        if args.label_dir:

            label_path = os.path.join(
                args.label_dir,
                base_name + ".txt"
            )

        else:

            label_path = os.path.splitext(
                image_path
            )[0] + ".txt"

        if not os.path.exists(label_path):

            print(
                f"[MISSING LABEL] {label_path}"
            )

            missing_labels += 1
            continue

        labels = load_yolo_labels(
            label_path
        )

        img_h, img_w = image.shape[:2]

        # Validate labels
        for class_id, xc, yc, w, h in labels:

            if not (
                0 <= xc <= 1 and
                0 <= yc <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):
                invalid_boxes += 1

        annotated = draw_annotations(
            image,
            labels
        )

        combined = create_side_by_side(
            image,
            annotated
        )

        # Add header
        header_height = 45

        header = (
            255 *
            cv2.UMat(
                header_height,
                combined.shape[1],
                cv2.CV_8UC3
            ).get()
        )

        header_text = (
            f"{index}/{len(selected_images)}  "
            f"{os.path.basename(image_path)}  |  "
            f"Annotations: {len(labels)}"
        )

        cv2.putText(
            header,
            header_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        display = cv2.vconcat([
            header,
            combined
        ])

        # Print annotation values
        print("=" * 70)
        print(f"IMAGE : {image_path}")
        print(f"LABEL : {label_path}")
        print(
            f"SIZE  : {img_w} x {img_h}"
        )

        for i, label in enumerate(labels):

            class_id, xc, yc, w, h = label

            x1, y1, x2, y2 = yolo_to_xyxy(
                xc, yc, w, h,
                img_w,
                img_h
            )

            print(
                f"  Box {i+1}: "
                f"class={class_id}, "
                f"YOLO=({xc:.6f}, {yc:.6f}, "
                f"{w:.6f}, {h:.6f}), "
                f"PIXEL=({x1}, {y1}, {x2}, {y2})"
            )

        # Save
        if args.save_dir:

            save_path = os.path.join(
                args.save_dir,
                base_name + "_check.jpg"
            )

            cv2.imwrite(
                save_path,
                display
            )

            print(
                f"Saved: {save_path}"
            )

        # Display
        window_name = (
            "Synthetic Annotation Check - "
            "Press Q to quit / SPACE for next"
        )

        cv2.imshow(
            window_name,
            display
        )

        while True:

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            if key == 32:
                break

            if key == 27:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Images checked       : {len(selected_images)}"
    )

    print(
        f"Missing labels       : {missing_labels}"
    )

    print(
        f"Invalid annotations  : {invalid_boxes}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
