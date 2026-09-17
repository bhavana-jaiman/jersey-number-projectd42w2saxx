import os
import csv
import argparse
from collections import Counter
from PIL import Image


def load_yolo_labels(label_path):
    """
    Read YOLO labels:
    class_id x_center y_center width height
    """

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            # Only digits 0-9
            if 0 <= cls <= 9:
                boxes.append({
                    "digit": str(cls),
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h
                })

    return boxes


def group_digits(boxes, y_tolerance=0.08, gap_factor=2.0):
    """
    Group digit bounding boxes belonging to the same jersey number.

    y_tolerance:
        Maximum normalized vertical-center difference.

    gap_factor:
        Maximum horizontal gap relative to average digit width.
    """

    if not boxes:
        return []

    # Sort approximately top-to-bottom
    boxes = sorted(boxes, key=lambda b: b["yc"])

    lines = []

    # ---------------------------------------------------------
    # STEP 1: Group boxes that are on approximately the same
    # horizontal line.
    # ---------------------------------------------------------
    for box in boxes:

        assigned = False

        for line in lines:

            avg_y = sum(b["yc"] for b in line) / len(line)

            if abs(box["yc"] - avg_y) <= y_tolerance:
                line.append(box)
                assigned = True
                break

        if not assigned:
            lines.append([box])

    groups = []

    # ---------------------------------------------------------
    # STEP 2: Within each line, sort digits left -> right
    # ---------------------------------------------------------
    for line in lines:

        line = sorted(line, key=lambda b: b["xc"])

        if len(line) == 1:
            groups.append(line)
            continue

        current_group = [line[0]]

        for i in range(1, len(line)):

            prev = line[i - 1]
            curr = line[i]

            # Right edge of previous box
            prev_right = prev["xc"] + prev["w"] / 2

            # Left edge of current box
            curr_left = curr["xc"] - curr["w"] / 2

            horizontal_gap = curr_left - prev_right

            # Average digit width
            avg_width = (prev["w"] + curr["w"]) / 2

            # If boxes overlap, definitely same group
            if horizontal_gap <= 0:
                same_number = True

            else:
                same_number = horizontal_gap <= gap_factor * avg_width

            if same_number:
                current_group.append(curr)

            else:
                groups.append(current_group)
                current_group = [curr]

        groups.append(current_group)

    return groups


def process_dataset(images_dir, labels_dir, output_csv,
                    y_tolerance=0.08,
                    gap_factor=2.0):

    results = []

    digit_count_distribution = Counter()
    number_distribution = Counter()

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images.")

    for index, image_name in enumerate(image_files, 1):

        image_path = os.path.join(images_dir, image_name)

        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        boxes = load_yolo_labels(label_path)

        if not boxes:
            continue

        groups = group_digits(
            boxes,
            y_tolerance=y_tolerance,
            gap_factor=gap_factor
        )

        for group_id, group in enumerate(groups, 1):

            # Sort digits left -> right
            group = sorted(group, key=lambda b: b["xc"])

            number = "".join(b["digit"] for b in group)

            digit_count = len(group)

            digit_count_distribution[digit_count] += 1
            number_distribution[number] += 1

            results.append({
                "image": image_name,
                "group_id": group_id,
                "number": number,
                "number_of_digits": digit_count,
                "digit_boxes": len(group)
            })

        if index % 1000 == 0:
            print(f"Processed {index}/{len(image_files)} images...")

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------

    with open(output_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "group_id",
                "number",
                "number_of_digits",
                "digit_boxes"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("JERSEY NUMBER DIGIT ANALYSIS")
    print("=" * 60)

    total_numbers = sum(digit_count_distribution.values())

    print(f"\nTotal detected jersey-number groups: {total_numbers}")

    print("\nDigit-count distribution:")

    for digits in sorted(digit_count_distribution):

        count = digit_count_distribution[digits]

        percentage = (
            count / total_numbers * 100
            if total_numbers > 0
            else 0
        )

        print(
            f"{digits}-digit numbers : "
            f"{count:6d} "
            f"({percentage:.2f}%)"
        )

    print("\n" + "-" * 60)
    print("Most common reconstructed numbers:")
    print("-" * 60)

    for number, count in number_distribution.most_common(30):

        print(
            f"{number:>5} : {count:6d}"
        )

    print("\nDetailed results saved to:")
    print(output_csv)


def main():

    parser = argparse.ArgumentParser(
        description="Group digit bounding boxes into jersey numbers."
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing images"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Directory containing YOLO label files"
    )

    parser.add_argument(
        "--output",
        default="jersey_number_digit_analysis.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--y_tolerance",
        type=float,
        default=0.08,
        help="Vertical grouping tolerance (default: 0.08)"
    )

    parser.add_argument(
        "--gap_factor",
        type=float,
        default=2.0,
        help="Maximum horizontal gap relative to digit width"
    )

    args = parser.parse_args()

    process_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_csv=args.output,
        y_tolerance=args.y_tolerance,
        gap_factor=args.gap_factor
    )


if __name__ == "__main__":
    main()
