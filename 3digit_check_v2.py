import os
import csv
import argparse
from collections import Counter, defaultdict

import cv2


# ============================================================
# LOAD YOLO LABELS
# ============================================================

def load_yolo_labels(label_path):
    """
    YOLO format:

        class_id x_center y_center width height

    All coordinates are normalized [0, 1].

    Assumption:
        class_id 0-9 = digit 0-9
    """

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

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

            # Only digit classes
            if not 0 <= cls <= 9:
                continue

            x1 = xc - w / 2
            x2 = xc + w / 2

            y1 = yc - h / 2
            y2 = yc + h / 2

            boxes.append({
                "digit": str(cls),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2
            })

    return boxes


# ============================================================
# CHECK WHETHER TWO DIGITS CAN BELONG TO SAME NUMBER
# ============================================================

def compatible_boxes(
    previous,
    current,
    y_tolerance=0.05,
    gap_factor=1.0,
    height_tolerance=0.30,
    area_tolerance=3.0
):
    """
    Determine whether two digit boxes are likely part
    of the same jersey number.
    """

    # --------------------------------------------------------
    # 1. They should be vertically aligned
    # --------------------------------------------------------

    y_difference = abs(previous["yc"] - current["yc"])

    if y_difference > y_tolerance:
        return False

    # --------------------------------------------------------
    # 2. Current digit should be to the right
    # --------------------------------------------------------

    if current["xc"] <= previous["xc"]:
        return False

    # --------------------------------------------------------
    # 3. Calculate horizontal gap
    # --------------------------------------------------------

    horizontal_gap = current["x1"] - previous["x2"]

    # If boxes overlap, consider gap = 0
    horizontal_gap = max(0.0, horizontal_gap)

    # --------------------------------------------------------
    # 4. Gap relative to digit width
    # --------------------------------------------------------

    average_width = (
        previous["w"] + current["w"]
    ) / 2

    max_allowed_gap = gap_factor * average_width

    if horizontal_gap > max_allowed_gap:
        return False

    # --------------------------------------------------------
    # 5. Height similarity
    # --------------------------------------------------------

    max_height = max(previous["h"], current["h"])
    min_height = min(previous["h"], current["h"])

    if max_height == 0:
        return False

    height_difference_ratio = (
        max_height - min_height
    ) / max_height

    if height_difference_ratio > height_tolerance:
        return False

    # --------------------------------------------------------
    # 6. Area similarity
    #
    # Box area = width × height
    #
    # area_tolerance = maximum allowed ratio
    #
    # Example:
    # 3.0 means largest area can be up to
    # 3 times the smallest.
    # --------------------------------------------------------

    area_previous = previous["w"] * previous["h"]
    area_current = current["w"] * current["h"]

    if area_previous == 0 or area_current == 0:
        return False

    area_ratio = max(
        area_previous / area_current,
        area_current / area_previous
    )

    if area_ratio > area_tolerance:
        return False

    return True


# ============================================================
# GROUP DIGITS INTO JERSEY NUMBERS
# ============================================================

def group_digits(
    boxes,
    y_tolerance=0.05,
    gap_factor=1.0,
    height_tolerance=0.30,
    area_tolerance=3.0
):

    if not boxes:
        return []

    # --------------------------------------------------------
    # Sort top-to-bottom first
    # --------------------------------------------------------

    boxes = sorted(boxes, key=lambda b: b["yc"])

    # --------------------------------------------------------
    # STEP 1:
    # Group boxes that are on approximately the same row.
    # --------------------------------------------------------

    rows = []

    for box in boxes:

        assigned = False

        for row in rows:

            avg_y = sum(
                b["yc"] for b in row
            ) / len(row)

            if abs(box["yc"] - avg_y) <= y_tolerance:

                row.append(box)
                assigned = True
                break

        if not assigned:
            rows.append([box])

    # --------------------------------------------------------
    # STEP 2:
    # Within each row, sort left-to-right
    # --------------------------------------------------------

    all_groups = []

    for row in rows:

        row = sorted(
            row,
            key=lambda b: b["xc"]
        )

        if len(row) == 1:
            all_groups.append(row)
            continue

        current_group = [row[0]]

        for i in range(1, len(row)):

            previous = row[i - 1]
            current = row[i]

            same_number = compatible_boxes(
                previous,
                current,
                y_tolerance=y_tolerance,
                gap_factor=gap_factor,
                height_tolerance=height_tolerance,
                area_tolerance=area_tolerance
            )

            if same_number:

                current_group.append(current)

            else:

                all_groups.append(current_group)

                current_group = [current]

        all_groups.append(current_group)

    return all_groups


# ============================================================
# DRAW RESULTS ON IMAGE
# ============================================================

def draw_groups(image_path, groups, output_path):

    image = cv2.imread(image_path)

    if image is None:
        return

    height, width = image.shape[:2]

    for group_id, group in enumerate(groups, 1):

        # Sort left -> right
        group = sorted(
            group,
            key=lambda b: b["xc"]
        )

        number = "".join(
            b["digit"] for b in group
        )

        # Group bounding box
        x1 = min(b["x1"] for b in group)
        y1 = min(b["y1"] for b in group)

        x2 = max(b["x2"] for b in group)
        y2 = max(b["y2"] for b in group)

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # Draw individual digit boxes
        for box in group:

            bx1 = int(box["x1"] * width)
            by1 = int(box["y1"] * height)

            bx2 = int(box["x2"] * width)
            by2 = int(box["y2"] * height)

            cv2.rectangle(
                image,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                box["digit"],
                (bx1, max(20, by1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Draw complete-number box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            image,
            f"{number} ({len(group)} digit)",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imwrite(output_path, image)


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    images_dir,
    labels_dir,
    output_csv,
    three_digit_csv,
    visualization_dir=None,
    y_tolerance=0.05,
    gap_factor=1.0,
    height_tolerance=0.30,
    area_tolerance=3.0
):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower()
        in image_extensions
    ]

    print("=" * 70)
    print("JERSEY NUMBER DIGIT ANALYSIS")
    print("=" * 70)

    print(f"Images directory : {images_dir}")
    print(f"Labels directory : {labels_dir}")
    print(f"Images found     : {len(image_files)}")

    if visualization_dir:
        os.makedirs(
            visualization_dir,
            exist_ok=True
        )

    results = []

    digit_distribution = Counter()
    number_distribution = Counter()

    images_with_3_digit = set()

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, image_name in enumerate(
        image_files,
        1
    ):

        base_name = os.path.splitext(
            image_name
        )[0]

        label_path = os.path.join(
            labels_dir,
            base_name + ".txt"
        )

        boxes = load_yolo_labels(
            label_path
        )

        if not boxes:
            continue

        groups = group_digits(
            boxes,
            y_tolerance=y_tolerance,
            gap_factor=gap_factor,
            height_tolerance=height_tolerance,
            area_tolerance=area_tolerance
        )

        # ----------------------------------------------------
        # Process each reconstructed number
        # ----------------------------------------------------

        for group_id, group in enumerate(
            groups,
            1
        ):

            group = sorted(
                group,
                key=lambda b: b["xc"]
            )

            number = "".join(
                b["digit"] for b in group
            )

            digit_count = len(group)

            # ------------------------------------------------
            # Calculate statistics for this group
            # ------------------------------------------------

            widths = [
                b["w"]
                for b in group
            ]

            heights = [
                b["h"]
                for b in group
            ]

            areas = [
                b["w"] * b["h"]
                for b in group
            ]

            # Maximum horizontal gap
            gaps = []

            for i in range(
                len(group) - 1
            ):

                gap = (
                    group[i + 1]["x1"]
                    - group[i]["x2"]
                )

                gaps.append(
                    max(0.0, gap)
                )

            max_gap = (
                max(gaps)
                if gaps
                else 0.0
            )

            min_area = min(areas)
            max_area = max(areas)

            area_ratio = (
                max_area / min_area
                if min_area > 0
                else 0
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append({

                "image": image_name,

                "group_id": group_id,

                "number": number,

                "number_of_digits": digit_count,

                "digit_boxes": digit_count,

                "max_horizontal_gap": round(
                    max_gap,
                    6
                ),

                "min_box_area": round(
                    min_area,
                    6
                ),

                "max_box_area": round(
                    max_area,
                    6
                ),

                "area_ratio": round(
                    area_ratio,
                    3
                ),

                "avg_width": round(
                    sum(widths) / len(widths),
                    6
                ),

                "avg_height": round(
                    sum(heights) / len(heights),
                    6
                )
            })

            digit_distribution[
                digit_count
            ] += 1

            number_distribution[
                number
            ] += 1

            if digit_count == 3:

                images_with_3_digit.add(
                    image_name
                )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        if visualization_dir:

            output_image = os.path.join(
                visualization_dir,
                image_name
            )

            image_path = os.path.join(
                images_dir,
                image_name
            )

            draw_groups(
                image_path,
                groups,
                output_image
            )

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index}/{len(image_files)} images..."
            )

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    fieldnames = [
        "image",
        "group_id",
        "number",
        "number_of_digits",
        "digit_boxes",
        "max_horizontal_gap",
        "min_box_area",
        "max_box_area",
        "area_ratio",
        "avg_width",
        "avg_height"
    ]

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # SAVE ONLY 3-DIGIT NUMBERS
    # ========================================================

    three_digit_results = [
        r for r in results
        if r["number_of_digits"] == 3
    ]

    with open(
        three_digit_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            three_digit_results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_numbers = sum(
        digit_distribution.values()
    )

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal reconstructed jersey numbers: "
        f"{total_numbers}"
    )

    print(
        "\nDigit-count distribution:"
    )

    for digit_count in sorted(
        digit_distribution
    ):

        count = digit_distribution[
            digit_count
        ]

        percentage = (
            count / total_numbers * 100
            if total_numbers > 0
            else 0
        )

        print(
            f"{digit_count}-digit numbers : "
            f"{count:8d} "
            f"({percentage:6.2f}%)"
        )

    print("\n" + "-" * 70)

    print(
        f"3-digit number groups : "
        f"{len(three_digit_results)}"
    )

    print(
        f"Images containing 3-digit groups : "
        f"{len(images_with_3_digit)}"
    )

    print("\n" + "-" * 70)

    print(
        "Most common reconstructed numbers:"
    )

    for number, count in (
        number_distribution
        .most_common(30)
    ):

        print(
            f"{number:>6} : {count:8d}"
        )

    print("\n" + "-" * 70)

    print(
        "All results saved to:"
    )

    print(
        output_csv
    )

    print(
        "\n3-digit results saved to:"
    )

    print(
        three_digit_csv
    )

    if visualization_dir:

        print(
            "\nVisualizations saved to:"
        )

        print(
            visualization_dir
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze digit-level YOLO annotations "
            "and reconstruct jersey numbers."
        )
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing images"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="Directory containing YOLO labels"
    )

    parser.add_argument(
        "--output",
        default="jersey_number_analysis.csv",
        help="CSV containing all reconstructed numbers"
    )

    parser.add_argument(
        "--three_digit_output",
        default="three_digit_numbers.csv",
        help="CSV containing only 3-digit numbers"
    )

    parser.add_argument(
        "--visualize",
        default=None,
        help=(
            "Directory where annotated images "
            "will be saved"
        )
    )

    # --------------------------------------------------------
    # Grouping parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--gap_factor",
        type=float,
        default=1.0,
        help=(
            "Maximum horizontal gap as a multiple "
            "of average digit width. "
            "Default=1.0"
        )
    )

    parser.add_argument(
        "--y_tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum normalized Y-center difference. "
            "Default=0.05"
        )
    )

    parser.add_argument(
        "--height_tolerance",
        type=float,
        default=0.30,
        help=(
            "Maximum relative height difference. "
            "Default=0.30"
        )
    )

    parser.add_argument(
        "--area_tolerance",
        type=float,
        default=3.0,
        help=(
            "Maximum ratio between largest and "
            "smallest digit box area. "
            "Default=3.0"
        )
    )

    args = parser.parse_args()

    process_dataset(

        images_dir=args.images,

        labels_dir=args.labels,

        output_csv=args.output,

        three_digit_csv=args.three_digit_output,

        visualization_dir=args.visualize,

        y_tolerance=args.y_tolerance,

        gap_factor=args.gap_factor,

        height_tolerance=args.height_tolerance,

        area_tolerance=args.area_tolerance
    )


if __name__ == "__main__":
    main()
