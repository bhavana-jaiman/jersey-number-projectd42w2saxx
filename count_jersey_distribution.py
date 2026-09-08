#!/usr/bin/env python3

import os
import csv
import argparse
from collections import Counter, defaultdict


# ============================================================
# LABEL READING
# ============================================================

def read_label_file(label_path):
    """
    Read YOLO-style label file.

    Expected format:
        class_id x_center y_center width height

    Example:
        2 0.45 0.50 0.10 0.20
        8 0.56 0.50 0.10 0.20

    Returns:
        list of dictionaries
    """

    objects = []

    if not os.path.isfile(label_path):
        return objects

    with open(label_path, "r") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[WARNING] Invalid line in {label_path} "
                    f"(line {line_number}): {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                print(
                    f"[WARNING] Could not parse line in {label_path} "
                    f"(line {line_number}): {line}"
                )
                continue

            objects.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    return objects


# ============================================================
# RECONSTRUCT NUMBER FROM DIGIT LABELS
# ============================================================

def reconstruct_numbers(objects):
    """
    Reconstruct jersey numbers from digit annotations.

    Single digit:
        7
        -> jersey number 7

    Double digit:
        2 ... 
        8 ...
        -> jersey number 28

    Digits are sorted from LEFT to RIGHT using x_center.

    Returns:
        list of reconstructed jersey numbers
    """

    if not objects:
        return []

    # Sort all digit boxes from left to right
    objects = sorted(objects, key=lambda x: x["x_center"])

    # --------------------------------------------------------
    # Important assumption:
    #
    # Each label file represents ONE jersey number.
    #
    # Therefore:
    #   1 object  -> single digit
    #   2 objects -> double digit
    #
    # If there are more than 2 objects, we handle it
    # conservatively by grouping them into pairs.
    # --------------------------------------------------------

    if len(objects) == 1:

        digit = objects[0]["class_id"]

        if 0 <= digit <= 9:
            return [digit]

        return []

    if len(objects) == 2:

        d1 = objects[0]["class_id"]
        d2 = objects[1]["class_id"]

        if 0 <= d1 <= 9 and 0 <= d2 <= 9:

            # No leading zero
            if d1 == 0:
                return [d2]

            number = d1 * 10 + d2

            if 10 <= number <= 99:
                return [number]

        return []

    # --------------------------------------------------------
    # More than 2 boxes
    # --------------------------------------------------------
    #
    # This can happen if an image/label file contains
    # multiple detected jersey-number objects.
    #
    # We group nearby boxes based on their x positions.
    # --------------------------------------------------------

    numbers = []

    i = 0

    while i < len(objects):

        current = objects[i]

        d1 = current["class_id"]

        if not (0 <= d1 <= 9):
            i += 1
            continue

        # Try to pair with next digit
        if i + 1 < len(objects):

            next_obj = objects[i + 1]

            d2 = next_obj["class_id"]

            if 0 <= d2 <= 9:

                number = d1 * 10 + d2

                if d1 != 0 and 10 <= number <= 99:
                    numbers.append(number)
                    i += 2
                    continue

        # Otherwise treat as single digit
        numbers.append(d1)

        i += 1

    return numbers


# ============================================================
# PROCESS DATASET
# ============================================================

def analyze_dataset(dataset_path):

    number_counts = Counter()
    digit_counts = Counter()

    single_digit_count = 0
    double_digit_count = 0

    image_count = 0
    label_count = 0

    image_details = []

    # --------------------------------------------------------
    # Find label directory
    # --------------------------------------------------------

    possible_label_dirs = [
        os.path.join(dataset_path, "labels"),
        dataset_path
    ]

    label_dir = None

    for candidate in possible_label_dirs:

        if os.path.isdir(candidate):

            # Check whether txt files exist
            txt_files = []

            for root, _, files in os.walk(candidate):

                for file in files:

                    if file.lower().endswith(".txt"):
                        txt_files.append(
                            os.path.join(root, file)
                        )

            if txt_files:

                label_dir = candidate
                break

    if label_dir is None:

        print(f"[WARNING] No label files found in: {dataset_path}")

        return {
            "number_counts": number_counts,
            "digit_counts": digit_counts,
            "single_digit_count": 0,
            "double_digit_count": 0,
            "image_count": 0,
            "label_count": 0,
            "image_details": []
        }

    # --------------------------------------------------------
    # Process all label files
    # --------------------------------------------------------

    label_files = []

    for root, _, files in os.walk(label_dir):

        for file in files:

            if file.lower().endswith(".txt"):

                label_files.append(
                    os.path.join(root, file)
                )

    label_files.sort()

    for label_path in label_files:

        label_count += 1

        objects = read_label_file(label_path)

        if not objects:
            continue

        image_count += 1

        # ====================================================
        # DIGIT-WISE COUNTING
        # ====================================================
        #
        # Every digit annotation contributes independently.
        #
        # Example:
        #
        # 28
        #
        # labels:
        #   2 ...
        #   8 ...
        #
        # contributes:
        #
        # 2 -> +1
        # 8 -> +1
        # ====================================================

        for obj in objects:

            digit = obj["class_id"]

            if 0 <= digit <= 9:
                digit_counts[digit] += 1

        # ====================================================
        # NUMBER-WISE COUNTING
        # ====================================================

        reconstructed = reconstruct_numbers(objects)

        for number in reconstructed:

            number_counts[number] += 1

            if 0 <= number <= 9:

                single_digit_count += 1

            elif 10 <= number <= 99:

                double_digit_count += 1

        # Save image-level information
        image_details.append({
            "label_file": label_path,
            "objects": len(objects),
            "numbers": reconstructed
        })

    return {
        "number_counts": number_counts,
        "digit_counts": digit_counts,
        "single_digit_count": single_digit_count,
        "double_digit_count": double_digit_count,
        "image_count": image_count,
        "label_count": label_count,
        "image_details": image_details
    }


# ============================================================
# PRINT NUMBER-WISE RESULTS
# ============================================================

def print_number_counts(title, results):

    print("\n")
    print("=" * 65)
    print(title)
    print("NUMBER-WISE COUNTING")
    print("=" * 65)

    print(f"{'Jersey Number':<18} {'Count':>10}")

    print("-" * 30)

    for number in range(100):

        count = results["number_counts"].get(number, 0)

        print(
            f"{number:<18} {count:>10}"
        )


# ============================================================
# PRINT DIGIT-WISE RESULTS
# ============================================================

def print_digit_counts(title, results):

    print("\n")
    print("=" * 65)
    print(title)
    print("DIGIT-WISE COUNTING")
    print("(Single + Double digit occurrences combined)")
    print("=" * 65)

    print(f"{'Digit':<18} {'Count':>10}")

    print("-" * 30)

    for digit in range(10):

        count = results["digit_counts"].get(digit, 0)

        print(
            f"{digit:<18} {count:>10}"
        )


# ============================================================
# PRINT SINGLE VS DOUBLE
# ============================================================

def print_single_double(title, results):

    single = results["single_digit_count"]
    double = results["double_digit_count"]

    total = single + double

    print("\n")
    print("=" * 65)
    print(title)
    print("SINGLE DIGIT vs DOUBLE DIGIT")
    print("=" * 65)

    if total > 0:

        single_percentage = (
            single / total
        ) * 100

        double_percentage = (
            double / total
        ) * 100

    else:

        single_percentage = 0
        double_percentage = 0

    print(
        f"Single digit (0-9)  : "
        f"{single:8d} "
        f"({single_percentage:6.2f}%)"
    )

    print(
        f"Double digit (10-99): "
        f"{double:8d} "
        f"({double_percentage:6.2f}%)"
    )

    print("-" * 40)

    print(
        f"Total                : "
        f"{total:8d}"
    )


# ============================================================
# SAVE NUMBER CSV
# ============================================================

def save_number_csv(results, output_path):

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "jersey_number",
            "count"
        ])

        for number in range(100):

            writer.writerow([
                number,
                results["number_counts"].get(number, 0)
            ])


# ============================================================
# SAVE DIGIT CSV
# ============================================================

def save_digit_csv(results, output_path):

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "digit",
            "count"
        ])

        for digit in range(10):

            writer.writerow([
                digit,
                results["digit_counts"].get(digit, 0)
            ])


# ============================================================
# SAVE IMAGE DETAILS
# ============================================================

def save_image_details(results, output_path):

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "label_file",
            "number_of_objects",
            "reconstructed_numbers"
        ])

        for item in results["image_details"]:

            writer.writerow([
                item["label_file"],
                item["objects"],
                ",".join(
                    str(x)
                    for x in item["numbers"]
                )
            ])


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze jersey number distribution "
            "for training_dataset_Ying and "
            "validation_dataset_Ying."
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to training_dataset_Ying"
    )

    parser.add_argument(
        "--val",
        required=True,
        help="Path to validation_dataset_Ying"
    )

    parser.add_argument(
        "--output",
        default="jersey_distribution_results",
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\nAnalyzing training dataset...")

    train_results = analyze_dataset(
        args.train
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\nAnalyzing validation dataset...")

    val_results = analyze_dataset(
        args.val
    )

    # ========================================================
    # COMBINED
    # ========================================================

    combined_number_counts = (
        train_results["number_counts"]
        +
        val_results["number_counts"]
    )

    combined_digit_counts = (
        train_results["digit_counts"]
        +
        val_results["digit_counts"]
    )

    combined_results = {

        "number_counts":
            combined_number_counts,

        "digit_counts":
            combined_digit_counts,

        "single_digit_count":
            train_results["single_digit_count"]
            +
            val_results["single_digit_count"],

        "double_digit_count":
            train_results["double_digit_count"]
            +
            val_results["double_digit_count"],

        "image_count":
            train_results["image_count"]
            +
            val_results["image_count"],

        "label_count":
            train_results["label_count"]
            +
            val_results["label_count"],

        "image_details":
            train_results["image_details"]
            +
            val_results["image_details"]
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n\n")
    print("#" * 70)
    print("# TRAINING DATASET")
    print("#" * 70)

    print(
        f"\nLabel files : "
        f"{train_results['label_count']}"
    )

    print(
        f"Non-empty  : "
        f"{train_results['image_count']}"
    )

    print_number_counts(
        "TRAINING DATASET",
        train_results
    )

    print_digit_counts(
        "TRAINING DATASET",
        train_results
    )

    print_single_double(
        "TRAINING DATASET",
        train_results
    )

    # ========================================================

    print("\n\n")
    print("#" * 70)
    print("# VALIDATION DATASET")
    print("#" * 70)

    print(
        f"\nLabel files : "
        f"{val_results['label_count']}"
    )

    print(
        f"Non-empty  : "
        f"{val_results['image_count']}"
    )

    print_number_counts(
        "VALIDATION DATASET",
        val_results
    )

    print_digit_counts(
        "VALIDATION DATASET",
        val_results
    )

    print_single_double(
        "VALIDATION DATASET",
        val_results
    )

    # ========================================================

    print("\n\n")
    print("#" * 70)
    print("# COMBINED DATASET")
    print("#" * 70)

    print_number_counts(
        "TRAIN + VALIDATION",
        combined_results
    )

    print_digit_counts(
        "TRAIN + VALIDATION",
        combined_results
    )

    print_single_double(
        "TRAIN + VALIDATION",
        combined_results
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    save_number_csv(
        train_results,
        os.path.join(
            args.output,
            "train_number_counts.csv"
        )
    )

    save_number_csv(
        val_results,
        os.path.join(
            args.output,
            "validation_number_counts.csv"
        )
    )

    save_number_csv(
        combined_results,
        os.path.join(
            args.output,
            "combined_number_counts.csv"
        )
    )

    save_digit_csv(
        train_results,
        os.path.join(
            args.output,
            "train_digit_counts.csv"
        )
    )

    save_digit_csv(
        val_results,
        os.path.join(
            args.output,
            "validation_digit_counts.csv"
        )
    )

    save_digit_csv(
        combined_results,
        os.path.join(
            args.output,
            "combined_digit_counts.csv"
        )
    )

    save_image_details(
        train_results,
        os.path.join(
            args.output,
            "train_image_details.csv"
        )
    )

    save_image_details(
        val_results,
        os.path.join(
            args.output,
            "validation_image_details.csv"
        )
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{os.path.abspath(args.output)}"
    )

    print("\nGenerated files:")

    for file in sorted(
        os.listdir(args.output)
    ):

        print(
            f"  {file}"
        )


if __name__ == "__main__":
    main()
