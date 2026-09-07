import os
import glob
import numpy as np


THRESHOLD = 0.25


def check_dataset(dataset_name, root):
    label_files = sorted(glob.glob(os.path.join(root, "*.txt")))

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print(f"PATH: {root}")
    print("=" * 80)

    total_files = 0
    one_box = 0
    two_boxes = 0
    three_plus = 0

    double_digit = 0
    single_digit_by_distance = 0

    distances_x = []
    distances_y = []
    bbox_gaps = []

    examples = []

    for label_path in label_files:
        total_files += 1

        try:
            label = np.loadtxt(label_path, delimiter=" ", dtype=np.float32)
        except Exception:
            continue

        label = np.atleast_2d(label)

        if label.shape[1] < 5:
            continue

        raw_count = label.shape[0]

        if raw_count == 1:
            one_box += 1
            continue

        if raw_count >= 3:
            three_plus += 1
            continue

        # Exactly 2 bounding boxes
        two_boxes += 1

        # Sort from left to right
        order = np.argsort(label[:, 1])
        label = label[order]

        # YOLO format:
        # [class, center_x, center_y, width, height]

        x1 = label[0][1]
        y1 = label[0][2]
        w1 = label[0][3]
        h1 = label[0][4]

        x2 = label[1][1]
        y2 = label[1][2]
        w2 = label[1][3]
        h2 = label[1][4]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Horizontal gap between the two bounding boxes
        gap = dx - (w1 + w2) / 2

        distances_x.append(dx)
        distances_y.append(dy)
        bbox_gaps.append(gap)

        # This is exactly the logic currently used in your code
        if dx > THRESHOLD or dy > THRESHOLD:
            single_digit_by_distance += 1

            if len(examples) < 20:
                examples.append(
                    (
                        os.path.basename(label_path),
                        dx,
                        dy,
                        gap,
                        int(label[0][0]),
                        int(label[1][0])
                    )
                )
        else:
            double_digit += 1

    print(f"Total label files       : {total_files}")
    print(f"1 bounding box          : {one_box}")
    print(f"2 bounding boxes        : {two_boxes}")
    print(f"3+ bounding boxes       : {three_plus}")

    print()
    print(f"DOUBLE according to threshold : {double_digit}")
    print(f"SINGLE + 10 according to threshold : {single_digit_by_distance}")

    if distances_x:
        print()
        print("Horizontal center distance (ΔX)")
        print(f"  Min    : {min(distances_x):.4f}")
        print(f"  Max    : {max(distances_x):.4f}")
        print(f"  Mean   : {np.mean(distances_x):.4f}")
        print(f"  Median : {np.median(distances_x):.4f}")

        print()
        print("Vertical center distance (ΔY)")
        print(f"  Min    : {min(distances_y):.4f}")
        print(f"  Max    : {max(distances_y):.4f}")
        print(f"  Mean   : {np.mean(distances_y):.4f}")
        print(f"  Median : {np.median(distances_y):.4f}")

        print()
        print("Bounding-box horizontal gap")
        print(f"  Min    : {min(bbox_gaps):.4f}")
        print(f"  Max    : {max(bbox_gaps):.4f}")
        print(f"  Mean   : {np.mean(bbox_gaps):.4f}")
        print(f"  Median : {np.median(bbox_gaps):.4f}")

        print()
        print("Examples classified as SINGLE + 10:")
        print("-" * 80)

        for name, dx, dy, gap, d1, d2 in examples:
            print(
                f"{name:30s} "
                f"digits={d1},{d2}  "
                f"dx={dx:.4f}  "
                f"dy={dy:.4f}  "
                f"gap={gap:.4f}"
            )


datasets = {
    "training_dataset_Ying":
        "/home/ying/Desktop/Dataset/jersey_number_detection/training_dataset_Ying",

    "validation_dataset_Ying":
        "/home/ying/Desktop/Dataset/jersey_number_detection/validation_dataset_Ying",

    "simple2D_train":
        "/home/ying/Desktop/Dataset/jersey_number_detection/simple2D_train",

    "simple2D_validation":
        "/home/ying/Desktop/Dataset/jersey_number_detection/simple2D_validation",
}


for name, path in datasets.items():
    if os.path.isdir(path):
        check_dataset(name, path)
    else:
        print(f"\nWARNING: Dataset not found: {path}")
