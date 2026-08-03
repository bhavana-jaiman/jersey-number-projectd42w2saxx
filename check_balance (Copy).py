"""
check_balance.py

Standalone class-balance checker for the jersey-number dataset.

It reads the YOLO-style .txt label files directly (same format used by
jerseyWithState_Dataset in dataset.py: rows of
    class_id  x_center  y_center  width  height
where class_id in [0-9] is a digit and class_id == 10 means "no visible
jersey number") and reproduces the same state/label logic used at
training time, WITHOUT touching any images. This makes it fast enough to
run over a full dataset in seconds.

It reports:
  - state distribution:
        0 = no jersey number visible
        1 = a clean, readable 1-2 digit jersey number
        2 = ambiguous / 3+ raw boxes (dropped / treated as invalid)
  - per-digit histogram (0-9), counted once per digit box in state==1 samples
  - whole jersey-number histogram (0-99), counted once per state==1 sample
  - jersey-number-length histogram (0, 1, 2) for state==1 samples

Usage:
    python check_balance.py --root /path/to/training_dataset_Ying
    python check_balance.py --root /path/to/dataset --outdir report
"""

import os
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def classify_label(label: np.ndarray):
    """
    Reproduce the state / digit-sequence logic from
    jerseyWithState_Dataset.__getitem__ in dataset.py, using only the raw
    label rows (before any image-based augmentation).

    Returns:
        state (int): 0, 1, or 2
        digits (list[int]): the digit sequence for state==1 samples
                             (1 or 2 ints), otherwise [] or a placeholder
        whole_number (int | None): combined 1-2 digit number for state==1
                                    samples, else None
    """
    raw_box_count = label.shape[0]

    if label[0][0] == 10:
        return 0, [], None

    if raw_box_count >= 3:
        return 2, [int(x) for x in label[:, 0]], None

    # raw_box_count is 1 or 2 here
    if raw_box_count == 1:
        digit = int(label[0][0])
        return 1, [digit], digit

    # raw_box_count == 2
    far_apart = (
        abs(label[0][1] - label[1][1]) > 0.25
        or abs(label[0][2] - label[1][2]) > 0.25
    )
    if far_apart:
        # Treated as a single stray digit box (matches dataset.py behaviour)
        digit = int(label[0][0])
        return 1, [digit], digit

    # Two boxes belonging to the same 2-digit jersey number
    sorted_indices = np.argsort(label[:, 1])
    digits = [int(label[i, 0]) for i in sorted_indices]
    whole_number = digits[0] * 10 + digits[1]
    return 1, digits, whole_number


def scan_dataset(root: str):
    label_files = [
        os.path.join(root, f) for f in os.listdir(root) if f.endswith(".txt")
    ]

    state_counts = Counter()
    digit_counts = Counter()
    length_counts = Counter()
    whole_number_counts = Counter()
    bad_files = []

    for path in label_files:
        try:
            if os.path.getsize(path) == 0:
                label = np.array([[10, 0, 0, 0, 0]], dtype=np.float32)
            else:
                label = np.loadtxt(path).reshape(-1, 5)
        except Exception as e:
            bad_files.append((path, str(e)))
            continue

        state, digits, whole_number = classify_label(label)
        state_counts[state] += 1

        if state == 1:
            length_counts[len(digits)] += 1
            for d in digits:
                digit_counts[d] += 1
            whole_number_counts[whole_number] += 1

    return {
        "num_files": len(label_files),
        "state_counts": state_counts,
        "digit_counts": digit_counts,
        "length_counts": length_counts,
        "whole_number_counts": whole_number_counts,
        "bad_files": bad_files,
    }


def print_report(stats):
    print(f"\nTotal label files scanned: {stats['num_files']}")
    if stats["bad_files"]:
        print(f"Files that failed to parse: {len(stats['bad_files'])}")
        for path, err in stats["bad_files"][:10]:
            print(f"  - {path}: {err}")
        if len(stats["bad_files"]) > 10:
            print(f"  ... and {len(stats['bad_files']) - 10} more")

    print("\n--- State distribution ---")
    state_names = {0: "state 0 (no number)", 1: "state 1 (valid 1-2 digit)", 2: "state 2 (ambiguous/3+ boxes)"}
    total = sum(stats["state_counts"].values()) or 1
    for state in sorted(stats["state_counts"]):
        count = stats["state_counts"][state]
        pct = 100 * count / total
        print(f"  {state_names.get(state, state)}: {count} ({pct:.1f}%)")

    print("\n--- Jersey-number length distribution (state==1 only) ---")
    total_len = sum(stats["length_counts"].values()) or 1
    for length in sorted(stats["length_counts"]):
        count = stats["length_counts"][length]
        pct = 100 * count / total_len
        print(f"  {length}-digit: {count} ({pct:.1f}%)")

    print("\n--- Per-digit histogram (0-9), counted per digit box, state==1 only ---")
    total_digits = sum(stats["digit_counts"].values()) or 1
    for digit in range(10):
        count = stats["digit_counts"].get(digit, 0)
        pct = 100 * count / total_digits
        print(f"  digit {digit}: {count} ({pct:.1f}%)")

    print("\n--- Whole jersey-number distribution (state==1 only) ---")
    numbers = stats["whole_number_counts"]
    if numbers:
        counts = list(numbers.values())
        print(f"  distinct jersey numbers seen: {len(numbers)}")
        print(f"  most common: {numbers.most_common(5)}")
        print(f"  least common: {sorted(numbers.items(), key=lambda kv: kv[1])[:5]}")
        print(f"  mean count per number: {np.mean(counts):.1f}, std: {np.std(counts):.1f}")
        print(f"  min: {min(counts)}, max: {max(counts)}")


def save_plots(stats, outdir):
    os.makedirs(outdir, exist_ok=True)

    # State distribution
    state_names = {0: "no number", 1: "valid 1-2 digit", 2: "ambiguous"}
    states = sorted(stats["state_counts"])
    plt.figure(figsize=(6, 4))
    plt.bar([state_names.get(s, s) for s in states], [stats["state_counts"][s] for s in states])
    plt.title("Sample count by state")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "state_distribution.png"))
    plt.close()

    # Digit histogram
    digits = list(range(10))
    plt.figure(figsize=(8, 4))
    plt.bar(digits, [stats["digit_counts"].get(d, 0) for d in digits])
    plt.title("Digit histogram (state==1 samples)")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.xticks(digits)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "digit_distribution.png"))
    plt.close()

    # Whole jersey number histogram
    numbers = stats["whole_number_counts"]
    if numbers:
        xs = sorted(numbers.keys())
        ys = [numbers[x] for x in xs]
        plt.figure(figsize=(12, 4))
        plt.bar(xs, ys, width=0.8)
        plt.title("Whole jersey-number distribution (state==1 samples)")
        plt.xlabel("Jersey number")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "whole_number_distribution.png"))
        plt.close()

    print(f"\nSaved plots to: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Check jersey-number dataset class balance")
    parser.add_argument("--root", required=True, help="Path to the folder containing .jpg/.txt pairs")
    parser.add_argument("--outdir", default="balance_report", help="Where to save plots")
    args = parser.parse_args()

    stats = scan_dataset(args.root)
    print_report(stats)
    save_plots(stats, args.outdir)


if __name__ == "__main__":
    main()
