"""
digit_position_counts.py

Computes per-DIGIT-POSITION counts (class 0-10 for digit1, class 0-10 for
digit2) directly from the label files -- exactly the two 11-element lists
loss_patched.py's make_loss_fn(digit1_counts=..., digit2_counts=...) needs.

This is DIFFERENT from class_counts.py, which reports the WHOLE jersey
number (0-99, e.g. "47" appeared 12 times). This script instead reports,
separately:
  - digit1 position: how many samples have a '4' in the first digit slot
  - digit2 position: how many samples have a '7' in the second digit slot
  - class 10 in either position = "digit absent" (single-digit numbers
    pad the missing second digit with class 10, matching how
    MultiTaskLearner's 11-way digit heads are set up)

Uses the same classify_label() logic as check_balance.py / dataset_audit.py
/ class_counts.py, so results are consistent with everything else you've
already run.

Usage:
    python digit_position_counts.py --root ./datasets/training_dataset_Ying

    # also print ready-to-paste Python list literals for loss_patched.py:
    python digit_position_counts.py --root ./datasets/training_dataset_Ying --print-python
"""

import os
import argparse
from collections import Counter

import numpy as np


def classify_label(label: np.ndarray):
    """Same logic as check_balance.py / dataset_audit.py / class_counts.py."""
    raw_box_count = label.shape[0]

    if label[0][0] == 10:
        return 0, [], None

    if raw_box_count >= 3:
        return 2, [int(x) for x in label[:, 0]], None

    if raw_box_count == 1:
        digit = int(label[0][0])
        return 1, [digit], digit

    far_apart = (
        abs(label[0][1] - label[1][1]) > 0.25
        or abs(label[0][2] - label[1][2]) > 0.25
    )
    if far_apart:
        digit = int(label[0][0])
        return 1, [digit], digit

    sorted_indices = np.argsort(label[:, 1])
    digits = [int(label[i, 0]) for i in sorted_indices]
    whole_number = digits[0] * 10 + digits[1]
    return 1, digits, whole_number


def load_label(path: str) -> np.ndarray:
    if os.path.getsize(path) == 0:
        return np.array([[10, 0, 0, 0, 0]], dtype=np.float32)
    return np.loadtxt(path).reshape(-1, 5).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Per-digit-position (digit1/digit2) class counts")
    parser.add_argument("--root", required=True)
    parser.add_argument("--print-python", action="store_true",
                         help="Also print ready-to-paste Python list literals")
    args = parser.parse_args()

    label_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".txt")]
    print(f"Found {len(label_files)} label files\n")

    digit1_counts = Counter()
    digit2_counts = Counter()
    bad_files = []
    n_state0, n_state1, n_state2 = 0, 0, 0

    for path in label_files:
        try:
            label = load_label(path)
            state, digits, whole_number = classify_label(label)
        except Exception as e:
            bad_files.append((path, str(e)))
            continue

        if state == 0:
            n_state0 += 1
            continue
        if state == 2:
            n_state2 += 1
            continue
        n_state1 += 1

        # digit1 is always present for a valid (state==1) sample
        digit1_counts[digits[0]] += 1

        if len(digits) == 2:
            digit2_counts[digits[1]] += 1
        else:
            # single-digit number -> digit2 slot is "absent" (class 10),
            # matching how MultiTaskLearner's digit_2 (11-way) head and
            # jersey_Dataset's padding-to-length-2 behavior already work
            digit2_counts[10] += 1

    if bad_files:
        print(f"⚠ {len(bad_files)} files failed to parse:")
        for p, err in bad_files[:5]:
            print(f"  - {p}: {err}")
        print()

    print(f"State 0 (no number): {n_state0}   State 1 (valid): {n_state1}   State 2 (ambiguous): {n_state2}\n")

    print(f"{'Class':>6} | {'Digit1 count':>13} | {'Digit1 %':>9} | {'Digit2 count':>13} | {'Digit2 %':>9}")
    print("-" * 62)
    d1_total = sum(digit1_counts.values()) or 1
    d2_total = sum(digit2_counts.values()) or 1
    d1_list = []
    d2_list = []
    for c in range(11):
        c1 = digit1_counts.get(c, 0)
        c2 = digit2_counts.get(c, 0)
        d1_list.append(c1)
        d2_list.append(c2)
        label = "absent" if c == 10 else str(c)
        print(f"{label:>6} | {c1:>13} | {100*c1/d1_total:>8.2f}% | {c2:>13} | {100*c2/d2_total:>8.2f}%")

    zero_d1 = [i for i, c in enumerate(d1_list) if c == 0]
    zero_d2 = [i for i, c in enumerate(d2_list) if c == 0]
    if zero_d1:
        print(f"\n⚠ digit1 classes with ZERO samples: {zero_d1}")
    if zero_d2:
        print(f"⚠ digit2 classes with ZERO samples: {zero_d2}")

    if args.print_python:
        print("\n--- Paste directly into loss_patched.py's make_loss_fn(...) call ---")
        print(f"digit1_counts = {d1_list}")
        print(f"digit2_counts = {d2_list}")
        print(
            "\nNote: calculate_weights() divides by each count, so a class with 0 "
            "samples would cause a divide-by-zero. If any class above shows 0, "
            "replace that 0 with 1 in the pasted list before using it (a small, "
            "harmless floor — there's no real data to weight correctly anyway)."
        )


if __name__ == "__main__":
    main()
