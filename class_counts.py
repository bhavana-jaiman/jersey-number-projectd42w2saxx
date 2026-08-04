"""
class_counts.py  (v2)

Reports how many samples exist for EACH whole jersey-number class (0-99)
in a dataset folder (e.g. training_dataset_Ying), reading directly from
the label files next to each image.

Auto-detects the label format present in --root:
    - .txt files   -> classic YOLO rows: "class_id x_center y_center width height"
    - .json files  -> same YOLO-style data, but wrapped in JSON. Several
                      common JSON shapes are supported automatically (see
                      `load_label_json` below). If your JSON has a
                      different shape, run with --inspect first to see the
                      raw structure and adjust `load_label_json` accordingly.

By default this reports ALL classes 0-99 (not just a handful) — sorted by
class number so you can scan top-to-bottom for gaps, or use --sort count
to see the busiest classes first.

Usage:
    # First, if you're not 100% sure of the label format on disk:
    python class_counts.py --root /path/to/training_dataset_Ying --inspect

    # Full table, every class 0-99, auto-detects .txt vs .json
    python class_counts.py --root /path/to/training_dataset_Ying

    # Force a specific format instead of auto-detecting
    python class_counts.py --root /path/to/training_dataset_Ying --format json

    # Busiest classes first, and save to CSV
    python class_counts.py --root /path/to/training_dataset_Ying --sort count --csv counts.csv

    # Still supported: only specific classes
    python class_counts.py --root /path/to/training_dataset_Ying --classes 2,56,78,66,77,99
"""

import os
import json
import argparse
import csv
import glob
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Label loading — supports .txt (classic YOLO) and .json (YOLO data in JSON)
# ---------------------------------------------------------------------------

def load_label_txt(path: str) -> np.ndarray:
    if os.path.getsize(path) == 0:
        return np.array([[10, 0, 0, 0, 0]], dtype=np.float32)
    return np.loadtxt(path).reshape(-1, 5).astype(np.float32)


def load_label_json(path: str) -> np.ndarray:
    """
    Attempts to read YOLO-style rows out of a JSON label file. Handles a
    few common shapes:

      1. A bare list of [class_id, x_center, y_center, width, height]:
         [[7, 0.3, 0.5, 0.1, 0.2], [5, 0.6, 0.5, 0.1, 0.2]]

      2. A list of dicts with flexible key names:
         [{"class_id": 7, "x_center": 0.3, "y_center": 0.5, "width": 0.1, "height": 0.2}, ...]
         (also accepts "class"/"cls" and "x"/"y"/"w"/"h" as key aliases)

      3. A dict wrapping the list under a common key:
         {"boxes": [...]} or {"labels": [...]} or {"annotations": [...]}

    If your JSON uses a different shape, run with --inspect to see a raw
    sample and adjust this function to match.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Shape 3: unwrap a dict wrapper
    if isinstance(data, dict):
        for key in ("boxes", "labels", "annotations", "objects", "data"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list) or len(data) == 0:
        # Empty / no boxes -> treat as "no visible jersey number"
        return np.array([[10, 0, 0, 0, 0]], dtype=np.float32)

    rows = []
    for item in data:
        if isinstance(item, (list, tuple)):
            # Shape 1
            cls, x, y, w, h = item[:5]
        elif isinstance(item, dict):
            # Shape 2, with flexible key aliases
            cls = item.get("class_id", item.get("class", item.get("cls")))
            x = item.get("x_center", item.get("x"))
            y = item.get("y_center", item.get("y"))
            w = item.get("width", item.get("w"))
            h = item.get("height", item.get("h"))
            if None in (cls, x, y, w, h):
                raise ValueError(f"Unrecognized JSON box shape in {path}: {item}")
        else:
            raise ValueError(f"Unrecognized JSON structure in {path}: {item}")
        rows.append([float(cls), float(x), float(y), float(w), float(h)])

    return np.array(rows, dtype=np.float32)


def detect_format(root: str) -> str:
    n_txt = len(glob.glob(os.path.join(root, "*.txt")))
    n_json = len(glob.glob(os.path.join(root, "*.json")))
    print(f"Detected in {root}:  {n_txt} .txt files,  {n_json} .json files")
    if n_txt == 0 and n_json == 0:
        raise SystemExit(
            "No .txt or .json label files found directly in --root. "
            "Check the path, or whether labels live in a subfolder (e.g. labels/)."
        )
    if n_txt >= n_json:
        return "txt"
    return "json"


def find_label_files(root: str, fmt: str):
    ext = ".txt" if fmt == "txt" else ".json"
    return [os.path.join(root, f) for f in os.listdir(root) if f.endswith(ext)]


def load_label(path: str, fmt: str) -> np.ndarray:
    return load_label_txt(path) if fmt == "txt" else load_label_json(path)


def inspect_samples(root: str, fmt: str, n: int = 3):
    files = find_label_files(root, fmt)
    print(f"\n--- Raw content of {min(n, len(files))} sample {fmt} files ---")
    for path in files[:n]:
        print(f"\n{path}")
        with open(path, "r") as f:
            content = f.read()
        print(content[:800] + ("... (truncated)" if len(content) > 800 else ""))
    print(
        "\nCompare this to the shapes documented at the top of load_label_json() "
        "if using JSON. If it doesn't match, the script will raise a clear error "
        "on that file when you run the full scan (rather than silently miscounting)."
    )


# ---------------------------------------------------------------------------
# Same classify_label logic as check_balance.py / dataset_audit.py, kept in
# sync deliberately so numbers always agree across all scripts.
# ---------------------------------------------------------------------------

def classify_label(label: np.ndarray):
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


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_whole_numbers(label_files, fmt: str):
    whole_number_counts = Counter()
    bad_files = []
    n_state0 = 0
    n_state2 = 0

    for path in label_files:
        try:
            label = load_label(path, fmt)
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

        whole_number_counts[whole_number] += 1

    return whole_number_counts, n_state0, n_state2, bad_files


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(counts: Counter, total_valid: int, classes=None, sort_by="class"):
    if classes is not None:
        rows = [(c, counts.get(c, 0)) for c in classes]
        header = f"Requested classes ({len(classes)})"
    else:
        rows = [(c, counts.get(c, 0)) for c in range(100)]
        header = "All classes 0-99"

    if sort_by == "count":
        rows.sort(key=lambda r: r[1], reverse=True)
    else:
        rows.sort(key=lambda r: r[0])

    print(f"\n{header}")
    print(f"{'Class':>6} | {'Count':>7} | {'% of valid samples':>18}")
    print("-" * 40)
    zero_classes = []
    for cls, count in rows:
        pct = 100 * count / total_valid if total_valid else 0.0
        flag = ""
        if count == 0:
            flag = "  <-- zero samples"
            zero_classes.append(cls)
        print(f"{cls:>6} | {count:>7} | {pct:>17.2f}%{flag}")

    if zero_classes:
        print(f"\n⚠ {len(zero_classes)} class(es) have ZERO samples: {zero_classes}")


def save_csv(counts: Counter, path: str, classes=None):
    rows = classes if classes is not None else list(range(100))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        for c in rows:
            writer.writerow([c, counts.get(c, 0)])
    print(f"\nSaved CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-class (whole jersey number 0-99) sample counts from label files"
    )
    parser.add_argument("--root", required=True, help="Path to folder containing image/label pairs")
    parser.add_argument(
        "--format",
        choices=["auto", "txt", "json"],
        default="auto",
        help="Label file format. 'auto' detects based on what's actually in --root (default).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print raw content of a few sample label files and exit, without running the full scan. "
             "Use this first if you're unsure of the exact label schema.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated list of specific classes to report, e.g. 2,56,78,66,77,99 "
             "(default: report ALL classes 0-99)",
    )
    parser.add_argument(
        "--sort",
        choices=["class", "count"],
        default="class",
        help="Sort table by class number (default) or by count, busiest first",
    )
    parser.add_argument("--csv", default=None, help="Optional path to also save results as CSV")
    args = parser.parse_args()

    fmt = detect_format(args.root) if args.format == "auto" else args.format

    if args.inspect:
        inspect_samples(args.root, fmt)
        return

    classes = None
    if args.classes:
        classes = [int(x.strip()) for x in args.classes.split(",") if x.strip() != ""]

    label_files = find_label_files(args.root, fmt)
    print(f"Using format: {fmt}   |   Found {len(label_files)} label files")

    counts, n_state0, n_state2, bad_files = scan_whole_numbers(label_files, fmt)
    total_valid = sum(counts.values())

    print(f"\nValid 1-2 digit samples: {total_valid}")
    print(f"State 0 (no number visible): {n_state0}")
    print(f"State 2 (ambiguous/3+ boxes): {n_state2}")
    if bad_files:
        print(f"\n⚠ {len(bad_files)} files failed to parse — schema may not match what "
              f"load_label_{fmt}() expects. Re-run with --inspect to check the raw format.")
        for p, err in bad_files[:10]:
            print(f"  - {p}: {err}")

    print_table(counts, total_valid, classes=classes, sort_by=args.sort)

    if args.csv:
        save_csv(counts, args.csv, classes=classes)


if __name__ == "__main__":
    main()
