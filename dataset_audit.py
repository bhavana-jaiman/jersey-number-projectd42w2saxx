"""
dataset_audit.py

Follow-up audit for the jersey-number dataset, extending check_balance.py
with two targeted checks:

  1. "No number visible" (state 0 / class_id == 10) verification
     - exact count and percentage of label files that actually contain a
       class_id == 10 row
     - lists a few matching files so you can eyeball them
     - flags if the count is suspiciously low/zero

  2. Whole jersey-number "spike" investigation
     - re-derives the whole-number histogram (0-99) using the same
       classify_label logic as check_balance.py
     - for the top-N most common numbers, finds every matching label file
     - hashes the matching images (MD5) to catch EXACT duplicates
     - saves a 3x3 visual grid of sample crops per spike number so you can
       eyeball near-duplicates (heavily augmented copies of the same photo)

Nothing here modifies any files in the dataset — everything is read-only,
outputs go to --outdir.

Usage:
    python dataset_audit.py --root /path/to/training_dataset_Ying
    python dataset_audit.py --root /path/to/training_dataset_Ying \
        --outdir audit_report --top-n 5 --samples-per-spike 9
"""

import os
import argparse
import hashlib
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless-server safe: never tries to open a window
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Same label-parsing logic as check_balance.py, kept in sync deliberately so
# the numbers in this report always agree with the balance report.
# ---------------------------------------------------------------------------

def classify_label(label: np.ndarray):
    """
    Returns:
        state (int): 0, 1, or 2
        digits (list[int]): digit sequence for state==1 samples (1 or 2
            ints), otherwise [] or a placeholder
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


def find_label_files(root: str):
    return [
        os.path.join(root, f) for f in os.listdir(root) if f.endswith(".txt")
    ]


# ---------------------------------------------------------------------------
# Check 1 — "no number visible" (state 0) verification
# ---------------------------------------------------------------------------

def check_absent_digit(root: str, label_files, sample_count: int = 5):
    print("\n" + "=" * 70)
    print("CHECK 1 — 'No number visible' (state 0 / class_id == 10)")
    print("=" * 70)

    state0_files = []
    for path in label_files:
        try:
            label = load_label(path)
        except Exception:
            continue
        # Flag any file that contains AT LEAST ONE class_id == 10 row,
        # even if classify_label() would ultimately call the sample
        # state 2 (ambiguous) because of extra boxes.
        if np.any(label[:, 0] == 10):
            state0_files.append(path)

    total = len(label_files)
    n = len(state0_files)
    pct = 100 * n / total if total else 0.0

    print(f"Total label files scanned : {total}")
    print(f"Files containing class_id==10 : {n}  ({pct:.2f}%)")

    if n == 0:
        print(
            "\n⚠  NO 'no visible jersey number' examples found in this "
            "dataset. The model has zero real negative examples for the "
            "'absent digit' class (class 10 in digit1/digit2 heads) — "
            "this is a genuine data gap, not a chart-rendering artifact."
        )
    elif pct < 1.0:
        print(
            f"\n⚠  Only {pct:.2f}% of samples represent 'no number visible' "
            "— very thin signal for that class. Worth reviewing whether "
            "this matches real-world frequency of occluded/blurry jerseys."
        )
    else:
        print(
            f"\n✓ {pct:.2f}% of samples cover the 'no number visible' case "
            "— present, though still worth sanity-checking a few examples."
        )

    if state0_files:
        print(f"\nSample matching files (up to {sample_count}):")
        for p in state0_files[:sample_count]:
            img_path = p.replace(".txt", ".jpg")
            exists = "✓ image found" if os.path.exists(img_path) else "✗ MISSING IMAGE"
            print(f"  - {p}   [{exists}]")

    return {"total": total, "state0_count": n, "state0_pct": pct, "state0_files": state0_files}


# ---------------------------------------------------------------------------
# Check 2 — whole-number spike investigation
# ---------------------------------------------------------------------------

def scan_whole_numbers(label_files):
    """Re-derive the whole-number histogram, keeping a map of
    whole_number -> list of contributing label file paths, so we can go
    straight from 'this number spikes' to 'here are the exact files'."""
    whole_number_files = {}
    whole_number_counts = Counter()
    length_counts = Counter()
    bad_files = []

    for path in label_files:
        try:
            label = load_label(path)
        except Exception as e:
            bad_files.append((path, str(e)))
            continue

        state, digits, whole_number = classify_label(label)
        if state != 1:
            continue

        length_counts[len(digits)] += 1
        whole_number_counts[whole_number] += 1
        whole_number_files.setdefault(whole_number, []).append(path)

    return whole_number_counts, whole_number_files, length_counts, bad_files


def md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_spike(number: int, files, outdir: str, samples_per_spike: int = 9):
    print(f"\n--- Spike check: jersey number {number:02d}  ({len(files)} label files) ---")

    img_paths = []
    for txt_path in files:
        jpg_path = txt_path.replace(".txt", ".jpg")
        if os.path.exists(jpg_path):
            img_paths.append(jpg_path)
        else:
            print(f"  ✗ missing image for {txt_path}")

    # --- exact duplicate check via MD5 ---
    hashes = Counter()
    hash_to_files = {}
    for p in img_paths:
        try:
            h = md5_of_file(p)
        except Exception as e:
            print(f"  ✗ could not hash {p}: {e}")
            continue
        hashes[h] += 1
        hash_to_files.setdefault(h, []).append(p)

    exact_dupe_groups = {h: fs for h, fs in hash_to_files.items() if len(fs) > 1}
    if exact_dupe_groups:
        total_dupes = sum(len(fs) for fs in exact_dupe_groups.values())
        print(f"  ⚠ EXACT duplicate images found: {len(exact_dupe_groups)} groups, "
              f"{total_dupes} files total are byte-for-byte identical copies")
        for h, fs in list(exact_dupe_groups.items())[:3]:
            print(f"      identical group ({len(fs)} files): {fs[:3]}{' ...' if len(fs) > 3 else ''}")
    else:
        print(f"  ✓ No exact (byte-for-byte) duplicate images among {len(img_paths)} files")

    # --- visual grid for manual near-duplicate check ---
    sample = img_paths[:samples_per_spike]
    if sample:
        cols = 3
        rows = (len(sample) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, p in zip(axes, sample):
            try:
                img = Image.open(p)
                ax.imshow(img)
                ax.set_title(os.path.basename(p), fontsize=7)
            except Exception as e:
                ax.set_title(f"error: {e}", fontsize=7)
            ax.axis("off")
        for ax in axes[len(sample):]:
            ax.axis("off")
        plt.suptitle(f"Jersey number {number:02d} — {len(sample)} of {len(img_paths)} samples")
        plt.tight_layout()
        grid_path = os.path.join(outdir, f"spike_{number:02d}_samples.png")
        plt.savefig(grid_path, dpi=120)
        plt.close()
        print(f"  Saved visual grid: {grid_path}  (inspect by eye for near-duplicates)")

    return {
        "number": number,
        "n_files": len(files),
        "n_images_found": len(img_paths),
        "exact_dupe_groups": len(exact_dupe_groups),
        "exact_dupe_files": sum(len(fs) for fs in exact_dupe_groups.values()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit jersey-number dataset for absent-digit coverage and duplicate/spike numbers"
    )
    parser.add_argument("--root", required=True, help="Path to the folder containing .jpg/.txt pairs")
    parser.add_argument("--outdir", default="audit_report", help="Where to save grids and the text report")
    parser.add_argument("--top-n", type=int, default=5, help="How many top whole-numbers to investigate")
    parser.add_argument("--samples-per-spike", type=int, default=9, help="Images per visual grid")
    parser.add_argument("--state0-samples", type=int, default=5, help="How many state-0 example paths to print")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Scanning: {args.root}")
    label_files = find_label_files(args.root)
    print(f"Found {len(label_files)} label files")

    # Check 1
    absent_digit_stats = check_absent_digit(args.root, label_files, args.state0_samples)

    # Check 2 — rebuild whole-number histogram + file index
    print("\n" + "=" * 70)
    print("CHECK 2 — Whole jersey-number spike investigation")
    print("=" * 70)
    whole_counts, whole_files, length_counts, bad_files = scan_whole_numbers(label_files)

    if bad_files:
        print(f"\n⚠ {len(bad_files)} label files failed to parse and were skipped:")
        for p, err in bad_files[:10]:
            print(f"  - {p}: {err}")

    print("\nJersey-number length breakdown (state==1 samples):")
    total_len = sum(length_counts.values()) or 1
    for length in sorted(length_counts):
        c = length_counts[length]
        print(f"  {length}-digit: {c}  ({100*c/total_len:.1f}%)")

    top_numbers = whole_counts.most_common(args.top_n)
    print(f"\nTop {args.top_n} most common whole numbers:")
    for num, count in top_numbers:
        print(f"  #{num:02d}: {count} samples")

    spike_results = []
    for num, count in top_numbers:
        result = check_spike(num, whole_files[num], args.outdir, args.samples_per_spike)
        spike_results.append(result)

    # --- final summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Absent-digit (state 0) coverage: {absent_digit_stats['state0_count']} / "
          f"{absent_digit_stats['total']} files ({absent_digit_stats['state0_pct']:.2f}%)")
    for r in spike_results:
        flag = "⚠ DUPLICATES FOUND" if r["exact_dupe_groups"] > 0 else "✓ clean"
        print(f"  Jersey #{r['number']:02d}: {r['n_files']} samples, "
              f"{r['exact_dupe_files']} exact-duplicate files  [{flag}]")
    print(f"\nVisual grids and this report are saved under: {args.outdir}/")
    print("Review the grid PNGs by eye for NEAR-duplicates (same photo, "
          "rotated/color-shifted) — MD5 hashing only catches EXACT copies.")


if __name__ == "__main__":
    main()
