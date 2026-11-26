"""Inspect a folder of .txt label files and cross-check digit counts.

Usage:
    python check_labels.py /path/to/folder_with_txt
    python check_labels.py /path/to/folder_with_txt --show 10
    python check_labels.py /path/to/folder_with_txt --col 0

Works on .txt files directly in the folder, or in a 'labels/' subfolder.
"""
import os, re, argparse
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--show", type=int, default=5, help="how many sample files to print")
    ap.add_argument("--col", type=int, default=0, help="which whitespace-separated column is the label")
    a = ap.parse_args()

    root = os.path.expanduser(a.root)
    ldir = os.path.join(root, "labels")
    if not os.path.isdir(ldir):
        ldir = root

    files = sorted(f for f in os.listdir(ldir) if f.endswith(".txt"))
    print(f"Folder          : {ldir}")
    print(f"Total .txt files: {len(files)}")
    if not files:
        print("No .txt files found. Check the path.")
        return

    empty, multi_line, multi_col = 0, 0, 0
    raw_values = Counter()
    col_counts = Counter()
    samples = []

    for f in files:
        with open(os.path.join(ldir, f), errors="replace") as fh:
            txt = fh.read().strip()
        if not txt:
            empty += 1
            raw_values["<empty>"] += 1
            if len(samples) < a.show:
                samples.append((f, "<empty>"))
            continue
        lines = txt.splitlines()
        if len(lines) > 1:
            multi_line += 1
        toks = lines[0].split()
        col_counts[len(toks)] += 1
        if len(toks) > 1:
            multi_col += 1
        val = toks[a.col] if a.col < len(toks) else "<missing col>"
        raw_values[val] += 1
        if len(samples) < a.show:
            samples.append((f, txt.replace("\n", " | ")))

    print(f"\nSample files (first {a.show}):")
    for f, c in samples:
        print(f"  {f}  ->  {c}")

    print(f"\nEmpty files          : {empty}")
    print(f"Files with >1 line   : {multi_line}")
    print(f"Files with >1 column : {multi_col}")
    print(f"Columns per file     : {dict(sorted(col_counts.items()))}")
    if multi_line or multi_col:
        print("  WARNING: files have extra lines/columns. Check which column is the "
              "jersey number (use --col) before trusting the counts.")

    print(f"\nLabel value counts (column {a.col}):")
    def sort_key(kv):
        k = kv[0]
        return (0, int(k)) if re.fullmatch(r"-?\d+", k) else (1, 0)
    for k, c in sorted(raw_values.items(), key=sort_key):
        print(f"  {k}: {c}")

    numeric = {k: c for k, c in raw_values.items() if re.fullmatch(r"\d+", k)}
    non_numeric = sum(c for k, c in raw_values.items() if k not in numeric)
    per_digit = Counter()
    length = Counter()
    for k, c in numeric.items():
        length[len(k)] += c
        for ch in k:
            per_digit[ch] += c

    print(f"\nNumeric labels       : {sum(numeric.values())}")
    print(f"Empty / -1 / other   : {non_numeric}")
    print(f"Check (numeric + other = total): {sum(numeric.values()) + non_numeric} vs {len(files)}")
    print(f"Digits per jersey    : {dict(sorted(length.items()))}")
    print("Digit counts (0-9)   :", {d: per_digit[d] for d in '0123456789'})


if __name__ == "__main__":
    main()
