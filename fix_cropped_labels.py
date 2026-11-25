"""
Rewrites label files in a folder like validation_body_cropped_whole/labels/
(currently holding "class cx cy w h" box lines, e.g. "23 0.454500 0.288600
0.331600 0.143200") into plain digit-sequence labels (e.g. "2 3"), matching
the format already used in validation/labels/ and 3_digit_validation/labels/
(e.g. "7 7").

The class number on each box line already IS the jersey number read
left-to-right (that's how merge_digit_boxes_v2.py built it: digits in boxes
it merged, concatenated in left-to-right order, e.g. "2"+"3" -> class "23").
So converting is just: take that class number, split it into its individual
digits, and write them space-separated. No coordinates are kept.

Class "0" is treated as empty/no-number (merge_digit_boxes_v2.py uses 0 for
files with no boxes) and written out as an empty label file, matching how
your dataset represents empty-jersey samples.

If a label file has more than one line (shouldn't happen after merging --
every file should be exactly one merged box per the "boxes per file AFTER:
{1: 9603}" result you already confirmed), this script only reads the FIRST
line's class and reports how many such files it saw; it does not silently
concatenate multiple lines together.

This REWRITES files in --labels_dir in place (that's the already-cropped
output folder, not your original source data, so this is safe -- but a
dry-run is available first).

Usage (dry run -- prints what would change for a few files, writes nothing):
    python fix_cropped_labels.py --labels_dir validation_body_cropped_whole/labels --dry_run --sample 10

Then for real (rewrites every file in --labels_dir):
    python fix_cropped_labels.py --labels_dir validation_body_cropped_whole/labels
"""
import argparse
import glob
import os


def class_to_digit_sequence(cls_str):
    """'23' -> '2 3', '0' -> '' (empty/no-number), '7' -> '7'."""
    if cls_str == "0":
        return ""
    return " ".join(list(cls_str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="Folder of label .txt files to rewrite in place, e.g. validation_body_cropped_whole/labels")
    ap.add_argument("--dry_run", action="store_true", help="Only print planned changes for --sample files; write nothing")
    ap.add_argument("--sample", type=int, default=10, help="Number of files to show in --dry_run (default: 10)")
    args = ap.parse_args()

    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"ERROR: no .txt files found in {args.labels_dir}")

    print(f"Found {len(label_files)} label files in {args.labels_dir}")
    print()

    todo = label_files[: args.sample] if args.dry_run else label_files
    if args.dry_run:
        print(f"--dry_run: showing up to {args.sample} samples, writing NOTHING.\n")

    n_changed = 0
    n_multiline = 0
    n_unparseable = 0

    for lf in todo:
        with open(lf, "r", encoding="utf8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            continue

        if len(lines) > 1:
            n_multiline += 1

        parts = lines[0].split()
        if not parts:
            n_unparseable += 1
            continue
        cls_str = parts[0]
        if not cls_str.isdigit():
            n_unparseable += 1
            if args.dry_run:
                print(f"{os.path.basename(lf)}: class '{cls_str}' is not numeric, skipped")
            continue

        new_content = class_to_digit_sequence(cls_str)

        if args.dry_run:
            print(f"{os.path.basename(lf)}: '{lines[0]}'  ->  '{new_content}'")
            continue

        with open(lf, "w", encoding="utf8") as f:
            f.write(new_content + "\n" if new_content else "")
        n_changed += 1

    if args.dry_run:
        print(f"\nPlanned {len(todo)} rewrites shown.")
        print("Re-run without --dry_run to actually rewrite all label files.")
    else:
        print(f"Done. Rewrote {n_changed} label files to digit-sequence format in:")
        print(f"  {args.labels_dir}")
        if n_multiline:
            print(f"  NOTE: {n_multiline} files had more than one line; only the first line's class was used.")
        if n_unparseable:
            print(f"  NOTE: {n_unparseable} files had a non-numeric class and were skipped (left unchanged).")


if __name__ == "__main__":
    main()
