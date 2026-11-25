"""
Widens the margin of existing YOLO-format bounding boxes in
~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train (or any
folder of the same format: flat <name>.jpg + <name>.txt pairs, each .txt
line = "class cx cy w h", all normalized 0-1).

This does NOT touch images -- only rewrites the box numbers in the .txt
labels, so each box covers a larger area (more surrounding jersey/body)
around the same center point. Width and height are each grown by `--pad`
as a fraction of their current size (default 0.20 = 20% larger on each of
width and height, i.e. the box grows outward on all sides), then clamped
so it never exceeds the image's 0-1 normalized bounds.

Originals are NEVER modified in place: writes to a separate --out_dir,
mirroring the --labels_dir's images alongside the new labels (images are
unchanged, just copied through, so the folder is ready to use/inspect/compare).

Usage:
    python widen_yolo_boxes.py \
        --data_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/1768_digits_train_widened \
        --pad 0.20

Dry run first (prints before/after for a few samples, writes nothing):
    python widen_yolo_boxes.py --data_dir ... --out_dir ... --pad 0.20 --dry_run --sample 5
"""
import argparse
import glob
import os
import shutil


def widen_box(cx, cy, w, h, pad):
    """Grow w/h by `pad` fraction (e.g. 0.20 = +20%), keep center fixed,
    then clamp to [0,1] normalized image bounds. Returns new (cx, cy, w, h).
    If clamping would shrink the box asymmetrically (center too close to an
    edge), the center is nudged so the full widened box still fits."""
    new_w = min(1.0, w * (1.0 + pad))
    new_h = min(1.0, h * (1.0 + pad))

    half_w, half_h = new_w / 2.0, new_h / 2.0

    new_cx = min(max(cx, half_w), 1.0 - half_w)
    new_cy = min(max(cy, half_h), 1.0 - half_h)

    return new_cx, new_cy, new_w, new_h


def process_label_file(src_path, dst_path, pad):
    """Returns list of (orig_line_values, new_line_values) for reporting."""
    changes = []
    out_lines = []
    with open(src_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                out_lines.append(line)
                continue
            cls = parts[0]
            try:
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                out_lines.append(line)
                continue
            new_cx, new_cy, new_w, new_h = widen_box(cx, cy, w, h, pad)
            out_lines.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
            changes.append(((cx, cy, w, h), (new_cx, new_cy, new_w, new_h)))

    if dst_path is not None:
        with open(dst_path, "w", encoding="utf8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with flat <name>.jpg + <name>.txt pairs (YOLO format)")
    ap.add_argument("--out_dir", required=True, help="Output folder (must differ from --data_dir); images copied through unchanged, labels widened")
    ap.add_argument("--pad", type=float, default=0.20, help="Fractional extra margin on width/height, e.g. 0.20 = +20%% (default: 0.20)")
    ap.add_argument("--dry_run", action="store_true", help="Only print before/after for --sample files; write nothing")
    ap.add_argument("--sample", type=int, default=5, help="Number of files to show in --dry_run (default: 5)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    if data_dir == out_dir:
        raise SystemExit("ERROR: --out_dir must be different from --data_dir (originals are never overwritten).")

    label_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not label_files:
        raise SystemExit(f"ERROR: no .txt label files found in {data_dir}")

    print(f"Found {len(label_files)} label files in {data_dir}")
    print(f"Padding: +{args.pad*100:.0f}% on width and height (center-preserving, clamped to image bounds)")
    print()

    if args.dry_run:
        print(f"--dry_run: showing up to {args.sample} samples, writing NOTHING.\n")
        for lf in label_files[: args.sample]:
            changes = process_label_file(lf, None, args.pad)
            name = os.path.basename(lf)
            print(f"{name}:")
            for (cx, cy, w, h), (ncx, ncy, nw, nh) in changes:
                print(f"  before: cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
                w_pct = (nw / w - 1.0) * 100 if w > 0 else 0.0
                h_pct = (nh / h - 1.0) * 100 if h > 0 else 0.0
                print(f"  after:  cx={ncx:.4f} cy={ncy:.4f} w={nw:.4f} h={nh:.4f}  "
                      f"(w {'+' if w_pct>=0 else ''}{w_pct:.1f}%, h {'+' if h_pct>=0 else ''}{h_pct:.1f}%)")
            print()
        print("Re-run without --dry_run to actually write the widened dataset.")
        return

    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    n_images_copied = 0
    for lf in label_files:
        name = os.path.splitext(os.path.basename(lf))[0]
        dst_label = os.path.join(out_dir, name + ".txt")
        process_label_file(lf, dst_label, args.pad)
        n_written += 1

        # copy the matching image through unchanged, whatever extension it has
        src_img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(data_dir, name + ext)
            if os.path.exists(cand):
                src_img = cand
                break
        if src_img is not None:
            shutil.copy2(src_img, os.path.join(out_dir, os.path.basename(src_img)))
            n_images_copied += 1

    print(f"Done. Wrote {n_written} widened label files and copied {n_images_copied} images to:")
    print(f"  {out_dir}")
    if n_images_copied != n_written:
        print(f"  NOTE: {n_written - n_images_copied} label files had no matching image found alongside them.")


if __name__ == "__main__":
    main()
