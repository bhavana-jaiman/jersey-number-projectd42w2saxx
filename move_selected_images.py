"""
Pick N image names from a txt list, then MOVE those images and their label
files out of validation_final into a new folder (keeping the same sub-folder
layout, e.g. images/ and labels/).

Usage example:
  python move_selected_images.py \
      --list my_list.txt \
      --src ../Datasets/dataset_8_10_2026/validation_final \
      --dst ../Datasets/dataset_8_10_2026/validation_70 \
      --n 70 --dry_run

Remove --dry_run once the printed plan looks right.
"""
import argparse
import os
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

p = argparse.ArgumentParser()
p.add_argument("--list", required=True, help="txt file with image names (one per line)")
p.add_argument("--src", required=True, help="validation_final folder")
p.add_argument("--dst", required=True, help="new folder to move into")
p.add_argument("--n", type=int, default=70)
p.add_argument("--mode", choices=["random", "first"], default="random",
               help="random sample (reproducible with --seed) or first N lines")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--dry_run", action="store_true", help="only print, don't move")
args = p.parse_args()

# 1. Read the list. A line may be just "img.jpg" or "img.jpg <label>" / "path/img.jpg,label".
lines = []
with open(args.list) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name = os.path.basename(line.replace(",", " ").split()[0])
        lines.append((name, line))

# 2. Index every file under src by file name and by stem.
by_name, by_stem = {}, {}
for root, _, files in os.walk(args.src):
    for fn in files:
        full = os.path.join(root, fn)
        by_name.setdefault(fn, []).append(full)
        by_stem.setdefault(os.path.splitext(fn)[0], []).append(full)

# Keep only list entries whose image actually exists.
available = [(n, l) for n, l in lines if n in by_name]
missing = [n for n, _ in lines if n not in by_name]
print(f"List entries: {len(lines)} | found in src: {len(available)} | missing: {len(missing)}")
if missing:
    print("  e.g. missing:", missing[:5])
if len(available) < args.n:
    raise SystemExit(f"Only {len(available)} images available, need {args.n}.")

# 3. Select N.
if args.mode == "random":
    random.seed(args.seed)
    selected = random.sample(available, args.n)
else:
    selected = available[: args.n]

# 4. Move each image + every file sharing its stem (labels: .txt/.json/.xml, ...).
moved = 0
for name, _ in selected:
    stem = os.path.splitext(name)[0]
    for path in by_stem.get(stem, []):
        rel = os.path.relpath(path, args.src)          # keeps images/ labels/ layout
        out = os.path.join(args.dst, rel)
        kind = "IMG  " if os.path.splitext(path)[1].lower() in IMG_EXTS else "LABEL"
        print(f"{kind} {rel}")
        if not args.dry_run:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.move(path, out)
            moved += 1

# 5. Save the selected lines (with any labels that were in the txt) for reference.
if not args.dry_run:
    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "selected_list.txt"), "w") as f:
        f.write("\n".join(l for _, l in selected) + "\n")
    print(f"\nMoved {moved} files for {len(selected)} images -> {args.dst}")
else:
    print("\nDry run only. Nothing moved.")
