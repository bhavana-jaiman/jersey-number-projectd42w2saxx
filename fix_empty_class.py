"""
fix_empty_class.py  -- changes the class -1 to 10 (State 0, "no number") in every YOLO label
of a folder. Only the first column is changed; the box (cx cy w h) stays the same.
A backup of the folder is made first: <folder>_backup  (only if it does not exist yet).

Run:
  python fix_empty_class.py --dir ../workspace_bhavana/Datasets/70_30_dataset/train_empty_additions --dry_run
  python fix_empty_class.py --dir ../workspace_bhavana/Datasets/70_30_dataset/train_empty_additions
Options:
  --old -1 --new 10   (defaults) to change another class value
"""
import argparse
import collections
import glob
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="folder with .txt labels (jpg + txt, or labels/)")
ap.add_argument("--old", default="-1", help="class value to replace")
ap.add_argument("--new", default="10", help="new class value")
ap.add_argument("--dry_run", action="store_true", help="only count, change nothing")
opt = ap.parse_args()

d = opt.dir.rstrip("/")
lbl_dir = os.path.join(d, "labels") if os.path.isdir(os.path.join(d, "labels")) else d
files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
if not files:
    raise SystemExit(f"no .txt labels in {lbl_dir}")


def is_old(v):
    try:
        return float(v) == float(opt.old)
    except ValueError:
        return False


if not opt.dry_run:
    bak = d + "_backup"
    if not os.path.exists(bak):
        shutil.copytree(d, bak)
        print("backup ->", bak)
    else:
        print("backup already exists, not overwritten:", bak)

changed, bad, empty = 0, [], []
for p in files:
    rows = [r.split() for r in open(p).read().splitlines() if r.strip()]
    if not rows:
        empty.append(os.path.basename(p))
        continue
    edit = False
    for r in rows:
        if is_old(r[0]):
            r[0] = opt.new
            edit = True
        if len(r) != 5:
            bad.append(os.path.basename(p))
    if edit:
        changed += 1
        if not opt.dry_run:
            with open(p, "w") as f:
                f.write("\n".join(" ".join(r) for r in rows) + "\n")

counts = collections.Counter()
for p in files:
    for r in open(p).read().splitlines():
        if r.strip():
            counts[r.split()[0]] += 1

print(f"label files: {len(files)}")
print(f"{'would change' if opt.dry_run else 'changed'}: {changed} files ({opt.old} -> {opt.new})")
print("class counts now:", dict(counts))
if empty:
    print(f"WARNING: {len(empty)} empty label files, e.g. {empty[:3]}")
if bad:
    print(f"WARNING: {len(set(bad))} files with rows that do not have 5 values, e.g. {sorted(set(bad))[:3]}")
