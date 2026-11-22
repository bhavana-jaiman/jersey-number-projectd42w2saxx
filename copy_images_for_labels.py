#!/usr/bin/env python3
"""Copy the image for every label file into a dataset folder (next to its label).

Usage:
  python copy_images_for_labels.py --labels syn_digits_train_whole --images syn_digits_train \
         --dest syn_digits_train_whole [--exclude vis debug annot preview viz] [--dry-run] [--symlink]

- Looks up each label's image by file name (stem) under --images (recursive).
- Paths containing any --exclude word (default list below) are ignored.
- If a stem still matches MORE than one image, it is NOT copied and is listed in
  ambiguous.txt so a pre-annotated copy can never slip in.
- Originals are never modified or deleted."""
import os, sys, shutil, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--dest", required=True)
ap.add_argument("--exclude", nargs="*", default=["vis", "debug", "annot", "preview", "viz", "drawn", "boxes"])
ap.add_argument("--symlink", action="store_true", help="symlink instead of copy (saves disk)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
labels, images, dest = (os.path.abspath(p) for p in (a.labels, a.images, a.dest))

idx = defaultdict(list)
for d, _, fs in os.walk(images):
    if any(w in os.path.relpath(d, images).lower() for w in a.exclude) and d != images:
        continue
    for f in fs:
        s, e = os.path.splitext(f)
        if e.lower() in IMG: idx[s].append(os.path.join(d, f))

ok = missing = 0; amb = []
for d, _, fs in os.walk(labels):
    for f in fs:
        s, e = os.path.splitext(f)
        if e != ".txt" or f.lower() in ("classes.txt", "labels.txt"): continue
        c = idx.get(s, [])
        if not c: missing += 1; continue
        if len(c) > 1: amb.append((s, c)); continue
        out = os.path.join(dest, os.path.relpath(d, labels), os.path.basename(c[0]))
        if not a.dry_run:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if not os.path.exists(out):
                (os.symlink(c[0], out) if a.symlink else shutil.copy2(c[0], out))
        ok += 1
print(f"labels with exactly one image: {ok} | no image found: {missing} | ambiguous: {len(amb)}")
if amb:
    with open("ambiguous.txt", "w") as fh:
        for s, c in amb: fh.write(s + "\n  " + "\n  ".join(c) + "\n")
    print("ambiguous names listed in ambiguous.txt (first one):", amb[0][0], amb[0][1])
print("dry run, nothing copied" if a.dry_run else f"images placed in: {dest}")
