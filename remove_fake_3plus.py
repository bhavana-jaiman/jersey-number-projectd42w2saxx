"""
remove_fake_3plus.py
Moves every 3+ image (label with 3 or more digits) OUT of the validation folder.
Nothing is deleted: files go to <root>_removed_3plus/images and /labels,
so you can look at them and move a real 3-digit number back if you find one.

Run (from New_arcitecture_2):
    python remove_fake_3plus.py --root ../Datasets/70_30_dataset/validation_body --dry_run
    python remove_fake_3plus.py --root ../Datasets/70_30_dataset/validation_body
"""
import argparse
import glob
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="validation folder with images/ and labels/")
ap.add_argument("--dry_run", action="store_true", help="only count, move nothing")
a = ap.parse_args()

root = a.root.rstrip("/")
dst_img = os.path.join(root + "_removed_3plus", "images")
dst_lbl = os.path.join(root + "_removed_3plus", "labels")
if not a.dry_run:
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

n = 0
for lp in sorted(glob.glob(os.path.join(root, "labels", "*.txt"))):
    base = os.path.splitext(os.path.basename(lp))[0]
    if base.startswith("syn3_"):              # synthetic 3+ made by make_synthetic_3plus_val.py
        continue
    if np.loadtxt(lp, ndmin=1).size <= 2:     # 1-2 digits or "10" (no number) -> keep
        continue
    ip = os.path.join(root, "images", base + ".jpg")
    n += 1
    if a.dry_run:
        continue
    shutil.move(lp, os.path.join(dst_lbl, base + ".txt"))
    if os.path.exists(ip):
        shutil.move(ip, os.path.join(dst_img, base + ".jpg"))
    else:
        print("image not found for", lp)

print(("would move " if a.dry_run else "moved ") + f"{n} 3+ images" +
      ("" if a.dry_run else f" to {root}_removed_3plus"))
