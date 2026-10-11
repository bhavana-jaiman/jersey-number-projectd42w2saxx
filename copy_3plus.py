"""
copy_3plus.py
Copies every validation image whose label has 3 or more digits into check_3plus/,
with the label in the file name (e.g. 2_8_3__image.jpg), so you can look at them.

Run:
    python copy_3plus.py --root ../workspace_bhavana/Datasets/70_30_dataset/validation_body
"""
import argparse
import glob
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="folder with images/ and labels/")
ap.add_argument("--out", default="check_3plus")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
n = 0
for lp in sorted(glob.glob(os.path.join(a.root, "labels", "*.txt"))):
    d = np.loadtxt(lp, ndmin=1)
    if d.size <= 2:
        continue
    base = os.path.splitext(os.path.basename(lp))[0]
    ip = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(a.root, "images", base + ext)
        if os.path.exists(p):
            ip = p
            break
    if ip is None:
        print("image not found for", lp)
        continue
    label = "_".join(str(int(x)) for x in d)
    shutil.copy(ip, os.path.join(a.out, f"{label}__{os.path.basename(ip)}"))
    n += 1
print("copied", n, "images to", a.out)
