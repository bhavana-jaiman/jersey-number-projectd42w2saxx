"""
remove_syn3_images.py
Removes chosen images from syn3_dataset together with their label (and debug image).
Nothing is deleted: everything goes to <root>_removed/images, /labels, /debug.

Two ways to choose:
  A) --list bad.txt   one image name per line (with or without .jpg, path allowed)
     --names a.jpg b    or the names directly on the command line
  B) --orphans        you already deleted bad images from images/ by hand
                      -> moves every label (and debug image) that has no image left

Run (from workspace_bhavana_New_arc_bodycrop):
    python remove_syn3_images.py --root ../workspace_bhavana/Datasets/70_30_dataset/syn3_dataset --list bad.txt --dry_run
    python remove_syn3_images.py --root ../workspace_bhavana/Datasets/70_30_dataset/syn3_dataset --list bad.txt
    python remove_syn3_images.py --root ../workspace_bhavana/Datasets/70_30_dataset/syn3_dataset --orphans
"""
import argparse
import glob
import os
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="syn3_dataset folder (images/, labels/, debug/)")
ap.add_argument("--list", help="text file with the image names to remove")
ap.add_argument("--names", nargs="*", default=[], help="image names to remove (instead of --list)")
ap.add_argument("--orphans", action="store_true", help="remove labels whose image was already deleted")
ap.add_argument("--dry_run", action="store_true", help="only show what would be moved")
a = ap.parse_args()

root = os.path.abspath(a.root.rstrip("/"))
img_dir, lbl_dir, dbg_dir = (os.path.join(root, d) for d in ("images", "labels", "debug"))
if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
    sys.exit(f"ERROR: {root} must contain images/ and labels/")
if not a.list and not a.names and not a.orphans:
    sys.exit("ERROR: give --list bad.txt, --names ... or --orphans")

if a.list or a.names:
    names = {os.path.splitext(os.path.basename(n.strip()))[0] for n in a.names if n.strip()}
    if a.list:
        with open(a.list) as f:
            names |= {os.path.splitext(os.path.basename(l.strip()))[0] for l in f if l.strip()}
else:
    have = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(img_dir, "*.jpg"))}
    names = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(lbl_dir, "*.txt"))} - have

dst = root + "_removed"
moved, missing = 0, []
for n in sorted(names):
    files = [(os.path.join(img_dir, n + ".jpg"), "images"),
             (os.path.join(lbl_dir, n + ".txt"), "labels"),
             (os.path.join(dbg_dir, n + ".jpg"), "debug")]
    found = [(p, sub) for p, sub in files if os.path.exists(p)]
    if not found:
        missing.append(n)
        continue
    for p, sub in found:
        if a.dry_run:
            print("would move", p)
        else:
            os.makedirs(os.path.join(dst, sub), exist_ok=True)
            shutil.move(p, os.path.join(dst, sub, os.path.basename(p)))
    moved += 1

left_img = len(glob.glob(os.path.join(img_dir, "*.jpg")))
left_lbl = len(glob.glob(os.path.join(lbl_dir, "*.txt")))
print(f"\n{'would remove' if a.dry_run else 'removed'}: {moved} images (+ labels / debug) -> {dst}")
if missing:
    print(f"not found ({len(missing)}): {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
print(f"left in syn3_dataset: {left_img} images, {left_lbl} labels"
      + ("" if left_img == left_lbl else "   <- NOT equal, check!"))
