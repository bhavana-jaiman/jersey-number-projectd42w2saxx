"""
remove_images.py  -- removes chosen images (+ label + debug image) from ANY dataset folder.
Nothing is deleted: files are MOVED to <root>_removed/ (same sub-folders), so you can undo.

Works for both layouts (detected automatically):
  A) <root>/images/x.jpg + <root>/labels/x.txt   (+ <root>/debug/x.jpg)   e.g. syn3_dataset
  B) <root>/x.jpg + <root>/x.txt                 (+ <root>/debug/x.jpg)   e.g. syn_digits_train

Choose what to remove:
  --list bad.txt     one image name per line (with/without .jpg, full path or review name ok)
  --names a.jpg b    names directly on the command line
  --orphans          you already deleted bad images by hand -> moves labels/debug left without image

Run (always --dry_run first):
  python remove_images.py --root <folder> --list bad.txt --dry_run
  python remove_images.py --root <folder> --list bad.txt
  python remove_images.py --root <folder> --orphans
"""
import argparse
import glob
import os
import shutil
import sys

EXTS = (".jpg", ".jpeg", ".png", ".bmp")

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="dataset folder")
ap.add_argument("--list", help="text file with image names to remove")
ap.add_argument("--names", nargs="*", default=[], help="image names to remove")
ap.add_argument("--orphans", action="store_true", help="move labels/debug whose image is gone")
ap.add_argument("--dry_run", action="store_true", help="only show what would be moved")
a = ap.parse_args()

root = os.path.abspath(a.root.rstrip("/"))
if os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels")):
    img_dir, lbl_dir, layout = os.path.join(root, "images"), os.path.join(root, "labels"), "images/ + labels/"
    img_sub, lbl_sub = "images", "labels"
else:
    img_dir = lbl_dir = root
    layout, img_sub, lbl_sub = "jpg + txt in one folder", "", ""
dbg_dir = os.path.join(root, "debug")
if not a.list and not a.names and not a.orphans:
    sys.exit("ERROR: give --list bad.txt, --names ... or --orphans")


def stem(s):
    s = os.path.basename(s.strip()).split("__", 1)[-1]        # review name -> original name
    return os.path.splitext(s)[0]


def images_in(d):
    return {os.path.splitext(f)[0]: os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXTS)}


imgs = images_in(img_dir)
if a.orphans:
    labels = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")}
    names = labels - set(imgs)
else:
    names = {stem(n) for n in a.names if n.strip()}
    if a.list:
        with open(a.list, encoding="utf-8", errors="ignore") as f:
            names |= {stem(l) for l in f if l.strip() and not l.startswith("#")}

dst = root + "_removed"
moved, missing = 0, []
for n in sorted(names):
    files = []
    if n in imgs:
        files.append((imgs[n], img_sub))
    lp = os.path.join(lbl_dir, n + ".txt")
    if os.path.exists(lp):
        files.append((lp, lbl_sub))
    for e in EXTS:
        dp = os.path.join(dbg_dir, n + e)
        if os.path.exists(dp):
            files.append((dp, "debug"))
    if not files:
        missing.append(n)
        continue
    for p, sub in files:
        if a.dry_run:
            print("would move", p)
        else:
            os.makedirs(os.path.join(dst, sub), exist_ok=True)
            shutil.move(p, os.path.join(dst, sub, os.path.basename(p)))
    moved += 1

left_img = len(images_in(img_dir))
left_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])
print(f"\nlayout: {layout}")
print(f"{'would remove' if a.dry_run else 'removed'}: {moved} images (+ labels / debug) -> {dst}")
if missing:
    print(f"not found ({len(missing)}): {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
print(f"left in {os.path.basename(root)}: {left_img} images, {left_lbl} labels"
      + ("" if left_img == left_lbl else "   <- NOT equal, check!"))
