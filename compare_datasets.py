"""
compare_datasets.py  -- checks whether two dataset folders contain the same samples.
Read-only. Works with images/ + labels/  or  jpg + txt in one folder.

Reports:
  same file name            -> made from the same base photo
  same name + same label    -> exact duplicate sample
  same name, other label    -> same photo, different pasted digit (near-duplicate)
  identical image content   -> byte-identical images (md5), even with different names
Writes <out>/overlap_names.txt  (names of B that also exist in A) -> use with remove_images.py

Run:
  python compare_datasets.py --a <folder A> --b <folder B>
"""
import argparse
import collections
import hashlib
import os

EXTS = (".jpg", ".jpeg", ".png")


def load(root):
    img_dir = os.path.join(root, "images") if os.path.isdir(os.path.join(root, "images")) else root
    lbl_dir = os.path.join(root, "labels") if os.path.isdir(os.path.join(root, "labels")) else root
    data = {}
    for f in os.listdir(img_dir):
        if not f.lower().endswith(EXTS) or f == "preview.png":
            continue
        n = os.path.splitext(f)[0]
        lp = os.path.join(lbl_dir, n + ".txt")
        lab = " ".join(open(lp).read().split()) if os.path.exists(lp) else None
        with open(os.path.join(img_dir, f), "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        data[n] = (lab, md5)
    return data


def last_digit_hist(d):
    c = collections.Counter(l.split()[-1] for l, _ in d.values() if l)
    return " ".join(f"{k}:{c[k]}" for k in sorted(c))


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--out", default=".")
opt = ap.parse_args()

A, B = load(opt.a), load(opt.b)
common = sorted(set(A) & set(B))
same_lab = [n for n in common if A[n][0] == B[n][0]]
diff_lab = [n for n in common if A[n][0] != B[n][0]]
same_px = len({m for _, m in A.values()} & {m for _, m in B.values()})

print(f"A: {len(A):5d} images  {opt.a}")
print(f"B: {len(B):5d} images  {opt.b}")
print(f"same file name (same base photo)  : {len(common)}")
print(f"  ... with the SAME label         : {len(same_lab)}   <- exact duplicates")
print(f"  ... with a DIFFERENT label      : {len(diff_lab)}   <- same photo, other pasted digit")
print(f"byte-identical images (md5)       : {same_px}")
print(f"unique to B (not in A)            : {len(set(B) - set(A))}")
print(f"last-digit counts A: {last_digit_hist(A)}")
print(f"last-digit counts B: {last_digit_hist(B)}")
for n in diff_lab[:5]:
    print(f"  e.g. {n}: A='{A[n][0]}'  B='{B[n][0]}'")

os.makedirs(opt.out, exist_ok=True)
p = os.path.join(opt.out, "overlap_names.txt")
with open(p, "w") as f:
    f.write("\n".join(common) + ("\n" if common else ""))
print(f"\nnames of B that also exist in A -> {p}")
