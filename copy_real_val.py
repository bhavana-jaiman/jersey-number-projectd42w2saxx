"""
copy_real_val.py  -- copies ONLY the REAL validation images (+ their labels) into a NEW
folder with images/ and labels/ sub-folders. The source folder is never modified.

Rule: a file is SYNTHETIC if its name ends with "_<number>" before the extension
      e.g.  0A834BC1...D81271_7.jpg   -> synthetic (skipped)
            0A32915A...B6F37A1.jpg    -> real      (copied)

Works with both source layouts:
  <src>/images/x.jpg + <src>/labels/x.txt      or      <src>/x.jpg + <src>/x.txt

Run (check first with --dry_run):
  python copy_real_val.py --src <validation folder> --out <validation folder>_real --dry_run
  python copy_real_val.py --src <validation folder> --out <validation folder>_real
"""
import argparse
import os
import re
import shutil

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SYN = re.compile(r"_\d+$")            # name (without extension) ends with _<digits>


def find_label(img_path, src):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    d = os.path.dirname(img_path)
    for lp in (os.path.join(d, stem + ".txt"),                                   # same folder
               os.path.join(os.path.dirname(d), "labels", stem + ".txt"),        # ../labels/
               os.path.join(src, "labels", stem + ".txt")):                      # <src>/labels/
        if os.path.exists(lp):
            return lp
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing validation folder (not modified)")
    ap.add_argument("--out", required=True, help="NEW folder (gets images/ and labels/)")
    ap.add_argument("--dry_run", action="store_true", help="only count, copy nothing")
    opt = ap.parse_args()

    src, out = os.path.abspath(opt.src), os.path.abspath(opt.out)
    if out == src or out.startswith(src + os.sep):
        raise SystemExit("--out must be a NEW folder outside --src")

    imgs = []
    for d, _, files in os.walk(src):
        imgs += [os.path.join(d, f) for f in files if f.lower().endswith(EXTS)]
    imgs.sort()
    if not imgs:
        raise SystemExit(f"no images found in {src}")

    real, syn, no_label, dup = 0, 0, [], 0
    seen = set()
    if not opt.dry_run:
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, "labels"), exist_ok=True)

    for ip in imgs:
        name = os.path.basename(ip)
        stem = os.path.splitext(name)[0]
        if SYN.search(stem):
            syn += 1
            continue
        if name in seen:                    # same name in two sub-folders -> keep first
            dup += 1
            continue
        seen.add(name)
        lp = find_label(ip, src)
        if lp is None:
            no_label.append(name)
            continue                        # image without label is not copied
        real += 1
        if not opt.dry_run:
            shutil.copy2(ip, os.path.join(out, "images", name))
            shutil.copy2(lp, os.path.join(out, "labels", stem + ".txt"))

    print(f"source images          : {len(imgs)}")
    print(f"synthetic (_<n>) skipped: {syn}")
    print(f"REAL copied            : {real}" + ("  (dry run, nothing copied)" if opt.dry_run else f"  -> {out}"))
    if dup:
        print(f"duplicate names skipped: {dup}")
    if no_label:
        print(f"WARNING: {len(no_label)} real images have no label file (not copied), e.g. {no_label[:5]}")


if __name__ == "__main__":
    main()
