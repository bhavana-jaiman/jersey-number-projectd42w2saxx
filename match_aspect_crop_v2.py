#!/usr/bin/env python3
"""
match_aspect_crop.py  --  bring existing crops closer to a target aspect ratio (w/h) by CROPPING ONLY.

Guarantees
  * NO padding, NO mirroring, NO blur fill, NO resizing/stretching: every output pixel is an original pixel.
  * Same file names as the input -> the existing labels still match. Labels are never modified.
  * Images already inside the tolerance band are copied byte-for-byte (not even re-encoded).
  * Cropping only removes the part that is least likely to hold the number:
        too tall/narrow  -> trim from the BOTTOM  (legs; the number is in the upper body)
        too wide         -> trim equally from LEFT and RIGHT
  * --min-keep-h / --min-keep-w cap how much may be removed. If the target cannot be reached within that cap,
    the image is cropped only as far as the cap allows (action '...-limited') -- it is never padded to make up for it.

Usage
  python match_aspect_crop.py SRC --out OUT --target-aspect 0.75 --dry-run          # just count, write nothing
  python match_aspect_crop.py SRC --out OUT --target-aspect 0.75 --copy-labels --preview 12
SRC may be the folder with the images, or the dataset folder that contains images/ (and labels/).
"""
import argparse, csv, math, os, random, shutil
from collections import Counter
from PIL import Image

EXT = (".jpg", ".jpeg", ".png")


def plan(w, h, target, tol, aim, min_keep_h, min_keep_w):
    """Return (action, box) ; box = (x0, y0, x1, y1) inside the original image."""
    asp = w / h
    lo, hi = target * (1 - tol), target * (1 + tol)
    if lo <= asp <= hi:
        return "keep", (0, 0, w, h)
    goal = target if aim == "target" else (lo if asp < lo else hi)      # aim at the target, or just the band edge
    if asp < lo:                                                         # too tall/narrow -> trim bottom
        want_h = math.floor(w / goal + 1e-9)          # floor => aspect >= goal, never just outside the band
        floor_h = math.ceil(min_keep_h * h - 1e-9)
        new_h = max(want_h, floor_h)
        return ("crop-bottom" if want_h >= floor_h else "crop-bottom-limited"), (0, 0, w, min(h, new_h))
    want_w = math.floor(h * goal + 1e-9)                                 # floor => aspect <= goal
    floor_w = math.ceil(min_keep_w * w - 1e-9)
    new_w = max(want_w, floor_w)
    x0 = (w - new_w) // 2
    return ("crop-sides" if want_w >= floor_w else "crop-sides-limited"), (x0, 0, x0 + new_w, h)


def save_crop(im, box, path, src_path):
    ext = os.path.splitext(path)[1].lower()
    crop = im.crop(box)
    if ext in (".jpg", ".jpeg"):
        crop.save(path, quality=100, subsampling=0)                      # minimal re-compression loss
    else:
        crop.save(path)                                                  # png: lossless


def contact_sheet(pairs, out_path, H=220, cols=3):
    tiles = []
    for b, n in pairs:
        b = b.resize((max(1, round(b.width * H / b.height)), H)); n = n.resize((max(1, round(n.width * H / n.height)), H))
        t = Image.new("RGB", (b.width + n.width + 24, H), (255, 255, 255)); t.paste(b, (0, 0)); t.paste(n, (b.width + 24, 0))
        tiles.append(t)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    W = max(sum(t.width + 30 for t in r) for r in rows)
    sheet = Image.new("RGB", (W, len(rows) * (H + 20)), (200, 200, 200))
    for ri, r in enumerate(rows):
        x = 0
        for t in r:
            sheet.paste(t, (x, ri * (H + 20))); x += t.width + 30
    sheet.save(out_path, quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("--out", required=True)
    p.add_argument("--target-aspect", type=float, required=True, help="w/h, e.g. 0.75 (median of your test data)")
    p.add_argument("--tol", type=float, default=0.10, help="images within +-tol of the target are left untouched")
    p.add_argument("--aim", choices=["target", "edge"], default="target",
                   help="target: crop to the exact target; edge: crop only to the nearest edge of the tolerance band (removes less)")
    p.add_argument("--min-keep-h", type=float, default=0.60, help="never keep less than this fraction of the height")
    p.add_argument("--min-keep-w", type=float, default=0.80, help="never keep less than this fraction of the width")
    p.add_argument("--copy-labels", action="store_true", help="copy labels/ next to the new images, unchanged")
    p.add_argument("--preview", type=int, default=0, help="save contact sheets: N most-cropped + N random (left=before, right=after)")
    p.add_argument("--dry-run", action="store_true", help="only report what would happen; write nothing")
    p.add_argument("--report-dir", default="crop_report")
    a = p.parse_args()

    src = os.path.join(a.src, "images") if os.path.isdir(os.path.join(a.src, "images")) else a.src
    out_img = os.path.join(a.out, "images")
    if not a.dry_run:
        if os.path.exists(a.out) and os.listdir(a.out):
            raise SystemExit(f"refusing to write into non-empty {a.out}; choose a new --out folder")
        os.makedirs(out_img, exist_ok=True)

    rows, acts = [], Counter()
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXT))
    for f in files:
        sp = os.path.join(src, f)
        with Image.open(sp) as im:
            im.load(); w, h = im.size
            act, box = plan(w, h, a.target_aspect, a.tol, a.aim, a.min_keep_h, a.min_keep_w)
            nw, nh = box[2] - box[0], box[3] - box[1]
            if not a.dry_run:
                dp = os.path.join(out_img, f)
                if act == "keep": shutil.copy2(sp, dp)                   # byte-for-byte
                else: save_crop(im, box, dp, sp)
        acts[act] += 1
        final = nw / nh
        in_band = abs(final - a.target_aspect) / a.target_aspect <= a.tol + 1e-9
        rows.append(dict(file=f, w=w, h=h, aspect=round(w / h, 3), action=act, new_w=nw, new_h=nh,
                         new_aspect=round(final, 3), in_tolerance=in_band, pixels_removed_pct=round(100 * (1 - nw * nh / (w * h)), 1)))

    os.makedirs(a.report_dir, exist_ok=True)
    rp = os.path.join(a.report_dir, "match_aspect_report.csv")
    with open(rp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    n = len(rows); ok = sum(r["in_tolerance"] for r in rows)
    print(f"{'[dry-run] ' if a.dry_run else ''}{n} images   target aspect {a.target_aspect} (+-{int(a.tol*100)}%)")
    for k, v in acts.most_common(): print(f"  {k:<22}{v:>6}  {100*v/n:4.1f}%")
    print(f"  reach the tolerance band: {ok} ({100*ok/n:.1f}%);  stay outside it because of the keep limits: {n-ok}")
    rm = [r["pixels_removed_pct"] for r in rows if r["action"] != "keep"]
    if rm: print(f"  pixels removed in cropped images: median {sorted(rm)[len(rm)//2]}%  max {max(rm)}%")
    print(f"report: {rp}")

    if a.copy_labels and not a.dry_run:
        lab = os.path.join(os.path.dirname(os.path.normpath(src)), "labels")
        if os.path.isdir(lab):
            shutil.copytree(lab, os.path.join(a.out, "labels")); print(f"labels copied unchanged -> {os.path.join(a.out, 'labels')}")
        else:
            print(f"(no labels/ folder next to {src}; nothing copied)")

    if a.preview and not a.dry_run:
        cropped = [r for r in rows if r["action"] != "keep"]
        random.seed(0)
        sets = {"preview_most_cropped.jpg": sorted(cropped, key=lambda r: -r["pixels_removed_pct"])[:a.preview],
                "preview_random.jpg": random.sample(cropped, min(a.preview, len(cropped)))}
        for name, sel in sets.items():
            if not sel: continue
            pairs = [(Image.open(os.path.join(src, r["file"])).convert("RGB"),
                      Image.open(os.path.join(out_img, r["file"])).convert("RGB")) for r in sel]
            contact_sheet(pairs, os.path.join(a.report_dir, name))
        print(f"previews (left=before, right=after): {a.report_dir}/preview_most_cropped.jpg and preview_random.jpg")


if __name__ == "__main__":
    main()
