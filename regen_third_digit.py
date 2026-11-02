"""
regen_third_digit.py  -- re-generates the overlapping samples of a 3-digit folder (B) so that
their THIRD digit differs from the one in the reference folder (A).

For every name in --names (e.g. overlap_names.txt from compare_datasets.py):
  1. load the ORIGINAL base photo (2 real digits) from --src
  2. paste a new 3rd digit at the same place as before (right of digit 2, same gap,
     same height/top, colour of digit 2) -- the digit is chosen so it is DIFFERENT
     from A's 3rd digit (and from B's current one)
  3. same body crop as the first run (--crop_seed = seed used when B was made, default 42),
     so only the 3rd digit changes; save into B
Old files of B are not deleted: they are moved to <B>_replaced/ (images, labels, debug).

Needs make_syn3_dataset.py in the same folder (uses its drawing / crop functions).

Run (dry run first):
  python regen_third_digit.py --src <validation photos with YOLO labels> \
      --ref <folder A> --target <folder B> --names overlap_check/overlap_names.txt --seed 7 --dry_run
  python regen_third_digit.py ... (same, without --dry_run)
"""
import argparse
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_syn3_dataset import (body_like_crop, digit_colour, find_pairs, glyph_aspect,  # noqa: E402
                               load_font, make_digit, read_yolo, to_px)


def read_digits(path):
    if not os.path.exists(path):
        return None
    vals = open(path).read().split()
    return [int(v) for v in vals] if len(vals) >= 3 else None


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder with the ORIGINAL base photos + YOLO labels")
ap.add_argument("--ref", required=True, help="folder A (its 3rd digits must be avoided)")
ap.add_argument("--target", required=True, help="folder B (overlapping samples are re-made here)")
ap.add_argument("--names", required=True, help="names to re-make (one per line)")
ap.add_argument("--seed", type=int, default=7, help="seed for choosing the new 3rd digit")
ap.add_argument("--crop_seed", type=int, default=42,
                help="seed used when B was generated -> identical crop (only the digit changes)")
ap.add_argument("--font", default=None, help="bold .ttf (another font adds more variety)")
ap.add_argument("--dry_run", action="store_true")
opt = ap.parse_args()

tgt = os.path.abspath(opt.target.rstrip("/"))
bak = tgt + "_replaced"
names = [os.path.splitext(os.path.basename(l.strip()))[0] for l in open(opt.names) if l.strip()]
src = {os.path.splitext(os.path.basename(ip))[0]: (ip, lp) for ip, lp in find_pairs(opt.src)}
font = load_font(opt.font, 200)
rng = random.Random(opt.seed)

done, skipped, counts = 0, [], [0] * 10
for n in names:
    if n not in src:
        skipped.append((n, "base photo not in --src"))
        continue
    ip, lp = src[n]
    lab = read_yolo(lp)
    if lab is None or lab.shape[0] != 2:
        skipped.append((n, "base label is not a clean 2-digit number"))
        continue
    lab = lab[np.argsort(lab[:, 1])]
    img = Image.open(ip).convert("RGB")
    W, H = img.size
    b1, b2 = to_px(lab[0], W, H), to_px(lab[1], W, H)
    th = int(round(b2[3] - b2[1]))
    gap = max(0.0, b2[0] - b1[2])
    real_ar = np.mean([(b[2] - b[0]) / max(1.0, b[3] - b[1]) for b in (b1, b2)])
    font_ar = np.mean([glyph_aspect(int(lab[k][0]), font) for k in (0, 1)])
    w_scale = float(np.clip(real_ar / max(1e-3, font_ar), 0.6, 1.6))

    avoid = set()
    for folder in (opt.ref, opt.target):
        d = read_digits(os.path.join(folder, "labels", n + ".txt"))
        if d:
            avoid.add(d[2])
    d3 = rng.choice([d for d in range(10) if d not in avoid])

    sticker = make_digit(d3, digit_colour(img, b2), th, w_scale, opt.font)
    px, py = int(round(b2[2] + gap)), int(round(b2[1]))
    if th < 2 or px + sticker.width > W or py + th > H:
        skipped.append((n, "new digit does not fit"))
        continue
    img.paste(sticker, (px, py), sticker)
    x1, y1 = min(b1[0], b2[0]), min(b1[1], b2[1])
    y2 = max(b1[3], b2[3], py + th)
    crop = body_like_crop((x1, y1, float(px + sticker.width), y2), W, H, random.Random(f"{opt.crop_seed}-{n}"))
    digits = [int(lab[0][0]), int(lab[1][0]), d3]

    if opt.dry_run:
        print(f"would re-make {n}: 3rd digit {sorted(avoid)} -> {d3}")
    else:
        for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("debug", ".jpg")):
            old = os.path.join(tgt, sub, n + ext)
            if os.path.exists(old):
                os.makedirs(os.path.join(bak, sub), exist_ok=True)
                shutil.move(old, os.path.join(bak, sub, n + ext))
        os.makedirs(os.path.join(tgt, "images"), exist_ok=True)
        os.makedirs(os.path.join(tgt, "labels"), exist_ok=True)
        img.crop(crop).save(os.path.join(tgt, "images", n + ".jpg"), quality=95)
        with open(os.path.join(tgt, "labels", n + ".txt"), "w") as f:
            f.write(" ".join(map(str, digits)) + "\n")
        if os.path.isdir(os.path.join(tgt, "debug")) or os.path.isdir(os.path.join(bak, "debug")):
            dbg = img.copy()
            dr = ImageDraw.Draw(dbg)
            for b in (b1, b2):
                dr.rectangle(b, outline=(255, 0, 0), width=2)
            dr.rectangle((px, py, px + sticker.width, py + th), outline=(0, 255, 0), width=2)
            dr.rectangle(crop, outline=(0, 128, 255), width=2)
            os.makedirs(os.path.join(tgt, "debug"), exist_ok=True)
            dbg.save(os.path.join(tgt, "debug", n + ".jpg"), quality=90)
    counts[d3] += 1
    done += 1

print(f"\n{'would re-make' if opt.dry_run else 're-made'}: {done} | skipped: {len(skipped)}")
print(f"new 3rd-digit counts 0-9: {counts}")
for n, why in skipped[:10]:
    print(f"  skipped {n}: {why}")
if not opt.dry_run and done:
    print(f"old versions moved to: {bak}")
