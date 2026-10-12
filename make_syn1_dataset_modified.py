"""
make_syn1_dataset.py   (standalone, not imported by the project)

Makes synthetic TRAINING images with an extra "1" (and some "7") next to a real
single-digit number, saved in training format (full photo + YOLO label).

For every real image with ONE digit (YOLO label with 1 box):
  1. take the real digit box: height, top line, colour of its strokes
  2. gap: sampled from the REAL gaps between digits of the 2-digit images in --src
     (only the middle 10-90 % of them, and at most --max_gap x height; the result
     must also pass the training rule: box centres within 0.25 of each other)
  3. draw a "1" (or "7") with the same height / top / colour / STROKE WIDTH as the
     real digit (measured from its pixels); a "7" also gets the real digit's width,
     before or after the
     real digit  ->  "14" / "41" / "11"  (or "74" / "47" / "77")
  4. "1" shapes : plain bar | bar + flag | bar + flag + base
     "7" shapes : top bar + diagonal | + small cross bar
     (drawn in code: no fonts needed)
  5. save the full photo + YOLO label (real box + new box), training format

Output (training format, jpg + txt in the SAME folder, like training_dataset_Ying):
    <out>/syn1_<name>.jpg  + .txt      (YOLO: cls cx cy w h, one row per digit)
    <out>/syn7_<name>.jpg  + .txt
    <out>/debug/...jpg                 red = real digit, green = drawn digit (--debug)

Run:
    python make_syn1_dataset.py --src <training folder> --out ./syn1_dataset \
        --num1 1500 --num7 750 --debug
"""
import argparse
import glob
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ------------------------------------------------------------------ io
def find_pairs(src):
    imgs = sorted(glob.glob(os.path.join(src, "images", "*.jpg")))
    if imgs:
        return [(ip, os.path.join(src, "labels", os.path.basename(ip)[:-4] + ".txt")) for ip in imgs]
    return [(ip, ip[:-4] + ".txt") for ip in sorted(glob.glob(os.path.join(src, "*.jpg")))]


def read_yolo(lp):
    if not os.path.exists(lp):
        return None
    try:
        d = np.loadtxt(lp, ndmin=2)
    except Exception:
        return None
    return d if d.size and d.shape[1] == 5 else None


def to_px(r, W, H):
    return ((r[1] - r[3] / 2) * W, (r[2] - r[4] / 2) * H,
            (r[1] + r[3] / 2) * W, (r[2] + r[4] / 2) * H)


def to_yolo(cls, b, W, H):
    x1, y1, x2, y2 = b
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def digit_colour(img, box):
    """Colour of the digit strokes in a box (pixels least like the box border)."""
    patch = np.asarray(img.crop(tuple(int(round(v)) for v in box)), dtype=np.float32)
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return (255, 255, 255)
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    bg = np.median(border, axis=0)
    a = patch.reshape(-1, 3)
    dist = np.linalg.norm(a - bg, axis=1)
    return tuple(int(c) for c in a[dist >= np.percentile(dist, 80)].mean(axis=0))


def stroke_width(img, box):
    """Stroke thickness of the REAL digit (px): 2 x typical distance-to-edge inside the strokes."""
    patch = np.asarray(img.crop(tuple(int(round(v)) for v in box)), dtype=np.float32)
    if patch.shape[0] < 6 or patch.shape[1] < 4:
        return None
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    dist = np.linalg.norm(patch - np.median(border, axis=0), axis=2)
    dist = (255 * dist / max(1e-6, dist.max())).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() < 5:                              # nothing found
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    return float(2 * np.percentile(vals, 90)) if vals.size else None


# ------------------------------------------------------------------ drawing
def draw_glyph(digit, h, colour, rng, sw_px, real_w, style=None):
    """RGBA image of a '1' or '7', glyph height = h px, tight box. Returns (img, style).
    sw_px  = stroke width of the real digit (px), real_w = width of the real digit box (px)."""
    S = 4                                           # draw 4x bigger, then downscale (anti-alias)
    H = h * S
    sw = sw_px * S * rng.uniform(0.9, 1.1)           # SAME thickness as the real digit (+-10 %)
    if digit == 1:
        style = style or rng.choice(["plain", "flag", "flag_base"])
        flag_w = min(H * rng.uniform(0.18, 0.28), real_w * S * 0.6) if style != "plain" else 0
        base_w = min(H * rng.uniform(0.40, 0.50), real_w * S) if style == "flag_base" else 0
        W = int(max(sw + flag_w, base_w, sw) + sw) + 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = W - sw / 2 - (base_w / 2 - sw / 2 if base_w else 0) - sw / 2   # vertical bar x
        x = max(x, sw / 2 + flag_w)
        d.line([(x, 0), (x, H)], fill=colour + (255,), width=int(sw))
        if flag_w:                                  # short slanted flag at the top-left
            ang = math.radians(rng.uniform(25, 45))
            d.line([(x, sw * 0.4), (x - flag_w, sw * 0.4 + flag_w * math.tan(ang))],
                   fill=colour + (255,), width=int(sw * 0.9))
        if base_w:                                  # horizontal foot
            d.line([(x - base_w / 2, H - sw / 2), (x + base_w / 2, H - sw / 2)],
                   fill=colour + (255,), width=int(sw * 0.9))
    else:                                           # 7
        style = style or rng.choice(["plain", "plain", "cross"])
        top_w = real_w * S * rng.uniform(0.9, 1.05) - sw   # same width as the real digit
        W = int(top_w + sw) + 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, x1 = sw / 2, W - sw / 2
        d.line([(x0, sw / 2), (x1, sw / 2)], fill=colour + (255,), width=int(sw))          # top bar
        xb = x0 + (x1 - x0) * rng.uniform(0.15, 0.35)                                     # foot x
        d.line([(x1, sw / 2), (xb, H)], fill=colour + (255,), width=int(sw))              # diagonal
        if style == "cross":
            ym = H * 0.5
            xm = x1 + (xb - x1) * (ym / H)
            d.line([(xm - top_w * 0.25, ym), (xm + top_w * 0.25, ym)], fill=colour + (255,), width=int(sw * 0.8))
    img = img.resize((max(1, round(img.width / S)), h), Image.LANCZOS)
    alpha = img.getchannel("A")
    bb = alpha.getbbox()                             # tight box around the drawn strokes
    if bb:
        img = img.crop((bb[0], 0, bb[2], h))
    return img, f"sw{sw_px:.1f}px_{style}"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="real training folder (YOLO labels)")
    ap.add_argument("--out", default="./syn1_dataset", help="NEW output folder (training format)")
    ap.add_argument("--num1", type=int, default=1500, help="images with an added '1'")
    ap.add_argument("--num7", type=int, default=750, help="images with an added '7'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_digit_h", type=int, default=10, help="skip digits smaller than this (px)")
    ap.add_argument("--max_gap", type=float, default=0.30, help="largest gap allowed, x digit height")
    ap.add_argument("--debug", action="store_true", help="save full images with boxes in <out>/debug")
    opt = ap.parse_args()
    rng = random.Random(opt.seed)

    singles, gaps = [], []
    for ip, lp in find_pairs(opt.src):
        lab = read_yolo(lp)
        if lab is None or int(lab[0][0]) == 10:
            continue
        if lab.shape[0] == 1:
            singles.append((ip, lab[0]))
        elif lab.shape[0] == 2 and abs(lab[0][1] - lab[1][1]) <= 0.25 and abs(lab[0][2] - lab[1][2]) <= 0.25:
            l = lab[np.argsort(lab[:, 1])]          # real gap / height, from normalised boxes
            hgt = max(l[0][4], l[1][4])
            with Image.open(ip) as im:
                W, H = im.size
            b1, b2 = to_px(l[0], W, H), to_px(l[1], W, H)
            gaps.append(max(0.0, b2[0] - b1[2]) / max(1.0, max(b1[3] - b1[1], b2[3] - b2[1])))
    if not gaps:
        gaps = [0.08]
    lo, hi = np.percentile(gaps, [10, 90])           # drop odd boxes / outliers
    gaps = [g for g in gaps if lo <= g <= min(hi, opt.max_gap)] or [min(np.median(gaps), opt.max_gap)]
    print(f"single-digit images: {len(singles)} | real gaps measured: {len(gaps)} "
          f"(median {np.median(gaps):.2f} x height)")
    if not singles:
        print("nothing to do (needs YOLO labels with 1 box)")
        return

    os.makedirs(opt.out, exist_ok=True)
    if opt.debug:
        os.makedirs(os.path.join(opt.out, "debug"), exist_ok=True)

    jobs = [1] * opt.num1 + [7] * opt.num7
    rng.shuffle(jobs)
    order = singles[:]
    rng.shuffle(order)
    made, skipped, k = {1: 0, 7: 0}, 0, 0
    counts = {}
    for j, new_d in enumerate(jobs):
        for _ in range(10):                          # try a few bases until one fits
            ip, r = order[k % len(order)]
            k += 1
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            bx = to_px(r, W, H)
            h = int(round(bx[3] - bx[1]))
            if h < opt.min_digit_h or bx[0] < 0 or bx[2] > W:
                continue
            sw = stroke_width(img, bx)
            if sw is None:
                continue
            sw = float(np.clip(sw, 0.06 * h, 0.25 * h))
            glyph, style = draw_glyph(new_d, h, digit_colour(img, bx), rng, sw, bx[2] - bx[0])
            gap = rng.choice(gaps) * h
            before = rng.random() < 0.5
            px = int(round(bx[0] - gap - glyph.width)) if before else int(round(bx[2] + gap))
            py = int(round(bx[1]))
            if px < 0 or px + glyph.width > W or py < 0 or py + h > H:
                continue
            # training rule: 2 boxes are ONE number only if centres are within 0.25 (x and y)
            ncx = (px + glyph.width / 2) / W
            if abs(ncx - r[1]) > 0.25:
                continue
            if rng.random() < 0.5:                   # slight blur, like a real photo
                glyph = glyph.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
            img.paste(glyph, (px, py), glyph)
            nb = (px, py, px + glyph.width, py + h)
            real_d = int(r[0])
            number = f"{new_d}{real_d}" if before else f"{real_d}{new_d}"
            name = f"syn{new_d}_{j:05d}_{os.path.splitext(os.path.basename(ip))[0]}"
            img.save(os.path.join(opt.out, name + ".jpg"), quality=95)
            with open(os.path.join(opt.out, name + ".txt"), "w") as f:
                f.write(to_yolo(real_d, bx, W, H) + "\n" + to_yolo(new_d, nb, W, H) + "\n")
            if opt.debug:
                dbg = img.copy()
                dr = ImageDraw.Draw(dbg)
                dr.rectangle(bx, outline=(255, 0, 0), width=2)
                dr.rectangle(nb, outline=(0, 255, 0), width=2)
                dr.text((min(bx[0], nb[0]), max(0, py - 12)), f"{number} {style}", fill=(255, 255, 0))
                dbg.save(os.path.join(opt.out, "debug", name + ".jpg"), quality=90)
            made[new_d] += 1
            counts[number] = counts.get(number, 0) + 1
            break
        else:
            skipped += 1

    print(f"written: '1' = {made[1]}, '7' = {made[7]} | skipped: {skipped}")
    print("numbers made:", dict(sorted(counts.items(), key=lambda x: -x[1])[:20]), "...")
    print(f"saved to: {opt.out}  (jpg + txt, training format)")


if __name__ == "__main__":
    main()
