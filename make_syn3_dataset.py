"""
make_syn3_dataset.py   (standalone, not imported by the project)

Makes a SEPARATE synthetic 3-digit dataset from real images with 2-digit numbers.
For every image with a clean 2-digit number (YOLO labels):
  1. pick a random digit 0-9 (NOT taken from any other image: it is drawn with a
     font, in the colour of the image's own 2nd digit)
  2. sticker = the digit with the training margins: 5% left/right, 15% top/bottom
  3. resize the sticker to the number height, paste it right after the 2nd digit:
     gap = 0.1 x height, top = y1 - 0.1 x height          (same as training)
  4. cut a body crop around the 3 digits (same as make_body_validation.py)

Output (validation format, in a NEW folder):
    <out>/images/<name>.jpg      body crop
    <out>/labels/<name>.txt      "d1 d2 d3"
    <out>/debug/<name>.jpg       full image: red = 2 real digits, green = pasted digit,
                                 blue = body crop   (with --debug, for checking)

Run:
    python make_syn3_dataset.py --src <dataset folder> --out ./syn3_dataset --num 400 --debug
--src: jpg + txt in the same folder, or images/ + labels/ ; labels "cls cx cy w h".
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "C:/Windows/Fonts/arialbd.ttf", "arialbd.ttf", "Arial Bold.ttf"]


def load_font(path, size):
    for p in ([path] if path else []) + FONTS:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    raise SystemExit("No TrueType font found: pass one with --font /path/to/bold.ttf")


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
    return (max(0.0, (r[1] - r[3] / 2) * W), max(0.0, (r[2] - r[4] / 2) * H),
            min(float(W), (r[1] + r[3] / 2) * W), min(float(H), (r[2] + r[4] / 2) * H))


def digit_colour(img, box):
    """Colour of the digit strokes inside a digit box (pixels least like the box border)."""
    a = np.asarray(img.crop(tuple(int(v) for v in box)), dtype=np.float32).reshape(-1, 3)
    patch = np.asarray(img.crop(tuple(int(v) for v in box)), dtype=np.float32)
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a - bg, axis=1)
    stroke = a[dist >= np.percentile(dist, 80)]
    return tuple(int(c) for c in stroke.mean(axis=0))


def make_sticker(digit, colour, height, font_path):
    """RGBA sticker: digit glyph + training margins (5% x, 15% y), scaled to `height`."""
    big = 200
    font = load_font(font_path, big)
    l, t, r, b = font.getbbox(str(digit))
    gw, gh = r - l, b - t
    mx, my = int(round(gw * 0.05)), int(round(gh * 0.15))
    canvas = Image.new("RGBA", (gw + 2 * mx, gh + 2 * my), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((mx - l, my - t), str(digit), font=font, fill=colour + (255,))
    w = max(1, int(round(canvas.width * height / canvas.height)))
    return canvas.resize((w, height), Image.LANCZOS)


def body_like_crop(box, W, H, rng, ratio=(0.15, 0.40), vpos=(0.15, 0.80)):
    """Same as make_body_validation.body_like_crop."""
    x1, y1, x2, y2 = box
    num_h, num_w = max(1.0, y2 - y1), max(1.0, x2 - x1)
    crop_h = num_h / rng.uniform(*ratio)
    v_pos = rng.uniform(*vpos)
    crop_w = max(num_w * 1.15, crop_h * rng.uniform(0.75, 1.10))
    y_min = y1 - (crop_h - num_h) * v_pos
    x_min = x1 - (crop_w - num_w) * rng.uniform(0.30, 0.70)
    y_max, x_max = y_min + crop_h, x_min + crop_w
    return (int(max(0.0, x_min)), int(max(0.0, y_min)),
            int(round(min(float(W), x_max))), int(round(min(float(H), y_max))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset folder with YOLO labels")
    ap.add_argument("--out", default="./syn3_dataset", help="NEW output folder")
    ap.add_argument("--num", type=int, default=400, help="max images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_digit_h", type=int, default=10, help="skip numbers smaller than this (px)")
    ap.add_argument("--font", default=None, help="bold .ttf font (default: DejaVu/Arial bold)")
    ap.add_argument("--debug", action="store_true", help="also save full images with boxes")
    opt = ap.parse_args()
    load_font(opt.font, 20)                                   # fail early if no font

    pairs = find_pairs(opt.src)
    bases = []
    for ip, lp in pairs:
        lab = read_yolo(lp)
        if lab is None or lab.shape[0] != 2 or int(lab[0][0]) == 10:
            continue
        if abs(lab[0][1] - lab[1][1]) > 0.25 or abs(lab[0][2] - lab[1][2]) > 0.25:
            continue                                          # 2 boxes far apart -> not one number
        bases.append((ip, lab[np.argsort(lab[:, 1])]))
    print(f"source images: {len(pairs)} | clean 2-digit: {len(bases)}")

    for sub in ("images", "labels") + (("debug",) if opt.debug else ()):
        os.makedirs(os.path.join(opt.out, sub), exist_ok=True)

    rng = random.Random(opt.seed)
    rng.shuffle(bases)
    made, skipped, counts = 0, 0, [0] * 10
    for ip, lab in bases:
        if opt.num and made >= opt.num:
            break
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        b1, b2 = to_px(lab[0], W, H), to_px(lab[1], W, H)
        x1, y1 = min(b1[0], b2[0]), min(b1[1], b2[1])
        x2, y2 = max(b1[2], b2[2]), max(b1[3], b2[3])
        th = int(y2 - y1)                                     # number height
        if th < opt.min_digit_h or b2[2] - b2[0] < 2:
            skipped += 1
            continue

        d3 = rng.randint(0, 9)                                # random 3rd digit
        sticker = make_sticker(d3, digit_colour(img, b2), th, opt.font)
        px, py = int(x2 + 0.1 * th), max(0, int(y1 - 0.1 * th))
        if px + sticker.width > W or py + th > H:
            skipped += 1                                      # does not fit in the photo
            continue
        img.paste(sticker, (px, py), sticker)                 # alpha: only the digit strokes

        name = os.path.splitext(os.path.basename(ip))[0]
        crop = body_like_crop((x1, y1, float(px + sticker.width), y2), W, H,
                              random.Random(f"{opt.seed}-{name}"))
        if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
            skipped += 1
            continue

        img.crop(crop).save(os.path.join(opt.out, "images", name + ".jpg"), quality=95)
        digits = [int(lab[0][0]), int(lab[1][0]), d3]
        with open(os.path.join(opt.out, "labels", name + ".txt"), "w") as f:
            f.write(" ".join(map(str, digits)) + "\n")

        if opt.debug:
            dbg = img.copy()
            dr = ImageDraw.Draw(dbg)
            for b in (b1, b2):
                dr.rectangle(b, outline=(255, 0, 0), width=2)
            dr.rectangle((px, py, px + sticker.width, py + th), outline=(0, 255, 0), width=2)
            dr.rectangle(crop, outline=(0, 128, 255), width=2)
            dr.text((crop[0] + 3, crop[1] + 3), " ".join(map(str, digits)), fill=(255, 255, 0))
            dbg.save(os.path.join(opt.out, "debug", name + ".jpg"), quality=90)

        counts[d3] += 1
        made += 1

    print(f"written: {made} | skipped: {skipped} | 3rd digit counts 0-9: {counts}")
    print(f"saved to: {opt.out}  (images/, labels/" + (", debug/)" if opt.debug else ")"))


if __name__ == "__main__":
    main()
