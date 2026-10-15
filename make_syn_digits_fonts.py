"""
make_syn_digits_fonts.py   (standalone, not imported by the project)   -- v2
Extension of make_syn1_dataset_modified.py: same placement logic (real single-digit
photo, same height / top / ink colour / stroke width / real gap, training rule), now
with a PLAN of how many images per DIGIT x FONT GROUP, for digits 6, 7, 1, 8.

Font groups (edit FONT_GROUPS; fonts are searched on this machine, missing ones skipped):
  varsity : collegiate block, outlined (fill + light ring + dark ring)   <- your "6" example
  confusable: drawn 1 / 7 pair from your examples (1 and 7 only):
            1 = heavy vertical stem + short thick almost-horizontal flag, no base
            7 = longer top bar + steep, almost vertical stem, no cross bar
  mono    : typewriter / slab digits
  bib     : Arial / Helvetica bold, 1 with a flag, race bibs
  athletic: bold condensed shirt numbers
  slab    : serif feet (1 with a base), heavy curves
  thin    : light regular weight (thin 7 / 1)
  rounded : rounded / handwritten-like (only if such fonts exist)
  drawn   : the old line-drawn 1 / 7 shapes (1, 7 only)
A group without fonts gives its count to the other groups of that digit (printed).
"varsity" without a collegiate font falls back to athletic/slab fonts + outline.

Output (never overwrites: --out must be new or empty, and must not be inside --src):
  --mode train : <out>/syn{d}_*.jpg + .txt   full photo + YOLO "cls cx cy w h" (2 rows)
  --mode val   : <out>/images/*.jpg + <out>/labels/*.txt   body crop + "d1 d2"
  <out>/preview.png   grid: one row per digit x font group, zoomed on the number
  <out>/log.csv       name, digit, number, group, font, style, base image
  <out>/debug/        (with --debug) full photo with boxes
Report printed at the end: original count / generated / final for each digit.

Run:
  python make_syn_digits_fonts.py --list_fonts
  python make_syn_digits_fonts.py --mode train --src <train YOLO folder> --out <...>/syn_digits_train --debug
  python make_syn_digits_fonts.py --mode val   --src <val YOLO folder>   --out <...>/syn_digits_val --scale 0.15
  custom plan:  --plan "1:drawn=300,bib=300;6:varsity=150"   (digit:group=count, ';' between digits)
"""
import argparse
import csv
import glob
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------ config (edit here)
FONT_GROUPS = {
    "varsity":  ["Graduate-Regular.ttf", "Freshman.ttf", "College.ttf", "collegeb.ttf", "Varsity.ttf",
                 "VarsityTeam-Bold.otf", "Staatliches-Regular.ttf", "BlackOpsOne-Regular.ttf",
                 "AlfaSlabOne-Regular.ttf", "Bungee-Regular.ttf"],
    "mono":     ["FreeMonoBold.ttf", "courbd.ttf", "CourierPrime-Bold.ttf", "LiberationMono-Bold.ttf",
                 "DejaVuSansMono-Bold.ttf", "LiberationMono-Regular.ttf"],
    "bib":      ["LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "FreeSansBold.ttf",
                 "DejaVuSans-Bold.ttf", "Roboto-Bold.ttf", "Inter-Bold.otf", "NotoSans-Bold.ttf"],
    "athletic": ["Oswald-Bold.ttf", "Oswald-SemiBold.ttf", "RobotoCondensed-Bold.ttf",
                 "LiberationSansNarrow-Bold.ttf", "DejaVuSansCondensed-Bold.ttf",
                 "ARIALNB.TTF", "arialnb.ttf", "impact.ttf", "Impact.ttf", "bahnschrift.ttf"],
    "slab":     ["DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf", "FreeSerifBold.ttf",
                 "timesbd.ttf", "RobotoSlab-Bold.ttf", "NotoSerif-Bold.ttf", "georgiab.ttf"],
    "thin":     ["DejaVuSans.ttf", "DejaVuSansCondensed.ttf", "LiberationSans-Regular.ttf",
                 "LiberationSansNarrow-Regular.ttf", "arial.ttf", "Roboto-Regular.ttf",
                 "Inter-Regular.otf", "FreeSans.ttf"],
    "rounded":  ["ARLRDBD.TTF", "Nunito-Bold.ttf", "VarelaRound-Regular.ttf", "Comfortaa-Bold.ttf",
                 "Quicksand-Bold.ttf", "ComicNeue-Bold.ttf", "comicbd.ttf", "Kalam-Bold.ttf",
                 "PatrickHand-Regular.ttf"],
}
OUTLINE_GROUPS = {"varsity"}          # drawn with fill + light ring + dark ring
# training plan: images per digit x group (original counts 1:424, 6:3214, 7:3444, 8:2069)
#   1 -> +1600 (rarest digit, read as 7)   8 -> +400   6 -> +300   7 -> +250 (styles only)
#   "confusable" = your 1 / 7 examples: block 1 with short flat flag vs 7 with steep stem
PLAN = {
    1: {"confusable": 500, "drawn": 200, "bib": 250, "thin": 200, "slab": 200, "athletic": 150,
        "varsity": 50, "mono": 50},
    6: {"varsity": 100, "athletic": 60, "bib": 50, "slab": 50, "rounded": 40},
    7: {"confusable": 150, "thin": 30, "athletic": 30, "mono": 20, "varsity": 20},
    8: {"varsity": 100, "athletic": 100, "bib": 80, "slab": 80, "rounded": 40},
}
FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
             os.path.expanduser("~/.local/share/fonts"), "C:/Windows/Fonts", "/Library/Fonts",
             "/System/Library/Fonts"]
ITALIC_P = 0.20          # share of font glyphs given a small italic slant
ITALIC_MAX = 0.18        # max shear (about 10 degrees)
PREVIEW_N = 8            # samples per row in preview.png


# ------------------------------------------------------------------ fonts
def index_fonts(extra_dirs):
    idx = {}
    for d in extra_dirs + FONT_DIRS:
        for ext in ("ttf", "otf", "TTF", "OTF"):
            for p in glob.glob(os.path.join(d, "**", "*." + ext), recursive=True):
                idx.setdefault(os.path.basename(p).lower(), p)
    return idx


def font_ok(path, size=64):
    """real digit glyphs: every digit non-empty and digits not all the same shape (no tofu)."""
    try:
        f = ImageFont.truetype(path, size)
        masks = []
        for c in "0123456789":
            im = Image.new("L", (size * 2, size * 2), 0)
            ImageDraw.Draw(im).text((size // 2, size // 4), c, font=f, fill=255)
            masks.append(np.asarray(im) > 0)
    except Exception:
        return False
    return all(m.sum() >= 20 for m in masks) and len({m.tobytes() for m in masks}) >= 9


def resolve_fonts(extra_dirs, extra_files):
    idx = index_fonts(extra_dirs)
    groups = {}
    for g, names in FONT_GROUPS.items():
        groups[g] = []
        for n in names:
            p = idx.get(n.lower())
            if p and p not in groups[g] and font_ok(p):
                groups[g].append(p)
    for spec in extra_files:                     # --fonts group=path
        g, p = spec.split("=", 1) if "=" in spec else ("bib", spec)
        if os.path.exists(p) and font_ok(p):
            groups.setdefault(g, []).append(p)
        else:
            print(f"  skipped (missing / no digit glyphs): {p}")
    return groups


# ------------------------------------------------------------------ io (unchanged)
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


def box_colours(img, box):
    """(ink colour of the digit strokes, background colour around them)."""
    patch = np.asarray(img.crop(tuple(int(round(v)) for v in box)), dtype=np.float32)
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return (255, 255, 255), (0, 0, 0)
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    bg = np.median(border, axis=0)
    a = patch.reshape(-1, 3)
    dist = np.linalg.norm(a - bg, axis=1)
    # stroke core (Otsu mask, eroded): edge pixels mixed with the background do not shift it
    d8 = (255 * dist / max(1e-6, dist.max())).astype(np.uint8).reshape(patch.shape[:2])
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8)) if min(mask.shape) > 8 else mask
    sel = a[core.reshape(-1) > 0]
    if sel.shape[0] < 5:
        sel = a[dist >= np.percentile(dist, 80)]
    return tuple(int(c) for c in np.median(sel, axis=0)), tuple(int(c) for c in bg)


def stroke_width(img, box):
    """Stroke thickness of the REAL digit (px): 2 x typical distance-to-edge inside the strokes."""
    patch = np.asarray(img.crop(tuple(int(round(v)) for v in box)), dtype=np.float32)
    if patch.shape[0] < 6 or patch.shape[1] < 4:
        return None
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    dist = np.linalg.norm(patch - np.median(border, axis=0), axis=2)
    dist = (255 * dist / max(1e-6, dist.max())).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() < 5:
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    return float(2 * np.percentile(vals, 90)) if vals.size else None


def mask_stroke(mask):
    dt = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt[dt > 0]
    return float(2 * np.percentile(vals, 90)) if vals.size else 0.0


def body_like_crop(box, W, H, rng, ratio=(0.15, 0.40), vpos=(0.15, 0.80)):
    """Same as make_body_validation.body_like_crop / make_syn3_dataset."""
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


# ------------------------------------------------------------------ glyphs
def draw_confusable(digit, h, colour, rng, sw_px):
    """The 1 / 7 pair from your examples, drawn so the ONLY difference is the real one:
    1 = vertical stem + SHORT, thick, almost horizontal flag at the top-left (no base)
    7 = LONGER top bar + STEEP, almost vertical stem going down-left (no cross bar)"""
    S = 4
    H = h * S
    sw = sw_px * S * rng.uniform(0.95, 1.15)                     # heavy, like the examples
    poly = []
    if digit == 1:
        L = H * rng.uniform(0.18, 0.30)                          # flag left of the stem (x height)
        ft = sw * rng.uniform(0.75, 1.0)                          # flag thickness
        drop = L * math.tan(math.radians(rng.uniform(0, 15)))   # flag almost horizontal
        W = int(L + sw) + 2
        poly.append([(L, 0), (L + sw, 0), (L + sw, H), (L, H)])                 # stem
        poly.append([(0, drop), (L + 1, 0), (L + 1, ft), (0, drop + ft)])       # flag
        style = "block_flag"
    else:
        Wt = sw + H * rng.uniform(0.36, 0.50)                     # bar reaches 0.36-0.50 h left of the stem
        ang = math.radians(rng.uniform(6, 18))                    # stem angle from vertical
        W = int(Wt) + 2
        x1 = Wt                                                   # stem starts at the bar's right end
        xb = max(sw, x1 - (H - sw) * math.tan(ang))               # foot x
        poly.append([(0, 0), (x1, 0), (x1, sw), (0, sw)])                       # top bar
        poly.append([(x1 - sw * 1.05, 0), (x1, 0), (xb, H), (xb - sw * 1.05, H)])  # steep stem
        style = "steep"
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for p in poly:
        d.polygon(p, fill=colour + (255,))
    img = img.resize((max(1, round(W / S)), h), Image.LANCZOS)
    bb = img.getchannel("A").getbbox()
    if bb:
        img = img.crop((bb[0], 0, bb[2], h))
    return img, "drawn", style


def draw_glyph(digit, h, colour, rng, sw_px, real_w):
    """Old drawn '1' / '7' (unchanged logic)."""
    S = 4
    H = h * S
    sw = sw_px * S * rng.uniform(0.9, 1.1)
    if digit == 1:
        style = rng.choice(["plain", "flag", "flag_base"])
        flag_w = min(H * rng.uniform(0.18, 0.28), real_w * S * 0.6) if style != "plain" else 0
        base_w = min(H * rng.uniform(0.40, 0.50), real_w * S) if style == "flag_base" else 0
        W = int(max(sw + flag_w, base_w, sw) + sw) + 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = W - sw / 2 - (base_w / 2 - sw / 2 if base_w else 0) - sw / 2
        x = max(x, sw / 2 + flag_w)
        d.line([(x, 0), (x, H)], fill=colour + (255,), width=int(sw))
        if flag_w:
            ang = math.radians(rng.uniform(25, 45))
            d.line([(x, sw * 0.4), (x - flag_w, sw * 0.4 + flag_w * math.tan(ang))],
                   fill=colour + (255,), width=int(sw * 0.9))
        if base_w:
            d.line([(x - base_w / 2, H - sw / 2), (x + base_w / 2, H - sw / 2)],
                   fill=colour + (255,), width=int(sw * 0.9))
    else:
        style = rng.choice(["plain", "plain", "cross"])
        top_w = real_w * S * rng.uniform(0.9, 1.05) - sw
        W = int(top_w + sw) + 2
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, x1 = sw / 2, W - sw / 2
        d.line([(x0, sw / 2), (x1, sw / 2)], fill=colour + (255,), width=int(sw))
        xb = x0 + (x1 - x0) * rng.uniform(0.15, 0.35)
        d.line([(x1, sw / 2), (xb, H)], fill=colour + (255,), width=int(sw))
        if style == "cross":
            ym = H * 0.5
            xm = x1 + (xb - x1) * (ym / H)
            d.line([(xm - top_w * 0.25, ym), (xm + top_w * 0.25, ym)], fill=colour + (255,), width=int(sw * 0.8))
    img = img.resize((max(1, round(img.width / S)), h), Image.LANCZOS)
    bb = img.getchannel("A").getbbox()
    if bb:
        img = img.crop((bb[0], 0, bb[2], h))
    return img, "drawn", f"drawn_{style}"


_FONT_CACHE = {}


def tight(a):
    ys, xs = np.where(a > 20)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else None


def font_glyph(digit, h, colour, bg, rng, sw_px, font_path, outline):
    """Font digit, total height h, stroke matched to the real digit; outline = varsity style."""
    S = 4
    key = (font_path, h)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(font_path, int(h * S * 1.4))
    f = _FONT_CACHE[key]
    l, t, r, b = f.getbbox(str(digit))
    pad = int(h * S * 0.3)
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), str(digit), font=f, fill=255)
    a = np.asarray(m)
    shear = 0.0
    if rng.random() < ITALIC_P:
        shear = rng.uniform(0.08, ITALIC_MAX)
        M = np.float32([[1, -shear, shear * a.shape[0] / 2], [0, 1, 0]])
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
    a = tight(a)
    if a is None:
        return None, "", ""
    # outline rings take part of the height: fill height = h minus the rings
    ring1 = ring2 = 0.0
    if outline:
        ring1, ring2 = sw_px * rng.uniform(0.18, 0.28), sw_px * rng.uniform(0.15, 0.25)
    hf = max(4, int(round(h - 2 * (ring1 + ring2))))
    w = max(1, int(round(a.shape[1] * hf / a.shape[0] * rng.uniform(0.92, 1.08))))
    a = cv2.resize(a, (w * S, hf * S), interpolation=cv2.INTER_AREA)
    # stroke of the FILL matched to the real digit (+-10 %)
    target = sw_px * S * rng.uniform(0.9, 1.1) * (0.75 if outline else 1.0)
    cur = mask_stroke(a)
    if cur > 0 and abs(target - cur) / target > 0.15:
        k = int(round(abs(target - cur) / 2))
        if k >= 1:
            a = np.pad(a, k + 1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            a = cv2.dilate(a, ker) if target > cur else cv2.erode(a, ker)
            a = tight(a)
            if a is None:
                return None, "", ""
            a = cv2.resize(a, (max(1, int(round(a.shape[1] * hf * S / a.shape[0]))), hf * S),
                           interpolation=cv2.INTER_AREA)
    if outline:
        r1, r2 = max(1, int(round(ring1 * S))), max(1, int(round(ring2 * S)))
        a = np.pad(a, r1 + r2 + 1)
        e1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r1 + 1, 2 * r1 + 1))
        e2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (r1 + r2) + 1, 2 * (r1 + r2) + 1))
        d1, d2 = cv2.dilate(a, e1), cv2.dilate(a, e2)
        lum = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
        light = (240, 240, 240) if lum < 128 else tuple(int(c * 0.35) for c in bg)
        rgb = np.zeros(a.shape + (3,), np.float32)
        for c, (fc, rc) in enumerate(zip(colour, light)):          # outer ring = fill colour
            ch = np.full(a.shape, fc, np.float32)
            ring = (d1.astype(np.float32) - a) / 255.0
            ch = ch * (1 - ring) + rc * ring
            rgb[..., c] = ch
        alpha = d2
        ys, xs = np.where(alpha > 20)
        rgb, alpha = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1], alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ow = max(1, int(round(alpha.shape[1] * h / alpha.shape[0])))
        rgb = cv2.resize(rgb, (ow, h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (ow, h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]), "RGBA")
    else:
        a = cv2.resize(a, (max(1, round(a.shape[1] / S)), hf), interpolation=cv2.INTER_AREA)
        img = Image.new("RGBA", (a.shape[1], hf), colour + (0,))
        img.putalpha(Image.fromarray(a))
    name = os.path.splitext(os.path.basename(font_path))[0]
    return img, name, ("outline" if outline else "plain") + ("_ital" if shear else "")


# ------------------------------------------------------------------ helpers
def parse_plan(s):
    plan = {d: dict(g) for d, g in PLAN.items()}
    if s:
        plan = {}
        for part in s.split(";"):
            d, items = part.split(":")
            plan[int(d)] = {g: int(n) for g, n in (it.split("=") for it in items.split(","))}
    return plan


def count_digits(src):
    c = [0] * 11
    for _, lp in find_pairs(src):
        lab = read_yolo(lp)
        if lab is not None:
            for row in lab:
                if 0 <= int(row[0]) <= 10:
                    c[int(row[0])] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="real YOLO folder (train folder for --mode train, val folder for --mode val)")
    ap.add_argument("--out", default="./syn_digits", help="NEW output folder (must be new or empty)")
    ap.add_argument("--mode", choices=["train", "val"], default="train")
    ap.add_argument("--plan", default="", help='e.g. "1:drawn=300,bib=300;6:varsity=150" (default: PLAN)')
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every plan count (val: 0.15)")
    ap.add_argument("--font_dir", action="append", default=[], help="extra folder with .ttf/.otf")
    ap.add_argument("--fonts", nargs="*", default=[], help="extra fonts as group=path")
    ap.add_argument("--list_fonts", action="store_true", help="print fonts found per group and exit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_digit_h", type=int, default=10)
    ap.add_argument("--max_gap", type=float, default=0.30)
    ap.add_argument("--debug", action="store_true")
    opt = ap.parse_args()

    groups = resolve_fonts(opt.font_dir, opt.fonts)
    print("fonts found:")
    for g, ps in groups.items():
        print(f"  {g:9s}: " + (", ".join(os.path.basename(p) for p in ps) if ps else "-- none --"))
    if opt.list_fonts:
        return
    if not opt.src:
        raise SystemExit("--src is required")
    out, src = os.path.abspath(opt.out), os.path.abspath(opt.src)
    if out == src or out.startswith(src + os.sep):
        raise SystemExit("--out must not be the source folder or inside it")
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} is not empty: choose a new --out (nothing is overwritten)")

    # plan -> jobs (digit, group, font list, outline); redistribute groups without fonts
    plan = parse_plan(opt.plan)
    jobs = []
    print("plan (digit: group=count):")
    for d, gs in plan.items():
        gs = {g: int(round(n * opt.scale)) for g, n in gs.items()}
        avail = {}
        for g, n in gs.items():
            if g in ("drawn", "confusable") and d in (1, 7):
                avail[g] = n
            elif groups.get(g):
                avail[g] = n
            elif g == "varsity" and (groups.get("athletic") or groups.get("slab")):
                avail[g] = n                                     # fallback fonts + outline
            else:
                print(f"  digit {d}: no fonts for '{g}', its {n} images go to the other groups")
        lost = sum(gs.values()) - sum(avail.values())
        if not avail:
            raise SystemExit(f"no font available for digit {d}")
        for i, g in enumerate(list(avail)):
            avail[g] += lost // len(avail) + (1 if i < lost % len(avail) else 0)
        print(f"  {d}: " + ", ".join(f"{g}={n}" for g, n in avail.items()) + f"  (total {sum(avail.values())})")
        for g, n in avail.items():
            if g == "varsity" and not groups.get("varsity"):
                fonts = groups.get("athletic", []) + groups.get("slab", [])
            else:
                fonts = groups.get(g, [])
            jobs += [(d, g, fonts, g in OUTLINE_GROUPS)] * n

    singles, gaps = [], []
    for ip, lp in find_pairs(opt.src):
        lab = read_yolo(lp)
        if lab is None or int(lab[0][0]) == 10:
            continue
        if lab.shape[0] == 1:
            singles.append((ip, lab[0]))
        elif lab.shape[0] == 2 and abs(lab[0][1] - lab[1][1]) <= 0.25 and abs(lab[0][2] - lab[1][2]) <= 0.25:
            l = lab[np.argsort(lab[:, 1])]
            with Image.open(ip) as im:
                W, H = im.size
            b1, b2 = to_px(l[0], W, H), to_px(l[1], W, H)
            gaps.append(max(0.0, b2[0] - b1[2]) / max(1.0, max(b1[3] - b1[1], b2[3] - b2[1])))
    if not gaps:
        gaps = [0.08]
    lo, hi = np.percentile(gaps, [10, 90])
    gaps = [g for g in gaps if lo <= g <= min(hi, opt.max_gap)] or [min(np.median(gaps), opt.max_gap)]
    print(f"single-digit images: {len(singles)} | real gaps: {len(gaps)} (median {np.median(gaps):.2f} x height)")
    if not singles:
        print("nothing to do (needs YOLO labels with 1 box)")
        return
    orig = count_digits(opt.src)

    subs = ("images", "labels") if opt.mode == "val" else ()
    os.makedirs(out, exist_ok=True)
    for s in subs + (("debug",) if opt.debug else ()):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    log = open(os.path.join(out, "log.csv"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["name", "new_digit", "number", "group", "font", "style", "base_image"])

    rng = random.Random(opt.seed)
    rng.shuffle(jobs)
    order = singles[:]
    rng.shuffle(order)
    used = set()
    made, per_group, previews = {}, {}, {}
    skipped, k = 0, 0
    for j, (new_d, g, fonts, outline) in enumerate(jobs):
        for _ in range(20):
            ip, r = order[k % len(order)]
            k += 1
            if (ip, new_d) in used and len(used) < len(order) * len(plan):
                continue                                         # no duplicate base + digit
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            bx = to_px(r, W, H)
            h = int(round(bx[3] - bx[1]))
            if h < opt.min_digit_h or bx[0] < 0 or bx[2] > W or bx[1] < 0 or bx[3] > H:
                continue
            sw = stroke_width(img, bx)
            if sw is None:
                continue
            sw = float(np.clip(sw, 0.06 * h, 0.25 * h))
            colour, bg = box_colours(img, bx)
            if g == "confusable":
                glyph, font, style = draw_confusable(new_d, h, colour, rng, sw)
            elif g == "drawn":
                glyph, font, style = draw_glyph(new_d, h, colour, rng, sw, bx[2] - bx[0])
            else:
                glyph, font, style = font_glyph(new_d, h, colour, bg, rng, sw, rng.choice(fonts), outline)
            if glyph is None or glyph.width > 1.6 * h or glyph.width < 2:
                continue
            gap = rng.choice(gaps) * h
            before = rng.random() < 0.5
            gh = glyph.height
            px = int(round(bx[0] - gap - glyph.width)) if before else int(round(bx[2] + gap))
            py = int(round(bx[3] - gh))                          # same bottom line as the real digit
            if px < 0 or px + glyph.width > W or py < 0 or py + gh > H:
                continue                                         # would be clipped
            if abs((px + glyph.width / 2) / W - r[1]) > 0.25:   # training rule
                continue
            if rng.random() < 0.5:
                glyph = glyph.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
            img.paste(glyph, (px, py), glyph)
            nb = (px, py, px + glyph.width, py + gh)
            real_d = int(r[0])
            number = f"{new_d}{real_d}" if before else f"{real_d}{new_d}"
            base = os.path.splitext(os.path.basename(ip))[0]
            name = f"syn{new_d}_{g}_{opt.mode}_s{opt.seed}_{j:05d}_{base}"
            ub = (min(bx[0], nb[0]), min(bx[1], nb[1]), max(bx[2], nb[2]), max(bx[3], nb[3]))

            crop = None
            if opt.mode == "train":
                img.save(os.path.join(out, name + ".jpg"), quality=95)
                with open(os.path.join(out, name + ".txt"), "w") as f:
                    f.write(to_yolo(real_d, bx, W, H) + "\n" + to_yolo(new_d, nb, W, H) + "\n")
            else:
                crop = body_like_crop(ub, W, H, random.Random(f"{opt.seed}-{name}"))
                if crop[0] > ub[0] or crop[1] > ub[1] or crop[2] < ub[2] or crop[3] < ub[3]:
                    continue                                     # number would be clipped
                img.crop(crop).save(os.path.join(out, "images", name + ".jpg"), quality=95)
                with open(os.path.join(out, "labels", name + ".txt"), "w") as f:
                    f.write(" ".join(number) + "\n")

            if opt.debug:
                dbg = img.copy()
                dr = ImageDraw.Draw(dbg)
                dr.rectangle(bx, outline=(255, 0, 0), width=2)
                dr.rectangle(nb, outline=(0, 255, 0), width=2)
                if crop:
                    dr.rectangle(crop, outline=(0, 128, 255), width=2)
                dr.text((ub[0], max(0, ub[1] - 12)), f"{number} {g}:{font}", fill=(255, 255, 0))
                dbg.save(os.path.join(out, "debug", name + ".jpg"), quality=90)

            key = (new_d, g)
            if len(previews.setdefault(key, [])) < PREVIEW_N:
                m = (ub[3] - ub[1]) * 0.35
                tile = img.crop((int(max(0, ub[0] - m)), int(max(0, ub[1] - m)),
                                 int(min(W, ub[2] + m)), int(min(H, ub[3] + m)))).resize((120, 84))
                ImageDraw.Draw(tile).text((2, 1), number, fill=(255, 255, 0))
                previews[key].append(tile)
            wr.writerow([name, new_d, number, g, font, style, os.path.basename(ip)])
            used.add((ip, new_d))
            made[new_d] = made.get(new_d, 0) + 1
            per_group[key] = per_group.get(key, 0) + 1
            break
        else:
            skipped += 1
    log.close()

    # preview grid: one row per digit x group
    rows = sorted(previews)
    if rows:
        lw = 110
        grid = Image.new("RGB", (lw + 120 * PREVIEW_N, 84 * len(rows)), (30, 30, 30))
        dr = ImageDraw.Draw(grid)
        for i, key in enumerate(rows):
            dr.text((4, i * 84 + 34), f"digit {key[0]} | {key[1]}", fill=(255, 255, 255))
            for c, t in enumerate(previews[key]):
                grid.paste(t, (lw + c * 120, i * 84))
        grid.save(os.path.join(out, "preview.png"))

    print(f"\nskipped jobs (no base image fitted): {skipped}")
    print(f"{'digit':>5} | {'original':>8} | {'generated':>9} | {'final':>6} | by group")
    for d in plan:
        gen = made.get(d, 0)
        by = ", ".join(f"{g}={n}" for (dd, g), n in sorted(per_group.items()) if dd == d)
        print(f"{d:>5} | {orig[d]:>8} | {gen:>9} | {orig[d] + gen:>6} | {by}")
    print("(original = digit boxes in --src; each generated image adds one box of that digit)")
    print(f"saved to: {out}  ({'images/ + labels/' if opt.mode == 'val' else 'jpg + txt'}, preview.png, log.csv)")


if __name__ == "__main__":
    main()
