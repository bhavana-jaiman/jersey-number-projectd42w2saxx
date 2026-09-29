#!/usr/bin/env python3
"""
Augment jersey-number images together with their JSON labels, and save
visualisations with the digit bounding boxes drawn on top.

Expected input layout
    <root>/images/<name>.jpg
    <root>/labels/<name>.json     (one JSON per image)

Label format (keys may use spaces or underscores, both are handled):
    {"file_name": "201810_image_000645.jpg",
     "dimensions": [426, 640],
     "data": [{"jersey": [{"body": {"rect": [140,145,137,202]},
                           "jersey_number": [2, 2, 6],
                           "numbers": [{"digits": 2, "rect": [152,245,19,35]},
                                       {"digits": 2, "rect": [172,245,20,35]},
                                       {"digits": 6, "rect": [195,246,19,37]}]}]}]}
    rect = [x, y, width, height] in pixels.

Only files that contain a non-empty jersey number array are processed.

Output layout
    <output>/images/       augmented images
    <output>/labels/       updated JSON labels (same structure as input)
    <output>/visualized/   images with body + digit boxes drawn

Usage
    python augment_jersey.py --images data/images --labels data/labels \
                             --output data/output --num-aug 5
Requires: pip install opencv-python numpy
"""
import argparse
import copy
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Label helpers (tolerant to "jersey number" vs "jersey_number")
# --------------------------------------------------------------------------- #
def _variants(key):
    return (key, key.replace(" ", "_"), key.replace("_", " "))


def get(d, key, default=None):
    if not isinstance(d, dict):
        return default
    for k in _variants(key):
        if k in d:
            return d[k]
    return default


def find_key(d, key):
    for k in _variants(key):
        if k in d:
            return k
    return None


def iter_jerseys(label):
    for entry in get(label, "data", []) or []:
        for jersey in get(entry, "jersey", []) or []:
            yield jersey


def has_jersey_number(label):
    return any(get(j, "jersey number") and get(j, "numbers")
               for j in iter_jerseys(label))


def collect_rects(label):
    """Return [(dict_that_holds_rect, kind, digit_value), ...] for all boxes."""
    out = []
    for j in iter_jerseys(label):
        body = get(j, "body")
        if isinstance(body, dict) and "rect" in body:
            out.append((body, "body", None))
        for n in get(j, "numbers", []) or []:
            if isinstance(n, dict) and "rect" in n:
                out.append((n, "digit", n.get("digits")))
    return out


# --------------------------------------------------------------------------- #
# Geometric augmentation
# --------------------------------------------------------------------------- #
def random_affine(w, h, cfg):
    angle = random.uniform(-cfg.rotate, cfg.rotate)
    scale = random.uniform(1 - cfg.scale, 1 + cfg.scale)
    tx = random.uniform(-cfg.translate, cfg.translate) * w
    ty = random.uniform(-cfg.translate, cfg.translate) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def transform_rect(rect, M, w, h):
    """Transform [x,y,w,h] by affine M; return (clipped_rect, visible_fraction)."""
    x, y, bw, bh = [float(v) for v in rect]
    pts = np.array([[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]],
                   dtype=np.float64)
    pts = np.hstack([pts, np.ones((4, 1))]) @ M.T
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    full_area = max((x2 - x1) * (y2 - y1), 1e-6)

    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(w), x2), min(float(h), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, 0.0
    visible = (cx2 - cx1) * (cy2 - cy1) / full_area
    new_rect = [int(round(cx1)), int(round(cy1)),
                int(round(cx2 - cx1)), int(round(cy2 - cy1))]
    return new_rect, visible


# --------------------------------------------------------------------------- #
# Photometric augmentation (does not move boxes)
# --------------------------------------------------------------------------- #
def photometric(img):
    out = img.astype(np.float32)

    if random.random() < 0.8:  # brightness / contrast
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        out = out * alpha + beta

    out = np.clip(out, 0, 255).astype(np.uint8)

    if random.random() < 0.5:  # hue / saturation
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.3:  # blur
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    if random.random() < 0.3:  # gaussian noise
        noise = np.random.normal(0, random.uniform(3, 12), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def augment_once(img, label, cfg, max_tries=15):
    """Return (aug_img, aug_label) or (None, None) if no valid sample found."""
    h, w = img.shape[:2]
    for _ in range(max_tries):
        M = random_affine(w, h, cfg)
        new_label = copy.deepcopy(label)
        ok = True
        for holder, kind, _ in collect_rects(new_label):
            new_rect, vis = transform_rect(holder["rect"], M, w, h)
            # every digit must stay (mostly) inside the image
            if new_rect is None or (kind == "digit" and vis < cfg.min_visibility):
                ok = False
                break
            holder["rect"] = new_rect
        if not ok:
            continue
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return photometric(warped), new_label
    return None, None


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def draw_boxes(img, label):
    vis = img.copy()
    for j in iter_jerseys(label):
        body = get(j, "body")
        number = get(j, "jersey number") or []
        if isinstance(body, dict) and "rect" in body:
            x, y, bw, bh = map(int, body["rect"])
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 128, 0), 2)
            txt = "#" + "".join(str(d) for d in number)
            cv2.putText(vis, txt, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        for n in get(j, "numbers", []) or []:
            if "rect" not in n:
                continue
            x, y, bw, bh = map(int, n["rect"])
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
            cv2.putText(vis, str(n.get("digits", "?")), (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return vis


# --------------------------------------------------------------------------- #
# File matching
# --------------------------------------------------------------------------- #
def norm(name):
    return Path(name).stem.lower().replace(" ", "_")


def build_image_index(images_dir):
    return {norm(p.name): p for p in Path(images_dir).rglob("*")
            if p.suffix.lower() in IMG_EXTS}


def find_image(label_path, label, index):
    img = index.get(norm(label_path.name))            # same stem as json
    if img is None and get(label, "file name"):
        img = index.get(norm(get(label, "file name")))  # file_name field
    return img


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with images")
    ap.add_argument("--labels", required=True, help="folder with JSON labels")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--num-aug", type=int, default=5, help="augmentations per image")
    ap.add_argument("--rotate", type=float, default=8.0, help="max rotation (deg)")
    ap.add_argument("--scale", type=float, default=0.15, help="max scale change (+/-)")
    ap.add_argument("--translate", type=float, default=0.08, help="max shift (fraction)")
    ap.add_argument("--min-visibility", type=float, default=0.9,
                    help="min fraction of each digit box that must remain in frame")
    ap.add_argument("--no-originals", action="store_true",
                    help="do not save visualisation of the original images")
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = Path(cfg.output)
    out_img, out_lbl, out_vis = out / "images", out / "labels", out / "visualized"
    for d in (out_img, out_lbl, out_vis):
        d.mkdir(parents=True, exist_ok=True)

    index = build_image_index(cfg.images)
    label_files = sorted(Path(cfg.labels).rglob("*.json"))
    stats = dict(total=0, with_number=0, missing_img=0, bad=0, generated=0)

    for lp in label_files:
        stats["total"] += 1
        try:
            label = json.loads(lp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] cannot read {lp.name}: {e}")
            stats["bad"] += 1
            continue

        if not has_jersey_number(label):
            continue
        stats["with_number"] += 1

        img_path = find_image(lp, label, index)
        if img_path is None:
            print(f"[skip] no image found for {lp.name}")
            stats["missing_img"] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot open {img_path}")
            stats["bad"] += 1
            continue

        stem = img_path.stem
        if not cfg.no_originals:
            cv2.imwrite(str(out_vis / f"{stem}_orig.jpg"), draw_boxes(img, label))

        for i in range(cfg.num_aug):
            aug_img, aug_lbl = augment_once(img, label, cfg)
            if aug_img is None:
                print(f"[warn] {stem}: could not keep digits in frame (aug {i})")
                continue

            name = f"{stem}_aug{i}"
            fkey = find_key(aug_lbl, "file name")
            if fkey:
                aug_lbl[fkey] = f"{name}{img_path.suffix}"

            cv2.imwrite(str(out_img / f"{name}{img_path.suffix}"), aug_img)
            (out_lbl / f"{name}.json").write_text(json.dumps(aug_lbl), encoding="utf-8")
            cv2.imwrite(str(out_vis / f"{name}.jpg"), draw_boxes(aug_img, aug_lbl))
            stats["generated"] += 1

    print("\nDone.")
    print(f"  label files scanned      : {stats['total']}")
    print(f"  with jersey number array : {stats['with_number']}")
    print(f"  missing images           : {stats['missing_img']}")
    print(f"  unreadable files         : {stats['bad']}")
    print(f"  augmented samples saved  : {stats['generated']}")
    print(f"  output                   : {out.resolve()}")


if __name__ == "__main__":
    main()
