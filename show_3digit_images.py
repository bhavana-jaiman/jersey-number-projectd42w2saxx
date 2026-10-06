"""
show_3digit_images.py
---------------------
Finds every training image whose YOLO label has 3 or more digit boxes
(the real "3+ digits" / state-2 images), draws the boxes on it, and saves:

  <out>/images/<name>.jpg        each image with its digit boxes and number
  <out>/sheets/sheet_XXX.jpg     contact sheets (many images per page) for fast checking
  <out>/three_digit_list.csv     one row per image + automatic warnings

Box colours on the images:
  red    = each digit box, with the digit written above it
  green  = one box around all digits (the "number" the dataset uses)
  yellow = number read left -> right, written at the top

Warnings in the CSV (worth a look, possible label problems):
  far_apart      boxes are far apart -> maybe two different numbers / two players
  height_differs largest box is > 1.6x the smallest -> maybe a digit from another number
  overlap        two boxes overlap a lot -> maybe a duplicated box
  out_of_image   a box goes outside the image

Run from ~/workspace_bhavana/New_arcitecture:
    python show_3digit_images.py
    python show_3digit_images.py --train ../Datasets/70_30_dataset/training --out debug_3digit
"""
import os
import csv
import glob
import argparse
import warnings
from collections import Counter

import cv2
import numpy as np

RED, GREEN, YELLOW, BLACK = (0, 0, 255), (0, 200, 0), (0, 255, 255), (0, 0, 0)


def read_yolo(path):
    warnings.filterwarnings("ignore")
    a = np.atleast_1d(np.loadtxt(path))
    if a.size == 0 or a.size % 5 != 0:
        return None
    return a.reshape(-1, 5)


def to_pixels(lbl, W, H):
    """YOLO rows -> list of (digit, x1, y1, x2, y2) in pixels, sorted left -> right."""
    out = []
    for c, cx, cy, w, h in lbl:
        out.append((int(c), (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return sorted(out, key=lambda b: b[1])


def warnings_for(boxes, W, H):
    flags = []
    heights = [b[4] - b[2] for b in boxes]
    widths = [b[3] - b[1] for b in boxes]
    if min(heights) > 0 and max(heights) / min(heights) > 1.6:
        flags.append("height_differs")
    # gap between neighbouring boxes compared with the typical digit width
    typ_w = float(np.median(widths)) if widths else 1.0
    for a, b in zip(boxes, boxes[1:]):
        gap = b[1] - a[3]
        vgap = abs((a[2] + a[4]) / 2 - (b[2] + b[4]) / 2)
        if gap > 1.5 * typ_w or vgap > 0.8 * float(np.median(heights)):
            flags.append("far_apart")
            break
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
            inter = ix * iy
            small = min((a[3] - a[1]) * (a[4] - a[2]), (b[3] - b[1]) * (b[4] - b[2]))
            if small > 0 and inter / small > 0.5:
                flags.append("overlap")
                break
        if "overlap" in flags:
            break
    if any(b[1] < -2 or b[2] < -2 or b[3] > W + 2 or b[4] > H + 2 for b in boxes):
        flags.append("out_of_image")
    return flags


def draw(img, boxes, number, flags):
    H, W = img.shape[:2]
    t = max(1, round(min(W, H) / 250))                 # line thickness scales with image size
    fs = max(0.4, min(W, H) / 500)                     # font scale scales with image size
    ux1 = min(b[1] for b in boxes); uy1 = min(b[2] for b in boxes)
    ux2 = max(b[3] for b in boxes); uy2 = max(b[4] for b in boxes)
    cv2.rectangle(img, (int(ux1) - 2 * t, int(uy1) - 2 * t), (int(ux2) + 2 * t, int(uy2) + 2 * t), GREEN, t)
    for d, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), RED, t)
        cv2.putText(img, str(d), (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, RED, t, cv2.LINE_AA)
    text = number + ("  ! " + ",".join(flags) if flags else "")
    cv2.rectangle(img, (0, 0), (W, int(28 * fs) + 8), BLACK, -1)
    cv2.putText(img, text, (4, int(22 * fs) + 2), cv2.FONT_HERSHEY_SIMPLEX, fs, YELLOW, t, cv2.LINE_AA)
    return img


def make_sheets(paths, out_dir, cols=6, tile=220):
    os.makedirs(out_dir, exist_ok=True)
    per = cols * 5
    for s in range(0, len(paths), per):
        tiles = []
        for p in paths[s:s + per]:
            im = cv2.imread(p)
            h, w = im.shape[:2]
            k = tile / max(h, w)
            im = cv2.resize(im, (max(1, int(w * k)), max(1, int(h * k))))
            canvas = np.full((tile + 18, tile, 3), 255, np.uint8)
            canvas[:im.shape[0], :im.shape[1]] = im
            cv2.putText(canvas, os.path.basename(p)[:30], (2, tile + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, BLACK, 1, cv2.LINE_AA)
            tiles.append(canvas)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(out_dir, f"sheet_{s // per + 1:03d}.jpg"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../Datasets/70_30_dataset/training",
                    help="training folder with .jpg and .txt (YOLO) side by side")
    ap.add_argument("--out", default="debug_3digit")
    ap.add_argument("--min_digits", type=int, default=3)
    opt = ap.parse_args()

    os.makedirs(os.path.join(opt.out, "images"), exist_ok=True)
    rows, saved, flag_count, n_digits = [], [], Counter(), Counter()

    for lp in sorted(glob.glob(os.path.join(opt.train, "*.txt"))):
        lbl = read_yolo(lp)
        if lbl is None or len(lbl) < opt.min_digits or int(lbl[0][0]) == 10:
            continue
        ip = os.path.splitext(lp)[0] + ".jpg"
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = to_pixels(lbl, W, H)
        number = "".join(str(b[0]) for b in boxes)
        flags = warnings_for(boxes, W, H)
        n_digits[len(boxes)] += 1
        for f in flags:
            flag_count[f] += 1

        name = os.path.basename(ip)
        out_path = os.path.join(opt.out, "images", ("FLAG_" if flags else "") + name)
        cv2.imwrite(out_path, draw(img, boxes, number, flags))
        saved.append(out_path)
        rows.append({"image": name, "number": "'" + number, "n_digits": len(boxes),
                     "img_w": W, "img_h": H,
                     "digit_h_px": round(float(np.median([b[4] - b[2] for b in boxes])), 1),
                     "warnings": ",".join(flags)})

    with open(os.path.join(opt.out, "three_digit_list.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    # flagged images first in the contact sheets, so problems are seen early
    saved.sort(key=lambda p: (not os.path.basename(p).startswith("FLAG_"), p))
    if saved:
        make_sheets(saved, os.path.join(opt.out, "sheets"))

    print(f"images with {opt.min_digits}+ digit boxes: {len(rows)}")
    print("by number of boxes:", dict(sorted(n_digits.items())))
    print("warnings:", dict(flag_count) if flag_count else "none")
    print(f"saved: {opt.out}/images/ (FLAG_ = has a warning), {opt.out}/sheets/, "
          f"{opt.out}/three_digit_list.csv")


if __name__ == "__main__":
    main()
