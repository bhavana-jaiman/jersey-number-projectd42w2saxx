"""
check_state_errors_test.py

Finds test bodies that are really 1-2 digit numbers (state 1) but are predicted
as state 2 (3+ digits), and helps you see WHY.

Works on the TEST set format (same as test_New_2.py):
    <test_image_path>/images/xxx.jpg
    <test_image_path>/labels/xxx.json   (data -> jersey -> body.rect, jersey_number, numbers[j].digits/rect)

Crops and targets are built exactly like test_New_2.py:
    --crop number : union of the digit boxes + margins (like training)   [default]
    --crop body   : whole body box
    image converted BGR -> RGB, resized to 96x96, ToTensor()

Outputs (in --out):
    state1_as_state2.csv   one row per error
    images/                one picture per error:
                             left  = what the model saw (enlarged)
                             right = the body with boxes: green = body,
                                     red = digit boxes, yellow = crop given to the model
Usage (from ~/workspace_bhavana/New_arcitecture):
    python check_state_errors_test.py \
        --checkpoint checkpoints/<run>/<folder>/jerseyNumberRecognizer_best.pth \
        --test_image_path ../Datasets/JerseyNumber_Validation_Dataset \
        --crop number
"""
import os
import csv
import glob
import json
import argparse
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as Transforms

from subModules.backbone_New import MultiTaskLearnerWithState

BLANK = 10
VIEW_H = 192          # height of the saved pictures


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def load_model(path, device):
    model = MultiTaskLearnerWithState().to(device)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # _best.pth = plain state_dict; _ckp_ files keep the weights under "state_dict"
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ----------------------------------------------------------------------
# labels (same rules as test_New_2.py build_targets)
# ----------------------------------------------------------------------
def label_path_for(img_path):
    root = os.path.dirname(os.path.dirname(img_path))
    name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    return os.path.join(root, "labels", name)


def build_targets(entry):
    body = [float(v) for v in entry["body"]["rect"]]
    digits = []
    for j in range(len(entry["jersey_number"])):
        num = entry["numbers"][j]
        digits.append((int(num["digits"]), [float(v) for v in num["rect"]]))
    digits.sort(key=lambda d: d[1][0])                 # left -> right, like training
    n = len(digits)
    state = 0 if n == 0 else (1 if n <= 2 else 2)
    return body, state, [d for d, _ in digits], [r for _, r in digits]


def to_abs(rects, body, coords):
    """Digit rects [x,y,w,h] -> image coords. coords: auto | image | body (same as test_New_2.py)."""
    if not rects:
        return [], False
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[0] + r[2] for r in rects)
    y2 = max(r[1] + r[3] for r in rects)
    bx, by, bw, bh = body
    relative = coords == "body"
    if coords == "auto":
        overlaps = not (x2 < bx or x1 > bx + bw or y2 < by or y1 > by + bh)
        relative = not overlaps
    if relative:
        rects = [[r[0] + bx, r[1] + by, r[2], r[3]] for r in rects]
    return rects, relative


# ----------------------------------------------------------------------
# crops (same as test_New_2.py crop_region / crop_body / crop_number)
# ----------------------------------------------------------------------
def crop_region(img_bgr, x1, y1, x2, y2, size):
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop, (x1, y1, x2, y2)


def make_input(img_bgr, body, rects_abs, mode, margins, size, rel_margins=None):
    bx, by, bw, bh = body
    if mode == "body" or not rects_abs:
        return crop_region(img_bgr, bx, by, bx + bw, by + bh, size)
    x1 = min(r[0] for r in rects_abs)
    y1 = min(r[1] for r in rects_abs)
    x2 = max(r[0] + r[2] for r in rects_abs)
    y2 = max(r[1] + r[3] for r in rects_abs)
    if mode == "number_rel":
        # margins proportional to the digit height -> same digits/crop ratio as training
        H = y2 - y1
        ml, mt, mr, mb = [f * H for f in rel_margins]
    else:
        ml, mt, mr, mb = margins
    return crop_region(img_bgr, x1 - ml, y1 - mt, x2 + mr, y2 + mb, size)


def make_panel(img_bgr, crop_rgb, body, rects_abs, crop_box, text):
    """left: model input (enlarged); right: body area with boxes drawn."""
    left = cv2.resize(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), (VIEW_H, VIEW_H),
                      interpolation=cv2.INTER_NEAREST)

    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = body
    pad = int(0.15 * max(bw, bh))
    X1, Y1 = max(0, int(bx) - pad), max(0, int(by) - pad)
    X2, Y2 = min(w, int(bx + bw) + pad), min(h, int(by + bh) + pad)
    ctx = img_bgr[Y1:Y2, X1:X2].copy()

    def rect(x1, y1, x2, y2, color, t=2):
        cv2.rectangle(ctx, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), color, t)

    rect(bx, by, bx + bw, by + bh, (0, 200, 0))                     # body: green
    for r in rects_abs:
        rect(r[0], r[1], r[0] + r[2], r[1] + r[3], (0, 0, 255), 1)   # digits: red
    if crop_box is not None:
        rect(*crop_box, (0, 255, 255))                               # model crop: yellow

    scale = VIEW_H / max(ctx.shape[0], 1)
    right = cv2.resize(ctx, (max(1, int(ctx.shape[1] * scale)), VIEW_H))
    panel = np.hstack([left, np.full((VIEW_H, 6, 3), 255, np.uint8), right])
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_image_path", required=True, help="folder with images/ and labels/ (json)")
    ap.add_argument("--crop", default="number", choices=["number", "number_rel", "body"],
                    help="number = fixed pixel margins (like test_New_2.py); "
                         "number_rel = margins relative to digit height (--rel_margins); body = body box")
    ap.add_argument("--rel_margins", type=float, nargs=4, default=None,
                    metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    help="for --crop number_rel: margins as a fraction of the digit height "
                         "(get them from measure_crop_ratio.py)")
    ap.add_argument("--margins", type=int, nargs=4, default=[23, 30, 30, 33],
                    help="left top right bottom (number crop), same as training")
    ap.add_argument("--digit_coords", default="auto", choices=["auto", "image", "body"])
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--out", default="state_errors_test")
    ap.add_argument("--max_save", type=int, default=1000, help="max error pictures to save")
    opt = ap.parse_args()
    if opt.crop == "number_rel" and opt.rel_margins is None:
        ap.error("--crop number_rel needs --rel_margins (run measure_crop_ratio.py first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.checkpoint, device)
    to_tensor = Transforms.ToTensor()
    os.makedirs(os.path.join(opt.out, "images"), exist_ok=True)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    print(f"Found {len(images)} images | crop mode: {opt.crop} | input {opt.img_size}x{opt.img_size}")

    conf = np.zeros((3, 3), dtype=int)       # rows true state, cols predicted state
    total_s1 = Counter()                     # true state-1 bodies by n digits
    to_s2 = Counter()                        # ... predicted state 2
    to_s2_swapped = Counter()                # ... predicted state 2 with BGR<->RGB swapped input
    conf_bins = Counter()
    true_numbers = Counter()
    digits_ok_in_err = 0
    skipped = relative_used = saved = 0
    rows = []

    with torch.no_grad():
        for img_path in images:
            lp = label_path_for(img_path)
            if not os.path.exists(lp):
                skipped += 1
                continue
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            with open(lp, "r", encoding="utf8") as f:
                data = json.load(f)

            for b_idx, entry in enumerate(data["data"]["jersey"]):
                body, state, digits, rects = build_targets(entry)
                rects_abs, relative = to_abs(rects, body, opt.digit_coords)
                relative_used += relative
                crop, crop_box = make_input(img, body, rects_abs, opt.crop, opt.margins, opt.img_size,
                                             opt.rel_margins)
                if crop is None:
                    skipped += 1
                    continue

                x = to_tensor(crop).unsqueeze(0).to(device)
                _, l1, l2, ls = model(x)
                p = torch.softmax(ls, 1)[0]
                pred_state = int(p.argmax())
                conf[state, pred_state] += 1

                if state != 1:
                    continue
                n = len(digits)
                total_s1[n] += 1

                _, _, _, ls_sw = model(x[:, [2, 1, 0]])
                if int(ls_sw.argmax()) == 2:
                    to_s2_swapped[n] += 1

                if pred_state != 2:
                    continue

                # ---- state 1 predicted as state 2 ----
                to_s2[n] += 1
                p2 = float(p[2])
                conf_bins[">0.9" if p2 > 0.9 else "0.7-0.9" if p2 > 0.7 else "0.5-0.7" if p2 > 0.5 else "<0.5"] += 1
                d1 = digits[0]
                d2 = digits[1] if n == 2 else BLANK
                pd1, pd2 = int(l1.argmax()), int(l2.argmax())
                ok = (pd1 == d1) and (pd2 == d2)
                digits_ok_in_err += ok
                true_str = "".join(map(str, digits))
                true_numbers[true_str] += 1

                base = os.path.splitext(os.path.basename(img_path))[0]
                pic = f"p{p2:.2f}_true{true_str}_{base}_b{b_idx}.jpg"
                if saved < opt.max_save:
                    pred_str = f"{pd1}" + ("" if pd2 == BLANK else f"{pd2}")
                    panel = make_panel(img, crop, body, rects_abs, crop_box,
                                       f"true {true_str}  pred {pred_str}  p(s2)={p2:.2f}")
                    cv2.imwrite(os.path.join(opt.out, "images", pic), panel)
                    saved += 1

                rows.append({"image": img_path, "body_index": b_idx, "true_number": "'" + true_str,
                             "n_digits": n, "p_state2": round(p2, 3), "pred_d1": pd1,
                             "pred_d2": "" if pd2 == BLANK else pd2, "digits_correct": ok,
                             "picture": pic})

    rows.sort(key=lambda r: -r["p_state2"])
    with open(os.path.join(opt.out, "state1_as_state2.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # ---------------- summary ----------------
    print("\n==== state confusion on the test set (rows = true 0/1/2, cols = predicted 0/1/2) ====")
    print(conf)
    if skipped:
        print(f"(skipped {skipped} bodies/images: missing label, unreadable image or empty crop)")
    if relative_used:
        print(f"(digit boxes read as body-relative for {relative_used} bodies)")

    print("\n==== true state 1 (1-2 digits) predicted as state 2 ====")
    all_s1 = sum(total_s1.values())
    print(f"test state-1 bodies: {all_s1}")
    for n in (1, 2):
        if total_s1[n]:
            print(f"  {n}-digit numbers: {to_s2[n]:5d} / {total_s1[n]:5d}  ({100*to_s2[n]/total_s1[n]:.2f}%) -> state 2")
    e = sum(to_s2.values())
    print(f"  total -> state 2: {e}")

    print("\nconfidence of the wrong 'state 2' (softmax prob):")
    for k in [">0.9", "0.7-0.9", "0.5-0.7", "<0.5"]:
        print(f"  {k:>8}: {conf_bins[k]}")

    if e:
        print(f"\ndigits still correct in these errors: {digits_ok_in_err} / {e} ({100*digits_ok_in_err/e:.1f}%)")
        two = [t for t in true_numbers.elements() if len(t) == 2]
        rep = sum(1 for t in two if t[0] == t[1])
        if two:
            print(f"repeated-digit numbers (77, 44, ...) among 2-digit errors: {rep} / {len(two)} ({100*rep/len(two):.1f}%)")
        lead0 = sum(1 for t in two if t[0] == "0")
        if lead0:
            print(f"2-digit errors whose label starts with 0 (e.g. '02'): {lead0}")
    print("most common true numbers among errors:", true_numbers.most_common(10))

    print("\ncolour-order test (same crops, channels swapped RGB<->BGR):")
    print(f"  -> state 2 with normal input : {e}")
    print(f"  -> state 2 with swapped input: {sum(to_s2_swapped.values())}")

    print(f"\nsaved: {opt.out}/state1_as_state2.csv and {saved} pictures in {opt.out}/images/")
    print("pictures: left = model input, right = body (green), digit boxes (red), model crop (yellow)")


if __name__ == "__main__":
    main()
