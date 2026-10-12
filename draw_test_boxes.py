"""
draw_test_boxes.py
Draws, on every client test image:
  * body box   (green = prediction correct, red = wrong, yellow = 3+ body)
                text: "GT 23 | P 28 s1 0.97"  (s = predicted state, prob of that state)
  * digit boxes (cyan if fully inside its body, magenta if partly outside)
                text under the body box, one line per digit:
                "3: iou 0.03 in 0.33"  (IoU with body, inside = part of digit box inside the body)
Also writes boxes.csv with one row per digit box (IoU / inside per digit).

NOTE: the model only classifies the 96x96 body crop, it does not predict boxes.
      So IoU here = GT digit box vs GT body box (checks the labels / crop),
      and the prediction is the model output for that body crop.

Run (from New_arcitecture_2, same args as the testing script):
    python draw_test_boxes.py --checkpoint checkpoints/<run>/jerseyNumberRecognitior_best.pth \
        --test_image_path ../workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset \
        --out results/Test_boxes --only_wrong
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as Transforms

from subModules.backbone_ying import MultiTaskLearnerWithState

STATE_TH = 0.9          # same 0.9 threshold for "3+" as testing / validation


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    inside = inter / (a[2] * a[3]) if a[2] * a[3] > 0 else 0.0      # part of a inside b
    return (inter / union if union > 0 else 0.0), inside


def put(img, text, x, y, color, scale):
    y = max(int(14 * scale), int(y))
    cv2.putText(img, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0),
                max(1, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(img, text, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color,
                max(1, int(1 * scale)), cv2.LINE_AA)


def predict(model, body_rgb, device):
    x = Transforms.ToTensor()(cv2.resize(body_rgb, (96, 96))).unsqueeze(0).to(device)
    _, l1, l2, ls = model(x)
    p_state = torch.softmax(ls, 1)[0]
    s = int(p_state.argmax())
    if s == 2 and p_state[2] < STATE_TH:                 # weak 3+ -> best of state 0/1
        s = int(p_state[:2].argmax())
    return int(l1.argmax(1)), int(l2.argmax(1)), s, float(p_state[s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_image_path", required=True)
    ap.add_argument("--out", default="results/Test_boxes")
    ap.add_argument("--only_wrong", action="store_true", help="save only images with a wrong body")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskLearnerWithState().to(device)
    sd = torch.load(opt.checkpoint, map_location=device)
    model.load_state_dict(sd.get("state_dict", sd), strict=True)
    model.eval()

    os.makedirs(opt.out, exist_ok=True)
    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    if opt.max_images:
        images = images[:opt.max_images]

    rows, saved, n_body, n_wrong = [], 0, 0, 0
    with torch.no_grad():
        for ip in images:
            with open(ip.replace("images", "labels").replace(".jpg", ".json"), encoding="utf8") as f:
                data = json.load(f)
            org = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)     # model input (RGB)
            canvas = cv2.imread(ip)                                    # drawing (BGR)
            scale = max(1.0, canvas.shape[1] / 1280)
            any_wrong = False

            for bi, jer in enumerate(data["data"]["jersey"]):
                bx, by, bw, bh = [float(v) for v in jer["body"]["rect"]]
                gt = [int(jer["numbers"][j]["digits"]) for j in range(len(jer["jersey_number"]))]
                gt_state = 2 if len(gt) > 2 else (1 if gt else 0)

                x0, y0 = max(0, int(bx)), max(0, int(by))
                x1, y1 = int(bx + bw), int(by + bh)
                body = org[y0:y1, x0:x1]
                if body.size == 0:
                    continue
                d1, d2, s, ps = predict(model, body, device)
                n_body += 1

                if gt_state == 1:
                    gt2 = gt + [10] * (2 - len(gt))
                    ok = (s == 1 and d1 == gt2[0] and d2 == gt2[1])
                else:
                    ok = (s == gt_state)
                any_wrong |= not ok
                n_wrong += not ok

                col = (0, 255, 255) if gt_state == 2 else ((0, 200, 0) if ok else (0, 0, 255))
                cv2.rectangle(canvas, (x0, y0), (x1, y1), col, max(1, int(2 * scale)))
                gt_txt = "".join(map(str, gt)) if gt else "-"
                p_txt = (f"{d1}{'' if d2 == 10 else d2}" if s == 1 else ("3+" if s == 2 else "-"))
                put(canvas, f"GT {gt_txt} | P {p_txt} s{s} {ps:.2f}", x0, y0 - 4, col, scale)

                for j in range(len(gt)):
                    r = [float(v) for v in jer["numbers"][j]["rect"]]
                    iou, inside = iou_xywh(r, [bx, by, bw, bh])
                    dcol = (255, 255, 0) if inside >= 0.9 else (255, 0, 255)
                    cv2.rectangle(canvas, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])),
                                  dcol, max(1, int(1 * scale)))
                    # one text line per digit, stacked under the digit boxes (no overlap)
                    put(canvas, f"{gt[j]}: iou {iou:.2f} in {inside:.2f}", x0 + 3,
                        y1 + (j + 1) * 14 * scale, dcol, scale * 0.8)
                    rows.append([os.path.basename(ip), bi, gt_txt, p_txt, s, int(ok), j, gt[j],
                                 round(iou, 3), round(inside, 3),
                                 round(r[3] * 96 / bh, 1) if bh > 0 else 0])   # digit height at 96

            if any_wrong or not opt.only_wrong:
                cv2.imwrite(os.path.join(opt.out, os.path.basename(ip)), canvas)
                saved += 1

    with open(os.path.join(opt.out, "boxes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "body", "gt", "pred", "pred_state", "correct", "digit_idx", "digit",
                    "iou_with_body", "inside_body", "digit_h_px_at96"])
        w.writerows(rows)

    ins = np.array([r[9] for r in rows]) if rows else np.zeros(0)
    print(f"bodies: {n_body}, wrong: {n_wrong}, images saved: {saved} -> {opt.out}")
    if ins.size:
        print(f"digit boxes: {ins.size}, partly outside body (inside < 0.9): {(ins < 0.9).sum()}")


if __name__ == "__main__":
    main()
