import os
import sys
import json
import cv2
import pandas as pd

# usage: python show_errors.py output/exp6/wrong_predictions.csv
csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/exp6/wrong_predictions.csv"
N = 30          # images saved per error type
BLANK = 10

df = pd.read_csv(csv_path)
out_root = os.path.join(os.path.dirname(csv_path), "error_images")

groups = {
    "state_1_pred_as_2": df[(df.true_state == 1) & (df.pred_state == 2)],
    "state_2_pred_as_1": df[(df.true_state == 2) & (df.pred_state == 1)],
    "2digit_2nd_pred_blank": df[(df.true_state == 1) & (df.pred_state == 1)
                                & (df.true_d2 != BLANK) & (df.pred_d2 == BLANK)],
}


def label_path_for(img_path):
    root = os.path.dirname(os.path.dirname(img_path))
    name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    return os.path.join(root, "labels", name)


for gname, rows in groups.items():
    out_dir = os.path.join(out_root, gname)
    os.makedirs(out_dir, exist_ok=True)
    rows = rows.sample(min(N, len(rows)), random_state=0)

    for k, r in enumerate(rows.itertuples()):
        img = cv2.imread(r.image)
        if img is None:
            continue
        with open(label_path_for(r.image), "r", encoding="utf8") as f:
            entry = json.load(f)["data"]["jersey"][int(r.body_index)]

        # body box (green)
        x, y, w, h = [int(float(v)) for v in entry["body"]["rect"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # every labelled digit box (red) with its digit
        for j in range(len(entry["jersey_number"])):
            num = entry["numbers"][j]
            dx, dy, dw, dh = [int(float(v)) for v in num["rect"]]
            cv2.rectangle(img, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 2)
            cv2.putText(img, str(num["digits"]), (dx, max(15, dy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # cut out the body area (+ a little context) so it is easy to look at
        pad = 20
        H, W = img.shape[:2]
        crop = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if crop.size == 0:
            continue
        scale = 400 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale)

        text = (f"true {r.true_digits} s{r.true_state} | "
                f"pred s{r.pred_state} d1={r.pred_d1} d2={r.pred_d2}")
        canvas = cv2.copyMakeBorder(crop, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(canvas, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"{k:03d}.jpg"), canvas)

    print(f"{gname}: {len(rows)} images -> {out_dir}")
