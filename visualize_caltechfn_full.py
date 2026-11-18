"""
Visualize the ENTIRE caltechfn portion of caltechfn_sample_350_clean (350
samples) with:
  - body.rect box drawn (green = correct state prediction, red = wrong)
  - ground-truth digits + ground-truth state (gt) and predicted state (pred)
    labeled above the box

Runs inference once per sample so predicted state is real, not guessed.
Saves individual annotated images plus contact sheets (20 per sheet) for
fast visual review of the whole caltechfn set at once.

Usage:
    python visualize_caltechfn_full.py \
        --model ~/old_server_backup_16_10_2026/New_16_10_2026/kd_79.74_s2b1.25.onnx \
        --test_image_path ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/ \
        --out_dir caltechfn_full_viz
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test_image_path", required=True)
    p.add_argument("--out_dir", default="caltechfn_full_viz")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--rows", type=int, default=4)
    opt = p.parse_args()

    annotated_dir = os.path.join(opt.out_dir, "annotated")
    sheets_dir = os.path.join(opt.out_dir, "sheets")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(sheets_dir, exist_ok=True)

    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    print("providers:", sess.get_providers())

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    caltech_images = [p for p in images if not os.path.basename(p).startswith(("set2_org_", "set3_org_"))]
    print(f"Found {len(caltech_images)} caltechfn images")

    annotated_paths = []
    n_correct = n_total = 0

    for img_path in caltech_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org_bgr = cv2.imread(img_path)
        if org_bgr is None:
            continue
        org = cv2.cvtColor(org_bgr, cv2.COLOR_BGR2RGB)
        H, W = org_bgr.shape[:2]

        canvas = org_bgr.copy()

        for item in data["data"]["jersey"]:
            digits_gt = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits_gt)
            gt_state = 2 if n_digits > 2 else 1
            gt_str = "".join(str(d) for d in digits_gt)

            x, y, w, h = item["body"]["rect"]
            crop_raw = org[int(y): int(y + h), int(x): int(x + w)]
            if crop_raw.size == 0:
                continue
            crop = cv2.resize(crop_raw, (96, 96)).astype(np.float32) / 255.0
            inp = crop.transpose(2, 0, 1)[None]
            out = dict(zip(names, sess.run(None, {in_name: inp})))
            st = int(out["state_logits"][0].argmax())

            correct = st == gt_state
            n_total += 1
            n_correct += correct
            color = (0, 200, 0) if correct else (0, 0, 255)  # BGR: green / red

            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{gt_str} gt={gt_state} pred={st}"
            ty = max(15, y1 - 6)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(canvas, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(annotated_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, canvas)
        annotated_paths.append(out_path)

    print(f"\nAnnotated {len(annotated_paths)} images -> {annotated_dir}/")
    print(f"Overall state accuracy on caltechfn (all digit counts): {n_correct}/{n_total} = {100*n_correct/n_total:.2f}%")

    # build contact sheets
    per_sheet = opt.cols * opt.rows
    cell_w, cell_h, label_h = 220, 280, 16
    for s in range(0, len(annotated_paths), per_sheet):
        batch = annotated_paths[s:s + per_sheet]
        sheet = np.full((opt.rows * (cell_h + label_h), opt.cols * cell_w, 3), 30, dtype=np.uint8)
        for i, path in enumerate(batch):
            r, c = divmod(i, opt.cols)
            im = cv2.imread(path)
            ih, iw = im.shape[:2]
            scale = min((cell_w - 8) / iw, (cell_h - 8) / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            im_small = cv2.resize(im, (nw, nh))
            x0 = c * cell_w + (cell_w - nw) // 2
            y0 = r * (cell_h + label_h) + (cell_h - nh) // 2
            sheet[y0:y0 + nh, x0:x0 + nw] = im_small
            stem = os.path.splitext(os.path.basename(path))[0]
            cv2.putText(sheet, stem[:20], (c * cell_w + 3, r * (cell_h + label_h) + cell_h + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        sheet_path = os.path.join(sheets_dir, f"sheet_{s // per_sheet + 1}.jpg")
        cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    n_sheets = (len(annotated_paths) + per_sheet - 1) // per_sheet
    print(f"Built {n_sheets} contact sheets ({per_sheet} images each) -> {sheets_dir}/")


if __name__ == "__main__":
    main()
