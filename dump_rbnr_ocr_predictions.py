"""
Diagnostic v2: dump per-sample state predictions for the 89 EasyOCR-cropped
RBNR images inside caltechfn_sample_350_clean, using the exact same
preprocessing as test_onnx_accuracy.py (crop body.rect -> resize 96x96 ->
/255, RGB).

Goal: now that the crop geometry matches the real dataset's tight
number-box style (median ~56x74px from JerseyNumber_Validation_Dataset),
find out whether the REMAINING wrong predictions (state2 = 63.10%, so
~33/89 still wrong) are due to:
  (a) crop still slightly loose/offset (visible in the saved crop images)
  (b) genuinely hard photos (motion blur, curved bib, harsh angle/lighting)
  (c) a specific digit-count bias (model still under-predicting state2)

This saves, for every WRONG RBNR prediction:
  - the actual 96x96 crop that was fed to the model (as the model sees it)
  - a padded, upscaled version for easier visual judgment
  - predicted state + raw state_logits
  - ground truth digits

Usage:
    python dump_rbnr_ocr_predictions.py \
        --model kd_79.74_s2b1.25.onnx \
        --test_image_path ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/ \
        --out_dir rbnr_ocr_wrong_crops
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
    p.add_argument("--out_dir", default="rbnr_ocr_wrong_crops")
    opt = p.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    print("providers:", sess.get_providers(), "| outputs:", names)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    rbnr_images = [p for p in images if os.path.basename(p).startswith(("set2_org_", "set3_org_"))]
    print(f"Found {len(rbnr_images)} RBNR images out of {len(images)} total")

    pred_state_hist = {}
    rows = []

    for img_path in rbnr_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org_bgr = cv2.imread(img_path)
        org = cv2.cvtColor(org_bgr, cv2.COLOR_BGR2RGB)

        for item in data["data"]["jersey"]:
            digits_gt = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits_gt)
            gt_state = 2 if n_digits > 2 else 1

            x, y, w, h = item["body"]["rect"]
            crop_raw = org[int(y): int(y + h), int(x): int(x + w)]
            crop_96 = cv2.resize(crop_raw, (96, 96))
            crop = crop_96.astype(np.float32) / 255.0
            inp = crop.transpose(2, 0, 1)[None]

            out = dict(zip(names, sess.run(None, {in_name: inp})))
            st_logits = out["state_logits"][0]
            st = int(st_logits.argmax())
            correct = st == gt_state

            pred_state_hist[st] = pred_state_hist.get(st, 0) + 1

            row = {
                "file": os.path.basename(img_path),
                "gt_digits": "".join(str(d) for d in digits_gt),
                "gt_state": gt_state,
                "pred_state": st,
                "state_logits": [round(float(v), 2) for v in st_logits],
                "correct": correct,
                "body_rect": [round(float(v), 1) for v in (x, y, w, h)],
            }
            rows.append(row)

            if not correct:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                # save EXACTLY what the model saw (the 96x96 input, upscaled for viewing)
                exact_input = cv2.cvtColor(crop_96, cv2.COLOR_RGB2BGR)
                exact_input_big = cv2.resize(exact_input, (384, 384), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(opt.out_dir, f"{stem}__model_input.jpg"), exact_input_big)

                # also save a padded context view from the original image for judgment
                H, W = org_bgr.shape[:2]
                pad_x, pad_y = w * 0.6, h * 0.6
                cx1 = max(0, int(x - pad_x)); cy1 = max(0, int(y - pad_y))
                cx2 = min(W, int(x + w + pad_x)); cy2 = min(H, int(y + h + pad_y))
                context = org_bgr[cy1:cy2, cx1:cx2].copy()
                # draw the actual body.rect used, relative to this padded context
                rx1, ry1 = int(x - cx1), int(y - cy1)
                rx2, ry2 = int(x + w - cx1), int(y + h - cy1)
                cv2.rectangle(context, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                scale = max(1, 400 // max(1, context.shape[1]))
                context_big = cv2.resize(context, (context.shape[1] * scale, context.shape[0] * scale))
                cv2.imwrite(os.path.join(opt.out_dir, f"{stem}__context.jpg"), context_big)

    n_correct = sum(r["correct"] for r in rows)
    print()
    print(f"Total RBNR samples checked: {len(rows)} (all ground-truth state2)")
    print(f"Correct (pred_state==2): {n_correct} ({100*n_correct/len(rows):.2f}%)")
    print()
    print("Predicted-state distribution:")
    for k in sorted(pred_state_hist):
        print(f"  predicted state={k}: {pred_state_hist[k]} ({100*pred_state_hist[k]/len(rows):.1f}%)")

    print()
    wrong = [r for r in rows if not r["correct"]]
    print(f"WRONG predictions ({len(wrong)}):")
    for r in wrong:
        print(f"  {r['file']:30s} gt_digits={r['gt_digits']:6s} gt_state={r['gt_state']} "
              f"pred_state={r['pred_state']} state_logits={r['state_logits']} body_rect={r['body_rect']}")

    with open(os.path.join(opt.out_dir, "rbnr_ocr_predictions_full.json"), "w", encoding="utf8") as f:
        json.dump(rows, f, indent=2)

    print()
    print(f"Saved {len(wrong)*2} crop images (model-input + context) to: {opt.out_dir}/")
    print(f"Full per-sample dump: {opt.out_dir}/rbnr_ocr_predictions_full.json")


if __name__ == "__main__":
    main()
