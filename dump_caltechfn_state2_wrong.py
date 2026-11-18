"""
Diagnostic: dump per-sample state predictions for caltechfn's OWN 3+ digit
(state2) samples inside caltechfn_sample_350_clean (the ones NOT from
RBNR), using the exact same preprocessing as test_onnx_accuracy.py.

Context: state2_breakdown_by_source.py showed RBNR is now ~98-100%
accurate on state2, but caltechfn's own 98 state2 samples are only
29.59% accurate. This is a SEPARATE problem from the RBNR crop-geometry
work -- these are the project's own annotator-labeled caltechfn samples
(filtered earlier with filter_single_number.py), so the crop should
already match the expected style. This script finds out WHY they're
failing:
  (a) still a duplicate/composite-number labeling artifact that slipped
      through the earlier geometric filter (e.g. "21"+"21" -> "2121")
  (b) a genuinely hard crop (small, blurry, odd angle)
  (c) a state-head bias (confirm via predicted-state distribution)

Saves, for every WRONG caltechfn state2 prediction:
  - the exact 96x96 crop fed to the model (model-input)
  - a padded context view with body.rect box drawn (context)
  - raw state_logits

Usage:
    python dump_caltechfn_state2_wrong.py \
        --model ~/old_server_backup_16_10_2026/New_16_10_2026/kd_79.74_s2b1.25.onnx \
        --test_image_path ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/ \
        --out_dir caltechfn_state2_wrong_crops
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
    p.add_argument("--out_dir", default="caltechfn_state2_wrong_crops")
    opt = p.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    print("providers:", sess.get_providers(), "| outputs:", names)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    caltech_images = [p for p in images if not os.path.basename(p).startswith(("set2_org_", "set3_org_"))]
    print(f"Found {len(caltech_images)} caltechfn images out of {len(images)} total")

    pred_state_hist = {}
    rows = []

    for img_path in caltech_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org_bgr = cv2.imread(img_path)
        if org_bgr is None:
            continue
        org = cv2.cvtColor(org_bgr, cv2.COLOR_BGR2RGB)

        for item in data["data"]["jersey"]:
            digits_gt = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits_gt)
            if n_digits <= 2:
                continue  # only state2 samples
            gt_state = 2

            x, y, w, h = item["body"]["rect"]
            crop_raw = org[int(y): int(y + h), int(x): int(x + w)]
            if crop_raw.size == 0:
                continue
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
                exact_input = cv2.cvtColor(crop_96, cv2.COLOR_RGB2BGR)
                exact_input_big = cv2.resize(exact_input, (384, 384), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(opt.out_dir, f"{stem}__model_input.jpg"), exact_input_big)

                H, W = org_bgr.shape[:2]
                pad_x, pad_y = w * 0.6, h * 0.6
                cx1 = max(0, int(x - pad_x)); cy1 = max(0, int(y - pad_y))
                cx2 = min(W, int(x + w + pad_x)); cy2 = min(H, int(y + h + pad_y))
                context = org_bgr[cy1:cy2, cx1:cx2].copy()
                rx1, ry1 = int(x - cx1), int(y - cy1)
                rx2, ry2 = int(x + w - cx1), int(y + h - cy1)
                cv2.rectangle(context, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                scale = max(1, 400 // max(1, context.shape[1]))
                context_big = cv2.resize(context, (context.shape[1] * scale, context.shape[0] * scale))
                cv2.imwrite(os.path.join(opt.out_dir, f"{stem}__context.jpg"), context_big)

    n_correct = sum(r["correct"] for r in rows)
    print()
    print(f"Total caltechfn state2 samples checked: {len(rows)}")
    print(f"Correct: {n_correct} ({100*n_correct/len(rows):.2f}%)")
    print()
    print("Predicted-state distribution:")
    for k in sorted(pred_state_hist):
        print(f"  predicted state={k}: {pred_state_hist[k]} ({100*pred_state_hist[k]/len(rows):.1f}%)")

    print()
    wrong = [r for r in rows if not r["correct"]]
    print(f"WRONG predictions ({len(wrong)}):")
    for r in wrong:
        print(f"  {r['file']:30s} gt_digits={r['gt_digits']:8s} pred_state={r['pred_state']} "
              f"state_logits={r['state_logits']} body_rect={r['body_rect']}")

    with open(os.path.join(opt.out_dir, "caltechfn_state2_predictions_full.json"), "w", encoding="utf8") as f:
        json.dump(rows, f, indent=2)

    print()
    print(f"Saved {len(wrong)*2} crop images (model-input + context) to: {opt.out_dir}/")
    print(f"Full per-sample dump: {opt.out_dir}/caltechfn_state2_predictions_full.json")


if __name__ == "__main__":
    main()
