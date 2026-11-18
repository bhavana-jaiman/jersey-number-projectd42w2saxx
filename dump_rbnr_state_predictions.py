"""
Diagnostic: dump per-sample state predictions ONLY for the 101 RBNR images
inside caltechfn_sample_350_clean, using the exact same preprocessing as
test_onnx_accuracy.py (crop body.rect -> resize 96x96 -> /255, RGB).

Purpose: find out WHY state2 accuracy collapsed on unseen data (24%) while
raw digit accuracy barely moved (79%). This prints, for every RBNR sample
(all ground-truth state2, i.e. 3+ digit bibs):
  - predicted state (1 or 2)
  - predicted digit_1, digit_2
  - whether digit_1/digit_2 look like they read the FIRST TWO digits of the
    bib correctly (even though state2 samples aren't scored on digits in
    the real script, this tells us whether the digit head is reading
    something sane or is confused -- a sanity cross-check)

It also prints a confusion summary: how many predicted-state==1 vs ==2,
so we can see whether this is a state-head generalization problem.

Usage:
    python dump_rbnr_state_predictions.py \
        --model kd_79.74_s2b1.25.onnx \
        --test_image_path ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/
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
    opt = p.parse_args()

    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    print("providers:", sess.get_providers(), "| outputs:", names)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    # RBNR files are named set2_org_... / set3_org_...
    rbnr_images = [p for p in images if os.path.basename(p).startswith(("set2_org_", "set3_org_"))]
    print(f"Found {len(rbnr_images)} RBNR images out of {len(images)} total")

    pred_state_hist = {}
    rows = []

    for img_path in rbnr_images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for item in data["data"]["jersey"]:
            digits_gt = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits_gt)
            gt_state = 2 if n_digits > 2 else 1

            x, y, w, h = item["body"]["rect"]
            crop = org[int(y): int(y + h), int(x): int(x + w)]
            crop = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.0
            inp = crop.transpose(2, 0, 1)[None]

            out = dict(zip(names, sess.run(None, {in_name: inp})))
            d1 = int(out["digit_1_logits"].argmax())
            d2 = int(out["digit_2_logits"].argmax())
            st_logits = out["state_logits"][0]
            st = int(st_logits.argmax())

            pred_state_hist[st] = pred_state_hist.get(st, 0) + 1

            rows.append({
                "file": os.path.basename(img_path),
                "gt_digits": "".join(str(d) for d in digits_gt),
                "gt_state": gt_state,
                "pred_state": st,
                "pred_d1": d1,
                "pred_d2": d2,
                "state_logits": [round(float(x), 2) for x in st_logits],
                "correct": st == gt_state,
            })

    n_correct = sum(r["correct"] for r in rows)
    print()
    print(f"Total RBNR samples checked: {len(rows)} (all ground-truth state2)")
    print(f"Correct (pred_state==2): {n_correct} ({100*n_correct/len(rows):.2f}%)")
    print()
    print("Predicted-state distribution across all RBNR (ground-truth-state2) samples:")
    for k in sorted(pred_state_hist):
        print(f"  predicted state={k}: {pred_state_hist[k]} ({100*pred_state_hist[k]/len(rows):.1f}%)")

    print()
    print("Sample of WRONG predictions (first 15):")
    wrong = [r for r in rows if not r["correct"]]
    for r in wrong[:15]:
        print(f"  {r['file']:35s} gt_digits={r['gt_digits']:6s} gt_state={r['gt_state']} "
              f"pred_state={r['pred_state']} state_logits={r['state_logits']}")

    out_path = "rbnr_state_predictions.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(rows, f, indent=2)
    print()
    print(f"Full per-sample dump written to: {out_path}")


if __name__ == "__main__":
    main()
