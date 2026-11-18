"""
Breaks down state2 accuracy by SOURCE (caltechfn vs RBNR) on the merged
caltechfn_sample_350_clean dataset, using the exact same logic as
test_onnx_accuracy.py.

Why: dump_rbnr_ocr_predictions.py showed RBNR alone scores 100% (89/89) on
state2 with confident correct logits. But test_onnx_accuracy.py on the full
439-sample merged folder reported state2 = 63.10%. Since test_onnx_accuracy.py
computes state2 accuracy over ALL samples with n_digits > 2 (not just RBNR),
if caltechfn also contributes some 3+ digit samples to that denominator,
and those are performing worse, the combined average would be dragged down
even though RBNR itself is perfect. This script checks that directly.

Usage:
    python state2_breakdown_by_source.py \
        --model ~/old_server_backup_16_10_2026/New_16_10_2026/kd_79.74_s2b1.25.onnx \
        --test_image_path ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


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
    assert images, "no images found"

    # per-source tallies
    stats = {
        "rbnr": {"n_3": 0, "s2_correct": 0, "wrong_files": []},
        "caltechfn": {"n_3": 0, "s2_correct": 0, "wrong_files": []},
    }

    for img_path in tqdm(images, desc="Testing..."):
        fname = os.path.basename(img_path)
        source = "rbnr" if fname.startswith(("set2_org_", "set3_org_")) else "caltechfn"

        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for item in data["data"]["jersey"]:
            digits = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits)
            if n_digits <= 2:
                continue  # only care about state2 (n_digits > 2) here

            stats[source]["n_3"] += 1

            x, y, w, h = item["body"]["rect"]
            crop = org[int(y): int(y + h), int(x): int(x + w)]
            crop = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.0
            inp = crop.transpose(2, 0, 1)[None]

            out = dict(zip(names, sess.run(None, {in_name: inp})))
            st = int(out["state_logits"][0].argmax())

            if st == 2:
                stats[source]["s2_correct"] += 1
            else:
                stats[source]["wrong_files"].append((fname, "".join(str(d) for d in digits), st))

    print("\n==== state2 accuracy breakdown by source ====")
    total_n3 = sum(s["n_3"] for s in stats.values())
    total_correct = sum(s["s2_correct"] for s in stats.values())
    for src, s in stats.items():
        if s["n_3"] == 0:
            print(f"{src}: no 3+ digit samples")
            continue
        acc = 100 * s["s2_correct"] / s["n_3"]
        print(f"{src}: {s['s2_correct']}/{s['n_3']} = {acc:.2f}%")
        if s["wrong_files"]:
            print(f"  wrong examples (first 10): {s['wrong_files'][:10]}")
    print(f"\nCOMBINED (matches test_onnx_accuracy.py's state2 number): "
          f"{total_correct}/{total_n3} = {100*total_correct/total_n3:.2f}%")


if __name__ == "__main__":
    main()
