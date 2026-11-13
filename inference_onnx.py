"""
ONNX inference for the jersey-number model (same command style as the client's scripts).

  python scripts/inference_onnx.py --model <model.onnx> --input <image_or_image_directory>

--input : a single image or a directory of images (body crops).
Results are saved in output/output_inference_onnx/ (re-created before each run):
  predictions.json, annotated images.
Needs only: onnxruntime, opencv-python, numpy (no PyTorch, no model code).
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 96


def preprocess(img_bgr):
    x = cv2.resize(img_bgr, (SIZE, SIZE)).astype(np.float32) / 255.0   # BGR kept, no mean/std
    return x.transpose(2, 0, 1)[None]                                  # NCHW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output/output_inference_onnx")
    opt = p.parse_args()

    files = [opt.input] if os.path.isfile(opt.input) else sorted(
        f for e in ("*.jpg", "*.jpeg", "*.png") for f in glob.glob(os.path.join(opt.input, e)))
    assert files, "no images found in --input"

    shutil.rmtree(opt.output, ignore_errors=True)
    os.makedirs(opt.output)

    prov = [p_ for p_ in ("CUDAExecutionProvider", "CPUExecutionProvider") if p_ in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]          # features, digit_1, digit_2, state

    results = []
    for f in files:
        img = cv2.imread(f)
        out = dict(zip(names, sess.run(None, {in_name: preprocess(img)})))
        d1 = int(out["digit_1_logits"].argmax())
        d2 = int(out["digit_2_logits"].argmax())          # 10 = blank
        state = int(out["state_logits"].argmax())         # 0 none, 1 = 1-2 digits, 2 = 3+ digits
        number = "" if state == 0 else (str(d1) if d2 == 10 else f"{d1}{d2}")
        results.append({"image": os.path.basename(f), "digit_1": d1, "digit_2": d2, "state": state, "number": number})
        cv2.putText(img, f"{number or '-'} (state {state})", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(opt.output, os.path.basename(f)), img)

    with open(os.path.join(opt.output, "predictions.json"), "w", encoding="utf8") as fp:
        json.dump(results, fp, indent=2)
    print(f"{len(results)} images -> {opt.output}/predictions.json")


if __name__ == "__main__":
    main()
