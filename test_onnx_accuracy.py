"""
Accuracy of an exported ONNX model on the jersey test set (same logic as
jerseyNumberRecognitiorWithState_Testing.py, but with onnxruntime only).

  python scripts/test_onnx_accuracy.py --model kd_79.74_s2b1.25_dig.onnx \
      --test_image_path /home/eng_bhavana/workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset/

Prints the same four numbers as the testing script:
  2 digit / state / state2 / state1 accuracy.
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
    p.add_argument("--numberOfDigits", type=int, default=2)
    opt = p.parse_args()

    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
    sess = ort.InferenceSession(opt.model, providers=prov)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    print("providers:", sess.get_providers(), "| outputs:", names)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    assert images, "no images found in <test_image_path>/images"

    total_body = state_correct = n_le2 = n_3 = 0
    two_digit_correct = s1_correct = s2_correct = 0

    for img_path in tqdm(images, desc="Testing ONNX..."):
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org = cv2.imread(img_path)

        for item in data["data"]["jersey"]:
            total_body += 1
            digits = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            n_digits = len(digits)
            state = 2 if n_digits > 2 else 1
            if n_digits < opt.numberOfDigits:                       # blank padding (10), as in the testing script
                digits = digits + [10] * (opt.numberOfDigits - n_digits)
            if n_digits > 2:
                n_3 += 1
            else:
                n_le2 += 1

            x, y, w, h = item["body"]["rect"]
            crop = org[int(y): int(y + h), int(x): int(x + w)]
            crop = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.0   # BGR kept, /255 only
            inp = crop.transpose(2, 0, 1)[None]

            out = dict(zip(names, sess.run(None, {in_name: inp})))
            d1 = int(out["digit_1_logits"].argmax())
            d2 = int(out["digit_2_logits"].argmax())
            st = int(out["state_logits"].argmax())

            state_correct += st == state
            if state == 2 and st == 2:
                s2_correct += 1
            elif state == 1 and st == 1:
                s1_correct += 1
            if n_digits <= 2 and d1 == digits[0] and d2 == digits[1]:
                two_digit_correct += 1

    print("----")
    print("the accuracy of the 2 digit is %2f" % (100 * two_digit_correct / n_le2))
    print("the accuracy of the state is %.2f" % (100 * state_correct / total_body))
    print("the accuracy of the state2 is %.2f" % (100 * s2_correct / n_3))
    print("the accuracy of the state1 is %.2f" % (100 * s1_correct / n_le2))


if __name__ == "__main__":
    main()
