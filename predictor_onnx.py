"""predictor_onnx.py for safe_crop_onnx.py -- uses your ONNX jersey-number model, with the SAME preprocessing and
decoding as inference_onnx.py (96x96 resize, BGR kept, /255, no mean/std; digit_1/digit_2/state logits).

Set the model path either by editing MODEL_PATH below or with an environment variable:
    export JERSEY_ONNX=/path/to/your_model.onnx
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get("JERSEY_ONNX", "PUT_YOUR_MODEL.onnx_PATH_HERE")
SIZE = 96
BATCH = 64

_sess = None
_in_name = None
_fixed_batch1 = False


def load():
    global _sess, _in_name, _fixed_batch1
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"ONNX model not found: {MODEL_PATH}\n"
                         f"Set it with:  export JERSEY_ONNX=/path/to/model.onnx   (or edit MODEL_PATH in predictor.py)")
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    _sess = ort.InferenceSession(MODEL_PATH, providers=prov)
    inp = _sess.get_inputs()[0]
    _in_name = inp.name
    _fixed_batch1 = isinstance(inp.shape[0], int) and inp.shape[0] == 1     # exported with a fixed batch of 1
    print(f"predictor: {os.path.basename(MODEL_PATH)}  providers={prov}  batch={'1 (fixed)' if _fixed_batch1 else BATCH}")


def _prep(pil_rgb):
    bgr = np.asarray(pil_rgb)[..., ::-1]                                   # PIL RGB -> cv2 BGR
    x = cv2.resize(np.ascontiguousarray(bgr), (SIZE, SIZE)).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)                                            # CHW


def _decode(d1, d2, state):
    # 0 = no number, 10 = blank second digit (same rule as inference_onnx.py)
    return ["" if s == 0 else (str(a) if b == 10 else f"{a}{b}") for a, b, s in zip(d1, d2, state)]


def predict(pil_images):
    """list[PIL.Image RGB] -> list[str] such as ['7', '77', '16'] ('' = model sees no number)."""
    names = [o.name for o in _sess.get_outputs()]
    res = []
    step = 1 if _fixed_batch1 else BATCH
    for i in range(0, len(pil_images), step):
        x = np.stack([_prep(im) for im in pil_images[i:i + step]])
        out = dict(zip(names, _sess.run(None, {_in_name: x})))
        res += _decode(out["digit_1_logits"].argmax(1), out["digit_2_logits"].argmax(1), out["state_logits"].argmax(1))
    return res
