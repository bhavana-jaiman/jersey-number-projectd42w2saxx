"""
Export a JerseyNumber (MultiTaskLearnerWithState) checkpoint to ONNX and verify it.

Run from the project folder that contains `subModules/backbone_New.py`
(or pass --project_dir). Works on CPU only.

Example:
  python scripts/export_onnx.py \
      --checkpoint kd_79.74_s2b1.25_dig.pth \
      --out kd_79.74_s2b1.25_dig.onnx \
      --verify_images <test_folder>/images

What the ONNX file contains
  input  : "input"          float32 [N, 3, 96, 96]  (RGB, divided by 255, NCHW)
  output : "features"       pooled feature vector (kept so the output order matches the .pth model)
           "digit_1_logits" [N, 10]   digits 0-9
           "digit_2_logits" [N, 11]   digits 0-9, index 10 = blank
           "state_logits"   [N, 3]    0 = no number, 1 = 1-2 digits, 2 = 3+ digits
  All biases (incl. the calibrated classifier_state.4.bias / digit_1.bias / digit_2.bias)
  are stored in the graph. Decision = plain argmax of each logits output.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

INPUT_NAMES = ["input"]
OUTPUT_NAMES = ["features", "digit_1_logits", "digit_2_logits", "state_logits"]
SIZE = 96


def load_model(project_dir, checkpoint, backbone_module):
    import importlib
    sys.path.insert(0, os.path.abspath(project_dir))
    # same module as the testing script imports MultiTaskLearnerWithState from
    MultiTaskLearnerWithState = importlib.import_module(backbone_module).MultiTaskLearnerWithState

    model = MultiTaskLearnerWithState()
    sd = torch.load(checkpoint, map_location="cpu")        # CPU: works without a usable GPU
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)                  # wrong checkpoint -> error, not random heads
    model.eval()                                            # BatchNorm uses stored stats, dropout off
    return model, sd


def export(model, out, opset, dynamic_batch):
    dummy = torch.zeros(1, 3, SIZE, SIZE, dtype=torch.float32)
    kwargs = dict(input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                  opset_version=opset, do_constant_folding=True)
    if dynamic_batch:
        kwargs["dynamic_axes"] = {n: {0: "batch"} for n in INPUT_NAMES + OUTPUT_NAMES}
    try:
        torch.onnx.export(model, dummy, out, dynamo=False, **kwargs)   # classic exporter, single file
    except TypeError:                                                  # older torch without `dynamo=`
        torch.onnx.export(model, dummy, out, **kwargs)


def finalize(out, ckpt_name):
    """Single self-contained .onnx (no external .data), metadata, checker."""
    proto = onnx.load(out)                                  # also loads external data if any
    for k, v in {
        "model_class": "MultiTaskLearnerWithState",
        "checkpoint": os.path.basename(ckpt_name),
        "preprocessing": "cv2.imread -> BGR2RGB -> crop body box -> resize 96x96 -> float32 / 255 -> NCHW, no mean/std",
        "outputs": "features; digit_1_logits[10]; digit_2_logits[11] (10=blank); state_logits[3]",
        "decision": "argmax of each logits output, no post-processing",
    }.items():
        e = proto.metadata_props.add()
        e.key, e.value = k, v
    onnx.save_model(proto, out, save_as_external_data=False)
    for extra in (out + ".data", out + "_data"):
        if os.path.exists(extra):
            os.remove(extra)
    onnx.checker.check_model(onnx.load(out))


def bias_present(out, sd):
    """Check that the calibrated biases are stored as weights in the ONNX graph."""
    inits = [onnx.numpy_helper.to_array(i) for i in onnx.load(out).graph.initializer]
    res = {}
    for name in ("classifier_state.4.bias", "digit_1.bias", "digit_2.bias"):
        b = sd[name].detach().cpu().numpy()
        res[name] = any(a.shape == b.shape and np.allclose(a, b, atol=1e-6) for a in inits)
    return res


def load_images(folder, limit):
    import cv2
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))[:limit]
    arr = []
    for f in files:
        im = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (SIZE, SIZE)).astype(np.float32) / 255.0
        arr.append(im.transpose(2, 0, 1))
    if not arr:
        print("WARNING: no .jpg/.png found in", folder, "- did you mean <folder>/images ? (parity check uses random inputs only)")
        return None
    return np.stack(arr)


def verify(model, out, images, n_random):
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.random((n_random, 3, SIZE, SIZE), dtype=np.float32)
    if images is not None:
        x = np.concatenate([images, x])
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(x))]
    got = sess.run(None, {INPUT_NAMES[0]: x})
    ok = True
    print("\nPyTorch vs ONNX on %d inputs (%d real images):" % (len(x), 0 if images is None else len(images)))
    for name, r, g in zip(OUTPUT_NAMES, ref, got):
        diff = float(np.abs(r - g).max())
        same = "-" if name == "features" else "%.2f%%" % (100 * (r.argmax(1) == g.argmax(1)).mean())
        flag = diff < 1e-3
        ok &= flag
        print("  %-15s max|diff| = %.2e   argmax agreement = %s   %s" % (name, diff, same, "OK" if flag else "MISMATCH"))
    # also test another batch size to confirm the dynamic batch axis
    got1 = sess.run(None, {INPUT_NAMES[0]: x[:1]})
    print("  batch=1 run: output shapes", [tuple(g.shape) for g in got1])
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--project_dir", default=".", help="folder that contains subModules/backbone_New.py")
    p.add_argument("--backbone_module", default="subModules.backbone_New",
                   help="module that defines MultiTaskLearnerWithState (see the import in the testing script)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--fixed_batch", action="store_true", help="export with batch size fixed to 1")
    p.add_argument("--verify_images", default=None, help="optional folder of body crops/images for the parity check")
    p.add_argument("--verify_limit", type=int, default=200)
    opt = p.parse_args()

    model, sd = load_model(opt.project_dir, opt.checkpoint, opt.backbone_module)
    export(model, opt.out, opt.opset, not opt.fixed_batch)
    finalize(opt.out, opt.checkpoint)

    print("Exported:", opt.out, "(%.2f MB)" % (os.path.getsize(opt.out) / 1e6))
    print("Calibrated biases stored in ONNX:", bias_present(opt.out, sd))
    imgs = load_images(opt.verify_images, opt.verify_limit) if opt.verify_images else None
    ok = verify(model, opt.out, imgs, n_random=32)
    print("\nRESULT:", "PASSED - ONNX matches the PyTorch checkpoint" if ok else "FAILED - do not use this ONNX")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
