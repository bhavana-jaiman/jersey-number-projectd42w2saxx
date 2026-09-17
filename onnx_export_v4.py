import torch
import torch.onnx

# ----------------------------------------------------------
# Import your actual model definitions.
# Replace 'your_model_file' with the actual filename (no .py)
# where MultiTaskLearner, featureExtractor_Ying, block(), and
# weight_init_kaiming are defined.
# ----------------------------------------------------------
from your_model_file import MultiTaskLearner  # <-- update this import

# ----------------------------
# Config
# ----------------------------
PTH_PATH = "path/to/your_model.pth"
ONNX_PATH = "path/to/multitasklearner.onnx"
DEVICE = "cpu"
OUT_CHANNELS = 256  # matches out_channels=256 used in __init__

# Input size — based on your val/test pipeline (resize 114 -> center-crop 96)
DUMMY_INPUT_SHAPE = (1, 3, 96, 96)

# ----------------------------
# Load model
# ----------------------------
model = MultiTaskLearner(out_channels=OUT_CHANNELS)

checkpoint = torch.load(PTH_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # already a raw state_dict

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ----------------------------
# Dummy input for tracing
# ----------------------------
dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, device=DEVICE)

# forward() returns: x (flattened feature), digital, digit_1, digit_2
input_names = ["input"]
output_names = ["features", "digital", "digit_1", "digit_2"]

# ----------------------------
# Export
# ----------------------------
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input": {0: "batch_size"},
        **{name: {0: "batch_size"} for name in output_names},
    },
)

print(f"Model exported to {ONNX_PATH}")

# ----------------------------
# Verify + sanity-check against PyTorch output
# ----------------------------
import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model structure is valid.")

with torch.no_grad():
    torch_outputs = model(dummy_input)

ort_session = ort.InferenceSession(ONNX_PATH)
ort_outputs = ort_session.run(None, {"input": dummy_input.numpy()})

for name, t_out, o_out in zip(output_names, torch_outputs, ort_outputs):
    max_diff = np.abs(t_out.numpy() - o_out).max()
    print(f"{name}: max abs diff = {max_diff:.6f}")
