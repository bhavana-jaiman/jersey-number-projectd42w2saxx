#!/usr/bin/env python3
"""
Export the Jersey MultiTaskLearnerWithState PyTorch model to ONNX
for visualization in Netron.

Run from the project root:
    python onnx_export.py

Or specify a checkpoint/output:
    python onnx_export.py \
        --checkpoint checkpoints/202506161701456finetuning_best.pth \
        --output jersey_model.onnx
"""

import argparse
from pathlib import Path

import torch

# Your model definition
from subModules.backbone_ying import MultiTaskLearnerWithState


def load_checkpoint(model, checkpoint_path):
    """Load a normal state_dict or a checkpoint containing state_dict/model_state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Your checkpoint shown in the terminal is already a state_dict.
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    # Remove DataParallel/DDP prefix if it exists.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        print("\nWARNING: Missing keys:")
        for key in missing:
            print("  ", key)

    if unexpected:
        print("\nWARNING: Unexpected keys:")
        for key in unexpected:
            print("  ", key)

    if not missing and not unexpected:
        print("Checkpoint loaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/202506161701456finetuning_best.pth",
        help="Path to the trained .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        default="jersey_model.onnx",
        help="Output ONNX file",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=96,
        help="Input image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=96,
        help="Input image width",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n  {checkpoint_path.resolve()}\n"
            "Use --checkpoint to give the correct .pth path."
        )

    # The class in your backbone_ying.py has out_channels=128 by default.
    model = MultiTaskLearnerWithState(out_channels=128)
    load_checkpoint(model, checkpoint_path)

    model.eval()

    # Your dataset/model uses 96x96 RGB images.
    dummy_input = torch.randn(
        1, 3, args.height, args.width,
        dtype=torch.float32
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nExporting ONNX model...")
    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Output      : {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "features",
                "whole_number_logits",
                "digit_1_logits",
                "digit_2_logits",
                "state_logits",
            ],
            dynamic_axes={
                "image": {0: "batch_size"},
                "features": {0: "batch_size"},
                "whole_number_logits": {0: "batch_size"},
                "digit_1_logits": {0: "batch_size"},
                "digit_2_logits": {0: "batch_size"},
                "state_logits": {0: "batch_size"},
            },
        )

    print("\nSUCCESS!")
    print(f"ONNX file created at:\n  {output_path.resolve()}")

    # Optional ONNX validation.
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        print("ONNX checker: PASSED")
        print("\nYou can now open the .onnx file in Netron.")
    except ImportError:
        print(
            "\nONNX package is not installed, so validation was skipped."
        )
        print("Install it with:")
        print("  pip install onnx")


if __name__ == "__main__":
    main()
