!/usr/bin/env python3
"""
Jersey Number Recognition - PyTorch -> ONNX exporter

This exporter lets you provide BOTH the checkpoint and backbone file
from the command line.

Example:
python onnx_export_v2.py \
    --checkpoint checkpoints/backbone_v2_train_15092026_v1/jerseyNumberRecognition_best.pth \
    --backbone subModules/backbone_ying_v2.py \
    --output onnx_model/jersey_model_v2.onnx

The backbone module must define:
    MultiTaskLearnerWithState

The model is instantiated with out_channels=256 because the checkpoint
you showed previously contains 256-channel feature-extractor/classifier
weights.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import torch


def load_backbone_class(backbone_path):
    """Load MultiTaskLearnerWithState from a user-specified .py file."""
    backbone_path = Path(backbone_path).resolve()

    if not backbone_path.exists():
        raise FileNotFoundError(
            f"Backbone file not found: {backbone_path}"
        )

    if backbone_path.suffix != ".py":
        raise ValueError(
            f"Backbone must be a Python .py file: {backbone_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "user_backbone_ying_v2",
        str(backbone_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load backbone module: {backbone_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_backbone_ying_v2"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "MultiTaskLearnerWithState"):
        raise AttributeError(
            f"{backbone_path} does not contain "
            f"'MultiTaskLearnerWithState'."
        )

    return module.MultiTaskLearnerWithState


def load_checkpoint(model, checkpoint_path):
    """Load a raw state_dict or state_dict inside a checkpoint dictionary."""
    checkpoint_path = Path(checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # The checkpoint itself may already be a state_dict.
            state_dict = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    print("Loading checkpoint with strict=True...")

    # Strict loading is intentional.
    # It prevents exporting the wrong architecture by accident.
    model.load_state_dict(cleaned_state_dict, strict=True)

    return model


def export_onnx(model, output_path, height, width, opset):
    model.eval()
    model.cpu()

    dummy_input = torch.randn(
        1, 3, height, width,
        dtype=torch.float32,
    )

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Your MultiTaskLearnerWithState forward() returns:
    # x, digital, digit_1, digit_2, logits_state
    output_names = [
        "features",
        "digital",
        "digit_1",
        "digit_2",
        "logits_state",
    ]

    print()
    print("Starting ONNX export...")
    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Output path : {output_path}")
    print(f"Opset       : {opset}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes={
                "input": {0: "batch"},
                "features": {0: "batch"},
                "digital": {0: "batch"},
                "digit_1": {0: "batch"},
                "digit_2": {0: "batch"},
                "logits_state": {0: "batch"},
            },
        )

    print()
    print("=" * 70)
    print("ONNX EXPORT SUCCESSFUL")
    print("=" * 70)
    print(f"Saved model: {output_path}")
    print(f"File size  : {output_path.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export MultiTaskLearnerWithState to ONNX."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained .pth checkpoint.",
    )

    parser.add_argument(
        "--backbone",
        required=True,
        help=(
            "Path to the backbone Python file containing "
            "MultiTaskLearnerWithState, e.g. "
            "subModules/backbone_ying_v2.py"
        ),
    )

    parser.add_argument(
        "--output",
        default="onnx_model/jersey_model_v2.onnx",
        help="Output ONNX file path.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=96,
        help="Input image height. Default: 96.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=96,
        help="Input image width. Default: 96.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17.",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=256,
        help=(
            "Model feature/output channels. Default: 256. "
            "Must match the checkpoint architecture."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("JERSEY NUMBER RECOGNITION - ONNX EXPORT")
    print("=" * 70)
    print(f"Checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Backbone   : {Path(args.backbone).resolve()}")
    print(f"Channels   : {args.channels}")
    print(f"Input      : 3 x {args.height} x {args.width}")
    print(f"Output     : {Path(args.output).resolve()}")
    print("=" * 70)

    # Load the requested backbone file.
    ModelClass = load_backbone_class(args.backbone)

    print()
    print(f"Loaded model class: {ModelClass.__name__}")

    # Your shown backbone defaults to 256, but we pass it explicitly.
    try:
        model = ModelClass(out_channels=args.channels)
    except TypeError as e:
        raise TypeError(
            "Could not construct MultiTaskLearnerWithState with "
            f"out_channels={args.channels}. "
            "Check the constructor in the supplied backbone file."
        ) from e

    print(f"Created model with out_channels={args.channels}")

    # Load trained weights.
    model = load_checkpoint(model, args.checkpoint)

    print("Checkpoint loaded successfully.")

    # Export.
    export_onnx(
        model=model,
        output_path=args.output,
        height=args.height,
        width=args.width,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
