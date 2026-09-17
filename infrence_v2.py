#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


# ============================================================
# Constants
# ============================================================

INPUT_SIZE = 96

# IMPORTANT:
# Your current checkpoint model uses 256-dimensional features.
FEATURE_DIM = 256

# Digit heads in the PyTorch architecture shown in backbone_ying.py
DIGIT1_CLASSES = 11
DIGIT2_CLASSES = 11

# Class index 10 is treated as "no digit" when the 11-class
# digit head is used.
BLANK_CLASS = 10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Argument parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Jersey Number Classification Inference"
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model path. Can be .onnx, .pth or .pt"
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or directory containing images",
    )

    parser.add_argument(
        "--output_dir",
        default="inference_results",
        help="Directory where inference results are saved",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum confidence required to mark prediction "
            "as accepted. Default: 0.0"
        ),
    )

    parser.add_argument(
        "--feature_dim",
        type=int,
        default=FEATURE_DIM,
        choices=[128, 256],
        help="Feature dimension of PyTorch checkpoint",
    )

    parser.add_argument(
        "--checkpoint_key",
        default=None,
        help=(
            "Optional checkpoint key containing model state_dict. "
            "Examples: state_dict, model_state_dict, model"
        ),
    )

    parser.add_argument(
        "--no_annotation",
        action="store_true",
        help="Do not save annotated images",
    )

    return parser.parse_args()


# ============================================================
# Image utilities
# ============================================================

def collect_images(input_path):

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if input_path.is_file():

        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {input_path}"
            )

        return [input_path]

    image_paths = sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {input_path}"
        )

    return image_paths


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image):

    # Original image remains untouched for annotation.
    #
    # Model input:
    #     3 x 96 x 96
    #
    # Resize directly to 96x96.
    resized = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # BGR -> RGB
    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    resized = resized.astype(np.float32) / 255.0

    # IMPORTANT:
    # This normalization must match the normalization used
    # during training/testing.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    resized = (resized - mean) / std

    # HWC -> CHW
    tensor = resized.transpose(2, 0, 1)

    # Add batch dimension
    tensor = np.expand_dims(
        tensor,
        axis=0,
    ).astype(np.float32)

    return tensor


# ============================================================
# Softmax
# ============================================================

def softmax(logits):

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
    )


# ============================================================
# Prediction decoding
# ============================================================

def decode_digit(logits):

    probabilities = softmax(logits)

    class_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_id]
    )

    return class_id, confidence, probabilities


def decode_number(digit1_logits, digit2_logits):

    digit1, conf1, probs1 = decode_digit(
        digit1_logits
    )

    digit2, conf2, probs2 = decode_digit(
        digit2_logits
    )

    # --------------------------------------------------------
    # Single digit
    # --------------------------------------------------------

    if digit2 == BLANK_CLASS:

        number = str(digit1)

        confidence = conf1

        return {
            "number": number,
            "digit1": digit1,
            "digit2": None,
            "digit1_confidence": conf1,
            "digit2_confidence": None,
            "confidence": confidence,
            "digit1_probabilities": probs1.tolist(),
            "digit2_probabilities": probs2.tolist(),
        }

    # --------------------------------------------------------
    # Two digit
    # --------------------------------------------------------

    number = f"{digit1}{digit2}"

    confidence = min(
        conf1,
        conf2,
    )

    return {
        "number": number,
        "digit1": digit1,
        "digit2": digit2,
        "digit1_confidence": conf1,
        "digit2_confidence": conf2,
        "confidence": confidence,
        "digit1_probabilities": probs1.tolist(),
        "digit2_probabilities": probs2.tolist(),
    }


# ============================================================
# PyTorch checkpoint model
# ============================================================

def build_pytorch_model(feature_dim, device):

    import torch

    # Import YOUR backbone implementation.
    #
    # This must be the same backbone used during training.
    from subModules.backbone_ying import MultiTaskLearner

    model = MultiTaskLearner(
        out_channels=feature_dim
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# Checkpoint loading
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        # Remove common DataParallel prefix.
        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def load_checkpoint(
    model,
    checkpoint_path,
    device,
    checkpoint_key=None,
):

    import torch

    print(
        f"\nLoading checkpoint:\n"
        f"  {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Case 1:
    # checkpoint itself is state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if checkpoint_key is not None:

            if checkpoint_key not in checkpoint:

                raise KeyError(
                    f"Checkpoint key '{checkpoint_key}' "
                    f"not found.\n"
                    f"Available keys:\n"
                    f"{list(checkpoint.keys())}"
                )

            state_dict = checkpoint[
                checkpoint_key
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"],
            dict,
        ):

            state_dict = checkpoint[
                "model"
            ]

        else:

            # Assume checkpoint itself is state_dict.
            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format. "
            "Expected a PyTorch state_dict/checkpoint dictionary."
        )

    state_dict = clean_state_dict(
        state_dict
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    if missing_keys:

        print(
            "\nWARNING: Missing checkpoint keys:"
        )

        for key in missing_keys:
            print(
                f"  {key}"
            )

    if unexpected_keys:

        print(
            "\nWARNING: Unexpected checkpoint keys:"
        )

        for key in unexpected_keys:
            print(
                f"  {key}"
            )

    if not missing_keys and not unexpected_keys:

        print(
            "Checkpoint loaded successfully."
        )

    else:

        print(
            "\nCheckpoint loaded with "
            "missing/unexpected keys."
        )

    model.eval()

    return model


# ============================================================
# PyTorch inference
# ============================================================

def run_pytorch_inference(
    model,
    tensor,
    device,
):

    import torch

    tensor = torch.from_numpy(
        tensor
    ).to(device)

    with torch.no_grad():

        output = model(
            tensor
        )

    # Expected:
    #
    # x
    # digital
    # digit_1
    # digit_2
    #
    # The model source returns these four values.
    if not isinstance(output, (tuple, list)):

        raise RuntimeError(
            "Unexpected PyTorch model output. "
            "Expected tuple/list containing "
            "features and classification heads."
        )

    if len(output) < 4:

        raise RuntimeError(
            f"Unexpected number of model outputs: "
            f"{len(output)}"
        )

    features = output[0]

    digital_logits = output[1]

    digit1_logits = output[2]

    digit2_logits = output[3]

    return (
        features,
        digital_logits,
        digit1_logits,
        digit2_logits,
    )


# ============================================================
# ONNX model
# ============================================================

def create_onnx_session(
    model_path,
    device,
):

    import onnxruntime as ort

    if device == "cuda":

        available = (
            ort.get_available_providers()
        )

        if (
            "CUDAExecutionProvider"
            in available
        ):

            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        else:

            print(
                "\nCUDAExecutionProvider is "
                "not available."
            )

            print(
                "Falling back to CPU."
            )

            providers = [
                "CPUExecutionProvider"
            ]

    else:

        providers = [
            "CPUExecutionProvider"
        ]

    print(
        "\nONNX Runtime providers:"
    )

    print(
        providers
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    print(
        "Active providers:"
    )

    print(
        session.get_providers()
    )

    return session


# ============================================================
# ONNX output identification
# ============================================================

def identify_onnx_outputs(session):

    outputs = session.get_outputs()

    print(
        "\nONNX outputs:"
    )

    for i, output in enumerate(outputs):

        print(
            f"  [{i}] "
            f"{output.name} "
            f"{output.shape}"
        )

    names = [
        output.name
        for output in outputs
    ]

    digit1_name = None
    digit2_name = None
    feature_name = None

    # Exact/near-exact name matching.
    for name in names:

        lower = name.lower()

        if (
            lower in {
                "digit1",
                "digit_1",
            }
            and digit1_name is None
        ):
            digit1_name = name

        elif (
            lower == "digit_2"
            and digit2_name is None
        ):
            digit2_name = name

        elif (
            lower in {
                "features",
                "feature",
                "feat",
            }
            and feature_name is None
        ):
            feature_name = name

    # --------------------------------------------------------
    # If exact names are unavailable, identify using shapes.
    # --------------------------------------------------------

    if digit1_name is None:

        candidates = []

        for output in outputs:

            shape = output.shape

            if (
                len(shape) == 2
                and shape[-1] == 11
            ):

                candidates.append(
                    output.name
                )

        if len(candidates) >= 1:

            digit1_name = candidates[0]

    if digit2_name is None:

        candidates = []

        for output in outputs:

            shape = output.shape

            if (
                len(shape) == 2
                and shape[-1] == 11
            ):

                if output.name != digit1_name:

                    candidates.append(
                        output.name
                    )

        if candidates:

            digit2_name = candidates[0]

    if (
        digit1_name is None
        or digit2_name is None
    ):

        raise RuntimeError(
            "\nCould not automatically identify "
            "digit_1 and digit_2 ONNX outputs.\n\n"
            "Available outputs:\n"
            + "\n".join(
                f"  {o.name}: {o.shape}"
                for o in outputs
            )
            + "\n\n"
            "Use the output names from your "
            "actual ONNX model."
        )

    print(
        f"\nUsing ONNX digit outputs:"
    )

    print(
        f"  Digit 1 : {digit1_name}"
    )

    print(
        f"  Digit 2 : {digit2_name}"
    )

    if feature_name:

        print(
            f"  Feature : {feature_name}"
        )

    return (
        digit1_name,
        digit2_name,
        feature_name,
    )


# ============================================================
# ONNX inference
# ============================================================

def run_onnx_inference(
    session,
    tensor,
    digit1_name,
    digit2_name,
    feature_name,
):

    input_name = (
        session.get_inputs()[0].name
    )

    output_names = [
        digit1_name,
        digit2_name,
    ]

    if feature_name:

        output_names.append(
            feature_name
        )

    outputs = session.run(
        output_names,
        {
            input_name: tensor
        },
    )

    digit1_logits = outputs[0]

    digit2_logits = outputs[1]

    features = None

    if feature_name:

        features = outputs[2]

    return (
        features,
        digit1_logits,
        digit2_logits,
    )


# ============================================================
# Annotation
# ============================================================

def draw_prediction(
    image,
    prediction,
):

    # VERY IMPORTANT:
    # Always start with a copy of the ORIGINAL image.
    annotated = image.copy()

    number = prediction["number"]

    confidence = prediction[
        "confidence"
    ]

    text = (
        f"Number: {number} | "
        f"Confidence: {confidence:.3f}"
    )

    # Background rectangle.
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

    thickness = 2

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = 5
    y = 5

    cv2.rectangle(
        annotated,
        (
            x,
            y,
        ),
        (
            x + text_width + 10,
            y + text_height + baseline + 10,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        annotated,
        text,
        (
            x + 5,
            y + text_height + 5,
        ),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    return annotated


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    model_path = Path(
        args.model
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model does not exist: "
            f"{model_path}"
        )

    input_paths = collect_images(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    if output_dir.exists():

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated_dir = (
        output_dir / "annotated"
    )

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = {}

    suffix = (
        model_path.suffix.lower()
    )

    # ========================================================
    # Load model
    # ========================================================

    pytorch_model = None
    onnx_session = None

    digit1_name = None
    digit2_name = None
    feature_name = None

    device = args.device

    if suffix == ".onnx":

        print(
            "\n========================================"
        )

        print(
            "Loading ONNX model"
        )

        print(
            "========================================"
        )

        onnx_session = (
            create_onnx_session(
                model_path,
                device,
            )
        )

        input_meta = (
            onnx_session.get_inputs()[0]
        )

        print(
            f"ONNX input: "
            f"{input_meta.name} "
            f"{input_meta.shape}"
        )

        (
            digit1_name,
            digit2_name,
            feature_name,
        ) = identify_onnx_outputs(
            onnx_session
        )

    elif suffix in {
        ".pth",
        ".pt",
    }:

        print(
            "\n========================================"
        )

        print(
            "Loading PyTorch checkpoint"
        )

        print(
            "========================================"
        )

        import torch

        if (
            device == "cuda"
            and torch.cuda.is_available()
        ):

            torch_device = torch.device(
                "cuda"
            )

        else:

            torch_device = torch.device(
                "cpu"
            )

            if device == "cuda":

                print(
                    "CUDA is not available "
                    "to PyTorch."
                )

                print(
                    "Using CPU."
                )

        print(
            f"PyTorch device: "
            f"{torch_device}"
        )

        print(
            f"Feature dimension: "
            f"{args.feature_dim}"
        )

        pytorch_model = (
            build_pytorch_model(
                args.feature_dim,
                torch_device,
            )
        )

        pytorch_model = load_checkpoint(
            pytorch_model,
            model_path,
            torch_device,
            args.checkpoint_key,
        )

    else:

        raise ValueError(
            "Unsupported model format.\n"
            "Use .onnx, .pth or .pt"
        )

    # ========================================================
    # Process images
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        f"Images to process: "
        f"{len(input_paths)}"
    )

    print(
        "========================================\n"
    )

    for index, image_path in enumerate(
        input_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(input_paths)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "  ERROR: Could not read image."
            )

            continue

        # Keep original image for annotation.
        original_image = image.copy()

        # Preprocess.
        tensor = preprocess_image(
            image
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        if onnx_session is not None:

            (
                features,
                digit1_logits,
                digit2_logits,
            ) = run_onnx_inference(
                onnx_session,
                tensor,
                digit1_name,
                digit2_name,
                feature_name,
            )

        else:

            (
                features,
                _digital_logits,
                digit1_logits,
                digit2_logits,
            ) = run_pytorch_inference(
                pytorch_model,
                tensor,
                torch_device,
            )

            digit1_logits = (
                digit1_logits
                .detach()
                .cpu()
                .numpy()
            )

            digit2_logits = (
                digit2_logits
                .detach()
                .cpu()
                .numpy()
            )

            if features is not None:

                features = (
                    features
                    .detach()
                    .cpu()
                    .numpy()
                )

        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        digit1_logits = np.asarray(
            digit1_logits
        )

        digit2_logits = np.asarray(
            digit2_logits
        )

        if digit1_logits.ndim == 2:

            digit1_logits = (
                digit1_logits[0]
            )

        if digit2_logits.ndim == 2:

            digit2_logits = (
                digit2_logits[0]
            )

        # ----------------------------------------------------
        # Validate feature dimension
        # ----------------------------------------------------

        feature_dimension_detected = None

        if features is not None:

            features = np.asarray(
                features
            )

            if features.ndim == 2:

                feature_dimension_detected = (
                    int(features.shape[-1])
                )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        prediction = decode_number(
            digit1_logits,
            digit2_logits,
        )

        prediction[
            "model_type"
        ] = (
            "onnx"
            if onnx_session is not None
            else "pytorch_checkpoint"
        )

        prediction[
            "input_size"
        ] = [
            INPUT_SIZE,
            INPUT_SIZE,
        ]

        prediction[
            "feature_dimension"
        ] = feature_dimension_detected

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        prediction[
            "accepted"
        ] = (
            prediction["confidence"]
            >= args.threshold
        )

        # ----------------------------------------------------
        # Save annotation
        # ----------------------------------------------------

        if not args.no_annotation:

            annotated = draw_prediction(
                original_image,
                prediction,
            )

            output_image_path = (
                annotated_dir
                / image_path.name
            )

            success = cv2.imwrite(
                str(output_image_path),
                annotated,
            )

            if not success:

                print(
                    "  WARNING: Failed to save "
                    "annotated image."
                )

            else:

                print(
                    f"  Annotated: "
                    f"{output_image_path}"
                )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        json_prediction = dict(
            prediction
        )

        # Probability arrays can make JSON
        # unnecessarily large. Keep them because
        # they can be useful for debugging.
        predictions[
            str(image_path)
        ] = json_prediction

        print(
            f"  Prediction : "
            f"{prediction['number']}"
        )

        print(
            f"  Confidence : "
            f"{prediction['confidence']:.4f}"
        )

    # ========================================================
    # Save JSON
    # ========================================================

    json_path = (
        output_dir
        / "predictions.json"
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
        )

    print(
        "\n========================================"
    )

    print(
        "Inference completed."
    )

    print(
        f"JSON       : {json_path}"
    )

    if not args.no_annotation:

        print(
            f"Annotated  : {annotated_dir}"
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
