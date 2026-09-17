#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# Constants
# ============================================================

INPUT_WIDTH = 96
INPUT_HEIGHT = 96

# ImageNet normalization.
# Keep these identical to the preprocessing used during
# training/testing of your jersey model.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Your digit heads have 11 classes:
# 0-9 = digits
# 10  = blank
BLANK_CLASS = 10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Argument Parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Jersey Number ONNX Runtime Inference"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or directory containing images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/inference_onnx",
        help="Directory where inference results are saved"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Inference device"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum confidence required for a prediction"
    )

    parser.add_argument(
        "--keep_ratio",
        action="store_true",
        help="Keep aspect ratio while resizing and pad to 96x96"
    )

    return parser.parse_args()


# ============================================================
# ONNX Runtime Session
# ============================================================

def create_session(model_path, device):

    print("\n========================================")
    print("ONNX Runtime Inference")
    print("========================================")

    available_providers = ort.get_available_providers()

    print("Available providers:")
    for provider in available_providers:
        print("  -", provider)

    if device == "cuda":

        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "\nCUDAExecutionProvider is not available.\n"
                "Install the CUDA-enabled onnxruntime package "
                "or use --device cpu."
            )

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    else:
        providers = [
            "CPUExecutionProvider",
        ]

    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    print("\nModel:", model_path)
    print("Execution providers:")
    print(session.get_providers())

    # --------------------------------------------------------
    # Input information
    # --------------------------------------------------------

    inputs = session.get_inputs()

    print("\nModel inputs:")

    for inp in inputs:
        print(
            f"  Name : {inp.name}\n"
            f"  Shape: {inp.shape}\n"
            f"  Type : {inp.type}"
        )

    # --------------------------------------------------------
    # Output information
    # --------------------------------------------------------

    outputs = session.get_outputs()

    print("\nModel outputs:")

    for out in outputs:
        print(
            f"  Name : {out.name}\n"
            f"  Shape: {out.shape}\n"
            f"  Type : {out.type}"
        )

    print("========================================\n")

    return session


# ============================================================
# Image Preprocessing
# ============================================================

def resize_keep_ratio(image):

    """
    Resize image while keeping aspect ratio and pad to 96x96.
    """

    h, w = image.shape[:2]

    scale = min(
        INPUT_WIDTH / w,
        INPUT_HEIGHT / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (INPUT_HEIGHT, INPUT_WIDTH, 3),
        dtype=np.uint8
    )

    x_offset = (INPUT_WIDTH - new_w) // 2
    y_offset = (INPUT_HEIGHT - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


def preprocess(image, keep_ratio=False):

    """
    Preprocess image exactly for the classifier input.

    Input:
        BGR OpenCV image

    Output:
        NCHW float32 tensor
        Shape = (1, 3, 96, 96)
    """

    if image is None:
        raise ValueError("Input image is None")

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    if keep_ratio:
        image = resize_keep_ratio(image)

    else:
        image = cv2.resize(
            image,
            (INPUT_WIDTH, INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # uint8 -> float32
    # --------------------------------------------------------

    image = image.astype(np.float32) / 255.0

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = (image - MEAN) / STD

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(np.float32)


# ============================================================
# Output Utilities
# ============================================================

def softmax(logits):

    logits = np.asarray(logits)

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=-1,
        keepdims=True
    )


def get_prediction(logits):

    """
    Convert classifier logits into:

        predicted class
        confidence
    """

    logits = np.asarray(logits)

    # Remove batch dimension if present
    if logits.ndim > 1:
        logits = logits[0]

    probabilities = softmax(logits)

    class_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_id]
    )

    return class_id, confidence


# ============================================================
# Output Name Detection
# ============================================================

def find_output_by_name(outputs, keywords):

    """
    Find an ONNX output using its name.

    Example:
        digit1_logits
        digit_1
        digit1
    """

    for name, value in outputs.items():

        name_lower = name.lower()

        for keyword in keywords:

            if keyword.lower() in name_lower:
                return value

    return None


# ============================================================
# Model Output Parsing
# ============================================================

def parse_model_outputs(session, raw_outputs):

    """
    Parse the outputs of the jersey classifier.

    The function supports named ONNX outputs such as:

        digit1
        digit_1
        digit1_logits

        digit2
        digit_2
        digit2_logits

    A state output is also accepted if your WithState model
    exports one.

    The state output is not required to construct the jersey
    number.
    """

    output_info = session.get_outputs()

    outputs = {}

    for info, value in zip(
        output_info,
        raw_outputs
    ):
        outputs[info.name] = value

    # --------------------------------------------------------
    # Digit 1
    # --------------------------------------------------------

    digit1_logits = find_output_by_name(
        outputs,
        [
            "digit1",
            "digit_1",
            "digit1_logits",
            "digit_1_logits",
            "first_digit",
        ]
    )

    # --------------------------------------------------------
    # Digit 2
    # --------------------------------------------------------

    digit2_logits = find_output_by_name(
        outputs,
        [
            "digit2",
            "digit_2",
            "digit2_logits",
            "digit_2_logits",
            "second_digit",
        ]
    )

    # --------------------------------------------------------
    # If names are not descriptive, use output order.
    #
    # For the jersey classifier the first two classification
    # outputs are treated as digit-1 and digit-2.
    # --------------------------------------------------------

    if digit1_logits is None or digit2_logits is None:

        if len(raw_outputs) >= 2:

            digit1_logits = raw_outputs[0]
            digit2_logits = raw_outputs[1]

        else:

            raise RuntimeError(
                "Could not identify digit-1 and digit-2 "
                "outputs from the ONNX model."
            )

    return digit1_logits, digit2_logits, outputs


# ============================================================
# Jersey Number Construction
# ============================================================

def construct_jersey_number(
    digit1,
    digit2
):

    """
    Construct the final jersey number.

    Class 10 is treated as BLANK.

    Examples:

        2 + 3  -> "23"
        5 + blank -> "5"
        blank + 7 -> "7"
        blank + blank -> "UNKNOWN"
    """

    first_blank = digit1 == BLANK_CLASS
    second_blank = digit2 == BLANK_CLASS

    if first_blank and second_blank:
        return "UNKNOWN"

    if first_blank:
        return str(digit2)

    if second_blank:
        return str(digit1)

    return f"{digit1}{digit2}"


# ============================================================
# Single Image Inference
# ============================================================

def inference_single_image(
    session,
    image_path,
    keep_ratio=False,
    threshold=0.0
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image: {image_path}"
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = preprocess(
        image,
        keep_ratio=keep_ratio
    )

    # --------------------------------------------------------
    # Get ONNX input name
    # --------------------------------------------------------

    input_name = session.get_inputs()[0].name

    # --------------------------------------------------------
    # ONNX Runtime inference
    # --------------------------------------------------------

    raw_outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    # --------------------------------------------------------
    # Parse model outputs
    # --------------------------------------------------------

    digit1_logits, digit2_logits, all_outputs = (
        parse_model_outputs(
            session,
            raw_outputs
        )
    )

    # --------------------------------------------------------
    # Digit predictions
    # --------------------------------------------------------

    digit1, confidence1 = get_prediction(
        digit1_logits
    )

    digit2, confidence2 = get_prediction(
        digit2_logits
    )

    # --------------------------------------------------------
    # Final jersey number
    # --------------------------------------------------------

    jersey_number = construct_jersey_number(
        digit1,
        digit2
    )

    # Overall confidence
    confidence = (
        confidence1 + confidence2
    ) / 2.0

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if confidence < threshold:

        final_number = "UNKNOWN"

    else:

        final_number = jersey_number

    result = {
        "image": image_path.name,

        "digit1": digit1,
        "digit1_confidence": confidence1,

        "digit2": digit2,
        "digit2_confidence": confidence2,

        "jersey_number": final_number,
        "confidence": confidence,

        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
    }

    return image, result


# ============================================================
# Annotation
# ============================================================

def annotate_image(
    image,
    result
):

    output = image.copy()

    jersey_number = result["jersey_number"]
    confidence = result["confidence"]

    digit1 = result["digit1"]
    digit2 = result["digit2"]

    confidence1 = result["digit1_confidence"]
    confidence2 = result["digit2_confidence"]

    # --------------------------------------------------------
    # Main label
    # --------------------------------------------------------

    text = (
        f"Jersey: {jersey_number} "
        f"({confidence * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Digit information
    # --------------------------------------------------------

    digit_text = (
        f"D1: {digit1} ({confidence1 * 100:.1f}%)  "
        f"D2: {digit2} ({confidence2 * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # Text position
    # --------------------------------------------------------

    x = 10
    y = 30

    # --------------------------------------------------------
    # Background rectangle
    # --------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.7
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    (digit_w, digit_h), _ = cv2.getTextSize(
        digit_text,
        font,
        0.45,
        1
    )

    box_w = max(
        text_w,
        digit_w
    ) + 20

    box_h = text_h + digit_h + 35

    cv2.rectangle(
        output,
        (5, 5),
        (5 + box_w, 5 + box_h),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Main jersey number
    # --------------------------------------------------------

    cv2.putText(
        output,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Digit information
    # --------------------------------------------------------

    cv2.putText(
        output,
        digit_text,
        (x, y + 25),
        font,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return output


# ============================================================
# Input Image Discovery
# ============================================================

def get_input_images(input_path):

    input_path = Path(input_path)

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    if input_path.is_file():

        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:

            raise ValueError(
                f"Unsupported image format: "
                f"{input_path.suffix}"
            )

        return [input_path]

    # --------------------------------------------------------
    # Directory
    # --------------------------------------------------------

    if input_path.is_dir():

        images = sorted(
            [
                p
                for p in input_path.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        if len(images) == 0:

            raise RuntimeError(
                f"No supported images found in: "
                f"{input_path}"
            )

        return images

    raise FileNotFoundError(
        f"Input path does not exist: {input_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    model_path = Path(
        args.model
    )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    annotated_dir = (
        output_dir / "annotated"
    )

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"ONNX model not found: "
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    if output_dir.exists():

        print(
            f"Removing existing output directory: "
            f"{output_dir}"
        )

        shutil.rmtree(
            output_dir
        )

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create ONNX session
    # --------------------------------------------------------

    session = create_session(
        str(model_path),
        args.device
    )

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    image_paths = get_input_images(
        input_path
    )

    print(
        f"\nFound {len(image_paths)} image(s)"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = {}

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            image, result = inference_single_image(
                session=session,
                image_path=image_path,
                keep_ratio=args.keep_ratio,
                threshold=args.threshold
            )

            # ------------------------------------------------
            # Annotate
            # ------------------------------------------------

            annotated = annotate_image(
                image,
                result
            )

            # ------------------------------------------------
            # Save annotated image
            # ------------------------------------------------

            output_image_path = (
                annotated_dir /
                image_path.name
            )

            cv2.imwrite(
                str(output_image_path),
                annotated
            )

            # ------------------------------------------------
            # Store JSON result
            # ------------------------------------------------

            predictions[
                image_path.name
            ] = result

            successful += 1

            print(
                f"    Jersey Number : "
                f"{result['jersey_number']}"
            )

            print(
                f"    Digit 1       : "
                f"{result['digit1']} "
                f"({result['digit1_confidence']:.4f})"
            )

            print(
                f"    Digit 2       : "
                f"{result['digit2']} "
                f"({result['digit2_confidence']:.4f})"
            )

            print(
                f"    Confidence    : "
                f"{result['confidence']:.4f}"
            )

        except Exception as e:

            failed += 1

            print(
                f"    ERROR: {e}"
            )

            predictions[
                image_path.name
            ] = {
                "image": image_path.name,
                "error": str(e),
            }

    # --------------------------------------------------------
    # Save predictions.json
    # --------------------------------------------------------

    json_path = (
        output_dir /
        "predictions.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("Inference completed")
    print("========================================")

    print(
        f"Total images : {len(image_paths)}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"Annotated    : {annotated_dir}"
    )

    print(
        f"JSON         : {json_path}"
    )

    print("========================================")


if __name__ == "__main__":
    main()
