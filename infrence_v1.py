#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SIZE = 96

# 11 classes:
# 0-9  -> digits
# 10   -> blank
BLANK_CLASS = 10

# IMPORTANT:
# These must match your existing testing preprocessing.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Jersey Number ONNX Runtime Inference"
    )

    parser.add_argument(
        "--model",
        default="output/MultiTaskLerner.onnx",
        help="Path to MultiTaskLerner.onnx"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or directory"
    )

    parser.add_argument(
        "--output_dir",
        default="output/inference",
        help="Output directory"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Inference device"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum confidence"
    )

    return parser.parse_args()


# ============================================================
# ONNX SESSION
# ============================================================

def create_session(model_path, device):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found:\n{model_path}"
        )

    # Your model uses external ONNX data.
    # It must stay beside the .onnx file.
    data_path = Path(
        str(model_path) + ".data"
    )

    if data_path.exists():

        print(
            f"External data file found:\n"
            f"  {data_path}"
        )

    else:

        print(
            "WARNING: External data file not found:\n"
            f"  {data_path}"
        )

    available = ort.get_available_providers()

    print("\nAvailable ONNX Runtime providers:")

    for provider in available:
        print(f"  {provider}")

    if device == "cuda":

        if "CUDAExecutionProvider" not in available:

            raise RuntimeError(
                "\nCUDAExecutionProvider is not available.\n"
                f"Available providers: {available}\n\n"
                "Use --device cpu or install CUDA-enabled "
                "ONNX Runtime."
            )

        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    else:

        providers = [
            "CPUExecutionProvider"
        ]

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    print("\n========================================")
    print("       JERSEY NUMBER ONNX INFERENCE")
    print("========================================")

    print(f"Model     : {model_path}")
    print(f"Data file : {data_path}")
    print(
        f"Provider  : {session.get_providers()}"
    )

    print("\nONNX INPUT")

    for item in session.get_inputs():

        print(
            f"  Name : {item.name}"
        )

        print(
            f"  Shape: {item.shape}"
        )

        print(
            f"  Type : {item.type}"
        )

    print("\nONNX OUTPUTS")

    for item in session.get_outputs():

        print(
            f"  Name : {item.name}"
        )

        print(
            f"  Shape: {item.shape}"
        )

        print(
            f"  Type : {item.type}"
        )

    print("========================================\n")

    return session


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):

    """
    OpenCV BGR image
            ↓
    RGB
            ↓
    Resize 96 x 96
            ↓
    Normalize
            ↓
    CHW
            ↓
    NCHW

    Final shape:
        [1, 3, 96, 96]
    """

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # uint8 -> float32
    image = (
        image.astype(np.float32)
        / 255.0
    )

    # Normalize
    image = (
        image - MEAN
    ) / STD

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # CHW -> NCHW
    image = np.expand_dims(
        image,
        axis=0
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32
    )


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=np.float32
    )

    logits = np.squeeze(logits)

    logits = (
        logits
        - np.max(logits)
    )

    exp_logits = np.exp(
        logits
    )

    probabilities = (
        exp_logits
        / (
            np.sum(exp_logits)
            + 1e-12
        )
    )

    return probabilities


# ============================================================
# CLASS PREDICTION
# ============================================================

def classify(logits):

    probabilities = softmax(
        logits
    )

    class_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_id]
    )

    return (
        class_id,
        confidence,
        probabilities
    )


# ============================================================
# FIND ONNX OUTPUT
# ============================================================

def find_output(
    outputs,
    keywords
):

    for name, value in outputs.items():

        name_lower = name.lower()

        for keyword in keywords:

            if keyword in name_lower:

                return value

    return None


# ============================================================
# PARSE MODEL OUTPUTS
# ============================================================

def parse_outputs(
    session,
    raw_outputs
):

    """
    Expected jersey model outputs include digit-1
    and digit-2 classification heads.

    The function first looks at output names.

    If names are generic, it falls back to finding
    outputs having 11 classes.
    """

    output_info = (
        session.get_outputs()
    )

    named_outputs = {
        info.name: value
        for info, value in zip(
            output_info,
            raw_outputs
        )
    }

    # --------------------------------------------------------
    # Digit 1
    # --------------------------------------------------------

    digit1 = find_output(
        named_outputs,
        [
            "digit1_logits",
            "digit_1_logits",
            "digit1",
            "digit_1",
            "first_digit",
        ]
    )

    # --------------------------------------------------------
    # Digit 2
    # --------------------------------------------------------

    digit2 = find_output(
        named_outputs,
        [
            "digit2_logits",
            "digit_2_logits",
            "digit2",
            "digit_2",
            "second_digit",
        ]
    )

    # --------------------------------------------------------
    # Optional whole-number output
    # --------------------------------------------------------

    whole = find_output(
        named_outputs,
        [
            "whole_logits",
            "whole",
            "digital_logits",
            "digital",
        ]
    )

    # --------------------------------------------------------
    # Generic output fallback
    # --------------------------------------------------------

    candidates = []

    for value in raw_outputs:

        arr = np.squeeze(
            np.asarray(value)
        )

        if (
            arr.ndim == 1
            and arr.shape[0] == 11
        ):

            candidates.append(value)

    if digit1 is None and len(candidates) >= 1:

        digit1 = candidates[0]

    if digit2 is None and len(candidates) >= 2:

        digit2 = candidates[1]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if digit1 is None or digit2 is None:

        available = []

        for info in output_info:

            available.append(
                f"{info.name}: {info.shape}"
            )

        raise RuntimeError(
            "\nCould not identify Digit-1 and Digit-2 "
            "outputs.\n\n"
            "Available ONNX outputs:\n"
            + "\n".join(available)
        )

    return (
        digit1,
        digit2,
        whole
    )


# ============================================================
# JERSEY NUMBER
# ============================================================

def construct_jersey_number(
    digit1,
    digit2
):

    """
    0-9  = digit
    10   = blank
    """

    if (
        digit1 == BLANK_CLASS
        and digit2 == BLANK_CLASS
    ):

        return "UNKNOWN"

    if digit1 == BLANK_CLASS:

        return str(digit2)

    if digit2 == BLANK_CLASS:

        return str(digit1)

    return f"{digit1}{digit2}"


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def infer_image(
    session,
    image_path,
    threshold
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    input_tensor = preprocess(
        image
    )

    # --------------------------------------------------------
    # Input name
    # --------------------------------------------------------

    input_name = (
        session.get_inputs()[0].name
    )

    # --------------------------------------------------------
    # ONNX inference
    # --------------------------------------------------------

    raw_outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    # --------------------------------------------------------
    # Parse outputs
    # --------------------------------------------------------

    (
        digit1_logits,
        digit2_logits,
        whole_logits
    ) = parse_outputs(
        session,
        raw_outputs
    )

    # --------------------------------------------------------
    # Digit 1
    # --------------------------------------------------------

    digit1, confidence1, _ = classify(
        digit1_logits
    )

    # --------------------------------------------------------
    # Digit 2
    # --------------------------------------------------------

    digit2, confidence2, _ = classify(
        digit2_logits
    )

    # --------------------------------------------------------
    # Jersey number
    # --------------------------------------------------------

    number = construct_jersey_number(
        digit1,
        digit2
    )

    # Average digit confidence
    confidence = (
        confidence1
        + confidence2
    ) / 2.0

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if confidence < threshold:

        final_number = "UNKNOWN"

    else:

        final_number = number

    result = {

        "image": image_path.name,

        "digit1": digit1,

        "digit1_confidence": confidence1,

        "digit2": digit2,

        "digit2_confidence": confidence2,

        "jersey_number": final_number,

        "confidence": confidence,

        "input_size": [
            INPUT_SIZE,
            INPUT_SIZE
        ]
    }

    # --------------------------------------------------------
    # Optional whole-number head
    # --------------------------------------------------------

    if whole_logits is not None:

        whole_class, whole_confidence, _ = (
            classify(whole_logits)
        )

        result[
            "whole_class"
        ] = whole_class

        result[
            "whole_confidence"
        ] = whole_confidence

    return image, result


# ============================================================
# ANNOTATION
# ============================================================

def annotate_image(
    image,
    result
):

    output = image.copy()

    jersey = result[
        "jersey_number"
    ]

    confidence = result[
        "confidence"
    ]

    digit1 = result[
        "digit1"
    ]

    digit2 = result[
        "digit2"
    ]

    confidence1 = result[
        "digit1_confidence"
    ]

    confidence2 = result[
        "digit2_confidence"
    ]

    # Main prediction
    main_text = (
        f"Jersey: {jersey} "
        f"({confidence * 100:.2f}%)"
    )

    # Individual predictions
    detail_text = (
        f"D1: {digit1} "
        f"({confidence1 * 100:.1f}%)  "
        f"D2: {digit2} "
        f"({confidence2 * 100:.1f}%)"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main jersey number
    cv2.putText(
        output,
        main_text,
        (8, 25),
        font,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Digit confidence
    cv2.putText(
        output,
        detail_text,
        (8, 47),
        font,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return output


# ============================================================
# FIND INPUT IMAGES
# ============================================================

def get_images(
    input_path
):

    path = Path(
        input_path
    )

    # Single image
    if path.is_file():

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported image format: "
                f"{path.suffix}"
            )

        return [path]

    # Directory
    if path.is_dir():

        image_paths = sorted(
            p
            for p in path.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        if not image_paths:

            raise RuntimeError(
                f"No supported images found in: "
                f"{path}"
            )

        return image_paths

    raise FileNotFoundError(
        f"Input path does not exist: "
        f"{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    model_path = Path(
        args.model
    )

    output_dir = Path(
        args.output_dir
    )

    annotated_dir = (
        output_dir
        / "annotated"
    )

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    session = create_session(
        model_path,
        args.device
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = get_images(
        args.input
    )

    print(
        f"\nFound {len(image_paths)} image(s).\n"
    )

    predictions = {}

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Inference loop
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

            image, result = infer_image(
                session,
                image_path,
                args.threshold
            )

            # Annotate
            annotated = annotate_image(
                image,
                result
            )

            # Save annotated image
            output_image = (
                annotated_dir
                / image_path.name
            )

            cv2.imwrite(
                str(output_image),
                annotated
            )

            # Store result
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

        except Exception as exc:

            failed += 1

            print(
                f"    ERROR: {exc}"
            )

            predictions[
                image_path.name
            ] = {
                "image": image_path.name,
                "error": str(exc)
            }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        output_dir
        / "predictions.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            predictions,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "       INFERENCE COMPLETED"
    )

    print(
        "========================================"
    )

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

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
