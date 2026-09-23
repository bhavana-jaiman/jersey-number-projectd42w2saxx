import argparse
import os
import cv2
import numpy as np
import glob
import json
from flops import flops
from thop import profile, clever_format
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns

from subModules.backbone.model_drop import MultiTaskLearner
from utils.util import count_parameters

import torch
import torchvision.transforms as Transforms
import torch.nn.functional as F

import onnxruntime


def evaluate_model_parameters(model, input_size=(3, 96, 96)):
    flops_value, params = flops(
        model=model,
        in_size=input_size,
    )
    input_tensor = torch.randn(input_size).unsqueeze(0).cuda()
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    print(
        "model FLOPs:%s  Params:%s  MACs:%s"
        % (flops_value, params, macs / 1e9)
    )


def plot_heatmap(
    predicted_jerseyNumber_1: np.array,
    predicted_jerseyNumber_2: np.array,
    target_1: np.array,
    target_2: np.array,
):
    confusion_matrix_1 = confusion_matrix(
        target_1, predicted_jerseyNumber_1
    )
    confusion_matrix_2 = confusion_matrix(
        target_2, predicted_jerseyNumber_2
    )

    f, axes = plt.subplots(1, 2)

    axes[0].set_title("The First digit of the Jersey Number")
    sns.heatmap(
        confusion_matrix_1,
        cmap="Blues",
        annot=True,
        fmt="d",
        square=True,
        ax=axes[0],
    )

    axes[1].set_title("The Second digit of the Jersey Number")
    sns.heatmap(
        confusion_matrix_2,
        cmap="Blues",
        annot=True,
        fmt="d",
        square=True,
        ax=axes[1],
    )

    plt.show()


def plot_graph(count_jerseyNumber, corrected_jerseyNumber):
    x = np.array([i for i in range(0, 10)])
    test = np.array(
        [
            count_jerseyNumber[i] - corrected_jerseyNumber[i]
            for i in range(0, 10)
        ]
    )

    fig, ax = plt.subplots()

    stock_1 = ax.bar(
        x,
        corrected_jerseyNumber,
        label="Corrected JerseyNumber",
        tick_label=x,
    )
    stock_2 = ax.bar(
        x,
        test,
        bottom=corrected_jerseyNumber,
        label="Total",
        tick_label=x,
    )

    ax.bar_label(stock_1)
    ax.bar_label(stock_2)
    ax.set_xlabel("Jersey Number")
    ax.set_ylabel("Number of Jersey Numbers")
    ax.legend()
    plt.show()


# ---------------------------------------------------------------------------
# ADDED METRICS
# ---------------------------------------------------------------------------

def box_iou(box_a, box_b):
    """
    IoU for boxes in [x, y, width, height] format.

    Returns:
        float in [0, 1].
    """
    if box_a is None or box_b is None:
        return None

    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def update_binary_detection_counts(
    predicted_box,
    target_box,
    iou_threshold,
    counts,
):
    """
    Binary detection FP/FN/TP/TN based on IoU.

    A prediction is considered a TP when a predicted box exists,
    a target box exists, and IoU >= threshold.

    This is intentionally separate from digit-classification FP/FN.
    """
    target_exists = target_box is not None
    prediction_exists = predicted_box is not None

    if target_exists and prediction_exists:
        iou = box_iou(predicted_box, target_box)

        if iou >= iou_threshold:
            counts["tp"] += 1
        else:
            counts["fp"] += 1
            counts["fn"] += 1

        return iou

    if prediction_exists and not target_exists:
        counts["fp"] += 1
        return None

    if target_exists and not prediction_exists:
        counts["fn"] += 1
        return None

    counts["tn"] += 1
    return None


def classification_fp_fn(targets, predictions, num_classes=10):
    """
    One-vs-rest FP/FN for each digit class.

    Returns:
        dict[class_id] = {"fp": ..., "fn": ...}
    """
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    result = {}
    for cls in range(num_classes):
        target_positive = targets == cls
        prediction_positive = predictions == cls

        fp = np.logical_and(
            prediction_positive,
            np.logical_not(target_positive),
        ).sum()

        fn = np.logical_and(
            np.logical_not(prediction_positive),
            target_positive,
        ).sum()

        result[cls] = {
            "fp": int(fp),
            "fn": int(fn),
        }

    return result


def print_fp_fn_metrics(
    target_1,
    target_2,
    predicted_1,
    predicted_2,
    iou_values,
    detection_counts,
    iou_threshold,
):
    digit_1 = classification_fp_fn(target_1, predicted_1)
    digit_2 = classification_fp_fn(target_2, predicted_2)

    print("\n--- Digit classification FP/FN ---")
    print("Digit | First FP | First FN | Second FP | Second FN")
    for digit in range(10):
        print(
            f"{digit:5d} | "
            f"{digit_1[digit]['fp']:8d} | "
            f"{digit_1[digit]['fn']:8d} | "
            f"{digit_2[digit]['fp']:9d} | "
            f"{digit_2[digit]['fn']:9d}"
        )

    print("\n--- Detection FP/FN/IoU ---")
    print(
        f"TP: {detection_counts['tp']}  "
        f"FP: {detection_counts['fp']}  "
        f"FN: {detection_counts['fn']}  "
        f"TN: {detection_counts['tn']}"
    )

    valid_iou = [v for v in iou_values if v is not None]
    if valid_iou:
        print(f"IoU threshold: {iou_threshold:.3f}")
        print(f"Mean IoU: {np.mean(valid_iou):.4f}")
        print(f"Min IoU:  {np.min(valid_iou):.4f}")
        print(f"Max IoU:  {np.max(valid_iou):.4f}")
        print(
            f"IoU >= threshold: "
            f"{sum(v >= iou_threshold for v in valid_iou)}/{len(valid_iou)}"
        )
    else:
        print(
            "IoU: not calculated. "
            "The current model code does not return a predicted bounding box."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to checkpoint",
    )
    parser.add_argument(
        "--test_image_path",
        type=str,
        default="./datasets/JerseyNumber_Validation_Dataset",
        help="path to test image",
    )
    parser.add_argument(
        "--numberOfDigits",
        type=int,
        default=2,
        help="path to checkpoint",
    )
    parser.add_argument(
        "--numberLocation",
        action="store_true",
        help="able to locate number",
    )
    parser.add_argument(
        "--plot_CAM",
        action="store_true",
        default=False,
        help="plot CAM",
    )
    parser.add_argument(
        "--plot_graph",
        action="store_true",
        default=False,
        help="plot correct prediction",
    )
    parser.add_argument(
        "--plot_heatmap",
        action="store_true",
        default=False,
        help="plot heatmap",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.50,
        help="IoU threshold used for detection TP/FP/FN",
    )

    opt = parser.parse_args()

    model = MultiTaskLearner()

    assert os.path.exists(opt.checkpoint)

    if opt.checkpoint.endswith(".pth"):
        print("---")
        print("Testing pytorch model")
        print("---\n")

        pretrained_dict = torch.load(opt.checkpoint)

        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]

        model.load_state_dict(pretrained_dict, strict=False)
        model.cuda()

    elif opt.checkpoint.endswith(".onnx"):
        print("---")
        print("Testing onnx model")
        print("---\n")

        session = onnxruntime.InferenceSession(
            opt.checkpoint,
            providers=["CUDAExecutionProvider"],
        )

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_name = os.path.splitext(
            os.path.basename(opt.checkpoint)
        )[0]

    else:
        raise ValueError("Checkpoint must end with .pth or .onnx")

    assert os.path.exists(
        opt.test_image_path
    ), "the file is not existed"

    image_path = glob.glob(
        opt.test_image_path + "/images" + "/*.jpg"
    )

    if opt.checkpoint.endswith(".pth"):
        evaluate_model_parameters(model)

    total_body = 0

    jerseyNumber_correct = 0
    located_jerseyNumber_correct = 0
    post_processed_correct = 0
    jerseyNumber_correct = 0
    digital_correct = 0
    located_number = 0

    count_jerseyNumber = [0 for i in range(0, 10)]
    corrected_jerseyNumber = [0 for i in range(0, 10)]

    target_jerseyNumber_1 = []
    target_jerseyNumber_2 = []
    predicted_jerseyNumber_1 = []
    predicted_jerseyNumber_2 = []

    predicted_PostProcess_jerseyNumber_1 = []
    predicted_PostProcess_jerseyNumber_2 = []

    target_PostProcess_jerseyNumber_1 = []
    target_PostProcess_jerseyNumber_2 = []

    # ADDED: detection metric storage.
    iou_values = []
    detection_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    if opt.checkpoint.endswith(".pth"):
        model.eval()

    with torch.no_grad():
        for img in tqdm(image_path, desc="Testing..."):
            assert os.path.exists(img)

            label_info = img.replace(
                "images", "labels"
            ).replace(".jpg", ".json")

            with open(label_info, "r", encoding="utf8") as f:
                data = json.load(f)

            body_cnt = len(data["data"]["jersey"])
            total_body += body_cnt

            org_img = cv2.imread(img)
            org_img = cv2.cvtColor(
                org_img, cv2.COLOR_BGR2RGB
            )

            # Process every labeled jersey in the image.
            for i in data["data"]["jersey"]:
                body_rect = i["body"]["rect"]
                jersey_number = i["jersey_number"]
                digit_bboxes = [
                    i["numbers"][j]["rect"]
                    for j in range(len(jersey_number))
                ]

                if len(jersey_number) > opt.numberOfDigits:
                    continue

                count_jerseyNumber[jersey_number[0]] += 1

                if len(jersey_number) >= 2:
                    count_jerseyNumber[jersey_number[1]] += 1

                digital_label = (
                    jersey_number[0] * 10 + jersey_number[1]
                    if len(jersey_number) >= 2
                    else jersey_number[0]
                )

                # Pad missing digits with 10, as in the original script.
                jersey_number = list(jersey_number)
                if len(jersey_number) < opt.numberOfDigits:
                    for _ in range(
                        opt.numberOfDigits - len(jersey_number)
                    ):
                        jersey_number.append(10)

                minx = int(body_rect[0])
                miny = int(body_rect[1])
                maxx = int(body_rect[0] + body_rect[2])
                maxy = int(body_rect[1] + body_rect[3])

                body_img = org_img[miny:maxy, minx:maxx]

                if body_img.size == 0:
                    continue

                body_img = body_img.astype(np.float32) / 255.0

                resized_image = cv2.resize(
                    body_img,
                    (114, 114),
                    interpolation=cv2.INTER_LINEAR,
                )

                top = (114 - 96) // 2
                left = (114 - 96) // 2
                body_img = resized_image[
                    top:top + 96,
                    left:left + 96,
                ]

                # HWC -> CHW
                body_img = body_img.transpose(2, 0, 1)

                ratio_w = 114 / body_rect[2]
                ratio_h = 114 / body_rect[3]

                digit_bboxes_resized = np.asarray(
                    digit_bboxes,
                    dtype=np.float32,
                ).copy()

                if len(digit_bboxes_resized) > 0:
                    digit_bboxes_resized[:, 2] *= ratio_w
                    digit_bboxes_resized[:, 3] *= ratio_h
                    digit_bboxes_resized[:, 0] *= ratio_w - left
                    digit_bboxes_resized[:, 1] *= ratio_h - top

                predicted_box = None

                if opt.checkpoint.endswith(".pth"):
                    body_tensor = torch.from_numpy(
                        body_img
                    ).unsqueeze(0).cuda()

                    body_tensor = Transforms.ToTensor()(
                        body_img.transpose(1, 2, 0)
                    ).unsqueeze(0).cuda()

                    # The original model returns digit predictions.
                    outputs = model(body_tensor)

                    # Support either a tuple/list or a dictionary output.
                    if isinstance(outputs, dict):
                        digit1 = outputs.get("digit1")
                        digit2 = outputs.get("digit2")
                        predicted_box = outputs.get(
                            "number_box",
                            outputs.get("bbox"),
                        )
                    else:
                        try:
                            digit1, digit2 = outputs[0], outputs[1]
                        except (TypeError, IndexError):
                            digit1, digit2 = outputs

                    digit1_prediction = torch.max(
                        digit1, dim=1
                    )[1]
                    digit2_prediction = torch.max(
                        digit2, dim=1
                    )[1]

                    jerseyNumber_correct += int(
                        (
                            digit1_prediction
                            == jersey_number[0]
                        ).item()
                        and (
                            digit2_prediction
                            == jersey_number[1]
                        ).item()
                    )

                    if digit1_prediction.item() == jersey_number[0]:
                        corrected_jerseyNumber[
                            jersey_number[0]
                        ] += 1

                    if (
                        len(jersey_number) >= 2
                        and digit2_prediction.item()
                        == jersey_number[1]
                        and jersey_number[1] != 10
                    ):
                        corrected_jerseyNumber[
                            jersey_number[1]
                        ] += 1

                    predicted_jerseyNumber_1.append(
                        digit1_prediction.item()
                    )
                    predicted_jerseyNumber_2.append(
                        digit2_prediction.item()
                    )

                    target_jerseyNumber_1.append(
                        jersey_number[0]
                    )
                    target_jerseyNumber_2.append(
                        jersey_number[1]
                    )

                elif opt.checkpoint.endswith(".onnx"):
                    # ONNX model expects NCHW.
                    body_input = body_img[np.newaxis, ...].astype(
                        np.float32
                    )

                    outputs = session.run(
                        None,
                        {input_name: body_input},
                    )

                    if len(outputs) < 2:
                        raise RuntimeError(
                            "ONNX model must return at least two "
                            "outputs for digit-1 and digit-2."
                        )

                    digit1, digit2 = outputs[0], outputs[1]

                    digit1_prediction = np.argmax(
                        digit1, axis=1
                    )
                    digit2_prediction = np.argmax(
                        digit2, axis=1
                    )

                    p1 = int(
                        np.asarray(digit1_prediction).reshape(-1)[0]
                    )
                    p2 = int(
                        np.asarray(digit2_prediction).reshape(-1)[0]
                    )

                    jerseyNumber_correct += int(
                        p1 == jersey_number[0]
                        and p2 == jersey_number[1]
                    )

                    if p1 == jersey_number[0]:
                        corrected_jerseyNumber[
                            jersey_number[0]
                        ] += 1

                    if (
                        len(jersey_number) >= 2
                        and p2 == jersey_number[1]
                        and jersey_number[1] != 10
                    ):
                        corrected_jerseyNumber[
                            jersey_number[1]
                        ] += 1

                    predicted_jerseyNumber_1.append(p1)
                    predicted_jerseyNumber_2.append(p2)

                    target_jerseyNumber_1.append(
                        jersey_number[0]
                    )
                    target_jerseyNumber_2.append(
                        jersey_number[1]
                    )

                    # If your ONNX model also returns a predicted box,
                    # assign it to predicted_box here.
                    # Example:
                    # predicted_box = outputs[2][0]

                # -------------------------------------------------------
                # ADDED: IoU / detection FP / FN
                # -------------------------------------------------------
                #
                # IMPORTANT:
                # The screenshots show body_rect as ground truth, but
                # the current model output only contains digit classes.
                # Therefore there is NO predicted bounding box to compare
                # against body_rect. IoU is only calculated when the model
                # actually returns a bbox/number_box.
                #
                if predicted_box is not None:
                    predicted_box = np.asarray(
                        predicted_box
                    ).reshape(-1)

                    if predicted_box.size >= 4:
                        predicted_box = predicted_box[:4]
                        iou = update_binary_detection_counts(
                            predicted_box,
                            body_rect,
                            opt.iou_threshold,
                            detection_counts,
                        )
                        iou_values.append(iou)
                    else:
                        predicted_box = None

                # Keep original behavior: if there is no predicted box,
                # do not count an IoU FP/FN from the classifier itself.

    if opt.plot_heatmap:
        plot_heatmap(
            predicted_jerseyNumber_1,
            predicted_jerseyNumber_2,
            target_jerseyNumber_1,
            target_jerseyNumber_2,
        )

    if opt.plot_graph:
        plot_graph(
            count_jerseyNumber,
            corrected_jerseyNumber,
        )

    print("---")
    if total_body > 0:
        print(
            "the number of the correctly predicted jersey is %d, "
            "the number of jersey is %d"
            % (
                jerseyNumber_correct,
                total_body,
            )
        )

        print(
            "the accuracy is %2f %%"
            % (
                (jerseyNumber_correct / total_body) * 100
            )
        )

    # ADDED: FP/FN report.
    if predicted_jerseyNumber_1 and predicted_jerseyNumber_2:
        print_fp_fn_metrics(
            target_jerseyNumber_1,
            target_jerseyNumber_2,
            predicted_jerseyNumber_1,
            predicted_jerseyNumber_2,
            iou_values,
            detection_counts,
            opt.iou_threshold,
        )

    print("\nNote:")
    print(
        "Digit FP/FN are classification metrics. "
        "IoU/box FP/FN require the model to output a predicted "
        "bounding box. The current screenshot code uses body_rect "
        "only as the ground-truth crop, so IoU cannot be meaningfully "
        "computed until a predicted box is available."
    )
