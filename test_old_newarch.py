import argparse

import os
import cv2
import numpy as np
import glob
import json
from flopth import flopth
from thop import profile, clever_format
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn

# ---------------HCL_Changes---------------
# new backbone file; MultiTaskLearner = model WITHOUT state
# (digit1 = 10, digit2 = 11 outputs; whole-number head removed)
# from subModules.backbone_ying import MultiTaskLearner
from subModules.backbone_New import MultiTaskLearner
# -------------------------------------------
from utils.util import count_parameters

import torch
import torchvision.transforms as Transforms
import torch.nn.functional as F

import onnxruntime


def Evaluate_model_parameters(model, input_size=(3, 96, 96)):
    flops, params = flopth(
        model=model,
        in_size=input_size,
    )
    input = torch.randn(input_size).unsqueeze_(0).cuda()
    macs, _ = profile(model, inputs=(input,))
    print("model FLOPs:%s   Params:%s Macs:%s\n" % (flops, params, macs / 1e9))


def plot_Heatmap(
    predicted_jerseyNumber_1: np.array,
    predicted_jerseyNumber_2: np.array,
    target_1: np.array,
    target_2: np.array,
):
    confusion_matrix_1 = confusion_matrix(target_1, predicted_jerseyNumber_1)
    confusion_matrix_2 = confusion_matrix(target_2, predicted_jerseyNumber_2)
    f, axes = plt.subplots(1, 2)
    axes[0].set_title("The First digit of the jersey Number")
    sn.heatmap(
        confusion_matrix_1, cmap="Blues", annot=True, fmt="d", square=True, ax=axes[0]
    )

    axes[1].set_title("The Second digit of the jersey Number")
    sn.heatmap(
        confusion_matrix_2, cmap="Blues", annot=True, fmt="d", square=True, ax=axes[1]
    )
    plt.show()


def plot_Graph(count_jerseyNumber, corrected_jerseyNumber):
    x = np.array([i for i in range(0, 10)])
    test = [count_jerseyNumber[i] - corrected_jerseyNumber[i] for i in range(0, 10)]
    fig, ax = plt.subplots()
    storck_1 = ax.bar(
        x, corrected_jerseyNumber, label="Corrected JerseyNumber", tick_label=x
    )
    storck_2 = ax.bar(
        x, test, bottom=corrected_jerseyNumber, label="Total", tick_label=x
    )
    ax.bar_label(storck_1)
    ax.bar_label(storck_2)
    ax.set_xlabel("Jersey Number")
    ax.set_ylabel("Number of Jersey Numbers")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument(
        "--test_image_path",
        type=str,
        default="./datasets/JerseyNumber_Validation_Dataset",
        help="path to test image",
    )
    parser.add_argument(
        "--numberOfDigits", type=int, default=2, help="path to checkpoint"
    )
    parser.add_argument(
        "--numberLocation", action="store_true", help="able to locate number"
    )
    parser.add_argument(
        "--plot_CAM", action="store_true", default=False, help="plot CAM"
    )
    parser.add_argument(
        "--plot_graph",
        action="store_true",
        default=False,
        help="plot correct prediction",
    )
    parser.add_argument(
        "--plot_heatmap", action="store_true", default=False, help="plot heatmap"
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
            opt.checkpoint, providers=["CUDAExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        model_name = os.path.splitext(os.path.basename(opt.checkpoint))[0]

    assert os.path.exists(opt.test_image_path), "the file is not exited"
    image_path = glob.glob(opt.test_image_path + "/images" + "/*.jpg")

    Evaluate_model_parameters(model)

    total_body = 0
    jerseyNumber_correct, located_jerseyNumber_correct, post_processed_correct = 0, 0, 0
    jerseyLen_correct, digital_correct = 0, 0
    located_number = 0
    count_jerseyNumber, corrected_jerseyNumber = [0 for i in range(0, 10)], [
        0 for i in range(0, 10)
    ]

    target_jerseyNumber_1, target_jerseyNumber_2, predicted_jerseyNumber_1, predicted_jerseyNumber_2 = [], [], [], []
    predicted_PostProcess_jerseyNumber_1, predicted_PostProcess_jerseyNumber_2 = [], []

    model.eval()

    with torch.no_grad():
        for img in tqdm(image_path, desc=f"Testing..."):
            assert os.path.exists(img)
            label_info = img.replace("images", "labels").replace(".jpg", ".json")

            with open(label_info, "r", encoding="utf8") as f:
                data = json.load(f)

            body_cnt = len(data["data"]["jersey"])
            total_body += body_cnt

            org_img = cv2.imread(img)
            # cv2.imshow('body', org_img)
            # cv2.waitKey(0)

            for i in data["data"]["jersey"]:
                body_rect = i["body"]["rect"]
                jersey_number = i["jersey_number"]

                if len(jersey_number) > opt.numberOfDigits:
                    total_body -= 1
                    continue

                count_jerseyNumber[jersey_number[0]] += 1
                if len(jersey_number) >= 2:
                    count_jerseyNumber[jersey_number[1]] += 1

                digital_label = (
                    jersey_number[0] * 10 + jersey_number[1]
                    if len(jersey_number) >= 2
                    else jersey_number[0]
                )

                if len(jersey_number) < opt.numberOfDigits:
                    for _ in range(opt.numberOfDigits - len(jersey_number)):
                        jersey_number.append(10)

                minx = int(body_rect[0])
                miny = int(body_rect[1])
                maxx = int(body_rect[0] + body_rect[2])
                maxy = int(body_rect[1] + body_rect[3])

                body_img = org_img[miny:maxy, minx:maxx]
                body_img = cv2.resize(body_img, (96, 96))

                # cv2.imshow('body', body_img)
                # cv2.waitKey(0)
                target_jerseyNumber_1.append(jersey_number[0])
                target_jerseyNumber_2.append(jersey_number[1])
                # print(jersey_number)
                # print(img)
                # cv2.imshow('body', number_part)
                # cv2.waitKey(0)

                if opt.checkpoint.endswith(".pth"):
                    body_img = Transforms.ToTensor()(body_img).unsqueeze(0).cuda()
                    # ---------------HCL_Changes---------------
                    # new output order: feat, digit1, digit2 (digital removed)
                    # feat, digital, digit1, digit2 = model(
                    #     body_img
                    # )  # , digit3, digit_len
                    feat, digit1, digit2 = model(body_img)
                    # -------------------------------------------
                    # digital_prediction = torch.max(digital, dim=1)[1]
                    digit1_prediction = torch.max(digit1, dim=1)[1]
                    digit2_prediction = torch.max(digit2, dim=1)[1]

                    jerseyNumber_correct += digit1_prediction.eq(
                        jersey_number[0]
                    ).logical_and(digit2_prediction.eq(jersey_number[1]))

                    if digit1_prediction == jersey_number[0]:
                        corrected_jerseyNumber[jersey_number[0]] += 1
                    if digit2_prediction == jersey_number[1] and jersey_number[1] != 10:
                        corrected_jerseyNumber[jersey_number[1]] += 1

                elif opt.checkpoint.endswith(".onnx"):
                    body_img = body_img.astype(np.float32)
                    body_img /= 255
                    body_img = body_img.transpose(2, 0, 1)  # h,w,c -> c, h, w
                    body_img = body_img[np.newaxis, :, :, :]
                    # ---------------HCL_Changes---------------
                    # new output order: feat, digit1, digit2 (+ state if exported
                    # from MultiTaskLearnerWithState) -> take outputs 1 and 2
                    # _, _, digit1, digit2 = session.run(
                    #     None, input_feed={input_name: body_img}
                    # )
                    outputs = session.run(
                        None, input_feed={input_name: body_img}
                    )
                    digit1, digit2 = outputs[1], outputs[2]
                    # -------------------------------------------
                    digit1_prediction = np.argmax(digit1)
                    digit2_prediction = np.argmax(digit2)

                    jerseyNumber_correct += np.logical_and(
                        np.equal(digit1_prediction, jersey_number[0]),
                        np.equal(digit2_prediction, jersey_number[1]),
                    )

                predicted_jerseyNumber_1.append(digit1_prediction.item())
                predicted_jerseyNumber_2.append(digit2_prediction.item())

    if opt.plot_heatmap:
        plot_Heatmap(
            predicted_jerseyNumber_1,
            predicted_jerseyNumber_2,
            target_jerseyNumber_1,
            target_jerseyNumber_2,
        )
        plot_Heatmap(
            predicted_PostProcess_jerseyNumber_1,
            predicted_PostProcess_jerseyNumber_2,
            target_jerseyNumber_1,
            target_jerseyNumber_2,
        )
    if opt.plot_graph:
        plot_Graph(count_jerseyNumber, corrected_jerseyNumber)

    print("----")
    print(
        "the number of the correctly predicted jersey is %d, the number of jersey is %d"
        % (jerseyNumber_correct.item(), total_body)
    )
    print("the accuracy is %2f" % ((jerseyNumber_correct.item() / total_body) * 100))
