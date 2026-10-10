import os
import cv2
import numpy as np
import glob
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from Analysis_plugin import plug_save_image
import torch
import torchvision.transforms as Transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# ---------------HCL_Changes---------------
# new backbone file (MultiTaskLearner = no state: returns x, digit_1 (10), digit_2 (11))
# from subModules.backbone_ying import MultiTaskLearnerWithState, MultiTaskLearner
from subModules.backbone_New import MultiTaskLearnerWithState, MultiTaskLearner
# -------------------------------------------


class ScaleRobustnessEvaluator:
    def __init__(self, name="Digit_Scale"):
        self.name = name
        self.scales = []
        self.corrects = []

    def update(self, logits, targets, bboxes, ignore_index=-100):
        if isinstance(targets, int):
            targets = torch.tensor([targets], device=logits.device)
        if isinstance(bboxes, list):
            bboxes = torch.tensor([bboxes], device=logits.device)
        valid_mask = targets != ignore_index

        if not valid_mask:
            return

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        valid_bboxes = bboxes[valid_mask]

        preds = torch.argmax(valid_logits, dim=1)
        correct = (preds == valid_targets).cpu().numpy().astype(int)

        widths = valid_bboxes[:, 2]
        heights = valid_bboxes[:, 3]

        areas = torch.clamp(widths * heights, min=1.0)
        scales = torch.sqrt(areas).cpu().numpy()

        self.scales.append(scales)
        self.corrects.append(correct)

    def plot_scale_analysis(self, num_bins=10, save_path=None):
        if len(self.scales) == 0:
            print('No data collected')
            return

        scales = np.array(self.scales)
        corrects = np.array(self.corrects)

        min_scale = np.floor(scales.min())
        max_scale = np.floor(scales.max())

        bins = np.linspace(min_scale, max_scale, num_bins + 1)

        bin_centers = []
        bin_accs = []
        bin_counts = []

        x_labels = []
        for i in range(num_bins):
            mask = (scales >= bins[i]) & (scales < bins[i + 1])

            if i == num_bins - 1:
                mask = (scales >= bins[i]) & (scales <= bins[i+1])

            count = mask.sum()
            bin_counts.append(count)

            bin_centers.append((bins[i] + bins[i+1]) / 2)

            x_labels.append(f"{int(bins[i])} - {int(bins[i+1])}")

            if count > 0:
                bin_accs.append(corrects[mask].mean())
            else:
                bin_accs.append(np.nan)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'Accuracy vs. Digit Scale (BBox Size) - {self.name}', fontsize=16, fontweight='bold')

        ax2 = ax1.twinx()
        width = (max_scale - min_scale) / num_bins * 0.8
        bars = ax2.bar(bin_centers, bin_counts, width=width, color='gray', alpha=0.3, label='Sample Count')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{int(height)}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 4),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=10, color='#666666', fontweight='bold')
        ax2.set_yscale('symlog', linthresh=1.0)
        ax2.set_ylabel('Number of Samples', color='gray', fontsize=12)
        ax2.tick_params('y', colors='gray')
        max_count = max(bin_counts) if len(bin_counts) > 0 else 10
        ax2.set_ylim(0, max_count * 5)

        ax1.plot(bin_centers, bin_accs, marker='o', markersize=8, linewidth=3, color='#d62728', label='Accuracy')
        ax1.set_ylabel('Accuracy', color='#d62728', fontsize=14)
        ax1.tick_params('y', colors='#d62728')
        ax1.set_ylim(0.5, 1.05)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xticks(bin_centers)
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11, fontweight='500')
        ax1.set_xlabel('Digit Scale Range ( pixels, $\\sqrt{W \\times H}$ )', fontsize=14, labelpad=10)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[{self.name}] figure save to : {save_path}")
        else:
            plt.show()


class AngleRobustnessEvaluator:
    def __init__(self,
                 name='Digit1_Angle',
                 angles=[-45, -30, -15, 0, 15, 30, 45]):
        self.name = name
        self.angles = angles
        self.correct_counts = {angle: 0 for angle in angles}
        self.total_counts = {angle: 0 for angle in angles}

    def evaluate_digit1(self, model, images, labels, device, ignore_index=-100):

        if isinstance(labels, int):
            labels = torch.tensor([labels], device=device)

        valid_mask = labels != ignore_index
        if not valid_mask:
            return
        valid_labels = labels[valid_mask]

        for angle in self.angles:
            rotated_images = TF.rotate(images, angle, interpolation=TF.InterpolationMode.BILINEAR).to(device)

            with torch.no_grad():
                # ---------------HCL_Changes---------------
                # digital head removed from the model
                # _, logits_digital, logits_d1, logits_d2 = model(rotated_images)#, _
                _, logits_d1, logits_d2 = model(rotated_images)
                # -------------------------------------------

            valid_logits = logits_d1[valid_mask]

            preds = torch.argmax(valid_logits, dim=1)
            correct = (preds == valid_labels).sum().item()

            self.correct_counts[angle] += correct
            self.total_counts[angle] += len(valid_labels)

    def evaluate_digit2(self, model, images, labels, device, ignore_index=-100):

        if isinstance(labels, int):
            labels = torch.tensor([labels], device=device)

        valid_mask = labels != ignore_index
        if not valid_mask:
            return
        valid_labels = labels[valid_mask]

        for angle in self.angles:
            rotated_images = TF.rotate(images, angle, interpolation=TF.InterpolationMode.BILINEAR).to(device)

            with torch.no_grad():
                # ---------------HCL_Changes---------------
                # digital head removed from the model
                # _, logits_digital, logits_d1, logits_d2 = model(rotated_images)#, _
                _, logits_d1, logits_d2 = model(rotated_images)
                # -------------------------------------------

            valid_logits = logits_d2[valid_mask]

            preds = torch.argmax(valid_logits, dim=1)
            correct = (preds == valid_labels).sum().item()

            self.correct_counts[angle] += correct
            self.total_counts[angle] += len(valid_labels)

    def plot_robustness_curve(self, save_path=None):
        accuracies = []
        for angle in self.angles:
            if self.total_counts[angle] > 0:
                acc = self.correct_counts[angle] / self.total_counts[angle]
            else:
                acc = 0
            accuracies.append(acc)

        plt.figure(figsize=(10, 6))

        plt.plot(self.angles, accuracies, marker='o', markersize=8, linewidth=3, color="#1f77b4", label='Model Accuracy')

        zero_index = self.angles.index(0) if 0 in self.angles else -1
        if zero_index != -1:
            baseline_acc = accuracies[zero_index]
            plt.axhline(y=baseline_acc, color='r', linestyle='--', alpha=0.5, label=f'Baseline(0°) Acc:{baseline_acc:.3f}')

        plt.fill_between(self.angles, accuracies, color='#1f77b4', alpha=0.1)

        plt.title(f"Accuracy vs. Rotation Angle - {self.name}", fontsize=16, fontweight='bold')
        plt.xlabel("Rotation Angle (Degrees)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)

        plt.xticks(self.angles, fontsize=12)
        plt.yticks(fontsize=12)

        plt.ylim(0.5, 1.05)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Anagle robustness analysis figure save to : {save_path}")
        else:
            plt.show()


class ConfidenceEvaluator:
    def __init__(self,
                 name='Digit'):
        self.name = name
        self.confidences = []
        self.predicitions = []
        self.labels = []

    def update(self, logits, targets, ignore_index=-100):

        if isinstance(targets, int):
            targets = torch.tensor([targets], device=logits.device)
        valid_mask = targets != ignore_index

        if not valid_mask:
            return

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        probs = F.softmax(valid_logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        self.confidences.extend(confs.detach().cpu().numpy())
        self.predicitions.extend(preds.detach().cpu().numpy())
        self.labels.extend(valid_targets.detach().cpu().numpy())

    def plot_analysis(self, save_path=None):
        confs = np.array(self.confidences)
        preds = np.array(self.predicitions)
        labels = np.array(self.labels)
        corrects = (preds == labels).astype(int)

        if len(confs) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Confidence Analysis for {self.name}', fontsize=16)
        bins = np.linspace(0, 1.0, 11)
        bin_accs, bin_confs, bin_counts = [], [], []

        for i in range(len(bins) - 1):
            mask = (confs >= bins[i]) & (confs < bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(corrects[mask].mean())
                bin_confs.append(confs[mask].mean())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
            bin_counts.append(mask.sum())

        ax1.bar(bins[:-1], bin_accs, width=0.1, align='edge', alpha=0.7, edgecolor='black', label='Actual Accuracy')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Confidence Bin')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram (Calibration)')
        ax1.set_xlim(0, 1.0)
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        thresholds = np.linspace(0.1, 0.99, 50)
        acc_at_th, coverage_at_th = [], []

        for th in thresholds:
            mask = confs >= th
            coverage = mask.sum() / len(confs)
            acc = corrects[mask].mean() if mask.sum() > 0 else 0.0
            acc_at_th.append(acc)
            coverage_at_th.append(coverage)

        ax2.plot(thresholds, acc_at_th, 'b-', linewidth=2, label='Accuracy (Retained)')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Accuracy', color='b')
        ax2.tick_params('y', colors='b')
        ax2.set_ylim(0.5, 1.01)

        ax2_right = ax2.twinx()
        ax2_right.plot(thresholds, coverage_at_th, 'g--', linewidth=2, label='Coverage (Retained %)')
        ax2_right.set_ylabel('Coverage', color='g')
        ax2_right.tick_params('y', colors='g')
        ax2_right.set_ylim(0, 1.05)

        ax2.set_title('Accuracy & Coverage vs Threshold')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"[{self.name}] figure save to : {save_path}")
        else:
            plt.show()

    def plot_per_class_analysis(self, target_classes=range(11), save_path=None):
        confs = np.array(self.confidences)
        preds = np.array(self.predicitions)
        labels = np.array(self.labels)
        corrects = (preds == labels).astype(int)

        if len(confs) == 0:
            return

        thresholds = np.linspace(0.1, 0.99, 40)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Per-Class Threshold Analysis for {self.name}', fontsize=16, fontweight='bold')

        cmap = plt.get_cmap('tab10')

        for c in target_classes:
            class_mask = (labels == c)
            if class_mask.sum() == 0:
                continue

            class_confs = confs[class_mask]
            class_corrects = corrects[class_mask]
            total_class_samples = len(class_confs)

            acc_at_th, cov_at_th = [], []

            for th in thresholds:
                th_mask = class_confs > th
                retained_samples = th_mask.sum()

                coverage = retained_samples / total_class_samples

                acc = class_corrects[th_mask].mean() if retained_samples > 0 else np.nan

                acc_at_th.append(acc)
                cov_at_th.append(coverage)

            color = cmap(c % 10)

            ax1.plot(thresholds, acc_at_th, label=f'Digit {c}', color=color, linewidth=2, marker='o', markersize=3)

            ax2.plot(thresholds, cov_at_th, label=f'Digit {c}', color=color, linewidth=2, linestyle='--')

        ax1.set_xlabel('Confidence Threshold', fontsize=12)
        ax1.set_ylabel('Accuracy (Retained)', fontsize=12)
        ax1.set_title('Accuracy vs Threshold (Per Class)', fontsize=14)
        ax1.set_xlim(0.1, 1.0)
        ax1.set_ylim(0.5, 1.01)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        ax2.set_xlabel('Confidence Threshold', fontsize=12)
        ax2.set_ylabel('Coverage (Retention %)', fontsize=12)
        ax2.set_title('Coverage vs Threshold (Per Class)', fontsize=14)
        ax2.set_xlim(0.1, 1.0)
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[{self.name}] per-class analysis figure is saved in: {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        help="path to checkpoint")
    parser.add_argument("--test_image_path", type=str,
                        help="path to test image")
    parser.add_argument("--numberOfDigits", type=int, default=2,
                        help="path to checkpoint")

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MultiTaskLearnerWithState().to(device)
    model = MultiTaskLearner().to(device)

    assert os.path.exists(opt.checkpoint)
    if opt.checkpoint.endswith('.pth'):

        print('---')
        print('Testing pytorch model')
        print('---\n')

        pretrained_dict = torch.load(opt.checkpoint)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict, strict=False)
        model.cuda()

    assert os.path.exists(opt.test_image_path), 'the file is not exited'
    image_path = glob.glob(opt.test_image_path + '/images' + '/*.jpg')

    model.eval()

    jerseyNumber_correct, state_correct, total_body, count_2digit_bodies, cont_3digit_bodies = 0, 0, 0, 0, 0
    state_correct_3digit, state_correct_less3digit = 0, 0

    evaluator_d1 = ConfidenceEvaluator(name="Digit_1")
    evaluator_d2 = ConfidenceEvaluator(name="Digit_2")

    anlge_evaluator_d1 = AngleRobustnessEvaluator(name='Digit_1')
    anlge_evaluator_d2 = AngleRobustnessEvaluator(name='Digit_2')

    scale_evalutaor_d1 = ScaleRobustnessEvaluator(name='Digit_1')
    scale_evalutaor_d2 = ScaleRobustnessEvaluator(name='Digit_2')
    scale_evalutaor_state = ScaleRobustnessEvaluator(name='State')

    TARGET_W, TARGET_H = 96, 96
    with torch.no_grad():
        for img in tqdm(image_path, desc=f"Testing..."):
            assert os.path.exists(img)
            label_info = img.replace('images', 'labels').replace('.jpg', '.json')

            with open(label_info, 'r', encoding='utf8') as f:
                data = json.load(f)

            body_cnt = len(data["data"]["jersey"])
            total_body += body_cnt

            org_img = cv2.imread(img)
            # print("Original Image ----",org_img.shape)
            for i in data["data"]["jersey"]:
                body_rect = i["body"]["rect"]
                all_jersey_numbers = i["jersey_number"]

                jersey_number = []
                digit_bboxes = []
                for j in range(len(all_jersey_numbers)):
                    digit = i['numbers'][j]['digits']
                    digit_bbox = i['numbers'][j]['rect']
                    jersey_number.append(digit)
                    digit_bboxes.append(digit_bbox)

                if len(jersey_number) > 2:
                    cont_3digit_bodies += 1
                    state = 2
                elif len(jersey_number) <= 2:
                    count_2digit_bodies += 1
                    state = 1
                else:
                    state = 0

                if len(jersey_number) < opt.numberOfDigits:
                    for _ in range(opt.numberOfDigits - len(jersey_number)):
                        jersey_number.append(10)
                        digit_bboxes.append([0, 0, 0, 0])

                minx = int(body_rect[0])
                miny = int(body_rect[1])
                maxx = int(body_rect[0] + body_rect[2])
                maxy = int(body_rect[1] + body_rect[3])

                body_img = org_img[miny:maxy, minx:maxx]
                # plt.imshow(body_img)#.permute(1, 2, 0)
                # plt.show()
                body_img = cv2.resize(body_img, (96, 96))
                # plt.imshow(body_img)#.permute(1, 2, 0)
                # plt.show()
                ratio_w = TARGET_W / body_rect[2]
                ratio_h = TARGET_H / body_rect[3]

                digit_bboxes_resized = digit_bboxes.copy()
                digit_bboxes_resized = np.array(digit_bboxes_resized)

                digit_bboxes_resized[:, 2] = digit_bboxes_resized[:, 2] * ratio_w
                digit_bboxes_resized[:, 3] = digit_bboxes_resized[:, 3] * ratio_h

                if opt.checkpoint.endswith('.pth'):
                    body_img = Transforms.ToTensor()(body_img).unsqueeze(0).cuda()
                    # path_save = plug_save_image(body_img, "Testing1")
                    # print("Body Image -----",path_save)

                    # ---------------HCL_Changes---------------
                    # digital head removed from the model
                    # _, logits_digital, logits_d1, logits_d2 = model(body_img)#, logits_state
                    _, logits_d1, logits_d2 = model(body_img)
                    # -------------------------------------------

                    # print("End--------------")
                    evaluator_d1.update(logits_d1, jersey_number[0], ignore_index=-100)
                    evaluator_d2.update(logits_d2, jersey_number[1], ignore_index=-100)

                    anlge_evaluator_d1.evaluate_digit1(model, body_img, jersey_number[0], device)
                    anlge_evaluator_d2.evaluate_digit2(model, body_img, jersey_number[1], device)

                    scale_evalutaor_d1.update(logits_d1, jersey_number[0], digit_bboxes_resized.tolist()[0], ignore_index=-100)
                    scale_evalutaor_d2.update(logits_d2, jersey_number[1], digit_bboxes_resized.tolist()[1], ignore_index=10)
                    # scale_evalutaor_state.update(logits_state, state, body_rect, ignore_index=-100)

                    # _, predicted_state = torch.max(logits_state, 1)
                    _, pred_d1 = torch.max(logits_d1, 1)
                    _, pred_d2 = torch.max(logits_d2, 1)
                    # print("--------------------------------------------------")
                    # print(pred_d1.item())
                    # print(pred_d2.item())
                    image_name = f"{pred_d1.item()}_{pred_d2.item()}"
                    # print(image_name)
                    plug_save_image(body_img, image_name, "Testing3")
                    # state_correct += predicted_state.eq(state)
                    # if state == 2 and predicted_state.eq(state):
                    #     state_correct_3digit += 1
                    # elif state == 1 and predicted_state.eq(state):
                    #     state_correct_less3digit += 1
                    if len(jersey_number) <= 2:
                        jerseyNumber_correct += pred_d1.eq(jersey_number[0]).logical_and(
                            pred_d2.eq(jersey_number[1]))
    print("----")
    print("the accuracy of the 2 digit is %2f" % ((jerseyNumber_correct.item()/count_2digit_bodies)*100))
    # print(f'the accuracy of the state is {100 * (state_correct.item()/total_body):.2f}')
    print(f'the accuracy of the state2 is {100 * (state_correct_3digit/cont_3digit_bodies):.2f}')
    print(f'the accuracy of the state1 is {100 * (state_correct_less3digit/count_2digit_bodies):.2f}')

'''
    os.makedirs('results/Fig', exist_ok=True)
    evaluator_d1.plot_per_class_analysis(save_path="./results/Test/per_class_d1.png")
    evaluator_d2.plot_per_class_analysis(save_path="./results/Test/per_class_d2.png")

    anlge_evaluator_d1.plot_robustness_curve(save_path="./results/Test/angleAnaly_d1.png")
    anlge_evaluator_d2.plot_robustness_curve(save_path="./results/Test/angleAnaly_d2.png")
    scale_evalutaor_d1.plot_scale_analysis(num_bins=10, save_path='./results/Test/scaleAnaly_d1_resized.png')
    scale_evalutaor_d2.plot_scale_analysis(num_bins=10, save_path='./results/Test/scaleAnaly_d2_resized.png')
    # scale_evalutaor_state.plot_scale_analysis(num_bins=10, save_path='./results/Fig/scaleAnaly_state.png')
'''
