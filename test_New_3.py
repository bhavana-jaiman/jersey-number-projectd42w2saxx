import os
import csv
import cv2
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # safe on SSH / no display
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# CHANGED: new backbone file + same class as training
from subModules.backbone_New import MultiTaskLearnerWithState

# ---------------HCL_Changes---------------
# Must match loss_New.py / jersey_dataset_New.py
#   digit1 head : 10 classes (0-9)
#   digit2 head : 11 classes (0-9, 10 = blank)
#   state head  : 3 classes (0 = no digit, 1 = one/two digits, 2 = three+ digits)
# Model forward returns: feat, digit_1, digit_2, logits_state
IGNORE_INDEX = -100
BLANK_DIGIT = 10
TARGET_W, TARGET_H = 96, 96  # overwritten by --img_size in __main__
STATE_NAMES = {0: "No digit", 1: "1-2 digits", 2: "3+ digits"}
# -------------------------------------------


# ======================================================================
# Helpers
# ======================================================================
def _to_target_tensor(targets, device):
    if isinstance(targets, (int, np.integer)):
        return torch.tensor([int(targets)], device=device, dtype=torch.long)
    return torch.as_tensor(targets, device=device).long().view(-1)


def _valid_mask(targets, ignore):
    """ignore can be a single int or a list of ints."""
    if isinstance(ignore, (int, np.integer)):
        ignore = [int(ignore)]
    mask = torch.ones_like(targets, dtype=torch.bool)
    for v in ignore:
        mask &= targets != v
    return mask


def _finish_figure(fig, save_path, msg):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{msg} saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _pct(a, b):
    return 100.0 * a / b if b > 0 else float("nan")


def _class_name(c):
    return "Blank (10)" if c == BLANK_DIGIT else f"Digit {c}"


# ======================================================================
# Scale robustness
# ======================================================================
class ScaleRobustnessEvaluator:
    def __init__(self, name="Digit_Scale"):
        self.name = name
        self.scales = []
        self.corrects = []

    def update(self, logits, targets, bboxes, ignore_index=IGNORE_INDEX):
        targets = _to_target_tensor(targets, logits.device)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=logits.device).view(-1, 4)

        valid = _valid_mask(targets, ignore_index)
        if not valid.any():
            return

        preds = torch.argmax(logits[valid], dim=1)
        correct = (preds == targets[valid]).cpu().numpy().astype(int)

        b = bboxes[valid]
        areas = torch.clamp(b[:, 2] * b[:, 3], min=1.0)
        scales = torch.sqrt(areas).cpu().numpy()

        self.scales.append(scales)
        self.corrects.append(correct)

    def plot_scale_analysis(self, num_bins=10, save_path=None):
        if len(self.scales) == 0:
            print(f"[{self.name}] No scale data collected")
            return

        # concatenate (not np.array) so batches of any size work
        scales = np.concatenate(self.scales)
        corrects = np.concatenate(self.corrects)

        min_scale = np.floor(scales.min())
        max_scale = np.ceil(scales.max())  # was floor -> largest samples were dropped
        if max_scale <= min_scale:
            max_scale = min_scale + 1

        bins = np.linspace(min_scale, max_scale, num_bins + 1)
        bin_centers, bin_accs, bin_counts, x_labels = [], [], [], []

        for i in range(num_bins):
            if i == num_bins - 1:
                mask = (scales >= bins[i]) & (scales <= bins[i + 1])
            else:
                mask = (scales >= bins[i]) & (scales < bins[i + 1])
            count = int(mask.sum())
            bin_counts.append(count)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            x_labels.append(f"{int(bins[i])} - {int(bins[i + 1])}")
            bin_accs.append(corrects[mask].mean() if count > 0 else np.nan)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Accuracy vs. Digit Scale (BBox Size) - {self.name}",
                     fontsize=16, fontweight="bold")

        ax2 = ax1.twinx()
        width = (max_scale - min_scale) / num_bins * 0.8
        bars = ax2.bar(bin_centers, bin_counts, width=width, color="gray",
                       alpha=0.3, label="Sample Count")
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f"{int(height)}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 4), textcoords="offset points",
                             ha="center", va="bottom",
                             fontsize=10, color="#666666", fontweight="bold")
        ax2.set_yscale("symlog", linthresh=1.0)
        ax2.set_ylabel("Number of Samples", color="gray", fontsize=12)
        ax2.tick_params("y", colors="gray")
        max_count = max(bin_counts) if len(bin_counts) > 0 else 10
        ax2.set_ylim(0, max(max_count, 1) * 5)

        ax1.plot(bin_centers, bin_accs, marker="o", markersize=8, linewidth=3,
                 color="#d62728", label="Accuracy")
        ax1.set_ylabel("Accuracy", color="#d62728", fontsize=14)
        ax1.tick_params("y", colors="#d62728")
        ax1.set_ylim(0.0, 1.05)  # was 0.5 -> hid bins below 50%
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_xticks(bin_centers)
        ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=11, fontweight="500")
        ax1.set_xlabel(r"Digit Scale Range ( pixels, $\sqrt{W \times H}$ )", fontsize=14, labelpad=10)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", fontsize=12)

        fig.tight_layout()
        _finish_figure(fig, save_path, f"[{self.name}] scale figure")


# ======================================================================
# Angle robustness
# ======================================================================
class AngleRobustnessEvaluator:
    # ---------------HCL_Changes---------------
    # evaluate_digit1 / evaluate_digit2 merged into evaluate(); `head` picks
    # which output to score. Model unpacking updated to the new forward():
    #   feat, digit_1, digit_2, logits_state
    # (old code read `_, logits_digital, logits_d1, logits_d2`, which with the
    # new model silently scored digit_2 as d1 and the STATE head as d2)
    def __init__(self, name="Digit1_Angle", head="d1",
                 angles=(-45, -30, -15, 0, 15, 30, 45), ignore_index=IGNORE_INDEX):
        assert head in ("d1", "d2")
        self.name = name
        self.head = head
        self.angles = list(angles)
        self.ignore_index = ignore_index
        self.correct_counts = {a: 0 for a in self.angles}
        self.total_counts = {a: 0 for a in self.angles}

    def evaluate(self, model, images, labels, device):
        labels = _to_target_tensor(labels, device)
        valid = _valid_mask(labels, self.ignore_index)
        if not valid.any():
            return
        valid_labels = labels[valid]
        images = images.to(device)

        with torch.no_grad():
            for angle in self.angles:
                rotated = TF.rotate(images, angle, interpolation=InterpolationMode.BILINEAR)
                _, logits_d1, logits_d2, _ = model(rotated)
                logits = logits_d1 if self.head == "d1" else logits_d2
                preds = torch.argmax(logits[valid], dim=1)
                self.correct_counts[angle] += (preds == valid_labels).sum().item()
                self.total_counts[angle] += valid_labels.numel()
    # -------------------------------------------

    def plot_robustness_curve(self, save_path=None):
        accuracies = [
            self.correct_counts[a] / self.total_counts[a] if self.total_counts[a] > 0 else np.nan
            for a in self.angles
        ]

        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.angles, accuracies, marker="o", markersize=8, linewidth=3,
                 color="#1f77b4", label="Model Accuracy")

        if 0 in self.angles:
            baseline_acc = accuracies[self.angles.index(0)]
            if not np.isnan(baseline_acc):
                plt.axhline(y=baseline_acc, color="r", linestyle="--", alpha=0.5,
                            label=f"Baseline(0°) Acc:{baseline_acc:.3f}")

        plt.fill_between(self.angles, np.nan_to_num(accuracies), color="#1f77b4", alpha=0.1)
        plt.title(f"Accuracy vs. Rotation Angle - {self.name}", fontsize=16, fontweight="bold")
        plt.xlabel("Rotation Angle (Degrees)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(self.angles, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        _finish_figure(fig, save_path, f"[{self.name}] angle figure")


# ======================================================================
# Confidence / calibration
# ======================================================================
class ConfidenceEvaluator:
    # ---------------HCL_Changes---------------
    # num_classes added: digit1 = 10, digit2 = 11 (10 = blank)
    def __init__(self, name="Digit", num_classes=10):
        self.name = name
        self.num_classes = num_classes
        self.confidences = []
        self.predictions = []
        self.labels = []
    # -------------------------------------------

    def update(self, logits, targets, ignore_index=IGNORE_INDEX):
        targets = _to_target_tensor(targets, logits.device)
        valid = _valid_mask(targets, ignore_index)
        if not valid.any():
            return

        probs = F.softmax(logits[valid].float(), dim=1)
        confs, preds = torch.max(probs, dim=1)

        self.confidences.extend(confs.detach().cpu().numpy())
        self.predictions.extend(preds.detach().cpu().numpy())
        self.labels.extend(targets[valid].detach().cpu().numpy())

    def _arrays(self):
        confs = np.array(self.confidences)
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        return confs, (preds == labels).astype(int), labels

    def plot_analysis(self, save_path=None):
        confs, corrects, _ = self._arrays()
        if len(confs) == 0:
            print(f"[{self.name}] No confidence data collected")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Confidence Analysis for {self.name}", fontsize=16)

        bins = np.linspace(0, 1.0, 11)
        bin_accs = []
        for i in range(len(bins) - 1):
            upper = (confs <= bins[i + 1]) if i == len(bins) - 2 else (confs < bins[i + 1])
            mask = (confs >= bins[i]) & upper
            bin_accs.append(corrects[mask].mean() if mask.sum() > 0 else 0)

        ax1.bar(bins[:-1], bin_accs, width=0.1, align="edge", alpha=0.7,
                edgecolor="black", label="Actual Accuracy")
        ax1.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
        ax1.set_xlabel("Confidence Bin")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Reliability Diagram (Calibration)")
        ax1.set_xlim(0, 1.0)
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        thresholds = np.linspace(0.1, 0.99, 50)
        acc_at_th, coverage_at_th = [], []
        for th in thresholds:
            mask = confs >= th
            coverage_at_th.append(mask.sum() / len(confs))
            acc_at_th.append(corrects[mask].mean() if mask.sum() > 0 else np.nan)

        ax2.plot(thresholds, acc_at_th, "b-", linewidth=2, label="Accuracy (Retained)")
        ax2.set_xlabel("Confidence Threshold")
        ax2.set_ylabel("Accuracy", color="b")
        ax2.tick_params("y", colors="b")
        ax2.set_ylim(0.0, 1.01)

        ax2_right = ax2.twinx()
        ax2_right.plot(thresholds, coverage_at_th, "g--", linewidth=2, label="Coverage (Retained %)")
        ax2_right.set_ylabel("Coverage", color="g")
        ax2_right.tick_params("y", colors="g")
        ax2_right.set_ylim(0, 1.05)

        ax2.set_title("Accuracy & Coverage vs Threshold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        _finish_figure(fig, save_path, f"[{self.name}] confidence figure")

    def plot_per_class_analysis(self, target_classes=None, save_path=None):
        confs, corrects, labels = self._arrays()
        if len(confs) == 0:
            print(f"[{self.name}] No confidence data collected")
            return
        if target_classes is None:
            target_classes = range(self.num_classes)

        thresholds = np.linspace(0.1, 0.99, 40)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Per-Class Threshold Analysis for {self.name}", fontsize=16, fontweight="bold")
        cmap = plt.get_cmap("tab20")

        for c in target_classes:
            class_mask = labels == c
            if class_mask.sum() == 0:
                continue
            class_confs = confs[class_mask]
            class_corrects = corrects[class_mask]
            total = len(class_confs)

            acc_at_th, cov_at_th = [], []
            for th in thresholds:
                th_mask = class_confs >= th  # >= to match plot_analysis
                kept = th_mask.sum()
                cov_at_th.append(kept / total)
                acc_at_th.append(class_corrects[th_mask].mean() if kept > 0 else np.nan)

            color = cmap(c % 20)
            ax1.plot(thresholds, acc_at_th, label=_class_name(c), color=color,
                     linewidth=2, marker="o", markersize=3)
            ax2.plot(thresholds, cov_at_th, label=_class_name(c), color=color,
                     linewidth=2, linestyle="--")

        ax1.set_xlabel("Confidence Threshold", fontsize=12)
        ax1.set_ylabel("Accuracy (Retained)", fontsize=12)
        ax1.set_title("Accuracy vs Threshold (Per Class)", fontsize=14)
        ax1.set_xlim(0.1, 1.0)
        ax1.set_ylim(0.0, 1.01)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

        ax2.set_xlabel("Confidence Threshold", fontsize=12)
        ax2.set_ylabel("Coverage (Retention %)", fontsize=12)
        ax2.set_title("Coverage vs Threshold (Per Class)", fontsize=14)
        ax2.set_xlim(0.1, 1.0)
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        _finish_figure(fig, save_path, f"[{self.name}] per-class figure")


# ======================================================================
# State head (new)
# ======================================================================
class StateEvaluator:
    def __init__(self, name="State", num_states=3):
        self.name = name
        self.num_states = num_states
        self.confusion = np.zeros((num_states, num_states), dtype=int)  # rows = true, cols = pred

    def update(self, pred_state, true_state):
        self.confusion[int(true_state), int(pred_state)] += 1

    def accuracy(self):
        total = self.confusion.sum()
        return np.trace(self.confusion) / total if total > 0 else float("nan")

    def plot_confusion(self, save_path=None):
        if self.confusion.sum() == 0:
            print(f"[{self.name}] No state data collected")
            return
        row_sum = self.confusion.sum(axis=1, keepdims=True)
        norm = np.divide(self.confusion, row_sum, out=np.zeros_like(self.confusion, dtype=float),
                         where=row_sum > 0)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        names = [STATE_NAMES[i] for i in range(self.num_states)]
        ax.set_xticks(range(self.num_states))
        ax.set_yticks(range(self.num_states))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted State")
        ax.set_ylabel("True State")
        ax.set_title(f"State Confusion Matrix (acc {100 * self.accuracy():.2f}%)")
        for i in range(self.num_states):
            for j in range(self.num_states):
                ax.text(j, i, f"{self.confusion[i, j]}\n({100 * norm[i, j]:.1f}%)",
                        ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _finish_figure(fig, save_path, f"[{self.name}] state confusion figure")


# ======================================================================
# Data / model helpers
# ======================================================================
def load_model(ckpt_path, device, allow_partial=False):
    model = MultiTaskLearnerWithState().to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # ---------------HCL_Changes---------------
    # strict=False used to hide architecture mismatches (e.g. testing an OLD
    # checkpoint with the digital head / 11-class digit1). Report and stop instead.
    model_state = model.state_dict()
    missing = [k for k in model_state if k not in state]
    unexpected = [k for k in state if k not in model_state]
    shape_mismatch = [k for k in state if k in model_state and state[k].shape != model_state[k].shape]

    if missing or unexpected or shape_mismatch:
        print("WARNING: checkpoint does not match the current MultiTaskLearnerWithState")
        print("  missing   :", missing)
        print("  unexpected:", unexpected)
        print("  shape diff:", shape_mismatch)
        if not allow_partial:
            raise RuntimeError("Checkpoint/architecture mismatch. Use a checkpoint trained with "
                               "train_New.py, or pass --allow_partial to load matching tensors only.")
        state = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=True)
        print("Checkpoint loaded (all keys matched).")
    # -------------------------------------------

    model.eval()
    return model


def label_path_for(img_path):
    # <root>/images/x.jpg -> <root>/labels/x.json (only the folder name is replaced,
    # not every 'images' substring in the full path)
    root = os.path.dirname(os.path.dirname(img_path))
    name = os.path.splitext(os.path.basename(img_path))[0] + ".json"
    return os.path.join(root, "labels", name)


def build_targets(entry):
    """
    Returns body_rect, state, (d1, d2), (box1, box2), true_digits, all_digit_rects
    Same rules as JerseyWithState_Dataset:
      0 digits -> state 0, digit targets IGNORE
      1-2      -> state 1, d2 = BLANK if single digit
      3+       -> state 2, digit targets IGNORE
    """
    body_rect = [float(v) for v in entry["body"]["rect"]]

    digits = []
    for j in range(len(entry["jersey_number"])):
        num = entry["numbers"][j]
        digits.append((int(num["digits"]), [float(v) for v in num["rect"]]))
    # sort left -> right, same as training (argsort on x-centre)
    digits.sort(key=lambda d: d[1][0])

    n = len(digits)
    state = 0 if n == 0 else (1 if n <= 2 else 2)

    zero_box = [0.0, 0.0, 0.0, 0.0]
    if state == 1:
        d1, box1 = digits[0]
        if n == 2:
            d2, box2 = digits[1]
        else:
            d2, box2 = BLANK_DIGIT, zero_box
    else:
        d1 = d2 = IGNORE_INDEX
        box1 = box2 = zero_box

    all_rects = [r for _, r in digits]
    return body_rect, state, (d1, d2), (box1, box2), [d for d, _ in digits], all_rects


def crop_region(org_img_bgr, x1, y1, x2, y2):
    """Crop [x1,y1,x2,y2] (clamped to the image), BGR->RGB, resize to TARGET size."""
    h_img, w_img = org_img_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    crop = org_img_bgr[y1:y2, x1:x2]
    # training uses PIL .convert("RGB") -> model expects RGB, cv2 gives BGR
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return crop, (x2 - x1), (y2 - y1)


def crop_body(org_img_bgr, body_rect):
    x, y, bw, bh = body_rect
    return crop_region(org_img_bgr, x, y, x + bw, y + bh)


# ---------------HCL_Changes---------------
# --crop number : ORACLE diagnostic. Crops the union of the ground-truth digit
# boxes plus the same pixel margins JerseyWithState_Dataset uses in training
# (left 23, top 30, right 30, bottom 33). Not usable in deployment (it needs the
# GT boxes); it only tells us how much accuracy the crop mismatch costs.
def digit_union_abs(all_rects, body_rect, coords):
    """Union of digit rects [x,y,w,h] in IMAGE pixel coords.
    coords = 'image' | 'body' | 'auto' (auto: if the union does not overlap the
    body box when read as image coords, treat it as relative to the body)."""
    x1 = min(r[0] for r in all_rects)
    y1 = min(r[1] for r in all_rects)
    x2 = max(r[0] + r[2] for r in all_rects)
    y2 = max(r[1] + r[3] for r in all_rects)
    bx, by, bw, bh = body_rect

    relative = coords == "body"
    if coords == "auto":
        overlaps = not (x2 < bx or x1 > bx + bw or y2 < by or y1 > by + bh)
        relative = not overlaps
    if relative:
        x1, x2, y1, y2 = x1 + bx, x2 + bx, y1 + by, y2 + by
    return (x1, y1, x2, y2), relative


def crop_number(org_img_bgr, body_rect, all_rects, margins, coords):
    """Returns crop, crop_w, crop_h, used_relative_coords"""
    if len(all_rects) == 0:
        # no digit: training uses the whole image for label 10 -> whole body here
        crop, w, h = crop_body(org_img_bgr, body_rect)
        return crop, w, h, False
    (x1, y1, x2, y2), relative = digit_union_abs(all_rects, body_rect, coords)
    ml, mt, mr, mb = margins
    crop, w, h = crop_region(org_img_bgr, x1 - ml, y1 - mt, x2 + mr, y2 + mb)
    return crop, w, h, relative


# CHANGED (new): --crop number_rel
# Same as crop_number, but the margins are a FRACTION OF THE DIGIT HEIGHT instead
# of fixed pixels. Test digits are ~3x smaller than training digits (median 22 px
# vs 62 px), so fixed 23/30/30/33 px margins made the digits fill only ~26% of the
# crop (training: ~50%) and pulled in sponsor text / logos -> false "state 2".
# Default fractions 0.37/0.48/0.48/0.53 come from measure_crop_ratio.py
# (training medians), so test crops have the same digits/crop ratio as training.
def crop_number_rel(org_img_bgr, body_rect, all_rects, rel_margins, coords):
    """Returns crop, crop_w, crop_h, used_relative_coords"""
    if len(all_rects) == 0:
        crop, w, h = crop_body(org_img_bgr, body_rect)
        return crop, w, h, False
    (x1, y1, x2, y2), relative = digit_union_abs(all_rects, body_rect, coords)
    H = y2 - y1
    ml, mt, mr, mb = [f * H for f in rel_margins]
    crop, w, h = crop_region(org_img_bgr, x1 - ml, y1 - mt, x2 + mr, y2 + mb)
    return crop, w, h, relative
# -------------------------------------------


def scale_box(box, ratio_w, ratio_h):
    return [box[0], box[1], box[2] * ratio_w, box[3] * ratio_h]


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint (.pth)")
    parser.add_argument("--test_image_path", type=str, required=True,
                        help="dataset root containing images/ and labels/")
    parser.add_argument("--output", type=str, default="./results/Test", help="folder for figures/CSV")
    parser.add_argument("--no_angle", action="store_true",
                        help="skip the rotation test (runs the model 7 extra times per body)")
    parser.add_argument("--allow_partial", action="store_true",
                        help="load only matching tensors if the checkpoint does not fully match")
    # ---------------HCL_Changes---------------
    # CHANGED: default is now number_rel (matches training framing)
    parser.add_argument("--crop", type=str, default="number_rel", choices=["body", "number", "number_rel"],
                        help="number_rel = GT digit boxes + margins relative to digit height (default); "
                             "number = GT digit boxes + fixed pixel margins; "
                             "body = whole body box (deployment-like)")
    parser.add_argument("--rel_margins", type=float, nargs=4, default=[0.37, 0.48, 0.48, 0.53],
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="for --crop number_rel: margins as a fraction of the digit height "
                             "(re-measure with measure_crop_ratio.py if the training data changes)")
    parser.add_argument("--margins", type=int, nargs=4, default=[23, 30, 30, 33],
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="pixel margins around the digit union for --crop number")
    parser.add_argument("--digit_coords", type=str, default="auto", choices=["auto", "image", "body"],
                        help="are the JSON digit rects in image coords or relative to the body box")
    parser.add_argument("--img_size", type=int, default=96,
                        help="model input size; must match what the checkpoint was trained with")
    parser.add_argument("--save_crops", type=int, default=0,
                        help="save the first N model-input crops to <output>/crops for visual checks")
    # -------------------------------------------
    opt = parser.parse_args()
    TARGET_W = TARGET_H = opt.img_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    os.makedirs(opt.output, exist_ok=True)

    assert os.path.exists(opt.checkpoint), f"checkpoint not found: {opt.checkpoint}"
    assert os.path.exists(opt.test_image_path), f"test path not found: {opt.test_image_path}"

    print("---\nTesting pytorch model\n---\n")
    model = load_model(opt.checkpoint, device, opt.allow_partial)

    image_paths = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    print(f"Found {len(image_paths)} images")

    # evaluators
    conf_d1 = ConfidenceEvaluator(name="Digit_1", num_classes=10)
    conf_d2 = ConfidenceEvaluator(name="Digit_2", num_classes=11)
    angle_d1 = AngleRobustnessEvaluator(name="Digit_1", head="d1")
    angle_d2 = AngleRobustnessEvaluator(name="Digit_2", head="d2")
    scale_d1 = ScaleRobustnessEvaluator(name="Digit_1")
    scale_d2 = ScaleRobustnessEvaluator(name="Digit_2")
    scale_state = ScaleRobustnessEvaluator(name="State")
    state_eval = StateEvaluator(name="State")

    # counters
    total_bodies = 0
    skipped_bodies = 0
    state_correct = 0
    end_to_end_correct = 0          # state right AND (digits right if state 1)
    number_total = number_correct = 0   # true state 1 only, both digit heads right
    d1_correct = d2_correct = 0
    len_total = {1: 0, 2: 0}
    len_correct = {1: 0, 2: 0}
    wrong_rows = []

    to_tensor = Transforms.ToTensor()

    # ---------------HCL_Changes---------------
    relative_coord_count = 0
    saved_crops = 0
    crops_dir = os.path.join(opt.output, "crops")
    if opt.save_crops > 0:
        os.makedirs(crops_dir, exist_ok=True)
    print(f"crop mode: {opt.crop} | input size: {TARGET_W}x{TARGET_H}"
          + (f" | rel_margins: {opt.rel_margins}" if opt.crop == "number_rel" else ""))
    # -------------------------------------------

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Testing..."):
            label_path = label_path_for(img_path)
            if not os.path.exists(label_path):
                print(f"\nlabel missing, skipped: {label_path}")
                continue

            with open(label_path, "r", encoding="utf8") as f:
                data = json.load(f)

            org_img = cv2.imread(img_path)
            if org_img is None:
                print(f"\ncould not read image, skipped: {img_path}")
                continue

            for body_i, entry in enumerate(data["data"]["jersey"]):
                body_rect, state, (d1, d2), (box1, box2), true_digits, all_rects = build_targets(entry)

                # ---------------HCL_Changes---------------
                if opt.crop == "number":
                    crop, crop_w, crop_h, used_rel = crop_number(
                        org_img, body_rect, all_rects, opt.margins, opt.digit_coords)
                    relative_coord_count += int(used_rel)
                elif opt.crop == "number_rel":                     # CHANGED (new)
                    crop, crop_w, crop_h, used_rel = crop_number_rel(
                        org_img, body_rect, all_rects, opt.rel_margins, opt.digit_coords)
                    relative_coord_count += int(used_rel)
                else:
                    crop, crop_w, crop_h = crop_body(org_img, body_rect)
                # -------------------------------------------
                if crop is None:
                    skipped_bodies += 1
                    continue
                total_bodies += 1

                if saved_crops < opt.save_crops:
                    name = (f"{saved_crops:04d}_state{state}_"
                            f"{''.join(map(str, true_digits)) or 'none'}.jpg")
                    cv2.imwrite(os.path.join(crops_dir, name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    saved_crops += 1

                body_tensor = to_tensor(crop).unsqueeze(0).to(device)

                # ---------------HCL_Changes---------------
                # new forward: feat, digit_1 (10), digit_2 (11), state (3)
                _, logits_d1, logits_d2, logits_state = model(body_tensor)
                # -------------------------------------------

                pred_d1 = int(logits_d1.argmax(dim=1)[0])
                pred_d2 = int(logits_d2.argmax(dim=1)[0])
                pred_state = int(logits_state.argmax(dim=1)[0])

                # ---- state ----
                state_ok = pred_state == state
                state_correct += int(state_ok)
                state_eval.update(pred_state, state)

                # ---- digits (only defined for true state 1) ----
                digits_ok = True
                if state == 1:
                    ok1, ok2 = pred_d1 == d1, pred_d2 == d2
                    digits_ok = ok1 and ok2
                    number_total += 1
                    number_correct += int(digits_ok)
                    d1_correct += int(ok1)
                    d2_correct += int(ok2)
                    n = len(true_digits)
                    len_total[n] += 1
                    len_correct[n] += int(digits_ok)

                sample_ok = state_ok and digits_ok
                end_to_end_correct += int(sample_ok)

                if not sample_ok:
                    wrong_rows.append({
                        "image": img_path,
                        "body_index": body_i,
                        "true_digits": "".join(map(str, true_digits)),
                        "true_state": state,
                        "pred_state": pred_state,
                        "true_d1": d1, "pred_d1": pred_d1,
                        "true_d2": d2, "pred_d2": pred_d2,
                    })

                # ---- evaluators ----
                conf_d1.update(logits_d1, d1, ignore_index=IGNORE_INDEX)
                conf_d2.update(logits_d2, d2, ignore_index=IGNORE_INDEX)  # blank (10) is a real d2 class

                if not opt.no_angle:
                    angle_d1.evaluate(model, body_tensor, d1, device)
                    angle_d2.evaluate(model, body_tensor, d2, device)

                ratio_w = TARGET_W / crop_w
                ratio_h = TARGET_H / crop_h
                scale_d1.update(logits_d1, d1, scale_box(box1, ratio_w, ratio_h),
                                ignore_index=IGNORE_INDEX)
                # blank d2 has no bbox -> exclude it from the scale plot
                scale_d2.update(logits_d2, d2, scale_box(box2, ratio_w, ratio_h),
                                ignore_index=[IGNORE_INDEX, BLANK_DIGIT])
                # state vs body size (original pixels)
                scale_state.update(logits_state, state, [0, 0, crop_w, crop_h],
                                   ignore_index=IGNORE_INDEX)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n---------------- Results ----------------")
    print(f"Crop mode                     : {opt.crop}  (input {TARGET_W}x{TARGET_H})")
    if opt.crop == "number_rel":                                  # CHANGED (new)
        print(f"Relative margins (L T R B)    : {opt.rel_margins}  x digit height")
    if opt.crop in ("number", "number_rel"):                      # CHANGED
        print(f"Digit rects read as body-relative: {relative_coord_count}  (--digit_coords {opt.digit_coords})")
    print(f"Bodies evaluated              : {total_bodies}  (skipped bad crops: {skipped_bodies})")
    print(f"End-to-end accuracy           : {_pct(end_to_end_correct, total_bodies):.2f}%"
          f"  (state correct AND digits correct when state=1)")
    print(f"State accuracy                : {_pct(state_correct, total_bodies):.2f}%")
    for s in range(3):
        row = state_eval.confusion[s]
        print(f"   state {s} ({STATE_NAMES[s]:<10}) : {_pct(row[s], row.sum()):.2f}%  (n={row.sum()})")
    print(f"Jersey number acc (state 1)   : {_pct(number_correct, number_total):.2f}%  (n={number_total})")
    print(f"   1-digit numbers            : {_pct(len_correct[1], len_total[1]):.2f}%  (n={len_total[1]})")
    print(f"   2-digit numbers            : {_pct(len_correct[2], len_total[2]):.2f}%  (n={len_total[2]})")
    print(f"   digit1 head                : {_pct(d1_correct, number_total):.2f}%")
    print(f"   digit2 head (incl. blank)  : {_pct(d2_correct, number_total):.2f}%")
    print("-----------------------------------------")

    metrics = {
        "crop_mode": opt.crop,
        "rel_margins": opt.rel_margins if opt.crop == "number_rel" else None,
        "img_size": TARGET_W,
        "bodies": total_bodies,
        "skipped_bodies": skipped_bodies,
        "end_to_end_acc": _pct(end_to_end_correct, total_bodies),
        "state_acc": _pct(state_correct, total_bodies),
        "number_acc_state1": _pct(number_correct, number_total),
        "one_digit_acc": _pct(len_correct[1], len_total[1]),
        "two_digit_acc": _pct(len_correct[2], len_total[2]),
        "digit1_acc": _pct(d1_correct, number_total),
        "digit2_acc": _pct(d2_correct, number_total),
        "state_confusion": state_eval.confusion.tolist(),
    }
    with open(os.path.join(opt.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if wrong_rows:
        with open(os.path.join(opt.output, "wrong_predictions.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=wrong_rows[0].keys())
            writer.writeheader()
            writer.writerows(wrong_rows)
        print(f"Saved {len(wrong_rows)} wrong predictions to wrong_predictions.csv")

    out = opt.output
    state_eval.plot_confusion(save_path=os.path.join(out, "state_confusion.png"))
    conf_d1.plot_analysis(save_path=os.path.join(out, "confidence_d1.png"))
    conf_d2.plot_analysis(save_path=os.path.join(out, "confidence_d2.png"))
    conf_d1.plot_per_class_analysis(save_path=os.path.join(out, "per_class_d1.png"))
    conf_d2.plot_per_class_analysis(save_path=os.path.join(out, "per_class_d2.png"))
    if not opt.no_angle:
        angle_d1.plot_robustness_curve(save_path=os.path.join(out, "angleAnaly_d1.png"))
        angle_d2.plot_robustness_curve(save_path=os.path.join(out, "angleAnaly_d2.png"))
    scale_d1.plot_scale_analysis(num_bins=10, save_path=os.path.join(out, "scaleAnaly_d1_resized.png"))
    scale_d2.plot_scale_analysis(num_bins=10, save_path=os.path.join(out, "scaleAnaly_d2_resized.png"))
    scale_state.plot_scale_analysis(num_bins=10, save_path=os.path.join(out, "scaleAnaly_state.png"))
