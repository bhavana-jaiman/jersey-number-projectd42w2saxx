"""
jersey_dataset_patched.py

Patched version of jerseyWithState_Dataset with the sticker-paste
orientation fix applied, plus a CLI at the bottom that actually GENERATES
a synthetic dataset to disk (jpg + txt pairs, same YOLO-style label
format as the original data) so you can run this file directly instead
of wiring it into training first.

WHAT CHANGED vs. the original:

  1. measure_tilt_stats(root, ...)
     NEW. Scans a sample of REAL images and estimates the actual tilt
     distribution in your data using edge/line detection, instead of
     guessing a fixed +/-8 degree range. Prints mean/std/percentiles so
     you can see the real numbers before trusting them.

  2. jerseyWithState_Dataset.__init__(..., tilt_mean=None, tilt_std=None)
     NEW optional args. If not supplied, falls back to a conservative
     default (mean=0, std=4) but PRINTS A WARNING so it's obvious the
     dataset is running on an unmeasured guess rather than real data.

  3. Inside __getitem__, right after the existing sticker resize:
     - the sticker is now rotated by an angle sampled from
       N(tilt_mean, tilt_std) instead of pasted dead-level
     - the paste is now feathered (soft edge blend) instead of a hard
       rectangular overwrite, so rotated corners don't show as black
       wedges

  Everything else (box math, padding, state/digit logic, collate_fn) is
  unchanged from your original file.

USAGE

  # Step 1 - measure your dataset's real tilt distribution first
  python jersey_dataset_patched.py measure --root /path/to/training_dataset_Ying

  # Step 2 - generate a synthetic sample batch using the measured stats
  python jersey_dataset_patched.py generate \
      --root /path/to/training_dataset_Ying \
      --outdir synthetic_output \
      --n-samples 500 \
      --tilt-mean 0.0 --tilt-std 4.2      # use the numbers Step 1 printed

  # (tilt-mean/tilt-std are optional - if omitted, generate will run
  #  measure_tilt_stats() itself first, automatically)
"""

import os
import random
import argparse
import json
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as Transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset

from utils.util import *  # noqa: F401,F403  (pad_to_square, draw_grid, etc.)


# ---------------------------------------------------------------------------
# NEW: measure the real tilt distribution in your dataset instead of
# guessing a fixed rotation range.
# ---------------------------------------------------------------------------

def _estimate_box_tilt(image_bgr: np.ndarray, box_xyxy) -> Optional[float]:
    """Estimate the dominant near-horizontal line angle inside a box region.
    Returns degrees, or None if no usable edges were found."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None

    region = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=5
    )
    if lines is None:
        return None

    angles = []
    for lx1, ly1, lx2, ly2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1))
        if abs(angle) < 45:  # keep near-horizontal lines, drop vertical seams/edges
            angles.append(angle)
    if not angles:
        return None
    return float(np.median(angles))


def measure_tilt_stats(root: str, sample_size: int = 500, seed: int = 42):
    """Scans up to `sample_size` real label/image pairs in `root`, estimates
    the tilt of each digit region, and returns (mean, std, all_angles)."""
    rng = random.Random(seed)
    txt_files = [f for f in os.listdir(root) if f.endswith(".txt")]
    rng.shuffle(txt_files)
    txt_files = txt_files[:sample_size]

    angles = []
    skipped = 0
    for fname in txt_files:
        label_path = os.path.join(root, fname)
        image_path = os.path.join(root, fname.replace(".txt", ".jpg"))
        if not os.path.exists(image_path) or os.path.getsize(label_path) == 0:
            skipped += 1
            continue
        try:
            label = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            skipped += 1
            continue
        if label.size == 0 or label[0][0] == 10:
            skipped += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        min_x = min(l[1] - l[3] / 2 for l in label) * w
        max_x = max(l[1] + l[3] / 2 for l in label) * w
        min_y = min(l[2] - l[4] / 2 for l in label) * h
        max_y = max(l[2] + l[4] / 2 for l in label) * h

        angle = _estimate_box_tilt(img, (min_x, min_y, max_x, max_y))
        if angle is not None:
            angles.append(angle)
        else:
            skipped += 1

    angles = np.array(angles)
    if len(angles) == 0:
        print("Could not estimate tilt from any sample — falling back to mean=0, std=4")
        return 0.0, 4.0, angles

    mean, std = float(np.mean(angles)), float(np.std(angles))
    print(f"\nMeasured tilt from {len(angles)} usable samples ({skipped} skipped):")
    print(f"  mean  = {mean:.2f} deg")
    print(f"  std   = {std:.2f} deg")
    print(f"  5th/95th percentile = {np.percentile(angles, 5):.2f} / {np.percentile(angles, 95):.2f} deg")
    print(f"  min/max = {angles.min():.2f} / {angles.max():.2f} deg")
    return mean, std, angles


# ---------------------------------------------------------------------------
# Dataset — same as your original, with the rotation + feathering patch
# applied inside __getitem__, and tilt_mean/tilt_std made configurable.
# ---------------------------------------------------------------------------

class jerseyWithState_Dataset(VisionDataset):
    def __init__(self, root, transform=None, tilt_mean=None, tilt_std=None,
                 single_digit_pairs_only=True) -> None:
        super(jerseyWithState_Dataset, self).__init__(root, transforms=transform)

        self.max_jerseyNumberLength = 2
        self.IGNORE_INDEX = -100
        self.single_digit_pairs_only = single_digit_pairs_only

        # Auto-detect dataset layout: flat (.jpg/.txt sitting together, e.g.
        # training_dataset_Ying) vs split into images/ + labels/ subfolders
        # (e.g. validation_dataset_Ying). Avoids a confusing "0 samples"
        # crash when pointed at a split-layout folder.
        images_dir = os.path.join(root, "images") if os.path.isdir(os.path.join(root, "images")) else root
        labels_dir = os.path.join(root, "labels") if os.path.isdir(os.path.join(root, "labels")) else root

        self.images_path = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")
        ]
        self.labels_path = [
            os.path.join(labels_dir, f.replace(".jpg", ".txt"))
            for f in os.listdir(images_dir)
            if f.endswith(".jpg")
        ]

        if len(self.images_path) == 0:
            raise SystemExit(
                f"No .jpg files found in '{images_dir}'. Checked for a flat layout "
                f"and an images/ subfolder under --root='{root}' — neither had any "
                f".jpg files. Confirm --root points at a folder that actually "
                f"contains image files (directly, or under images/)."
            )
        print(f"Found {len(self.images_path)} images in: {images_dir}")

        self.images_files = np.array(self.images_path)
        self.labels_files = np.array(self.labels_path)

        if tilt_mean is None or tilt_std is None:
            print(
                "\n⚠ tilt_mean/tilt_std not provided — using unmeasured default "
                "(mean=0, std=4). Run measure_tilt_stats() on your real data and "
                "pass the results in for a properly calibrated rotation range."
            )
            tilt_mean = 0.0 if tilt_mean is None else tilt_mean
            tilt_std = 4.0 if tilt_std is None else tilt_std

        self.tilt_mean = tilt_mean
        self.tilt_std = tilt_std

        # ------------------------------------------------------------
        # NEW: pre-scan which samples are genuinely single-digit, so
        # both the DESTINATION photo and the STICKER source photo are
        # single-digit real images. This is what "single digit pasted
        # on single digit" actually requires — sampling stickers from
        # the full dataset (old behavior) could pull a two-digit source
        # and end up pasting more than one digit, or pasting onto an
        # already-two-digit destination.
        # ------------------------------------------------------------
        self.single_digit_indices = []
        if self.single_digit_pairs_only:
            print("Scanning labels to find single-digit-only samples "
                  "(used both as paste targets and as sticker sources)...")
            for i, label_path in enumerate(self.labels_files):
                try:
                    if os.path.getsize(label_path) == 0:
                        continue
                    label = np.loadtxt(label_path).reshape(-1, 5)
                    if label.ndim == 1:
                        label = label.reshape(1, -1)
                    if label.shape[0] == 1 and label[0][0] != 10:
                        self.single_digit_indices.append(i)
                except Exception:
                    continue
            print(f"Found {len(self.single_digit_indices)} single-digit samples "
                  f"out of {len(self.labels_files)} total.")
            if len(self.single_digit_indices) == 0:
                print(
                    "⚠ No single-digit samples found — sticker-pasting augmentation "
                    "will be skipped entirely for every sample."
                )

    def _get_tight_digit_sticker(self, index):
        label_path = self.labels_files[index]
        label = np.loadtxt(label_path).reshape(-1, 5)
        if label.ndim == 1:
            label = label.reshape(1, -1)
        if label.size == 0 or label[0][0] == 10:
            return None, None, None

        sorted_indices = np.argsort(label[:, 1])
        sticker_digits = [int(label[i, 0]) for i in sorted_indices]

        image_path = self.images_files[index]
        if not os.path.exists(image_path):
            return None, None, None

        image = Image.open(image_path).convert("RGB")
        image = Transforms.ToTensor()(image)
        _, h, w = image.shape

        min_x = min(l[1] - l[3] / 2 for l in label) * w
        max_x = max(l[1] + l[3] / 2 for l in label) * w
        min_y = min(l[2] - l[4] / 2 for l in label) * h
        max_y = max(l[2] + l[4] / 2 for l in label) * h

        box_w = max_x - min_x
        box_h = max_y - min_y
        margin_x = box_w * 0.05
        margin_y = box_h * 0.15

        min_x, max_x = max(0, int(min_x - margin_x)), min(w, int(max_x + margin_x))
        min_y, max_y = max(0, int(min_y - margin_y)), min(h, int(max_y + margin_y))

        if max_x <= min_x or max_y <= min_y:
            return None, None, None

        sticker = image[:, min_y:max_y, min_x:max_x]
        mask = self._digit_foreground_mask(sticker)
        return sticker, mask, sticker_digits

    @staticmethod
    def _digit_foreground_mask(sticker_chw: torch.Tensor) -> torch.Tensor:
        """
        Isolates the digit stroke from its LOCAL background using Otsu
        thresholding, instead of assuming the background is black.

        Otsu automatically finds the threshold that best splits the crop
        into two dominant clusters (digit vs. background) — this works
        whether the background is black, white, or a bright team color,
        unlike a fixed 'near-black = background' heuristic (which is what
        caused solid black rectangles to get pasted whenever a sticker's
        source photo had a dark background).

        Returns a (H, W) tensor: 1 = digit stroke (foreground), 0 = background.
        """
        img_np = (sticker_chw.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu just gives two classes (0/255) — we still need to know WHICH
        # class is the background. Assumption: a tight digit crop's outer
        # border ring is mostly background (digit is roughly centered), so
        # whichever class dominates the border is background.
        border = np.concatenate(
            [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
        )
        border_is_white = border.mean() > 127
        foreground = (binary < 127) if border_is_white else (binary >= 127)

        return torch.from_numpy(foreground.astype(np.float32))

    @staticmethod
    def _feathered_paste(image, sticker, mask, paste_y, paste_x, h_paste, w_paste, feather=3):
        """Blend the sticker into `image` using the REAL digit-shaped mask
        (computed before rotation, then carried through the same resize +
        rotate as the sticker pixels) instead of a crude 'non-black' guess.
        This means the sticker's own background — whatever color it is —
        never gets pasted, only the digit stroke itself."""
        sticker_region = sticker[:, :h_paste, :w_paste]
        alpha = mask[:h_paste, :w_paste].clone()

        if feather > 0:
            alpha = TF.gaussian_blur(alpha.unsqueeze(0), kernel_size=2 * feather + 1).squeeze(0)

        region = image[:, paste_y:paste_y + h_paste, paste_x:paste_x + w_paste]
        blended = sticker_region * alpha + region * (1 - alpha)
        image[:, paste_y:paste_y + h_paste, paste_x:paste_x + w_paste] = blended
        return image

    def __getitem__(self, index: int):
        image_path = self.images_files[index]
        label_path = self.labels_files[index]
        label = np.loadtxt(label_path).reshape(-1, 5)

        raw_box_count = label.shape[0]
        if label[0][0] == 10:
            state = 0
            jerseyNumber_len = 0
        elif raw_box_count >= 3:
            state = 2
            jerseyNumber_len = raw_box_count
        else:
            state = 1
            jerseyNumber_len = raw_box_count

        center_x = label[0][1]
        center_y = label[0][2]
        new_width = label[0][3]
        new_heigh = label[0][4]

        digit_number = [int(l[0]) for l in label]

        if raw_box_count > 1 and label[0][0] != 10:
            if raw_box_count == 2 and (
                abs(label[0][1] - label[1][1]) > 0.25
                or abs(label[0][2] - label[1][2]) > 0.25
            ):
                digit_number = [int(label[0][0])]
                jerseyNumber_len = 1
                state = 1
            else:
                min_x = min(l[1] - l[3] / 2 for l in label)
                max_x = max(l[1] + l[3] / 2 for l in label)
                min_y = min(l[2] - l[4] / 2 for l in label)
                max_y = max(l[2] + l[4] / 2 for l in label)

                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                new_width = max_x - min_x
                new_heigh = max_y - min_y

                sorted_indices = np.argsort(label[:, 1])
                digit_number = [int(label[i, 0]) for i in sorted_indices]

        digital = self.IGNORE_INDEX  # default; overwritten below if state==1

        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert("RGB")
        image = Transforms.ToTensor()(image)

        _, h_factor, w_factor = image.shape
        image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = image.shape

        x1 = w_factor * (center_x - new_width / 2)
        y1 = h_factor * (center_y - new_heigh / 2)
        x2 = w_factor * (center_x + new_width / 2)
        y2 = h_factor * (center_y + new_heigh / 2)

        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]

        sticker = None
        x2_original = x2
        safe_gap = 0
        target_h = 0

        # ------------------------------------------------------------
        # CHANGED: only paste when BOTH sides are single-digit real
        # photos —
        #   - destination: state==1 AND jerseyNumber_len==1 (this
        #     specific sample is a single digit, not already 2 digits)
        #   - source: sampled only from self.single_digit_indices, so
        #     the sticker itself is guaranteed to carry exactly one
        #     digit, never two
        # Already-two-digit samples are left completely untouched.
        # ------------------------------------------------------------
        eligible_for_paste = (
            state == 1
            and jerseyNumber_len == 1
            and self.single_digit_pairs_only
            and len(self.single_digit_indices) > 0
        )

        if eligible_for_paste and random.random() < 0.5:
            random_idx = random.choice(self.single_digit_indices)
            sticker, sticker_mask, sticker_digits = self._get_tight_digit_sticker(random_idx)

            # defensive check: _get_tight_digit_sticker can still return
            # None (e.g. missing image file, degenerate box) even for an
            # index we pre-classified as single-digit
            if sticker is not None and len(sticker_digits) != 1:
                sticker = None

            if sticker is not None:
                target_h = int(label[0, 4] * h_factor)  # int(y2 - y1)

                if target_h > 0 and sticker.shape[1] > 0:
                    w_sticker = int(sticker.shape[2] * (target_h / sticker.shape[1]))

                    if w_sticker > 0:
                        sticker = Transforms.functional.resize(
                            sticker, (target_h, w_sticker), antialias=True
                        )
                        # mask travels through the SAME resize as the pixels,
                        # so it stays perfectly aligned with the digit shape.
                        # nearest-neighbor keeps it a clean 0/1 mask (no blur
                        # from bilinear interpolation at this stage).
                        sticker_mask = Transforms.functional.resize(
                            sticker_mask.unsqueeze(0), (target_h, w_sticker),
                            interpolation=Transforms.InterpolationMode.NEAREST,
                        ).squeeze(0)

                        # ------------------------------------------------
                        # PATCH: rotate the sticker to a plausible tilt
                        # instead of pasting it dead-level. Angle is drawn
                        # from the REAL measured distribution for this
                        # dataset (self.tilt_mean / self.tilt_std), not a
                        # fixed guess.
                        # ------------------------------------------------
                        tilt_angle = float(np.random.normal(self.tilt_mean, self.tilt_std))
                        sticker = TF.rotate(
                            sticker,
                            tilt_angle,
                            expand=True,
                            interpolation=Transforms.InterpolationMode.BILINEAR,
                            fill=0,
                        )
                        # rotate the mask by the exact same angle — fill=0
                        # means anything in the newly-exposed corner area
                        # (introduced by expand=True) is correctly marked
                        # background, with zero extra logic needed.
                        sticker_mask = TF.rotate(
                            sticker_mask.unsqueeze(0),
                            tilt_angle,
                            expand=True,
                            interpolation=Transforms.InterpolationMode.NEAREST,
                            fill=0,
                        ).squeeze(0)
                        # expand=True changes the sticker's shape — recompute
                        target_h, w_sticker = sticker.shape[1], sticker.shape[2]
                        # ------------------------------------------------

                        safe_gap = target_h * 0.1

                        x2 += safe_gap + w_sticker

                        new_total_len = jerseyNumber_len + len(sticker_digits)

                        if new_total_len <= 2:
                            state = 1
                            digit_number.extend(sticker_digits)
                            jerseyNumber_len = new_total_len
                        else:
                            state = 2
                            jerseyNumber_len = new_total_len
                            digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
                    else:
                        sticker = None
                else:
                    sticker = None

        label[0, 1] = ((x1 + x2) / 2) / padded_w
        label[0, 2] = ((y1 + y2) / 2) / padded_h
        label[0, 3] = abs(x2 - x1) / padded_w
        if label[0][0] != 10:
            label[0, 4] *= h_factor / padded_h

        if label[0][0] != 10:
            label[0, 3] *= 3
            label[0, 4] *= 1.7

        x_min = label[0, 1] * padded_w - label[0, 3] * padded_w / 2.0
        y_min = label[0, 2] * padded_h - label[0, 4] * padded_h / 2.0
        x_max = label[0, 1] * padded_w + label[0, 3] * padded_w / 2.0
        y_max = label[0, 2] * padded_h + label[0, 4] * padded_h / 2.0

        x_min -= 23
        y_min -= 30
        x_max += 30
        y_max += 33

        # ------------------------------------------------------------
        # FIX: clamp to the REAL photo content, not the padded square
        # canvas. pad_to_square() adds black filler on two sides to make
        # a non-square photo square; the box-widening above (*3 width,
        # *1.7 height, plus the fixed pixel offsets) can push the crop
        # bounds out past the real photo and into that black filler,
        # which is exactly the solid black bars visible on left/right.
        # pad = (left, right, top, bottom) added around the real image.
        # ------------------------------------------------------------
        content_x_min, content_x_max = pad[0], padded_w - pad[1]
        content_y_min, content_y_max = pad[2], padded_h - pad[3]

        if x_min < content_x_min:
            x_min = content_x_min
        if y_min < content_y_min:
            y_min = content_y_min
        if x_max > content_x_max:
            x_max = content_x_max
        if y_max > content_y_max:
            y_max = content_y_max
        # ------------------------------------------------------------

        image = image[:, int(y_min):int(y_max), int(x_min):int(x_max)]

        if sticker is not None:
            paste_y = int(y1 - y_min - target_h * 0.1)
            paste_y = max(0, paste_y)
            paste_x = int(x2_original + safe_gap - x_min)
            paste_x = max(0, paste_x)

            h_paste, w_paste = sticker.shape[1], sticker.shape[2]
            max_h = image.shape[1] - paste_y
            max_w = image.shape[2] - paste_x

            if max_h > 0 and max_w > 0:
                h_paste = min(h_paste, max_h)
                w_paste = min(w_paste, max_w)
                # PATCH: feathered blend using the real digit-shaped mask
                # (Otsu-derived, resized/rotated alongside the sticker
                # pixels) instead of a crude 'non-black' guess.
                image = self._feathered_paste(
                    image, sticker, sticker_mask, paste_y, paste_x, h_paste, w_paste
                )

        if state == 0 or state == 2:
            digital = self.IGNORE_INDEX
            digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
        else:
            if len(digit_number) < self.max_jerseyNumberLength:
                for _ in range(self.max_jerseyNumberLength - len(digit_number)):
                    digit_number.append(10)
            digit_number = digit_number[: self.max_jerseyNumberLength]

            if jerseyNumber_len == 1:
                digital = digit_number[0]
            else:
                digital = digit_number[0] * 10 + digit_number[1]

        if self.transforms is not None:
            image = Transforms.functional.to_pil_image(image)
            if np.random.random() < 0.5:
                image = draw_grid(image, (5, 5))
            image = self.transforms(image)

        return image, digital, digit_number, jerseyNumber_len, state

    def __len__(self):
        return len(self.images_files[:])

    def collate_fn(self, batch):
        images, digital, digit_number, jerseyNumber_len, state = list(zip(*batch))
        images = torch.stack(images, dim=0)
        digital = torch.tensor(digital, dtype=torch.int64)
        digit_number = torch.tensor(digit_number, dtype=torch.int64)
        jerseyNumber_len = torch.tensor(jerseyNumber_len, dtype=torch.int64)
        state = torch.tensor(state, dtype=torch.int64)
        return images, digital, digit_number, jerseyNumber_len, state


# ---------------------------------------------------------------------------
# CLI — actually generate a synthetic batch to disk
# ---------------------------------------------------------------------------

def _write_yolo_label(path: str, state: int, digit_number: List[int]):
    """Writes a label file in the same 5-column YOLO convention used
    elsewhere in this project, so check_balance.py / dataset_audit.py /
    class_counts.py can all audit the generated data without modification.

    NOTE: state==2 (ambiguous) must be written as 3+ raw boxes, not a
    single class-10 row — otherwise downstream audit scripts (which infer
    state purely from raw_box_count / class_id, same as this file's own
    logic) would misread it as state 0 ("no number visible") instead of
    state 2 ("ambiguous"). Caught by testing this script's own output
    through class_counts.py before shipping it.
    """
    if state == 0:
        rows = [[10, 0.5, 0.5, 1.0, 1.0]]
    elif state == 2:
        # 3 dummy boxes -> raw_box_count >= 3 -> correctly read back as
        # state 2 (ambiguous) by check_balance.py / dataset_audit.py / class_counts.py
        rows = [[0, 0.3, 0.5, 0.2, 0.3], [0, 0.5, 0.5, 0.2, 0.3], [0, 0.7, 0.5, 0.2, 0.3]]
    else:
        valid_digits = [d for d in digit_number if d != -100]
        if len(valid_digits) == 1:
            rows = [[valid_digits[0], 0.5, 0.5, 1.0, 1.0]]
        else:
            # two boxes close together in x (diff < 0.25) so downstream
            # audit scripts read them as one combined two-digit number
            rows = [
                [valid_digits[0], 0.35, 0.5, 0.3, 1.0],
                [valid_digits[1], 0.55, 0.5, 0.3, 1.0],
            ]
    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def generate_dataset(root: str, outdir: str, n_samples: int,
                      tilt_mean: float, tilt_std: float, seed: int = 42,
                      single_digit_pairs_only: bool = True):
    os.makedirs(outdir, exist_ok=True)

    dataset = jerseyWithState_Dataset(
        root, transform=None, tilt_mean=tilt_mean, tilt_std=tilt_std,
        single_digit_pairs_only=single_digit_pairs_only,
    )
    n = len(dataset)
    print(f"\nSource dataset has {n} samples. Generating {n_samples} synthetic outputs...")

    rng = random.Random(seed)

    # When restricted to single-digit pairs, draw destination indices from
    # the single-digit pool too — otherwise most randomly-picked indices
    # would be multi-digit/ambiguous/absent samples that never trigger the
    # paste logic at all, and you'd mostly get plain pass-through copies.
    if single_digit_pairs_only and dataset.single_digit_indices:
        pool = dataset.single_digit_indices
    else:
        pool = list(range(n))

    indices = [rng.choice(pool) for _ in range(n_samples)]

    meta = []
    saved = 0
    pasted_count = 0
    for i, idx in enumerate(indices):
        result = dataset[idx]
        if result is None:
            continue
        image_tensor, digital, digit_number, jerseyNumber_len, state = result

        img_pil = Transforms.functional.to_pil_image(image_tensor.clamp(0, 1))
        out_name = f"synth_{i:06d}"
        img_path = os.path.join(outdir, out_name + ".jpg")
        lbl_path = os.path.join(outdir, out_name + ".txt")

        img_pil.save(img_path, quality=95)
        _write_yolo_label(lbl_path, state, digit_number)

        was_pasted = jerseyNumber_len == 2  # since destination was always single-digit here
        if single_digit_pairs_only and was_pasted:
            pasted_count += 1

        meta.append({
            "file": out_name, "source_index": idx, "state": state,
            "digital": digital, "digit_number": digit_number,
            "jersey_number_len": jerseyNumber_len, "was_pasted": was_pasted,
        })
        saved += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} processed...")

    with open(os.path.join(outdir, "generation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if single_digit_pairs_only:
        print(f"\n{pasted_count}/{saved} outputs actually got a sticker pasted "
              f"(the rest hit the 50% skip coin-flip and stayed single-digit).")

    print(f"\nDone. Saved {saved} image/label pairs to: {outdir}/")
    print(f"Metadata (source index, digits, state per file): {outdir}/generation_metadata.json")
    print(
        "\nYou can now audit this synthetic-only folder the same way as the "
        f"real dataset, e.g.:\n  python dataset_audit.py --root {outdir} --top-n 5"
    )


SCRIPT_VERSION = "v3-otsu-mask-single-digit-only"


def main():
    print(f"=== jersey_dataset_patched.py running: {SCRIPT_VERSION} ===")
    print("(if this line doesn't print, you are running an old/different copy of this file)\n")
    parser = argparse.ArgumentParser(description="Patched jersey sticker-augmentation: measure real tilt, then generate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_measure = sub.add_parser("measure", help="Measure real tilt distribution from your dataset")
    p_measure.add_argument("--root", required=True)
    p_measure.add_argument("--sample-size", type=int, default=500)

    p_generate = sub.add_parser("generate", help="Generate a synthetic batch to disk")
    p_generate.add_argument("--root", required=True)
    p_generate.add_argument("--outdir", required=True)
    p_generate.add_argument("--n-samples", type=int, default=500)
    p_generate.add_argument("--tilt-mean", type=float, default=None,
                             help="If omitted, measured automatically from --root first")
    p_generate.add_argument("--tilt-std", type=float, default=None,
                             help="If omitted, measured automatically from --root first")
    p_generate.add_argument("--seed", type=int, default=42)
    p_generate.add_argument(
        "--single-digit-only", dest="single_digit_only", action="store_true", default=True,
        help="Only paste single-digit stickers onto single-digit destination images "
             "(default: on). Already-two-digit samples are never touched.",
    )
    p_generate.add_argument(
        "--allow-any-source", dest="single_digit_only", action="store_false",
        help="Disable the single-digit-only restriction (old behavior: sticker can "
             "come from/be pasted onto any state==1 sample, 1 or 2 digits).",
    )

    args = parser.parse_args()

    if args.command == "measure":
        measure_tilt_stats(args.root, sample_size=args.sample_size)

    elif args.command == "generate":
        tilt_mean, tilt_std = args.tilt_mean, args.tilt_std
        if tilt_mean is None or tilt_std is None:
            print("No --tilt-mean/--tilt-std supplied — measuring from --root first...")
            tilt_mean, tilt_std, _ = measure_tilt_stats(args.root)
        generate_dataset(
            args.root, args.outdir, args.n_samples, tilt_mean, tilt_std,
            seed=args.seed, single_digit_pairs_only=args.single_digit_only,
        )


if __name__ == "__main__":
    main()
