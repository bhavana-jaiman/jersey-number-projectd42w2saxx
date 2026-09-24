import os
import random
from tqdm import tqdm
from typing import List, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, Counter

from utils.util import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as Transforms
from torch.utils.data import Dataset

import cv2
import glob


class JerseyWithState_Dataset(VisionDataset):
    # ---------------HCL_Changes---------------
    # upper_body param added to match the __main__ call site
    # (train_dataset = JerseyWithState_Dataset(..., upper_body=True)).
    # JerseyDataset already declares this same parameter (stored but unused,
    # commented out); mirrored here for consistency. Not otherwise used yet.
    def __init__(self, root, transform=None, upper_body: bool = False) -> None:
        super(JerseyWithState_Dataset, self).__init__(root, transforms=transform)
        # self.upper_body = upper_body
        # -------------------------------------------

        self.max_jerseyNumberLength = 2

        self.IGNORE_INDEX = -100

        self.images_path = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jpg")]

        self.labels_path = [
            os.path.join(root, f).replace(".jpg", ".txt")
            for f in os.listdir(root)
            if f.endswith(".jpg")]

        self.images_files = np.array(self.images_path)
        self.labels_files = np.array(self.labels_path)

        # self.labels_list = [np.loadtxt(path).reshape(-1, 5)[:, 0]
        #                      for path in self.labels_files]

    def _get_tight_digit_sticker(self, index):
        label_path = self.labels_files[index]
        label = np.loadtxt(label_path).reshape(-1, 5)
        if label.ndim == 1:
            label = label.reshape(1, -1)
        if label.size == 0 or label[0][0] == 10:
            return None, None

        sorted_indices = np.argsort(label[:, 1])
        sticker_digits = [int(label[i, 0]) for i in sorted_indices]

        image_path = self.images_files[index]
        if not os.path.exists(image_path):
            return None, None

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
        margin_y = box_h * 0.10

        min_x, max_x = max(0, int(min_x - margin_x)), min(w, int(max_x + margin_x))
        min_y, max_y = max(0, int(min_y - margin_y)), min(h, int(max_y + margin_y))

        if max_x <= min_x or max_y <= min_y:
            return None, None

        return image[:, min_y:max_y, min_x:max_x], sticker_digits

    def __getitem__(self, index):
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

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image = Transforms.functional.to_tensor(image)

            _, h_factor, w_factor = image.shape
            image, pad = pad_to_square(image, 0)
            _, padded_h, padded_w = image.shape

            x1 = w_factor * (center_x - new_width / 2)
            y1 = h_factor * (center_y - new_heigh / 2)
            x2 = w_factor * (center_x + new_width / 2)
            y2 = h_factor * (center_y + new_heigh / 2)

            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[0]
            y2 += pad[2]

            sticker = None
            x_original = x2
            if state == 1 and random.random() < 0.5:
                random_idx = random.randint(0, len(self.images_files) - 1)
                sticker, sticker_digits = self._get_tight_digit_sticker(random_idx)
                if sticker is not None:
                    target_h = int(label[0, 4] * h_factor)  # int(y2-y1)
                    w_sticker = int(
                        sticker.shape[2] * (target_h / sticker.shape[1])
                    )
                    if target_h > 0 and w_sticker > 0:
                        sticker = Transforms.functional.resize(
                            sticker, (target_h, w_sticker), antialias=True
                        )
                        safe_gap = target_h * 0.1 * random.uniform(0.1, 0.15)  # placeholder
                        x2 = safe_gap + w_sticker

                        new_total_len = jerseyNumber_len + len(sticker_digits)

                        if new_total_len <= 2:
                            state = 1
                            digit_number.extend(sticker_digits)
                            jerseyNumber_len = new_total_len
                        else:
                            state = 2
                            jerseyNumber_len = new_total_len
                            # ---------------HCL_Changes---------------
                            # digital head removed from model; drop the combined 'digital' value
                            # digital = self.IGNORE_INDEX
                            digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
                            # -------------------------------------------
                else:
                    sticker = None
            else:
                sticker = None

            label[0, 1] = ((x1 + x2) / 2) / padded_w
            label[0, 2] = ((y1 + y2) / 2) / padded_h
            label[0, 3] = abs(x2 - x1) / padded_w
            if label[0][0] == 10:
                label[0, 4] = h_factor / padded_h  # (y2 - y1) / padded_h

            if label[0][3] == 10:
                label[0, 3] = 10
                label[0, 4] = 1.7

            x_min = label[0, 1] * padded_w - label[0, 3] * padded_w / 2.0
            y_min = label[0, 2] * padded_h - label[0, 4] * padded_h / 2.0
            x_max = label[0, 1] * padded_w + label[0, 3] * padded_w / 2.0
            y_max = label[0, 2] * padded_h + label[0, 4] * padded_h / 2.0

            x_min -= 23
            y_min -= 30
            x_max += 30
            y_max += 33

            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > padded_w:
                x_max = padded_w
            if y_max > padded_h:
                y_max = padded_h

            image = image[:, int(y_min):int(y_max), int(x_min):int(x_max)]

            if sticker is not None:
                paste_y = int(y1 - y_min - target_h * 0.1)
                paste_x = int(x2_original + safe_gap - x_min)
                paste_y = max(0, paste_y)
                paste_x = max(0, paste_x)

                h_paste, w_paste = sticker.shape[1], sticker.shape[2]
                max_h = image.shape[1] - paste_y
                max_w = image.shape[2] - paste_x
                if max_h > 0 and max_w > 0:
                    h_paste = min(h_paste, max_h)
                    w_paste = min(w_paste, max_w)
                    image[
                        :, paste_y:paste_y + h_paste, paste_x:paste_x + w_paste
                    ] = sticker[:, :h_paste, :w_paste]

            # ---------------HCL_Changes---------------
            # digital head removed from model output; combined 'digital' value
            # and its old 11-class/100-sentinel logic are gone.
            #
            # New state semantics (classifier_state, 3 classes):
            #   state 0 -> no digit visible
            #   state 1 -> one or two digits visible (normal case, feeds digit1/digit2)
            #   state 2 -> three digits visible (unsupported by the 2-digit head)
            #
            # digit1 is now 0-9 only (10 classes, no blank/ignore class).
            # digit2 stays 0-10 (11 classes; 10 is the pad/blank value).
            # A lone visible digit is always index 0 in digit_number (there is
            # nothing to compare it against), so it is always digit1; digit2 is
            # padded with the blank value 10 in that case -- this matches the
            # existing sort-by-x-position logic above for the 2-digit case.
            #
            # if state == 0 or state == 2:
            #     digital = self.IGNORE_INDEX
            #     digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
            # else:
            if state == 2:
                # three-or-more digits: unsupported by the 2-digit head, mask
                # both digit1 and digit2 out of the loss for this sample
                digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
            elif state == 0:
                # no digit visible: nothing for digit1/digit2 to predict
                digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
            else:
                # state == 1: one or two digits, pad digit2 with blank (10) if
                # only a single digit (digit1) is present
                if len(digit_number) < self.max_jerseyNumberLength:
                    for _ in range(self.max_jerseyNumberLength - len(digit_number)):
                        digit_number.append(10)

            digit_number = digit_number[:self.max_jerseyNumberLength]

            # if jerseyNumber_len == 1:
            #     digital = digit_number[0]
            # else:
            #     digital = digit_number[0] * 10 + digit_number[1]
            # -------------------------------------------

            if self.transforms is not None:
                image = Transforms.functional.to_pil_image(image)

                if np.random.random() < 0.2 and label[0][0] == 10:
                    image = Transforms.GaussianBlur(image, (3, 3))
                # image = np.asarray(image)
                # image = self.album(image=image)["image"]
                # image = Transforms.functional.to_pil_image(image)

                if np.random.random() < 0.5:
                    image = draw_grid(image, (5, 5))

                image = self.transforms(image)
                # plt.imshow(image.permute(1, 2, 0))
                # plt.show()
                # print(digit_number)
                # print(digital)
                # print(jerseyNumber_len)
                # print(image_path)
        # ---------------HCL_Changes---------------
        # 'digital' no longer returned; model no longer has a whole-digit head
        # return image, digital, digit_number, jerseyNumber_len, state
        return image, digit_number, jerseyNumber_len, state
        # -------------------------------------------

    def __len__(self):
        return len(self.images_files[:])

    def collate_fn(self, batch):
        # ---------------HCL_Changes---------------
        # images, digital, digit_number, jerseyNumber_len, state = list(zip(*batch))
        images, digit_number, jerseyNumber_len, state = list(zip(*batch))
        # -------------------------------------------
        images = torch.stack(images, dim=0)

        # digital = torch.tensor(digital, dtype=torch.int64)
        digit_number = torch.tensor(digit_number, dtype=torch.int64)
        jerseyNumber_len = torch.tensor(jerseyNumber_len, dtype=torch.int64)
        state = torch.tensor(state, dtype=torch.int64)

        # ---------------HCL_Changes---------------
        # return images, digital, digit_number, jerseyNumber_len, state
        return images, digit_number, jerseyNumber_len, state
        # -------------------------------------------


class JerseyWithState_ValidationDataset(JerseyWithState_Dataset):
    def __init__(self, root, transform=None):
        # NOTE: pre-existing bug from original source -- parent __init__ takes
        # keyword 'transform', not 'transforms'. Left as originally written;
        # will raise a TypeError as-is. Flagging rather than silently fixing.
        super().__init__(root, transforms=transform)

    def __getitem__(self, index):
        image_path = self.images_files[index]
        label_path = self.labels_files[index]

        try:
            label = np.loadtxt(label_path, delimiter=" ", dtype=np.float32).reshape(
                -1, 1
            )
        except Exception:
            label = np.array([[10]])

        if label.size == 0:
            label = np.array([[10]])

        raw_count = label.shape[0]
        if label[0][0] == 10:
            state = 0
            jerseyNumber_len = 0
        elif raw_count >= 3:
            state = 2
            jerseyNumber_len = raw_count
        else:
            state = 1
            jerseyNumber_len = raw_count

        digit_number = [int(l[0]) for l in label]

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image = Transforms.functional.resize(image, (96, 96), Image.BILINEAR)
            image_tensor = Transforms.functional.to_tensor(image)

            # ---------------HCL_Changes---------------
            # digital head removed from model output; combined value no longer needed.
            # Same state semantics as JerseyWithState_Dataset:
            #   state 0 -> no digit, state 1 -> one/two digits, state 2 -> three digits
            # if state == 0 or state == 2:
            #     digital = self.IGNORE_INDEX
            #     digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
            # else:
            if state == 2 or state == 0:
                digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
            else:
                if len(digit_number) < self.max_jerseyNumberLength:
                    for _ in range(self.max_jerseyNumberLength - len(digit_number)):
                        digit_number.append(10)

            digit_number = digit_number[:self.max_jerseyNumberLength]

            # if jerseyNumber_len == 1:
            #     digital = digit_number[0]
            # else:
            #     digital = digit_number[0] * 10 + digit_number[1]
            # -------------------------------------------

        # ---------------HCL_Changes---------------
        # return image_tensor, digital, digit_number, jerseyNumber_len, state
        return image_tensor, digit_number, jerseyNumber_len, state
        # -------------------------------------------


class JerseyDataset(VisionDataset):
    def __init__(self, root, transform=None) -> None:  # 8, upper_bodyFalse
        super(JerseyDataset, self).__init__(root, transforms=transform)

        self.max_jerseyNumberLength = 2
        # self.upper_body = upper_body

        self.images_path = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jpg")]

        self.labels_path = [
            os.path.join(root, f).replace(".jpg", ".txt")
            for f in os.listdir(root)
            if f.endswith(".jpg")]

        self.images_files = np.array(self.images_path)
        self.labels_files = np.array(self.labels_path)

        self.labels_list = [np.loadtxt(path).reshape(-1, 5)[:, 0]
                             for path in self.labels_files]

        # self.album = A.Compose([
        #     A.ElasticTransform(alpha=100),
        #     A.Affine(rotate=(-10, 10), shear=(-10, 10)),
        #     A.Erasing(scale=(0.02, 0.05), ratio=(0.3, 3.3), p=0.3),
        # ])

    def __getitem__(self, index):
        image_path = self.images_files[index]
        label_path = self.labels_files[index]
        label = np.loadtxt(label_path).reshape(-1, 5)

        center_x = label[0][1]
        center_y = label[0][2]
        new_width = label[0][3]
        new_heigh = label[0][4]

        digit_number = [i for i in self.labels_list[index][:2]]

        if label.shape[0] > 1 and label[0][0] != 10:
            if (
                abs(label[0][1] - label[1][1]) > 0.25
                or abs(label[0][2] - label[1][2]) > 0.25
            ):
                digit_number = []
                digit_number.append(label[0][0])
            else:
                center_x = (label[0, 1] + label[1, 1]) / 2
                center_y = (label[0, 2] + label[1, 2]) / 2
                new_width = (label[0, 3] + label[1, 3]) / 2
                new_heigh = (label[0, 4] + label[1, 4]) / 2 + abs(label[0, 2] - label[1, 2])
                if label[0, 2] > label[1, 2]:
                    digit_number = [digit_number[1], digit_number[0]]

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")  # np.array
            image = Transforms.ToTensor()(image)

            _, h_factor, w_factor = image.shape

            image, pad = pad_to_square(image, 128)  # 0

            _, padded_h, padded_w = image.shape

            # extract coordinates for image
            x1 = w_factor * (center_x - new_width / 2)
            y1 = h_factor * (center_y - new_heigh / 2)
            x2 = w_factor * (center_x + new_width / 2)
            y2 = h_factor * (center_y + new_heigh / 2)

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[0]
            y2 += pad[2]

            label[0, 1] = ((x1 + x2) / 2) / padded_w
            label[0, 2] = ((y1 + y2) / 2) / padded_h
            label[0, 3] = w_factor / padded_w
            label[0, 4] = h_factor / padded_h

            if label[0][0] == 10:
                label[0, 3] = 10
                label[0, 4] = 1.7

            x_min = label[0, 1] * padded_w - label[0, 3] * padded_w / 2.0
            y_min = label[0, 2] * padded_h - label[0, 4] * padded_h / 2.0
            x_max = label[0, 1] * padded_w + label[0, 3] * padded_w / 2.0
            y_max = label[0, 2] * padded_h + label[0, 4] * padded_h / 2.0

            x_min -= 23
            y_min -= 30
            x_max += 30
            y_max += 33

            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > padded_w:
                x_max = padded_w
            if y_max > padded_h:
                y_max = padded_h

            image = image[:, int(y_min):int(y_max), int(x_min):int(x_max)]

            jerseyNumber_len = len(digit_number)
            for _ in range(self.max_jerseyNumberLength - jerseyNumber_len):
                digit_number.append(10)

            if label[0][0] == 10:
                # ---------------HCL_Changes---------------
                # digital head removed from model output; combined value no longer needed
                # digital = 100
                pass
                # -------------------------------------------
            elif jerseyNumber_len == 1:
                # digital = digit_number[0]
                pass
            else:
                # digital = digit_number[0] * 10 + digit_number[1]
                pass

            if self.transforms is not None:
                image = Transforms.functional.to_pil_image(image)

                if np.random.random() < 0.2 and label[0][0] == 10:
                    image = Transforms.GaussianBlur(image, (3, 3))
                # image = np.asarray(image)
                # image = self.album(image=image)["image"]
                # image = Transforms.functional.to_pil_image(image)

                if np.random.random() < 0.5:
                    image = draw_grid(image, (5, 5))

                image = self.transforms(image)
                # plt.imshow(image.permute(1, 2, 0))
                # plt.show()
                # print(digit_number)
                # print(digital)
                # print(jerseyNumber_len)
                # print(image_path)
        # ---------------HCL_Changes---------------
        # return image, digital, digit_number, jerseyNumber_len
        return image, digit_number, jerseyNumber_len
        # -------------------------------------------

    def __len__(self):
        return len(self.images_files[:])

    def collate_fn(self, batch):
        # ---------------HCL_Changes---------------
        # images, digital, digit_number, jerseyNumber_len = list(zip(*batch))
        images, digit_number, jerseyNumber_len = list(zip(*batch))
        # -------------------------------------------
        images = torch.stack(images, dim=0)

        # digital = torch.tensor(digital, dtype=torch.int64)
        digit_number = torch.tensor(digit_number, dtype=torch.int64)
        jerseyNumber_len = torch.tensor(jerseyNumber_len, dtype=torch.int64)

        # ---------------HCL_Changes---------------
        # return images, digital, digit_number, jerseyNumber_len
        return images, digit_number, jerseyNumber_len
        # -------------------------------------------

    def checkAndPlot(self):
        digital_label = [i[0] for i in self.labels_list]
        text = [i[0] if len(i) == 1 else i[0] * 10 + i[1] for i in digital_label]
        counts = Counter(text)
        dit = dict(sorted(counts.items()))
        plt.plot(dit.keys(), dit.values())
        plt.xlabel('Jersey Number')
        plt.ylabel('Lines')
        plt.show()

    def countJerseyNumber(self):
        digital_label = [i[0] for i in self.labels_list]
        jerseyNumber = [
            i[0] if len(i) == 1 else i[0] * 10 + i[1] for i in digital_label
        ]
        return len(set(jerseyNumber))


class MultiGroupTwoDigitDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[callable] = None,
        target_size: int = 96,
        blank_token: int = 10,
        is_train: bool = True,
        scale_jitter: Tuple[float, float] = (1.2, 1.6),
        translation_jitter: float = 0.1,
    ):
        super(MultiGroupTwoDigitDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_size = target_size
        self.blank_token = blank_token
        self.is_train = is_train
        self.scale_jitter = scale_jitter
        self.translation_jitter = translation_jitter

        self.samples = self._creat_sample_index(root)

    def _creat_sample_index(self, root: str) -> List[Tuple]:
        sample_index = []
        label_files = [f for f in os.listdir(root) if f.endswith(".txt")]

        for label_file in label_files:
            img_base_name = os.path.splitext(label_file)[0]
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                potential_path = os.path.join(root, img_base_name + ext)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if not img_path:
                continue

            with open(os.path.join(root, label_file), "r") as f:
                all_bboxes = []
                for line in f:
                    parts = line.strip().split()
                    all_bboxes.append([float(p) for p in parts])

            all_bboxes.sort(key=lambda bbox: bbox[1])

            used_indicies = [False] * len(all_bboxes)

            for i in range(len(all_bboxes)):
                if used_indicies[i]:
                    continue

                bbox1 = all_bboxes[i]
                best_partner_j = -1

                for j in range(i + 1, len(all_bboxes)):
                    if used_indicies[j]:
                        continue

                    bbox2 = all_bboxes[j]

                    is_aligned = abs(bbox1[2] - bbox2[2]) < (bbox1[4] * 0.5)
                    is_adjacent = (
                        bbox2[1] - bbox1[1] < (bbox1[3] + bbox2[3])
                    )

                    if is_aligned and is_adjacent:
                        best_partner_j = j
                        break

                if best_partner_j != -1:
                    bbox1 = all_bboxes[i]
                    bbox2 = all_bboxes[best_partner_j]
                    digit1_label = int(bbox1[0])
                    digit2_label = int(bbox2[0])
                    individual_digits_label = (digit1_label, digit2_label)
                    whole_number_label = digit1_label * 10 + digit2_label
                    group_bboxes = [bbox1, bbox2]
                    sample_index.append(
                        (
                            img_path,
                            group_bboxes,
                            individual_digits_label,
                            whole_number_label,
                        )
                    )
                    used_indicies[i] = True
                    used_indicies[best_partner_j] = True
                else:
                    digit1_label = int(bbox1[0])
                    individual_digits_label = (digit1_label, self.blank_token)
                    whole_number_label = digit1_label
                    group_bboxes = [bbox1]
                    sample_index.append(
                        (
                            img_path,
                            group_bboxes,
                            individual_digits_label,
                            whole_number_label,
                        )
                    )
                    used_indicies[i] = True

        return sample_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        img_path, group_bboxes, individual_digits_label, whole_number_label = (
            self.samples[index]
        )

        individual_digits_label_tensor = torch.tensor(
            individual_digits_label, dtype=torch.long
        )
        whole_number_label_tensor = torch.tensor(whole_number_label, dtype=torch.long)

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for bbox in group_bboxes:
            cx, cy, w, h = bbox[1:]
            abs_cx, abs_cy = cx * img_w, cy * img_h
            abs_w, abs_h = w * img_w, h * img_h
            x1, y1 = abs_cx - abs_w / 2, abs_cy - abs_h / 2
            x2, y2 = abs_cx + abs_w / 2, abs_cy + abs_h / 2
            min_x, min_y = min(min_x, x1), min(min_y, y1)
            max_x, max_y = max(max_x, x2), max(max_y, y2)

        union_box = (min_x, min_y, max_x, max_y)

        center_x = union_box["x"] = (union_box[0] + union_box[2]) / 2
        center_y = union_box["y"] = (union_box[1] + union_box[3]) / 2
        max_dim = union_box["w"] = max(union_box[2] - union_box[0], union_box[3] - union_box[1])

        if self.is_train:
            scale = random.uniform(self.scale_jitter[0], self.scale_jitter[1])
            final_size = max_dim * scale
            offset_x = random.uniform(-1, 1) * self.translation_jitter * final_size
            offset_y = random.uniform(-1, 1) * self.translation_jitter * final_size
            center_x += offset_x
            center_y += offset_y
        else:
            final_size = max_dim * self.scale_jitter[0]

        crop_left = int(center_x - final_size / 2)
        crop_top = int(center_y - final_size / 2)

        pad_left = max(0, -crop_left)
        pad_top = max(0, -crop_top)

        image_padded = ImageOps.expand(
            image, border=(pad_left, pad_top, 0, 0), fill="black"
        )

        new_crop_left = crop_left + pad_left
        new_crop_top = crop_top + pad_top

        cropped_image = image_padded.crop(
            (
                new_crop_left,
                new_crop_top,
                new_crop_left + int(final_size),
                new_crop_top + int(final_size),
            )
        )

        final_image = cropped_image.resize(
            (self.target_size, self.target_size), Image.Resampling.LANCZOS
        )

        if self.transform:
            final_image = self.transform(final_image)

        # plt.imshow(final_image.permute(1, 2, 0))
        # plt.show()
        # print(individual_digits_label_tensor, whole_number_label_tensor)

        return final_image, individual_digits_label_tensor, whole_number_label_tensor


class JerseyNumber_ValidationDataset(VisionDataset):
    def __init__(self, root, transform=None, onlyNumber: bool = False) -> None:
        super(JerseyNumber_ValidationDataset, self).__init__(root, transforms=transform)

        self.max_jerseyNumberLength = 2
        self.onlyNumber = onlyNumber

        self.images_path = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".jpg")]

        self.labels_path = [
            os.path.join(root, f).replace(".jpg", ".txt")
            for f in os.listdir(root)
            if f.endswith(".jpg")]

        self.images_files = np.array(self.images_path)
        self.labels_files = np.array(self.labels_path)

        self.labels_list = [np.loadtxt(path).reshape(-1, 5)[:, 0]
                             for path in self.labels_files]

    def __getitem__(self, index: int):
        image_path = self.images_files[index]
        label_path = self.labels_files[index]
        label = np.loadtxt(label_path).reshape(-1, 5)

        # ---------------HCL_Changes---------------
        # digital head removed from model; digit1 is now always 0-9 (no blank/
        # ignore class). 'digital' here is a parsed label value, not the model's
        # removed head output, so left as-is for now -- flag for review once the
        # rest of this method (not visible in the provided screenshots) is supplied.
        digital = label[0][0]
        digit_number = []
        if digital <= 9:
            digit_number.append(digital)
        elif digital > 9 and digital < 100:
            digit_1 = digital // 10
            digit_2 = digital % 10
            digit_number.append(digit_1)
            digit_number.append(digit_2)
        elif digital == 100:
            digit_1 = digital // 10
            digit_2 = digital % 10 % 10
            digit_3 = digital // 10 % 10 % 10
            digit_number.append(digit_1)
            digit_number.append(digit_2)
            # digit_number.append(digit_3)
        # -------------------------------------------

        # INCOMPLETE — the rest of __getitem__, __len__, and collate_fn for
        # this class were not visible in the screenshots provided.


class JerseyNumber_ValidationDataset_V2(VisionDataset):
    def __init__(self, root):
        super(JerseyNumber_ValidationDataset_V2, self).__init__(root)
        self.image_dir = root + "/images"

        self.images_path = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(".jpg")]

        self.labels_path = [
            path.replace("images", "labels").replace(".jpg", ".txt")
            for path in self.images_path]

        print("IMAGE PATH:", self.images_path[:3])
        print("LABEL PATH:", self.labels_path[:3])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label_path = self.labels_path[index]

        # ---------------HCL_Changes---------------
        # Fallback code : If the expected label file does not exist add the extra "_" or check alternative path
        if not os.path.exists(label_path):
            alternative_path = os.path.splitext(image_path)[0] + ".txt"

            if os.path.exists(alternative_path):
                label_path = alternative_path
            else:
                raise FileNotFoundError(
                    f"Label file not found: {label_path}.\n"
                    f"Also checked: {alternative_path}"
                )
        # -------------------------------------------

        individual_digits_label = np.loadtxt(
            label_path, delimiter=" ", dtype=np.float32
        ).reshape(-1, 2)

        # ---------------HCL_Changes---------------
        # Fallback code for labels in simple 2D shape
        if individual_digits_label.shape[0] not in [1, 2]:
            label_data = np.loadtxt(
                label_path, delimiter=" ", dtype=np.float32
            )
            individual_digits_label = np.atleast_2d(label_data)
            individual_digits_label = label_data[:, 0].reshape(-1, 1)
            # print(individual_digits_label.shape)
        # -------------------------------------------

        if (
            individual_digits_label.shape[0] == 2
            and individual_digits_label.shape[1] == 1
        ):
            whole_number_label = (
                individual_digits_label[0] * 10 + individual_digits_label[1]
            )
        else:
            whole_number_label = individual_digits_label[0]
        image = cv2.imread(image_path)

        # ---------------HCL_Changes---------------
        # BGR 2RGB
        # if image is None:
        #     raise FileNotFoundError(f"Unable to read image: {image_path}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('test', resized_image)
        # cv2.waitKey()
        resized_image = cv2.resize(image, (114, 114), interpolation=cv2.INTER_LINEAR)
        top = (114 - 96) // 2
        left = (114 - 96) // 2
        resized_image = resized_image[top:top + 96, left:left + 96]
        resized_image = resized_image.transpose(2, 0, 1)
        return resized_image, individual_digits_label, whole_number_label

    def collate_fn(self, batch):
        images, individual_digits_labels, whole_number_labels = list(zip(*batch))
        images = torch.stack([torch.from_numpy(image) for image in images])

        collated_labels = []
        for label in individual_digits_labels:
            if label.shape[0] == 1 and label.shape[1] == 1:
                padded_label = np.append(
                    label, np.array([[10]], dtype=label.dtype), axis=0
                )
            elif label.shape[0] == 2 and label.shape[1] == 1:
                padded_label = label
            else:
                raise ValueError
            collated_labels.append(padded_label)
        final_collated_label = torch.stack(
            [torch.from_numpy(collated_label) for collated_label in collated_labels],
            dim=0,
        )
        final_collated_whole_label = torch.stack(
            [torch.from_numpy(whole_label) for whole_label in whole_number_labels],
            dim=0,
        )

        return images, final_collated_label, final_collated_whole_label

    def new_collate_fn(self, batch):
        images, individual_digits_labels, whole_number_labels = zip(*batch)
        images = torch.stack([torch.from_numpy(image) for image in images])

        collated_labels = []
        for label in individual_digits_labels:
            t_label = torch.from_numpy(label).view(-1)

            if t_label.size(0) == 1:
                t_label = F.pad(t_label, (0, 1), value=10)
            elif t_label.size(0) != 2:
                raise ValueError(f"Unexpected label shape: {label.shape}")

            collated_labels.append(t_label)
        final_collated_label = torch.stack(collated_labels, dim=0).long()
        final_collated_whole_label = torch.from_numpy(
            np.array(whole_number_labels)
        ).view(-1).long()

        return images, final_collated_label, final_collated_whole_label

    def __len__(self):
        return len(self.images_path[:])


if __name__ == "__main__":
    dataset_path = (
        "/home/ying/Desktop/Dataset/jersey_number_detection/training_dataset_Ying"
    )
    validation_dataset_path = "./datasets/validation_dataset_Ying"  # "./datasets/validation_dataset_Ying"  # "/home/ying/Desktop/Dataset/jersey_number_detection/validationDataset_for_state"  #
    transform_list = []
    transform_list += [
        Transforms.Resize((96, 96)),
        Transforms.RandomApply(
            [
                Transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.6, 1.2),
                    shear=(-10, 10, -10, 10),
                )
            ],
            p=0.8,
        ),
        Transforms.RandomApply(
            [Transforms.RandomPerspective(distortion_scale=0.4, p=1)], p=0.4
        ),
        Transforms.RandomApply(
            [Transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)],
            p=0.4,
        ),
        Transforms.RandomApply(
            [Transforms.GaussianBlur(kernel_size=3)], p=0.3
        ),
        Transforms.ToTensor(),
    ]
    transform = Transforms.Compose(transform_list)

    image_dataset = MultiGroupTwoDigitDataset(
        root=dataset_path, transform=transform, scale_jitter=(1.5, 2.), translation_jitter=0.2
    )
    train_dataset = JerseyWithState_Dataset(
        dataset_path, transform=transform, upper_body=True
    )
    validation_dataset = JerseyWithState_ValidationDataset(validation_dataset_path)

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=64,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    print(image_dataset.checkAndPlot())

    for i in range(0, 100):
        for batch, (images, digit_number_labels, whole_number_labels) in enumerate(validation_loader):
            print("-----")
            print(whole_number_labels)
            label = torch.squeeze(-1).log2()

    mean, std = get_mean_and_std(validation_loader)
    print(mean, std)

    count_digit1 = [0] * 11
    count_digit2 = [0] * 11
    count_whole = [0] * 100

    # ---------------HCL_Changes---------------
    # loop updated to match JerseyWithState_ValidationDataset's new 4-item
    # return (digital removed from the return tuple, digit1 is now 0-9 only,
    # state=0/2 samples carry IGNORE_INDEX in digit_number per the new logic)
    # for images, digital, digit_number, jerseyNumber_len, state in validation_loader:
    for images, digit_number, jerseyNumber_len, state in tqdm(
        validation_loader, desc="Start counting Jersey Number..."
    ):
        for label in digit_number:
            count_digit1[label[0]] += 1
            count_digit2[label[1]] += 1
        whole_number_labels = whole_number_labels.squeeze(-1)
        for whole_number in whole_number_labels:
            count_whole[int(whole_number.item())] += 1
    print(count_digit1, count_digit2)
    # -------------------------------------------
