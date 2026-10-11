# =====================================================================
# Validation augmentation (12 versions per image) - all changes
# Files: utils/jersey_dataset.py  and  jerseyNumberRecognitiorWithState_Training.py
# Labels are NOT changed: these transforms keep the digits the same.
# =====================================================================


# =====================================================================
# FILE: utils/jersey_dataset.py
# =====================================================================

# ---------------------------------------------------------------------
# CHANGE 1 (ADD) - top of file, outside any class, after main_number_group
#                  (or after body_like_box), starting at the left edge
# ---------------------------------------------------------------------
# ---------------HCL_Changes---------------
# validation augmentations (same transform types / ranges as training)
VAL_FILL = 128
VAL_AUG = [
    None,                                                                    # 0  clean
    Transforms.ColorJitter(brightness=(1.15, 1.3)),                          # 1  brighter
    Transforms.ColorJitter(brightness=(0.7, 0.85)),                          # 2  darker
    Transforms.ColorJitter(contrast=(1.15, 1.3)),                            # 3  more contrast
    Transforms.ColorJitter(contrast=(0.7, 0.85)),                            # 4  less contrast
    Transforms.RandomRotation(degrees=(8, 20), fill=VAL_FILL),               # 5  rotate +
    Transforms.RandomRotation(degrees=(-20, -8), fill=VAL_FILL),             # 6  rotate -
    Transforms.RandomPerspective(distortion_scale=0.4, p=1.0, fill=VAL_FILL),# 7  perspective
    Transforms.ElasticTransform(alpha=35.0, sigma=5.0, fill=VAL_FILL),       # 8  elastic
    Transforms.GaussianBlur(kernel_size=3, sigma=(0.6, 1.2)),                # 9  blur
    Transforms.Compose([                                                     # 10 combined
        Transforms.RandomApply([Transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=VAL_FILL)], p=0.5),
        Transforms.RandomApply([Transforms.RandomRotation(degrees=(-20, 20), fill=VAL_FILL)], p=0.4),
        Transforms.RandomApply([Transforms.ElasticTransform(alpha=35.0, sigma=5.0, fill=VAL_FILL)], p=0.6),
        Transforms.RandomApply([Transforms.RandomPerspective(distortion_scale=0.4, p=1.0, fill=VAL_FILL)], p=0.4),
        Transforms.RandomApply([Transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.5),
        Transforms.RandomApply([Transforms.GaussianBlur(kernel_size=3)], p=0.3),
    ]),
    Transforms.Compose([                                                     # 11 rotate + colour + blur
        Transforms.RandomRotation(degrees=(-15, 15), fill=VAL_FILL),
        Transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        Transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.8)),
    ]),
]
# -------------------------------------------


class jerseyNumber_ValidationDataset_V2(VisionDataset):

    # -----------------------------------------------------------------
    # CHANGE 2 (REPLACE) - first lines of __init__
    # -----------------------------------------------------------------
    # ---------------HCL_Changes---------------
    # def __init__(self, root):
    #     super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
    def __init__(self, root, n_aug=1):          # n_aug = versions per image (1 = clean only)
        super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
        self.n_aug = n_aug
        # -------------------------------------------
        # ... rest of __init__ unchanged (images_path, labels_path, 3+ skip) ...

    def __getitem__(self, index):
        # -------------------------------------------------------------
        # CHANGE 3a (REPLACE) - first lines of __getitem__
        # -------------------------------------------------------------
        # ---------------HCL_Changes---------------
        # image_path = self.images_path[index]
        # label_path = self.labels_path[index]
        img_idx, version = divmod(index, self.n_aug)   # index -> (image, version)
        image_path = self.images_path[img_idx]
        label_path = self.labels_path[img_idx]
        # -------------------------------------------

        # ... label loading unchanged ...

        image = cv2.imread(image_path)
        # (keep your BGR->RGB line here if you added one)

        # -------------------------------------------------------------
        # CHANGE 3b (REPLACE) - the 2 lines after cv2.imread
        # -------------------------------------------------------------
        # ---------------HCL_Changes---------------
        # image = image.astype("float32") / 255.0
        # resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        if version > 0:                                 # fixed seed -> same image every epoch
            random.seed(1000 * img_idx + version)
            torch.manual_seed(1000 * img_idx + version)
            image = np.array(VAL_AUG[version](Image.fromarray(image)))
        resized_image = image.astype("float32") / 255.0
        # -------------------------------------------
        resized_image = resized_image.transpose(2, 0, 1)   # unchanged
        return resized_image, individual_digits_label, whole_number_label   # unchanged

    # -----------------------------------------------------------------
    # CHANGE 4 (REPLACE) - __len__
    # -----------------------------------------------------------------
    def __len__(self):
        # ---------------HCL_Changes---------------
        # return len(self.images_path[:])
        return len(self.images_path) * self.n_aug
        # -------------------------------------------


# =====================================================================
# FILE: jerseyNumberRecognitiorWithState_Training.py
# =====================================================================

# ---------------------------------------------------------------------
# CHANGE 5 (REPLACE) - validation dataset + loader (in __main__)
# ---------------------------------------------------------------------
    # ---------------HCL_Changes---------------
    # val_dataset = jerseyNumber_ValidationDataset_V2(
    #         validation_path)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
    #                         num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)
    val_dataset = jerseyNumber_ValidationDataset_V2(validation_path, n_aug=12)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)
    # -------------------------------------------


# =====================================================================
# CHECK
#   python -m py_compile utils/jersey_dataset.py jerseyNumberRecognitiorWithState_Training.py
#   log: validation size = clean images x 12
# =====================================================================
