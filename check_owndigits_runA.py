"""Save training samples so you can look at them before training.
Run from New_arcitecture_2:  python check_owndigits_runA.py
  debug_own_digits/ : 1-2 digit images that were turned into 3 digits (own-digit copy)
  debug_runA/       : 40 random training samples (Run A crops)
"""
import os, random, warnings
import numpy as np
import torchvision.transforms as T
from utils.jersey_dataset_New_changed import JerseyWithState_Dataset

warnings.filterwarnings("ignore")
random.seed(0)
ds = JerseyWithState_Dataset("../Datasets/70_30_dataset/training/", transform=None)

# 1) own-digit copy
os.makedirs("debug_own_digits", exist_ok=True)
saved = 0
for i in random.sample(range(len(ds)), min(3000, len(ds))):
    n_boxes = np.atleast_1d(np.loadtxt(ds.labels_files[i])).size // 5
    img, dn, ln, st = ds[i]
    if n_boxes <= 2 and st == 2:
        T.functional.to_pil_image(img).save(f"debug_own_digits/{i:05d}_orig{n_boxes}digits.jpg")
        saved += 1
        if saved >= 40:
            break
print("own-digit copy: saved", saved, "images to debug_own_digits/")

# 2) Run A crops
os.makedirs("debug_runA", exist_ok=True)
for k, i in enumerate(random.sample(range(len(ds)), 40)):
    img, dn, ln, st = ds[i]
    T.functional.to_pil_image(img).save(f"debug_runA/{k:02d}_state{st}_{dn[0]}_{dn[1]}.jpg")
print("Run A: saved 40 images to debug_runA/")
