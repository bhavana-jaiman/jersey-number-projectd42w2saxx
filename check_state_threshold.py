"""
check_state_threshold.py   (standalone, runs next to training, does not change anything)

1) For a checkpoint, how many synthetic 3-digit images are called "3+" and how many
   normal body numbers are wrongly called "3+", for thresholds 0 (argmax only) .. 0.9
2) Saves the "doubled digit" mistakes (true 1 digit X -> predicted XX) of validation_body
   into --out/doubled/ so you can look at them.

Run (from workspace_bhavana_New_arc_bodycrop):
    python check_state_threshold.py --checkpoint checkpoints/exp7/<run>/jerseyNumberRecognitior_ckp_40.pth
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from subModules.backbone_ying import MultiTaskLearnerWithState
from utils.jersey_dataset import jerseyNumber_ValidationDataset_V2 as V2

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--body", default="../workspace_bhavana/Datasets/70_30_dataset/validation_body")
ap.add_argument("--syn3", default="../workspace_bhavana/Datasets/70_30_dataset/syn3_dataset")
ap.add_argument("--out", default="results/state_check")
opt = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskLearnerWithState().to(device)
sd = torch.load(opt.checkpoint, map_location=device)
model.load_state_dict(sd.get("state_dict", sd), strict=True)
model.eval()


def run(ds):
    """returns p(3+), argmax state, pred d1, pred d2, labels  for the whole dataset"""
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=ds.new_collate_fn)
    P2, S, D1, D2, L = [], [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            _, l1, l2, ls = model(images.to(device))
            prob = torch.softmax(ls, 1)
            P2.append(prob[:, 2].cpu()); S.append(prob.argmax(1).cpu())
            D1.append(l1.argmax(1).cpu()); D2.append(l2.argmax(1).cpu()); L.append(labels)
    return [torch.cat(x).numpy() for x in (P2, S, D1, D2, L)]


body = V2(opt.body)                       # skips 3+
syn3 = V2(opt.syn3, keep_3plus=True)
bP2, bS, bD1, bD2, bL = run(body)
sP2, sS, _, _, _ = run(syn3)

print(f"\nbody images: {len(bP2)}   synthetic 3-digit images: {len(sP2)}")
print(f"{'threshold':>10} | {'3-digit caught':>16} | {'normal called 3+':>17}")
for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
    caught = int(((sS == 2) & (sP2 >= t)).sum())
    false3 = int(((bS == 2) & (bP2 >= t)).sum())
    name = "argmax" if t == 0 else f"{t:.1f}"
    print(f"{name:>10} | {caught:5d} ({100*caught/len(sP2):5.1f}%) | {false3:5d} ({100*false3/len(bP2):5.2f}%)")
print("p(3+) on 3-digit images: median %.2f, p25 %.2f, p75 %.2f"
      % tuple(np.percentile(sP2, [50, 25, 75])))

# doubled-digit mistakes: true one digit X -> predicted X X
os.makedirs(os.path.join(opt.out, "doubled"), exist_ok=True)
n = 0
for i in range(len(bL)):
    t1, t2 = int(bL[i][0]), int(bL[i][1])
    if t2 == 10 and bD1[i] == t1 and bD2[i] == t1:
        img = cv2.imread(body.images_path[i])
        cv2.imwrite(os.path.join(opt.out, "doubled", f"{t1}_as_{t1}{t1}__{os.path.basename(body.images_path[i])}"), img)
        n += 1
print(f"\ndoubled-digit mistakes saved: {n} -> {opt.out}/doubled/")
