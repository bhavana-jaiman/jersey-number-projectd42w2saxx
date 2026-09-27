import os
import sys
import glob
import numpy as np
import pandas as pd

# usage: python check_state.py output/exp6/wrong_predictions.csv ./datasets/training_dataset_Ying
csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/exp6/wrong_predictions.csv"
train_dir = sys.argv[2] if len(sys.argv) > 2 else "./datasets/training_dataset_Ying"

# ------------------------------------------------------------------
# 1) Of the "1-2 digits predicted as 3+" errors, how many digits were read correctly?
# ------------------------------------------------------------------
df = pd.read_csv(csv_path)
s12 = df[(df.true_state == 1) & (df.pred_state == 2)]
digits_ok = s12[(s12.pred_d1 == s12.true_d1) & (s12.pred_d2 == s12.true_d2)]
print("=== Test: state 1 predicted as state 2 ===")
print(f"total                         : {len(s12)}")
print(f"digits were read correctly    : {len(digits_ok)}  "
      f"({100 * len(digits_ok) / max(1, len(s12)):.1f}%)  <- only the state head was wrong")
print()

# ------------------------------------------------------------------
# 2) How many REAL samples of each state are in the training set?
#    (same rules as JerseyWithState_Dataset, before sticker augmentation)
# ------------------------------------------------------------------
counts = {0: 0, 1: 0, 2: 0}
far_apart_pairs = 0
bad = 0
for txt in glob.glob(os.path.join(train_dir, "*.txt")):
    try:
        label = np.loadtxt(txt).reshape(-1, 5)
    except Exception:
        bad += 1
        continue
    if label.size == 0:
        bad += 1
        continue
    n = label.shape[0]
    if label[0][0] == 10:
        counts[0] += 1
    elif n >= 3:
        counts[2] += 1
    else:
        counts[1] += 1
        if n == 2 and (abs(label[0][1] - label[1][1]) > 0.25 or abs(label[0][2] - label[1][2]) > 0.25):
            far_apart_pairs += 1

total = sum(counts.values())
print("=== Training set: real samples per state ===")
for s, name in [(0, "no digit"), (1, "1-2 digits"), (2, "3+ digits")]:
    print(f"state {s} ({name:<10}): {counts[s]:6d}  ({100 * counts[s] / max(1, total):.1f}%)")
print(f"total                : {total}   (unreadable label files: {bad})")
print(f"2-digit pairs that are far apart (training keeps only the 1st digit): {far_apart_pairs}")
