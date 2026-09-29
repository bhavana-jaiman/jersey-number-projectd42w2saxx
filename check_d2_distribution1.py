import glob, os, warnings
import numpy as np
from collections import Counter

TRAIN_PATH = "./datasets/training_dataset_Ying"      # same as --data_config
warnings.filterwarnings("ignore")                     # hides "empty file" warnings

d1_count, d2_count, n_boxes = Counter(), Counter(), Counter()
odd_files = []

for p in glob.glob(os.path.join(TRAIN_PATH, "*.txt")):
    raw = np.atleast_1d(np.loadtxt(p))

    # --- files that are not full "class cx cy w h" lines ---
    if raw.size == 0:
        n_boxes["empty file"] += 1
        continue
    if raw.size % 5 != 0:
        if raw.size == 1 and int(raw[0]) == 10:
            n_boxes["no digit (only '10')"] += 1
        else:
            odd_files.append((os.path.basename(p), raw.tolist()))
        continue

    l = raw.reshape(-1, 5)
    if l[0][0] == 10:
        n_boxes["no digit (full line)"] += 1
        continue

    l = l[np.argsort(l[:, 1])]                        # left -> right, same as training
    n = len(l)
    n_boxes[n if n <= 2 else "3+"] += 1
    if n == 1:
        d1_count[int(l[0, 0])] += 1
        d2_count["blank"] += 1
    elif n == 2:
        d1_count[int(l[0, 0])] += 1
        d2_count[int(l[1, 0])] += 1

def show(title, c):
    total = sum(c.values())
    print(f"\n{title}  (total {total})")
    for k in list(range(10)) + ["blank"]:
        if k in c:
            print(f"  {str(k):>5}: {c[k]:6d}  ({100*c[k]/total:5.1f}%)  " + "#" * int(60*c[k]/total))

print("Images by label type:", dict(n_boxes))
show("digit1 (first digit) counts", d1_count)
show("digit2 (second digit) counts", d2_count)

if odd_files:
    print(f"\n{len(odd_files)} files with an unexpected format (first 10):")
    for name, vals in odd_files[:10]:
        print("  ", name, vals)
