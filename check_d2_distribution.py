import glob, os, warnings
import numpy as np
from collections import Counter

TRAIN_PATH = "./datasets/training_dataset_Ying"      # same as --data_config
warnings.filterwarnings("ignore")                     # hides "empty file" warnings

d1_count, d2_count, n_boxes = Counter(), Counter(), Counter()

for p in glob.glob(os.path.join(TRAIN_PATH, "*.txt")):
    l = np.loadtxt(p).reshape(-1, 5)
    if l.size == 0 or l[0][0] == 10:
        n_boxes["no digit"] += 1
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

print("Images by number of digit boxes:", dict(n_boxes))
show("digit1 (first digit) counts", d1_count)
show("digit2 (second digit) counts", d2_count)
