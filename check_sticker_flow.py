import random, numpy as np, torch
from collections import Counter
from utils.jersey_dataset_New import JerseyWithState_Dataset
from utils.loss_New import make_loss_fn

TRAIN_PATH = "../Datasets/balanced_dataset_70_30/training"    # your --data_config
random.seed(0); np.random.seed(0); torch.manual_seed(0)

ds = JerseyWithState_Dataset(TRAIN_PATH, transform=None)

# record which sticker was used for each sample
used = {}
orig = ds._get_tight_digit_sticker
def spy(idx):
    out = orig(idx)
    used["digits"] = out[1]
    return out
ds._get_tight_digit_sticker = spy

counts, examples = Counter(), {}
for i in random.sample(range(len(ds)), 3000):
    used.clear()
    img, dn, ln, st = ds[i]
    if not used.get("digits"):
        continue                                   # no sticker for this sample
    kind = "sticker -> state 1 (digit heads learn)" if st == 1 else "sticker -> state 2 (only state head learns)"
    counts[kind] += 1
    examples.setdefault(kind, (dn, st))

print("sticker samples in 3000 random draws:")
for k, v in counts.items():
    print(f"  {v:5d}  {k}")

# pass one example of each type through the real loss with random model outputs
crit = make_loss_fn()
d1, d2, s = torch.randn(1, 10), torch.randn(1, 11), torch.randn(1, 3)
print("\nloss parts for one example of each:")
for kind, (dn, st) in examples.items():
    crit(d1, d2, s, torch.tensor([dn]), torch.tensor([st]))
    parts = {k: round(float(v), 3) for k, v in crit.last_losses.items()}
    print(f"  {kind}\n     labels={dn} state={st} -> {parts}")
