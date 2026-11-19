# Patch 2 of 2: jerseyNumberRecognitiorWithState_Training.py — oversample confusable-digit samples

## What this does
Replaces `shuffle=True` on `train_loader` with a `WeightedRandomSampler`
(mutually exclusive with shuffle -- PyTorch requires you drop `shuffle`
when passing `sampler`). Any training sample whose label contains one of
the confusable digits {1,2,3,5,6,7,8,9} gets weight 2.0; everything else
(digit 0, 4, and blank-only samples) keeps weight 1.0. This means those
samples are drawn roughly twice as often per epoch, on top of the loss
reweighting in Patch 1.

This uses `jerseyWithState_Dataset.labels_list`, confirmed present on
every instance of this class (built in `__init__` from each sample's YOLO
label file, column 0 = digit class). Since `train_dataset`, `empty_train`,
and `syn_digits_train` are all this same class, we can walk each one's
`.labels_list` in the same order they're concatenated to build one
combined weight array matching `ConcatDataset`'s index order exactly.

## Where to find the current code

Around where `train_loader` is built (should be shortly after the
`syn_digits_train = jerseyWithState_Dataset(...)` line you just showed
me), find:

```python
train_loader = DataLoader(
    ConcatDataset([train_dataset] + [empty_train] + [syn_digits_train]),
    batch_size=opt.batch_size, shuffle=True,
    num_workers=opt.n_cpu, pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
```

(the exact kwarg order/line-wrapping may differ slightly from this --
match on `shuffle=True` inside the `train_loader = DataLoader(...)` call
that wraps `[train_dataset] + [empty_train] + [syn_digits_train]`)

## Replace it with:

```python
# ---------------HCL_Changes----------------
# Confusable-digit oversampling: WeightedRandomSampler so samples
# containing digits 1,2,3,5,6,7,8,9 (the classes involved in the
# confusion pairs found in validation failure analysis: 7<->1, 6->8,
# 3->2, 9->8, 2->1, 5->6) are drawn ~2x as often per epoch. Complements
# the loss-level reweighting in utils/loss.py (make_loss_fn). Digit
# class balance in training data was already confirmed even (3700-4200
# instances each), so this is about exposure frequency per epoch given
# fixed batch composition, not correcting a raw count imbalance.
CONFUSABLE_DIGITS_SET = {1, 2, 3, 5, 6, 7, 8, 9}
OVERSAMPLE_WEIGHT = 2.0

def _sample_weights_from_labels_list(dataset):
    weights = []
    for labels in dataset.labels_list:
        # labels is the array of digit class ids present in this sample
        # (column 0 of the YOLO .txt, loaded in jerseyWithState_Dataset.__init__)
        has_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))
        weights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)
    return weights

train_concat_dataset = ConcatDataset([train_dataset] + [empty_train] + [syn_digits_train])

all_sample_weights = (
    _sample_weights_from_labels_list(train_dataset)
    + _sample_weights_from_labels_list(empty_train)
    + _sample_weights_from_labels_list(syn_digits_train)
)
assert len(all_sample_weights) == len(train_concat_dataset), (
    f"sample weight count {len(all_sample_weights)} != dataset size {len(train_concat_dataset)} "
    f"-- ConcatDataset ordering must match the order weights were built in"
)
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=all_sample_weights, num_samples=len(all_sample_weights), replacement=True)
# ---------------------------------------------

train_loader = DataLoader(
    train_concat_dataset,
    batch_size=opt.batch_size, sampler=train_sampler,
    num_workers=opt.n_cpu, pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
```

## Two things to verify before running

1. **`numpy as np` must already be imported** in this file (it is -- `10:import numpy as np`
   was in the earlier import listing). No new import needed for that.

2. **`torch.utils.data.WeightedRandomSampler`** needs `torch.utils.data` accessible --
   the file already does `from torch.utils.data import DataLoader` and
   `from torch.utils.data import ConcatDataset`, so `torch` itself is imported
   (`33:import torch`), meaning `torch.utils.data.WeightedRandomSampler` will resolve
   fine without a new import line.

3. **Run a quick sanity check first** before a full training run --
   add a temporary print right after building `all_sample_weights`:
   ```python
   print(f"Oversampling: {sum(1 for w in all_sample_weights if w > 1)} / {len(all_sample_weights)} "
         f"samples flagged as containing a confusable digit")
   ```
   This should print a plausible fraction (expect well over half, since
   confusable digits 1,2,3,5,6,7,8,9 cover 8 of 10 possible digit classes --
   most 2-digit samples will contain at least one). If it prints 0 or the
   full dataset size, something is wrong with how `labels_list` is being
   read and we should stop before training on it.
