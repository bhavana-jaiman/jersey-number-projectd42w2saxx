"""
Programmatically patches jerseyNumberRecognitiorWithState_Training.py to
replace the shuffle=True train_loader with a WeightedRandomSampler that
oversamples training batches containing confusable digits
(1,2,3,5,6,7,8,9 -- the classes implicated in the 7<->1, 6->8, 3->2,
9->8, 2->1, 5->6 confusion pairs found in validation failure analysis).

Uses a REGEX match (tolerant of tabs vs spaces, and of exact kwarg
line-wrapping) on the train_loader = DataLoader(...) block, so it doesn't
break on whitespace differences between what we were shown earlier and the
real live file. Makes a timestamped .bak copy before writing.

Usage:
    python apply_sampler_patch_v2.py jerseyNumberRecognitiorWithState_Training.py

If the match fails, the script prints what it searched for and makes NO
changes -- safe to re-run after the match pattern is corrected.
"""
import re
import sys
import shutil
import time

# Matches, in order, allowing any whitespace between tokens:
#   train_loader = DataLoader(ConcatDataset([train_dataset] + [empty_train] + [empty_add] + [syn_digits_train]),
#       batch_size=opt.batch_size, <optional comment> shuffle=True, num_workers=opt.n_cpu,
#       pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
# Captures the ConcatDataset(...) list expression so we reuse exactly
# whatever dataset list the live file actually has (handles empty_add
# being present or absent, and any ordering).
PATTERN = re.compile(
    r"train_loader\s*=\s*DataLoader\(\s*ConcatDataset\(\s*(?P<concat_list>\[.*?\])\s*\)\s*,"
    r"\s*batch_size\s*=\s*opt\.batch_size\s*,"
    r"(?:\s*#[^\n]*\n)?"          # optional trailing comment on that line
    r"\s*shuffle\s*=\s*True\s*,"
    r"\s*num_workers\s*=\s*opt\.n_cpu\s*,"
    r"\s*pin_memory\s*=\s*True\s*,"
    r"\s*collate_fn\s*=\s*train_dataset\.collate_fn\s*,"
    r"\s*drop_last\s*=\s*True\s*\)",
    re.DOTALL,
)


def build_new_block(concat_list_expr):
    return f'''# ---------------HCL_Changes----------------
# Confusable-digit oversampling: WeightedRandomSampler so samples
# containing digits 1,2,3,5,6,7,8,9 (the classes involved in the
# confusion pairs found in validation failure analysis: 7<->1, 6->8,
# 3->2, 9->8, 2->1, 5->6) are drawn ~2x as often per epoch. Complements
# the loss-level reweighting in utils/loss.py (make_loss_fn). Digit
# class balance in training data was already confirmed even (3700-4200
# instances each), so this is about exposure frequency per epoch given
# fixed batch composition, not correcting a raw count imbalance.
CONFUSABLE_DIGITS_SET = {{1, 2, 3, 5, 6, 7, 8, 9}}
OVERSAMPLE_WEIGHT = 2.0

def _sample_weights_from_labels_list(dataset):
	weights = []
	for labels in dataset.labels_list:
		has_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))
		weights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)
	return weights

_hcl_concat_parts = {concat_list_expr}
train_concat_dataset = ConcatDataset(_hcl_concat_parts)

all_sample_weights = []
for _hcl_part in _hcl_concat_parts:
	all_sample_weights += _sample_weights_from_labels_list(_hcl_part)

assert len(all_sample_weights) == len(train_concat_dataset), (
	f"sample weight count {{len(all_sample_weights)}} != dataset size {{len(train_concat_dataset)}}"
)
print(f"Oversampling: {{sum(1 for w in all_sample_weights if w > 1)}} / {{len(all_sample_weights)}} "
	  f"samples flagged as containing a confusable digit")
train_sampler = torch.utils.data.WeightedRandomSampler(
	weights=all_sample_weights, num_samples=len(all_sample_weights), replacement=True)
# ---------------------------------------------
train_loader = DataLoader(train_concat_dataset, batch_size=opt.batch_size, sampler=train_sampler,
			num_workers=opt.n_cpu, pin_memory=True,collate_fn=train_dataset.collate_fn, drop_last=True)'''


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_sampler_patch_v2.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    matches = list(PATTERN.finditer(content))

    if not matches:
        print("ERROR: no match for the train_loader = DataLoader(...) block was found.")
        print("No changes made.")
        print()
        print("This regex looks for, in order (whitespace-insensitive):")
        print("  train_loader = DataLoader(ConcatDataset([...]), batch_size=opt.batch_size,")
        print("      shuffle=True, num_workers=opt.n_cpu, pin_memory=True,")
        print("      collate_fn=train_dataset.collate_fn, drop_last=True)")
        print()
        print("Run this and paste the output so the pattern can be corrected:")
        print("  grep -n -A3 'train_loader = DataLoader' <your_file.py>")
        sys.exit(1)

    if len(matches) > 1:
        print(f"ERROR: found {len(matches)} matches, expected exactly 1. "
              "Refusing to patch to avoid changing the wrong spot. "
              "Tell me and I'll make the match more specific.")
        sys.exit(1)

    m = matches[0]
    concat_list_expr = m.group("concat_list")
    print(f"Matched train_loader block. Dataset list found: {concat_list_expr}")

    new_block = build_new_block(concat_list_expr)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_content = content[: m.start()] + new_block + content[m.end():]
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: shuffle=True -> WeightedRandomSampler (2x oversampling on digits 1,2,3,5,6,7,8,9)")
    print()
    print("NOTE: the dataset list used is exactly what was found in your live file "
          f"({concat_list_expr}) -- so this correctly includes empty_add if your file has it.")


if __name__ == "__main__":
    main()
