"""
Programmatically patches jerseyNumberRecognitiorWithState_Training.py to
replace the shuffle=True train_loader with a WeightedRandomSampler that
oversamples training batches containing confusable digits
(1,2,3,5,6,7,8,9 -- the classes implicated in the 7<->1, 6->8, 3->2,
9->8, 2->1, 5->6 confusion pairs found in validation failure analysis).

Uses exact string match-and-replace (not line numbers, not manual
indentation) so there is no risk of broken indentation from copy-paste.
Makes a timestamped .bak copy of the original file before writing.

Usage:
    python apply_sampler_patch.py jerseyNumberRecognitiorWithState_Training.py

If the match fails, the script prints exactly what it was looking for and
makes NO changes -- safe to re-run after fixing the file back to a known
state.
"""
import sys
import shutil
import time

OLD_BLOCK = '''train_loader = DataLoader(ConcatDataset([train_dataset] + [empty_train] + [empty_add] + [syn_digits_train]),batch_size=opt.batch_size,# [train_dataset, empty_dataset]
			shuffle=True, num_workers=opt.n_cpu, pin_memory=True,collate_fn=train_dataset.collate_fn, drop_last=True)'''

NEW_BLOCK = '''# ---------------HCL_Changes----------------
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
		has_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))
		weights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)
	return weights

train_concat_dataset = ConcatDataset([train_dataset] + [empty_train] + [empty_add] + [syn_digits_train])

all_sample_weights = (
	_sample_weights_from_labels_list(train_dataset)
	+ _sample_weights_from_labels_list(empty_train)
	+ _sample_weights_from_labels_list(empty_add)
	+ _sample_weights_from_labels_list(syn_digits_train)
)
assert len(all_sample_weights) == len(train_concat_dataset), (
	f"sample weight count {len(all_sample_weights)} != dataset size {len(train_concat_dataset)}"
)
print(f"Oversampling: {sum(1 for w in all_sample_weights if w > 1)} / {len(all_sample_weights)} "
	  f"samples flagged as containing a confusable digit")
train_sampler = torch.utils.data.WeightedRandomSampler(
	weights=all_sample_weights, num_samples=len(all_sample_weights), replacement=True)
# ---------------------------------------------
train_loader = DataLoader(train_concat_dataset, batch_size=opt.batch_size, sampler=train_sampler,
			num_workers=opt.n_cpu, pin_memory=True,collate_fn=train_dataset.collate_fn, drop_last=True)'''


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_sampler_patch.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if OLD_BLOCK not in content:
        print("ERROR: exact match for the train_loader block was NOT found in the file.")
        print("No changes made. This means the file's current text differs from what")
        print("this script expects (even a single extra space breaks an exact match).")
        print()
        print("Expected to find exactly this block:")
        print("-" * 70)
        print(OLD_BLOCK)
        print("-" * 70)
        print()
        print("Run: grep -n -B2 -A3 'train_loader = DataLoader' <your_file.py>")
        print("and paste the output so the match string can be corrected.")
        sys.exit(1)

    occurrences = content.count(OLD_BLOCK)
    if occurrences > 1:
        print(f"ERROR: found {occurrences} matches, expected exactly 1. "
              "Refusing to patch to avoid changing the wrong spot. "
              "Tell me and I'll make the match more specific.")
        sys.exit(1)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_content = content.replace(OLD_BLOCK, NEW_BLOCK)
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: shuffle=True -> WeightedRandomSampler (2x oversampling on digits 1,2,3,5,6,7,8,9)")


if __name__ == "__main__":
    main()
