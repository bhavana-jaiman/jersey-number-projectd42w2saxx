"""
Programmatically patches jerseyNumberRecognitiorWithState_Training.py to
replace the shuffle=True train_loader with a WeightedRandomSampler that
oversamples training batches containing confusable digits
(1,2,3,5,6,7,8,9).

v3 fix: the replacement block is now built so that EVERY line -- including
the final `train_loader = DataLoader(...)` line and its continuation --
carries the same leading indent as the original matched line. v2 left the
final train_loader line at column 0 (no leading tab), which parsed as a
dedent relative to the `model = MultiTaskLearnerWithState(...)` line that
follows it, causing "IndentationError: unexpected indent" on that next
line. v3 detects the real indent from the matched text and applies it to
every line of the new block, not just some of them.

Usage:
    python apply_sampler_patch_v3.py jerseyNumberRecognitiorWithState_Training.py
"""
import re
import sys
import shutil
import time

PATTERN = re.compile(
    r"train_loader\s*=\s*DataLoader\(\s*ConcatDataset\(\s*(?P<concat_list>\[.*?\])\s*\)\s*,"
    r"\s*batch_size\s*=\s*opt\.batch_size\s*,"
    r"(?:\s*#[^\n]*\n)?"
    r"\s*shuffle\s*=\s*True\s*,"
    r"\s*num_workers\s*=\s*opt\.n_cpu\s*,"
    r"\s*pin_memory\s*=\s*True\s*,"
    r"\s*collate_fn\s*=\s*train_dataset\.collate_fn\s*,"
    r"\s*drop_last\s*=\s*True\s*\)",
    re.DOTALL,
)


def build_new_block(concat_list_expr, indent):
    lines = [
        "# ---------------HCL_Changes----------------",
        "# Confusable-digit oversampling: WeightedRandomSampler so samples",
        "# containing digits 1,2,3,5,6,7,8,9 (the classes involved in the",
        "# confusion pairs found in validation failure analysis: 7<->1, 6->8,",
        "# 3->2, 9->8, 2->1, 5->6) are drawn ~2x as often per epoch. Complements",
        "# the loss-level reweighting in utils/loss.py (make_loss_fn). Digit",
        "# class balance in training data was already confirmed even (3700-4200",
        "# instances each), so this is about exposure frequency per epoch given",
        "# fixed batch composition, not correcting a raw count imbalance.",
        "CONFUSABLE_DIGITS_SET = {1, 2, 3, 5, 6, 7, 8, 9}",
        "OVERSAMPLE_WEIGHT = 2.0",
        "",
        "def _sample_weights_from_labels_list(dataset):",
        indent + "weights = []",
        indent + "for labels in dataset.labels_list:",
        indent + indent + "has_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))",
        indent + indent + "weights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)",
        indent + "return weights",
        "",
        f"_hcl_concat_parts = {concat_list_expr}",
        "train_concat_dataset = ConcatDataset(_hcl_concat_parts)",
        "",
        "all_sample_weights = []",
        "for _hcl_part in _hcl_concat_parts:",
        indent + "all_sample_weights += _sample_weights_from_labels_list(_hcl_part)",
        "",
        "assert len(all_sample_weights) == len(train_concat_dataset), (",
        indent + 'f"sample weight count {len(all_sample_weights)} != dataset size {len(train_concat_dataset)}"',
        ")",
        'print(f"Oversampling: {sum(1 for w in all_sample_weights if w > 1)} / {len(all_sample_weights)} "',
        indent + indent + 'f"samples flagged as containing a confusable digit")',
        "train_sampler = torch.utils.data.WeightedRandomSampler(",
        indent + "weights=all_sample_weights, num_samples=len(all_sample_weights), replacement=True)",
        "# ---------------------------------------------",
        "train_loader = DataLoader(train_concat_dataset, batch_size=opt.batch_size, sampler=train_sampler,",
        indent + "num_workers=opt.n_cpu, pin_memory=True,collate_fn=train_dataset.collate_fn, drop_last=True)",
    ]
    # Apply the base indent (matching the original matched line's indent)
    # to every line so the whole block sits at the correct level.
    return "\n".join((indent + ln) if ln else "" for ln in lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_sampler_patch_v3.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    matches = list(PATTERN.finditer(content))

    if not matches:
        print("ERROR: no match for the train_loader = DataLoader(...) block was found.")
        print("No changes made.")
        sys.exit(1)

    if len(matches) > 1:
        print(f"ERROR: found {len(matches)} matches, expected exactly 1.")
        sys.exit(1)

    m = matches[0]
    concat_list_expr = m.group("concat_list")

    # Detect the real indent of the matched line (from the start of its
    # line in the file, not from the regex match start).
    line_start = content.rfind("\n", 0, m.start()) + 1
    indent_match = re.match(r"[ \t]*", content[line_start:m.start()])
    indent = indent_match.group(0) if indent_match else ""

    print(f"Matched train_loader block. Dataset list found: {concat_list_expr}")
    print(f"Detected indent: {repr(indent)}")

    new_block = build_new_block(concat_list_expr, indent)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    # Replace from the start of the line (so we don't leave a stray old
    # indent before the new block) through the end of the match.
    new_content = content[:line_start] + new_block + content[m.end():]
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: shuffle=True -> WeightedRandomSampler (2x oversampling on digits 1,2,3,5,6,7,8,9)")


if __name__ == "__main__":
    main()
