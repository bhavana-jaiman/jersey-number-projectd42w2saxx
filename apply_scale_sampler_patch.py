"""
Programmatically patches jerseyNumberRecognitiorWithState_Training.py to
replace the train_loader's shuffle=True with a SCALE-based
WeightedRandomSampler.

Why: validation-side failure analysis found accuracy collapses from ~0.95
(large-scale digits) to ~0.5-0.58 (small-scale, bbox <=84px), and that
regime holds ~70% of validation samples. check_train_scale_distribution.py
confirmed training data already has a similar/higher small-scale share
(~78%). The earlier confusable-DIGIT reweighting/oversampling approach
(run both aggressively and lightly, the lighter run also including the
existing RandomLowRes augmentation) did not clear the 79.74% baseline and
is being set aside. This patch is a clean, ISOLATED test of scale-based
oversampling alone: it does not touch utils/jersey_dataset.py's crop,
margin, resize, or RandomLowRes/RandomResizedCrop logic in any way -- it
only changes how often each existing training sample is drawn per epoch,
weighting samples whose digit bbox is small (<=84px, same cutoff used
throughout this analysis) more heavily.

Uses dataset.labels_files (array of label file paths) and
dataset.images_files (parallel array of image file paths) -- both
confirmed present on jerseyWithState_Dataset via:
  self.labels_files = np.array(self.labels_path)   (line ~98)
  image_path = self.images_files[index]            (line ~114/143/...)

Run this on a FRESH copy of the ORIGINAL baseline training script (the one
that produced the 79.74% checkpoint, with NO confusable-digit patches
applied), so the result is interpretable on its own.

Usage:
    python apply_scale_sampler_patch.py jerseyNumberRecognitiorWithState_Training.py
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
        "# ---------------HCL_Changes (scale-based oversampling)----------------",
        "# check_train_scale_distribution.py confirmed training data is already",
        "# ~78% small-scale (<=84px bbox), similar to validation's ~70% -- yet",
        "# the original failure-mode analysis found accuracy collapses from",
        "# ~0.95 (large scale) to ~0.5-0.58 (small scale) in that regime.",
        "# This oversamples small-scale samples further so the model sees",
        "# proportionally more of the regime it struggles with per epoch.",
        "# Isolated test: confusable-digit reweighting/oversampling REMOVED,",
        "# this is the only new variable vs. the clean 79.74% baseline run",
        "# (RandomLowRes augmentation in jersey_dataset.py is left as-is).",
        "SMALL_SCALE_PX_CUTOFF = 84",
        "SMALL_SCALE_OVERSAMPLE_WEIGHT = 2.0",
        "",
        "def _sample_weights_by_scale(dataset):",
        indent + "weights = []",
        indent + "for lpath, ipath in zip(dataset.labels_files, dataset.images_files):",
        indent + indent + "is_small = False",
        indent + indent + "try:",
        indent + indent + indent + "arr = np.loadtxt(lpath).reshape(-1, 5)",
        indent + indent + indent + "img = cv2.imread(ipath)",
        indent + indent + indent + "if img is not None and arr.size:",
        indent + indent + indent + indent + "H, W = img.shape[0], img.shape[1]",
        indent + indent + indent + indent + "box_w_px = arr[:, 3] * W",
        indent + indent + indent + indent + "box_h_px = arr[:, 4] * H",
        indent + indent + indent + indent + "scale_px = float(np.max(np.maximum(box_w_px, box_h_px)))",
        indent + indent + indent + indent + "is_small = scale_px <= SMALL_SCALE_PX_CUTOFF",
        indent + indent + "except Exception:",
        indent + indent + indent + "pass",
        indent + indent + "weights.append(SMALL_SCALE_OVERSAMPLE_WEIGHT if is_small else 1.0)",
        indent + "return weights",
        "",
        f"_hcl_concat_parts = {concat_list_expr}",
        "train_concat_dataset = ConcatDataset(_hcl_concat_parts)",
        "",
        "all_sample_weights = []",
        "for _hcl_part in _hcl_concat_parts:",
        indent + "all_sample_weights += _sample_weights_by_scale(_hcl_part)",
        "",
        "assert len(all_sample_weights) == len(train_concat_dataset), (",
        indent + 'f"sample weight count {len(all_sample_weights)} != dataset size {len(train_concat_dataset)}"',
        ")",
        'print(f"Scale oversampling: {sum(1 for w in all_sample_weights if w > 1)} / {len(all_sample_weights)} "',
        indent + indent + 'f"samples flagged as small-scale (<={SMALL_SCALE_PX_CUTOFF}px)")',
        "train_sampler = torch.utils.data.WeightedRandomSampler(",
        indent + "weights=all_sample_weights, num_samples=len(all_sample_weights), replacement=True)",
        "# ---------------------------------------------",
        "train_loader = DataLoader(train_concat_dataset, batch_size=opt.batch_size, sampler=train_sampler,",
        indent + "num_workers=opt.n_cpu, pin_memory=True,collate_fn=train_dataset.collate_fn, drop_last=True)",
    ]
    return "\n".join((indent + ln) if ln else "" for ln in lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_scale_sampler_patch.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    matches = list(PATTERN.finditer(content))

    if not matches:
        print("ERROR: no match for the train_loader = DataLoader(...) block was found.")
        print("No changes made.")
        print()
        print("Run this and paste the output so the pattern can be corrected:")
        print("  grep -n -A3 'train_loader = DataLoader' <your_file.py>")
        sys.exit(1)

    if len(matches) > 1:
        print(f"ERROR: found {len(matches)} matches, expected exactly 1. Refusing to patch.")
        sys.exit(1)

    m = matches[0]
    concat_list_expr = m.group("concat_list")

    line_start = content.rfind("\n", 0, m.start()) + 1
    indent_match = re.match(r"[ \t]*", content[line_start:m.start()])
    indent = indent_match.group(0) if indent_match else ""

    print(f"Matched train_loader block. Dataset list found: {concat_list_expr}")
    print(f"Detected indent: {repr(indent)}")

    for mod, needed in (("import cv2", "cv2.imread"), ("import numpy as np", "np.loadtxt")):
        if mod not in content:
            print(f"WARNING: '{mod}' not found at top of file -- required for {needed} in the new block. "
                  f"Add it manually near the top if missing.")

    new_block = build_new_block(concat_list_expr, indent)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_content = content[:line_start] + new_block + content[m.end():]
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: shuffle=True -> WeightedRandomSampler, weighted by bbox SCALE "
          f"(2x oversampling for samples with bbox <=84px), not digit identity.")


if __name__ == "__main__":
    main()
