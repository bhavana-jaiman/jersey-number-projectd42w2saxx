"""
Split training_dataset_Ying_V3 into a 70:30 train:val dataset, accounting for
a fixed pool of synthetic images that already lives elsewhere (not moved by
this script) so the FINAL combined ratio across everything is exactly 70:30.

Background / rules this script encodes (confirmed with the user):

  - Inside training_dataset_Ying_V3/upper_labels, a file name ending in
    "_<digits>.txt" (e.g. foo_0.txt, foo_1.txt) is SYNTHETIC.
    A plain "<hash>.txt" (no such suffix) is REAL.
  - Images are the source of truth: a sample only exists if its image file
    exists. Stray number_labels with no matching image/upper_label are
    ignored (there were 33 such stray files found).
  - External synthetic pools already placed elsewhere and NOT touched by
    this script, but counted in the ratio math:
        train side: 1500 (empty) + 1199 (syn 1,7,6,8)          = 2699
        val   side: 593 + 912 + 891 (syn 3-digit) + 500 (empty) = 2896
  - V3's own real (8822) and synthetic (27279) pools are EACH split so that,
    once added to the fixed external pool above, the grand total across
    train+val lands at exactly 70:30. This means V3 itself is split at
    ~73.37% train / ~26.63% val (not a flat 70:30), because the external
    pool is itself skewed close to 48:52.
  - Output layout (confirmed with the user):
      70_30_dataset_v3/
        training/                  <- images (.jpg) + upper_labels (.txt), flat, same basename
        training_number_labels/    <- number_labels .txt (full bbox format, UNCHANGED)
        validation/images/         <- images (.jpg)
        validation/labels/         <- upper_labels .txt, STRIPPED to number-only (first token),
                                       and ONLY for samples whose upper_labels file has exactly
                                       ONE line (multi-detection files are dropped from validation
                                       entirely, per user's instruction)
        validation_number_labels/  <- number_labels .txt (full bbox format, UNCHANGED)
  - Files are COPIED (not symlinked), so the new folder is self-contained.
  - Split is randomized with a fixed seed for reproducibility, done
    independently within the real pool and the synthetic pool (so both
    pools keep the same train/val proportion inside each, by construction
    of how many of each are drawn -- see TRAIN_FRAC_REAL / TRAIN_FRAC_SYNTH
    computed below from the exact counts).

Usage:
    python split_training_dataset_v3.py \
        --src ~/workspace_bhavana/Datasets/training_dataset_Ying_V3 \
        --dst ~/workspace_bhavana/Datasets/70_30_dataset_v3 \
        --seed 42
"""
import argparse
import random
import re
import shutil
from pathlib import Path

SYNTH_SUFFIX_RE = re.compile(r"_[0-9]+$")  # matches "..._0", "..._12" (stem, no extension)

# Fixed external synthetic pools (already placed elsewhere, NOT moved by this script,
# only used here to compute the exact counts V3 must contribute).
EXTERNAL_TRAIN = 1500 + 1199   # 2699
EXTERNAL_VAL = 593 + 912 + 891 + 500  # 2896


def _external_counts():
    """Allow overriding EXTERNAL_TRAIN/EXTERNAL_VAL via env for testing at scale."""
    import os
    t = os.environ.get("TEST_EXTERNAL_TRAIN")
    v = os.environ.get("TEST_EXTERNAL_VAL")
    if t is not None and v is not None:
        return int(t), int(v)
    return EXTERNAL_TRAIN, EXTERNAL_VAL


def is_synthetic(stem: str) -> bool:
    return bool(SYNTH_SUFFIX_RE.search(stem))


def find_image_for_stem(images_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def strip_upper_label_to_numbers(src_path: Path, dst_path: Path) -> bool:
    """
    Returns True if the file was single-line (kept), False if multi-line
    (caller should skip/drop this sample from validation entirely).
    """
    lines = [ln.strip() for ln in src_path.read_text(encoding="utf8").splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    number_token = lines[0].split()[0]
    dst_path.write_text(number_token + "\n", encoding="utf8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="training_dataset_Ying_V3 folder")
    ap.add_argument("--dst", required=True, help="output 70_30_dataset_v3 folder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    images_dir = src / "images"
    number_labels_dir = src / "number_labels"
    upper_labels_dir = src / "upper_labels"

    # --- 1. Enumerate samples: source of truth = images that have a matching upper_labels file ---
    real_stems, synth_stems = [], []
    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        stem = img_path.stem
        upper_path = upper_labels_dir / f"{stem}.txt"
        if not upper_path.exists():
            continue  # no label, skip (shouldn't happen given counts matched)
        (synth_stems if is_synthetic(stem) else real_stems).append(stem)

    n_real = len(real_stems)
    n_synth = len(synth_stems)
    print(f"Found {n_real} real samples, {n_synth} synthetic samples in {src}")

    external_train, external_val = _external_counts()

    # --- 2. Work out exact train/val counts for V3 so grand total hits 70:30 ---
    v3_total = n_real + n_synth
    grand_total = v3_total + external_train + external_val
    target_train_total = round(grand_total * 0.70)
    v3_train_needed = target_train_total - external_train
    v3_val_needed = v3_total - v3_train_needed

    frac_train = v3_train_needed / v3_total
    real_train_n = round(n_real * frac_train)
    real_val_n = n_real - real_train_n
    synth_train_n = v3_train_needed - real_train_n
    synth_val_n = n_synth - synth_train_n

    print(f"V3 must contribute -> train: {v3_train_needed}  val: {v3_val_needed}")
    print(f"  real  -> train: {real_train_n}  val: {real_val_n}")
    print(f"  synth -> train: {synth_train_n}  val: {synth_val_n}")

    total_train = real_train_n + synth_train_n + external_train
    total_val = real_val_n + synth_val_n + external_val
    total = total_train + total_val
    print(f"FINAL (incl. external fixed pool) train: {total_train} ({100*total_train/total:.2f}%)"
          f"  val: {total_val} ({100*total_val/total:.2f}%)")

    # --- 3. Randomly assign stems to train/val within each pool ---
    rng = random.Random(args.seed)
    real_shuffled = real_stems[:]
    synth_shuffled = synth_stems[:]
    rng.shuffle(real_shuffled)
    rng.shuffle(synth_shuffled)

    train_stems = set(real_shuffled[:real_train_n]) | set(synth_shuffled[:synth_train_n])
    val_stems = (set(real_shuffled[real_train_n:]) | set(synth_shuffled[synth_train_n:]))
    assert len(train_stems & val_stems) == 0
    assert len(train_stems) == total_train - external_train
    assert len(val_stems) == total_val - external_val

    # --- 4. Set up output dirs ---
    train_dir = dst / "training"
    train_num_labels_dir = dst / "training_number_labels"
    val_img_dir = dst / "validation" / "images"
    val_label_dir = dst / "validation" / "labels"
    val_num_labels_dir = dst / "validation_number_labels"
    for d in (train_dir, train_num_labels_dir, val_img_dir, val_label_dir, val_num_labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 5. Copy TRAIN samples: image + upper_labels flat; number_labels to its own folder ---
    n_copied_train = 0
    for stem in train_stems:
        img_path = find_image_for_stem(images_dir, stem)
        upper_path = upper_labels_dir / f"{stem}.txt"
        num_path = number_labels_dir / f"{stem}.txt"
        if img_path is None or not upper_path.exists():
            continue
        shutil.copy2(img_path, train_dir / img_path.name)
        shutil.copy2(upper_path, train_dir / upper_path.name)
        if num_path.exists():
            shutil.copy2(num_path, train_num_labels_dir / num_path.name)
        n_copied_train += 1

    # --- 6. Copy VAL samples: image to images/, upper_labels STRIPPED to labels/
    #         (only if single-line; multi-line files are dropped entirely),
    #         number_labels to its own folder (full, unchanged) ---
    n_copied_val = 0
    n_dropped_multiline = 0
    for stem in val_stems:
        img_path = find_image_for_stem(images_dir, stem)
        upper_path = upper_labels_dir / f"{stem}.txt"
        num_path = number_labels_dir / f"{stem}.txt"
        if img_path is None or not upper_path.exists():
            continue
        stripped_dst = val_label_dir / upper_path.name
        kept = strip_upper_label_to_numbers(upper_path, stripped_dst)
        if not kept:
            n_dropped_multiline += 1
            if stripped_dst.exists():
                stripped_dst.unlink()
            continue  # drop this sample entirely from validation, per user's instruction
        shutil.copy2(img_path, val_img_dir / img_path.name)
        if num_path.exists():
            shutil.copy2(num_path, val_num_labels_dir / num_path.name)
        n_copied_val += 1

    print()
    print(f"Copied to training/: {n_copied_train} samples")
    print(f"Copied to validation/: {n_copied_val} samples "
          f"({n_dropped_multiline} dropped for having multiple detections per label file)")
    print(f"Done. Output at: {dst}")


if __name__ == "__main__":
    main()
