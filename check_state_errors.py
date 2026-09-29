"""
Find validation samples that are really 1-2 digit numbers (state 1)
but are predicted as state 2 (3+ digits), and analyse why.

Uses exactly the same validation dataset + preprocessing as train_New.py
(JerseyNumber_ValidationDataset_V2 + new_collate_fn).

Usage (from ~/workspace_bhavana/New_arcitecture):
    python check_state_errors.py \
        --checkpoint checkpoints/<run>/<folder>/jerseyNumberRecognizer_best.pth \
        --val_path  <same folder as validation_path in train_New.py>
"""
import os
import csv
import shutil
import argparse
from collections import Counter

import torch

from utils.jersey_dataset_New import JerseyNumber_ValidationDataset_V2
from subModules.backbone_New import MultiTaskLearnerWithState

BLANK = 10


def load_model(path, device):
    model = MultiTaskLearnerWithState().to(device)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    # _best.pth = plain state_dict; _ckp_ files = dict with the weights inside
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_path", required=True, help="same as validation_path in train_New.py")
    ap.add_argument("--out", default="state_errors", help="output folder")
    ap.add_argument("--save_images", type=int, default=200, help="how many error images to copy")
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.checkpoint, device)
    ds = JerseyNumber_ValidationDataset_V2(opt.val_path)
    os.makedirs(os.path.join(opt.out, "images"), exist_ok=True)

    total = Counter()          # true state-1 samples by number of digits
    to_state2 = Counter()      # ... predicted as state 2 (normal eval, BGR like training eval)
    to_state0 = Counter()      # ... predicted as state 0
    to_state2_rgb = Counter()  # same, but with RGB input (colour-order test)
    conf_bins = Counter()
    true_numbers = Counter()
    digits_ok_in_errors = 0
    rows = []

    with torch.no_grad():
        for i in range(len(ds)):
            img, dl, _ = ds.new_collate_fn([ds[i]])
            d1, d2 = int(dl[0, 0]), int(dl[0, 1])
            if d1 == BLANK:
                continue                               # state 0 sample (none expected)
            n = 1 if d2 == BLANK else 2
            total[n] += 1
            img = img.to(device)

            # 1) exactly like eval_one_epoch (image as the dataset gives it)
            _, l1, l2, ls = model(img)
            p = torch.softmax(ls, 1)[0]
            pred_state = int(p.argmax())
            p1, p2 = int(l1.argmax()), int(l2.argmax())

            # 2) same image with colour channels swapped (BGR <-> RGB)
            _, _, _, ls_rgb = model(img[:, [2, 1, 0]])
            if int(ls_rgb.argmax()) == 2:
                to_state2_rgb[n] += 1

            if pred_state == 0:
                to_state0[n] += 1
            if pred_state != 2:
                continue

            to_state2[n] += 1
            p_s2 = float(p[2])
            conf_bins[">0.9" if p_s2 > 0.9 else "0.7-0.9" if p_s2 > 0.7 else "0.5-0.7" if p_s2 > 0.5 else "<0.5"] += 1
            true_str = f"{d1}" if n == 1 else f"{d1}{d2}"
            true_numbers[true_str] += 1
            digits_ok = (p1 == d1) and (p2 == d2)
            digits_ok_in_errors += digits_ok

            fname = ds.images_path[i]
            rows.append({"filename": fname, "true_number": true_str, "n_digits": n,
                         "p_state2": round(p_s2, 3), "pred_d1": p1,
                         "pred_d2": "" if p2 == BLANK else p2, "digits_correct": digits_ok})

    # save csv + highest-confidence error images
    rows.sort(key=lambda r: -r["p_state2"])
    with open(os.path.join(opt.out, "state1_as_state2.csv"), "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    for r in rows[:opt.save_images]:
        base = os.path.basename(r["filename"])
        shutil.copy(r["filename"], os.path.join(
            opt.out, "images", f"p{r['p_state2']:.2f}_true{r['true_number']}_{base}"))

    # summary
    all_n = sum(total.values())
    print("\n==== true state 1 (1-2 digits) predicted as state 2 ====")
    print(f"validation state-1 samples: {all_n}")
    for n in (1, 2):
        if total[n]:
            print(f"  {n}-digit numbers: {to_state2[n]:5d} / {total[n]:5d}  ({100*to_state2[n]/total[n]:.2f}%) -> state 2")
    print(f"  total -> state 2: {sum(to_state2.values())}   (-> state 0: {sum(to_state0.values())})")

    print("\nconfidence of the wrong 'state 2' (softmax prob):")
    for k in [">0.9", "0.7-0.9", "0.5-0.7", "<0.5"]:
        print(f"  {k:>8}: {conf_bins[k]}")

    e = sum(to_state2.values())
    if e:
        print(f"\ndigits still correct in these errors: {digits_ok_in_errors} / {e} ({100*digits_ok_in_errors/e:.1f}%)")
    print("most common true numbers among errors:", true_numbers.most_common(10))

    print("\ncolour-order test (same images, channels swapped BGR<->RGB):")
    print(f"  -> state 2 with normal input : {sum(to_state2.values())}")
    print(f"  -> state 2 with swapped input: {sum(to_state2_rgb.values())}")

    print(f"\nsaved: {opt.out}/state1_as_state2.csv and top {min(len(rows), opt.save_images)} images in {opt.out}/images/")


if __name__ == "__main__":
    main()
