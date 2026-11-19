"""
Visual diagnostic for the two failure modes found in the failure-mode
analysis plots (per_class_d1/d2, angleAnaly_d1/d2, scaleAnaly_d1/d2/state):

  1. SCALE: accuracy collapses at small digit-scale buckets (~17-84px,
     which is ~70% of the validation set by sample count). Need to see
     real examples to tell whether this is a genuine resolution/detail
     problem (crop is just too small to read) or a crop/labeling artifact
     (similar to the caltechfn truncation issue found earlier).

  2. CLASS: Digit_7 (and to a lesser extent Digit_4) is consistently the
     weakest class across the confidence-threshold range in BOTH digit
     positions. Need to see real misclassifications to check what Digit_7
     is being confused for.

For each failure mode, saves annotated crops (model input at 96x96,
upscaled for visibility, plus the body.rect box drawn on original-scale
context) so they can be visually reviewed before deciding on a fix.

Usage:
    python inspect_small_scale_failures.py \
        --model post_bias_calibration/kd_79.74_s2b1.25_dig.pth \
        --test_image_path ~/workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset/ \
        --out_dir failure_mode_crops
"""
import argparse
import glob
import json
import math
import os

import cv2
import numpy as np
import torch


def load_model(checkpoint_path, device, high_res=False, script_dir="."):
    """Load the PyTorch checkpoint exactly as
    jerseyNumberRecognitiorWithState_Testing.py does:

        from subModules.backbone_ying import MultiTaskLearnerWithState
        model = MultiTaskLearnerWithState(high_res=opt.high_res).to(device)
        pretrained_dict = torch.load(opt.checkpoint)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict, strict=True)
    """
    import sys
    sys.path.insert(0, script_dir)
    from subModules.backbone_ying import MultiTaskLearnerWithState  # noqa

    model = MultiTaskLearnerWithState(high_res=high_res).to(device)

    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in pretrained_dict:
        pretrained_dict = pretrained_dict["state_dict"]
    model.load_state_dict(pretrained_dict, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_image_path", required=True)
    ap.add_argument("--out_dir", default="failure_mode_crops")
    ap.add_argument("--n_per_bucket", type=int, default=40,
                     help="how many example crops to save per category")
    ap.add_argument("--sharpness_thresh", type=float, default=30.0,
                     help="Laplacian-variance threshold below which a crop is "
                          "flagged 'likely illegible' (blurry/low detail/occluded). "
                          "Lower = stricter (fewer flagged). Tune by checking a few "
                          "borderline examples visually.")
    ap.add_argument("--high_res", action="store_true",
                     help="pass this if the checkpoint was trained with --high_res")
    ap.add_argument("--script_dir", default=".",
                     help="directory containing subModules/backbone_ying.py "
                          "(usually old_server_backup_16_10_2026/New_16_10_2026)")
    opt = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    small_scale_dir = os.path.join(opt.out_dir, "small_scale_wrong")
    digit7_dir = os.path.join(opt.out_dir, "digit7_wrong")
    likely_illegible_dir = os.path.join(small_scale_dir, "likely_illegible")
    likely_legible_wrong_dir = os.path.join(small_scale_dir, "likely_legible_wrong")
    os.makedirs(likely_illegible_dir, exist_ok=True)
    os.makedirs(likely_legible_wrong_dir, exist_ok=True)
    os.makedirs(digit7_dir, exist_ok=True)

    model = load_model(opt.model, device, high_res=opt.high_res, script_dir=opt.script_dir)

    images = sorted(glob.glob(os.path.join(opt.test_image_path, "images", "*.jpg")))
    print(f"Found {len(images)} images")

    small_scale_saved = 0
    digit7_saved = 0
    small_scale_total = 0
    small_scale_wrong = 0
    n_illegible = 0
    n_legible_wrong = 0
    confusion_pairs = {}  # (gt_digit, pred_digit) -> count, for legible-wrong only (small-scale only)

    # GLOBAL (all scales) confusion tally + per-digit ground-truth frequency,
    # to check whether the "collapse toward/from 1" pattern is scale-specific
    # or a dataset-wide class-imbalance / decision-boundary issue.
    global_confusion_pairs = {}  # (gt_digit, pred_digit) -> count, ALL scales, ALL wrong samples
    gt_digit_freq = {}           # digit -> count of appearances in ground truth (both positions)
    gt_digit_wrong_count = {}    # digit -> count of times this gt digit was predicted wrong

    for img_path in images:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".json")
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)
        org_bgr = cv2.imread(img_path)
        if org_bgr is None:
            continue
        org = cv2.cvtColor(org_bgr, cv2.COLOR_BGR2RGB)
        H, W = org_bgr.shape[:2]

        for item in data["data"]["jersey"]:
            digits_gt = [n["digits"] for n in item["numbers"]][: len(item["jersey_number"])]
            if len(digits_gt) == 0 or len(digits_gt) > 2:
                continue  # focus on 1-2 digit samples where d1/d2 heads apply
            x, y, w, h = item["body"]["rect"]
            if w <= 0 or h <= 0:
                continue
            scale = math.sqrt(w * h)

            crop_raw = org[int(y): int(y + h), int(x): int(x + w)]
            if crop_raw.size == 0:
                continue
            crop_96 = cv2.resize(crop_raw, (96, 96))
            crop = crop_96.astype(np.float32) / 255.0
            inp = torch.from_numpy(crop.transpose(2, 0, 1)[None]).float().to(device)

            with torch.no_grad():
                _, logits_d1, logits_d2, logits_state = model(inp)

            pred_d1 = int(torch.max(logits_d1, 1)[1].item())
            gt_d1 = digits_gt[0]
            correct_d1 = pred_d1 == gt_d1

            pred_d2 = None
            gt_d2 = None
            correct_d2 = True
            if len(digits_gt) == 2:
                pred_d2 = int(torch.max(logits_d2, 1)[1].item())
                gt_d2 = digits_gt[1]
                correct_d2 = pred_d2 == gt_d2

            overall_correct = correct_d1 and correct_d2

            # ---- global (all scales) class-frequency + confusion tally ----
            gt_digit_freq[gt_d1] = gt_digit_freq.get(gt_d1, 0) + 1
            if not correct_d1:
                gt_digit_wrong_count[gt_d1] = gt_digit_wrong_count.get(gt_d1, 0) + 1
                global_confusion_pairs[(gt_d1, pred_d1)] = global_confusion_pairs.get((gt_d1, pred_d1), 0) + 1
            if gt_d2 is not None:
                gt_digit_freq[gt_d2] = gt_digit_freq.get(gt_d2, 0) + 1
                if not correct_d2:
                    gt_digit_wrong_count[gt_d2] = gt_digit_wrong_count.get(gt_d2, 0) + 1
                    global_confusion_pairs[(gt_d2, pred_d2)] = global_confusion_pairs.get((gt_d2, pred_d2), 0) + 1

            # ---- category 1: small scale (<=84px) ----
            if scale <= 84:
                small_scale_total += 1
                if not overall_correct:
                    small_scale_wrong += 1

                    # Heuristic legibility check: Laplacian variance on the
                    # grayscale crop as a sharpness/detail proxy. Low value
                    # => blurry / low-detail / possibly occluded crop, where
                    # the ground-truth digit is unlikely to be readable even
                    # by a human. This is a heuristic signal to PRIORITIZE
                    # review, not a hard verdict -- it will misfire on some
                    # samples (e.g. a sharp but occluding hand/arm scores
                    # high sharpness despite hiding the number).
                    gray = cv2.cvtColor(crop_96, cv2.COLOR_RGB2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    is_likely_illegible = sharpness < opt.sharpness_thresh

                    if is_likely_illegible:
                        n_illegible += 1
                    else:
                        n_legible_wrong += 1
                        if pred_d1 != gt_d1:
                            confusion_pairs[(gt_d1, pred_d1)] = confusion_pairs.get((gt_d1, pred_d1), 0) + 1
                        if pred_d2 is not None and pred_d2 != gt_d2:
                            confusion_pairs[(gt_d2, pred_d2)] = confusion_pairs.get((gt_d2, pred_d2), 0) + 1

                    if small_scale_saved < opt.n_per_bucket:
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        gt_str = "".join(str(d) for d in digits_gt)
                        pred_str = str(pred_d1) + (str(pred_d2) if pred_d2 is not None else "")
                        exact_input = cv2.cvtColor(crop_96, cv2.COLOR_RGB2BGR)
                        exact_input_big = cv2.resize(exact_input, (384, 384), interpolation=cv2.INTER_NEAREST)
                        label_txt = f"gt={gt_str} pred={pred_str} scale={scale:.0f}px sharp={sharpness:.0f}"
                        cv2.putText(exact_input_big, label_txt, (5, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                        target_dir = likely_illegible_dir if is_likely_illegible else likely_legible_wrong_dir
                        cv2.imwrite(os.path.join(target_dir, f"{stem}__{gt_str}_vs_{pred_str}.jpg"),
                                    exact_input_big)
                        small_scale_saved += 1

            # ---- category 2: digit-7 confusions (either position) ----
            is_digit7_case = (gt_d1 == 7 and not correct_d1) or (gt_d2 == 7 and not correct_d2)
            if is_digit7_case and digit7_saved < opt.n_per_bucket:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                gt_str = "".join(str(d) for d in digits_gt)
                pred_str = str(pred_d1) + (str(pred_d2) if pred_d2 is not None else "")
                exact_input = cv2.cvtColor(crop_96, cv2.COLOR_RGB2BGR)
                exact_input_big = cv2.resize(exact_input, (384, 384), interpolation=cv2.INTER_NEAREST)
                label_txt = f"gt={gt_str} pred={pred_str} scale={scale:.0f}px"
                cv2.putText(exact_input_big, label_txt, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(digit7_dir, f"{stem}__{gt_str}_vs_{pred_str}.jpg"),
                            exact_input_big)
                digit7_saved += 1

    print()
    print(f"Small-scale (<=84px) samples: {small_scale_total}, wrong: {small_scale_wrong} "
          f"({100*small_scale_wrong/max(1,small_scale_total):.1f}%)")
    print()
    print(f"Of the {small_scale_wrong} wrong small-scale samples (heuristic, sharpness_thresh={opt.sharpness_thresh}):")
    print(f"  Likely ILLEGIBLE (blurry/occluded/low-detail, not a real model fault): "
          f"{n_illegible} ({100*n_illegible/max(1,small_scale_wrong):.1f}%)")
    print(f"  Likely LEGIBLE but still wrong (genuine model error, fixable): "
          f"{n_legible_wrong} ({100*n_legible_wrong/max(1,small_scale_wrong):.1f}%)")
    print(f"  (this split is a heuristic signal to prioritize review, not a hard verdict --")
    print(f"   check the saved crops in both subfolders to calibrate/sanity-check it)")
    print()
    if confusion_pairs:
        print("Top confusion pairs among likely-legible-but-wrong samples (gt -> pred : count):")
        for (gt, pred), cnt in sorted(confusion_pairs.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {gt} -> {pred} : {cnt}")
    print()
    print(f"Saved {small_scale_saved} example crops -> {likely_illegible_dir}/ and {likely_legible_wrong_dir}/")
    print(f"Saved {digit7_saved} digit-7 confusion crops -> {digit7_dir}/")

    # ==================================================================
    # GLOBAL (all scales, all 1-2 digit samples) analysis -- answers:
    # is the "collapse toward/from 1" pattern scale-specific, or does it
    # show up everywhere? And is digit 1 over/under-represented in the
    # ground-truth label distribution (class imbalance check)?
    # ==================================================================
    print()
    print("=" * 70)
    print("GLOBAL ANALYSIS (all scales, not just <=84px)")
    print("=" * 70)
    print()
    print("Ground-truth digit frequency (how often each digit 0-9 appears as a label):")
    total_digit_appearances = sum(gt_digit_freq.values())
    for d in range(10):
        freq = gt_digit_freq.get(d, 0)
        wrong = gt_digit_wrong_count.get(d, 0)
        pct_of_data = 100 * freq / max(1, total_digit_appearances)
        err_rate = 100 * wrong / max(1, freq)
        print(f"  digit {d}: appears {freq:5d} times ({pct_of_data:4.1f}% of all digit labels), "
              f"wrong {wrong:4d} times (error rate within this digit: {err_rate:5.1f}%)")

    print()
    print("Top GLOBAL confusion pairs (gt -> pred : count), ALL scales combined:")
    for (gt, pred), cnt in sorted(global_confusion_pairs.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {gt} -> {pred} : {cnt}")

    # Specifically answer: of ALL wrong predictions (any scale), what
    # fraction predicted "1", and what fraction had gt="1"?
    total_wrong_global = sum(global_confusion_pairs.values())
    predicted_as_1 = sum(cnt for (gt, pred), cnt in global_confusion_pairs.items() if pred == 1)
    gt_was_1 = sum(cnt for (gt, pred), cnt in global_confusion_pairs.items() if gt == 1)
    print()
    print(f"Of {total_wrong_global} total wrong digit predictions (all scales):")
    print(f"  predicted as '1' (regardless of true digit): {predicted_as_1} "
          f"({100*predicted_as_1/max(1,total_wrong_global):.1f}%)")
    print(f"  true digit was '1' (regardless of wrong prediction): {gt_was_1} "
          f"({100*gt_was_1/max(1,total_wrong_global):.1f}%)")
    print(f"  (compare to digit 1's share of all ground-truth labels: "
          f"{100*gt_digit_freq.get(1,0)/max(1,total_digit_appearances):.1f}% -- "
          f"if '1' is predicted far more than its label share, that's a bias, not just confusability)")


if __name__ == "__main__":
    main()
