"""
Compare crop geometry between:
  (a) your real JerseyNumber_Validation_Dataset (the model's training/
      validation-style annotator-drawn body.rect boxes)
  (b) the 89 RBNR samples (EasyOCR-detected tight number boxes) inside
      caltechfn_sample_350_clean

Checks TWO separate things, since they are not the same question:
  1. LABEL SCHEMA: does each RBNR label JSON have the same structure/keys
     as a real validation-dataset label JSON (tool_version, tool_type,
     file_name, dimensions, data.jersey[].body.rect, .jersey_number,
     .numbers[].digits/.rect)?
  2. CROP GEOMETRY: are the actual body.rect box sizes/aspect ratios in the
     same range, or is RBNR's crop tighter/looser than the real dataset's?
     Both ultimately get fed through the IDENTICAL crop-resize-96x96
     pipeline in test_onnx_accuracy.py, so this only matters for whether
     the boxes are *drawn* the same way, not whether they're *processed*
     the same way (that part is already guaranteed identical by using the
     same test script unmodified).

Usage:
    python compare_crop_geometry.py \
        --val_dataset ~/workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset/ \
        --merged_dataset ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean/
"""
import argparse
import glob
import json
import os
import statistics


def load_rects(dataset_dir, filter_fn=None):
    label_files = sorted(glob.glob(os.path.join(dataset_dir, "labels", "*.json")))
    rects = []
    schema_keys_seen = set()
    sample_label = None
    for lp in label_files:
        fname_base = os.path.basename(lp)
        if filter_fn and not filter_fn(fname_base):
            continue
        with open(lp, "r", encoding="utf8") as f:
            data = json.load(f)
        if sample_label is None:
            sample_label = data
        schema_keys_seen.update(data.keys())
        for item in data["data"]["jersey"]:
            x, y, w, h = item["body"]["rect"]
            if w <= 0 or h <= 0:
                continue
            rects.append({
                "w": w, "h": h, "aspect": w / h,
                "n_digits": len(item["numbers"]),
                "img_dims": data.get("dimensions"),
            })
    return rects, schema_keys_seen, sample_label


def report(name, rects):
    if not rects:
        print(f"{name}: no samples found")
        return
    ws = [r["w"] for r in rects]
    hs = [r["h"] for r in rects]
    ars = [r["aspect"] for r in rects]
    print(f"\n{name} (n={len(rects)}):")
    print(f"  width  -- median: {statistics.median(ws):.1f}  mean: {statistics.mean(ws):.1f}  "
          f"range: {min(ws):.1f}-{max(ws):.1f}")
    print(f"  height -- median: {statistics.median(hs):.1f}  mean: {statistics.mean(hs):.1f}  "
          f"range: {min(hs):.1f}-{max(hs):.1f}")
    print(f"  aspect ratio (w/h) -- median: {statistics.median(ars):.3f}  mean: {statistics.mean(ars):.3f}")
    rel = [(r["w"] / r["img_dims"][0], r["h"] / r["img_dims"][1])
           for r in rects if r.get("img_dims")]
    if rel:
        rel_w = [r[0] for r in rel]
        rel_h = [r[1] for r in rel]
        print(f"  relative width  (w/img_W) -- median: {statistics.median(rel_w):.3f}")
        print(f"  relative height (h/img_H) -- median: {statistics.median(rel_h):.3f}")
    return {
        "median_w": statistics.median(ws), "median_h": statistics.median(hs),
        "median_aspect": statistics.median(ars),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dataset", required=True)
    ap.add_argument("--merged_dataset", required=True)
    args = ap.parse_args()

    val_rects, val_keys, val_sample = load_rects(args.val_dataset)
    rbnr_rects, rbnr_keys, rbnr_sample = load_rects(
        args.merged_dataset,
        filter_fn=lambda f: f.startswith(("set2_org_", "set3_org_"))
    )
    caltech_rects, caltech_keys, caltech_sample = load_rects(
        args.merged_dataset,
        filter_fn=lambda f: not f.startswith(("set2_org_", "set3_org_"))
    )

    print("==== 1. LABEL SCHEMA CHECK ====")
    print(f"Validation dataset top-level keys: {sorted(val_keys)}")
    print(f"RBNR (in merged folder) top-level keys: {sorted(rbnr_keys)}")
    print(f"caltechfn (in merged folder) top-level keys: {sorted(caltech_keys)}")
    if val_keys == rbnr_keys:
        print("-> SAME schema keys as validation dataset.")
    else:
        print(f"-> DIFFERENT. Only in val: {val_keys - rbnr_keys} | Only in RBNR: {rbnr_keys - val_keys}")

    print("\nSample validation label (first one found):")
    print(json.dumps(val_sample, indent=2)[:800])
    print("\nSample RBNR label (first one found):")
    print(json.dumps(rbnr_sample, indent=2)[:800])

    print("\n\n==== 2. CROP GEOMETRY CHECK ====")
    val_stats = report("Validation dataset (ALL samples)", val_rects)
    val_state2_rects = [r for r in val_rects if r["n_digits"] > 2]
    val_state2_stats = report("Validation dataset (state2 / 3+ digit ONLY, fairest comparison)", val_state2_rects)
    rbnr_stats = report("RBNR (89 samples, EasyOCR tight crop)", rbnr_rects)
    caltech_state2_rects = [r for r in caltech_rects if r["n_digits"] > 2]
    report("caltechfn in merged folder (state2 / 3+ digit ONLY)", caltech_state2_rects)

    print("\n\n==== SUMMARY ====")
    if val_state2_stats and rbnr_stats:
        w_ratio = rbnr_stats["median_w"] / val_state2_stats["median_w"]
        h_ratio = rbnr_stats["median_h"] / val_state2_stats["median_h"]
        print(f"RBNR median box is {w_ratio:.2f}x the width and {h_ratio:.2f}x the height "
              f"of validation-set's state2 (3+ digit) boxes.")
        if 0.7 <= w_ratio <= 1.4 and 0.7 <= h_ratio <= 1.4:
            print("-> Crop tightness is COMPARABLE (within ~40%).")
        elif w_ratio < 0.7 or h_ratio < 0.7:
            print("-> RBNR crop is TIGHTER than the validation dataset's own state2 boxes.")
        else:
            print("-> RBNR crop is LOOSER than the validation dataset's own state2 boxes.")


if __name__ == "__main__":
    main()
