"""
Step 1: Measure the typical body.rect geometry (width, height, aspect ratio,
and position relative to image size) from your REAL validation dataset
(JerseyNumber_Validation_Dataset), since that's the crop distribution the
model was actually trained/calibrated on.

Step 2: Rebuild RBNR's body.rect boxes to match that geometry as closely as
possible, starting from the existing YOLO full-person box (yolo_boxes.json,
already computed) and tightening it toward the measured aspect ratio /
relative size -- instead of using the full person box, which we now know
triggers the model's "state=0 / not a number" reject class.

Approach for step 2 (per RBNR image):
  - Take the YOLO person box [x1,y1,x2,y2].
  - The number/bib typically sits in the upper-middle portion of a person
    box (chest height), so we crop a sub-region of the YOLO box:
      - horizontally: centered, width = --width_frac of the YOLO box width
      - vertically: starts at --top_frac down from the top of the YOLO box,
        height = --height_frac of the YOLO box height
  - Then we adjust that sub-region's aspect ratio to match the MEASURED
    median aspect ratio from your real dataset (pad the shorter dimension,
    not crop the longer one, so we don't cut into the number).

This prints the measured stats first so you (and I) can sanity check them
before anything is rebuilt, then writes the new RBNR label JSONs.

Usage:
    # Step 1 only (just measure, don't rebuild anything yet):
    python measure_and_rebuild_rbnr_crops.py --measure_only \
        --val_dataset ~/workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset/

    # Full run (measure + rebuild RBNR crops):
    python measure_and_rebuild_rbnr_crops.py \
        --val_dataset ~/workspace_bhavana/Datasets/JerseyNumber_Validation_Dataset/ \
        --rbnr_clean_dir <path to where clean/, labels_clean/, yolo_boxes.json live> \
        --dst <output folder for rebuilt RBNR dataset>
"""
import argparse
import glob
import json
import os
import statistics
from pathlib import Path

from PIL import Image


def measure_validation_geometry(val_dataset):
    label_files = sorted(glob.glob(os.path.join(val_dataset, "labels", "*.json")))
    assert label_files, f"no label JSONs found under {val_dataset}/labels"

    ws, hs, aspect_ratios, rel_w, rel_h = [], [], [], [], []

    for lp in label_files:
        with open(lp, "r", encoding="utf8") as f:
            data = json.load(f)
        dims = data.get("dimensions")
        for item in data["data"]["jersey"]:
            x, y, w, h = item["body"]["rect"]
            if w <= 0 or h <= 0:
                continue
            ws.append(w)
            hs.append(h)
            aspect_ratios.append(w / h)
            if dims:
                W, H = dims
                if W > 0 and H > 0:
                    rel_w.append(w / W)
                    rel_h.append(h / H)

    print(f"Measured from {len(ws)} body.rect boxes in {val_dataset}")
    print(f"  width  -- median: {statistics.median(ws):.1f}  mean: {statistics.mean(ws):.1f}")
    print(f"  height -- median: {statistics.median(hs):.1f}  mean: {statistics.mean(hs):.1f}")
    print(f"  aspect ratio (w/h) -- median: {statistics.median(aspect_ratios):.3f}  "
          f"mean: {statistics.mean(aspect_ratios):.3f}  "
          f"range: {min(aspect_ratios):.3f}-{max(aspect_ratios):.3f}")
    if rel_w:
        print(f"  relative width (w/img_W)  -- median: {statistics.median(rel_w):.3f}")
        print(f"  relative height (h/img_H) -- median: {statistics.median(rel_h):.3f}")

    return {
        "median_aspect_ratio": statistics.median(aspect_ratios),
        "median_rel_w": statistics.median(rel_w) if rel_w else None,
        "median_rel_h": statistics.median(rel_h) if rel_h else None,
    }


def rebuild_rbnr_crops(rbnr_clean_dir, dst, geom, top_frac, width_frac, height_frac):
    rbnr_clean_dir = Path(rbnr_clean_dir)
    dst = Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    yolo_boxes = json.loads((rbnr_clean_dir / "yolo_boxes.json").read_text())

    MISMATCHES = {
        "set2_org_set2_28_0", "set2_org_set2_31_0",
        "set2_org_set2_63_0", "set3_org_set3_04_0",
    }

    target_ar = geom["median_aspect_ratio"]
    n_ok = 0

    for label_path in sorted((rbnr_clean_dir / "labels_clean").glob("*.json")):
        stem = label_path.stem
        if stem in MISMATCHES:
            continue

        data = json.loads(label_path.read_text())
        img_name = data["file_name"]

        box = yolo_boxes.get(img_name) or yolo_boxes.get(stem)
        if box is None:
            for k in yolo_boxes:
                if Path(k).stem == stem:
                    box = yolo_boxes[k]
                    break
        if box is None:
            continue

        x1, y1, x2, y2 = box
        yw, yh = x2 - x1, y2 - y1

        # sub-region: centered horizontally, upper portion vertically
        sub_w = yw * width_frac
        sub_h = yh * height_frac
        sub_x1 = x1 + (yw - sub_w) / 2
        sub_y1 = y1 + yh * top_frac

        # adjust to target aspect ratio by padding the shorter side
        cur_ar = sub_w / sub_h if sub_h > 0 else target_ar
        if cur_ar > target_ar:
            # too wide -> increase height
            new_h = sub_w / target_ar
            sub_y1 -= (new_h - sub_h) / 2
            sub_h = new_h
        else:
            # too tall -> increase width
            new_w = sub_h * target_ar
            sub_x1 -= (new_w - sub_w) / 2
            sub_w = new_w

        img_path = rbnr_clean_dir / "clean" / img_name
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            W, H = im.size
            fx1 = max(0.0, min(sub_x1, W))
            fy1 = max(0.0, min(sub_y1, H))
            fx2 = max(0.0, min(sub_x1 + sub_w, W))
            fy2 = max(0.0, min(sub_y1 + sub_h, H))
            if fx2 <= fx1 or fy2 <= fy1:
                continue
            body_rect = [round(fx1, 1), round(fy1, 1), round(fx2 - fx1, 1), round(fy2 - fy1, 1)]

            jersey = data["data"]["jersey"][0]
            jersey["body"]["rect"] = body_rect
            for n in jersey["numbers"]:
                n["rect"] = body_rect

            im.convert("RGB").save(dst / "images" / img_name, quality=95)
            (dst / "labels" / label_path.name).write_text(json.dumps(data))
            n_ok += 1

    print(f"\nRebuilt {n_ok} RBNR samples with torso-tightened crops at: {dst}")
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dataset", required=True)
    ap.add_argument("--measure_only", action="store_true")
    ap.add_argument("--rbnr_clean_dir", default=None,
                     help="folder containing clean/, labels_clean/, yolo_boxes.json")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--top_frac", type=float, default=0.15,
                     help="start the sub-crop this far down from the top of the YOLO box (chest region)")
    ap.add_argument("--width_frac", type=float, default=0.75,
                     help="sub-crop width as a fraction of the YOLO box width")
    ap.add_argument("--height_frac", type=float, default=0.35,
                     help="sub-crop height as a fraction of the YOLO box height")
    args = ap.parse_args()

    geom = measure_validation_geometry(args.val_dataset)

    if args.measure_only:
        return

    if not args.rbnr_clean_dir or not args.dst:
        print("\n--rbnr_clean_dir and --dst are required to rebuild crops (omit --measure_only).")
        return

    rebuild_rbnr_crops(args.rbnr_clean_dir, args.dst, geom, args.top_frac, args.width_frac, args.height_frac)


if __name__ == "__main__":
    main()
