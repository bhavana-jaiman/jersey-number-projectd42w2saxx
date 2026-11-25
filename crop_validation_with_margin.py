"""
Re-crops validation images with a wider margin around the jersey number,
using the existing YOLO boxes in extra/validation_number_labels/ as the
box source (read-only -- this script never writes there), and the full
images in validation/images/ (also read-only -- never modified).

For each <hash>.txt in --box_labels_dir:
  1. Read its YOLO box(es): "class cx cy w h" (normalized 0-1, polygon
     lines also accepted, same parsing as merge_digit_boxes_v2.py).
  2. Widen each box by --pad (fraction of its own w/h, default 0.20),
     same center-preserving + edge-clamping logic as widen_yolo_boxes.py.
  3. Open the matching image from --images_dir, convert the widened
     normalized box to pixel coords, crop, and save to --out_dir/images/.
  4. Copy the ORIGINAL (unwidened) label line through to --out_dir/labels/
     for reference/traceability -- not used for cropping itself, just kept
     alongside so you can see what box the crop was based on.

If a .txt has multiple box lines (multiple jersey numbers in one image),
EACH one is cropped separately, suffixed _0, _1, ... on the output filename
-- same convention already used elsewhere in this dataset
(e.g. 00115DE78D3D130F0100D1DBC63D69EE3_0.jpg).

Nothing in extra/validation_number_labels/ or validation/images/ is ever
modified -- both are opened read-only. Output goes to a brand-new folder.

Usage (dry run first -- prints planned crops, writes nothing):
    python crop_validation_with_margin.py \
        --box_labels_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/extra/validation_number_labels \
        --images_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation/images \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_body_cropped_v2 \
        --pad 0.20 --dry_run --sample 5

Then for real:
    python crop_validation_with_margin.py \
        --box_labels_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/extra/validation_number_labels \
        --images_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation/images \
        --out_dir ~/workspace_bhavana/Datasets/70_30_dataset_v3/validation_body_cropped_v2 \
        --pad 0.20
"""
import argparse
import glob
import os

from PIL import Image


def read_boxes(path):
    """Same parsing as merge_digit_boxes_v2.py: returns list of
    (class_str, (cx, cy, w, h)) in normalized coords. Accepts plain YOLO
    (class cx cy w h [+conf]) and polygon lines (class x1 y1 x2 y2 ... ->
    its bounding box)."""
    boxes = []
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            t = line.split()
            if len(t) < 5:
                continue
            cls = t[0]
            v = [float(x) for x in t[1:]]
            if len(v) in (4, 5):
                cx, cy, w, h = v[:4]
                boxes.append((cls, (cx, cy, w, h)))
            elif len(v) >= 6 and len(v) % 2 == 0:
                xs, ys = v[0::2], v[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes.append((cls, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)))
    return boxes


def widen_box(cx, cy, w, h, pad):
    """Same center-preserving, edge-clamping widen logic as
    widen_yolo_boxes.py."""
    new_w = min(1.0, w * (1.0 + pad))
    new_h = min(1.0, h * (1.0 + pad))
    half_w, half_h = new_w / 2.0, new_h / 2.0
    new_cx = min(max(cx, half_w), 1.0 - half_w)
    new_cy = min(max(cy, half_h), 1.0 - half_h)
    return new_cx, new_cy, new_w, new_h


def find_image(images_dir, name):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = os.path.join(images_dir, name + ext)
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box_labels_dir", required=True, help="Folder of YOLO .txt boxes (read-only source), e.g. extra/validation_number_labels")
    ap.add_argument("--images_dir", required=True, help="Folder of full source images (read-only), e.g. validation/images")
    ap.add_argument("--out_dir", required=True, help="Output folder; images/ and labels/ subfolders are created here")
    ap.add_argument("--pad", type=float, default=0.20, help="Fractional extra margin on width/height before cropping (default: 0.20 = +20%%)")
    ap.add_argument("--dry_run", action="store_true", help="Only print planned crops for --sample files; write nothing")
    ap.add_argument("--sample", type=int, default=5, help="Number of files to show in --dry_run (default: 5)")
    args = ap.parse_args()

    box_files = sorted(glob.glob(os.path.join(args.box_labels_dir, "*.txt")))
    if not box_files:
        raise SystemExit(f"ERROR: no .txt files found in {args.box_labels_dir}")

    print(f"Found {len(box_files)} box label files in {args.box_labels_dir}")
    print(f"Reading images from: {args.images_dir}")
    print(f"Padding: +{args.pad*100:.0f}% on width/height before crop")
    print()

    todo = box_files[: args.sample] if args.dry_run else box_files
    if args.dry_run:
        print(f"--dry_run: showing up to {args.sample} samples, writing NOTHING.\n")

    n_planned = 0
    n_missing_image = 0
    n_no_boxes = 0
    n_written = 0

    if not args.dry_run:
        os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "labels"), exist_ok=True)

    for bf in todo:
        name = os.path.splitext(os.path.basename(bf))[0]
        boxes = read_boxes(bf)
        if not boxes:
            n_no_boxes += 1
            continue

        img_path = find_image(args.images_dir, name)
        if img_path is None:
            n_missing_image += 1
            if args.dry_run:
                print(f"{name}: NO MATCHING IMAGE FOUND in {args.images_dir} -- skipped")
            continue

        with Image.open(img_path) as im:
            W, H = im.size
            for i, (cls, (cx, cy, w, h)) in enumerate(boxes):
                ncx, ncy, nw, nh = widen_box(cx, cy, w, h, args.pad)
                left = int(round((ncx - nw / 2) * W))
                top = int(round((ncy - nh / 2) * H))
                right = int(round((ncx + nw / 2) * W))
                bottom = int(round((ncy + nh / 2) * H))
                left, top = max(0, left), max(0, top)
                right, bottom = min(W, right), min(H, bottom)

                suffix = f"_{i}" if len(boxes) > 1 else ""
                out_name = f"{name}{suffix}"

                if args.dry_run:
                    print(f"{name} box[{i}] (class={cls}):")
                    print(f"  image size: {W}x{H}")
                    print(f"  orig box (norm): cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
                    print(f"  widened (norm):  cx={ncx:.4f} cy={ncy:.4f} w={nw:.4f} h={nh:.4f}")
                    print(f"  crop pixels: left={left} top={top} right={right} bottom={bottom} "
                          f"({right-left}x{bottom-top})")
                    print(f"  would write: images/{out_name}.jpg + labels/{out_name}.txt")
                    print()
                    n_planned += 1
                    continue

                if right <= left or bottom <= top:
                    continue  # degenerate box, skip

                crop = im.crop((left, top, right, bottom))
                crop.save(os.path.join(args.out_dir, "images", out_name + ".jpg"))
                with open(os.path.join(args.out_dir, "labels", out_name + ".txt"), "w", encoding="utf8") as lf:
                    lf.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_written += 1

    if args.dry_run:
        print(f"Planned {n_planned} crops from {len(todo)} label files shown "
              f"({n_no_boxes} had no parseable box, {n_missing_image} had no matching image).")
        print("Re-run without --dry_run to actually write the cropped dataset.")
    else:
        print(f"Done. Wrote {n_written} cropped images + labels to:")
        print(f"  {os.path.join(args.out_dir, 'images')}")
        print(f"  {os.path.join(args.out_dir, 'labels')}")
        if n_no_boxes:
            print(f"  NOTE: {n_no_boxes} label files had no parseable box (skipped).")
        if n_missing_image:
            print(f"  NOTE: {n_missing_image} label files had no matching image in {args.images_dir} (skipped).")


if __name__ == "__main__":
    main()
