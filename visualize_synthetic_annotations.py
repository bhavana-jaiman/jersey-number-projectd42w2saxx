#!/usr/bin/env python3
"""
Visualize YOLO annotations on synthetic jersey images WITHOUT cv2.imshow().
The script reads each image and its matching .txt file, draws every
bounding box + digit/class on the image, and SAVES the visualization.

No XCB / GUI is required.

Example:
python visualize_synthetic_annotations.py \
    --input-dir demo3/complex2D \
    --output-dir demo3/complex2D_visualized \
    --num-images 20

To process all images, omit --num-images.
"""

import argparse
import random
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yolo_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"[WARNING] Invalid label: {label_path}:{line_no}")
                continue

            try:
                cls_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(f"[WARNING] Cannot parse: {label_path}:{line_no}")
                continue

            labels.append((cls_id, xc, yc, w, h))

    return labels


def draw_annotations(image, labels):
    img = image.copy()
    H, W = img.shape[:2]

    for cls_id, xc, yc, w, h in labels:
        # YOLO normalized -> pixel coordinates
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)

        # Clamp to image
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Digit/class text
        text = f"{cls_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(1.0, W / 900.0))
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Put text above the box when possible
        tx = x1
        ty = y1 - 6

        if ty - th < 0:
            ty = y1 + th + 6

        # Background behind text
        cv2.rectangle(
            img,
            (tx, ty - th - baseline),
            (tx + tw + 4, ty + 3),
            (0, 255, 0),
            -1,
        )

        # Black text for readability
        cv2.putText(
            img,
            text,
            (tx + 2, ty),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return img


def collect_images(input_dir):
    return sorted(
        p for p in Path(input_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Draw YOLO annotations on synthetic images and save them."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to visualize. Omit to process all images.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select images instead of taking the first N.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)

    if not images:
        print(f"[ERROR] No images found in: {input_dir}")
        return

    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(images)

    if args.num_images is not None:
        images = images[:args.num_images]

    print("=" * 70)
    print("ANNOTATION VISUALIZATION")
    print("=" * 70)
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")
    print("=" * 70)

    processed = 0
    missing_labels = 0
    total_boxes = 0

    for idx, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Cannot read image: {image_path}")
            continue

        label_path = image_path.with_suffix(".txt")
        labels = read_yolo_labels(label_path)

        if not label_path.exists():
            missing_labels += 1

        total_boxes += len(labels)

        annotated = draw_annotations(image, labels)

        # Preserve relative folder structure
        relative = image_path.relative_to(input_dir)
        out_path = output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JPG/PNG using original extension
        ok = cv2.imwrite(str(out_path), annotated)

        if not ok:
            print(f"[SKIP] Could not save: {out_path}")
            continue

        processed += 1

        print(
            f"[{idx}/{len(images)}] "
            f"{image_path.name} | boxes={len(labels)} | "
            f"saved={out_path}"
        )

    print("=" * 70)
    print("FINISHED")
    print(f"Processed images : {processed}")
    print(f"Missing labels   : {missing_labels}")
    print(f"Total boxes      : {total_boxes}")
    print(f"Output directory : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
