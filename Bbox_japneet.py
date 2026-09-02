"""
Draw YOLO-format bounding boxes on images to verify annotation correctness.

Expected layout:
    images/  -> img1.jpg, img2.png, ...
    labels/  -> img1.txt, img2.txt, ...   (one .txt per image, same stem)

Label line format (normalized, space-separated):
    class_id x_center y_center width height

Usage:
    python visualize_yolo_labels.py \
        --images /path/to/images \
        --labels /path/to/labels \
        --out /path/to/output_vis \
        --classes class0 class1 class2   # optional, else shows raw class_id
        --num 50                          # optional, limit how many to process
"""

import argparse
import os
import random
import cv2


def load_class_names(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def yolo_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h
    x1 = int(round(x_c - w / 2))
    y1 = int(round(y_c - h / 2))
    x2 = int(round(x_c + w / 2))
    y2 = int(round(y_c + h / 2))
    return x1, y1, x2, y2


def draw_annotations(img_path, label_path, class_names=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] could not read image: {img_path}")
        return None

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"[WARN] no label file for: {img_path}")
        return img

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            print(f"[WARN] malformed line in {label_path}: '{line}'")
            continue

        cls_id = int(float(parts[0]))
        x_c, y_c, bw, bh = map(float, parts[1:5])

        x1, y1, x2, y2 = yolo_to_xyxy(x_c, y_c, bw, bh, w, h)

        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir containing images")
    ap.add_argument("--labels", required=True, help="dir containing YOLO .txt labels")
    ap.add_argument("--out", required=True, help="output dir for visualized images")
    ap.add_argument("--classes", default=None, help="path to classes.txt (one name per line), optional")
    ap.add_argument("--num", type=int, default=None, help="limit number of images processed (random sample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    class_names = load_class_names(args.classes)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(valid_ext))

    if args.num is not None and args.num < len(img_files):
        random.seed(args.seed)
        img_files = random.sample(img_files, args.num)

    processed = 0
    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.images, fname)
        label_path = os.path.join(args.labels, stem + ".txt")

        vis = draw_annotations(img_path, label_path, class_names)
        if vis is None:
            continue

        cv2.imwrite(os.path.join(args.out, fname), vis)
        processed += 1

    print(f"Done. {processed}/{len(img_files)} images written to {args.out}")


if __name__ == "__main__":
    main()
 
