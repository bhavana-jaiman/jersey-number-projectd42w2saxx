#!/usr/bin/env python3
"""
Draw jersey body boxes and digit boxes from JSON labels onto the images.

Usage:
    python draw_boxes.py --images path/to/images --labels path/to/labels --output path/to/vis
    python draw_boxes.py ... --only-numbers     # only images that have a jersey number

rect format assumed: [x, y, width, height]
"""
import argparse
import json
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get(d, key, default=None):
    for k in (key, key.replace(" ", "_"), key.replace("_", " ")):
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def walk(obj):
    """Yield every dict at any nesting depth."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def flatten(obj):
    """Turn [[{..}, {..}]] into a flat list of dicts."""
    if isinstance(obj, list):
        for v in obj:
            yield from flatten(v)
    elif isinstance(obj, dict):
        yield obj


def jerseys(label):
    for d in walk(label):
        if get(d, "jersey number") is not None or get(d, "numbers") is not None:
            yield d


def draw(img, label):
    found_number = False
    for j in jerseys(label):
        number = get(j, "jersey number") or []
        found_number |= bool(number)

        # Body box (blue) + full jersey number text
        for body in flatten(get(j, "body")):
            if "rect" in body:
                x, y, w, h = map(int, body["rect"])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, "#" + "".join(map(str, number)), (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Digit boxes (green) + digit value (red)
        for n in flatten(get(j, "numbers", [])):
            if "rect" in n:
                x, y, w, h = map(int, n["rect"])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, str(n.get("digits", "?")), (x, max(y - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return img, found_number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only-numbers", action="store_true",
                    help="save only images whose label has a jersey number")
    args = ap.parse_args()

    norm = lambda name: Path(name).stem.lower().replace(" ", "_")
    images = {norm(p.name): p for p in Path(args.images).rglob("*")
              if p.suffix.lower() in IMG_EXTS}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    saved = missing = 0
    for lp in sorted(Path(args.labels).rglob("*.json")):
        label = json.loads(lp.read_text(encoding="utf-8"))
        img_path = images.get(norm(lp.name)) or images.get(norm(get(label, "file name", "")))
        if img_path is None:
            missing += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img, has_number = draw(img, label)
        if args.only_numbers and not has_number:
            continue
        cv2.imwrite(str(out / f"{img_path.stem}_vis.jpg"), img)
        saved += 1

    print(f"Saved {saved} images to {out.resolve()}  (labels without image: {missing})")


if __name__ == "__main__":
    main()
