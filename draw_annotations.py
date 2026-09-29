"""
Draw test-set annotations (body boxes + digit boxes) and flag suspicious labels.

Expected layout:
    DATASET_ROOT/images/xxx.jpg
    DATASET_ROOT/labels/xxx.json

Usage:
    python draw_annotations.py --root ../Datasets/JerseyNumber_Validation_Dataset --out annotation_check
    python draw_annotations.py --root ... --out ... --max 200          # only first 200 images
    python draw_annotations.py --root ... --out ... --only_problems    # save only flagged images
"""
import os
import glob
import json
import argparse
from collections import Counter

import cv2

GREEN = (0, 200, 0)      # body box
RED = (0, 0, 255)        # digit box
YELLOW = (0, 255, 255)   # text
ORANGE = (0, 140, 255)   # problem text


def inside(d, b, tol=3):
    """digit rect d fully inside body rect b (both [x, y, w, h]), with a small tolerance."""
    return (d[0] >= b[0] - tol and d[1] >= b[1] - tol and
            d[0] + d[2] <= b[0] + b[2] + tol and d[1] + d[3] <= b[1] + b[3] + tol)


def check_body(entry, img_w, img_h):
    """Return (problems list, digits list [(digit, rect)])."""
    problems = []
    body = [float(v) for v in entry["body"]["rect"]]
    jn = entry.get("jersey_number", [])
    nums = entry.get("numbers", [])

    if len(nums) < len(jn):
        problems.append(f"numbers({len(nums)}) < jersey_number({len(jn)})")
    digits = []
    for j in range(min(len(jn), len(nums))):
        d = int(nums[j]["digits"])
        r = [float(v) for v in nums[j]["rect"]]
        digits.append((d, r))
        if d != int(jn[j]):
            problems.append(f"digit mismatch: numbers={d} jersey_number={jn[j]}")
        if r[2] <= 0 or r[3] <= 0:
            problems.append(f"digit {d} has empty box")
        elif not inside(r, body):
            problems.append(f"digit {d} box outside body")

    if body[2] <= 0 or body[3] <= 0:
        problems.append("empty body box")
    if body[0] < 0 or body[1] < 0 or body[0] + body[2] > img_w + 1 or body[1] + body[3] > img_h + 1:
        problems.append("body box outside image")

    xs = [r[0] for _, r in digits]
    if xs != sorted(xs):
        problems.append("digits not in left->right order")
    if len(digits) == 0:
        problems.append("no digits (state 0)")
    return problems, digits


def draw(img, entry, digits, problems, idx):
    bx, by, bw, bh = [int(round(float(v))) for v in entry["body"]["rect"]]
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), GREEN, 2)
    number = "".join(str(d) for d, _ in sorted(digits, key=lambda t: t[1][0])) or "none"
    cv2.putText(img, f"#{idx}: {number}", (bx, max(12, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    for d, r in digits:
        x, y, w, h = [int(round(v)) for v in r]
        cv2.rectangle(img, (x, y), (x + w, y + h), RED, 1)
        cv2.putText(img, str(d), (x, y + h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)
    if problems:
        cv2.putText(img, "!", (bx + bw - 10, by + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder with images/ and labels/")
    ap.add_argument("--out", default="annotation_check", help="output folder")
    ap.add_argument("--max", type=int, default=None, help="only process first N images")
    ap.add_argument("--only_problems", action="store_true", help="save only images with problems")
    opt = ap.parse_args()

    os.makedirs(os.path.join(opt.out, "all"), exist_ok=True)
    os.makedirs(os.path.join(opt.out, "problems"), exist_ok=True)

    images = sorted(glob.glob(os.path.join(opt.root, "images", "*.jpg")))
    if opt.max:
        images = images[:opt.max]

    stats = Counter()
    problem_lines = []

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(opt.root, "labels", name + ".json")
        if not os.path.exists(label_path):
            problem_lines.append(f"{name}: missing label json")
            stats["missing label"] += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            problem_lines.append(f"{name}: cannot read image")
            stats["unreadable image"] += 1
            continue
        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf8") as f:
            data = json.load(f)

        dims = data.get("dimensions")
        if dims and (int(dims[0]) != w or int(dims[1]) != h):
            problem_lines.append(f"{name}: json dimensions {dims} != image {w}x{h}")
            stats["dimension mismatch"] += 1

        image_has_problem = False
        for idx, entry in enumerate(data["data"]["jersey"]):
            problems, digits = check_body(entry, w, h)
            n = len(digits)
            stats["bodies"] += 1
            stats[f"{n if n <= 2 else '3+'} digits"] += 1
            real = [p for p in problems if not p.startswith("no digits")]
            if real:
                image_has_problem = True
                stats["bodies with problems"] += 1
                for p in real:
                    problem_lines.append(f"{name} body#{idx}: {p}")
            draw(img, entry, digits, problems, idx)

        stats["images"] += 1
        if image_has_problem:
            stats["images with problems"] += 1
            cv2.imwrite(os.path.join(opt.out, "problems", name + ".jpg"), img)
        if not opt.only_problems:
            cv2.imwrite(os.path.join(opt.out, "all", name + ".jpg"), img)

    with open(os.path.join(opt.out, "problems.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(problem_lines))

    def kind(line):
        msg = line.split(": ", 1)[1]
        for key in ["digit mismatch", "box outside body", "empty box", "numbers(",
                    "empty body box", "body box outside image", "not in left->right order",
                    "missing label", "cannot read image", "json dimensions"]:
            if key in msg:
                return key.rstrip("(")
        return msg
    kinds = Counter(kind(line) for line in problem_lines)
    print("\n---- summary ----")
    for k in ["images", "bodies", "1 digits", "2 digits", "3+ digits", "0 digits",
              "images with problems", "bodies with problems",
              "missing label", "unreadable image", "dimension mismatch"]:
        if stats[k]:
            print(f"{k:<22}: {stats[k]}")
    if kinds:
        print("\nproblem types:")
        for k, v in kinds.most_common():
            print(f"  {v:5d}  {k}")
    print(f"\ndrawn images   : {opt.out}/all/ (unless --only_problems)")
    print(f"flagged images : {opt.out}/problems/")
    print(f"problem list   : {opt.out}/problems.txt")


if __name__ == "__main__":
    main()
