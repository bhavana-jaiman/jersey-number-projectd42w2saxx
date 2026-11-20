#!/usr/bin/env python3
"""
safe_crop_onnx.py -- aspect-matching CROP ONLY (no padding/resize), verified by YOUR model so digits are never cut.

For each image: try windows (tightest -> loosest) at 3 positions; keep the tightest window whose prediction
still equals the label. If the model can't read the ORIGINAL correctly, the image is copied unchanged ('unverified').
Same file names, labels copied untouched, the original folder is never modified.

Edit predictor_onnx.py (load() and predict()) to call your recognizer.
  python safe_crop_onnx.py SRC --out OUT --target-aspect 0.786 --tol 0.15 --limit 50      # try 50 images first
  python safe_crop_onnx.py SRC --out OUT --target-aspect 0.786 --tol 0.15 --copy-labels
"""
import argparse, csv, math, os, shutil, sys
from collections import Counter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predictor_onnx as predictor

EXT = (".jpg", ".jpeg", ".png")


def read_label(lab_dir, stem):
    p = os.path.join(lab_dir, stem + ".txt")
    return "".join(open(p).read().split()) if os.path.exists(p) else None


def candidates(w, h, target, tol, step):
    """Return [(name, box)] from tightest to loosest. Tall -> trim height; wide -> trim width."""
    asp = w / h
    lo, hi = target * (1 - tol), target * (1 + tol)
    out = []
    if lo <= asp <= hi:
        return out
    if asp < lo:                                    # trim height; tightest = band edge
        min_h = math.floor(w / lo + 1e-9)
        n = max(1, round((h - min_h) / (step * h)))
        for i in range(n + 1):
            H = round(min_h + (h - min_h) * i / n)
            if H >= h:
                continue
            for pos, y0 in (("top", 0), ("mid", (h - H) // 2), ("bot", h - H)):
                out.append((f"h{H}-{pos}", (0, y0, w, y0 + H)))
    else:                                           # trim width
        min_w = math.floor(h * hi + 1e-9)
        n = max(1, round((w - min_w) / (step * w)))
        for i in range(n + 1):
            W = round(min_w + (w - min_w) * i / n)
            if W >= w:
                continue
            for pos, x0 in (("mid", (w - W) // 2), ("left", 0), ("right", w - W)):
                out.append((f"w{W}-{pos}", (x0, 0, x0 + W, h)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-aspect", type=float, required=True)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--step", type=float, default=0.05, help="window size step as a fraction of the side")
    ap.add_argument("--min-keep", type=float, default=0.40, help="never keep less than this fraction of a side")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N images (test run)")
    ap.add_argument("--copy-labels", action="store_true")
    ap.add_argument("--report-dir", default="crop_report")
    a = ap.parse_args()

    src = os.path.join(a.src, "images") if os.path.isdir(os.path.join(a.src, "images")) else a.src
    lab_dir = os.path.join(os.path.dirname(os.path.normpath(src)), "labels")
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty; choose a new --out")
    out_img = os.path.join(a.out, "images")
    os.makedirs(out_img, exist_ok=True)
    predictor.load()

    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXT))
    if a.limit:
        files = files[:a.limit]
    rows, acts = [], Counter()
    for k, f in enumerate(files):
        sp = os.path.join(src, f)
        stem = os.path.splitext(f)[0]
        label = read_label(lab_dir, stem)
        im = Image.open(sp).convert("RGB")
        w, h = im.size
        act, box, why = "keep", (0, 0, w, h), ""
        cands = [(n, b) for n, b in candidates(w, h, a.target_aspect, a.tol, a.step)
                 if (b[3] - b[1]) >= a.min_keep * h and (b[2] - b[0]) >= a.min_keep * w]
        if cands:
            if label is None:
                act, why = "unverified", "no label file"
            elif predictor.predict([im])[0] != label:
                act, why = "unverified", "model misreads the original"
            else:
                preds = predictor.predict([im.crop(b) for _, b in cands])     # one batch, tightest first
                act, why = "keep-no-safe-crop", "every crop changed the prediction"
                for (n, b), p in zip(cands, preds):
                    if p == label:
                        act, box, why = "crop-verified", b, n
                        break
        if act == "crop-verified":
            im.crop(box).save(os.path.join(out_img, f), quality=100, subsampling=0)
        else:
            shutil.copy2(sp, os.path.join(out_img, f))                        # byte-for-byte, untouched
        nw, nh = box[2] - box[0], box[3] - box[1]
        acts[act] += 1
        rows.append(dict(file=f, label=label, w=w, h=h, action=act, detail=why, new_w=nw, new_h=nh,
                         new_aspect=round(nw / nh, 3), removed_pct=round(100 * (1 - nw * nh / (w * h)), 1)))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    os.makedirs(a.report_dir, exist_ok=True)
    with open(os.path.join(a.report_dir, "safe_crop_report.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    if a.copy_labels and os.path.isdir(lab_dir):
        shutil.copytree(lab_dir, os.path.join(a.out, "labels"))
    n = len(rows)
    print(f"\n{n} images")
    for k, v in acts.most_common():
        print(f"  {k:<20}{v:>6}  {100 * v / n:4.1f}%")
    ok = [r for r in rows if r["action"] == "crop-verified"]
    if ok:
        print(f"  verified crops remove median {sorted(r['removed_pct'] for r in ok)[len(ok) // 2]}% of pixels")
    print(f"report: {a.report_dir}/safe_crop_report.csv")


if __name__ == "__main__":
    main()
