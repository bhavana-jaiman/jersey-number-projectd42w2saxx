#!/usr/bin/env python3
"""
safe_aspect_crop.py -- crop existing images toward a target aspect ratio WITHOUT cutting the jersey digits.

How it works
  1. A digit detector finds the number in each image (EasyOCR by default). The label (labels/<stem>.txt, e.g. "7 7")
     is used to keep only detections that belong to the number, and to require that ALL digits were found.
  2. The usual aspect-ratio crop is planned (trim bottom if too narrow, trim sides if too wide)...
  3. ...then limited so the protected region (digit boxes + safety margin) always stays fully inside the crop.
  4. If the digits could not be found/verified, the image is NOT cropped (copied unchanged) -- or use --fallback.

Never pads, mirrors, blurs or resizes. Same file names. Labels are not modified. Refuses a non-empty --out.

  python safe_aspect_crop.py SRC --out OUT --target-aspect 0.786 --tol 0.15 --aim edge --dry-run
  python safe_aspect_crop.py SRC --out OUT --target-aspect 0.786 --tol 0.15 --aim edge --copy-labels --preview 24

Detection results are cached (OUT/../detections_cache.json by default) so re-runs with other settings are fast.
"""
import argparse, csv, json, math, os, random, shutil
from collections import Counter
from PIL import Image

EXT = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------ detectors
class EasyOCRDetector:
    def __init__(self, gpu, upscale):
        import easyocr
        self.r = easyocr.Reader(["en"], gpu=gpu, verbose=False); self.up = upscale
    def __call__(self, im):
        import numpy as np
        w, h = im.size; big = im.resize((w * self.up, h * self.up), Image.LANCZOS)
        out = []
        for pts, txt, conf in self.r.readtext(np.asarray(big), allowlist="0123456789", detail=1, paragraph=False,
                                              text_threshold=0.4, low_text=0.3, mag_ratio=1.0):
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            out.append(dict(text="".join(c for c in txt if c.isdigit()), conf=float(conf),
                            box=[min(xs) / self.up, min(ys) / self.up, max(xs) / self.up, max(ys) / self.up]))
        return out


class TesseractDetector:                       # lighter, for quick tests
    def __init__(self, upscale):
        import pytesseract; self.t = pytesseract; self.up = upscale
    def __call__(self, im):
        w, h = im.size; big = im.resize((w * self.up, h * self.up), Image.LANCZOS)
        d = self.t.image_to_data(big, config="--psm 11 -c tessedit_char_whitelist=0123456789", output_type=self.t.Output.DICT)
        out = []
        for i, txt in enumerate(d["text"]):
            txt = "".join(c for c in txt if c.isdigit())
            if txt and float(d["conf"][i]) > 0:
                x, y, bw, bh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                out.append(dict(text=txt, conf=float(d["conf"][i]) / 100, box=[x / self.up, y / self.up, (x + bw) / self.up, (y + bh) / self.up]))
        return out


def read_label(labels_dir, stem):
    if not labels_dir: return ""
    p = os.path.join(labels_dir, stem + ".txt")
    return "".join(c for c in open(p).read() if c.isdigit()) if os.path.exists(p) else ""


def protected_region(dets, label, min_conf):
    """Return (box or None, status). Needs ALL label digits found (when a label exists)."""
    ok = [d for d in dets if d["conf"] >= min_conf and d["text"] and (not label or d["text"] in label)]
    if not ok: return None, "not-found"
    if label and sum(len(d["text"]) for d in ok) < len(label): return None, "partial"
    b = [d["box"] for d in ok]
    return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)], "found"


# ------------------------------------------------------------------ planning
def plan(w, h, target, tol, aim, keep_h, keep_w, prot, margin):
    asp = w / h; lo, hi = target * (1 - tol), target * (1 + tol)
    if lo <= asp <= hi: return "keep", (0, 0, w, h)
    goal = target if aim == "target" else (lo if asp < lo else hi)
    if prot:                                              # grow the protected box by a safety margin (fraction of its size)
        px0, py0, px1, py1 = prot; pw, ph = px1 - px0, py1 - py0
        px0 -= margin * ph; px1 += margin * ph; py0 -= margin * ph; py1 += margin * ph
        px0, py0, px1, py1 = max(0, px0), max(0, py0), min(w, px1), min(h, py1)
    else:
        px0 = py0 = 0; px1, py1 = w, h
    if asp < lo:                                          # too narrow -> trim bottom
        want = math.floor(w / goal + 1e-9)
        floor_ = max(math.ceil(keep_h * h - 1e-9), math.ceil(py1 - 1e-9) if prot else 0)
        new_h = min(h, max(want, floor_))
        lim = "-digit-limited" if (prot and new_h == floor_ and floor_ > want) else ("-limited" if want < floor_ else "")
        return "crop-bottom" + lim, (0, 0, w, new_h)
    want = math.floor(h * goal + 1e-9)                    # too wide -> trim sides (window may shift to hold the digits)
    floor_ = max(math.ceil(keep_w * w - 1e-9), math.ceil(px1 - px0 - 1e-9) if prot else 0)
    new_w = min(w, max(want, floor_))
    x0 = (w - new_w) // 2
    if prot:
        x0 = min(x0, math.floor(px0)); x0 = max(x0, math.ceil(px1) - new_w); x0 = max(0, min(x0, w - new_w))
    lim = "-digit-limited" if (prot and new_w == floor_ and floor_ > want) else ("-limited" if want < floor_ else "")
    return "crop-sides" + lim, (x0, 0, x0 + new_w, h)


def contact_sheet(pairs, out, H=220, cols=3, prots=None):
    from PIL import ImageDraw
    tiles = []
    for (b, n), pr in zip(pairs, prots or [None] * len(pairs)):
        if pr:
            b = b.copy(); ImageDraw.Draw(b).rectangle(pr, outline=(255, 0, 0), width=max(1, b.width // 60))
        b = b.resize((max(1, round(b.width * H / b.height)), H)); n = n.resize((max(1, round(n.width * H / n.height)), H))
        t = Image.new("RGB", (b.width + n.width + 24, H), (255, 255, 255)); t.paste(b, (0, 0)); t.paste(n, (b.width + 24, 0)); tiles.append(t)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    W = max(sum(t.width + 30 for t in r) for r in rows); sh = Image.new("RGB", (W, len(rows) * (H + 20)), (200, 200, 200))
    for ri, r in enumerate(rows):
        x = 0
        for t in r: sh.paste(t, (x, ri * (H + 20))); x += t.width + 30
    sh.save(out, quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("--out", required=True)
    p.add_argument("--labels", help="labels folder (default: labels/ next to images/)")
    p.add_argument("--target-aspect", type=float, required=True); p.add_argument("--tol", type=float, default=0.10)
    p.add_argument("--aim", choices=["target", "edge"], default="edge")
    p.add_argument("--min-keep-h", type=float, default=0.40, help="hard floor; the digit protection is the real limit")
    p.add_argument("--min-keep-w", type=float, default=0.70)
    p.add_argument("--margin", type=float, default=0.5, help="safety margin around the digits, as a fraction of digit height")
    p.add_argument("--min-conf", type=float, default=0.30)
    p.add_argument("--fallback", choices=["keep", "conservative"], default="keep",
                   help="when digits are not found: keep = copy unchanged; conservative = crop but never below 85%% height / 90%% width")
    p.add_argument("--detector", choices=["easyocr", "tesseract"], default="easyocr")
    p.add_argument("--gpu", action="store_true"); p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--cache"); p.add_argument("--copy-labels", action="store_true")
    p.add_argument("--preview", type=int, default=0); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-dir", default="crop_report")
    a = p.parse_args()

    src = os.path.join(a.src, "images") if os.path.isdir(os.path.join(a.src, "images")) else a.src
    labels = a.labels or (os.path.join(os.path.dirname(os.path.normpath(src)), "labels")
                          if os.path.isdir(os.path.join(os.path.dirname(os.path.normpath(src)), "labels")) else None)
    out_img = os.path.join(a.out, "images")
    if not a.dry_run:
        if os.path.exists(a.out) and os.listdir(a.out): raise SystemExit(f"refusing to write into non-empty {a.out}")
        os.makedirs(out_img, exist_ok=True)
    cache_p = a.cache or os.path.join(os.path.dirname(os.path.normpath(a.out)), "detections_cache.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    det = None
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXT))
    print(f"labels: {labels or 'none (digit text cannot be verified)'}   cache: {cache_p}")

    rows, acts, prots = [], Counter(), {}
    for i, f in enumerate(files):
        sp = os.path.join(src, f); stem = os.path.splitext(f)[0]
        with Image.open(sp) as im:
            im.load(); im = im.convert("RGB"); w, h = im.size
            if f not in cache:
                if det is None: det = EasyOCRDetector(a.gpu, a.upscale) if a.detector == "easyocr" else TesseractDetector(a.upscale)
                cache[f] = det(im)
                if i % 500 == 0: json.dump(cache, open(cache_p, "w"))
            label = read_label(labels, stem)
            prot, status = protected_region(cache[f], label, a.min_conf)
            if prot is None:
                if a.fallback == "keep": act, box = "keep-no-digits", (0, 0, w, h)
                else:
                    act, box = plan(w, h, a.target_aspect, a.tol, a.aim, 0.85, 0.90, None, a.margin); act += "-nodigits"
            else:
                act, box = plan(w, h, a.target_aspect, a.tol, a.aim, a.min_keep_h, a.min_keep_w, prot, a.margin)
            nw, nh = box[2] - box[0], box[3] - box[1]
            if prot:                                       # hard safety assertion: digits fully inside the crop
                assert box[0] <= prot[0] + 1 and box[1] <= prot[1] + 1 and box[2] >= prot[2] - 1 and box[3] >= prot[3] - 1, (f, box, prot)
            if not a.dry_run:
                dp = os.path.join(out_img, f)
                if (nw, nh) == (w, h): shutil.copy2(sp, dp)
                else:
                    c = im.crop(box); c.save(dp, quality=100, subsampling=0) if f.lower().endswith((".jpg", ".jpeg")) else c.save(dp)
        prots[f] = prot
        acts[act] += 1; fin = nw / nh
        rows.append(dict(file=f, label=label, digits=status, w=w, h=h, aspect=round(w / h, 3), action=act, new_w=nw, new_h=nh,
                         new_aspect=round(fin, 3), in_band=abs(fin - a.target_aspect) / a.target_aspect <= a.tol + 1e-9,
                         pixels_removed_pct=round(100 * (1 - nw * nh / (w * h)), 1),
                         digit_box="" if not prot else [round(v, 1) for v in prot]))
    json.dump(cache, open(cache_p, "w"))

    os.makedirs(a.report_dir, exist_ok=True); rp = os.path.join(a.report_dir, "safe_crop_report.csv")
    with open(rp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    n = len(rows); st = Counter(r["digits"] for r in rows)
    print(f"{'[dry-run] ' if a.dry_run else ''}{n} images, target {a.target_aspect} +-{int(a.tol*100)}%")
    print(f"  digits verified: {st['found']} ({100*st['found']/n:.1f}%)   partial: {st['partial']}   not found: {st['not-found']}")
    for k, v in acts.most_common(): print(f"  {k:<30}{v:>6}  {100*v/n:4.1f}%")
    ib = sum(r["in_band"] for r in rows); print(f"  reach the band: {ib} ({100*ib/n:.1f}%)")
    rm = [r["pixels_removed_pct"] for r in rows if r["pixels_removed_pct"] > 0]
    if rm: print(f"  pixels removed in cropped images: median {sorted(rm)[len(rm)//2]}%  max {max(rm)}%")
    print(f"report: {rp}   (images where digits were NOT verified are left alone and listed there)")

    if a.copy_labels and not a.dry_run and labels: shutil.copytree(labels, os.path.join(a.out, "labels")); print("labels copied unchanged")
    if a.preview and not a.dry_run:
        cr = [r for r in rows if r["pixels_removed_pct"] > 0]; random.seed(0)
        for name, sel in {"safe_preview_most_cropped.jpg": sorted(cr, key=lambda r: -r["pixels_removed_pct"])[:a.preview],
                          "safe_preview_random.jpg": random.sample(cr, min(a.preview, len(cr)))}.items():
            if sel:
                contact_sheet([(Image.open(os.path.join(src, r["file"])).convert("RGB"), Image.open(os.path.join(out_img, r["file"])).convert("RGB")) for r in sel],
                              os.path.join(a.report_dir, name), prots=[tuple(prots[r["file"]]) if prots[r["file"]] else None for r in sel])
        print(f"previews (left=before with red digit box, right=after): {a.report_dir}/safe_preview_*.jpg")


if __name__ == "__main__":
    main()
