#!/usr/bin/env python3
"""
crop_tool.py -- inspect test-set body crops and build an aspect-ratio-safe validation set.
Reads the per-image label JSONs (tool_type "number_rect"):
  {file_name, dimensions:[W,H], data:{jersey:[{body:{rect:[x,y,w,h]}, jersey_number:[1,6],
                                               numbers:[{digits:1, rect:[x,y,w,h]}, ...]}]}}

  inspect  <labels_dir> [--img-dir D] [--ref-dir validation_body/images]
  build    <labels_dir> --img-dir D --out OUT [--mode target|keep] [--target-aspect A] [--pad P]
"""
import argparse, csv, glob, json, os
from collections import Counter

# ---------------------------------------------------------------- loading
def load_records(labels_dir):
    recs = []
    files = sorted(glob.glob(os.path.join(labels_dir, "*.json"))) if os.path.isdir(labels_dir) else [labels_dir]
    for jf in files:
        d = json.load(open(jf))
        iw, ih = (d.get("dimensions") or [None, None])[:2]
        for k, j in enumerate((d.get("data") or {}).get("jersey", [])):
            body = (j.get("body") or {}).get("rect")
            if not body or len(body) < 4: continue
            x, y, w, h = [float(v) for v in body[:4]]
            nums = []
            for n in j.get("numbers", []):
                r = n.get("rect")
                if r and len(r) >= 4:
                    nx, ny, nw, nh = [float(v) for v in r[:4]]
                    nums.append((n.get("digits"), nx, ny, nx + nw, ny + nh))
            label = "".join(str(v) for v in j.get("jersey_number", []))
            recs.append(dict(json=os.path.basename(jf), file=d.get("file_name", ""), idx=k, iw=iw, ih=ih,
                             box=(x, y, x + w, y + h), nums=nums, label=label))
    return recs

def default_img_dir(labels_dir):
    for c in ("images", "imgs", "Images"):
        p = os.path.join(os.path.dirname(os.path.normpath(labels_dir)), c)
        if os.path.isdir(p): return p
    return None

# ---------------------------------------------------------------- stats helpers
def pct(vals, ps=(0, 5, 25, 50, 75, 95, 100)):
    s = sorted(vals); n = len(s)
    return {p: s[min(n - 1, int(round(p / 100 * (n - 1))))] for p in ps}

def show(name, vals, fmt="{:.2f}"):
    if vals:
        print(f"  {name:<12}" + "  ".join(f"p{k}={fmt.format(v)}" for k, v in pct(vals).items()))

def hist(vals, bins, labels=None):
    c = Counter(next((i for i in range(len(bins) - 1) if bins[i] <= v < bins[i + 1]), len(bins) - 2) for v in vals)
    for i in range(len(bins) - 1):
        lab = labels[i] if labels else f"{bins[i]:>5}-{bins[i+1]:<5}"
        print(f"    {lab:<26}{'#' * int(50 * c[i] / max(1, len(vals))):<50} {c[i]:>5} ({100*c[i]/max(1,len(vals)):4.1f}%)")

def union(nums):
    return (min(n[1] for n in nums), min(n[2] for n in nums), max(n[3] for n in nums), max(n[4] for n in nums))

def folder_sizes(d):
    from PIL import Image
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            with Image.open(os.path.join(d, f)) as im: out.append(im.size)
    return out

# ---------------------------------------------------------------- inspect
def cmd_inspect(a):
    recs = load_records(a.labels)
    print(f"{len(recs)} jersey/body entries from {len(set(r['json'] for r in recs))} label files")
    rows = []
    for r in recs:
        x0, y0, x1, y1 = r["box"]; w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: continue
        row = dict(json=r["json"], file=r["file"], label=r["label"], ndigits=len(r["label"]),
                   x=x0, y=y0, w=w, h=h, aspect=w / h,
                   h_rel_img=h / r["ih"] if r["ih"] else "", num_h_rel_body="", body_over_num_h="", num_cy_rel_body="")
        if r["nums"]:
            ux0, uy0, ux1, uy1 = union(r["nums"]); nh = uy1 - uy0
            if nh > 0:
                row.update(num_h_rel_body=nh / h, body_over_num_h=h / nh, num_cy_rel_body=((uy0 + uy1) / 2 - y0) / h)
        rows.append(row)
    print(f"\n=== TEST DATA body crops: {len(rows)} ===")
    show("body w px", [r["w"] for r in rows], "{:.0f}"); show("body h px", [r["h"] for r in rows], "{:.0f}")
    show("aspect w/h", [r["aspect"] for r in rows])
    print(f"  -> TEST-DATA median aspect: {sorted(r['aspect'] for r in rows)[len(rows)//2]:.3f}")
    show("h / image h", [r["h_rel_img"] for r in rows if r["h_rel_img"] != ""])
    print("\n  aspect (w/h) histogram:")
    hist([r["aspect"] for r in rows], [0, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.5, 99])
    print("\n  digits per jersey:", dict(sorted(Counter(r["ndigits"] for r in rows).items())))
    bn = [r["body_over_num_h"] for r in rows if r["body_over_num_h"] != ""]
    print(f"\n  crop style = body height / number height  ({len(bn)} entries with number boxes):")
    show("body/num h", bn)
    hist(bn, [0, 3, 6, 10, 16, 1e9],
         ["close-up (<3x)", "torso (3-6x)", "upper/mid body (6-10x)", "most of body (10-16x)", "full/far body (>16x)"])
    show("num centre y", [r["num_cy_rel_body"] for r in rows if r["num_cy_rel_body"] != ""])
    print("    (num centre y = vertical position of the number inside the crop, 0=top 1=bottom)")
    if a.ref_dir:
        ref = folder_sizes(a.ref_dir)
        print(f"\n=== REFERENCE {a.ref_dir}: {len(ref)} images ===")
        show("width px", [w for w, h in ref], "{:.0f}"); show("height px", [h for w, h in ref], "{:.0f}")
        show("aspect w/h", [w / h for w, h in ref])
        print("  aspect histogram:"); hist([w / h for w, h in ref], [0, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.5, 99])
        print(f"\n  -> suggested --target-aspect {sorted(w/h for w,h in ref)[len(ref)//2]:.3f} (reference median)")
    os.makedirs(a.report_dir, exist_ok=True)
    p = os.path.join(a.report_dir, "test_crop_stats.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\nper-crop CSV: {p}")

# ---------------------------------------------------------------- build
def cmd_build(a):
    from PIL import Image
    recs = load_records(a.labels)
    img_dir = a.img_dir or default_img_dir(a.labels)
    if not img_dir: raise SystemExit("pass --img-dir (folder with the full images)")
    out_img = os.path.join(a.out, "images"); os.makedirs(out_img, exist_ok=True)
    seen, done, skipped, manifest = Counter(), 0, Counter(), []
    for r in recs:
        src = os.path.join(img_dir, os.path.basename(r["file"]))
        if not os.path.exists(src): skipped["missing image"] += 1; continue
        x0, y0, x1, y1 = r["box"]; w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: skipped["bad box"] += 1; continue
        if h < a.min_h or w < a.min_w: skipped["too small"] += 1; continue
        if a.require_label and not r["label"]: skipped["no label"] += 1; continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = w * (1 + a.pad), h * (1 + a.pad)
        if a.mode == "target":                       # only ever GROW a side; never squash
            t = a.target_aspect
            if w / h > t: h = w / t
            else:         w = h * t
        bx0, by0, bx1, by1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        with Image.open(src) as im:
            im = im.convert("RGB"); W, H = im.size
            if bx1 - bx0 <= W:                       # shift inside the image, size unchanged
                dx = max(0, -bx0) - max(0, bx1 - W); bx0 += dx; bx1 += dx
            if by1 - by0 <= H:
                dy = max(0, -by0) - max(0, by1 - H); by0 += dy; by1 += dy
            ix0, iy0, ix1, iy1 = round(bx0), round(by0), round(bx1), round(by1)
            canvas = Image.new("RGB", (ix1 - ix0, iy1 - iy0), (a.fill,) * 3)   # pad only if box > image
            canvas.paste(im, (-ix0, -iy0))
        s = 1.0
        if a.out_h:
            s = a.out_h / canvas.height
            canvas = canvas.resize((max(1, round(canvas.width * s)), a.out_h), Image.LANCZOS)
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        base = f"{stem}_{r['label']}" if r["label"] else stem
        seen[base] += 1
        name = base + (f"__{seen[base]}" if seen[base] > 1 else "") + ".jpg"
        canvas.save(os.path.join(out_img, name), quality=95); done += 1
        manifest.append(dict(file=name, label=r["label"], src=r["file"], crop_w=canvas.width, crop_h=canvas.height,
                             numbers=[dict(digit=n[0], rect=[round((n[1]-ix0)*s, 1), round((n[2]-iy0)*s, 1),
                                                             round((n[3]-n[1])*s, 1), round((n[4]-n[2])*s, 1)])
                                      for n in r["nums"]]))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"saved {done} crops -> {out_img}   skipped: {dict(skipped) or 0}")
    print(f"manifest (label + digit boxes in new crop coordinates): {os.path.join(a.out, 'manifest.json')}")

# ---------------------------------------------------------------- restyle (existing crops -> new aspect, same names)
def _pad(im, l, t, r, b, how, fill):
    import numpy as np
    from PIL import Image, ImageFilter
    arr = np.asarray(im)
    if how == "black":
        mode = dict(mode="constant", constant_values=fill)
    else:
        mode = dict(mode=how)
    try:
        out = np.pad(arr, ((t, b), (l, r), (0, 0)), **mode)
    except ValueError:                                   # reflect needs pad < size -> fall back to edge
        out = np.pad(arr, ((t, b), (l, r), (0, 0)), mode="edge")
    return Image.fromarray(out)

def cmd_restyle(a):
    import shutil
    from PIL import Image
    src = a.src if not os.path.isdir(os.path.join(a.src, "images")) else os.path.join(a.src, "images")
    out_img = os.path.join(a.out, "images"); os.makedirs(out_img, exist_ok=True)
    acts, rows = Counter(), []
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
        im = Image.open(os.path.join(src, f)).convert("RGB"); w, h = im.size; ow, oh = w, h; asp = w / h; t = a.target_aspect
        act = "keep"
        if abs(asp - t) / t > a.tol:
            if asp < t:                                  # too narrow/tall -> need wider
                new_h = w / t                            # trim from the BOTTOM only (number sits in the upper body)
                if a.how in ("crop", "auto") and new_h >= a.max_trim * h:
                    im = im.crop((0, 0, w, round(new_h))); act = "crop-bottom"
                else:
                    if a.how == "crop": new_h = max(new_h, h)
                    if a.how in ("crop", "auto") and new_h < h:        # trim as far as allowed, pad the rest
                        keep = round(a.max_trim * h); im = im.crop((0, 0, w, keep)); w, h = im.size
                    need_w = round(h * t); extra = max(0, need_w - w)
                    im = _pad(im, extra // 2, 0, extra - extra // 2, 0, a.pad_mode, a.fill); act = "pad-sides"
            else:                                        # too wide -> need taller
                new_w = h * t                            # trim sides symmetrically (number is horizontally central)
                if a.how in ("crop", "auto") and new_w >= a.max_trim * w:
                    x0 = round((w - new_w) / 2); im = im.crop((x0, 0, x0 + round(new_w), h)); act = "crop-sides"
                else:
                    if a.how in ("crop", "auto"):
                        keep = round(a.max_trim * w); x0 = (w - keep) // 2; im = im.crop((x0, 0, x0 + keep, h)); w, h = im.size
                    need_h = round(w / t); extra = max(0, need_h - h)
                    im = _pad(im, 0, extra // 2, 0, extra - extra // 2, a.pad_mode, a.fill); act = "pad-topbottom"
        if a.out_h:
            im = im.resize((max(1, round(im.width * a.out_h / im.height)), a.out_h), Image.LANCZOS)
        acts[act] += 1; rows.append((f, ow, oh, round(asp, 3), act, im.width, im.height))
        im.save(os.path.join(out_img, f), quality=95)     # SAME file name -> existing labels still match
    if a.copy_labels:
        lab = os.path.join(os.path.dirname(os.path.normpath(src)), "labels")
        if os.path.isdir(lab): shutil.copytree(lab, os.path.join(a.out, "labels"), dirs_exist_ok=True)
    os.makedirs(a.report_dir, exist_ok=True)
    with open(os.path.join(a.report_dir, "restyle_report.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["file", "w", "h", "aspect", "action", "new_w", "new_h"]); wr.writerows(rows)
    print(f"restyled {len(rows)} images -> {out_img}   actions: {dict(acts)}")
    print(f"report: {os.path.join(a.report_dir, 'restyle_report.csv')}")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect"); s.add_argument("labels"); s.add_argument("--img-dir")
    s.add_argument("--ref-dir"); s.add_argument("--report-dir", default="crop_report"); s.set_defaults(f=cmd_inspect)
    s = sub.add_parser("build"); s.add_argument("labels"); s.add_argument("--img-dir"); s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["keep", "target"], default="target")
    s.add_argument("--target-aspect", type=float, default=0.55)
    s.add_argument("--pad", type=float, default=0.0, help="extra context, fraction of box size (0.1 = +10%%)")
    s.add_argument("--min-w", type=int, default=0); s.add_argument("--min-h", type=int, default=0)
    s.add_argument("--out-h", type=int, default=0, help="optional resize by height (aspect preserved)")
    s.add_argument("--fill", type=int, default=0); s.add_argument("--require-label", action="store_true")
    s.set_defaults(f=cmd_build)
    s = sub.add_parser("restyle", help="re-aspect EXISTING crops (same file names, labels untouched)")
    s.add_argument("src", help="folder with images (or a dataset folder that has images/)"); s.add_argument("--out", required=True)
    s.add_argument("--target-aspect", type=float, required=True, help="w/h, from `inspect`")
    s.add_argument("--how", choices=["auto", "crop", "pad"], default="auto")
    s.add_argument("--max-trim", type=float, default=0.75, help="never keep less than this fraction of a side when cropping")
    s.add_argument("--tol", type=float, default=0.05, help="leave images within this relative aspect error unchanged")
    s.add_argument("--pad-mode", choices=["reflect", "edge", "black"], default="reflect"); s.add_argument("--fill", type=int, default=0)
    s.add_argument("--out-h", type=int, default=0); s.add_argument("--copy-labels", action="store_true", help="copy labels/ unchanged")
    s.add_argument("--report-dir", default="crop_report"); s.set_defaults(f=cmd_restyle)
    a = p.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
