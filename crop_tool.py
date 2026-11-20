#!/usr/bin/env python3
"""
crop_tool.py  --  inspect test-set body crops and build an aspect-ratio-safe validation set.

Subcommands
  schema   <json>                      print the JSON structure (use this first)
  inspect  <json> [--ref-dir DIR]      crop size / aspect / body-type stats (+ compare with validation_body)
  build    <json> --out DIR            re-crop from the source images, never stretching

Supported JSON layouts (auto-detected):
  * COCO          {"images":[...], "annotations":[{"image_id","bbox":[x,y,w,h],...}]}
  * list of dicts [{"file_name"/"image"/"path": ..., "bbox": [...], "label"/"number": ...}]
  * dict keyed by file name {"a.jpg": {"bbox": [...]} or [{"bbox": ...}, ...]}
Override with --file-key / --bbox-key / --label-key / --bbox-format if the guess is wrong.
"""
import argparse, csv, json, os, sys
from collections import Counter

FILE_KEYS  = ["file_name", "filename", "image", "image_path", "img_path", "path", "file", "img"]
BBOX_KEYS  = ["bbox", "box", "body_bbox", "person_bbox", "crop", "xywh", "xyxy"]
LABEL_KEYS = ["label", "number", "jersey_number", "jersey", "gt", "text"]
W_KEYS, H_KEYS = ["width", "img_w", "image_width", "w"], ["height", "img_h", "image_height", "h"]

# ---------------------------------------------------------------- loading
def first(d, keys):
    for k in keys:
        if k in d: return k
    return None

def to_xyxy(b, fmt):
    x0, y0, a, b2 = [float(v) for v in b[:4]]
    return (x0, y0, x0 + a, y0 + b2) if fmt == "xywh" else (x0, y0, a, b2)

def load_records(path, a):
    data = json.load(open(path))
    recs = []
    if isinstance(data, dict) and "images" in data and "annotations" in data:      # COCO
        imgs = {i["id"]: i for i in data["images"]}
        for an in data["annotations"]:
            im = imgs.get(an["image_id"], {})
            bk = a.bbox_key or first(an, BBOX_KEYS)
            if bk is None: continue
            lk = a.label_key or first(an, LABEL_KEYS)
            recs.append(dict(file=im.get(a.file_key or "file_name", ""),
                             box=to_xyxy(an[bk], a.bbox_format or "xywh"),
                             iw=im.get("width"), ih=im.get("height"),
                             label=an.get(lk) if lk else None))
        return recs
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for k, v in data.items():
            for e in (v if isinstance(v, list) else [v]):
                if isinstance(e, dict): items.append({**e, "_key": k})
    for e in items:
        if not isinstance(e, dict): continue
        fk = a.file_key or first(e, FILE_KEYS)
        bk = a.bbox_key or first(e, BBOX_KEYS)
        if bk is None: continue
        lk = a.label_key or first(e, LABEL_KEYS)
        wk, hk = first(e, W_KEYS), first(e, H_KEYS)
        recs.append(dict(file=e.get(fk, e.get("_key", "")) if fk else e.get("_key", ""),
                         box=to_xyxy(e[bk], a.bbox_format or "xywh"),
                         iw=e.get(wk) if wk else None, ih=e.get(hk) if hk else None,
                         label=e.get(lk) if lk else None))
    return recs

def attach_sizes(recs, img_dir):
    from PIL import Image
    for r in recs:
        if (r["iw"] is None or r["ih"] is None) and img_dir:
            p = os.path.join(img_dir, os.path.basename(str(r["file"])))
            if os.path.exists(p):
                with Image.open(p) as im: r["iw"], r["ih"] = im.size
    return recs

# ---------------------------------------------------------------- body type
def body_type(w, h, iw, ih, small_h, ):
    asp = w / h if h else 0
    if h < small_h:            size = "small"
    else:                      size = "ok"
    if asp < 0.50:             shape = "tall/thin(full body)"
    elif asp < 0.75:           shape = "portrait(full/3-4 body)"
    elif asp < 1.10:           shape = "squarish(torso)"
    else:                      shape = "wide(torso/occluded)"
    return f"{shape}|{size}"

def pct(vals, ps=(0, 5, 25, 50, 75, 95, 100)):
    s = sorted(vals); n = len(s)
    return {p: s[min(n - 1, int(round(p / 100 * (n - 1))))] for p in ps}

def show(name, vals, fmt="{:.2f}"):
    if not vals: return
    p = pct(vals)
    print(f"  {name:<10}" + "  ".join(f"p{k}={fmt.format(v)}" for k, v in p.items()))

# ---------------------------------------------------------------- commands
def cmd_schema(a):
    data = json.load(open(a.json))
    def desc(o, d=0, maxd=3):
        pad = "  " * d
        if isinstance(o, dict):
            print(f"{pad}dict({len(o)}) keys: {list(o)[:12]}")
            if d < maxd and o:
                k = next(iter(o)); print(f"{pad}  first value under {k!r}:"); desc(o[k], d + 2, maxd)
        elif isinstance(o, list):
            print(f"{pad}list({len(o)})")
            if d < maxd and o: desc(o[0], d + 1, maxd)
        else:
            print(f"{pad}{type(o).__name__}: {str(o)[:80]}")
    desc(data)
    recs = load_records(a.json, a)
    print(f"\nparsed {len(recs)} crop records; first 3:")
    for r in recs[:3]: print("  ", r)

def folder_sizes(d):
    from PIL import Image
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            with Image.open(os.path.join(d, f)) as im: out.append(im.size)
    return out

def cmd_inspect(a):
    recs = attach_sizes(load_records(a.json, a), a.img_dir)
    rows = []
    for r in recs:
        x0, y0, x1, y1 = r["box"]; w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: continue
        rows.append(dict(file=r["file"], label=r["label"], x0=x0, y0=y0, x1=x1, y1=y1, w=w, h=h,
                         aspect=w / h, h_rel=h / r["ih"] if r["ih"] else "",
                         btype=body_type(w, h, r["iw"], r["ih"], a.small_h)))
    print(f"\n=== TEST DATA: {len(rows)} crops ===")
    show("width", [r["w"] for r in rows], "{:.0f}"); show("height", [r["h"] for r in rows], "{:.0f}")
    show("w/h", [r["aspect"] for r in rows])
    print("\n  body types:")
    for k, v in Counter(r["btype"] for r in rows).most_common():
        print(f"    {k:<38}{v:>6}  {100*v/len(rows):5.1f}%")
    print("\n  aspect histogram (w/h):")
    bins = [0, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.5, 99]
    c = Counter(next(i for i in range(len(bins) - 1) if bins[i] <= r["aspect"] < bins[i + 1]) for r in rows)
    for i in range(len(bins) - 1):
        print(f"    {bins[i]:>4}-{bins[i+1]:<4} {'#' * int(60 * c[i] / len(rows)):<60} {c[i]}")
    if a.ref_dir:
        ref = folder_sizes(a.ref_dir)
        print(f"\n=== REFERENCE ({a.ref_dir}): {len(ref)} images ===")
        show("width", [w for w, h in ref], "{:.0f}"); show("height", [h for w, h in ref], "{:.0f}")
        show("w/h", [w / h for w, h in ref])
        print(f"\n  -> suggested --target-aspect {sorted(w/h for w,h in ref)[len(ref)//2]:.3f} (reference median)")
    os.makedirs(a.report_dir, exist_ok=True)
    p = os.path.join(a.report_dir, "test_crop_stats.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\nper-crop CSV: {p}")

def cmd_build(a):
    from PIL import Image
    recs = attach_sizes(load_records(a.json, a), a.img_dir)
    out_img = os.path.join(a.out, "images"); os.makedirs(out_img, exist_ok=True)
    seen, done, skipped = Counter(), 0, 0
    for r in recs:
        src = os.path.join(a.img_dir, os.path.basename(str(r["file"])))
        if not os.path.exists(src): skipped += 1; continue
        x0, y0, x1, y1 = r["box"]; w, h = x1 - x0, y1 - y0
        if w < a.min_w or h < a.min_h: skipped += 1; continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = w * (1 + a.pad), h * (1 + a.pad)                   # optional context margin
        if a.mode == "target":                                    # grow (never squash) to target aspect
            t = a.target_aspect
            if w / h > t: h = w / t
            else:         w = h * t
        bx0, by0, bx1, by1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        with Image.open(src) as im:
            im = im.convert("RGB"); W, H = im.size
            if bx1 - bx0 <= W and by1 - by0 <= H:                 # shift box inside image, keep size
                dx = max(0, -bx0) - max(0, bx1 - W); dy = max(0, -by0) - max(0, by1 - H)
                bx0, bx1, by0, by1 = bx0 + dx, bx1 + dx, by0 + dy, by1 + dy
            ix0, iy0, ix1, iy1 = int(round(bx0)), int(round(by0)), int(round(bx1)), int(round(by1))
            canvas = Image.new("RGB", (ix1 - ix0, iy1 - iy0), (a.fill,) * 3)   # pad only if still outside
            canvas.paste(im, (-ix0, -iy0))
        if a.out_h:                                               # resize by height, aspect preserved
            canvas = canvas.resize((max(1, round(canvas.width * a.out_h / canvas.height)), a.out_h), Image.LANCZOS)
        stem = os.path.splitext(os.path.basename(str(r["file"])))[0]
        lab = "" if r["label"] is None else f"_{r['label']}"
        seen[stem + lab] += 1
        name = f"{stem}{lab}" + (f"__{seen[stem+lab]}" if seen[stem + lab] > 1 else "") + ".jpg"
        canvas.save(os.path.join(out_img, name), quality=95); done += 1
    print(f"saved {done} crops to {out_img}  (skipped {skipped})")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(s):
        s.add_argument("json"); s.add_argument("--file-key"); s.add_argument("--bbox-key")
        s.add_argument("--label-key"); s.add_argument("--bbox-format", choices=["xywh", "xyxy"])
        s.add_argument("--img-dir", help="folder with the full source images")
    s = sub.add_parser("schema"); common(s); s.set_defaults(f=cmd_schema)
    s = sub.add_parser("inspect"); common(s)
    s.add_argument("--ref-dir", help="e.g. .../validation_body/images"); s.add_argument("--small-h", type=int, default=64)
    s.add_argument("--report-dir", default="crop_report"); s.set_defaults(f=cmd_inspect)
    s = sub.add_parser("build"); common(s)
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["keep", "target"], default="target",
                   help="keep: each crop keeps its own aspect; target: grow every box to --target-aspect")
    s.add_argument("--target-aspect", type=float, default=0.55, help="w/h; use the reference median from `inspect`")
    s.add_argument("--pad", type=float, default=0.0, help="extra context margin, fraction of box size")
    s.add_argument("--min-w", type=int, default=0); s.add_argument("--min-h", type=int, default=0)
    s.add_argument("--out-h", type=int, default=0, help="optional resize by height (aspect preserved)")
    s.add_argument("--fill", type=int, default=0, help="pad gray level when box leaves the image")
    s.set_defaults(f=cmd_build)
    a = p.parse_args(); a.f(a)

if __name__ == "__main__":
    main()
