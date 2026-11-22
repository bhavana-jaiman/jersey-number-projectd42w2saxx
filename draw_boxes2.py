#!/usr/bin/env python3
"""Draw bounding boxes from label files onto images.

Usage:
  python draw_boxes2.py --images <images_dir> --labels <labels_dir> --out <output_dir>

If images and labels are in the SAME folder, pass the same path to both.
Examples:
  python draw_boxes2.py --images 70_30_dataset_v3/validation_body_cropped/images \
                        --labels 70_30_dataset_v3/validation_body_cropped/labels --out vis_val
  python draw_boxes2.py --images 1768_fonts_training --labels 1768_fonts_training --out vis_fonts
Supports YOLO .txt (normalized or pixel, polygons, optional confidence) and Pascal VOC .xml.
Originals are never modified."""
import os, argparse, xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
img_root, lbl_root, out_root = (os.path.abspath(p) for p in (a.images, a.labels, a.out))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SKIP = {"classes.txt", "labels.txt", "obj.names", "readme.txt"}
COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "deepskyblue"]

def walk(root):
    for d, dirs, files in os.walk(root):
        if os.path.abspath(d).startswith(out_root):
            dirs[:] = []; continue
        for f in files: yield d, f

imgs = [os.path.join(d, f) for d, f in walk(img_root) if os.path.splitext(f)[1].lower() in IMG_EXT]
by_rel, by_stem = {}, defaultdict(list)
for d, f in walk(lbl_root):
    s, e = os.path.splitext(f)
    if e.lower() in (".txt", ".xml") and f.lower() not in SKIP:
        p = os.path.join(d, f)
        by_rel[os.path.join(os.path.relpath(d, lbl_root), s)] = p
        by_stem[s].append(p)

def find_label(ip):
    rel = os.path.splitext(os.path.relpath(ip, img_root))[0]
    if rel in by_rel: return by_rel[rel]
    return (by_stem.get(os.path.basename(rel)) or [None])[0]

def isnum(x):
    try: float(x); return True
    except ValueError: return False

def parse_txt(path, W, H):
    shapes = []
    for line in open(path, errors="ignore"):
        t = line.replace(",", " ").split()
        if not t: continue
        cls = t[0]; v = [float(x) for x in t[1:] if isnum(x)]
        if isnum(t[0]) and len(t) in (4,) :                 # no class column
            cls, v = "?", [float(x) for x in t]
        if len(v) in (4, 5):
            v = v[:4]
            if max(v) <= 1.0:                               # normalized xc yc w h
                xc, yc, w, h = v
                box = ((xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H)
            elif v[2] > v[0] and v[3] > v[1]: box = tuple(v)  # pixel x1 y1 x2 y2
            else:
                xc, yc, w, h = v; box = (xc-w/2, yc-h/2, xc+w/2, yc+h/2)
            shapes.append((cls, "box", box))
        elif len(v) >= 6 and len(v) % 2 == 0:               # polygon
            k = (W, H) if max(v) <= 1.0 else (1, 1)
            shapes.append((cls, "poly", [(v[i]*k[0], v[i+1]*k[1]) for i in range(0, len(v), 2)]))
    return shapes

def parse_xml(path):
    out = []
    for o in ET.parse(path).getroot().iter("object"):
        b = o.find("bndbox")
        if b is None: continue
        g = lambda k: float(b.find(k).text)
        out.append((o.findtext("name", "?"), "box", (g("xmin"), g("ymin"), g("xmax"), g("ymax"))))
    return out

done = nolabel = 0
for ip in sorted(imgs):
    lp = find_label(ip)
    if lp is None: nolabel += 1; continue
    im = Image.open(ip).convert("RGB"); W, H = im.size
    dr = ImageDraw.Draw(im)
    try: shapes = parse_xml(lp) if lp.lower().endswith(".xml") else parse_txt(lp, W, H)
    except Exception as e: print("skip", lp, e); continue
    lw = max(2, round(max(W, H) / 400))
    for cls, kind, g in shapes:
        col = COLORS[sum(map(ord, cls)) % len(COLORS)]
        if kind == "box": dr.rectangle(g, outline=col, width=lw); pos = (g[0], max(0, g[1]-12))
        else: dr.polygon(g, outline=col); pos = g[0]
        dr.text(pos, cls, fill=col)
    dst = os.path.join(out_root, os.path.relpath(ip, img_root))
    os.makedirs(os.path.dirname(dst), exist_ok=True); im.save(dst); done += 1

print(f"images found: {len(imgs)} | drawn: {done} | no label: {nolabel}\noutput: {out_root}")
