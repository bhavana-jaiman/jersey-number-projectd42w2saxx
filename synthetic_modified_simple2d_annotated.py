#!/usr/bin/env python3
"""Clean Simple2D + orientation-only Complex2D jersey-number generator."""

import argparse, csv, math, random
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG_COLORS = {
    "red_dark": (145,20,25), "navy": (12,32,85), "green": (20,105,55),
    "red": (205,35,35), "yellow": (235,205,45), "white": (245,245,245)
}
TEXT_COLORS = [(255,255,255),(245,245,245)]

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)

def list_images(root):
    if not root.exists(): return []
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def find_default_font():
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"):
        if Path(p).exists(): return p
    return None

def load_font(path, size):
    path=path or find_default_font()
    if not path: raise FileNotFoundError("No TTF/OTF font found. Use --font.")
    return ImageFont.truetype(path,size)

def choose_text_color(bg):
    lum=.2126*bg[0]+.7152*bg[1]+.0722*bg[2]
    return (25,25,25) if lum>205 else random.choice(TEXT_COLORS)

# --------------------------- NUMBER / SIMPLE2D ---------------------------

def render_number(number, image_size=100, font_path=None, bg_color=(20,30,60),
                  text_color=(255,255,255), scale_range=(.48,.70),
                  gap_ratio=(0,.05)):
    n=int(number)
    if not 0<=n<=99: raise ValueError("number must be 0-99")
    s=str(n)                         # no 07/03: 7/3 only
    font=load_font(font_path,max(8,int(image_size*random.uniform(*scale_range))))
    im=Image.new("RGB",(image_size,image_size),bg_color)
    d=ImageDraw.Draw(im)
    if len(s)==1:
        b=d.textbbox((0,0),s,font=font); w=b[2]-b[0]; h=b[3]-b[1]
        x=max(0,(image_size-w)//2); y=max(0,(image_size-h)//2)
        d.text((x-b[0],y-b[1]),s,font=font,fill=text_color)
    else:
        b1=d.textbbox((0,0),s[0],font=font); b2=d.textbbox((0,0),s[1],font=font)
        w1,w2=b1[2]-b1[0],b2[2]-b2[0]
        h=max(b1[3]-b1[1],b2[3]-b2[1])
        gap=int(random.uniform(*gap_ratio)*max(1,min(w1,w2)))
        x=max(0,(image_size-(w1+gap+w2))//2); y=max(0,(image_size-h)//2)
        d.text((x-b1[0],y-b1[1]),s[0],font=font,fill=text_color)
        d.text((x+w1+gap-b2[0],y-b2[1]),s[1],font=font,fill=text_color)
    return im

def render_number_with_mask(number, image_size=100, font_path=None, bg_color=(20,30,60),
                              text_color=(255,255,255), scale_range=(.48,.70),
                              gap_ratio=(0,.05)):
    """
    Render the complete jersey number as one object and return:
      image: RGB Simple2D image
      mask:  binary foreground mask of the jersey number
    The mask is used to calculate the exact jersey-number bounding box.
    """
    n=int(number)
    if not 0<=n<=99: raise ValueError("number must be 0-99")
    s=str(n)
    font=load_font(font_path,max(8,int(image_size*random.uniform(*scale_range))))
    im=Image.new("RGB",(image_size,image_size),bg_color)
    mask=Image.new("L",(image_size,image_size),0)
    d=ImageDraw.Draw(im)
    md=ImageDraw.Draw(mask)

    if len(s)==1:
        b=d.textbbox((0,0),s,font=font)
        w=b[2]-b[0]; h=b[3]-b[1]
        x=max(0,(image_size-w)//2); y=max(0,(image_size-h)//2)
        pos=(x-b[0],y-b[1])
        d.text(pos,s,font=font,fill=text_color)
        md.text(pos,s,font=font,fill=255)
    else:
        b1=d.textbbox((0,0),s[0],font=font)
        b2=d.textbbox((0,0),s[1],font=font)
        w1,w2=b1[2]-b1[0],b2[2]-b2[0]
        h=max(b1[3]-b1[1],b2[3]-b2[1])
        gap=int(random.uniform(*gap_ratio)*max(1,min(w1,w2)))
        x=max(0,(image_size-(w1+gap+w2))//2); y=max(0,(image_size-h)//2)

        p1=(x-b1[0],y-b1[1])
        p2=(x+w1+gap-b2[0],y-b2[1])
        d.text(p1,s[0],font=font,fill=text_color)
        d.text(p2,s[1],font=font,fill=text_color)
        md.text(p1,s[0],font=font,fill=255)
        md.text(p2,s[1],font=font,fill=255)

    return np.asarray(im).copy(), np.asarray(mask).copy()


def gaussian_noise(img,sigma=(2,10)):
    n=np.random.normal(0,random.uniform(*sigma),img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+n,0,255).astype(np.uint8)

def optical_distortion(img):
    h,w=img.shape[:2]; k1=random.uniform(-.18,.18); k2=random.uniform(-.08,.08)
    cam=np.array([[w,0,w/2],[0,h,h/2],[0,0,1]],np.float32)
    return cv2.undistort(img,cam,np.array([k1,k2,0,0,0],np.float32))

def grid_distortion(img,strength=.08):
    h,w=img.shape[:2]
    yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing="ij")
    ax=random.uniform(.01,strength)*w; ay=random.uniform(.01,strength)*h
    fx=random.uniform(.8,2); fy=random.uniform(.8,2)
    mx=xx+ax*np.sin(2*np.pi*yy/max(h,1)*fy)
    my=yy+ay*np.sin(2*np.pi*xx/max(w,1)*fx)
    return cv2.remap(img,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def apply_random_paper_augmentation(img):
    level=random.choices(["light","medium","hard"],weights=[1,5,5])[0]
    out=optical_distortion(img)
    if level!="light": out=grid_distortion(out,.07 if level=="medium" else .08)
    if level=="hard":
        h,w=out.shape[:2]; a=random.uniform(-18,18); sc=random.uniform(.82,1.18)
        M=cv2.getRotationMatrix2D((w/2,h/2),a,sc)
        M[0,2]+=random.uniform(-.12,.12)*w; M[1,2]+=random.uniform(-.12,.12)*h
        out=cv2.warpAffine(out,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    return gaussian_noise(out,(2,10) if level!="hard" else (1,8))

def counts_from_ratio(numbers,total,single=.30,double=.70):
    if abs(single+double-1)>1e-6: raise ValueError("ratios must sum to 1")
    singles=[n for n in numbers if 0<=n<=9]; doubles=[n for n in numbers if 10<=n<=99]
    if not singles or not doubles: raise ValueError("numbers must include singles and doubles")
    ns=round(total*single); nd=total-ns; out={n:0 for n in numbers}
    for group,amount in ((singles,ns),(doubles,nd)):
        q,r=divmod(amount,len(group))
        for n in group: out[n]=q
        for n in random.sample(group,r): out[n]+=1
    return out

def _mask_bbox(mask):
    """Return tight bbox (x1,y1,x2,y2) around non-zero mask pixels."""
    ys,xs=np.where(mask>0)
    if len(xs)==0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def generate_simple2d_class(number,out_dir,count=4000,image_size=100,font_path=None,annotation_rows=None):
    out_dir.mkdir(parents=True,exist_ok=True)
    colors=list(BG_COLORS.values())
    per=max(1,math.ceil(count/len(colors)))
    images=[]

    # Keep the image and its foreground mask together so the bbox remains exact.
    for bg in colors:
        for _ in range(per):
            im,mask=render_number_with_mask(
                number,image_size,font_path,bg,choose_text_color(bg)
            )
            images.append((im,mask))

    random.shuffle(images)

    for i,(im,mask) in enumerate(images[:count]):
        output_file=out_dir/f"{number}_{i:05d}.jpg"
        cv2.imwrite(
            str(output_file),
            cv2.cvtColor(im,cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY,95]
        )

        bbox=_mask_bbox(mask)
        if bbox is None:
            continue
        x1,y1,x2,y2=bbox
        annotation_rows.append([
            str(output_file.resolve()),
            int(number),
            x1,y1,x2,y2,
            int(x2-x1+1),
            int(y2-y1+1),
        ])


def generate_simple2d(root,count_per_class=4000,image_size=100,font_path=None,numbers=range(100),
                      total_count=None,single_ratio=.30,double_ratio=.70):
    root.mkdir(parents=True,exist_ok=True)
    numbers=list(numbers)
    counts=counts_from_ratio(numbers,total_count,single_ratio,double_ratio)
    annotation_rows=[]

    for n in numbers:
        if counts[n]:
            print(f"[Simple2D] {n}: {counts[n]}")
            generate_simple2d_class(
                n,root/str(n),counts[n],image_size,font_path,annotation_rows
            )

    write_number_annotations(
        annotation_rows,
        root/"annotations.csv",
        stage="Simple2D"
    )

def write_number_annotations(rows,out_file,stage="Synthetic"):
    """
    Write ONLY the requested annotation information:
      image path, jersey number, bounding box.
    bbox format is absolute pixel coordinates:
      bbox_x1, bbox_y1, bbox_x2, bbox_y2
    """
    out_file.parent.mkdir(parents=True,exist_ok=True)
    with out_file.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow([
            "image_path",
            "jersey_number",
            "bbox_x1","bbox_y1","bbox_x2","bbox_y2"
        ])
        w.writerows(rows)
    print(f"{stage} annotations: {out_file} ({len(rows)} images)")



def main():
    ap = argparse.ArgumentParser(description="Simple2D jersey-number generator with bbox annotations")
    ap.add_argument("--output", type=Path, default=Path("./synthetic_simple2d"))
    ap.add_argument("--count-per-class", type=int, default=4000)
    ap.add_argument("--total-count", type=int, default=None,
                    help="Total images across all requested classes. Use with --single-ratio/--double-ratio.")
    ap.add_argument("--single-ratio", type=float, default=0.30)
    ap.add_argument("--double-ratio", type=float, default=0.70)
    ap.add_argument("--image-size", type=int, default=100)
    ap.add_argument("--font", default=None)
    ap.add_argument("--numbers", default="all",
                    help='all, a-b, or comma-separated numbers, e.g. "0-99" or "1,7,10,29"')
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    seed_everything(a.seed)
    nums = parse_numbers(a.numbers)

    generate_simple2d(
        root=a.output,
        count_per_class=a.count_per_class,
        image_size=a.image_size,
        font_path=a.font,
        numbers=nums,
        total_count=a.total_count,
        single_ratio=a.single_ratio,
        double_ratio=a.double_ratio,
    )

    print("\nDone.")
    print(f"Images:      {a.output}/<jersey-number>/*.jpg")
    print(f"Annotations: {a.output}/annotations.csv")
    print("Annotation columns: image_path, jersey_number, bbox_x1, bbox_y1, bbox_x2, bbox_y2")


def parse_numbers(s):
    if s == "all":
        return list(range(100))
    if "-" in s and "," not in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    main()
