#!/usr/bin/env python3
"""Simple2D jersey-number generator with per-image annotations.

Each generated image is stored beside its annotation .txt file.
Annotation format (pixel bbox):
    <jersey_number> <x1> <y1> <x2> <y2>
A combined annotations.csv is also written for convenience.
"""
import argparse, csv, math, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG_COLORS={"red_dark":(145,20,25),"navy":(12,32,85),"green":(20,105,55),"red":(205,35,35),"yellow":(235,205,45),"white":(245,245,245)}
TEXT_COLORS=[(255,255,255),(245,245,245)]


def seed_everything(seed): random.seed(seed); np.random.seed(seed)

def find_default_font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
              str(Path.home()/".local/share/fonts/LiberationSans-Bold.ttf"),
              str(Path.home()/".local/share/fonts/FreeSansBold.ttf")):
        if Path(p).exists(): return p
    return None

def load_font(path,size):
    path=path or find_default_font()
    if not path: raise FileNotFoundError("No TTF/OTF font found. Use --font.")
    return ImageFont.truetype(path,size)

def choose_text_color(bg):
    lum=.2126*bg[0]+.7152*bg[1]+.0722*bg[2]
    return (25,25,25) if lum>205 else random.choice(TEXT_COLORS)

def render_number_with_mask(number,image_size=100,font_path=None,bg_color=(20,30,60),text_color=(255,255,255),scale_range=(.48,.70),gap_ratio=(0,.02)):
    n=int(number)
    if not 0<=n<=99: raise ValueError("number must be 0-99")
    s=str(n)  # no leading zeros
    font=load_font(font_path,max(8,int(image_size*random.uniform(*scale_range))))
    im=Image.new("RGB",(image_size,image_size),bg_color); mask=Image.new("L",(image_size,image_size),0)
    d=ImageDraw.Draw(im); md=ImageDraw.Draw(mask)
    if len(s)==1:
        b=d.textbbox((0,0),s,font=font); w=b[2]-b[0]; h=b[3]-b[1]
        x=max(0,(image_size-w)//2); y=max(0,(image_size-h)//2); pos=(x-b[0],y-b[1])
        d.text(pos,s,font=font,fill=text_color); md.text(pos,s,font=font,fill=255)
    else:
        b1=d.textbbox((0,0),s[0],font=font); b2=d.textbbox((0,0),s[1],font=font)
        w1,w2=b1[2]-b1[0],b2[2]-b2[0]; h=max(b1[3]-b1[1],b2[3]-b2[1])
        gap=int(random.uniform(*gap_ratio)*max(1,min(w1,w2)))
        x=max(0,(image_size-(w1+gap+w2))//2); y=max(0,(image_size-h)//2)
        p1=(x-b1[0],y-b1[1]); p2=(x+w1+gap-b2[0],y-b2[1])
        d.text(p1,s[0],font=font,fill=text_color); d.text(p2,s[1],font=font,fill=text_color)
        md.text(p1,s[0],font=font,fill=255); md.text(p2,s[1],font=font,fill=255)
    return np.asarray(im).copy(),np.asarray(mask).copy()

def _bbox(mask):
    ys,xs=np.where(mask>0)
    return None if len(xs)==0 else (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))

def counts_from_ratio(numbers,total,single=.30,double=.70):
    if abs(single+double-1)>1e-6: raise ValueError("--single-ratio + --double-ratio must equal 1")
    singles=[n for n in numbers if n<=9]; doubles=[n for n in numbers if n>=10]
    if not singles or not doubles: raise ValueError("numbers must include both single and double digits")
    ns=round(total*single); nd=total-ns; out={n:0 for n in numbers}
    for group,amount in ((singles,ns),(doubles,nd)):
        q,r=divmod(amount,len(group))
        for n in group: out[n]=q
        for n in random.sample(group,r): out[n]+=1
    return out

def write_annotation_txt(path,number,bbox,image_size=None):
    x1,y1,x2,y2=bbox
    if image_size is None:
        line=f"{number} {x1} {y1} {x2} {y2}\n"
    else:
        w,h=image_size
        xc=((x1+x2)/2)/w; yc=((y1+y2)/2)/h; bw=(x2-x1+1)/w; bh=(y2-y1+1)/h
        line=f"{number} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
    path.write_text(line)

def generate_simple2d_class(number,out_dir,count,image_size,font_path,rows):
    out_dir.mkdir(parents=True,exist_ok=True)
    colors=list(BG_COLORS.values()); per=max(1,math.ceil(count/len(colors))); samples=[]
    for bg in colors:
        for _ in range(per):
            im,mask=render_number_with_mask(number,image_size,font_path,bg,choose_text_color(bg))
            samples.append((im,mask))
    random.shuffle(samples)
    for i,(im,mask) in enumerate(samples[:count]):
        bbox=_bbox(mask)
        if bbox is None: continue
        stem=f"{number}_{i:05d}"; img_path=out_dir/f"{stem}.jpg"; txt_path=out_dir/f"{stem}.txt"
        cv2.imwrite(str(img_path),cv2.cvtColor(im,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
        write_annotation_txt(txt_path,number,bbox,(image_size,image_size))
        x1,y1,x2,y2=bbox
        rows.append([str(img_path.resolve()),number,x1,y1,x2,y2])

def write_csv(rows,path):
    with path.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["image_path","jersey_number","bbox_x1","bbox_y1","bbox_x2","bbox_y2"]); w.writerows(rows)

def parse_numbers(s):
    if s=="all": return list(range(100))
    if "-" in s and "," not in s:
        a,b=map(int,s.split("-")); return list(range(a,b+1))
    return [int(x) for x in s.split(",")]

def main():
    ap=argparse.ArgumentParser(description="Simple2D jersey-number generator with image-side annotations")
    ap.add_argument("--output",type=Path,default=Path("./synthetic_simple2d"))
    ap.add_argument("--count-per-class",type=int,default=4000)
    ap.add_argument("--total-count",type=int,default=None)
    ap.add_argument("--single-ratio",type=float,default=.30)
    ap.add_argument("--double-ratio",type=float,default=.70)
    ap.add_argument("--image-size",type=int,default=100)
    ap.add_argument("--font",default=None)
    ap.add_argument("--numbers",default="all")
    ap.add_argument("--gap-min",type=float,default=0.0,help="Minimum double-digit gap as fraction of digit width")
    ap.add_argument("--gap-max",type=float,default=0.02,help="Maximum double-digit gap as fraction of digit width")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); seed_everything(a.seed); nums=parse_numbers(a.numbers)
    total=a.total_count if a.total_count is not None else a.count_per_class*len(nums)
    counts=counts_from_ratio(nums,total,a.single_ratio,a.double_ratio) if a.total_count is not None else {n:a.count_per_class for n in nums}
    rows=[]; a.output.mkdir(parents=True,exist_ok=True)
    for n in nums:
        if counts[n]:
            print(f"[Simple2D] {n}: {counts[n]}")
            generate_simple2d_class(n,a.output/str(n),counts[n],a.image_size,a.font,rows)
    write_csv(rows,a.output/"annotations.csv")
    print(f"\nDone. Images + .txt labels: {a.output}/<number>/")
    print(f"CSV summary: {a.output/'annotations.csv'}")

if __name__=="__main__": main()
