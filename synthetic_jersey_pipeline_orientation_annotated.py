#!/usr/bin/env python3
"""Clean orientation-only Complex2D jersey-number generator with automatic annotations."""

import argparse, csv, json, math, random
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

def generate_simple2d_class(number,out_dir,count=4000,image_size=100,font_path=None):
    out_dir.mkdir(parents=True,exist_ok=True); colors=list(BG_COLORS.values())
    per=max(1,math.ceil(count/len(colors))); images=[]
    for bg in colors:
        for _ in range(per):
            # CLEAN Simple2D:
            # Keep the number itself clean. No paper distortion,
            # grid distortion, Gaussian noise, or artificial camera effects.
            im=render_number(number,image_size,font_path,bg,choose_text_color(bg))
            images.append(np.asarray(im).copy())
    random.shuffle(images)
    for i,im in enumerate(images[:count]):
        cv2.imwrite(str(out_dir/f"{number}_{i:05d}.jpg"),cv2.cvtColor(im,cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY,95])

def generate_simple2d(root,count_per_class=4000,image_size=100,font_path=None,numbers=range(100),
                      total_count=None,single_ratio=.30,double_ratio=.70):
    root.mkdir(parents=True,exist_ok=True); numbers=list(numbers)
    counts=counts_from_ratio(numbers,total_count,single_ratio,double_ratio) if total_count is not None else {n:count_per_class for n in numbers}
    for n in numbers:
        if counts[n]: print(f"[Simple2D] {n}: {counts[n]}"); generate_simple2d_class(n,root/str(n),counts[n],image_size,font_path)


# ---------------- AUTOMATIC ANNOTATION ----------------------------

def bbox_from_mask(mask, x, y):
    """Return exact final-image bbox from the final transformed number mask."""
    ys, xs = np.where(mask > 32)

    if len(xs) == 0:
        return None

    xmin = int(x + xs.min())
    ymin = int(y + ys.min())
    xmax = int(x + xs.max() + 1)
    ymax = int(y + ys.max() + 1)

    return xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin


def write_yolo_annotation(
    label_path,
    class_id,
    bbox,
    image_width,
    image_height,
):
    """Write YOLO: class x_center y_center width height."""
    xmin, ymin, xmax, ymax, _, _ = bbox

    xc = ((xmin + xmax) / 2.0) / image_width
    yc = ((ymin + ymax) / 2.0) / image_height
    w = (xmax - xmin) / image_width
    h = (ymax - ymin) / image_height

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w") as f:
        f.write(
            f"{class_id} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n"
        )


def write_full_image_mask(
    mask,
    x,
    y,
    image_size,
    output_path,
):
    """Save the final number mask in full-image coordinates."""
    full = np.zeros(
        (image_size, image_size),
        dtype=np.uint8,
    )

    h, w = mask.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_size, x + w)
    y2 = min(image_size, y + h)

    if x2 > x1 and y2 > y1:
        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        full[y1:y2, x1:x2] = mask[sy1:sy2, sx1:sx2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), full)


def write_annotation_row(
    writer,
    image_path,
    number,
    bbox,
    image_size,
):
    """Write classification + multitask digit + bbox information."""
    if number <= 9:
        digit1 = number
        digit2 = -1
        num_digits = 1
    else:
        digit1 = number // 10
        digit2 = number % 10
        num_digits = 2

    xmin, ymin, xmax, ymax, bw, bh = bbox

    writer.writerow([
        str(image_path.resolve()),
        int(number),
        str(number),
        int(digit1),
        int(digit2),
        int(num_digits),
        int(xmin),
        int(ymin),
        int(xmax),
        int(ymax),
        int(bw),
        int(bh),
        f"{((xmin + xmax) / 2.0) / image_size:.8f}",
        f"{((ymin + ymax) / 2.0) / image_size:.8f}",
        f"{bw / image_size:.8f}",
        f"{bh / image_size:.8f}",
    ])


# ---------------- COMPLEX2D: ORIENTATION ONLY -------------------

def extract_number_mask(img):
    x=img.astype(np.float32); h,w=x.shape[:2]; b=max(2,int(min(h,w)*.08))
    border=np.concatenate([x[:b].reshape(-1,3),x[-b:].reshape(-1,3),x[:,:b].reshape(-1,3),x[:,-b:].reshape(-1,3)])
    bg=np.median(border,axis=0); dist=np.linalg.norm(x-bg[None,None,:],axis=2)
    m=(dist>max(18,float(np.percentile(dist,82)))).astype(np.uint8)*255
    k=np.ones((3,3),np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
    return m

def tight_crop(img,mask,pad=2):
    ys,xs=np.where(mask>0)
    if len(xs)==0:return img,mask
    x1=max(0,int(xs.min())-pad); x2=min(img.shape[1],int(xs.max())+pad+1)
    y1=max(0,int(ys.min())-pad); y2=min(img.shape[0],int(ys.max())+pad+1)
    return img[y1:y2,x1:x2],mask[y1:y2,x1:x2]

def directional_bend(patch,mask,strength,direction):
    h,w=patch.shape[:2]
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dx=max(1,w*strength); dy=max(1,h*strength*.55)
    if direction=="left":
        dst=np.float32([[dx,dy*.35],[w-1,0],[w-1-dx*.55,h-1],[0,h-1-dy*.35]])
    elif direction=="right":
        dst=np.float32([[0,0],[w-1-dx,dy*.35],[w-1,h-1],[dx*.55,h-1-dy*.35]])
    else:
        j=.45
        dst=np.float32([[random.uniform(0,dx*j),random.uniform(0,dy*j)],
                        [w-1-random.uniform(0,dx*j),random.uniform(0,dy*j)],
                        [w-1-random.uniform(0,dx*j),h-1-random.uniform(0,dy*j)],
                        [random.uniform(0,dx*j),h-1-random.uniform(0,dy*j)]])
    H=cv2.getPerspectiveTransform(src,dst)
    p=cv2.warpPerspective(patch,H,(w,h),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    m=cv2.warpPerspective(mask,H,(w,h),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return p,np.clip(m,0,255).astype(np.uint8)

def surface_warp(patch,mask,strength):
    h,w=patch.shape[:2]
    yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing="ij")
    phase=random.uniform(0,2*np.pi)
    ax=random.uniform(.005,strength)*w; ay=random.uniform(.005,strength*.7)*h
    mx=xx+ax*np.sin(2*np.pi*yy/max(h,1)+phase); my=yy+ay*np.sin(2*np.pi*xx/max(w,1)+phase)
    p=cv2.remap(patch,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    m=cv2.remap(mask,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return p,np.clip(m,0,255).astype(np.uint8)

def warp_number_object(patch,mask,target_size=100):
    """
    ORIENTATION ONLY.

    Kept:
      - whole-number rotation
      - left/right/straight directional perspective

    Removed:
      - surface/fabric warp
      - grid distortion
      - noise
      - realistic texture effects

    The complete number and its mask always receive the same transform.
    """
    ph,pw=patch.shape[:2]

    # Small pre-scale so orientation does not change the final size wildly.
    sc=random.uniform(.98,1.02)
    nw=max(8,int(round(pw*sc)))
    nh=max(8,int(round(ph*sc)))

    patch=cv2.resize(
        patch,(nw,nh),
        interpolation=cv2.INTER_CUBIC
    )
    mask=cv2.resize(
        mask,(nw,nh),
        interpolation=cv2.INTER_NEAREST
    )

    # -------------------------------------------------------------
    # ROTATION
    # -------------------------------------------------------------
    # This is the main orientation control.
    angle=random.uniform(-15.0,15.0)

    # -------------------------------------------------------------
    # LEFT / RIGHT / STRAIGHT ORIENTATION
    # -------------------------------------------------------------
    direction=random.choices(
        ["straight","left","right"],
        weights=[.30,.35,.35],
        k=1,
    )[0]

    # Rotate the COMPLETE number object.
    M=cv2.getRotationMatrix2D(
        (nw/2.0,nh/2.0),
        angle,
        1.0,
    )

    c=abs(M[0,0])
    s=abs(M[0,1])

    rw=max(8,int(nh*s+nw*c))
    rh=max(8,int(nh*c+nw*s))

    M[0,2]+=rw/2.0-nw/2.0
    M[1,2]+=rh/2.0-nh/2.0

    patch=cv2.warpAffine(
        patch,M,(rw,rh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask=cv2.warpAffine(
        mask,M,(rw,rh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # -------------------------------------------------------------
    # LIGHT DIRECTIONAL PERSPECTIVE ONLY
    # -------------------------------------------------------------
    # Enough to simulate a jersey viewed from a left/right angle,
    # but deliberately much weaker than the realistic version.
    perspective_strength=random.uniform(.015,.045)

    patch,mask=directional_bend(
        patch,
        mask,
        perspective_strength,
        direction,
    )

    # NO surface_warp()
    # NO grid_distortion()
    # NO gaussian_noise()
    # NO blur here.

    return patch,np.clip(mask,0,255).astype(np.uint8)

def resize_number_for_jersey(patch,mask,image_size=100):
    ph,pw=patch.shape[:2]; target_h=int(image_size*random.uniform(.45,.55))
    sc=target_h/max(ph,1); nw=max(8,int(round(pw*sc))); nh=max(8,int(round(ph*sc)))
    patch=cv2.resize(patch,(nw,nh),cv2.INTER_CUBIC); mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
    if nh>image_size-8 or nw>image_size-8:
        sc=min((image_size-8)/nh,(image_size-8)/nw); nw=max(8,int(nw*sc)); nh=max(8,int(nh*sc))
        patch=cv2.resize(patch,(nw,nh),cv2.INTER_AREA); mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
    return patch,mask

def center_position(image_size,pw,ph):
    jx=random.uniform(-.06,.06)*image_size; jy=random.uniform(-.05,.06)*image_size
    x=round(image_size/2-pw/2+jx); y=round(image_size/2-ph/2+jy)
    return max(2,min(x,image_size-pw-2)),max(2,min(y,image_size-ph-2))

def blend_number(bg,patch,mask,x,y):
    ph,pw=patch.shape[:2]; x2=min(bg.shape[1],x+pw); y2=min(bg.shape[0],y+ph)
    if x2<=x or y2<=y:return bg
    p=patch[:y2-y,:x2-x]
    m=mask[:y2-y,:x2-x].astype(np.float32)/255.0

    # Clean composite: no foreground noise, no brightness randomization,
    # and no blur. Only a tiny mask feather avoids a hard rectangular edge.
    m=cv2.GaussianBlur(m,(3,3),0)

    base=bg[y:y2,x:x2].astype(np.float32)
    fg=p.astype(np.float32)

    bg[y:y2,x:x2]=np.clip(
        fg*m[...,None] + base*(1-m[...,None]),
        0,255
    ).astype(np.uint8)
    return bg

def generate_complex2d_class(
    number,
    simple_root,
    backgrounds,
    out_dir,
    count=4000,
    image_size=100,
    yolo_label_dir=None,
    mask_dir=None,
    annotation_writer=None,
):
    """
    Generate orientation-only Complex2D AND automatic annotations.

    The annotation bbox is calculated from the exact final number mask,
    after rotation, perspective, recropping, resizing and placement.
    Therefore manual annotation is not required.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    number_images = list_images(simple_root / str(number))

    if not number_images:
        raise FileNotFoundError(
            f"No Simple2D images found for class {number}: "
            f"{simple_root / str(number)}"
        )

    if not backgrounds:
        raise FileNotFoundError("No background images supplied.")

    if yolo_label_dir is not None:
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    attempts = 0
    max_attempts = max(100, count * 5)

    while made < count and attempts < max_attempts:
        attempts += 1

        bg = cv2.imread(str(random.choice(backgrounds)))
        number_img = cv2.imread(str(random.choice(number_images)))

        if bg is None or number_img is None:
            continue

        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        number_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2RGB)

        # ---------------- Background/player crop ----------------
        bh, bw = bg.shape[:2]

        if min(bh, bw) < image_size:
            scale = max(
                image_size / max(bh, 1),
                image_size / max(bw, 1),
            )
            bg = cv2.resize(
                bg,
                (
                    max(image_size, int(round(bw * scale))),
                    max(image_size, int(round(bh * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            bh, bw = bg.shape[:2]

        crop_size = min(bh, bw)
        x0 = (bw - crop_size) // 2
        y0 = (bh - crop_size) // 2

        bg = cv2.resize(
            bg[y0:y0 + crop_size, x0:x0 + crop_size],
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

        # ---------------- Extract clean number ----------------
        mask = extract_number_mask(number_img)
        patch, mask = tight_crop(number_img, mask, pad=2)

        if np.count_nonzero(mask) < 8:
            continue

        # ---------------- Orientation only ----------------
        patch, mask = warp_number_object(
            patch,
            mask,
            image_size,
        )

        # IMPORTANT:
        # Rotation/perspective introduces empty border area.
        # Remove it before calculating final number size.
        patch, mask = tight_crop(
            patch,
            mask,
            pad=2,
        )

        # ---------------- Consistent final size ----------------
        patch, mask = resize_number_for_jersey(
            patch,
            mask,
            image_size,
        )

        ph, pw = patch.shape[:2]

        if ph < 4 or pw < 4:
            continue

        # ---------------- Center placement ----------------
        x, y = center_position(
            image_size,
            pw,
            ph,
        )

        # ---------------- Exact bbox from final mask ----------------
        bbox = bbox_from_mask(
            mask,
            x,
            y,
        )

        if bbox is None:
            continue

        xmin, ymin, xmax, ymax, bw_box, bh_box = bbox

        # Clip bbox to final image.
        xmin = max(0, min(xmin, image_size))
        ymin = max(0, min(ymin, image_size))
        xmax = max(0, min(xmax, image_size))
        ymax = max(0, min(ymax, image_size))

        if xmax <= xmin or ymax <= ymin:
            continue

        bbox = (
            xmin,
            ymin,
            xmax,
            ymax,
            xmax - xmin,
            ymax - ymin,
        )

        # ---------------- Composite ----------------
        bg = blend_number(
            bg,
            patch,
            mask,
            x,
            y,
        )

        # No noise, no grid, no final blur:
        # this remains the clean orientation-only pipeline.

        # ---------------- Save image ----------------
        image_path = out_dir / f"{number}_{made:05d}.jpg"

        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(bg, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        # ---------------- Save YOLO bbox ----------------
        if yolo_label_dir is not None:
            write_yolo_annotation(
                yolo_label_dir / f"{number}_{made:05d}.txt",
                int(number),
                bbox,
                image_size,
                image_size,
            )

        # ---------------- Save segmentation mask ----------------
        if mask_dir is not None:
            write_full_image_mask(
                mask,
                x,
                y,
                image_size,
                mask_dir / f"{number}_{made:05d}.png",
            )

        # ---------------- Save CSV metadata ----------------
        if annotation_writer is not None:
            write_annotation_row(
                annotation_writer,
                image_path,
                number,
                bbox,
                image_size,
            )

        made += 1

    if made < count:
        raise RuntimeError(
            f"Generated only {made}/{count} for class {number}"
        )



def generate_complex2d(
    root,
    simple_root,
    bg_dir,
    count_per_class=4000,
    image_size=100,
    numbers=range(100),
    total_count=None,
    single_ratio=.30,
    double_ratio=.70,
    yolo_label_root=None,
    mask_root=None,
    annotation_csv=None,
):
    """Generate Complex2D images and all requested annotations."""
    backgrounds = list_images(bg_dir)

    if not backgrounds:
        raise FileNotFoundError(
            f"No background images under {bg_dir}"
        )

    root.mkdir(parents=True, exist_ok=True)

    numbers = list(numbers)

    counts = (
        counts_from_ratio(
            numbers,
            total_count,
            single_ratio,
            double_ratio,
        )
        if total_count is not None
        else {n: count_per_class for n in numbers}
    )

    csv_file = None
    writer = None

    if annotation_csv is not None:
        annotation_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        csv_file = annotation_csv.open(
            "w",
            newline="",
        )
        writer = csv.writer(csv_file)

        writer.writerow([
            "image_path",
            "whole",
            "number",
            "digit1",
            "digit2",
            "num_digits",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "bbox_width",
            "bbox_height",
            "bbox_xcenter_norm",
            "bbox_ycenter_norm",
            "bbox_width_norm",
            "bbox_height_norm",
        ])

    try:
        for number in numbers:
            count = counts[number]

            if count <= 0:
                continue

            print(
                f"[Complex2D] {number}: {count}"
            )

            generate_complex2d_class(
                number=number,
                simple_root=simple_root,
                backgrounds=backgrounds,
                out_dir=root / str(number),
                count=count,
                image_size=image_size,
                yolo_label_dir=yolo_label_root,
                mask_dir=mask_root,
                annotation_writer=writer,
            )
    finally:
        if csv_file is not None:
            csv_file.close()



def write_manifest(root,out):
    rows=[]
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.isdigit(): continue
        n=int(d.name)
        for p in sorted(list_images(d)):
            d1=n if n<=9 else n//10; d2=-1 if n<=9 else n%10; length=1 if n<=9 else 2
            rows.append([str(p.resolve()),n,str(n),d1,d2,length])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["path","whole","number","digit1","digit2","num_digits"]); w.writerows(rows)
    print(f"Manifest: {out} ({len(rows)} images)")

def parse_numbers(s):
    if s=="all": return list(range(100))
    if "-" in s and "," not in s:
        a,b=map(int,s.split("-")); return list(range(a,b+1))
    return [int(x) for x in s.split(",")]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["simple","complex","manifest","preview","all"],default="simple")
    ap.add_argument("--output",type=Path,default=Path("./synthetic_jersey"))
    ap.add_argument("--simple2d-dir",type=Path,default=None)
    ap.add_argument("--coco-dir",type=Path,default=None,help="Directory of player/jersey background crops")
    ap.add_argument("--font",default=None)
    ap.add_argument("--count-per-class",type=int,default=4000)
    ap.add_argument("--total-count",type=int,default=None)
    ap.add_argument("--single-ratio",type=float,default=.30)
    ap.add_argument("--double-ratio",type=float,default=.70)
    ap.add_argument("--image-size",type=int,default=100)
    ap.add_argument("--numbers",default="all")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument(
        "--no-yolo",
        action="store_true",
        help="Do not create YOLO .txt bounding-box labels.",
    )
    ap.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not create full-image PNG number masks.",
    )
    ap.add_argument(
        "--annotation-csv",
        type=Path,
        default=None,
        help="CSV file for whole/digit/bbox annotations.",
    )
    a=ap.parse_args(); seed_everything(a.seed); nums=parse_numbers(a.numbers)
    simple=a.simple2d_dir or a.output/"Simple2D"; complex_root=a.output/"Complex2D"
    if a.stage in ("simple","all"):
        generate_simple2d(simple,a.count_per_class,a.image_size,a.font,nums,a.total_count,a.single_ratio,a.double_ratio)
    if a.stage in ("complex","all"):
        if a.coco_dir is None: ap.error("--coco-dir is required for complex/all")
        annotation_csv = (
            a.annotation_csv
            if a.annotation_csv is not None
            else a.output / "complex2d_annotations.csv"
        )

        yolo_label_root = (
            None
            if a.no_yolo
            else a.output / "Complex2D_labels"
        )

        mask_root = (
            None
            if a.no_masks
            else a.output / "Complex2D_masks"
        )

        generate_complex2d(
            complex_root,
            simple,
            a.coco_dir,
            a.count_per_class,
            a.image_size,
            nums,
            a.total_count,
            a.single_ratio,
            a.double_ratio,
            yolo_label_root,
            mask_root,
            annotation_csv,
        )
    if a.stage=="manifest":
        if simple.exists(): write_manifest(simple,a.output/"simple2d_manifest.csv")
        if complex_root.exists(): write_manifest(complex_root,a.output/"complex2d_manifest.csv")
    if a.stage=="preview":
        root=complex_root if complex_root.exists() else simple
        make_preview(root,a.output/"preview.jpg")
    if a.stage=="all":
        write_manifest(simple,a.output/"simple2d_manifest.csv"); write_manifest(complex_root,a.output/"complex2d_manifest.csv")
        make_preview(simple,a.output/"simple2d_preview.jpg"); make_preview(complex_root,a.output/"complex2d_preview.jpg")

def make_preview(root,out):
    preferred=[0,1,7,8,12,23,27,29,42,44,66,77,91,99]
    tiles=[]
    for n in preferred:
        ps=list_images(root/str(n)); random.shuffle(ps)
        for p in ps[:3]: tiles.append((n,Image.open(p).convert("RGB").resize((160,160))))
    if not tiles: raise FileNotFoundError(f"No images in {root}")
    cols=5; rows=math.ceil(len(tiles)/cols); sheet=Image.new("RGB",(cols*160,rows*190),"white"); d=ImageDraw.Draw(sheet)
    for i,(n,im) in enumerate(tiles):
        x=i%cols*160; y=i//cols*190; sheet.paste(im,(x,y)); d.text((x+5,y+164),str(n),fill="black")
    out.parent.mkdir(parents=True,exist_ok=True); sheet.save(out); print(f"Preview: {out}")

if __name__=="__main__": main()
