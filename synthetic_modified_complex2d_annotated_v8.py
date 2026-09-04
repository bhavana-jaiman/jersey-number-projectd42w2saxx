#!/usr/bin/env python3
"""Complex2D jersey-number generator using real jersey crops with existing YOLO labels.

Version 5: robust placement for enlarged/rotated numbers; prevents ROI broadcasting errors.

Input directory contains images AND their .txt labels in the SAME directory.
Each input label line is standard YOLO:
    class_id x_center y_center width height
The default source class is 0. If a file contains multiple class-0 objects,
all class-0 boxes are combined into one old-number region. Other classes are ignored.

For every output image, a matching .txt file is written beside it:
    jersey_number x_center y_center width height
where the coordinates are normalized to the output image. By default the original
jersey-crop dimensions are preserved. Use --output-size resize to force a square
output. A CSV summary with pixel bboxes is also written.
"""
import argparse,csv,math,random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def seed_everything(seed): random.seed(seed); np.random.seed(seed)

def list_images(root):
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts] if root.exists() else []

def find_default_font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf","/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf","/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",str(Path.home()/".local/share/fonts/LiberationSans-Bold.ttf"),str(Path.home()/".local/share/fonts/FreeSansBold.ttf")):
        if Path(p).exists(): return p
    return None

def load_font(path,size):
    path=path or find_default_font()
    if not path: raise FileNotFoundError("No TTF/OTF font found. Use --font.")
    return ImageFont.truetype(path,size)

def choose_text_color(bg):
    lum=.2126*bg[0]+.7152*bg[1]+.0722*bg[2]
    return (25,25,25) if lum>205 else (255,255,255)

def render_number_with_mask(number,image_size=100,font_path=None,bg_color=(20,30,60),text_color=(255,255,255),scale_range=(.48,.70),gap_ratio=(0,.02)):
    n=int(number); s=str(n)
    font=load_font(font_path,max(8,int(image_size*random.uniform(*scale_range))))
    im=Image.new("RGB",(image_size,image_size),bg_color); mask=Image.new("L",(image_size,image_size),0)
    d=ImageDraw.Draw(im); md=ImageDraw.Draw(mask)
    if len(s)==1:
        b=d.textbbox((0,0),s,font=font); w=b[2]-b[0]; h=b[3]-b[1]; x=max(0,(image_size-w)//2); y=max(0,(image_size-h)//2); pos=(x-b[0],y-b[1]); d.text(pos,s,font=font,fill=text_color); md.text(pos,s,font=font,fill=255)
    else:
        b1=d.textbbox((0,0),s[0],font=font); b2=d.textbbox((0,0),s[1],font=font); w1=b1[2]-b1[0]; w2=b2[2]-b2[0]; h=max(b1[3]-b1[1],b2[3]-b2[1]); gap=int(random.uniform(*gap_ratio)*max(1,min(w1,w2))); x=max(0,(image_size-(w1+gap+w2))//2); y=max(0,(image_size-h)//2); p1=(x-b1[0],y-b1[1]); p2=(x+w1+gap-b2[0],y-b2[1]); d.text(p1,s[0],font=font,fill=text_color); d.text(p2,s[1],font=font,fill=text_color); md.text(p1,s[0],font=font,fill=255); md.text(p2,s[1],font=font,fill=255)
    return np.asarray(im).copy(),np.asarray(mask).copy()

def tight_crop(img,mask,pad=2):
    ys,xs=np.where(mask>0)
    if len(xs)==0:return img,mask
    x1=max(0,int(xs.min())-pad); x2=min(img.shape[1],int(xs.max())+pad+1); y1=max(0,int(ys.min())-pad); y2=min(img.shape[0],int(ys.max())+pad+1)
    return img[y1:y2,x1:x2],mask[y1:y2,x1:x2]

def bbox(mask):
    ys,xs=np.where(mask>0)
    return None if len(xs)==0 else (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))

def gaussian_noise(img,sigma=(2,12)):
    n=np.random.normal(0,random.uniform(*sigma),img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+n,0,255).astype(np.uint8)

def grid_distortion(img,strength=.10):
    h,w=img.shape[:2]; yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing="ij")
    ax=random.uniform(.01,strength)*w; ay=random.uniform(.01,strength)*h; fx=random.uniform(.8,2); fy=random.uniform(.8,2)
    mx=xx+ax*np.sin(2*np.pi*yy/max(h,1)*fy); my=yy+ay*np.sin(2*np.pi*xx/max(w,1)*fx)
    return cv2.remap(img,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def directional_bend(patch,mask,strength,direction):
    h,w=patch.shape[:2]; src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]); dx=max(1,w*strength); dy=max(1,h*strength*.55)
    if direction=="left": dst=np.float32([[dx,dy*.35],[w-1,0],[w-1-dx*.55,h-1],[0,h-1-dy*.35]])
    elif direction=="right": dst=np.float32([[0,0],[w-1-dx,dy*.35],[w-1,h-1],[dx*.55,h-1-dy*.35]])
    else: dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H=cv2.getPerspectiveTransform(src,dst)
    p=cv2.warpPerspective(patch,H,(w,h),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    m=cv2.warpPerspective(mask,H,(w,h),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return p,np.clip(m,0,255).astype(np.uint8)

def warp_number_object(patch,mask,rotation_min=-30,rotation_max=30,perspective_min=.025,perspective_max=.075,grid_prob=.40,grid_strength_min=.015,grid_strength_max=.035,noise_prob=.30,noise_min=.5,noise_max=2.0):
    ph,pw=patch.shape[:2]
    sc=random.uniform(.98,1.02); nw=max(8,int(pw*sc)); nh=max(8,int(ph*sc)); patch=cv2.resize(patch,(nw,nh),cv2.INTER_CUBIC); mask=cv2.resize(mask,(nw,nh),cv2.INTER_NEAREST)
    angle=random.uniform(rotation_min,rotation_max); direction=random.choices(["straight","left","right"],weights=[.30,.35,.35])[0]
    M=cv2.getRotationMatrix2D((nw/2,nh/2),angle,1.0); c=abs(M[0,0]); s=abs(M[0,1]); rw=max(8,int(nh*s+nw*c)); rh=max(8,int(nh*c+nw*s)); M[0,2]+=rw/2-nw/2; M[1,2]+=rh/2-nh/2
    patch=cv2.warpAffine(patch,M,(rw,rh),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101); mask=cv2.warpAffine(mask,M,(rw,rh),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    patch,mask=directional_bend(patch,mask,random.uniform(perspective_min,perspective_max),direction)
    if random.random()<grid_prob:
        # Slight cloth-like deformation, kept moderate.
        patch=grid_distortion(patch,random.uniform(grid_strength_min,grid_strength_max))
    if random.random()<noise_prob:
        patch=gaussian_noise(patch,(noise_min,noise_max))
    return patch,np.clip(mask,0,255).astype(np.uint8)

def read_yolo_boxes(txt_path,class_id=None):
    boxes=[]
    if not txt_path.exists(): return boxes
    for ln,line in enumerate(txt_path.read_text(errors="ignore").splitlines(),1):
        s=line.strip()
        if not s or s.startswith("#"): continue
        parts=s.split()
        if len(parts)<5: continue
        try:
            cls=int(float(parts[0])); xc,yc,w,h=map(float,parts[1:5])
        except ValueError: continue
        if class_id is not None and cls!=class_id: continue
        boxes.append((xc,yc,w,h))
    return boxes

def yolo_to_pixel(box,W,H):
    xc,yc,w,h=box; x1=int(round((xc-w/2)*W)); y1=int(round((yc-h/2)*H)); x2=int(round((xc+w/2)*W))-1; y2=int(round((yc+h/2)*H))-1
    return max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)

def box_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1+1); ih=max(0,iy2-iy1+1)
    inter=iw*ih
    aa=(ax2-ax1+1)*(ay2-ay1+1); ab=(bx2-bx1+1)*(by2-by1+1)
    return inter/max(aa+ab-inter,1)

def boxes_are_adjacent(a,b,distance_ratio=.75,y_center_ratio=.40,height_ratio=.55):
    """Check whether two labeled boxes plausibly form adjacent jersey digits."""
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ah=ay2-ay1+1; bh=by2-by1+1
    aw=ax2-ax1+1; bw=bx2-bx1+1
    acx=(ax1+ax2)/2; acy=(ay1+ay2)/2
    bcx=(bx1+bx2)/2; bcy=(by1+by2)/2
    left,right=sorted([(ax1,ax2),(bx1,bx2)],key=lambda z:z[0])
    horizontal_gap=max(0,right[0]-left[1]-1)
    avg_w=max((aw+bw)/2,1.0)
    avg_h=max((ah+bh)/2,1.0)
    center_y_gap=abs(acy-bcy)
    height_ratio_diff=abs(ah-bh)/avg_h
    vertical_overlap=max(0,min(ay2,by2)-max(ay1,by1)+1)/max(min(ah,bh),1)
    if box_iou(a,b)>0.05:
        return True
    return (horizontal_gap <= distance_ratio*avg_w
            and center_y_gap <= y_center_ratio*avg_h
            and height_ratio_diff <= height_ratio
            and (vertical_overlap >= 0.25 or center_y_gap <= 0.25*avg_h))

def group_number_boxes(boxes,W,H,distance_ratio=.75,y_center_ratio=.40,height_ratio=.55):
    """Group nearby labeled boxes into candidate jersey-number regions. The grouping is based on geometry, not class ID."""
    px=[yolo_to_pixel(b,W,H) for b in boxes]
    n=len(px)
    if n==0: return []
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj: parent[rj]=ri
    for i in range(n):
        for j in range(i+1,n):
            if boxes_are_adjacent(px[i],px[j],distance_ratio,y_center_ratio,height_ratio):
                union(i,j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(px[i])
    return list(groups.values())

def expanded_group_bbox(group,W,H,pad_ratio=.08):
    x1=min(b[0] for b in group); y1=min(b[1] for b in group)
    x2=max(b[2] for b in group); y2=max(b[3] for b in group)
    pw=x2-x1+1; ph=y2-y1+1
    pad=max(1,int(round(max(pw,ph)*pad_ratio)))
    return max(0,x1-pad),max(0,y1-pad),min(W-1,x2+pad),min(H-1,y2+pad)

def select_number_group(boxes,W,H,distance_ratio=.75,y_center_ratio=.40,height_ratio=.55):
    """Select the most plausible jersey-number group from all labeled boxes.

    Two nearby boxes are treated as the two digits of one double-digit number.
    Distant boxes are not merged. If multiple candidate groups exist, prefer a
    two-box group, then a single box, then the largest remaining group.
    """
    groups=group_number_boxes(boxes,W,H,distance_ratio,y_center_ratio,height_ratio)
    if not groups: return None,[]
    doubles=[g for g in groups if len(g)==2]
    if doubles:
        g=max(doubles,key=lambda z:sum((b[2]-b[0]+1)*(b[3]-b[1]+1) for b in z))
        return expanded_group_bbox(g,W,H),g
    singles=[g for g in groups if len(g)==1]
    if singles:
        g=max(singles,key=lambda z:(z[0][2]-z[0][0]+1)*(z[0][3]-z[0][1]+1))
        return expanded_group_bbox(g,W,H),g
    g=max(groups,key=lambda z:(len(z),sum((b[2]-b[0]+1)*(b[3]-b[1]+1) for b in z)))
    return expanded_group_bbox(g,W,H),g

def remove_number_region(image, old_bbox, method="inpaint", radius=5, blur_kernel=21):
    """Remove the complete old jersey number using the already-expanded bbox."""
    x1,y1,x2,y2=old_bbox
    mask=np.zeros(image.shape[:2],np.uint8)
    mask[y1:y2+1,x1:x2+1]=255
    if method == "blur":
        k=max(3,int(blur_kernel))
        if k % 2 == 0: k += 1
        blurred=cv2.GaussianBlur(image,(k,k),0)
        return np.where(mask[...,None]>0,blurred,image)
    k=max(1,int(radius))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1))
    mask=cv2.dilate(mask,kernel)
    return cv2.inpaint(image,mask,float(radius),cv2.INPAINT_TELEA)

def transform_to_box(patch,mask,target_w,target_h,keep_aspect=True):
    if keep_aspect:
        sc=min(target_w/max(patch.shape[1],1),target_h/max(patch.shape[0],1)); nw=max(2,int(round(patch.shape[1]*sc))); nh=max(2,int(round(patch.shape[0]*sc)))
    else: nw=max(2,target_w); nh=max(2,target_h)
    patch=cv2.resize(patch,(nw,nh),cv2.INTER_CUBIC); mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
    return patch,mask

def paste_center(bg,patch,mask,cx,cy,margin=2):
    """Paste a transformed number safely at (cx, cy).

    The previous implementation failed when an enlarged/rotated number became
    wider or taller than the available ROI.  In that case the destination ROI
    could be smaller than the patch, producing a NumPy broadcasting error.
    Here we first fit an oversized patch to the available image area, preserving
    aspect ratio, and then paste it. This only reduces the number when necessary
    to keep the COMPLETE number visible.
    """
    h,w=bg.shape[:2]
    ph,pw=patch.shape[:2]
    margin=max(0,int(margin))
    avail_w=max(2,w-2*margin)
    avail_h=max(2,h-2*margin)

    if pw>avail_w or ph>avail_h:
        sc=min(avail_w/max(pw,1),avail_h/max(ph,1))
        nw=max(2,int(round(pw*sc)))
        nh=max(2,int(round(ph*sc)))
        patch=cv2.resize(patch,(nw,nh),cv2.INTER_CUBIC)
        mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
        ph,pw=nh,nw

    x=int(round(cx-pw/2))
    y=int(round(cy-ph/2))
    x=max(margin,min(x,w-margin-pw))
    y=max(margin,min(y,h-margin-ph))

    roi=bg[y:y+ph,x:x+pw]
    a=(mask.astype(np.float32)/255.0)[...,None]
    blended=patch.astype(np.float32)*a+roi.astype(np.float32)*(1-a)
    bg[y:y+ph,x:x+pw]=np.clip(blended,0,255).astype(np.uint8)
    return bg,x,y

def process_source(image_path,label_path,simple_image_path,number,out_path,image_size,args):
    img=cv2.imread(str(image_path)); ni=cv2.imread(str(simple_image_path))
    if img is None or ni is None: return False,"read_error"
    H,W=img.shape[:2]; boxes=read_yolo_boxes(label_path,args.source_class_id)
    if not boxes: return False,"no_matching_labels"
    # IMPORTANT: consider ALL labeled objects by default. We do not assume that
    # jersey-number digits have a particular YOLO class ID. Nearby boxes are
    # grouped geometrically (e.g. two close digit boxes -> one double-digit region);
    # distant boxes are kept separate so unrelated annotations are not erased.
    old_bbox,number_group=select_number_group(
        boxes,W,H,
        args.digit_distance_ratio,
        args.digit_y_center_ratio,
        args.digit_height_ratio
    )
    if old_bbox is None:
        return False,"no_number_group"
    clean=remove_number_region(img,old_bbox,args.remove_method,args.inpaint_radius,args.blur_kernel)
    # Generate number from Simple2D image using color-distance mask from its border.
    ni_rgb=cv2.cvtColor(ni,cv2.COLOR_BGR2RGB); x=ni_rgb.astype(np.float32); b=max(2,int(min(ni.shape[:2])*.08)); border=np.concatenate([x[:b].reshape(-1,3),x[-b:].reshape(-1,3),x[:,:b].reshape(-1,3),x[:,-b:].reshape(-1,3)]); bg=np.median(border,axis=0); dist=np.linalg.norm(x-bg[None,None,:],axis=2); mask=(dist>max(18,float(np.percentile(dist,82)))).astype(np.uint8)*255; mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)); patch,mask=tight_crop(ni_rgb,mask,2)
    if np.count_nonzero(mask)<8:return False,"bad_simple2d_mask"
    patch,mask=warp_number_object(patch,mask,args.rotation_min,args.rotation_max,args.perspective_min,args.perspective_max,args.grid_prob,args.grid_strength_min,args.grid_strength_max,args.noise_prob,args.noise_min,args.noise_max)
    patch,mask=tight_crop(patch,mask,1)
    ox1,oy1,ox2,oy2=old_bbox; tw=ox2-ox1+1; th=oy2-oy1+1
    if args.size_mode == "original":
        # Fit the new number to the original annotated number region.
        patch,mask=transform_to_box(patch,mask,tw,th,args.keep_aspect)
        cx=(ox1+ox2)/2; cy=(oy1+oy2)/2
    elif args.size_mode == "enlarge":
        # Make the generated number slightly larger than the old bbox while
        # keeping its center at the original annotated location.
        target_w=max(8,int(round(tw*args.scale_factor)))
        target_h=max(8,int(round(th*args.scale_factor)))
        patch,mask=transform_to_box(patch,mask,target_w,target_h,args.keep_aspect)
        cx=(ox1+ox2)/2; cy=(oy1+oy2)/2
    else:
        # Fixed mode: use the same 45--55% image-height target used by
        # our previous Complex2D version, while keeping the number centered
        # at the original number location.
        target_h=max(8,int(H*random.uniform(.45,.55)))
        sc=target_h/max(patch.shape[0],1)
        nw=max(8,int(round(patch.shape[1]*sc))); nh=max(8,int(round(patch.shape[0]*sc)))
        patch=cv2.resize(patch,(nw,nh),cv2.INTER_CUBIC); mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
        cx=(ox1+ox2)/2; cy=(oy1+oy2)/2
        # Prevent fixed-size placement from going outside the source crop.
        if nw>W-4 or nh>H-4:
            sc=min((W-4)/max(nw,1),(H-4)/max(nh,1)); nw=max(8,int(nw*sc)); nh=max(8,int(nh*sc))
            patch=cv2.resize(patch,(nw,nh),cv2.INTER_AREA); mask=cv2.resize(mask,(nw,nh),cv2.INTER_LINEAR)
    clean,px,py=paste_center(clean,patch,mask,cx,cy,args.placement_margin)
    # By default preserve the REAL input jersey-crop dimensions.
    # Resize only when --output-size resize is explicitly requested.
    mb=bbox(mask)
    if mb is None:return False,"empty_mask"
    mx1,my1,mx2,my2=mb; fx1=px+mx1; fy1=py+my1; fx2=px+mx2; fy2=py+my2
    if args.output_size == "resize":
        sx=image_size/W; sy=image_size/H
        out=cv2.resize(clean,(image_size,image_size),cv2.INTER_AREA)
        x1=int(round(fx1*sx)); y1=int(round(fy1*sy)); x2=int(round(fx2*sx)); y2=int(round(fy2*sy))
        label_size=image_size
    else:
        out=clean
        x1,y1,x2,y2=fx1,fy1,fx2,fy2
        label_size=W  # returned separately below; actual H is used by caller
    out_h,out_w=out.shape[:2]
    x1=max(0,min(out_w-1,int(round(x1)))); y1=max(0,min(out_h-1,int(round(y1))))
    x2=max(x1,min(out_w-1,int(round(x2)))); y2=max(y1,min(out_h-1,int(round(y2))))
    cv2.imwrite(str(out_path),out,[cv2.IMWRITE_JPEG_QUALITY,95])
    return True,(x1,y1,x2,y2,out_w,out_h)

def write_label(path,number,bbox,image_w,image_h):
    x1,y1,x2,y2=bbox; xc=((x1+x2)/2)/image_w; yc=((y1+y2)/2)/image_h; w=(x2-x1+1)/image_w; h=(y2-y1+1)/image_h
    path.write_text(f"{number} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def parse_numbers(s):
    if s=="all":return list(range(100))
    if "-" in s and "," not in s:
        a,b=map(int,s.split("-"));return list(range(a,b+1))
    return [int(x) for x in s.split(",")]

def counts_from_ratio(numbers,total,single=.30,double=.70):
    if abs(single+double-1)>1e-6:raise ValueError("ratios must sum to 1")
    singles=[n for n in numbers if n<=9]; doubles=[n for n in numbers if n>=10]
    if not singles or not doubles:raise ValueError("numbers must include both singles and doubles")
    ns=round(total*single); nd=total-ns; out={n:0 for n in numbers}
    for g,a in ((singles,ns),(doubles,nd)):
        q,r=divmod(a,len(g));
        for n in g:out[n]=q
        for n in random.sample(g,r):out[n]+=1
    return out

def main():
    ap=argparse.ArgumentParser(description="Complex2D: replace existing jersey number in real jersey crops using YOLO labels")
    ap.add_argument("--simple2d-dir",type=Path,required=True,help="Simple2D directory containing number class folders")
    ap.add_argument("--jersey-dir",type=Path,required=True,help="Directory containing jersey images AND matching .txt labels in the SAME directory")
    ap.add_argument("--output",type=Path,default=Path("./synthetic_complex2d"))
    ap.add_argument("--count-per-class",type=int,default=4000)
    ap.add_argument("--total-count",type=int,default=None)
    ap.add_argument("--single-ratio",type=float,default=.30)
    ap.add_argument("--double-ratio",type=float,default=.70)
    ap.add_argument("--numbers",default="all")
    ap.add_argument("--image-size",type=int,default=100,help="Final size only when --output-size resize")
    ap.add_argument("--output-size",choices=["original","resize"],default="original",help="original=preserve source jersey-crop dimensions; resize=force image-size x image-size")
    ap.add_argument("--source-class-id",default="all",help="Which YOLO class IDs to consider when finding the old number: all (default) or a specific integer class ID")
    ap.add_argument("--digit-distance-ratio",dest="digit_distance_ratio",type=float,default=.75,help="Maximum horizontal gap between two label boxes, relative to average digit width, to consider them adjacent digits")
    ap.add_argument("--digit-y-center-ratio",dest="digit_y_center_ratio",type=float,default=.40,help="Maximum vertical center separation between candidate digit boxes, relative to average digit height")
    ap.add_argument("--digit-height-ratio",dest="digit_height_ratio",type=float,default=.55,help="Maximum relative height difference between boxes considered adjacent digits")
    ap.add_argument("--remove-pad-ratio",type=float,default=.08,help="Extra margin around the COMPLETE old-number bbox before removal; 0.08 = 8%%")
    ap.add_argument("--remove-method",choices=["inpaint","blur"],default="inpaint",help="How to remove the complete old-number region")
    ap.add_argument("--inpaint-radius",type=int,default=5,help="Inpainting radius when --remove-method inpaint")
    ap.add_argument("--blur-kernel",type=int,default=21,help="Odd Gaussian kernel when --remove-method blur")
    ap.add_argument("--size-mode",choices=["original","enlarge","fixed"],default="enlarge",help="original=same bbox; enlarge=slightly larger than old bbox; fixed=use fixed target fraction")
    ap.add_argument("--scale-factor",type=float,default=1.15,help="Number enlargement factor for --size-mode enlarge; 1.15 = 15%% larger")
    ap.add_argument("--placement-margin",type=int,default=2,help="Minimum pixel margin kept from crop edges when placing the generated number; oversized numbers are safely scaled down only if needed")
    ap.add_argument("--keep-aspect",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--rotation-min",type=float,default=-30.0); ap.add_argument("--rotation-max",type=float,default=30.0)
    ap.add_argument("--perspective-min",type=float,default=.025); ap.add_argument("--perspective-max",type=float,default=.075)
    ap.add_argument("--grid-prob",type=float,default=.40); ap.add_argument("--grid-strength-min",type=float,default=.015); ap.add_argument("--grid-strength-max",type=float,default=.035)
    ap.add_argument("--noise-prob",type=float,default=.30); ap.add_argument("--noise-min",type=float,default=.5); ap.add_argument("--noise-max",type=float,default=2.0)
    ap.add_argument("--font",default=None)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); seed_everything(a.seed); nums=parse_numbers(a.numbers)
    if str(a.source_class_id).lower() == "all":
        a.source_class_id = None
    else:
        a.source_class_id = int(a.source_class_id)
    images=list_images(a.jersey_dir)
    if not images: raise FileNotFoundError(f"No images found under {a.jersey_dir}")
    valid=[]
    for p in images:
        lp=p.with_suffix('.txt')
        if read_yolo_boxes(lp,a.source_class_id):valid.append((p,lp))
    if not valid: raise FileNotFoundError(f"No images with usable labels found in {a.jersey_dir}")
    total=a.total_count if a.total_count is not None else a.count_per_class*len(nums)
    counts=counts_from_ratio(nums,total,a.single_ratio,a.double_ratio) if a.total_count is not None else {n:a.count_per_class for n in nums}
    a.output.mkdir(parents=True,exist_ok=True); rows=[]
    for n in nums:
        if not counts[n]:continue
        src_nums=list_images(a.simple2d_dir/str(n))
        if not src_nums: raise FileNotFoundError(f"No Simple2D images for number {n}: {a.simple2d_dir/str(n)}")
        made=0; attempts=0
        print(f"[Complex2D] {n}: {counts[n]}")
        while made<counts[n] and attempts<counts[n]*20:
            attempts+=1; img_path,label_path=random.choice(valid); simple_path=random.choice(src_nums)
            out_dir=a.output/str(n); out_dir.mkdir(parents=True,exist_ok=True); out_path=out_dir/f"{n}_{made:05d}.jpg"
            ok,res=process_source(img_path,label_path,simple_path,n,out_path,a.image_size,a)
            if not ok:continue
            bbox_out=res; x1,y1,x2,y2,out_w,out_h=bbox_out; write_label(out_path.with_suffix('.txt'),n,(x1,y1,x2,y2),out_w,out_h); rows.append([str(out_path.resolve()),n,x1,y1,x2,y2,str(img_path.resolve()),str(label_path.resolve())]); made+=1
        if made<counts[n]:raise RuntimeError(f"Generated only {made}/{counts[n]} for class {n}. Check labels and crop quality.")
    with (a.output/"annotations.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["image_path","jersey_number","bbox_x1","bbox_y1","bbox_x2","bbox_y2","source_image","source_label"]);w.writerows(rows)
    print(f"\nDone. Images + .txt labels: {a.output}/<number>/")
    print(f"CSV summary: {a.output/'annotations.csv'}")

if __name__=="__main__":main()
