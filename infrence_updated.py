#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

SIZE = 96
FEATURE_DIM = 256
DIGIT_CLASSES = 11
BLANK = 10
NUMBER_CLASSES = 100
EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def images_from(p):
    p = Path(p)
    if not p.exists(): raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.is_file(): return [p]
    out = sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
    if not out: raise FileNotFoundError(f"No images found in: {p}")
    return out

def prep(im):
    im = im.convert("RGB").resize((SIZE,SIZE), Image.Resampling.LANCZOS)
    a = np.asarray(im,dtype=np.float32)/255.0
    return torch.from_numpy(np.transpose(a,(2,0,1))).unsqueeze(0)

def state_dict_from(c):
    if isinstance(c,dict):
        for k in ("state_dict","model_state_dict","model","net"):
            if isinstance(c.get(k),dict): c=c[k]; break
    if not isinstance(c,dict) or not c:
        raise RuntimeError("Checkpoint does not contain a usable state_dict.")
    return {(k[7:] if k.startswith("module.") else k):v for k,v in c.items()}

def load_model(path,device):
    root=Path(__file__).resolve().parent
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    try:
        from subModules.backbone_model_drop import MultiTaskLearner
    except Exception as e:
        raise ImportError("Cannot import subModules.backbone_model_drop.MultiTaskLearner. "
                          "Keep this script inside Release1 and keep subModules unchanged.") from e
    try: model=MultiTaskLearner(out_channels=FEATURE_DIM)
    except TypeError: model=MultiTaskLearner()
    c=torch.load(str(path),map_location=device)
    sd=state_dict_from(c)
    model.load_state_dict(sd,strict=True)
    model.to(device).eval()
    return model

def probs(x):
    x=x.detach().float().cpu().numpy()
    x=x-np.max(x,axis=1,keepdims=True)
    e=np.exp(x); return e/e.sum(axis=1,keepdims=True)

@torch.no_grad()
def predict(model,im,device,threshold):
    x=prep(im).to(device)
    out=model(x)
    if not isinstance(out,(tuple,list)) or len(out)<4:
        raise RuntimeError("Expected MultiTaskLearner output: features, digital, digit_1, digit_2.")
    features,digital,d1,d2=out[:4]
    if features.ndim!=2 or features.shape[-1]!=FEATURE_DIM:
        raise RuntimeError(f"Unexpected feature shape {tuple(features.shape)}; expected [B,{FEATURE_DIM}].")
    if digital.shape[-1]!=100 or d1.shape[-1]!=11 or d2.shape[-1]!=11:
        raise RuntimeError(f"Unexpected heads: digital={tuple(digital.shape)}, "
                           f"digit_1={tuple(d1.shape)}, digit_2={tuple(d2.shape)}")
    pn,p1,p2=probs(digital),probs(d1),probs(d2)
    number=int(np.argmax(pn[0])); nconf=float(pn[0,number])
    r1=int(np.argmax(p1[0])); c1=float(p1[0,r1])
    r2=int(np.argmax(p2[0])); c2=float(p2[0,r2])
    # Final number comes from the trained 100-way digital head.
    if number < 10: digit1,digit2=number,None
    else: digit1,digit2=number//10,number%10
    return {
        "jersey_number":str(number),
        "jersey_number_class":number,
        "confidence":round(nconf,6),
        "accepted":bool(nconf>=threshold),
        "threshold":threshold,
        "digit_1":digit1,
        "digit_1_confidence":round(c1,6),
        "digit_2":digit2,
        "digit_2_confidence":round(c2,6),
        "raw_digit_1_prediction":None if r1==BLANK else r1,
        "raw_digit_2_prediction":None if r2==BLANK else r2,
    }

def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def annotate(im,r,out):
    im=im.convert("RGB").copy(); d=ImageDraw.Draw(im)
    f1,f2=font(24),font(16)
    lines=[(f"Jersey Number: {r['jersey_number']}",f1),
           (f"Confidence: {r['confidence']:.3f}",f2),
           (f"Status: {'ACCEPTED' if r['accepted'] else 'LOW CONFIDENCE'}",f2)]
    widths=[d.textbbox((0,0),t,font=f)[2] for t,f in lines]
    heights=[d.textbbox((0,0),t,font=f)[3] for t,f in lines]
    w=max(widths)+24; h=sum(heights)+30
    d.rectangle((5,5,min(im.width-5,5+w),min(im.height-5,5+h)),fill=(0,0,0))
    y=12
    for (t,f),hh in zip(lines,heights):
        d.text((12,y),t,fill=(255,255,255),font=f); y+=hh+3
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out,quality=95)

def main():
    ap=argparse.ArgumentParser(description="Ying MultiTaskLearner jersey-number inference")
    ap.add_argument("--model",required=True,type=Path,help="Path to .pth/.pt/.ckpt checkpoint")
    ap.add_argument("--input",required=True,type=Path,help="Single image or image directory")
    ap.add_argument("--output_dir","--output-dir",default=Path("inference_results"),type=Path)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--threshold",type=float,default=0.0)
    args=ap.parse_args()
    if not 0<=args.threshold<=1: ap.error("--threshold must be between 0 and 1")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Use --device cpu.")
    dev=torch.device(args.device)
    model=load_model(args.model.expanduser().resolve(),dev)
    inp=args.input.expanduser().resolve()
    imgs=images_from(inp)
    outroot=args.output_dir.expanduser().resolve(); outroot.mkdir(parents=True,exist_ok=True)
    results=[]; ok=0
    print("="*70); print("JERSEY NUMBER CLASSIFICATION INFERENCE"); print("="*70)
    print(f"Model       : {args.model}")
    print(f"Input       : {inp}")
    print(f"Device      : {dev}")
    print(f"Input size  : {SIZE}x{SIZE}")
    print(f"Feature dim : {FEATURE_DIM}")
    print(f"Images      : {len(imgs)}"); print("="*70)
    for i,p in enumerate(imgs,1):
        rec={"image":p.name,"path":str(p)}
        try:
            im=Image.open(p).convert("RGB")
            r=predict(model,im,dev,args.threshold); rec.update(r)
            rel=p.relative_to(inp).parent if inp.is_dir() else Path(".")
            annotate(im,rec,outroot/rel/p.name); ok+=1
            print(f"[{i}/{len(imgs)}] {p.name} -> {r['jersey_number']} "
                  f"(conf={r['confidence']:.3f})")
        except Exception as e:
            rec.update({"jersey_number":None,"confidence":None,"accepted":False,"error":str(e)})
            print(f"[{i}/{len(imgs)}] ERROR {p.name}: {e}")
        results.append(rec)
    payload={"model":str(args.model.resolve()),"input":str(inp),
             "device":str(dev),"input_size":[SIZE,SIZE],"feature_dimension":FEATURE_DIM,
             "threshold":args.threshold,"total_images":len(imgs),
             "successful":ok,"failed":len(imgs)-ok,"predictions":results}
    jp=outroot/"predictions.json"
    jp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print("="*70); print(f"Annotated images : {outroot}")
    print(f"JSON             : {jp}")
    print(f"Successful       : {ok}/{len(imgs)}")
    print("="*70)

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}",file=sys.stderr); sys.exit(1)
