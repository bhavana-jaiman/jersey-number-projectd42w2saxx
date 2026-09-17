#!/usr/bin/env python3
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE=96
BLANK=10
EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def images_from(p):
    p=Path(p).expanduser().resolve()
    if not p.exists(): raise FileNotFoundError(f'Input path does not exist: {p}')
    if p.is_file(): return [p]
    imgs=sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in EXTS)
    if not imgs: raise FileNotFoundError(f'No images found in: {p}')
    return imgs

def prep(im):
    im=im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.LANCZOS)
    a=np.asarray(im,dtype=np.float32)/255.0
    return np.transpose(a,(2,0,1)).astype(np.float32)

def batch_np(ims): return np.stack([prep(im) for im in ims]).astype(np.float32)

def softmax(x):
    x=np.asarray(x,dtype=np.float32); x=x-x.max(axis=1,keepdims=True); e=np.exp(x); return e/e.sum(axis=1,keepdims=True)

def decode100(logits,threshold):
    p=softmax(logits); out=[]
    for row in p:
        n=int(row.argmax()); c=float(row[n]); d1=n if n<10 else n//10; d2=None if n<10 else n%10
        out.append({'jersey_number':n,'confidence':round(c,6),'accepted':c>=threshold,'source':'whole_number_head','digit_1':d1,'digit_1_confidence':None,'digit_2':d2,'digit_2_confidence':None,'single_digit_prediction':n if n<10 else None,'single_digit_confidence':round(c,6) if n<10 else None})
    return out

def decode_heads(single,d1log,d2log,threshold):
    ps,p1,p2=softmax(single),softmax(d1log),softmax(d2log); out=[]
    for a,b,c in zip(ps,p1,p2):
        sd=int(a.argmax()); sc=float(a[sd]); x=int(b.argmax()); xc=float(b[x]); y=int(c.argmax()); yc=float(c[y])
        two=None if y==BLANK else x*10+y; tc=0.0 if two is None else (xc+yc)/2.0
        if two is not None and tc>sc: n,conf,src=two,tc,'digit_heads'
        else: n,conf,src=sd,sc,'number_head'
        out.append({'jersey_number':n,'confidence':round(conf,6),'accepted':conf>=threshold,'source':src,'digit_1':x,'digit_1_confidence':round(xc,6),'digit_2':None if y==BLANK else y,'digit_2_confidence':round(yc,6),'single_digit_prediction':sd,'single_digit_confidence':round(sc,6)})
    return out

class ONNXRunner:
    def __init__(self,path,device):
        try: import onnxruntime as ort
        except ImportError as e: raise RuntimeError('onnxruntime is not installed. Install onnxruntime or onnxruntime-gpu.') from e
        avail=ort.get_available_providers()
        providers=['CUDAExecutionProvider','CPUExecutionProvider'] if device=='cuda' and 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider']
        if device=='cuda' and 'CUDAExecutionProvider' not in avail: print('[WARNING] CUDAExecutionProvider unavailable; using CPUExecutionProvider.')
        self.s=ort.InferenceSession(str(path),providers=providers); self.i=self.s.get_inputs()[0]
        print('\nONNX MODEL'); print('  model:',path); print('  input:',self.i.name,self.i.shape); print('  providers:',self.s.get_providers())
        for o in self.s.get_outputs(): print('  output:',o.name,o.shape)
    def predict(self,batch,threshold):
        vals=self.s.run(None,{self.i.name:batch})
        pairs=[(o.name.lower(),np.asarray(v)) for o,v in zip(self.s.get_outputs(),vals)]
        whole=single=d1=d2=None
        for name,v in pairs:
            if v.ndim!=2: continue
            k=v.shape[-1]
            if 'feature' in name: continue
            if name=='digit1' or name.endswith('/digit1'): single=v
            elif 'digit_1' in name: d1=v
            elif 'digit_2' in name: d2=v
            elif any(t in name for t in ('digital','whole','number')): whole=v
        # Dimension fallback for the observed exported layout: 256,10,10,11.
        if whole is None or single is None or d1 is None or d2 is None:
            for _,v in pairs:
                if v.ndim!=2: continue
                k=v.shape[-1]
                if k==100 and whole is None: whole=v
                elif k==10:
                    if single is None: single=v
                    elif d1 is None: d1=v
                elif k==11 and d2 is None: d2=v
        if whole is not None and whole.shape[-1]==100: return decode100(whole,threshold)
        if single is None or d1 is None or d2 is None:
            raise RuntimeError('Could not map ONNX outputs. Found: '+str({n:list(v.shape) for n,v in pairs}))
        return decode_heads(single,d1,d2,threshold)

def unwrap(c):
    if isinstance(c,dict):
        for k in ('state_dict','model_state_dict','model','net','weights'):
            if isinstance(c.get(k),dict): c=c[k]; break
    if not isinstance(c,dict) or not c: raise RuntimeError('Checkpoint does not contain a usable state_dict.')
    return {(k[7:] if k.startswith('module.') else k):v for k,v in c.items()}

def infer_dim(sd):
    for k in ('digital.weight','digit1.weight','digit_1.weight'):
        if k in sd and getattr(sd[k],'ndim',0)==2: return int(sd[k].shape[1])
    return None

class CheckpointRunner:
    def __init__(self,path,device,module_name,class_name):
        import torch
        self.torch=torch; self.device=torch.device(device)
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but torch.cuda.is_available() is False. Use --device cpu.')
        root=Path(__file__).resolve().parent
        if str(root) not in sys.path: sys.path.insert(0,str(root))
        c=torch.load(str(path),map_location=self.device); sd=unwrap(c); dim=infer_dim(sd)
        mod=__import__(module_name,fromlist=[class_name]); cls=getattr(mod,class_name)
        try: model=cls(out_channels=dim) if dim is not None else cls()
        except TypeError: model=cls()
        try: model.load_state_dict(sd,strict=True)
        except RuntimeError as e: raise RuntimeError(f'Checkpoint architecture mismatch for {module_name}.{class_name}. Detected feature dim={dim}. Original error: {e}') from e
        model.to(self.device).eval(); self.model=model
        print('\nPYTORCH CHECKPOINT'); print('  model:',path); print('  device:',self.device); print('  class:',module_name+'.'+class_name); print('  feature dim:',dim)
    def predict(self,batch,threshold):
        x=self.torch.from_numpy(batch).to(self.device)
        with self.torch.no_grad(): out=self.model(x)
        if not isinstance(out,(tuple,list)): raise RuntimeError(f'Model returned {type(out)}, expected tuple/list.')
        arr=[v.detach().float().cpu().numpy() for v in out if hasattr(v,'shape')]
        whole=single=d1=d2=None
        for v in arr:
            if v.ndim!=2: continue
            k=v.shape[-1]
            if k==100 and whole is None: whole=v
            elif k==10:
                if single is None: single=v
                elif d1 is None: d1=v
            elif k==11 and d2 is None: d2=v
        if whole is not None: return decode100(whole,threshold)
        if single is not None and d1 is not None and d2 is not None: return decode_heads(single,d1,d2,threshold)
        raise RuntimeError('Could not map checkpoint outputs. Returned shapes: '+str([list(v.shape) for v in arr]))

def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'):
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def gt_box(label,w,h):
    if not label.exists(): return None
    try: r=np.loadtxt(str(label)).reshape(-1,5)[0]
    except Exception: return None
    _,xc,yc,bw,bh=r
    return (int((xc-bw/2)*w),int((yc-bh/2)*h),int((xc+bw/2)*w),int((yc+bh/2)*h),int(r[0]))

def annotate(im,r,out,label=None):
    im=im.convert('RGB').copy(); d=ImageDraw.Draw(im); box=gt_box(label,im.width,im.height) if label else None
    if box:
        x1,y1,x2,y2,cl=box; d.rectangle((x1,y1,x2,y2),outline=(255,255,0),width=3); d.text((x1+3,max(0,y1-20)),f'GT class: {cl}',fill=(255,255,0),font=font(max(14,im.width//35)))
    lines=[f"Predicted: {r['jersey_number']}",f"Confidence: {r['confidence']:.3f}",f"Source: {r['source']}"]
    fs=[font(max(20,im.width//18)),font(max(14,im.width//30)),font(max(14,im.width//30))]
    bb=[d.textbbox((0,0),t,font=f) for t,f in zip(lines,fs)]; pw=min(im.width-10,max(b[2]-b[0] for b in bb)+24); ph=min(im.height-10,sum(b[3]-b[1] for b in bb)+28)
    d.rectangle((5,5,5+pw,5+ph),fill=(0,0,0)); y=12
    for t,f,b in zip(lines,fs,bb): d.text((12,y),t,fill=(255,255,255),font=f); y+=(b[3]-b[1])+3
    out.parent.mkdir(parents=True,exist_ok=True); im.save(out,quality=95)

def main():
    ap=argparse.ArgumentParser(description='Jersey-number inference: ONNX + PyTorch checkpoint')
    ap.add_argument('--model',required=True,type=Path); ap.add_argument('--input',required=True,type=Path); ap.add_argument('--output_dir','--output-dir',default=Path('inference_results'),type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda'); ap.add_argument('--threshold',type=float,default=0.0); ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--checkpoint-module',default='subModules.backbone_model_drop'); ap.add_argument('--checkpoint-class',default='MultiTaskLearner')
    a=ap.parse_args()
    if not 0<=a.threshold<=1: ap.error('--threshold must be between 0 and 1')
    if a.batch_size<1: ap.error('--batch-size must be >=1')
    model=a.model.expanduser().resolve(); inp=a.input.expanduser().resolve(); out=a.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    imgs=images_from(inp); ext=model.suffix.lower()
    if ext=='.onnx': runner=ONNXRunner(model,a.device); typ='onnx'
    elif ext in ('.pth','.pt','.ckpt'): runner=CheckpointRunner(model,a.device,a.checkpoint_module,a.checkpoint_class); typ='checkpoint'
    else: raise ValueError('Model must be .onnx, .pth, .pt or .ckpt')
    print('\n'+'='*78); print('JERSEY NUMBER CLASSIFICATION INFERENCE'); print('='*78); print('Model type:',typ); print('Model:',model); print('Input:',inp); print('Images:',len(imgs)); print('Output:',out); print('Input size: 96x96'); print('Threshold:',a.threshold); print('='*78)
    recs=[]; ok=0
    for s in range(0,len(imgs),a.batch_size):
        paths=imgs[s:s+a.batch_size]; ims=[]; good=[]
        for p in paths:
            try: ims.append(Image.open(p).convert('RGB')); good.append(p)
            except Exception as e: recs.append({'image':p.name,'path':str(p),'jersey_number':None,'confidence':None,'accepted':False,'error':f'image read error: {e}'}); print('[ERROR]',p.name,e)
        if not good: continue
        try:
            rs=runner.predict(batch_np(ims),a.threshold)
            for p,im,r in zip(good,ims,rs):
                rel=p.relative_to(inp).parent if inp.is_dir() else Path('.')
                apath=out/'annotated'/rel/p.name; lab=p.with_suffix('.txt')
                annotate(im,r,apath,lab if lab.exists() else None)
                recs.append({'image':p.name,'path':str(p),'annotated_image':str(apath),**r}); ok+=1
                print(f'[{ok}/{len(imgs)}] {p.name}: number={r["jersey_number"]} confidence={r["confidence"]:.3f} source={r["source"]}')
        except Exception as e:
            print(f'[WARNING] Batch failed ({len(good)} images): {e}'); print('[INFO] Retrying one image at a time...')
            for p,im in zip(good,ims):
                try:
                    r=runner.predict(batch_np([im]),a.threshold)[0]; rel=p.relative_to(inp).parent if inp.is_dir() else Path('.'); apath=out/'annotated'/rel/p.name; lab=p.with_suffix('.txt'); annotate(im,r,apath,lab if lab.exists() else None)
                    recs.append({'image':p.name,'path':str(p),'annotated_image':str(apath),**r}); ok+=1; print(f'[{ok}/{len(imgs)}] {p.name}: number={r["jersey_number"]} confidence={r["confidence"]:.3f} source={r["source"]}')
                except Exception as e2:
                    recs.append({'image':p.name,'path':str(p),'jersey_number':None,'confidence':None,'accepted':False,'error':str(e2)}); print('[ERROR]',p.name,e2)
    payload={'model_type':typ,'model':str(model),'input':str(inp),'device':a.device,'input_size':[SIZE,SIZE],'threshold':a.threshold,'batch_size':a.batch_size,'total_images':len(imgs),'successful':ok,'failed':len(imgs)-ok,'predictions':recs}
    jp=out/'predictions.json'; jp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    cp=out/'results.csv'; fields=['image','path','annotated_image','jersey_number','confidence','accepted','source','digit_1','digit_1_confidence','digit_2','digit_2_confidence','single_digit_prediction','single_digit_confidence','error']
    with cp.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(recs)
    print('\n'+'='*78); print('INFERENCE COMPLETE'); print('='*78); print(f'Total images: {len(imgs)}'); print(f'Successful: {ok}'); print(f'Failed: {len(imgs)-ok}'); print('Annotated images:',out/'annotated'); print('Predictions JSON:',jp); print('Results CSV:',cp); print('='*78)

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: sys.exit(130)
    except Exception as e: print(f'\n[FATAL ERROR] {e}',file=sys.stderr); sys.exit(1)
