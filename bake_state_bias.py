"""
bake_state_bias.py -- writes the calibration bias INTO a checkpoint, so the normal
(unchanged) testing script gives the calibrated result.

It adds the bias to the bias vector of the last layer of the state head (3 values:
State 0 / 1 / 2). Same model, same size, 0 extra FLOPs.

Run:
  python bake_state_bias.py --ckpt kd_best_79.74.pth --state2 1.25 --out kd_79.74_s2b1.25.pth
  then test --out with the normal testing script (6x6 model -> no --high_res)
Options:
  --state0 -0.5   also lower State 0 (only after checking on empty jerseys!)
  --key name      if the script cannot find the state-head bias by itself
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--state2", type=float, default=0.0)
ap.add_argument("--state0", type=float, default=0.0)
ap.add_argument("--key", default="")
opt = ap.parse_args()

ck = torch.load(opt.ckpt, map_location="cpu")
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

cands = [k for k, v in sd.items() if k.endswith("bias") and tuple(v.shape) == (3,)]
print("bias vectors with 3 values:", cands)
if opt.key:
    key = opt.key
elif len(cands) == 1:
    key = cands[0]
else:
    st = [k for k in cands if "state" in k.lower()]
    if len(st) != 1:
        raise SystemExit("ERROR: cannot decide which one is the state head -> send me the list above")
    key = st[0]

before = sd[key].clone()
sd[key] = sd[key] + torch.tensor([opt.state0, 0.0, opt.state2], dtype=sd[key].dtype)
print(f"{key}: {before.tolist()} -> {sd[key].tolist()}")
torch.save(ck, opt.out)
print("saved", opt.out)
