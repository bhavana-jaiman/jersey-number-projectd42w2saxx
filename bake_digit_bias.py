"""
bake_digit_bias.py -- writes the digit biases from calibrate_digits.py (digit_bias.json)
INTO a copy of a checkpoint: digit_1.bias (10 values) and digit_2.bias (11 values).
Same model, same size, 0 extra FLOPs. Can be applied on top of an already state-baked checkpoint.

Run:
  python bake_digit_bias.py --ckpt kd_79.74_s2b1.25.pth --json digit_bias.json --out kd_79.74_s2b1.25_dig.pth
"""
import argparse
import json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--out", required=True)
opt = ap.parse_args()

ck = torch.load(opt.ckpt, map_location="cpu")
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
bias = json.load(open(opt.json))

for name, size in (("digit1", 10), ("digit2", 11)):
    keys = [k for k, v in sd.items() if k.endswith("bias") and tuple(v.shape) == (size,)]
    if len(keys) != 1:
        raise SystemExit(f"ERROR: {len(keys)} bias vectors with {size} values: {keys} -> send me this")
    k = keys[0]
    sd[k] = sd[k] + torch.tensor(bias[name], dtype=sd[k].dtype)
    print(f"{k}: added {[round(v, 3) for v in bias[name]]}")
torch.save(ck, opt.out)
print("saved", opt.out)
