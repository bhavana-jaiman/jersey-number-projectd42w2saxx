"""
Quick sanity check for the new super_high_res teacher patch applied to
subModules/backbone_ying.py's featureExtractor_Ying class.

Runs a dummy 96x96 input through all 4 flag combinations and prints the
resulting feature map spatial size. Expected output:

    high_res=False super_high_res=False -> (6, 6)    (original student, UNCHANGED)
    high_res=False super_high_res=True  -> (12, 12)
    high_res=True  super_high_res=False -> (12, 12)   (existing teacher, UNCHANGED)
    high_res=True  super_high_res=True  -> (24, 24)   (NEW teacher)

Usage (run from the workspace root, so the subModules import resolves):
    python check_teacher_sizes.py
"""
import torch
from subModules.backbone_ying import featureExtractor_Ying

torch.set_grad_enabled(False)
x = torch.randn(1, 3, 96, 96)

combos = [(False, False), (False, True), (True, False), (True, True)]

for hr, shr in combos:
    m = featureExtractor_Ying(out_channels=32, high_res=hr, super_high_res=shr)
    m.eval()
    out = m(x)
    shape = tuple(out.shape[2:])
    print("high_res=" + str(hr) + " super_high_res=" + str(shr) + " -> " + str(shape))
