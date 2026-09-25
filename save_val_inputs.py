# save_val_inputs.py
# Saves EVERY validation image as its own PNG file, exactly as it goes into
# the model (after all preprocessing in JerseyNumber_ValidationDataset_V2).
#
# File name:  <index>_<label>_<original file name>.png
#   e.g.  0057_17_201810_image_000184_crop_1.png   -> index 57, number 17
#         0058_4_....png                            -> single digit 4
#         0059_none_....png                         -> no number
#
# Run from ~/workspace_bhavana/New_arcitecture:
#   python save_val_inputs.py              # 96x96, exactly the model input size
#   python save_val_inputs.py --scale 4    # same pixels, shown 4x bigger (384x384)

import os
import argparse
import numpy as np
from PIL import Image
from utils.jersey_dataset_New import JerseyNumber_ValidationDataset_V2 as V

parser = argparse.ArgumentParser()
parser.add_argument('--val_path', default='../Datasets/validation_dataset_Ying')
parser.add_argument('--out', default='val_model_input')
parser.add_argument('--scale', type=int, default=1,
                    help='enlarge for viewing (nearest neighbour: pixels are not changed, only made bigger)')
args = parser.parse_args()

ds = V(args.val_path)
paths = list(ds.images_path)
os.makedirs(args.out, exist_ok=True)

with open(os.path.join(args.out, 'index.txt'), 'w') as f:
    for i, p in enumerate(paths):
        f.write(f'{i}\t{p}\n')

for i in range(len(ds)):
    img, digits, _ = ds[i]                     # exactly what validation gives the model
    # CHW float 0-1  ->  HWC uint8 0-255 (only needed to save as an image file)
    arr = (np.transpose(img, (1, 2, 0)) * 255).round().clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if args.scale > 1:
        im = im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)

    lab = [int(d) for d in np.ravel(digits)]
    label = 'none' if lab[0] == 10 else ''.join(str(d) for d in lab)
    name = os.path.splitext(os.path.basename(paths[i]))[0]
    im.save(os.path.join(args.out, f'{i:04d}_{label}_{name}.png'))   # PNG = no compression loss

    if (i + 1) % 100 == 0:
        print(f'{i + 1}/{len(ds)} saved')

print(f'done: {len(ds)} images saved to {args.out}/  (size {96 * args.scale}x{96 * args.scale})')
