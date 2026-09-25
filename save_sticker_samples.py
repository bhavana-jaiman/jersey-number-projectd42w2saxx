# save_sticker_samples.py
# Saves training samples that received a pasted STICKER, each as its own PNG.
# Nothing in the dataset file is changed - the script only watches (and, with
# --force, triggers) the sticker code in JerseyWithState_Dataset.
#
# File name: <n>_idx<i>_<label>_s<state>_from<j>-<sticker digits>.png
#   e.g. 0003_idx5231_xx_s2_from18877-27.png
#        -> training image 5231, sticker "27" taken from image 18877,
#           result has 3+ digits -> state 2, digits ignored (x x)
#        0004_idx912_47_s1_from302-7.png
#        -> image 912 showed "4", sticker "7" added -> label 47, state 1
#   label: digit1 digit2  ('-' = blank digit2, 'x' = ignored)
#
# Run from ~/workspace_bhavana/New_arcitecture:
#   python save_sticker_samples.py --force --scale 3            # 200 sticker samples, 3x bigger
#   python save_sticker_samples.py --force --num 500 --compare   # original | with sticker, side by side
#   python save_sticker_samples.py --force --view aug           # also with training augmentation
#   (without --force only ~15% of 1-2 digit images get a sticker, so it takes longer)

import os
import argparse
import random as pyrandom
import numpy as np
from PIL import Image
import torchvision.transforms as T
import utils.jersey_dataset_New as jd

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='../Datasets/training_dataset_Ying')
parser.add_argument('--out', default='sticker_samples')
parser.add_argument('--num', type=int, default=200, help='how many sticker samples to save')
parser.add_argument('--force', action='store_true',
                    help='give EVERY 1-2 digit image a sticker (only for this script)')
parser.add_argument('--view', choices=['crop', 'aug'], default='crop',
                    help='crop = dataset output only, aug = plus training augmentation')
parser.add_argument('--compare', action='store_true',
                    help='save original file (left) next to the sample with sticker (right)')
parser.add_argument('--scale', type=int, default=1, help='enlarge for viewing (pixels not changed)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

pyrandom.seed(args.seed)
np.random.seed(args.seed)

if args.force:
    # the sticker decision is the only place that calls random.random() in the
    # dataset file, so returning 0.0 makes "random.random() < p" always true
    class _AlwaysSticker:
        def random(self):
            return 0.0

        def __getattr__(self, name):
            return getattr(pyrandom, name)

    jd.random = _AlwaysSticker()

if args.view == 'crop':
    transform = T.Compose([T.Resize((96, 96)), T.ToTensor()])
else:
    # keep in sync with train_New.py
    from utils.autoAugment import AutoAugment
    transform = T.Compose([
        T.RandomResizedCrop((96, 96), scale=(0.7, 1.0)),
        T.RandomApply([T.ElasticTransform(alpha=35.0, sigma=5.0)], p=0.6),
        AutoAugment(),
        T.RandomApply([T.RandomRotation(degrees=(-12, 12), fill=0)], p=0.5),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=1)], p=0.25),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.25,
                                     saturation=0.2, hue=0.03)], p=0.7),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.3),
        T.ToTensor(),
    ])

ds = jd.JerseyWithState_Dataset(args.train_path, transform=transform)

# watch the sticker function: remember which image the sticker came from
_original_get_sticker = ds._get_tight_digit_sticker


def _watched_get_sticker(idx):
    sticker, digits = _original_get_sticker(idx)
    ds._last_sticker = (idx, digits) if sticker is not None else None
    return sticker, digits


ds._get_tight_digit_sticker = _watched_get_sticker


def fmt(d):
    d = int(d)
    return '-' if d == 10 else ('x' if d == -100 else str(d))


def to_pil(x):
    arr = (x.numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


os.makedirs(args.out, exist_ok=True)
order = list(range(len(ds)))
pyrandom.shuffle(order)          # spread over the whole dataset, not only the first images

saved = checked = 0
for i in order:
    if saved >= args.num:
        break
    ds._last_sticker = None
    img, digits, _, state = ds[i]
    checked += 1
    if ds._last_sticker is None:
        continue                 # no sticker on this sample

    src, s_digits = ds._last_sticker
    im = to_pil(img)
    if args.compare:
        orig = Image.open(ds.images_files[i]).convert('RGB').resize(im.size)
        both = Image.new('RGB', (im.width * 2 + 4, im.height), 'white')
        both.paste(orig, (0, 0))
        both.paste(im, (im.width + 4, 0))
        im = both
    if args.scale > 1:
        im = im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)

    saved += 1
    label = f'{fmt(digits[0])}{fmt(digits[1])}'
    sticker_txt = ''.join(str(d) for d in s_digits)
    im.save(os.path.join(args.out, f'{saved:04d}_idx{i}_{label}_s{state}_from{src}-{sticker_txt}.png'))
    if saved % 50 == 0:
        print(f'{saved}/{args.num} saved ({checked} images checked)')

print(f'done: {saved} sticker samples saved to {args.out}/  ({checked} images checked)')
