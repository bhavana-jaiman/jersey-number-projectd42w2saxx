# check_pages.py
# Saves training or validation images on pages (100 per page), with the
# index number and label printed under every image.
#
# Run from ~/workspace_bhavana/New_arcitecture, for example:
#   python check_pages.py --set val                   # validation, exactly what the model sees
#   python check_pages.py --set val --view original   # validation, original files (nothing cut)
#   python check_pages.py --set train                 # training, dataset crop (no random augmentation)
#   python check_pages.py --set train --view aug      # training, with full training augmentation
#   python check_pages.py --set train --max 2000      # more training images (default 1000)
#
# Labels under each image:
#   validation:  "1 7" = 17,  "4" = single digit,  "none" = no number
#   training:    "1 7 s1" = 17 state 1,  "4 - s1" = single 4 (digit2 blank),
#                "x x s0" = no number,  "x x s2" = 3+ digits (digits masked)

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--set', choices=['train', 'val'], default='val')
parser.add_argument('--view', default=None,
                    help='train: crop (default) or aug | val: model (default) or original')
parser.add_argument('--max', type=int, default=None,
                    help='max images to show (default: all for val, 1000 for train)')
parser.add_argument('--train_path', default='../Datasets/training_dataset_Ying')
parser.add_argument('--val_path', default='../Datasets/validation_dataset_Ying')
args = parser.parse_args()

PER_PAGE, COLS, SIZE = 100, 10, 96


def to_pil(x):
    """CHW float image in 0-1 (numpy array or torch tensor) -> PIL image."""
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return Image.fromarray((np.transpose(x, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8))


def fmt_digit(d):
    d = int(d)
    if d == 10:
        return '-'
    if d == -100:
        return 'x'
    return str(d)


if args.set == 'val':
    view = args.view or 'model'
    if view not in ('model', 'original'):
        raise SystemExit('for --set val, --view must be model or original')
    from utils.jersey_dataset_New import JerseyNumber_ValidationDataset_V2 as V
    ds = V(args.val_path)
    paths = list(ds.images_path)
    out_dir = 'val_check' if view == 'model' else 'val_check_original'
    max_images = args.max

    def get(i):
        img, digits, _ = ds[i]
        if view == 'original':
            im = Image.open(paths[i]).convert('RGB').resize((SIZE, SIZE))
        else:
            im = to_pil(img)
        lab = [int(d) for d in np.ravel(digits)]
        text = 'none' if lab[0] == 10 else ' '.join(str(d) for d in lab)
        return im, text

else:
    view = args.view or 'crop'
    if view not in ('crop', 'aug'):
        raise SystemExit('for --set train, --view must be crop or aug')
    from utils.jersey_dataset_New import JerseyWithState_Dataset as D

    if view == 'crop':
        # dataset crop only (the dataset itself still adds stickers / grid lines at random)
        transform = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])
    else:
        # full training augmentation - keep this in sync with train_New.py
        from utils.autoAugment import AutoAugment
        transform = T.Compose([
            T.RandomResizedCrop((96, 96), scale=(0.7, 1.0)),
            # T.RandomApply([T.Resize(32), T.Resize((96, 96))], p=0.3),  # un-comment if added in train_New.py
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

    ds = D(args.train_path, transform=transform)
    paths = list(ds.images_files)
    out_dir = f'train_check_{view}'
    max_images = args.max if args.max is not None else 1000

    def get(i):
        img, digits, _, state = ds[i]
        return to_pil(img), f'{fmt_digit(digits[0])} {fmt_digit(digits[1])} s{state}'


os.makedirs(out_dir, exist_ok=True)

# index -> file path for every image in the set (move_cut.py uses val_check/index.txt)
with open(os.path.join(out_dir, 'index.txt'), 'w') as f:
    for i, p in enumerate(paths):
        f.write(f'{i}\t{p}\n')

# which images to show: all, or evenly spread across the whole set
total = len(ds)
if max_images is not None and max_images < total:
    step = total / max_images
    indices = [int(k * step) for k in range(max_images)]
else:
    indices = list(range(total))

pages = 0
for start in range(0, len(indices), PER_PAGE):
    chunk = indices[start:start + PER_PAGE]
    rows = (len(chunk) + COLS - 1) // COLS
    page = Image.new('RGB', (COLS * SIZE, rows * (SIZE + 16)), 'white')
    draw = ImageDraw.Draw(page)
    for k, i in enumerate(chunk):
        im, text = get(i)
        x, y = (k % COLS) * SIZE, (k // COLS) * (SIZE + 16)
        page.paste(im, (x, y))
        draw.text((x + 2, y + SIZE + 2), f'{i}: {text}', fill='black')
    pages += 1
    page.save(os.path.join(out_dir, f'page_{pages:02d}.png'))

print(f'{args.set} / {view}: saved {len(indices)} of {total} images into {pages} pages in {out_dir}/')
