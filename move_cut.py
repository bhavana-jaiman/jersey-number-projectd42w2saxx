# Script 2 of 2: move_cut.py
# Moves the validation images listed in cut_list.txt (one index per line)
# and their label files into ../Datasets/validation_cut/. Nothing is deleted.
#
# Needs val_check/index.txt from check_val_all.py.
# Run from ~/workspace_bhavana/New_arcitecture:
#     python move_cut.py

import os
import shutil

INDEX_FILE = 'val_check/index.txt'
CUT_LIST = 'cut_list.txt'
CUT_DIR = '../Datasets/validation_cut'

if not os.path.exists(INDEX_FILE):
    raise SystemExit(f'{INDEX_FILE} not found - run check_val_all.py first')
if not os.path.exists(CUT_LIST):
    raise SystemExit(f'{CUT_LIST} not found - create it with one index per line')

paths = {}
for line in open(INDEX_FILE):
    i, p = line.rstrip('\n').split('\t')
    paths[int(i)] = p

os.makedirs(os.path.join(CUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(CUT_DIR, 'labels'), exist_ok=True)

indices = sorted({int(l) for l in open(CUT_LIST) if l.strip()})
moved = 0
for i in indices:
    if i not in paths:
        print(f'skip {i}: no such index')
        continue
    img = paths[i]
    if not os.path.exists(img):
        print(f'skip {i}: image already moved or missing ({img})')
        continue
    lbl = img.replace('images', 'labels').replace('.jpg', '.txt')
    if not os.path.exists(lbl):
        lbl = os.path.splitext(img)[0] + '.txt'
    shutil.move(img, os.path.join(CUT_DIR, 'images'))
    if os.path.exists(lbl):
        shutil.move(lbl, os.path.join(CUT_DIR, 'labels'))
    else:
        print(f'note {i}: no label file found for {img}')
    print(f'moved {i}: {os.path.basename(img)}')
    moved += 1

print(f'moved {moved} images to {CUT_DIR}')
print('Run check_val_all.py again before moving more (index numbers change).')
