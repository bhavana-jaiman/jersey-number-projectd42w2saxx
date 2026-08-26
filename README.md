# Paper-faithful Simple2D + Complex2D synthetic jersey-number pipeline

This implements the synthetic-data generation procedure described in:

**Bhargavi, Gholami & Pelaez Coyotl (2022), "Jersey number detection using synthetic data in a low-data regime."**

The paper's approach is:

```text
Simple2D
   ↓
Complex2D
   ↓
Real jersey dataset
   ↓
Fine-tuning
```

The paper reports 4,000 images per class for 00-99:
- 400,000 Simple2D images
- 400,000 Complex2D images

## 1. Install dependencies

```bash
python3 -m venv jersey_synth_env
source jersey_synth_env/bin/activate

pip install numpy pillow opencv-python
```

## 2. Find a font

The paper used the Freshman font because it resembled the football jerseys. If you have that font, pass it with `--font`.

Otherwise the script automatically uses a system bold font.

Example:

```bash
fc-list | grep -i "DejaVuSans-Bold"
```

## 3. First run a SMALL test

Do NOT start with 400,000 images.

Generate 20 images/class for classes 00-09:

```bash
python synthetic_jersey_pipeline.py \
  --stage simple \
  --output ./synthetic_test \
  --count-per-class 20 \
  --numbers 0-9
```

You should get:

```text
synthetic_test/
└── Simple2D/
    ├── 00/
    ├── 01/
    ├── ...
    └── 09/
```

## 4. Generate full Simple2D

After visually checking the preview:

```bash
python synthetic_jersey_pipeline.py \
  --stage simple \
  --output ./synthetic_jersey \
  --count-per-class 4000 \
  --numbers all
```

This creates approximately:

```text
100 classes × 4,000 = 400,000 images
```

## 5. COCO images for Complex2D

Download/extract COCO train images and point the script at the image directory.

For example:

```text
coco/
├── 000000000001.jpg
├── 000000000002.jpg
├── ...
```

Then test Complex2D:

```bash
python synthetic_jersey_pipeline.py \
  --stage complex \
  --output ./synthetic_test \
  --coco-dir ./coco \
  --count-per-class 20 \
  --numbers 0-9
```

## 6. Full Complex2D

```bash
python synthetic_jersey_pipeline.py \
  --stage complex \
  --output ./synthetic_jersey \
  --coco-dir ./coco \
  --count-per-class 4000 \
  --numbers all
```

This creates approximately:

```text
100 classes × 4,000 = 400,000 images
```

## 7. Generate manifests

```bash
python synthetic_jersey_pipeline.py \
  --stage manifest \
  --output ./synthetic_jersey
```

Creates:

```text
simple2d_manifest.csv
complex2d_manifest.csv
```

with:

```text
path,label,number
.../00/00_00000.jpg,0,00
.../00/00_00001.jpg,0,00
...
.../23/23_00000.jpg,23,23
```

## 8. Create previews

```bash
python synthetic_jersey_pipeline.py \
  --stage preview \
  --output ./synthetic_jersey
```

Inspect the preview before training.

## 9. Recommended training strategy for your existing model

Do not immediately replace your current model.

Use the paper's curriculum-learning idea:

```text
Your existing model
       ↓
Pretrain on Simple2D
       ↓
Continue training on Complex2D
       ↓
Fine-tune on your REAL jersey dataset
       ↓
Final model
```

For your multitask architecture, you can keep your existing:

```text
MobileNet-style backbone
        ↓
SE
        ↓
SPP
        ↓
Digit-1 head
Digit-2 head
Whole-number head
```

Only the training data/order changes.

## Important differences from the paper

This is a faithful implementation of the *published algorithmic idea*, not the authors' original source code.

The paper states:
- Freshman font
- jersey-like colors
- 100x100 output
- 4,000/class
- Light: Gaussian noise + optical distortion
- Medium: Light + grid distortion
- Hard: Medium + RGB-channel shuffling + random shift-scale-rotation
- Complex2D: Simple2D number over random COCO image
- curriculum: Simple2D → Complex2D → real football data

Some low-level choices (exact distortion parameters, font availability, placement/opacity) are not fully specified in the paper and are therefore configurable/approximated here.

Most importantly, **inspect the generated images**. Synthetic data that looks unlike your real jerseys can create domain shift rather than solve it.
