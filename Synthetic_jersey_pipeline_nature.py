1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
#!/usr/bin/env python3

"""
synthetic_jersey_pipeline_nature.py

Generate a small synthetic jersey-number dataset by overlaying Simple2D
jersey-number images onto real upper-body crops from nature_bodies_9918.

Example:
    python synthetic_jersey_pipeline_nature.py \
        --nature-dir nature_bodies_9918 \
        --simple-dir simple2D \
        --output nature_complex2D \
        --count 100

By default, --count is the TOTAL number of images generated.

If --numbers is 0-99 and --count is 100, the script generates one sample
for each jersey-number class (00 ... 99).

If you want, for example, 20 samples from classes 07, 23 and 45:
    --numbers 7,23,45 --count 20
"""

from pathlib import Path
import argparse
import random
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path):
    """Return image files directly under root."""
    if not root.exists():
        return []

    return [
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def list_images_recursive(root: Path):
    """Return image files under root and its subdirectories."""
