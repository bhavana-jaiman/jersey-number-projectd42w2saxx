#!/usr/bin/env python3
"""
Paper-faithful Simple2D + Complex2D synthetic jersey-number pipeline.

Based on:
Bhargavi, Gholami & Pelaez Coyotl, "Jersey number detection using
synthetic data in a low-data regime", Frontiers in AI, 2022.

Implements the paper's Algorithms 1-3:

Algorithm 1: Number generation
Algorithm 2: Simple2D generation
Algorithm 3: Complex2D generation

The paper generated:
    - 00-99
    - 4,000 images/class
    - 100x100 images
    - Simple2D: 400,000 images
    - Complex2D: 400,000 images

This implementation is configurable so you can start with a small
debug dataset before generating the full 800,000 images.

Training-oriented modification:
    - 30% of generated images are single-digit (0-9).
    - 70% are double-digit (10-99).
    - Double digits are rendered and augmented as ONE object with tight
      spacing, never as two independently transformed 100x100 images.
    - Complex2D removes the Simple2D background before compositing so the
      result contains the number rather than a pasted coloured square.
    - Manifest includes whole, digit1, digit2 and num_digits labels.

Important:
- The original paper used a jersey-like "Freshman" font. This script
  accepts any .ttf/.otf font path; use the same font if you have it.
- The paper used jersey-like background colors. The defaults below are
  approximate and should be changed to match YOUR jersey dataset.
- The paper used Albumentations-like image transformations. We implement
  the stated Light/Medium/Hard operations using OpenCV/numpy/PIL.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

# Paper describes Red, Navy Blue, Green, Red, Yellow, White.
# The paper text contains Red twice; we keep six slots but use two
# slightly different red shades so the configuration is meaningful.
DEFAULT_BG_COLORS = {
    "red_dark": (145, 20, 25),
    "navy": (12, 32, 85),
    "green": (20, 105, 55),
    "red": (205, 35, 35),
    "yellow": (235, 205, 45),
    "white": (245, 245, 245),
}

# White is a common jersey-number foreground color. If your jersey
# uses a different color, change this.
DEFAULT_TEXT_COLORS = [
    (255, 255, 255),
    (245, 245, 245),
]


def choose_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Choose a number colour with enough contrast against the background."""
    luminance = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
    if luminance > 205:
        return (25, 25, 25)
    return random.choice(DEFAULT_TEXT_COLORS)


# ---------------------------------------------------------------------
# Font handling
# ---------------------------------------------------------------------

def find_default_font() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    path = font_path or find_default_font()
    if not path:
        raise FileNotFoundError(
            "No TTF/OTF font found. Pass --font /path/to/font.ttf"
        )
    return ImageFont.truetype(path, size=size)


# ---------------------------------------------------------------------
# Algorithm 1: Number generation
# ---------------------------------------------------------------------

def render_number(
    number: int | str,
    image_size: int = 100,
    font_path: str | None = None,
    bg_color: Tuple[int, int, int] = (20, 30, 60),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    scale_range: Tuple[float, float] = (0.48, 0.70),
    transparent: bool = False,
    gap_ratio: Tuple[float, float] = (0.00, 0.05),
) -> Image.Image:
    """Render the COMPLETE jersey number as ONE visual object.

    0-9  -> one digit
    10-99 -> two digits with very small, natural spacing

    The complete number is rendered first and is augmented later as a
    single object. This avoids the old behaviour where two independent
    100x100 digit images were augmented separately and then concatenated.
    """
    number_str = str(number)
    if not number_str.isdigit() or not 0 <= int(number_str) <= 99:
        raise ValueError(f"number must be an integer in [0, 99], got {number}")

    # Never create leading-zero jersey numbers.
    number_str = str(int(number_str))

    font_size = max(8, int(image_size * random.uniform(*scale_range)))
    font = load_font(font_path, font_size)

    # Measure each character so two-digit kerning can be controlled tightly.
    dummy = Image.new("RGB", (image_size * 2, image_size * 2), bg_color)
    draw = ImageDraw.Draw(dummy)

    if len(number_str) == 1:
        bbox = draw.textbbox((0, 0), number_str, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    else:
        b1 = draw.textbbox((0, 0), number_str[0], font=font)
        b2 = draw.textbbox((0, 0), number_str[1], font=font)
        w1 = b1[2] - b1[0]
        w2 = b2[2] - b2[0]
        gap = int(random.uniform(*gap_ratio) * min(w1, w2))
        tw = w1 + gap + w2
        th = max(b1[3] - b1[1], b2[3] - b2[1])

    if transparent:
        canvas = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        fill = text_color + (255,)
    else:
        canvas = Image.new("RGB", (image_size, image_size), bg_color)
        fill = text_color

    draw = ImageDraw.Draw(canvas)
    x = max(0, (image_size - tw) // 2)
    y = max(0, (image_size - th) // 2)

    if len(number_str) == 1:
        bbox = draw.textbbox((0, 0), number_str, font=font)
        draw.text((x - bbox[0], y - bbox[1]), number_str, font=font, fill=fill)
    else:
        b1 = draw.textbbox((0, 0), number_str[0], font=font)
        b2 = draw.textbbox((0, 0), number_str[1], font=font)
        w1 = b1[2] - b1[0]
        w2 = b2[2] - b2[0]
        gap = int(random.uniform(*gap_ratio) * min(w1, w2))

        draw.text((x - b1[0], y - b1[1]), number_str[0], font=font, fill=fill)
        draw.text((x + w1 + gap - b2[0], y - b2[1]), number_str[1], font=font, fill=fill)

    return canvas


def render_single_digit(
    digit: int,
    image_size: int = 100,
    font_path: str | None = None,
    bg_color: Tuple[int, int, int] = (20, 30, 60),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    scale_range: Tuple[float, float] = (0.48, 0.70),
    transparent: bool = False,
) -> Image.Image:
    """Backward-compatible wrapper for single-digit generation."""
    if not 0 <= int(digit) <= 9:
        raise ValueError("digit must be in [0, 9]")
    return render_number(
        number=int(digit),
        image_size=image_size,
        font_path=font_path,
        bg_color=bg_color,
        text_color=text_color,
        scale_range=scale_range,
        transparent=transparent,
    )


# ---------------------------------------------------------------------
# Augmentations described in Table 1
# ---------------------------------------------------------------------

def gaussian_noise(img: np.ndarray, sigma_range=(3.0, 18.0)) -> np.ndarray:
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def optical_distortion(img: np.ndarray) -> np.ndarray:
    """
    Approximation of optical distortion using OpenCV camera distortion.
    This is intentionally lightweight and does not require a calibration
    model. The paper specifies optical distortion, not a particular
    implementation.
    """
    h, w = img.shape[:2]
    k1 = random.uniform(-0.18, 0.18)
    k2 = random.uniform(-0.08, 0.08)

    fx = w
    fy = h
    cx = w / 2.0
    cy = h / 2.0

    camera = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
        dtype=np.float32,
    )
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    return cv2.undistort(img, camera, dist)


def grid_distortion(img: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """
    Lightweight grid/mesh-like warping. This approximates the stated
    grid distortion operation without adding a large dependency.
    """
    h, w = img.shape[:2]

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    # Smooth sinusoidal displacement.
    amp_x = random.uniform(0.01, strength) * w
    amp_y = random.uniform(0.01, strength) * h
    freq_x = random.uniform(0.8, 2.0)
    freq_y = random.uniform(0.8, 2.0)

    map_x = xx + amp_x * np.sin(2 * np.pi * yy / h * freq_y)
    map_y = yy + amp_y * np.sin(2 * np.pi * xx / w * freq_x)

    return cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rgb_channel_shuffle(img: np.ndarray) -> np.ndarray:
    order = random.choice([
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ])
    return img[:, :, list(order)]


def random_shift_scale_rotate(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    angle = random.uniform(-18, 18)
    scale = random.uniform(0.82, 1.18)
    tx = random.uniform(-0.12, 0.12) * w
    ty = random.uniform(-0.12, 0.12) * h

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_light(img: np.ndarray) -> np.ndarray:
    img = gaussian_noise(img)
    img = optical_distortion(img)
    return img


def apply_medium(img: np.ndarray) -> np.ndarray:
    img = apply_light(img)
    img = grid_distortion(img)
    return img


def apply_hard(img: np.ndarray) -> np.ndarray:
    img = apply_medium(img)
    img = rgb_channel_shuffle(img)
    img = random_shift_scale_rotate(img)
    return img


def apply_random_paper_augmentation(img: np.ndarray) -> np.ndarray:
    """Apply one paper-style augmentation level to the COMPLETE number.

    The original implementation independently augmented each digit. This
    version applies the same transform to the complete number object.
    RGB-channel shuffling is intentionally omitted because it can create
    unrealistic jersey-number colours; geometry/noise are retained.
    """
    level = random.choice([
        "light",
        "medium", "medium", "medium", "medium", "medium",
        "hard", "hard", "hard", "hard", "hard",
    ])
    if level == "light":
        return gaussian_noise(optical_distortion(img), (2.0, 10.0))
    if level == "medium":
        out = optical_distortion(img)
        out = grid_distortion(out, strength=0.07)
        return gaussian_noise(out, (2.0, 10.0))
    out = optical_distortion(img)
    out = grid_distortion(out, strength=0.08)
    out = random_shift_scale_rotate(out)
    return gaussian_noise(out, (1.0, 8.0))


# ---------------------------------------------------------------------
# Digit -> two-digit composition
# ---------------------------------------------------------------------

def concat_two_digits(
    left: np.ndarray,
    right: np.ndarray,
    canvas_size: int = 100,
) -> np.ndarray:
    """Backward-compatible helper; no longer used for new generation."""
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[:, :left.shape[1]] = left
    canvas[:, left.shape[1]:] = right
    return cv2.resize(canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)


def generate_simple2d_class(
    number: int,
    out_dir: Path,
    count: int = 4000,
    image_size: int = 100,
    font_path: str | None = None,
) -> None:
    """Generate one class, rendering the whole number as one object.

    Unlike the original paper-faithful implementation, this version does
    NOT create two independent 100x100 digits and concatenate them. For a
    double digit such as 28, it renders ``28`` tightly, then applies the
    same rotation/distortion/noise/scaling to the complete object.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = list(DEFAULT_BG_COLORS.values())
    per_color = max(1, math.ceil(count / len(colors)))
    candidates: List[np.ndarray] = []

    for bg in colors:
        for _ in range(per_color):
            number_img = render_number(
                number=number,
                image_size=image_size,
                font_path=font_path,
                bg_color=bg,
                text_color=choose_text_color(bg),
                scale_range=(0.48, 0.70),
                gap_ratio=(0.00, 0.05),
            )
            img = np.asarray(number_img).copy()
            # Apply all geometry/noise to the COMPLETE number.
            img = apply_random_paper_augmentation(img)
            candidates.append(img)

    random.shuffle(candidates)
    candidates = candidates[:count]
    label = str(number)  # no leading zero in the visual label/name
    for idx, img in enumerate(candidates):
        cv2.imwrite(
            str(out_dir / f"{label}_{idx:05d}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )


def _counts_from_ratio(
    numbers: List[int],
    total_count: int,
    single_ratio: float = 0.30,
    double_ratio: float = 0.70,
) -> dict[int, int]:
    """Distribute total images so 30% are 0-9 and 70% are 10-99.

    Within each group, images are distributed as evenly as possible and a
    small random remainder is assigned, so the final dataset has the exact
    requested group ratio (up to integer rounding).
    """
    if abs(single_ratio + double_ratio - 1.0) > 1e-6:
        raise ValueError("single_ratio + double_ratio must equal 1.0")

    singles = [n for n in numbers if 0 <= n <= 9]
    doubles = [n for n in numbers if 10 <= n <= 99]
    if not singles or not doubles:
        raise ValueError("numbers must contain both single and double digit classes")

    n_single = round(total_count * single_ratio)
    n_double = total_count - n_single

    counts = {n: 0 for n in numbers}
    for group, amount in ((singles, n_single), (doubles, n_double)):
        base, rem = divmod(amount, len(group))
        for n in group:
            counts[n] = base
        # Randomly distribute remainder while keeping reproducibility via seed_everything.
        for n in random.sample(group, rem):
            counts[n] += 1
    return counts


def generate_simple2d(
    root: Path,
    count_per_class: int = 4000,
    image_size: int = 100,
    font_path: str | None = None,
    numbers: Iterable[int] = range(100),
    total_count: int | None = None,
    single_ratio: float = 0.30,
    double_ratio: float = 0.70,
) -> None:
    """Generate Simple2D.

    If ``total_count`` is provided, exactly 30% of the images are assigned
    to single-digit classes (0-9) and 70% to double-digit classes (10-99).
    This is the recommended mode for training your multitask model.

    If ``total_count`` is omitted, the old ``count_per_class`` behaviour is
    retained for compatibility.
    """
    root.mkdir(parents=True, exist_ok=True)
    numbers = list(numbers)

    if total_count is not None:
        counts = _counts_from_ratio(numbers, total_count, single_ratio, double_ratio)
    else:
        counts = {n: count_per_class for n in numbers}

    for number in numbers:
        count = counts[number]
        if count <= 0:
            continue
        print(f"[Simple2D] Generating {number} ({count} images) ...")
        generate_simple2d_class(
            number=number,
            out_dir=root / str(number),
            count=count,
            image_size=image_size,
            font_path=font_path,
        )


# ---------------------------------------------------------------------
# Algorithm 3: Complex2D
# ---------------------------------------------------------------------

def list_images(root: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def _extract_number_mask(number_img: np.ndarray) -> np.ndarray:
    """Estimate the foreground digit mask from a Simple2D image.

    Simple2D has a mostly uniform jersey-like background. Estimating the
    border/background colour lets us remove that background before placing
    the number on a real image. This avoids pasting a coloured square.
    """
    img = number_img.astype(np.float32)
    h, w = img.shape[:2]
    b = max(2, int(min(h, w) * 0.08))
    border = np.concatenate([
        img[:b].reshape(-1, 3),
        img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3),
        img[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(img - bg[None, None, :], axis=2)
    # Adaptive threshold. The percentile guard handles noisy backgrounds.
    threshold = max(18.0, float(np.percentile(dist, 82)))
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _tight_crop(img: np.ndarray, mask: np.ndarray, pad: int = 2):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img, mask
    x1, x2 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _warp_number_object(
    patch: np.ndarray,
    mask: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale/rotate/perspective-warp the COMPLETE number object together."""
    ph, pw = patch.shape[:2]
    scale = random.uniform(0.65, 1.00)
    nw = max(8, int(pw * scale))
    nh = max(8, int(ph * scale))
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    angle = random.uniform(-8.0, 8.0)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    rw = max(8, int(nh * sin + nw * cos))
    rh = max(8, int(nh * cos + nw * sin))
    M[0, 2] += rw / 2 - nw / 2
    M[1, 2] += rh / 2 - nh / 2
    patch = cv2.warpAffine(patch, M, (rw, rh), borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    # Mild perspective distortion applied to the whole number, not per digit.
    h, w = patch.shape[:2]
    jitter = random.uniform(0.02, 0.07)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx, dy = w * jitter, h * jitter
    dst = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - 1 - random.uniform(0, dx), random.uniform(0, dy)],
        [w - 1 - random.uniform(0, dx), h - 1 - random.uniform(0, dy)],
        [random.uniform(0, dx), h - 1 - random.uniform(0, dy)],
    ])
    H = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(patch, H, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return patch, mask


def alpha_overlay_number(
    background: np.ndarray,
    number_img: np.ndarray,
    x: int,
    y: int,
    opacity: float = 0.90,
) -> np.ndarray:
    """Mask-aware number overlay; the Simple2D background is NOT pasted."""
    bg = background.copy()
    mask = _extract_number_mask(number_img)
    patch, mask = _tight_crop(number_img, mask)
    patch, mask = _warp_number_object(patch, mask, bg.shape[0])

    h, w = patch.shape[:2]
    x2, y2 = min(bg.shape[1], x + w), min(bg.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return bg
    patch = patch[:y2-y, :x2-x]
    mask = mask[:y2-y, :x2-x]

    alpha = (mask.astype(np.float32) / 255.0) * opacity
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
    base = bg[y:y2, x:x2].astype(np.float32)
    fg = patch.astype(np.float32)

    # Small local lighting variation makes the print less synthetic.
    brightness = random.uniform(0.82, 1.08)
    fg = np.clip(fg * brightness, 0, 255)

    blended = fg * alpha + base * (1.0 - alpha)
    bg[y:y2, x:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def generate_complex2d_class(
    number: int,
    simple_root: Path,
    coco_images: List[Path],
    out_dir: Path,
    count: int = 4000,
    image_size: int = 100,
) -> None:
    """Create realistic composites by placing a complete number object on images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    number_dir = simple_root / str(number)
    number_images = list_images(number_dir)
    if not number_images:
        raise FileNotFoundError(f"No Simple2D images found for class {number}: {number_dir}")
    if not coco_images:
        raise FileNotFoundError("No background images supplied.")

    for idx in range(count):
        coco_path = random.choice(coco_images)
        num_path = random.choice(number_images)
        bg = cv2.imread(str(coco_path))
        number_img = cv2.imread(str(num_path))
        if bg is None or number_img is None:
            continue
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        number_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2RGB)

        # Preserve background variety and create a square training crop.
        bh, bw = bg.shape[:2]
        crop = min(bh, bw)
        x0 = random.randint(0, bw - crop)
        y0 = random.randint(0, bh - crop)
        bg = cv2.resize(bg[y0:y0+crop, x0:x0+crop], (image_size, image_size), interpolation=cv2.INTER_AREA)

        mask = _extract_number_mask(number_img)
        number_patch, mask = _tight_crop(number_img, mask, pad=2)
        number_patch, mask = _warp_number_object(number_patch, mask, image_size)

        ph, pw = number_patch.shape[:2]
        # Keep number reasonably sized: roughly 18-55% of image width.
        desired_w = int(image_size * random.uniform(0.18, 0.55))
        if pw > 0:
            scale = desired_w / pw
            nh = max(8, int(ph * scale))
            nw = max(8, int(pw * scale))
            number_patch = cv2.resize(number_patch, (nw, nh), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        ph, pw = number_patch.shape[:2]
        if ph >= image_size or pw >= image_size:
            scale = min((image_size - 4) / max(ph, 1), (image_size - 4) / max(pw, 1))
            pw, ph = max(8, int(pw * scale)), max(8, int(ph * scale))
            number_patch = cv2.resize(number_patch, (pw, ph), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)

        x = random.randint(0, image_size - pw)
        y = random.randint(0, image_size - ph)

        # Blend only the number pixels, not the solid Simple2D square.
        local = bg[y:y+ph, x:x+pw].astype(np.float32)
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0)
        alpha *= random.uniform(0.78, 0.98)
        fg = number_patch.astype(np.float32)
        fg *= random.uniform(0.86, 1.08)
        bg[y:y+ph, x:x+pw] = np.clip(fg * alpha[..., None] + local * (1-alpha[..., None]), 0, 255).astype(np.uint8)

        # Mild image degradation, applied after compositing.
        if random.random() < 0.45:
            bg = cv2.GaussianBlur(bg, (3, 3), 0)
        if random.random() < 0.35:
            bg = gaussian_noise(bg, (1.0, 6.0))

        cv2.imwrite(str(out_dir / f"{number}_{idx:05d}.jpg"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def generate_complex2d(
    root: Path,
    simple_root: Path,
    coco_dir: Path,
    count_per_class: int = 4000,
    image_size: int = 100,
    numbers: Iterable[int] = range(100),
    total_count: int | None = None,
    single_ratio: float = 0.30,
    double_ratio: float = 0.70,
) -> None:
    coco_images = list_images(coco_dir)

    if not coco_images:
        raise FileNotFoundError(
            f"No images found under COCO directory: {coco_dir}"
        )

    root.mkdir(parents=True, exist_ok=True)

    print(f"[Complex2D] COCO images found: {len(coco_images)}")

    numbers = list(numbers)
    if total_count is not None:
        counts = _counts_from_ratio(numbers, total_count, single_ratio, double_ratio)
    else:
        counts = {n: count_per_class for n in numbers}

    for number in numbers:
        count = counts[number]
        if count <= 0:
            continue
        print(f"[Complex2D] Generating {number} ({count} images) ...")
        generate_complex2d_class(
            number=number,
            simple_root=simple_root,
            coco_images=coco_images,
            out_dir=root / str(number),
            count=count,
            image_size=image_size,
        )


# ---------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------

def write_manifest(dataset_root: Path, output_csv: Path) -> None:
    """Write labels for both the whole-number and multitask heads."""
    import csv

    rows = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        try:
            label = int(class_dir.name)
        except ValueError:
            continue

        for img in sorted(list_images(class_dir)):
            if label <= 9:
                digit1, digit2, num_digits = label, -1, 1
            else:
                digit1, digit2, num_digits = label // 10, label % 10, 2
            rows.append((str(img.resolve()), label, str(label), digit1, digit2, num_digits))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "whole", "number", "digit1", "digit2", "num_digits"])
        writer.writerows(rows)

    print(f"Manifest written: {output_csv}")
    print(f"Images: {len(rows)}")


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def make_preview(
    dataset_root: Path,
    output: Path,
    numbers: List[int] | None = None,
    per_number: int = 5,
) -> None:
    """Create a contact sheet for quick visual inspection."""
    if numbers is None:
        available = []
        for d in dataset_root.iterdir() if dataset_root.exists() else []:
            if d.is_dir() and d.name.isdigit():
                available.append(int(d.name))
        if not available:
            raise FileNotFoundError("No numbered class directories found for preview.")
        # Prefer representative single + double digit examples.
        preferred = [0, 1, 7, 8, 12, 23, 27, 42, 44, 66, 77, 91, 99]
        numbers = [n for n in preferred if n in available]
        if not numbers:
            numbers = available[:10]

    tiles = []
    for n in numbers:
        paths = list_images(dataset_root / str(n))
        random.shuffle(paths)
        for p in paths[:per_number]:
            img = Image.open(p).convert("RGB").resize((160, 160))
            tiles.append((n, img))

    if not tiles:
        raise FileNotFoundError("No images found for preview.")

    cols = 5
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 160, rows * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (n, img) in enumerate(tiles):
        x = (i % cols) * 160
        y = (i // cols) * 190
        sheet.paste(img, (x, y))
        draw.text((x + 5, y + 164), str(n), fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Preview saved: {output}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_numbers(spec: str) -> List[int]:
    """
    Examples:
        all
        0-99
        0,1,2,44,66
    """
    spec = spec.strip().lower()

    if spec == "all":
        return list(range(100))

    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))

    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple2D + Complex2D jersey synthetic-data pipeline"
    )

    parser.add_argument(
        "--stage",
        choices=["simple", "complex", "manifest", "preview", "all"],
        default="simple",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./synthetic_jersey"),
    )

    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=None,
        help="Directory containing extracted COCO images.",
    )

    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Path to a TTF/OTF jersey-like font. Default: system bold font.",
    )

    parser.add_argument(
        "--count-per-class",
        type=int,
        default=4000,
        help="Legacy mode: number of images per jersey number when --total-count is not used.",
    )

    parser.add_argument(
        "--total-count",
        type=int,
        default=None,
        help="Recommended training mode: total Simple2D/Complex2D images. Defaults to 30%% single-digit and 70%% double-digit.",
    )

    parser.add_argument(
        "--single-ratio",
        type=float,
        default=0.30,
        help="Fraction of total images for 0-9. Default: 0.30.",
    )

    parser.add_argument(
        "--double-ratio",
        type=float,
        default=0.70,
        help="Fraction of total images for 10-99. Default: 0.70.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--numbers",
        type=str,
        default="all",
        help="all, 0-99, or comma-separated classes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    numbers = parse_numbers(args.numbers)

    simple_root = args.output / "Simple2D"
    complex_root = args.output / "Complex2D"

    if args.stage in {"simple", "all"}:
        generate_simple2d(
            root=simple_root,
            count_per_class=args.count_per_class,
            image_size=args.image_size,
            font_path=args.font,
            numbers=numbers,
            total_count=args.total_count,
            single_ratio=args.single_ratio,
            double_ratio=args.double_ratio,
        )

    if args.stage in {"complex", "all"}:
        if args.coco_dir is None:
            parser.error("--coco-dir is required for --stage complex/all")

        generate_complex2d(
            root=complex_root,
            simple_root=simple_root,
            coco_dir=args.coco_dir,
            count_per_class=args.count_per_class,
            image_size=args.image_size,
            numbers=numbers,
            total_count=args.total_count,
            single_ratio=args.single_ratio,
            double_ratio=args.double_ratio,
        )

    if args.stage == "manifest":
        write_manifest(simple_root, args.output / "simple2d_manifest.csv")
        if complex_root.exists():
            write_manifest(complex_root, args.output / "complex2d_manifest.csv")

    if args.stage == "preview":
        which = simple_root if simple_root.exists() else complex_root
        make_preview(which, args.output / "preview.jpg")

    if args.stage == "all":
        write_manifest(simple_root, args.output / "simple2d_manifest.csv")
        write_manifest(complex_root, args.output / "complex2d_manifest.csv")
        make_preview(simple_root, args.output / "simple2d_preview.jpg")
        make_preview(complex_root, args.output / "complex2d_preview.jpg")


if __name__ == "__main__":
    main()
