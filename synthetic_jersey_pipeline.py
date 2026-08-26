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

def render_single_digit(
    digit: int,
    image_size: int = 100,
    font_path: str | None = None,
    bg_color: Tuple[int, int, int] = (20, 30, 60),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    scale_range: Tuple[float, float] = (0.45, 0.85),
    transparent: bool = False,
) -> Image.Image:
    """
    Algorithm 1:
      - choose jersey background + font/text color
      - choose a random font size/scaling factor
      - paste/render the single number

    Returns an RGB image or RGBA image if transparent=True.
    """
    scale = random.uniform(*scale_range)
    font_size = max(8, int(image_size * scale))
    font = load_font(font_path, font_size)

    if transparent:
        canvas = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (image_size, image_size), bg_color)

    draw = ImageDraw.Draw(canvas)

    # Find tight text bounding box.
    bbox = draw.textbbox((0, 0), str(digit), font=font, stroke_width=0)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Random position while trying to keep the digit in-frame.
    max_x = max(0, image_size - tw - 2)
    max_y = max(0, image_size - th - 2)
    x = random.randint(0, max_x) if max_x else 0
    y = random.randint(0, max_y) if max_y else 0

    draw.text(
        (x - bbox[0], y - bbox[1]),
        str(digit),
        fill=text_color + ((255,) if transparent else ()),
        font=font,
    )

    return canvas


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
    """
    The paper uses one Light, five Medium and five Hard augmentations
    on each digit. Here we randomly choose one of those levels for each
    generated instance. Use --augmentation fixed if you want a fixed
    level.
    """
    level = random.choice(["light", "medium", "medium", "medium",
                           "medium", "medium",
                           "hard", "hard", "hard", "hard", "hard"])
    if level == "light":
        return apply_light(img)
    if level == "medium":
        return apply_medium(img)
    return apply_hard(img)


# ---------------------------------------------------------------------
# Digit -> two-digit composition
# ---------------------------------------------------------------------

def concat_two_digits(
    left: np.ndarray,
    right: np.ndarray,
    canvas_size: int = 100,
) -> np.ndarray:
    """
    Concatenate two 100x100-ish digit images horizontally and resize
    back to 100x100, following Algorithm 2.
    """
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)

    h = max(left.shape[0], right.shape[0])
    canvas_w = left.shape[1] + right.shape[1]

    canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)

    # Preserve the left/right generated backgrounds rather than forcing
    # one common color.
    canvas[:, :left.shape[1]] = left
    canvas[:, left.shape[1]:] = right

    return cv2.resize(
        canvas,
        (canvas_size, canvas_size),
        interpolation=cv2.INTER_AREA,
    )


# ---------------------------------------------------------------------
# Algorithm 2: Simple2D
# ---------------------------------------------------------------------

def generate_simple2d_class(
    number: int,
    out_dir: Path,
    count: int = 4000,
    image_size: int = 100,
    font_path: str | None = None,
) -> None:
    """
    Generate one class (00-99).

    Paper Algorithm 2:
      for each number 0-99
        for each background color
          generate 1,000 images
          augment digits
          concatenate digits
          resize to 100x100
        randomly sample 4,000 images/class
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = number // 10
    d2 = number % 10

    colors = list(DEFAULT_BG_COLORS.values())

    # The paper's pseudocode says 1,000 per color and then sample 4,000.
    # For a smaller custom count, distribute generation over colors.
    per_color = max(1, math.ceil(count / len(colors)))

    candidates: List[np.ndarray] = []

    for bg in colors:
        for _ in range(per_color):
            # Algorithm 1 independently generates each digit.
            left = render_single_digit(
                d1,
                image_size=image_size,
                font_path=font_path,
                bg_color=bg,
                text_color=random.choice(DEFAULT_TEXT_COLORS),
            )
            right = render_single_digit(
                d2,
                image_size=image_size,
                font_path=font_path,
                bg_color=bg,
                text_color=random.choice(DEFAULT_TEXT_COLORS),
            )

            left_np = np.asarray(left).copy()
            right_np = np.asarray(right).copy()

            # Apply paper augmentation to each digit before concatenation.
            left_np = apply_random_paper_augmentation(left_np)
            right_np = apply_random_paper_augmentation(right_np)

            result = concat_two_digits(left_np, right_np, image_size)
            candidates.append(result)

    random.shuffle(candidates)
    candidates = candidates[:count]

    label = f"{number:02d}"
    for idx, img in enumerate(candidates):
        cv2.imwrite(
            str(out_dir / f"{label}_{idx:05d}.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )


def generate_simple2d(
    root: Path,
    count_per_class: int = 4000,
    image_size: int = 100,
    font_path: str | None = None,
    numbers: Iterable[int] = range(100),
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for number in numbers:
        print(f"[Simple2D] Generating {number:02d} ...")
        generate_simple2d_class(
            number=number,
            out_dir=root / f"{number:02d}",
            count=count_per_class,
            image_size=image_size,
            font_path=font_path,
        )


# ---------------------------------------------------------------------
# Algorithm 3: Complex2D
# ---------------------------------------------------------------------

def list_images(root: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in extensions]


def alpha_overlay_number(
    background: np.ndarray,
    number_img: np.ndarray,
    x: int,
    y: int,
    opacity: float = 0.85,
) -> np.ndarray:
    """
    Overlay a synthetic number image on a real COCO image.

    Because Simple2D has a solid jersey-like background, we use the
    number image as the pasted jersey/number patch. Opacity controls
    how strongly the underlying real image affects the composite.
    """
    bg = background.copy()
    h, w = number_img.shape[:2]

    x2 = min(bg.shape[1], x + w)
    y2 = min(bg.shape[0], y + h)

    if x2 <= x or y2 <= y:
        return bg

    crop_w = x2 - x
    crop_h = y2 - y

    patch = number_img[:crop_h, :crop_w].astype(np.float32)
    base = bg[y:y2, x:x2].astype(np.float32)

    alpha = opacity
    blended = alpha * patch + (1.0 - alpha) * base
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
    """
    Paper Algorithm 3:
      for each number 0-99
        choose random COCO image
        choose random jersey-number image
        superimpose number at random position
        resize to 100x100
        continue until 4,000/class
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    number_dir = simple_root / f"{number:02d}"
    number_images = list_images(number_dir)

    if not number_images:
        raise FileNotFoundError(
            f"No Simple2D images found for class {number:02d}: {number_dir}"
        )

    if not coco_images:
        raise FileNotFoundError(
            "No COCO images found. Pass --coco-dir with extracted COCO images."
        )

    for idx in range(count):
        coco_path = random.choice(coco_images)
        num_path = random.choice(number_images)

        bg = cv2.imread(str(coco_path))
        if bg is None:
            continue

        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        # Resize/crop background while preserving variety.
        bg_h, bg_w = bg.shape[:2]
        if bg_h < image_size or bg_w < image_size:
            bg = cv2.resize(bg, (max(bg_w, image_size), max(bg_h, image_size)))

        # Random crop to square before final resize.
        bg_h, bg_w = bg.shape[:2]
        crop_size = min(bg_h, bg_w)
        x0 = random.randint(0, bg_w - crop_size)
        y0 = random.randint(0, bg_h - crop_size)
        bg = bg[y0:y0 + crop_size, x0:x0 + crop_size]

        bg = cv2.resize(bg, (image_size, image_size))

        number_img = cv2.imread(str(num_path))
        if number_img is None:
            continue

        number_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2RGB)
        number_img = cv2.resize(number_img, (image_size, image_size))

        # In the paper the number is randomly positioned. To make that
        # operation meaningful, use a smaller patch from the 100x100
        # Simple2D image before compositing.
        patch_scale = random.uniform(0.25, 0.65)
        patch_size = max(16, int(image_size * patch_scale))
        number_patch = cv2.resize(
            number_img,
            (patch_size, patch_size),
            interpolation=cv2.INTER_AREA,
        )

        x = random.randint(0, image_size - patch_size)
        y = random.randint(0, image_size - patch_size)

        opacity = random.uniform(0.70, 1.0)
        result = alpha_overlay_number(
            bg,
            number_patch,
            x,
            y,
            opacity=opacity,
        )

        result = cv2.resize(
            result,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

        cv2.imwrite(
            str(out_dir / f"{number:02d}_{idx:05d}.jpg"),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )


def generate_complex2d(
    root: Path,
    simple_root: Path,
    coco_dir: Path,
    count_per_class: int = 4000,
    image_size: int = 100,
    numbers: Iterable[int] = range(100),
) -> None:
    coco_images = list_images(coco_dir)

    if not coco_images:
        raise FileNotFoundError(
            f"No images found under COCO directory: {coco_dir}"
        )

    root.mkdir(parents=True, exist_ok=True)

    print(f"[Complex2D] COCO images found: {len(coco_images)}")

    for number in numbers:
        print(f"[Complex2D] Generating {number:02d} ...")
        generate_complex2d_class(
            number=number,
            simple_root=simple_root,
            coco_images=coco_images,
            out_dir=root / f"{number:02d}",
            count=count_per_class,
            image_size=image_size,
        )


# ---------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------

def write_manifest(dataset_root: Path, output_csv: Path) -> None:
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
            rows.append((str(img.resolve()), label, f"{label:02d}"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "number"])
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
    """
    Creates one contact sheet for quick visual inspection.
    """
    if numbers is None:
        numbers = [0, 1, 12, 23, 44, 66, 77, 99]

    tiles = []

    for n in numbers:
        paths = list_images(dataset_root / f"{n:02d}")
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
        draw.text((x + 5, y + 164), f"{n:02d}", fill="black")

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
        help="Number of images per jersey number. Use 20/50 first for testing.",
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
