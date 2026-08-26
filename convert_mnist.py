from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from pathlib import Path


# =========================
# Paths
# =========================
MNIST_ROOT = "./data"
OUTPUT_ROOT = Path("./datasets/MNIST_Ying")

TRAIN_OUTPUT = OUTPUT_ROOT / "train"
TEST_OUTPUT = OUTPUT_ROOT / "test"


# =========================
# Convert one split
# =========================
def convert_split(dataset, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        image, label = dataset[idx]

        # Convert PIL image to numpy
        img = np.array(image)

        # Find foreground pixels
        ys, xs = np.where(img > 0)

        # Safety check
        if len(xs) == 0 or len(ys) == 0:
            # If image is completely black, use full image
            x_min, x_max = 0, img.shape[1] - 1
            y_min, y_max = 0, img.shape[0] - 1
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

        # Image dimensions
        width = img.shape[1]
        height = img.shape[0]

        # Bounding box center
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height

        # Bounding box width/height
        bbox_width = (x_max - x_min + 1) / width
        bbox_height = (y_max - y_min + 1) / height

        # File name
        filename = f"{idx:06d}"

        # Save image as RGB JPG
        image_rgb = image.convert("RGB")
        image_rgb.save(
            output_dir / f"{filename}.jpg",
            quality=95
        )

        # Save YOLO label
        with open(output_dir / f"{filename}.txt", "w") as f:
            f.write(
                f"{label} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bbox_width:.6f} "
                f"{bbox_height:.6f}\n"
            )

        if (idx + 1) % 10000 == 0:
            print(f"Converted {idx + 1}/{len(dataset)}")


# =========================
# Load MNIST
# =========================
print("Loading MNIST...")

train_dataset = MNIST(
    root=MNIST_ROOT,
    train=True,
    download=False
)

test_dataset = MNIST(
    root=MNIST_ROOT,
    train=False,
    download=False
)


# =========================
# Convert
# =========================
print("\nConverting training set...")
convert_split(train_dataset, TRAIN_OUTPUT)

print("\nConverting test set...")
convert_split(test_dataset, TEST_OUTPUT)

print("\nDone!")
print(f"Training dataset: {TRAIN_OUTPUT}")
print(f"Test dataset:     {TEST_OUTPUT}")
