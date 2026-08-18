import os
import shutil
import re

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying"

OUTPUT_DIR = "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying_separated"

IMAGE_DIR = os.path.join(SOURCE_DIR, "images")
LABEL_DIR = os.path.join(SOURCE_DIR, "labels")

# Output directories
REAL_IMAGE_DIR = os.path.join(OUTPUT_DIR, "real", "images")
REAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "real", "labels")

SYN_IMAGE_DIR = os.path.join(OUTPUT_DIR, "synthetic", "images")
SYN_LABEL_DIR = os.path.join(OUTPUT_DIR, "synthetic", "labels")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for folder in [
    REAL_IMAGE_DIR,
    REAL_LABEL_DIR,
    SYN_IMAGE_DIR,
    SYN_LABEL_DIR
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# CHECK SOURCE DIRECTORIES
# ============================================================

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(f"Images directory not found: {IMAGE_DIR}")

if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(f"Labels directory not found: {LABEL_DIR}")


# ============================================================
# SEPARATE DATASET
# ============================================================

real_count = 0
synthetic_count = 0
missing_labels = 0

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for filename in os.listdir(IMAGE_DIR):

    # Only process image files
    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)

    # Remove image extension
    base_name = os.path.splitext(filename)[0]

    # Corresponding label
    label_filename = base_name + ".txt"
    label_path = os.path.join(LABEL_DIR, label_filename)

    # Check whether label exists
    if not os.path.isfile(label_path):
        missing_labels += 1
        continue

    # ========================================================
    # SYNTHETIC DETECTION
    # Example:
    # ABC123.jpg      -> REAL
    # ABC123_0.jpg    -> SYNTHETIC
    # ABC123_1.jpg    -> SYNTHETIC
    # ========================================================

    if re.search(r"_\d+$", base_name):
        category = "synthetic"

        destination_image = os.path.join(
            SYN_IMAGE_DIR,
            filename
        )

        destination_label = os.path.join(
            SYN_LABEL_DIR,
            label_filename
        )

        synthetic_count += 1

    else:
        category = "real"

        destination_image = os.path.join(
            REAL_IMAGE_DIR,
            filename
        )

        destination_label = os.path.join(
            REAL_LABEL_DIR,
            label_filename
        )

        real_count += 1

    # ========================================================
    # COPY — ORIGINAL DATASET IS NOT MODIFIED
    # ========================================================

    shutil.copy2(image_path, destination_image)
    shutil.copy2(label_path, destination_label)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SEPARATION COMPLETED")
print("=" * 60)

print(f"Real image-label pairs       : {real_count}")
print(f"Synthetic image-label pairs  : {synthetic_count}")
print(f"Missing label files          : {missing_labels}")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Original dataset was NOT modified.")
print("=" * 60)
