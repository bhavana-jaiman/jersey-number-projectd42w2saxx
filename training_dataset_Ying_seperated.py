import os
import shutil
import re

# ============================================================
# PATHS - CHANGE THESE
# ============================================================

SOURCE_DIR = r"C:\path\to\training_dataset_Ying"

OUTPUT_DIR = r"C:\path\to\training_dataset_Ying_separated"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

real_images = os.path.join(OUTPUT_DIR, "real", "images")
real_labels = os.path.join(OUTPUT_DIR, "real", "labels")

synthetic_images = os.path.join(OUTPUT_DIR, "synthetic", "images")
synthetic_labels = os.path.join(OUTPUT_DIR, "synthetic", "labels")

for folder in [
    real_images,
    real_labels,
    synthetic_images,
    synthetic_labels
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# COUNTERS
# ============================================================

real_count = 0
synthetic_count = 0

missing_labels = []
other_files = []


# ============================================================
# PROCESS DATASET
# ============================================================

for filename in os.listdir(SOURCE_DIR):

    file_path = os.path.join(SOURCE_DIR, filename)

    # Ignore directories
    if not os.path.isfile(file_path):
        continue

    name, extension = os.path.splitext(filename)

    # Only process images
    if extension.lower() not in IMAGE_EXTENSIONS:
        continue

    # --------------------------------------------------------
    # Find corresponding label
    # --------------------------------------------------------

    label_filename = name + ".txt"
    label_path = os.path.join(SOURCE_DIR, label_filename)

    if not os.path.exists(label_path):
        missing_labels.append(filename)
        continue

    # --------------------------------------------------------
    # Determine REAL vs SYNTHETIC
    #
    # Real:
    #       ABC123.jpg
    #
    # Synthetic:
    #       ABC123_0.jpg
    #       ABC123_1.jpg
    #       ABC123_2.jpg
    # --------------------------------------------------------

    if re.search(r"_\d+$", name):

        # ==========================
        # SYNTHETIC
        # ==========================

        shutil.copy2(
            file_path,
            os.path.join(synthetic_images, filename)
        )

        shutil.copy2(
            label_path,
            os.path.join(synthetic_labels, label_filename)
        )

        synthetic_count += 1

    else:

        # ==========================
        # REAL
        # ==========================

        shutil.copy2(
            file_path,
            os.path.join(real_images, filename)
        )

        shutil.copy2(
            label_path,
            os.path.join(real_labels, label_filename)
        )

        real_count += 1


# ============================================================
# REPORT
# ============================================================

print("\n========================================")
print("DATASET SEPARATION COMPLETED")
print("========================================")

print(f"Real image-label pairs      : {real_count}")
print(f"Synthetic image-label pairs : {synthetic_count}")

print(f"\nMissing label files         : {len(missing_labels)}")

if missing_labels:
    print("\nImages with missing labels:")
    for file in missing_labels:
        print("  ", file)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("========================================")
