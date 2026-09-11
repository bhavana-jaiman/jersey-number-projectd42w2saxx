import os
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

# Your current dataset folder
SOURCE_DIR = r"D:\path\to\complex2D_1_train"

# Folder where selected data will be created
SELECTED_DIR = r"D:\path\to\selected_0_1_2_3_4_5_8"

# Jersey numbers you want
TARGET_NUMBERS = {"0", "1", "2", "3", "4", "5", "8"}

# Random seed so the same split can be reproduced
RANDOM_SEED = 42


# ============================================================
# STEP 1: CREATE SELECTED DATASET
# ============================================================

os.makedirs(SELECTED_DIR, exist_ok=True)

selected_files = []

for filename in os.listdir(SOURCE_DIR):

    # We only process JPG images
    if not filename.lower().endswith(".jpg"):
        continue

    # Example: 0_00015.jpg -> number = 0
    parts = filename.split("_")

    if len(parts) < 2:
        continue

    jersey_number = parts[0]

    # Check whether this is one of the required numbers
    if jersey_number in TARGET_NUMBERS:

        image_path = os.path.join(SOURCE_DIR, filename)

        # Corresponding TXT file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(SOURCE_DIR, txt_filename)

        # Only copy if both image and label exist
        if os.path.exists(txt_path):

            shutil.copy2(image_path,
                         os.path.join(SELECTED_DIR, filename))

            shutil.copy2(txt_path,
                         os.path.join(SELECTED_DIR, txt_filename))

            selected_files.append(filename)

print(f"Selected {len(selected_files)} images.")
print(f"Data copied to: {SELECTED_DIR}")


# ============================================================
# STEP 2: CREATE TWO 50-50 FOLDERS
# ============================================================

FOLDER_1 = os.path.join(SELECTED_DIR, "split_1")
FOLDER_2 = os.path.join(SELECTED_DIR, "split_2")

os.makedirs(FOLDER_1, exist_ok=True)
os.makedirs(FOLDER_2, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# STEP 3: SPLIT EACH JERSEY NUMBER 50-50
# ============================================================

for number in sorted(TARGET_NUMBERS):

    # Get images belonging to this number
    number_images = [
        f for f in selected_files
        if f.startswith(number + "_")
    ]

    # Shuffle
    random.shuffle(number_images)

    # 50-50 split
    midpoint = len(number_images) // 2

    part1 = number_images[:midpoint]
    part2 = number_images[midpoint:]

    # Copy first half
    for img in part1:

        txt = os.path.splitext(img)[0] + ".txt"

        shutil.copy2(
            os.path.join(SELECTED_DIR, img),
            os.path.join(FOLDER_1, img)
        )

        shutil.copy2(
            os.path.join(SELECTED_DIR, txt),
            os.path.join(FOLDER_1, txt)
        )

    # Copy second half
    for img in part2:

        txt = os.path.splitext(img)[0] + ".txt"

        shutil.copy2(
            os.path.join(SELECTED_DIR, img),
            os.path.join(FOLDER_2, img)
        )

        shutil.copy2(
            os.path.join(SELECTED_DIR, txt),
            os.path.join(FOLDER_2, txt)
        )

    print(
        f"Number {number}: "
        f"{len(part1)} -> split_1, "
        f"{len(part2)} -> split_2"
    )


# ============================================================
# DONE
# ============================================================

print("\n====================================")
print("DONE!")
print("====================================")
print(f"Selected dataset : {SELECTED_DIR}")
print(f"Split 1          : {FOLDER_1}")
print(f"Split 2          : {FOLDER_2}")
