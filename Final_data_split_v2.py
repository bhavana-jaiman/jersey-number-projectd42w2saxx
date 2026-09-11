import os
import shutil
import random

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/demo3/complex2D_1_train"

OUTPUT_DIR = "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/demo3/final_splited_data"

# Numbers required
TARGET_NUMBERS = {"0", "1", "2", "3", "4", "5", "8"}

RANDOM_SEED = 42


# ============================================================
# CREATE TWO SUBFOLDERS
# ============================================================

PART1_DIR = os.path.join(OUTPUT_DIR, "part_1")
PART2_DIR = os.path.join(OUTPUT_DIR, "part_2")

os.makedirs(PART1_DIR, exist_ok=True)
os.makedirs(PART2_DIR, exist_ok=True)


# ============================================================
# FIND IMAGE + LABEL PAIRS
# ============================================================

data = []

for filename in os.listdir(SOURCE_DIR):

    # Only JPG images
    if not filename.lower().endswith(".jpg"):
        continue

    # Example:
    # 0_00015.jpg
    # 1_00025.jpg
    # 8_00120.jpg

    parts = filename.split("_")

    if len(parts) < 2:
        continue

    jersey_number = parts[0]

    # Only numbers 0,1,2,3,4,5,8
    if jersey_number not in TARGET_NUMBERS:
        continue

    # Matching label
    label_name = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(SOURCE_DIR, label_name)

    # Make sure label exists
    if not os.path.exists(label_path):
        print("WARNING: Label missing:", label_name)
        continue

    # Store image + label pair
    data.append((filename, label_name))


print("Total selected image-label pairs:", len(data))


# ============================================================
# SHUFFLE DATA
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(data)


# ============================================================
# 50-50 SPLIT
# ============================================================

mid = len(data) // 2

part1 = data[:mid]
part2 = data[mid:]


# ============================================================
# COPY PART 1
# ============================================================

for image_name, label_name in part1:

    shutil.copy2(
        os.path.join(SOURCE_DIR, image_name),
        os.path.join(PART1_DIR, image_name)
    )

    shutil.copy2(
        os.path.join(SOURCE_DIR, label_name),
        os.path.join(PART1_DIR, label_name)
    )


# ============================================================
# COPY PART 2
# ============================================================

for image_name, label_name in part2:

    shutil.copy2(
        os.path.join(SOURCE_DIR, image_name),
        os.path.join(PART2_DIR, image_name)
    )

    shutil.copy2(
        os.path.join(SOURCE_DIR, label_name),
        os.path.join(PART2_DIR, label_name)
    )


# ============================================================
# RESULTS
# ============================================================

print("\n====================================")
print("DONE")
print("====================================")

print("Total pairs :", len(data))
print("Part 1      :", len(part1))
print("Part 2      :", len(part2))

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nPart 1:")
print(PART1_DIR)

print("\nPart 2:")
print(PART2_DIR)
