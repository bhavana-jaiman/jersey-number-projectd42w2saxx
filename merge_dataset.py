import os
import shutil
from pathlib import Path

# ============================================================
# CHANGE THESE 3 PATHS
# ============================================================

dataset1 = Path("/path/to/dataset1")
dataset2 = Path("/path/to/dataset2")
output_dir = Path("/path/to/merged_dataset")

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Label extensions
LABEL_EXTENSIONS = {".txt", ".xml", ".json"}


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(dataset_path):
    print(f"\nChecking: {dataset_path}")

    if not dataset_path.exists():
        print("❌ Dataset does not exist!")
        return False

    if not dataset_path.is_dir():
        print("❌ Path is not a directory!")
        return False

    # Check for subdirectories
    subdirs = [x for x in dataset_path.iterdir() if x.is_dir()]

    if subdirs:
        print("❌ Dataset contains subfolders:")
        for folder in subdirs:
            print("   ", folder.name)
        return False

    files = [x for x in dataset_path.iterdir() if x.is_file()]

    images = [
        x for x in files
        if x.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        x for x in files
        if x.suffix.lower() in LABEL_EXTENSIONS
    ]

    print(f"Images found : {len(images)}")
    print(f"Labels found : {len(labels)}")

    if len(images) == 0:
        print("❌ No images found!")
        return False

    if len(labels) == 0:
        print("❌ No labels found!")
        return False

    # Check corresponding image-label pairs
    image_stems = {x.stem for x in images}
    label_stems = {x.stem for x in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        print(f"⚠️ Images without labels: {len(missing_labels)}")

    if missing_images:
        print(f"⚠️ Labels without images: {len(missing_images)}")

    print("✅ Dataset structure is valid.")

    return True


# ============================================================
# CHECK BOTH DATASETS
# ============================================================

print("=" * 60)
print("CHECKING DATASETS")
print("=" * 60)

valid1 = check_dataset(dataset1)
valid2 = check_dataset(dataset2)

if not valid1 or not valid2:
    print("\n❌ Dataset check failed. Nothing was merged.")
    exit()


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)


# ============================================================
# COPY FILES
# ============================================================

def copy_dataset(source_dir, prefix):
    copied = 0

    for file in source_dir.iterdir():

        if not file.is_file():
            continue

        destination = output_dir / file.name

        # If filename already exists, add prefix
        if destination.exists():
            destination = output_dir / f"{prefix}_{file.name}"

        shutil.copy2(file, destination)
        copied += 1

    return copied


count1 = copy_dataset(dataset1, "dataset1")
count2 = copy_dataset(dataset2, "dataset2")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print(f"Dataset 1 files copied : {count1}")
print(f"Dataset 2 files copied : {count2}")

total_files = len([
    x for x in output_dir.iterdir()
    if x.is_file()
])

print(f"Total files in merged dataset : {total_files}")
print(f"\nMerged dataset location:")
print(output_dir)

print("\n✅ Done!")
