import os
import shutil
from pathlib import Path

# ============================================================
# CHANGE ONLY THESE 3 PATHS
# ============================================================

dataset1 = Path("/path/to/dataset1")
dataset2 = Path("/path/to/dataset2")

merged_dataset = Path("/path/to/merged_dataset")


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt", ".xml", ".json"}


# ============================================================
# CHECK DATASET STRUCTURE
# ============================================================

def check_dataset(dataset_path):

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    if not dataset_path.is_dir():
        raise ValueError(f"Not a directory: {dataset_path}")

    # Check for subdirectories
    subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]

    if subdirs:
        print(f"\n❌ {dataset_path} contains subfolders:")
        for folder in subdirs:
            print(f"   {folder.name}")
        return False

    files = [p for p in dataset_path.iterdir() if p.is_file()]

    images = [
        p for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = [
        p for p in files
        if p.suffix.lower() in LABEL_EXTENSIONS
    ]

    print(f"\nDataset: {dataset_path}")
    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    return True


# ============================================================
# GET FILE PAIRS
# ============================================================

def get_pairs(dataset_path):

    files = [p for p in dataset_path.iterdir() if p.is_file()]

    images = {
        p.stem: p
        for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        p.stem: p
        for p in files
        if p.suffix.lower() in LABEL_EXTENSIONS
    }

    common = set(images.keys()) & set(labels.keys())

    missing_labels = set(images.keys()) - set(labels.keys())
    missing_images = set(labels.keys()) - set(images.keys())

    if missing_labels:
        print("\n⚠️ Images without labels:")
        for name in sorted(missing_labels):
            print(f"   {name}")

    if missing_images:
        print("\n⚠️ Labels without images:")
        for name in sorted(missing_images):
            print(f"   {name}")

    pairs = []

    for stem in common:
        pairs.append((images[stem], labels[stem]))

    return pairs


# ============================================================
# COPY FILE SAFELY
# ============================================================

def copy_file_safe(source, destination):

    destination.mkdir(parents=True, exist_ok=True)

    target = destination / source.name

    # If filename already exists, create a new name
    if target.exists():

        counter = 1

        while True:

            new_name = f"{source.stem}_dataset2_{counter}{source.suffix}"
            target = destination / new_name

            if not target.exists():
                break

            counter += 1

    shutil.copy2(source, target)

    return target


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("CHECKING DATASET STRUCTURE")
print("=" * 60)

valid1 = check_dataset(dataset1)
valid2 = check_dataset(dataset2)

if not valid1 or not valid2:
    print("\n❌ Dataset structure is not compatible.")
    print("No files were copied.")
    exit()


print("\n" + "=" * 60)
print("CHECKING IMAGE-LABEL PAIRS")
print("=" * 60)

pairs1 = get_pairs(dataset1)
pairs2 = get_pairs(dataset2)

print(f"\nDataset 1 valid pairs: {len(pairs1)}")
print(f"Dataset 2 valid pairs: {len(pairs2)}")


# ============================================================
# CREATE NEW MERGED DIRECTORY
# ============================================================

if merged_dataset.exists():
    print(f"\n⚠️ Merged directory already exists:")
    print(f"   {merged_dataset}")

    response = input("Do you want to use it? (yes/no): ")

    if response.lower() != "yes":
        print("❌ Stopped. Original datasets were not changed.")
        exit()

else:
    merged_dataset.mkdir(parents=True)


# ============================================================
# COPY DATASET 1
# ============================================================

print("\n" + "=" * 60)
print("COPYING DATASET 1")
print("=" * 60)

count1 = 0

for image, label in pairs1:

    copy_file_safe(image, merged_dataset)
    copy_file_safe(label, merged_dataset)

    count1 += 1


# ============================================================
# COPY DATASET 2
# ============================================================

print("\n" + "=" * 60)
print("COPYING DATASET 2")
print("=" * 60)

count2 = 0

for image, label in pairs2:

    # Copy image
    new_image = copy_file_safe(image, merged_dataset)

    # Determine label name
    # If image was renamed because of duplicate,
    # rename its label with the same new stem.
    expected_label_name = new_image.stem + label.suffix

    label_destination = merged_dataset / expected_label_name

    if label_destination.exists():

        counter = 1

        while True:

            new_label_name = f"{label.stem}_dataset2_{counter}{label.suffix}"
            label_destination = merged_dataset / new_label_name

            if not label_destination.exists():
                break

            counter += 1

    shutil.copy2(label, label_destination)

    count2 += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

merged_images = [
    p for p in merged_dataset.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

merged_labels = [
    p for p in merged_dataset.iterdir()
    if p.is_file() and p.suffix.lower() in LABEL_EXTENSIONS
]

print("\n" + "=" * 60)
print("MERGING COMPLETE")
print("=" * 60)

print(f"Dataset 1 pairs copied : {count1}")
print(f"Dataset 2 pairs copied : {count2}")

print(f"\nMerged dataset:")
print(f"   {merged_dataset}")

print(f"\nTotal images : {len(merged_images)}")
print(f"Total labels : {len(merged_labels)}")

print("\n✅ Original Dataset 1 was NOT changed.")
print("✅ Original Dataset 2 was NOT changed.")
print("✅ Only the new merged directory was created/modified.")
