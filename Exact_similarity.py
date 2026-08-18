import os
import hashlib
import csv
from collections import defaultdict


# ============================================================
# DATASET PATH
# ============================================================

IMAGE_DIR = (
    "/home/eng_bhavana/workspace_bhavana/"
    "jersey_number_recognition_20260626/"
    "datasets/training_dataset_Ying/images"
)

# Output file
OUTPUT_CSV = (
    "/home/eng_bhavana/workspace_bhavana/"
    "jersey_number_recognition_20260626/"
    "datasets/exact_duplicates.csv"
)


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# SHA-256 FUNCTION
# ============================================================

def calculate_sha256(filepath):

    sha256 = hashlib.sha256()

    try:

        with open(filepath, "rb") as f:

            # Read in chunks so large images don't
            # consume a lot of memory.

            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception as e:

        print(
            f"ERROR reading {filepath}: {e}"
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EXACT IMAGE DUPLICATE CHECK")
    print("=" * 70)

    print("\nDataset:")
    print(IMAGE_DIR)

    print(
        "\nMethod: SHA-256"
    )

    print(
        "Jersey numbers are INCLUDED."
    )

    print(
        "No masking, resizing, or image modification is performed."
    )


    # ========================================================
    # CHECK DIRECTORY
    # ========================================================

    if not os.path.isdir(IMAGE_DIR):

        print(
            "\nERROR: Image directory does not exist:"
        )

        print(IMAGE_DIR)

        return


    # ========================================================
    # FIND IMAGES
    # ========================================================

    print("\nScanning images...")

    image_files = []

    for root, dirs, files in os.walk(
        IMAGE_DIR
    ):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(
                root,
                filename
            )

            image_files.append(
                filepath
            )


    total_images = len(
        image_files
    )

    print(
        f"\nTotal images found: "
        f"{total_images:,}"
    )


    # ========================================================
    # CALCULATE SHA-256
    # ========================================================

    print("\nCalculating SHA-256 hashes...")

    hash_groups = defaultdict(list)

    failed = 0

    for index, filepath in enumerate(
        image_files,
        start=1
    ):

        if index % 1000 == 0:

            print(
                f"Processed "
                f"{index:,}/{total_images:,}"
            )

        file_hash = calculate_sha256(
            filepath
        )

        if file_hash is None:

            failed += 1

            continue

        hash_groups[
            file_hash
        ].append(filepath)


    # ========================================================
    # FIND EXACT DUPLICATES
    # ========================================================

    duplicate_groups = []

    for file_hash, files in hash_groups.items():

        if len(files) > 1:

            duplicate_groups.append(
                (
                    file_hash,
                    files
                )
            )


    duplicate_groups.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )


    # ========================================================
    # CALCULATE STATISTICS
    # ========================================================

    duplicate_files = sum(
        len(files)
        for _, files in duplicate_groups
    )

    duplicate_extra_copies = sum(
        len(files) - 1
        for _, files in duplicate_groups
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    print("\nSaving duplicate report...")

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "duplicate_group",
            "sha256",
            "duplicate_count",
            "image_path"
        ])

        group_number = 1

        for file_hash, files in duplicate_groups:

            for filepath in files:

                writer.writerow([
                    group_number,
                    file_hash,
                    len(files),
                    filepath
                ])

            group_number += 1


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXACT DUPLICATE ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"Total images             : "
        f"{total_images:,}"
    )

    print(
        f"Successfully hashed      : "
        f"{total_images - failed:,}"
    )

    print(
        f"Failed                   : "
        f"{failed:,}"
    )

    print(
        f"Exact duplicate groups   : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images in duplicate groups: "
        f"{duplicate_files:,}"
    )

    print(
        f"Extra duplicate copies   : "
        f"{duplicate_extra_copies:,}"
    )

    print()
    print(
        "Report saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Only files with identical SHA-256 hashes "
        "are considered exact duplicates."
    )

    print(
        "The original dataset was NOT modified."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
