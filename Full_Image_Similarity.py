import os
import re
import csv
import hashlib
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import imagehash


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = (
    "/home/eng_bhavana/workspace_bhavana/"
    "jersey_number_recognition_20260626/"
    "datasets/training_dataset_Ying"
)

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

OUTPUT_DIR = (
    "/home/eng_bhavana/workspace_bhavana/"
    "jersey_number_recognition_20260626/"
    "datasets/similarity_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

# pHash distance used to identify candidate near-duplicates.
#
# IMPORTANT:
# This is NOT a final "duplicate" definition.
#
# Smaller distance = more similar.
#
# 0 = identical perceptual hash
# 4 = extremely similar
# 8 = strong similarity candidate
#
PHASH_DISTANCE_THRESHOLD = 8

# Number of strongest pairs to save in detailed report
MAX_DETAILED_PAIRS = 100000

# ORB configuration
ORB_FEATURES = 1500


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def get_image_type(filename):

    """
    Determine whether image is REAL or SYNTHETIC.

    Example:

        ABC123.jpg       -> real
        ABC123_0.jpg     -> synthetic
        ABC123_1.jpg     -> synthetic
        ABC123_15.jpg    -> synthetic
    """

    stem = os.path.splitext(filename)[0]

    if re.search(r"_\d+$", stem):
        return "synthetic"

    return "real"


def get_base_name(filename):

    """
    Convert:

        ABC123.jpg
        ABC123_0.jpg
        ABC123_15.jpg

    into:

        ABC123
    """

    stem = os.path.splitext(filename)[0]

    return re.sub(r"_\d+$", "", stem)


# ============================================================
# MD5
# ============================================================

def calculate_md5(filepath):

    try:

        md5 = hashlib.md5()

        with open(filepath, "rb") as f:

            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b""
            ):

                md5.update(chunk)

        return md5.hexdigest()

    except Exception:

        return None


# ============================================================
# pHASH
# ============================================================

def calculate_phash(filepath):

    try:

        with Image.open(filepath) as img:

            img = img.convert("RGB")

            return imagehash.phash(img)

    except Exception:

        return None


def phash_distance(hash1, hash2):

    return hash1 - hash2


def phash_similarity(hash1, hash2):

    distance = phash_distance(
        hash1,
        hash2
    )

    # pHash is normally 64 bits.
    #
    # This is a DERIVED similarity score.
    # It is NOT a probability.

    max_distance = 64

    score = (
        1 -
        distance / max_distance
    ) * 100

    return max(0.0, score)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(filepath):

    try:

        image = cv2.imread(filepath)

        return image

    except Exception:

        return None


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(image1, image2):

    try:

        # Resize both images to same size
        image1 = cv2.resize(
            image1,
            (256, 256)
        )

        image2 = cv2.resize(
            image2,
            (256, 256)
        )

        gray1 = cv2.cvtColor(
            image1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            image2,
            cv2.COLOR_BGR2GRAY
        )

        gray1 = gray1.astype(
            np.float64
        )

        gray2 = gray2.astype(
            np.float64
        )

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(
            gray1,
            (11, 11),
            1.5
        )

        mu2 = cv2.GaussianBlur(
            gray2,
            (11, 11),
            1.5
        )

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            cv2.GaussianBlur(
                gray1 * gray1,
                (11, 11),
                1.5
            )
            - mu1_sq
        )

        sigma2_sq = (
            cv2.GaussianBlur(
                gray2 * gray2,
                (11, 11),
                1.5
            )
            - mu2_sq
        )

        sigma12 = (
            cv2.GaussianBlur(
                gray1 * gray2,
                (11, 11),
                1.5
            )
            - mu1_mu2
        )

        numerator = (
            (2 * mu1_mu2 + C1) *
            (2 * sigma12 + C2)
        )

        denominator = (
            (mu1_sq + mu2_sq + C1) *
            (sigma1_sq + sigma2_sq + C2)
        )

        ssim_map = numerator / denominator

        score = np.mean(ssim_map)

        # Convert to percentage
        score = score * 100

        return float(
            np.clip(score, 0, 100)
        )

    except Exception:

        return None


# ============================================================
# ORB
# ============================================================

def calculate_orb_similarity(
    image1,
    image2
):

    try:

        image1 = cv2.resize(
            image1,
            (640, 640)
        )

        image2 = cv2.resize(
            image2,
            (640, 640)
        )

        gray1 = cv2.cvtColor(
            image1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            image2,
            cv2.COLOR_BGR2GRAY
        )

        orb = cv2.ORB_create(
            nfeatures=ORB_FEATURES
        )

        keypoints1, descriptors1 = (
            orb.detectAndCompute(
                gray1,
                None
            )
        )

        keypoints2, descriptors2 = (
            orb.detectAndCompute(
                gray2,
                None
            )
        )

        if descriptors1 is None:
            return 0.0

        if descriptors2 is None:
            return 0.0

        if len(descriptors1) < 2:
            return 0.0

        if len(descriptors2) < 2:
            return 0.0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        matches = matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            # Lowe ratio test
            if m.distance < 0.75 * n.distance:

                good_matches.append(m)

        denominator = min(
            len(keypoints1),
            len(keypoints2)
        )

        if denominator == 0:
            return 0.0

        score = (
            len(good_matches) /
            denominator
        ) * 100

        return min(score, 100.0)

    except Exception:

        return 0.0


# ============================================================
# GET IMAGES
# ============================================================

def get_images():

    images = []

    for filename in os.listdir(IMAGE_DIR):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        filepath = os.path.join(
            IMAGE_DIR,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        image_type = get_image_type(
            filename
        )

        base_name = get_base_name(
            filename
        )

        images.append({
            "filename": filename,
            "path": filepath,
            "type": image_type,
            "base": base_name
        })

    return images


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("FULL IMAGE SIMILARITY ANALYSIS")
    print("=" * 75)

    print("\nDataset:")
    print(DATASET_DIR)

    print("\nImages:")
    print(IMAGE_DIR)

    print("\nIMPORTANT:")
    print(
        "Jersey numbers ARE INCLUDED in all image comparisons."
    )

    print(
        "No number masking or number removal is performed."
    )


    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    if not os.path.isdir(IMAGE_DIR):

        print("\nERROR: Image directory does not exist.")

        print(IMAGE_DIR)

        return


    # ========================================================
    # GET IMAGES
    # ========================================================

    print("\nScanning dataset...")

    images = get_images()

    real_images = [
        x for x in images
        if x["type"] == "real"
    ]

    synthetic_images = [
        x for x in images
        if x["type"] == "synthetic"
    ]

    print(
        f"\nTotal images      : {len(images):,}"
    )

    print(
        f"Real images       : {len(real_images):,}"
    )

    print(
        f"Synthetic images  : {len(synthetic_images):,}"
    )


    # ========================================================
    # MD5 HASHES
    # ========================================================

    print("\nCalculating MD5 hashes...")

    md5_groups = defaultdict(list)

    failed_md5 = 0

    for index, item in enumerate(
        images,
        start=1
    ):

        if index % 5000 == 0:

            print(
                f"MD5: {index:,}/{len(images):,}"
            )

        md5 = calculate_md5(
            item["path"]
        )

        if md5 is None:

            failed_md5 += 1

            continue

        item["md5"] = md5

        md5_groups[md5].append(item)


    # ========================================================
    # EXACT DUPLICATES
    # ========================================================

    exact_duplicate_groups = []

    for group in md5_groups.values():

        if len(group) > 1:

            exact_duplicate_groups.append(
                group
            )


    exact_duplicate_images = sum(
        len(group) - 1
        for group in exact_duplicate_groups
    )


    print("\nExact duplicate groups:")
    print(
        len(exact_duplicate_groups)
    )

    print(
        "Duplicate images:",
        exact_duplicate_images
    )


    # ========================================================
    # SAVE EXACT DUPLICATES
    # ========================================================

    exact_csv = os.path.join(
        OUTPUT_DIR,
        "exact_duplicates.csv"
    )

    with open(
        exact_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "md5",
            "image_type",
            "filename"
        ])

        for group in exact_duplicate_groups:

            md5 = group[0]["md5"]

            for item in group:

                writer.writerow([
                    md5,
                    item["type"],
                    item["filename"]
                ])


    # ========================================================
    # CALCULATE pHASH
    # ========================================================

    print("\nCalculating pHash...")

    phash_buckets = defaultdict(list)

    failed_phash = 0

    for index, item in enumerate(
        images,
        start=1
    ):

        if index % 5000 == 0:

            print(
                f"pHash: {index:,}/{len(images):,}"
            )

        phash = calculate_phash(
            item["path"]
        )

        if phash is None:

            failed_phash += 1

            continue

        item["phash"] = phash

        # First four hexadecimal
        # characters are used as
        # a candidate bucket.

        bucket = str(phash)[:4]

        phash_buckets[bucket].append(
            item
        )


    # ========================================================
    # NEAR DUPLICATES
    # ========================================================

    print("\nFinding pHash candidate pairs...")

    near_duplicate_results = []

    bucket_count = len(
        phash_buckets
    )

    for bucket_index, group in enumerate(
        phash_buckets.values(),
        start=1
    ):

        if bucket_index % 100 == 0:

            print(
                f"Buckets: "
                f"{bucket_index:,}/{bucket_count:,}"
            )

        if len(group) < 2:
            continue

        for i in range(len(group)):

            for j in range(
                i + 1,
                len(group)
            ):

                item1 = group[i]
                item2 = group[j]

                distance = (
                    item1["phash"] -
                    item2["phash"]
                )

                if distance > PHASH_DISTANCE_THRESHOLD:
                    continue

                similarity = (
                    1 -
                    distance / 64
                ) * 100

                near_duplicate_results.append({
                    "image1":
                        item1["filename"],

                    "image2":
                        item2["filename"],

                    "type1":
                        item1["type"],

                    "type2":
                        item2["type"],

                    "phash_distance":
                        distance,

                    "phash_similarity":
                        round(
                            similarity,
                            2
                        )
                })


    # ========================================================
    # SORT NEAR DUPLICATES
    # ========================================================

    near_duplicate_results.sort(
        key=lambda x:
        x["phash_similarity"],
        reverse=True
    )


    # ========================================================
    # SAVE NEAR DUPLICATES
    # ========================================================

    near_csv = os.path.join(
        OUTPUT_DIR,
        "near_duplicates.csv"
    )

    with open(
        near_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image1",
                "image2",
                "type1",
                "type2",
                "phash_distance",
                "phash_similarity"
            ]
        )

        writer.writeheader()

        writer.writerows(
            near_duplicate_results[
                :MAX_DETAILED_PAIRS
            ]
        )


    print(
        "\nNear-duplicate candidates:",
        len(near_duplicate_results)
    )


    # ========================================================
    # REAL -> SYNTHETIC SOURCE MATCHING
    # ========================================================

    print()
    print("=" * 75)
    print("REAL ↔ SYNTHETIC SOURCE ANALYSIS")
    print("=" * 75)

    real_by_base = {}

    for item in real_images:

        real_by_base[
            item["base"]
        ] = item


    source_results = []

    matched_source_pairs = 0

    for index, synthetic in enumerate(
        synthetic_images,
        start=1
    ):

        if index % 1000 == 0:

            print(
                f"Source pairs: "
                f"{index:,}/{len(synthetic_images):,}"
            )

        real = real_by_base.get(
            synthetic["base"]
        )

        if real is None:
            continue

        matched_source_pairs += 1

        # ----------------------------------------------------
        # pHASH
        # ----------------------------------------------------

        if (
            "phash" in real
            and
            "phash" in synthetic
        ):

            distance = (
                real["phash"] -
                synthetic["phash"]
            )

            phash_score = (
                1 -
                distance / 64
            ) * 100

        else:

            distance = None
            phash_score = None


        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        real_img = load_image(
            real["path"]
        )

        synthetic_img = load_image(
            synthetic["path"]
        )

        if (
            real_img is not None
            and
            synthetic_img is not None
        ):

            ssim_score = calculate_ssim(
                real_img,
                synthetic_img
            )

            orb_score = (
                calculate_orb_similarity(
                    real_img,
                    synthetic_img
                )
            )

        else:

            ssim_score = None
            orb_score = None


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        source_results.append({

            "real_image":
                real["filename"],

            "synthetic_image":
                synthetic["filename"],

            "phash_distance":
                distance,

            "phash_similarity":
                round(
                    phash_score,
                    2
                )
                if phash_score is not None
                else None,

            "ssim_similarity":
                round(
                    ssim_score,
                    2
                )
                if ssim_score is not None
                else None,

            "orb_similarity":
                round(
                    orb_score,
                    2
                )
                if orb_score is not None
                else None
        })


    # ========================================================
    # SORT SOURCE RESULTS
    # ========================================================

    source_results.sort(
        key=lambda x:
        (
            x["phash_similarity"]
            if x["phash_similarity"]
            is not None
            else 0
        ),
        reverse=True
    )


    # ========================================================
    # SAVE SOURCE RESULTS
    # ========================================================

    source_csv = os.path.join(
        OUTPUT_DIR,
        "real_vs_synthetic_source.csv"
    )

    with open(
        source_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "real_image",
                "synthetic_image",
                "phash_distance",
                "phash_similarity",
                "ssim_similarity",
                "orb_similarity"
            ]
        )

        writer.writeheader()

        writer.writerows(
            source_results
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL SUMMARY")
    print("=" * 75)

    print(
        f"Total images          : "
        f"{len(images):,}"
    )

    print(
        f"Real images           : "
        f"{len(real_images):,}"
    )

    print(
        f"Synthetic images      : "
        f"{len(synthetic_images):,}"
    )

    print(
        f"Exact duplicate groups: "
        f"{len(exact_duplicate_groups):,}"
    )

    print(
        f"Exact duplicate images: "
        f"{exact_duplicate_images:,}"
    )

    print(
        f"Near-duplicate pairs  : "
        f"{len(near_duplicate_results):,}"
    )

    print(
        f"Real→Synthetic pairs  : "
        f"{matched_source_pairs:,}"
    )

    print()
    print("Reports:")
    print(
        exact_csv
    )

    print(
        near_csv
    )

    print(
        source_csv
    )

    print()
    print(
        "IMPORTANT: Jersey numbers were INCLUDED "
        "in every similarity calculation."
    )

    print(
        "No image was modified."
    )

    print(
        "No image was moved."
    )

    print(
        "No image was deleted."
    )

    print("=" * 75)


if __name__ == "__main__":

    main()
