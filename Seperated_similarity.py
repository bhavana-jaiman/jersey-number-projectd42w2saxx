import os
import re
import csv
import hashlib
from collections import defaultdict

from PIL import Image
import imagehash


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/datasets/training_dataset_Ying"

IMAGE_DIR = os.path.join(DATASET_DIR, "images")

OUTPUT_DIR = os.path.join(
    "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626",
    "similarity_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# pHash Hamming distance threshold
# Lower = more similar
PHASH_THRESHOLD = 8

# Maximum number of examples written to detailed CSV
MAX_EXAMPLES = 100000


# ============================================================
# IMAGE TYPE DETECTION
# ============================================================

# Example:
# ABC123.jpg       -> REAL
# ABC123_0.jpg     -> SYNTHETIC
# ABC123_1.jpg     -> SYNTHETIC
#
# The synthetic image must end with "_number".

SYNTHETIC_PATTERN = re.compile(r"_(\d+)$")


def get_image_type(filename):

    stem = os.path.splitext(filename)[0]

    if SYNTHETIC_PATTERN.search(stem):
        return "synthetic"

    return "real"


def get_base_name(filename):

    stem = os.path.splitext(filename)[0]

    # ABC123_0 -> ABC123
    # ABC123_10 -> ABC123

    return re.sub(r"_\d+$", "", stem)


# ============================================================
# HASH FUNCTIONS
# ============================================================

def calculate_md5(filepath):

    md5 = hashlib.md5()

    try:
        with open(filepath, "rb") as f:

            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return None


def calculate_phash(filepath):

    try:

        with Image.open(filepath) as img:

            img = img.convert("RGB")

            return str(imagehash.phash(img))

    except Exception:

        return None


# ============================================================
# PHASH HAMMING DISTANCE
# ============================================================

def hamming_distance(hash1, hash2):

    try:

        return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)

    except Exception:

        return 999


def similarity_percentage(distance):

    # pHash is normally 64 bits

    similarity = (1 - distance / 64) * 100

    return max(0, similarity)


# ============================================================
# GET ALL IMAGES
# ============================================================

print("=" * 70)
print("IMAGE SIMILARITY ANALYSIS")
print("=" * 70)

print("\nScanning images...")

images = []

for root, dirs, files in os.walk(IMAGE_DIR):

    for filename in files:

        ext = os.path.splitext(filename)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        filepath = os.path.join(root, filename)

        image_type = get_image_type(filename)

        base_name = get_base_name(filename)

        images.append({
            "path": filepath,
            "filename": filename,
            "type": image_type,
            "base": base_name
        })


print(f"\nTotal images found: {len(images):,}")


# ============================================================
# COUNTS
# ============================================================

real_images = [
    x for x in images
    if x["type"] == "real"
]

synthetic_images = [
    x for x in images
    if x["type"] == "synthetic"
]

print(f"Real images      : {len(real_images):,}")
print(f"Synthetic images : {len(synthetic_images):,}")


# ============================================================
# CALCULATE HASHES
# ============================================================

print("\nCalculating hashes...")

md5_groups = defaultdict(list)

phash_records = []

failed = 0

for i, item in enumerate(images, 1):

    if i % 5000 == 0:

        print(
            f"Processed {i:,}/{len(images):,}"
        )

    md5 = calculate_md5(item["path"])

    phash = calculate_phash(item["path"])

    if md5 is None or phash is None:

        failed += 1

        continue

    item["md5"] = md5
    item["phash"] = phash

    md5_groups[md5].append(item)

    phash_records.append(item)


print("\nHash calculation completed.")

print(f"Failed images: {failed:,}")


# ============================================================
# A) EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("A) EXACT DUPLICATES")
print("=" * 70)

exact_duplicate_groups = [
    group
    for group in md5_groups.values()
    if len(group) > 1
]

exact_duplicate_count = sum(
    len(group) - 1
    for group in exact_duplicate_groups
)

print(
    f"Exact duplicate groups : {len(exact_duplicate_groups):,}"
)

print(
    f"Duplicate images        : {exact_duplicate_count:,}"
)


# ============================================================
# CREATE PHASH BUCKETS
# ============================================================

print("\nCreating pHash buckets...")

# First 4 hexadecimal characters = 16 bits
# Similar images are likely to share this prefix.
#
# This avoids all-vs-all comparison.

phash_buckets = defaultdict(list)

for item in phash_records:

    key = item["phash"][:4]

    phash_buckets[key].append(item)


# ============================================================
# SIMILARITY ANALYSIS FUNCTION
# ============================================================

def analyze_pair_group(
    group1,
    group2,
    analysis_name,
    same_group=False
):

    print("\n" + "-" * 70)
    print(analysis_name)
    print("-" * 70)

    total_pairs = 0
    similar_pairs = 0

    similarity_values = []

    examples = []

    # --------------------------------------------------------
    # SAME GROUP
    # --------------------------------------------------------

    if same_group:

        for i in range(len(group1)):

            item1 = group1[i]

            for j in range(i + 1, len(group1)):

                item2 = group1[j]

                if item1["path"] == item2["path"]:
                    continue

                total_pairs += 1

                distance = hamming_distance(
                    item1["phash"],
                    item2["phash"]
                )

                if distance <= PHASH_THRESHOLD:

                    similar_pairs += 1

                    similarity = similarity_percentage(
                        distance
                    )

                    similarity_values.append(
                        similarity
                    )

                    if len(examples) < MAX_EXAMPLES:

                        examples.append({
                            "image1": item1["filename"],
                            "image2": item2["filename"],
                            "distance": distance,
                            "similarity": round(
                                similarity, 2
                            )
                        })

    # --------------------------------------------------------
    # CROSS GROUP
    # --------------------------------------------------------

    else:

        for item1 in group1:

            for item2 in group2:

                total_pairs += 1

                distance = hamming_distance(
                    item1["phash"],
                    item2["phash"]
                )

                if distance <= PHASH_THRESHOLD:

                    similar_pairs += 1

                    similarity = similarity_percentage(
                        distance
                    )

                    similarity_values.append(
                        similarity
                    )

                    if len(examples) < MAX_EXAMPLES:

                        examples.append({
                            "image1": item1["filename"],
                            "image2": item2["filename"],
                            "distance": distance,
                            "similarity": round(
                                similarity, 2
                            )
                        })

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"Pairs checked          : {total_pairs:,}"
    )

    print(
        f"Similar pairs           : {similar_pairs:,}"
    )

    if total_pairs > 0:

        percentage = (
            similar_pairs /
            total_pairs
        ) * 100

    else:

        percentage = 0

    print(
        f"Similar pair percentage : {percentage:.2f}%"
    )

    if similarity_values:

        print(
            f"Average similarity     : "
            f"{sum(similarity_values) / len(similarity_values):.2f}%"
        )

        print(
            f"Maximum similarity     : "
            f"{max(similarity_values):.2f}%"
        )

    return {
        "pairs_checked": total_pairs,
        "similar_pairs": similar_pairs,
        "percentage": percentage,
        "examples": examples
    }


# ============================================================
# A) REAL vs REAL
# ============================================================

print("\n" + "=" * 70)
print("A) REAL vs REAL")
print("=" * 70)

real_buckets = defaultdict(list)

for item in real_images:

    if "phash" in item:

        real_buckets[
            item["phash"][:4]
        ].append(item)


real_similar_pairs = 0
real_pairs_checked = 0
real_similarity_values = []
real_examples = []

for bucket in real_buckets.values():

    result = analyze_pair_group(
        bucket,
        bucket,
        "REAL ↔ REAL BUCKET",
        same_group=True
    )

    real_pairs_checked += result["pairs_checked"]
    real_similar_pairs += result["similar_pairs"]

    real_similarity_values.extend(
        [
            x["similarity"]
            for x in result["examples"]
        ]
    )

    real_examples.extend(
        result["examples"]
    )


# ============================================================
# B) SYNTHETIC vs SYNTHETIC
# ============================================================

print("\n" + "=" * 70)
print("B) SYNTHETIC vs SYNTHETIC")
print("=" * 70)

synthetic_buckets = defaultdict(list)

for item in synthetic_images:

    if "phash" in item:

        synthetic_buckets[
            item["phash"][:4]
        ].append(item)


synthetic_similar_pairs = 0
synthetic_pairs_checked = 0
synthetic_examples = []

for bucket in synthetic_buckets.values():

    result = analyze_pair_group(
        bucket,
        bucket,
        "SYNTHETIC ↔ SYNTHETIC BUCKET",
        same_group=True
    )

    synthetic_pairs_checked += result["pairs_checked"]
    synthetic_similar_pairs += result["similar_pairs"]

    synthetic_examples.extend(
        result["examples"]
    )


# ============================================================
# C) REAL vs SYNTHETIC
# ============================================================

print("\n" + "=" * 70)
print("C) REAL vs SYNTHETIC")
print("=" * 70)

# ------------------------------------------------------------
# First: compare synthetic image to its source real image
#
# ABC123_0.jpg -> ABC123.jpg
# ABC123_1.jpg -> ABC123.jpg
# ------------------------------------------------------------

real_by_base = {
    item["base"]: item
    for item in real_images
    if "phash" in item
}


source_pairs_checked = 0
source_similar_pairs = 0
source_similarity_values = []

source_examples = []

for item in synthetic_images:

    if "phash" not in item:
        continue

    source_real = real_by_base.get(
        item["base"]
    )

    if source_real is None:
        continue

    source_pairs_checked += 1

    distance = hamming_distance(
        source_real["phash"],
        item["phash"]
    )

    similarity = similarity_percentage(
        distance
    )

    source_similarity_values.append(
        similarity
    )

    if distance <= PHASH_THRESHOLD:

        source_similar_pairs += 1

    if len(source_examples) < MAX_EXAMPLES:

        source_examples.append({
            "real_image": source_real["filename"],
            "synthetic_image": item["filename"],
            "distance": distance,
            "similarity": round(
                similarity,
                2
            )
        })


print(
    f"Real → synthetic source pairs : "
    f"{source_pairs_checked:,}"
)

print(
    f"Highly similar source pairs   : "
    f"{source_similar_pairs:,}"
)

if source_similarity_values:

    print(
        f"Average source similarity     : "
        f"{sum(source_similarity_values) / len(source_similarity_values):.2f}%"
    )

    print(
        f"Maximum source similarity     : "
        f"{max(source_similarity_values):.2f}%"
    )

    print(
        f"Minimum source similarity     : "
        f"{min(source_similarity_values):.2f}%"
    )


# ============================================================
# SAVE HASH DATABASE
# ============================================================

hash_csv = os.path.join(
    OUTPUT_DIR,
    "image_hashes.csv"
)

print("\nSaving hash information...")

with open(
    hash_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "type",
        "base_name",
        "md5",
        "phash"
    ])

    for item in phash_records:

        writer.writerow([
            item["filename"],
            item["type"],
            item["base"],
            item["md5"],
            item["phash"]
        ])


# ============================================================
# SAVE REAL vs SYNTHETIC SOURCE RESULTS
# ============================================================

source_csv = os.path.join(
    OUTPUT_DIR,
    "real_vs_synthetic.csv"
)

with open(
    source_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "real_image",
        "synthetic_image",
        "phash_distance",
        "similarity_percent"
    ])

    for row in source_examples:

        writer.writerow([
            row["real_image"],
            row["synthetic_image"],
            row["distance"],
            row["similarity"]
        ])


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_csv = os.path.join(
    OUTPUT_DIR,
    "similarity_summary.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "analysis",
        "pairs_checked",
        "similar_pairs",
        "similar_percentage"
    ])

    writer.writerow([
        "Real vs Real",
        real_pairs_checked,
        real_similar_pairs,
        round(
            (
                real_similar_pairs /
                real_pairs_checked * 100
            )
            if real_pairs_checked
            else 0,
            2
        )
    ])

    writer.writerow([
        "Synthetic vs Synthetic",
        synthetic_pairs_checked,
        synthetic_similar_pairs,
        round(
            (
                synthetic_similar_pairs /
                synthetic_pairs_checked * 100
            )
            if synthetic_pairs_checked
            else 0,
            2
        )
    ])

    writer.writerow([
        "Real vs Synthetic source",
        source_pairs_checked,
        source_similar_pairs,
        round(
            (
                source_similar_pairs /
                source_pairs_checked * 100
            )
            if source_pairs_checked
            else 0,
            2
        )
    ])


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SIMILARITY ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Total images      : {len(images):,}"
)

print(
    f"Real images       : {len(real_images):,}"
)

print(
    f"Synthetic images  : {len(synthetic_images):,}"
)

print(
    f"\nExact duplicates  : "
    f"{exact_duplicate_count:,}"
)

print(
    f"\nReports saved in:"
)

print(
    OUTPUT_DIR
)

print("\nFiles:")

print(
    "  image_hashes.csv"
)

print(
    "  real_vs_synthetic.csv"
)

print(
    "  similarity_summary.csv"
)

print("=" * 70)
