#!/usr/bin/env bash
# Rebuild caltechfn_sample_350_clean from scratch: fresh caltechfn-only zip +
# the corrected rbnr_final_101.zip, merged once, cleanly.
#
# Usage:
#   bash rebuild_merge.sh <path-to-caltechfn-only-zip> <path-to-rbnr_final_101.zip> <dest-dir>
#
# Example:
#   bash rebuild_merge.sh \
#       ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350.zip \
#       ~/workspace_bhavana/jersey_number_recognition_20260626/jersey-number-projectd42w2saxx/rbnr_final_101.zip \
#       ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean

set -euo pipefail

CALTECH_ZIP="${1:?Usage: $0 <caltechfn_zip> <rbnr_final_101_zip> <dest_dir>}"
RBNR_ZIP="${2:?Usage: $0 <caltechfn_zip> <rbnr_final_101_zip> <dest_dir>}"
DEST="${3:?Usage: $0 <caltechfn_zip> <rbnr_final_101_zip> <dest_dir>}"

TMP_RBNR=$(mktemp -d)
TMP_CALTECH=$(mktemp -d)

echo "== 1. Removing old merged folder (if present): $DEST"
rm -rf "$DEST"
mkdir -p "$DEST/images" "$DEST/labels"

echo "== 2. Extracting caltechfn zip fresh into a temp dir to inspect structure"
unzip -q -o "$CALTECH_ZIP" -d "$TMP_CALTECH"

# Handle both possible layouts: images/ labels/ at top level, or nested one level deep.
CALTECH_IMAGES_DIR=$(find "$TMP_CALTECH" -type d -name images | head -n 1)
CALTECH_LABELS_DIR=$(find "$TMP_CALTECH" -type d -name labels | head -n 1)

if [[ -z "$CALTECH_IMAGES_DIR" || -z "$CALTECH_LABELS_DIR" ]]; then
    echo "ERROR: could not find images/ and labels/ folders inside $CALTECH_ZIP"
    echo "Contents found:"
    find "$TMP_CALTECH" -maxdepth 3
    exit 1
fi

echo "   found images: $CALTECH_IMAGES_DIR"
echo "   found labels: $CALTECH_LABELS_DIR"
cp "$CALTECH_IMAGES_DIR"/* "$DEST/images/"
cp "$CALTECH_LABELS_DIR"/* "$DEST/labels/"

echo "== 3. Extracting rbnr_final_101.zip"
unzip -q -o "$RBNR_ZIP" -d "$TMP_RBNR"

RBNR_IMAGES_DIR=$(find "$TMP_RBNR" -type d -name images | head -n 1)
RBNR_LABELS_DIR=$(find "$TMP_RBNR" -type d -name labels | head -n 1)

if [[ -z "$RBNR_IMAGES_DIR" || -z "$RBNR_LABELS_DIR" ]]; then
    echo "ERROR: could not find images/ and labels/ folders inside $RBNR_ZIP"
    find "$TMP_RBNR" -maxdepth 3
    exit 1
fi

echo "   found images: $RBNR_IMAGES_DIR"
echo "   found labels: $RBNR_LABELS_DIR"

echo "== 4. Copying RBNR images/labels into merged folder"
cp "$RBNR_IMAGES_DIR"/* "$DEST/images/"
cp "$RBNR_LABELS_DIR"/* "$DEST/labels/"

echo "== 5. Sanity check"
N_IMG=$(ls "$DEST/images" | wc -l)
N_LBL=$(ls "$DEST/labels" | wc -l)
echo "   images: $N_IMG"
echo "   labels: $N_LBL"

if [[ "$N_IMG" != "$N_LBL" ]]; then
    echo "WARNING: image count and label count do not match! Check for filename collisions or missing pairs."
fi

echo "== 6. Checking every label has a matching image (and vice versa)"
MISSING=0
for f in "$DEST"/labels/*.json; do
    stem=$(basename "$f" .json)
    if [[ ! -f "$DEST/images/$stem.jpg" ]]; then
        echo "   MISSING IMAGE for label: $stem"
        MISSING=$((MISSING + 1))
    fi
done
for f in "$DEST"/images/*.jpg; do
    stem=$(basename "$f" .jpg)
    if [[ ! -f "$DEST/labels/$stem.json" ]]; then
        echo "   MISSING LABEL for image: $stem"
        MISSING=$((MISSING + 1))
    fi
done

if [[ "$MISSING" -eq 0 ]]; then
    echo "   OK -- every image has a matching label and vice versa."
else
    echo "   $MISSING mismatches found -- see above."
fi

rm -rf "$TMP_RBNR" "$TMP_CALTECH"

echo
echo "Done. Merged dataset ready at: $DEST"
echo "  images: $N_IMG"
echo "  labels: $N_LBL"
