import os
from pathlib import Path

# ---- Configuration ----
BASE_DIR = Path("Datasets/70_30_dataset")
TRAIN_DIR = BASE_DIR / "training"            # images + labels directly inside
VAL_DIR = BASE_DIR / "validation_final"      # images/ and labels/ subfolders

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".txt", ".xml", ".json"}


def list_files(folder, exts):
    """Return a set of file stems in `folder` (non-recursive) matching `exts`."""
    if not folder.is_dir():
        print(f"  [!] Folder not found: {folder}")
        return {}
    return {
        f.stem: f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    }


def report(name, images, labels):
    img_stems, lbl_stems = set(images), set(labels)
    matched = img_stems & lbl_stems
    imgs_no_label = img_stems - lbl_stems
    labels_no_img = lbl_stems - img_stems

    print(f"\n===== {name} =====")
    print(f"Images              : {len(images)}")
    print(f"Labels              : {len(labels)}")
    print(f"Matched pairs       : {len(matched)}")
    print(f"Images w/o labels   : {len(imgs_no_label)}")
    print(f"Labels w/o images   : {len(labels_no_img)}")

    # Show a few examples of mismatches, if any
    if imgs_no_label:
        print("  e.g. images missing labels:", sorted(imgs_no_label)[:5])
    if labels_no_img:
        print("  e.g. labels missing images:", sorted(labels_no_img)[:5])

    return len(images), len(labels)


def main():
    # Training: images and labels are in the same folder
    train_imgs = list_files(TRAIN_DIR, IMAGE_EXTS)
    train_lbls = list_files(TRAIN_DIR, LABEL_EXTS)
    t_i, t_l = report(f"TRAINING ({TRAIN_DIR})", train_imgs, train_lbls)

    # Validation: images and labels are in separate subfolders
    val_imgs = list_files(VAL_DIR / "images", IMAGE_EXTS)
    val_lbls = list_files(VAL_DIR / "labels", LABEL_EXTS)
    v_i, v_l = report(f"VALIDATION ({VAL_DIR})", val_imgs, val_lbls)

    # Totals and split ratio
    total_imgs = t_i + v_i
    print("\n===== TOTAL =====")
    print(f"Images : {total_imgs}")
    print(f"Labels : {t_l + v_l}")
    if total_imgs:
        print(f"Split  : {t_i / total_imgs * 100:.1f}% train / "
              f"{v_i / total_imgs * 100:.1f}% validation")


if __name__ == "__main__":
    main()
