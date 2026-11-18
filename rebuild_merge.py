#!/usr/bin/env python3
"""
Rebuild caltechfn_sample_350_clean from scratch: fresh caltechfn-only zip +
the corrected rbnr_final_101.zip, merged once, cleanly.

Pure Python (uses the zipfile module) -- no dependency on the `unzip` CLI,
which isn't installed on this server. Also auto-fixes the Windows
backslash-in-filename issue that PowerShell's Compress-Archive sometimes
produces when extracted on Linux (e.g. "images\\foo.jpg" as one literal
filename instead of a real subfolder).

Usage:
    python3 rebuild_merge.py <caltechfn_zip> <rbnr_final_101_zip> <dest_dir>

Example:
    python3 rebuild_merge.py \\
        ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350.zip \\
        ~/workspace_bhavana/jersey_number_recognition_20260626/jersey-number-projectd42w2saxx/rbnr_final_101.zip \\
        ~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean
"""
import shutil
import sys
import zipfile
from pathlib import Path


def extract_fixing_backslashes(zip_path: Path, out_dir: Path):
    """Extract a zip, splitting any literal-backslash filenames (from
    Windows Compress-Archive) into real files under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue  # directory entry
            # normalize both kinds of embedded path separators
            clean_parts = name.replace("\\", "/").split("/")
            clean_name = clean_parts[-1]
            # figure out which "folder" this belongs to (images/labels/...)
            folder = None
            for part in clean_parts[:-1]:
                if part:
                    folder = part
            if folder is None and len(clean_parts) > 1:
                folder = clean_parts[-2]
            dest_dir = out_dir / folder if folder else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest_dir / clean_name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def find_dir(root: Path, name: str):
    if (root / name).is_dir():
        return root / name
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <caltechfn_zip> <rbnr_final_101_zip> <dest_dir>")
        sys.exit(1)

    caltech_zip = Path(sys.argv[1]).expanduser()
    rbnr_zip = Path(sys.argv[2]).expanduser()
    dest = Path(sys.argv[3]).expanduser()

    for p in (caltech_zip, rbnr_zip):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    workdir = dest.parent / ".rebuild_merge_tmp"
    if workdir.exists():
        shutil.rmtree(workdir)
    tmp_caltech = workdir / "caltech"
    tmp_rbnr = workdir / "rbnr"

    print(f"== 1. Removing old merged folder (if present): {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)

    print("== 2. Extracting caltechfn zip fresh")
    extract_fixing_backslashes(caltech_zip, tmp_caltech)
    caltech_images = find_dir(tmp_caltech, "images")
    caltech_labels = find_dir(tmp_caltech, "labels")
    if not caltech_images or not caltech_labels:
        print(f"ERROR: could not find images/ and labels/ inside {caltech_zip}")
        print("Found:", list(tmp_caltech.rglob("*")))
        sys.exit(1)
    print(f"   found images: {caltech_images}")
    print(f"   found labels: {caltech_labels}")

    n_caltech_img = 0
    for f in caltech_images.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / "images" / f.name)
            n_caltech_img += 1
    n_caltech_lbl = 0
    for f in caltech_labels.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / "labels" / f.name)
            n_caltech_lbl += 1
    print(f"   copied {n_caltech_img} caltechfn images, {n_caltech_lbl} labels")

    print("== 3. Extracting rbnr_final_101.zip")
    extract_fixing_backslashes(rbnr_zip, tmp_rbnr)
    rbnr_images = find_dir(tmp_rbnr, "images")
    rbnr_labels = find_dir(tmp_rbnr, "labels")
    if not rbnr_images or not rbnr_labels:
        print(f"ERROR: could not find images/ and labels/ inside {rbnr_zip}")
        print("Found:", list(tmp_rbnr.rglob("*")))
        sys.exit(1)
    print(f"   found images: {rbnr_images}")
    print(f"   found labels: {rbnr_labels}")

    n_rbnr_img = 0
    for f in rbnr_images.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / "images" / f.name)
            n_rbnr_img += 1
    n_rbnr_lbl = 0
    for f in rbnr_labels.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / "labels" / f.name)
            n_rbnr_lbl += 1
    print(f"   copied {n_rbnr_img} rbnr images, {n_rbnr_lbl} labels")

    print("== 4. Sanity check")
    img_files = {f.stem for f in (dest / "images").iterdir() if f.is_file()}
    lbl_files = {f.stem for f in (dest / "labels").iterdir() if f.is_file()}
    print(f"   images: {len(img_files)}")
    print(f"   labels: {len(lbl_files)}")

    missing_img = lbl_files - img_files
    missing_lbl = img_files - lbl_files
    if missing_img:
        print(f"   MISSING IMAGES for {len(missing_img)} labels: {sorted(missing_img)[:10]}...")
    if missing_lbl:
        print(f"   MISSING LABELS for {len(missing_lbl)} images: {sorted(missing_lbl)[:10]}...")
    if not missing_img and not missing_lbl:
        print("   OK -- every image has a matching label and vice versa.")

    shutil.rmtree(workdir)

    print()
    print(f"Done. Merged dataset ready at: {dest}")
    print(f"  images: {len(img_files)}")
    print(f"  labels: {len(lbl_files)}")
    print(f"  (caltechfn: {n_caltech_img}, rbnr: {n_rbnr_img})")


if __name__ == "__main__":
    main()
