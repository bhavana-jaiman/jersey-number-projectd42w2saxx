"""
extract_from_list.py  -- copies the images listed in a .txt file out of the training data
into a NEW folder (images/ + labels/) and zips it, so you can download one file.
The training folders are never modified.

The .txt file: one image per line. Any of these work:
    0A32915A7DBEB4C2A8D924B52B6F37A1.jpg      (file name)
    0A32915A7DBEB4C2A8D924B52B6F37A1          (name without extension)
    /home/.../real/0A32915A...jpg             (full or relative path)
    001_loss3.21_lab27_pred21__0A3291...jpg   (review-style name, part after "__" is used)
Empty lines and lines starting with # are ignored.

Run:
  python extract_from_list.py --list my_list.txt \
      --search ../datasets/training_dataset_Ying_separated/ --out extracted_images
  -> extracted_images/images, extracted_images/labels, extracted_images.zip
"""
import argparse
import os
import shutil

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def wanted_names(list_file):
    names = []
    for line in open(list_file, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.split(",")[0].strip()                 # csv line -> first column
        s = os.path.basename(s).split("__", 1)[-1]  # path / review name -> original name
        names.append(os.path.splitext(s)[0])
    return list(dict.fromkeys(names))               # unique, keep order


def label_for(img_path):
    stem = os.path.splitext(img_path)[0]
    if os.path.exists(stem + ".txt"):
        return stem + ".txt"
    d, n = os.path.split(stem)
    lp = os.path.join(os.path.dirname(d), "labels", n + ".txt")
    return lp if os.path.exists(lp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help=".txt file with the image names")
    ap.add_argument("--search", nargs="+", required=True, help="training folder(s) to search in")
    ap.add_argument("--out", default="extracted_images", help="NEW output folder")
    ap.add_argument("--no_labels", action="store_true", help="copy images only")
    ap.add_argument("--no_zip", action="store_true", help="do not create the .zip")
    opt = ap.parse_args()

    names = wanted_names(opt.list)
    print(f"names in list: {len(names)}")

    index = {}                                        # stem -> first image path found
    for root in opt.search:
        for d, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(EXTS):
                    index.setdefault(os.path.splitext(f)[0], os.path.join(d, f))
    print(f"images indexed in training folders: {len(index)}")

    os.makedirs(os.path.join(opt.out, "images"), exist_ok=True)
    if not opt.no_labels:
        os.makedirs(os.path.join(opt.out, "labels"), exist_ok=True)

    found, missing, no_label = 0, [], 0
    for n in names:
        ip = index.get(n)
        if ip is None:
            missing.append(n)
            continue
        shutil.copy2(ip, os.path.join(opt.out, "images", os.path.basename(ip)))
        if not opt.no_labels:
            lp = label_for(ip)
            if lp:
                shutil.copy2(lp, os.path.join(opt.out, "labels", n + ".txt"))
            else:
                no_label += 1
        found += 1

    print(f"copied: {found} images -> {os.path.abspath(opt.out)}")
    if no_label:
        print(f"  {no_label} of them had no label file")
    if missing:
        with open(os.path.join(opt.out, "not_found.txt"), "w") as f:
            f.write("\n".join(missing) + "\n")
        print(f"NOT FOUND: {len(missing)} (listed in {opt.out}/not_found.txt), e.g. {missing[:5]}")

    if not opt.no_zip and found:
        z = shutil.make_archive(os.path.abspath(opt.out), "zip", opt.out)
        print(f"zip for download: {z}")


if __name__ == "__main__":
    main()
