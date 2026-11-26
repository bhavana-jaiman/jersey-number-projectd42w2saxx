import os, re, csv, argparse
from collections import Counter

IMG_EXT = (".jpg", ".jpeg", ".png")


def get_labels(a, root):
    labels = []  # (image_name, label_string)
    if a.mode == "folder":
        for cls in sorted(os.listdir(root)):
            p = os.path.join(root, cls)
            if os.path.isdir(p):
                for dp, _, fs in os.walk(p):
                    for f in fs:
                        if f.lower().endswith(IMG_EXT):
                            labels.append((f, cls))
    elif a.mode == "filename":
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(IMG_EXT):
                    parts = os.path.splitext(f)[0].split(a.sep)
                    labels.append((f, parts[a.index]))
    elif a.mode == "labels":
        ldir = os.path.join(root, "labels")
        for f in os.listdir(ldir):
            if f.endswith(".txt"):
                with open(os.path.join(ldir, f)) as fh:
                    txt = fh.read().strip()
                m = re.search(r"-?\d+", txt)
                labels.append((f, m.group(0) if m else "-1"))
    elif a.mode == "csv":
        with open(os.path.expanduser(a.csv)) as fh:
            for row in csv.reader(fh):
                labels.append((row[a.img_col], row[a.label_col].strip()))
    return labels


def main():
    ap = argparse.ArgumentParser(description="Count jersey digits in a dataset")
    ap.add_argument("root")
    ap.add_argument("--mode", required=True, choices=["folder", "filename", "labels", "csv"])
    ap.add_argument("--sep", default="_", help="filename separator (filename mode)")
    ap.add_argument("--index", type=int, default=-1, help="filename part holding the label")
    ap.add_argument("--csv", help="csv path (csv mode)")
    ap.add_argument("--img_col", type=int, default=0)
    ap.add_argument("--label_col", type=int, default=1)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    labels = get_labels(a, root)
    whole, per_digit, tens, units, length = Counter(), Counter(), Counter(), Counter(), Counter()
    no_jersey = 0
    for _, lab in labels:
        if not re.fullmatch(r"\d+", lab):  # -1, empty, "none", etc.
            no_jersey += 1
            continue
        whole[lab] += 1
        length[len(lab)] += 1
        for ch in lab:
            per_digit[ch] += 1
        units[lab[-1]] += 1
        if len(lab) >= 2:
            tens[lab[-2]] += 1

    print(f"Total images/labels : {len(labels)}")
    print(f"No jersey (-1/other): {no_jersey}")
    print("\nNumber length (digits in jersey):", dict(sorted(length.items())))
    print("\nOverall digit count (0-9):")
    for d in "0123456789":
        print(f"  {d}: {per_digit[d]}")
    print("\nUnits-place digit:", {d: units[d] for d in "0123456789"})
    print("Tens-place digit :", {d: tens[d] for d in "0123456789"})
    print("\nWhole-number counts:")
    for n, c in sorted(whole.items(), key=lambda x: int(x[0])):
        print(f"  {n}: {c}")


if __name__ == "__main__":
    main()
