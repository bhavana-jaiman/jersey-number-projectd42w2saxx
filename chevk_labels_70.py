import glob, os
from collections import defaultdict

label_dir = "../Datasets/70_30_dataset/validation_final/labels"

files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
print("Found", len(files), "label files\n")

groups = defaultdict(list)
for f in files:
    with open(f) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    layout = (len(lines), tuple(len(l.split()) for l in lines))
    groups[layout].append(f)

# Most common layouts first; rare ones (likely bad) at the bottom
for (n_lines, per_line), fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(fs):6d} files | {n_lines} lines | values per line: {per_line}")
    for f in fs[:5]:
        with open(f) as fh:
            content = fh.read().strip().replace("\n", " \\n ")
        print(f"         {os.path.basename(f)}  ->  {content}")
    print()
