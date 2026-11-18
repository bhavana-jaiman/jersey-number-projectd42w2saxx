cd ~/workspace_bhavana/jersey_number_recognition_20260626/jersey-number-projectd42w2saxx

python3 << 'EOF'
import zipfile
import os
import shutil

# 1. extract the RBNR state2 zip
with zipfile.ZipFile("rbnr_state2.zip") as z:
    z.extractall("/tmp/rbnr_state2_raw")

# 2. merge its images/labels into the existing CaltechFN clean folder
src = "/tmp/rbnr_state2_raw"
dst = os.path.expanduser("~/workspace_bhavana/Datasets/generalization_test/caltechfn_sample_350_clean")

count = 0
for root, dirs, files in os.walk(src):
    for name in files:
        full = os.path.join(root, name)
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.move(full, f"{dst}/images/{name}")
            count += 1
        elif name.lower().endswith(".json"):
            shutil.move(full, f"{dst}/labels/{name}")
            count += 1

print(f"Merged {count} files into {dst}")

n_images = len(os.listdir(f"{dst}/images"))
n_labels = len(os.listdir(f"{dst}/labels"))
print(f"Final counts -> images: {n_images}, labels: {n_labels}")
EOF
