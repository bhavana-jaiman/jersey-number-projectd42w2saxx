cd ~/workspace_bhavana/Datasets/generalization_test
python3 << 'EOF'
import os

base = "."
new_root = "caltechfn_sample_350_clean"
os.makedirs(f"{new_root}/images", exist_ok=True)
os.makedirs(f"{new_root}/labels", exist_ok=True)

count = 0
for name in os.listdir(base):
    if not os.path.isfile(name):
        continue
    if "\\images\\" in name:
        newname = name.split("\\images\\")[-1]
        os.rename(name, f"{new_root}/images/{newname}")
        count += 1
    elif "\\labels\\" in name:
        newname = name.split("\\labels\\")[-1]
        os.rename(name, f"{new_root}/labels/{newname}")
        count += 1

print(f"Moved {count} files into {new_root}/")
EOF
