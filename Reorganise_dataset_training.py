import os, shutil

src = "training_dataset_Ying"

os.makedirs(f"{src}/images", exist_ok=True)
os.makedirs(f"{src}/labels", exist_ok=True)

for f in os.listdir(src):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        shutil.copy2(f"{src}/{f}", f"{src}/images/")
    elif f.endswith(".txt"):
        shutil.copy2(f"{src}/{f}", f"{src}/labels/")

print("Done!")
