import glob, os, shutil

base = "../Datasets/70_30_dataset/validation_final"
out = "../Datasets/70_30_dataset/validation_excluded"
DRY_RUN = True   # change to False to actually move files

os.makedirs(os.path.join(out, "labels"), exist_ok=True)
os.makedirs(os.path.join(out, "images"), exist_ok=True)

count = 0
for f in glob.glob(os.path.join(base, "labels", "*.txt")):
    with open(f) as fh:
        vals = fh.read().split()
    if len(vals) > 2:
        count += 1
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"{name}  ->  {' '.join(vals)}")
        if not DRY_RUN:
            shutil.move(f, os.path.join(out, "labels", os.path.basename(f)))
            for ext in (".jpg", ".jpeg", ".png"):
                img = os.path.join(base, "images", name + ext)
                if os.path.exists(img):
                    shutil.move(img, os.path.join(out, "images", name + ext))

print(f"\n{count} files {'would be' if DRY_RUN else 'were'} moved")
