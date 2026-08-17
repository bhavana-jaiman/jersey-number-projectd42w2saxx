import os

dataset_path = "."

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_count = 0
label_count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        ext = os.path.splitext(file)[1].lower()

        if ext in image_extensions:
            image_count += 1
        elif ext == ".txt":
            label_count += 1

print(f"Total Images : {image_count}")
print(f"Total Labels : {label_count}")
