import json
from pathlib import Path

labels = Path(
    "/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/"
    "datasets/JerseyNumber_Validation_Dataset/labels"
)

for json_file in labels.glob("*.json"):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        w, h = data["dimensions"]
        numbers = data["data"]["jersey"]["numbers"]

        txt_file = json_file.with_suffix(".txt")

        with open(txt_file, "w") as f:
            for item in numbers:
                digit = item["digits"]
                x, y, bw, bh = item["rect"]

                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                f.write(
                    f"{digit} {cx:.6f} {cy:.6f} "
                    f"{nw:.6f} {nh:.6f}\n"
                )

        print(f"Converted: {json_file.name}")

    except Exception as e:
        print(f"ERROR: {json_file.name}: {e}")

print("JSON -> TXT conversion completed.")
