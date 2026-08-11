python3 -c '
import json
from pathlib import Path

labels=Path("/home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/datasets/JerseyNumber_Validation_Dataset/labels")

for j in labels.glob("*.json"):
    d=json.load(open(j))
    w,h=d["dimensions"]
    with open(j.with_suffix(".txt"),"w") as f:
        for n in d["data"]["jersey"]["numbers"]:
            digit=n["digits"]
            x,y,bw,bh=n["rect"]
            f.write(f"{digit} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

print("JSON -> TXT conversion completed")
' && python3 jersey_dataset_patched_updated_2.py generate --root /home/eng_bhavana/workspace_bhavana/jersey_number_recognition_20260626/datasets/JerseyNumber_Validation_Dataset/ --outdir synthetic_output4 --n-samples 500 --tilt-mean 0 --tilt-std 4
