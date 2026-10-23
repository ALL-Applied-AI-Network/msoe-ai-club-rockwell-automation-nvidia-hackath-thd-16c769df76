# check_alignment.py
# Shows every vial_fill image in a time window, labeled with the clock time
# from the filename and what the label file says (in_fill, vial, recipe).
# Lets us see whether the labels line up with what the camera shows.

import pandas as pd
from PIL import Image, ImageDraw
import os, sys

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"
os.makedirs(OUT, exist_ok=True)

# Time window as HHMMSS (default: around vials 6-9). Can override:
#   python3 check_alignment.py 195230 195600
start = sys.argv[1] if len(sys.argv) > 1 else "195230"
end   = sys.argv[2] if len(sys.argv) > 2 else "195600"

labels = pd.read_csv(f"{DATA}/labels/video5_frame_labels.csv")

# Filename looks like: vial_fill_20260923_195325_005000.jpg
# Pull out the HHMMSS part (second-to-last piece when split on "_")
labels["hhmmss"] = labels["filename"].str.split("_").str[-2]
win = labels[(labels["hhmmss"] >= start) & (labels["hhmmss"] <= end)].sort_values("filename")

# Print the label rows so we can read them as a table too
print(win[["filename", "video_s", "in_fill", "vial", "recipe_id"]].to_string(index=False))

# Tile the images: 8 columns, each 240x180
cols, w, h = 8, 240, 180
rows = max(1, (len(win) + cols - 1) // cols)
sheet = Image.new("RGB", (cols * w, rows * h), "black")
for i, r in enumerate(win.itertuples()):
    img = Image.open(f"{DATA}/images/{r.filename}").convert("RGB").resize((w, h))
    d = ImageDraw.Draw(img)
    t = f"{r.hhmmss[:2]}:{r.hhmmss[2:4]}:{r.hhmmss[4:]}"
    tag = f"v{int(r.vial)} r{int(r.recipe_id)}" if r.in_fill == 1 else "idle"
    d.rectangle([0, 0, w, 16], fill="black")
    d.text((3, 2), f"{t}  {tag}", fill="yellow")
    sheet.paste(img, ((i % cols) * w, (i // cols) * h))

name = f"{OUT}/align_{start}_{end}.jpg"
sheet.save(name, quality=90)
print(f"\nSaved {len(win)} frames to {name}")
