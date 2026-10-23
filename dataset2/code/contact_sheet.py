# contact_sheet.py
# Makes one image showing the finished-fill frame of all 30 vials in video 5,
# labeled with vial number and recipe, so we can see where the vial sits
# and spot the discolored "first vial" of each recipe.

import pandas as pd
from PIL import Image, ImageDraw
import os

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"
os.makedirs(OUT, exist_ok=True)

# Read the per-frame labels (one row per sampled image)
labels = pd.read_csv(f"{DATA}/labels/video5_frame_labels.csv")

# Keep only frames where a vial is being filled
filling = labels[labels["in_fill"] == 1]

# For each vial, take its LAST filling frame = closest to the finished vial
last_frames = filling.sort_values("video_s").groupby("vial").tail(1).sort_values("vial")

# Build a 6-column grid of half-size images (320x240 each)
cols, w, h = 6, 320, 240
rows = (len(last_frames) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "black")

for i, row in enumerate(last_frames.itertuples()):
    img = Image.open(f"{DATA}/images/{row.filename}").convert("RGB").resize((w, h))
    d = ImageDraw.Draw(img)
    first = " <- FIRST" if (row.vial - 1) % 3 == 0 else ""
    d.rectangle([0, 0, w, 18], fill="black")
    d.text((4, 3), f"vial {row.vial} | recipe {row.recipe_id}{first}", fill="yellow")
    sheet.paste(img, ((i % cols) * w, (i // cols) * h))

sheet.save(f"{OUT}/contact_sheet.jpg", quality=90)
last_frames[["vial", "recipe_id", "filename", "video_s"]].to_csv(f"{OUT}/last_frames.csv", index=False)
print(f"Saved {len(last_frames)} vials to {OUT}/contact_sheet.jpg")
