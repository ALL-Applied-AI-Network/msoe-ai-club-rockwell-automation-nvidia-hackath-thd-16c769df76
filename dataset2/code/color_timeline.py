# color_timeline.py
# For every video-5 vial_fill image: measure how much strongly-colored liquid
# is in the region where the vial sits. Plot that over time next to what the
# labels claim, so we can see real fill events vs label windows.

import pandas as pd, numpy as np, os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")              # draw plots to a file (no screen on the Spark)
import matplotlib.pyplot as plt

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"
os.makedirs(OUT, exist_ok=True)

# Region of interest (ROI) = box around the bottom of the vial under the nozzle,
# in full-size 640x480 pixel coordinates: (left, top, right, bottom).
ROI = (230, 330, 380, 440)

labels = pd.read_csv(f"{DATA}/labels/video5_frame_labels.csv").sort_values("filename")
labels["hhmmss"] = labels["filename"].str.split("_").str[-2]

rows = []
for r in labels.itertuples():
    img = Image.open(f"{DATA}/images/{r.filename}").convert("RGB").crop(ROI)
    hsv = np.array(img.convert("HSV")).astype(float)   # Hue, Saturation, Value (0-255)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (s > 90) & (v > 40)        # strongly colored and not too dark
    frac = colored.mean()                # fraction of ROI pixels that are colored
    hue = np.median(h[colored]) if colored.any() else np.nan
    rows.append((r.filename, r.hhmmss, r.video_s, r.in_fill, r.vial, frac, hue))

df = pd.DataFrame(rows, columns=["filename","hhmmss","video_s","in_fill","vial","color_frac","hue"])
df.to_csv(f"{OUT}/color_timeline.csv", index=False)

fig, ax = plt.subplots(figsize=(22, 5))
ax.plot(df["video_s"], df["color_frac"], lw=1, color="black", label="colored fraction in ROI")
ax.fill_between(df["video_s"], 0, df["in_fill"] * df["color_frac"].max(),
                color="orange", alpha=0.25, label="label says in_fill")
ax.set_xlabel("video_s (seconds into video 5)")
ax.set_ylabel("colored fraction")
ax.legend(loc="upper right")
ax.set_title("Video 5: measured color vs label windows")
fig.tight_layout()
fig.savefig(f"{OUT}/color_timeline.png", dpi=90)

picks = df.iloc[np.linspace(0, len(df) - 1, 8).astype(int)]
sheet = Image.new("RGB", (4 * 320, 2 * 240))
for i, p in enumerate(picks.itertuples()):
    im = Image.open(f"{DATA}/images/{p.filename}").convert("RGB")
    ImageDraw.Draw(im).rectangle(ROI, outline="red", width=3)
    sheet.paste(im.resize((320, 240)), ((i % 4) * 320, (i // 4) * 240))
sheet.save(f"{OUT}/roi_check.jpg", quality=90)

print(f"Done: {len(df)} frames -> color_timeline.csv, color_timeline.png, roi_check.jpg")
