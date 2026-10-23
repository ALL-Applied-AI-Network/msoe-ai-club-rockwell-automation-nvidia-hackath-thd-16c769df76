# segment_fills.py
# Uses color_timeline.csv to find each fill event from the images themselves:
# a fill = a stretch of frames where liquid is visible in the ROI, ending when
# the vial leaves (sharp drop to ~0). Numbers the fills in order.

import pandas as pd, numpy as np, os
from PIL import Image, ImageDraw

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"
ROI  = (230, 330, 380, 440)

THRESH   = 0.015   # above this = liquid present (empty vial baseline is ~0.00)
MIN_LEN  = 3       # a fill must last at least 3 frames (~6 s)
MAX_GAP  = 0       # allow a 1-frame dip inside a fill (noise) without splitting it

df = pd.read_csv(f"{OUT}/color_timeline.csv").sort_values("video_s").reset_index(drop=True)
present = (df["color_frac"] > THRESH).to_numpy()

# Fill small gaps: if a single frame dips below threshold between two "present"
# frames, treat it as present too, so one fill isn't split in two.
p = present.copy()
for i in range(1, len(p) - MAX_GAP):
    if not p[i] and p[i - 1] and p[i + MAX_GAP]:
        p[i] = True

# Find runs of consecutive "present" frames
fills, start = [], None
for i, val in enumerate(p):
    if val and start is None:
        start = i
    if (not val or i == len(p) - 1) and start is not None:
        end = i - 1 if not val else i
        if end - start + 1 >= MIN_LEN:
            fills.append((start, end))
        start = None

rows = []
for n, (s, e) in enumerate(fills, 1):
    seg = df.iloc[s:e + 1]
    # "Finished vial" frame: 2 frames before the end (the last frame may be
    # motion blur as the vial leaves)
    rep_i = max(s, e - 2)
    rep = df.iloc[rep_i]
    # Typical color near the end of the fill (last 4 frames before leaving)
    end_hue = df.iloc[max(s, e - 4):e]["hue"].median()
    rows.append({
        "fill": n,
        "start_s": round(seg["video_s"].iloc[0], 1),
        "end_s": round(seg["video_s"].iloc[-1], 1),
        "duration_s": round(seg["video_s"].iloc[-1] - seg["video_s"].iloc[0], 1),
        "peak_frac": round(seg["color_frac"].max(), 3),
        "end_hue": round(end_hue, 1) if pd.notna(end_hue) else None,
        "rep_file": rep["filename"],
        "touches_video_edge": s == 0 or e == len(df) - 1,
    })

fills_df = pd.DataFrame(rows)
fills_df.to_csv(f"{OUT}/fills_detected.csv", index=False)
pd.set_option("display.width", 200)
print(fills_df.drop(columns=["rep_file"]).to_string(index=False))
print(f"\nDetected {len(fills_df)} fills (labels say 30).")

# Picture of each fill's finished vial, numbered
cols, w, h = 6, 320, 240
nrows = max(1, (len(fills_df) + cols - 1) // cols)
sheet = Image.new("RGB", (cols * w, nrows * h), "black")
for i, r in fills_df.iterrows():
    im = Image.open(f"{DATA}/images/{r['rep_file']}").convert("RGB")
    ImageDraw.Draw(im).rectangle(ROI, outline="red", width=3)
    im = im.resize((w, h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 18], fill="black")
    d.text((4, 3), f"fill {r['fill']} | {r['start_s']}-{r['end_s']}s | hue {r['end_hue']}", fill="yellow")
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(f"{OUT}/fills_sheet.jpg", quality=90)
print(f"Saved {OUT}/fills_sheet.jpg")
