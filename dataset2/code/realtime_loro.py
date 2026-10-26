# realtime_loro.py
# Real-time carryover detector: predict each vial's expected color from its
# recipe mix (learned tank colors), compare the actual vial to the prediction.
# Tested with leave-one-recipe-out (LORO): the recipe being tested is never
# used for learning.

import pandas as pd, numpy as np
from PIL import Image, ImageDraw

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"
ROI  = (230, 330, 380, 440)

lab = pd.read_csv(f"{OUT}/video5_labels_corrected.csv")
tl  = pd.read_csv(f"{OUT}/color_timeline.csv").sort_values("video_s")
lab = lab[lab["vial"] <= 24].copy()        # vial 25 is cut off by the end of the video


def liquid_rgb(fname):
    """Median RGB of the colored (liquid) pixels in the ROI of one image."""
    im  = Image.open(f"{DATA}/images/{fname}").convert("RGB").crop(ROI)
    rgb = np.array(im).reshape(-1, 3).astype(float)
    hsv = np.array(im.convert("HSV")).reshape(-1, 3).astype(float)
    m = (hsv[:, 1] > 60) & (hsv[:, 2] > 30)      # skip white background / gray metal
    if m.sum() < 50:                             # very pale vial: fall back to all pixels
        m = np.ones(len(rgb), bool)
    return np.median(rgb[m], axis=0)


def chroma(rgb):
    """Color proportions with brightness removed: R/(R+G+B), G/(R+G+B)."""
    rgb = np.atleast_2d(rgb)
    return rgb[:, :2] / rgb.sum(axis=1, keepdims=True)


# 1) Actual color of each vial: average of the last 3 frames before it leaves
colors = []
for r in lab.itertuples():
    fr = tl[(tl["video_s"] >= r.start_s) & (tl["video_s"] <= r.end_s)]
    pick = fr.iloc[-4:-1] if len(fr) >= 4 else fr   # skip the very last (motion blur) frame
    colors.append(np.mean([liquid_rgb(f) for f in pick["filename"]], axis=0))
lab[["R", "G", "B"]] = np.array(colors)

# Recipe mix as fractions: [orange (T200), blue (T400), clear (T100)]
X_all = lab[["t200_ml", "t400_ml", "t100_ml"]].div(lab["total_ml"], axis=0).to_numpy()
Y_all = lab[["R", "G", "B"]].to_numpy()

# 2) Leave-one-recipe-out
lab["pred_R"] = np.nan
lab["pred_G"] = np.nan
lab["pred_B"] = np.nan
clean = lab["pos_in_batch"].isin([2, 3]).to_numpy()   # clean vials = positions 2 and 3

for held in sorted(lab["recipe_id"].unique()):
    train = clean & (lab["recipe_id"] != held).to_numpy()   # never learn from the held-out recipe
    W, *_ = np.linalg.lstsq(X_all[train], Y_all[train], rcond=None)
    test = (lab["recipe_id"] == held).to_numpy()
    lab.loc[test, ["pred_R", "pred_G", "pred_B"]] = np.clip(X_all[test] @ W, 0, 255)

P = lab[["pred_R", "pred_G", "pred_B"]].to_numpy()
lab["dist"] = (np.linalg.norm(chroma(Y_all) - chroma(P), axis=1) * 100).round(2)

# Learned tank colors using ALL clean vials (sanity check + saved for the live monitor)
W_all, *_ = np.linalg.lstsq(X_all[clean], Y_all[clean], rcond=None)
np.save(f"{OUT}/tank_colors.npy", W_all)
for name, c in zip(["orange T200", "blue T400", "clear T100"], W_all):
    print(f"learned {name:12s} RGB = {np.clip(c, 0, 255).round(0)}")

# 3) Scores
S = lab[lab["scored"]]
print("\nPer vial (each recipe predicted WITHOUT seeing that recipe):")
print(S[["vial", "recipe_id", "pos_in_batch", "dist", "expected_carryover"]].to_string(index=False))

print("\nThreshold sweep (dist > T means 'flag as carryover'):")
print(f"{'T':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'miss rate':>9} {'false-alarm rate':>16}")
for T in [2, 3, 4, 5, 6, 8, 10, 12]:
    flag, truth = S["dist"] > T, S["expected_carryover"]
    TP = int((flag & truth).sum());  FN = int((~flag & truth).sum())
    FP = int((flag & ~truth).sum()); TN = int((~flag & ~truth).sum())
    print(f"{T:>4} {TP:>3} {FP:>3} {FN:>3} {TN:>3}  {FN/max(TP+FN,1):>9.0%} {FP/max(FP+TN,1):>16.0%}")

lab.to_csv(f"{OUT}/realtime_loro_results.csv", index=False)

# 4) Picture: actual vs predicted color for every scored vial
cols, cw, ch = 6, 220, 120
rows = (len(S) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cw, rows * ch), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(S.itertuples()):
    x, y = (i % cols) * cw, (i // cols) * ch
    tag = "  CARRYOVER?" if r.expected_carryover else ""
    d.text((x + 5, y + 3), f"vial {r.vial} r{r.recipe_id} d={r.dist}{tag}", fill="black")
    d.rectangle([x + 10, y + 20, x + 100, y + 110], fill=tuple(int(v) for v in (r.R, r.G, r.B)))
    d.rectangle([x + 115, y + 20, x + 205, y + 110],
                fill=tuple(int(v) for v in (r.pred_R, r.pred_G, r.pred_B)))
sheet.save(f"{OUT}/realtime_swatches.png")
print(f"\nSaved {OUT}/realtime_swatches.png  (left square = actual, right = predicted)")
