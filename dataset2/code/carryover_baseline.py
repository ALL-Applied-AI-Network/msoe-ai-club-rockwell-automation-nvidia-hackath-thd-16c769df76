# carryover_baseline.py
# 1) Corrected video-5 labels from image-detected fills
# 2) Batch-consistency carryover detector + metrics (false alarms / misses)
# 3) Predictive check: does recipe-change size predict carryover size?

import pandas as pd, numpy as np, datetime as dt

DATA = "/home/yt-thd0812/workspace/data/csi-vial-fill-dataset/vial_fill"
OUT  = "/home/yt-thd0812/workspace/out"

fills   = pd.read_csv(f"{OUT}/fills_detected.csv")
recipes = pd.read_csv(f"{DATA}/labels/video5_vial_recipes.csv")
tl      = pd.read_csv(f"{OUT}/color_timeline.csv").sort_values("video_s")

# ---------- 1) Corrected labels ----------
# Clock time of video start = first image time minus its video_s
f0 = tl.iloc[0]
d, hms, us = f0["filename"].split("_")[-3:]
t_img = dt.datetime.strptime(d + hms + us.split(".")[0], "%Y%m%d%H%M%S%f")
t0 = t_img - dt.timedelta(seconds=float(f0["video_s"]))

rec = recipes.drop_duplicates("recipe_id").set_index("recipe_id")[
    ["t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order"]].fillna({"t100_ml":0,"t200_ml":0,"t400_ml":0})

lab = fills.copy()
lab["vial"]   = lab["fill"]                        # fill N = vial N (verified by color)
lab["recipe_id"] = 5 + (lab["vial"] - 1) // 3      # README rule
lab["pos_in_batch"] = (lab["vial"] - 1) % 3 + 1    # 1 = first vial of the recipe
lab = lab.join(rec, on="recipe_id")
lab["start_clock"] = [(t0 + dt.timedelta(seconds=s)).strftime("%H:%M:%S") for s in lab["start_s"]]
lab["end_clock"]   = [(t0 + dt.timedelta(seconds=s)).strftime("%H:%M:%S") for s in lab["end_s"]]
lab["expected_carryover"] = (lab["pos_in_batch"] == 1) & (lab["vial"] > 1)
lab["scored"] = (lab["vial"] > 1) & (~lab["touches_video_edge"])
lab.to_csv(f"{OUT}/video5_labels_corrected.csv", index=False)
print(f"Video start clock: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Wrote corrected labels for {len(lab)} vials -> {OUT}/video5_labels_corrected.csv\n")

# ---------- 2) Batch-consistency detector ----------
def hue_diff(a, b):
    """Hue is a circle (0-255 wraps around), so take the shorter way round."""
    x = abs(a - b)
    return np.minimum(x, 256 - x)

batch_med = lab.groupby("recipe_id")["end_hue"].median()
lab["batch_hue"] = lab["recipe_id"].map(batch_med)
lab["hue_dev"]   = hue_diff(lab["end_hue"], lab["batch_hue"]).round(1)

S = lab[lab["scored"]]
print("Per-vial view (scored vials):")
print(S[["vial","recipe_id","pos_in_batch","end_hue","batch_hue","hue_dev","expected_carryover"]].to_string(index=False))

print("\nThreshold sweep (hue_dev > T means 'flag as carryover'):")
print(f"{'T':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  {'miss rate':>9} {'false-alarm rate':>16}")
for T in [10, 15, 20, 25, 30, 40, 50]:
    flag  = S["hue_dev"] > T
    truth = S["expected_carryover"]
    TP = int(( flag &  truth).sum()); FN = int((~flag &  truth).sum())
    FP = int(( flag & ~truth).sum()); TN = int((~flag & ~truth).sum())
    miss = FN / max(TP + FN, 1); fa = FP / max(FP + TN, 1)
    print(f"{T:>4} {TP:>3} {FP:>3} {FN:>3} {TN:>3}  {miss:>9.0%} {fa:>16.0%}")

# ---------- 3) Predictive check ----------
# Recipe "color mix" = share of orange (T200), blue (T400), clear (T100)
mix = rec[["t200_ml", "t400_ml", "t100_ml"]].div(rec["total_ml"], axis=0)
print("\nRecipe changes: predicted risk vs observed shift in the first vial")
print(f"{'prev->new':>10} {'change_score':>12} {'observed_hue_shift':>18}")
for r in lab[lab["expected_carryover"] & lab["scored"]].itertuples():
    prev, new = r.recipe_id - 1, r.recipe_id
    score = 0.5 * np.abs(mix.loc[prev] - mix.loc[new]).sum()   # 0 = same mix, 1 = totally different
    print(f"{prev:>4}->{new:<4} {score:>12.2f} {r.hue_dev:>18.1f}")
