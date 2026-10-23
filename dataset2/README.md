# Dataset 2 (Fluid Mixing): corrected labels + carryover baseline

## Key finding: the original video-5 labels are misaligned
`video5_frame_labels.csv` time windows drift from the vial_fill images (start ~10 s early at vial 7,
~40 s early by vial 9, growing over time). Detecting fills directly from the images shows
**video 5 contains vials 1-25 only** (vial 25 cut off at the end); vials 26-30 are not in the video.

**Use `results/video5_labels_corrected.csv` instead** (vial, recipe, tank volumes, real start/end).
Video 5 starts at 2026-09-23 19:47:10.

## Scripts (run on the Spark, in this order)
1. `contact_sheet.py`: last labeled frame per vial (showed the mismatch)
2. `check_alignment.py`: frames + labels in a time window (`python3 check_alignment.py 195230 195600`)
3. `color_timeline.py`: colored-pixel fraction + hue per frame in the vial ROI (230,330,380,440)
4. `segment_fills.py`: finds fills from the images (25 detected)
5. `carryover_baseline.py`: corrected labels + batch-consistency detector + predictive check

Paths assume the dataset at `/home/yt-thd0812/workspace/data/csi-vial-fill-dataset`.

## First results (23 scored vials, 7 carryover = first vial after a recipe change)
| Threshold | Caught | Missed | False alarms |
|---|---|---|---|
| hue_dev > 15 | 7/7 | 0% | 3 of 16 (19%) |
| hue_dev > 20 | 6/7 | 14% | 2 of 16 (12%) |

Recipe-change size vs observed color shift in the first vial: r about 0.88 (7 transitions).

## Limitations / next
- Batch check needs all 3 vials (after the fact). Next: real-time detector predicting expected
  color from tank mix, tested with leave-one-recipe-out.
- Ground truth assumes the first vial of each recipe is the carryover vial.
- Dark/pale vials give noisy hue.
