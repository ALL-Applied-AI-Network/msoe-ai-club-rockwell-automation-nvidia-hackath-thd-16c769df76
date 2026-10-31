# Recipe-conditioned carryover model v1

This practical first version preserves the existing `PREDICT -> DETECT -> CONFIRM` monitor and adds a compact neural model with three outputs:

- **Risk:** low, medium, or high recipe-transition risk before filling.
- **Observed error:** whether carryover is visibly occurring in the current frame.
- **Severity:** clean, mild, medium, or severe visible carryover.

The risk head receives previous/current recipe fractions and expected color. The observed-error and severity heads combine those recipe features with a cropped FillCam image. They also receive explicit color statistics from the vial's central stripe and the difference between observed and expected color; this makes the physical comparison available even for palettes held out of training. During training, recipe data is randomly masked for some samples so the vision path remains useful when the manually entered queue is missing or out of sync.

## Data safety

Training reads immutable dataset `v4.0.0` only when `READY.json` is present. Selected frames are decoded with FFmpeg into a temporary cache outside the dataset. The cache corrects one derived target: a color mismatch is considered visually observable only when liquid is also visible. Source manifests are never modified.

The frozen train/validation/test split holds out complete recipe transitions and complete palettes. Do not tune using the test metrics repeatedly.

## Training command

```bash
python3 train.py \
  --dataset-root /path/to/synthetic-carryover/v4.0.0 \
  --cache-dir /tmp/carryover-frame-cache \
  --output-dir /path/to/models/carryover_detector/experiment_v1
```

Run `python3 smoke_test.py` before a dataset training job.

## Deployment boundary

Synthetic metrics measure controlled generalization, not real-world accuracy. Calibrate the model on corrected real video 5, keep the rule-based monitor as a fallback, and complete a supervised hardware test before making deployment claims.
