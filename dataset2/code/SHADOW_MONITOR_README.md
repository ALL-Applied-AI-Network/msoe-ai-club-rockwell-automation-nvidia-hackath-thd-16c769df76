# Neural shadow monitor

`live_monitor.py` keeps the original `PREDICT -> DETECT -> CONFIRM` rules and optionally runs the frozen neural model alongside them. Neural alerts are advisory only and never control the PLC.

Current provisional configuration is recorded in `shadow_config.json`: recall-prioritized probability threshold `0.0071`, a five-visible-frame rolling window, and at least three frames at or above threshold before an alert. This operating point intentionally accepts more false positives to reduce missed defects.

Example replay:

```bash
python3 live_monitor.py \
  --source replay --speed 0 \
  --queue /path/to/queue.csv \
  --data-root /path/to/vial_fill \
  --out-dir /path/to/run-output \
  --model /path/to/model.ts \
  --model-threshold 0.0071 \
  --model-window 5 \
  --model-min-frames 3 \
  --tank-colors /path/to/tank_colors.npy
```

The recipe queue may contain `expected_r`, `expected_g`, and `expected_b` in either 0-1 or 0-255 form. Corrected Dataset 5 queues may instead contain leave-one-recipe-out `pred_R`, `pred_G`, and `pred_B`. If neither is present, the monitor can use the optional tank-color calibration; without any expected-color source it sets queue confidence to zero and uses the model's image-only fallback.

Outputs:

- `events.jsonl`: rule and advisory model alerts
- `neural_predictions.jsonl`: every neural frame prediction, smoothing state, expected-color source, and latency

The Dataset 5 tuning figures are development results, not final deployment accuracy. Recalibrate on a designated CSI calibration subset, then score a separate untouched CSI evaluation subset.
