# Real-time fluid carryover prediction and detection

Team THD's Rockwell Automation + NVIDIA Fall 2026 hackathon project turns the CSI Fluid Mixing Station 3 camera and recipe queue into an advisory quality monitor. It warns operators before a recipe transition is likely to produce carryover, checks the vial while it fills, and shows the evidence in a real-time dashboard.

The system is intentionally read-only: it never writes to the PLC or controls the line.

## What we built

1. **Recipe-transition risk:** a manually entered queue supplies the next recipe until a production PLC queue feed is available. Recipe changes generate an early low/medium/high carryover warning.
2. **Real-time visual check:** a frozen neural model and an interpretable color rule inspect each vial-fill frame.
3. **Operator dashboard:** the current recipe, queue, live camera frame, analysis regions, neural probability, event timeline, and advisory pop-ups update together.
4. **Synthetic-data pipeline:** reproducible render plans vary recipes, carryover severity, palettes, lighting, camera conditions, and nuisance factors without leaking transitions or palettes across splits.
5. **Evaluation pipeline:** corrected Dataset 5 timing, checksums, frozen thresholds, leave-one-recipe-out calibration, and per-frame/per-vial outputs keep exploratory and test results distinct.

## Current evidence

| Evaluation | Result |
|---|---:|
| Corrected Dataset 5 vials scored | 23 |
| Expected carryover vials | 7 |
| Interpretable batch rule | 7/7 expected carryover vials found |
| Integrated shadow pipeline at threshold 0.0265 | 6 TP, 2 FP, 1 FN, 14 TN |
| Integrated recall | 85.7% |
| Integrated specificity | 87.5% |
| Integrated precision | 75.0% |
| Integrated accuracy | 87.0% |
| Frozen neural ranking on Dataset 5 | ROC AUC 0.839 |
| Mean / p95 neural inference latency on Spark | 4.1 / 4.9 ms |

The integrated results use a three-of-five-frame persistence rule at the balanced evaluation threshold of `0.0265`. They are the values reported in the executive summary. The demonstration dashboard uses a separate sensitivity setting, so its warnings should not be assigned the balanced-threshold accuracy above.

These are prototype results, not a production acceptance claim. Dataset 5 ground truth uses expected carryover by batch position rather than an independent lab assay, and only 23 vials are scored. The `0.0265` threshold was selected after Dataset 5 analysis, so its performance is descriptive rather than an unbiased untouched-test result. Additional calibration experiments remain documented in the [model card](dataset2/MODEL_CARD.md) for technical review.

## Repository map

- [`dataset2/code`](dataset2/code): baseline, live monitor, and neural shadow integration
- [`dataset2/dashboard`](dataset2/dashboard): token-protected, localhost-only operator dashboard
- [`dataset2/model`](dataset2/model): training and evaluation code
- [`dataset2/model_artifacts`](dataset2/model_artifacts): configs, checksums, metrics, and predictions (weights excluded)
- [`dataset2/synthetic_data`](dataset2/synthetic_data): generation tools, frozen production plan, metadata, and compact samples
- [`dataset2/results`](dataset2/results): corrected Dataset 5 labels and baseline visualizations
- [`PROJECT_PROGRESS.md`](PROJECT_PROGRESS.md): chronological implementation summary and remaining work
- [`dataset2/DATASET_CARD.md`](dataset2/DATASET_CARD.md): provenance and data limitations
- [`dataset2/MODEL_CARD.md`](dataset2/MODEL_CARD.md): model scope, evaluation, and safety boundary
- [`submission`](submission): executive summary, video shot list, presentation visuals, and final-video notes

## Submission materials

- [`Executive summary (PDF)`](submission/Team_THD_Carryover_Monitor_Executive_Summary.pdf)
- [`Editable executive summary (DOCX)`](submission/Team_THD_Carryover_Monitor_Executive_Summary.docx)
- [`Video shot list`](submission/VIDEO_SHOT_LIST.md)
- [`Submission and final-video notes`](submission/README.md)

The final rendered video, original CSI recordings, and editable visual deck are intentionally stored outside Git as media-authoring artifacts. The repository contains the reproducible source, measured results, executive summary, and video plan.

## Quick start: recorded replay

The monitor expects the original Dataset 5 `vial_fill` folder, a queue CSV, and the frozen TorchScript artifact produced by `dataset2/model/train.py`.

```bash
python3 dataset2/code/live_monitor.py \
  --source replay \
  --speed 1 \
  --data-root /path/to/vial_fill \
  --queue /path/to/queue.csv \
  --out-dir /path/to/live-output \
  --model /path/to/model.ts \
  --tank-colors /path/to/tank_colors.npy
```

Then start the dashboard:

```bash
python3 dataset2/dashboard/dashboard_server.py \
  --queue /path/to/queue.csv \
  --events /path/to/live-output/events.jsonl \
  --neural-log /path/to/live-output/neural_predictions.jsonl \
  --latest-frame /path/to/live-output/latest_frame.jpg
```

The server binds to `127.0.0.1:20400` by default and prints a login token. Use an SSH tunnel for remote viewing; do not expose it directly to a shared network.

## Live CSI path

`dataset2/dashboard/start_live_monitor.sh` watches a camera JPEG folder with file-stability checks, runs the frozen model in the NVIDIA PyTorch container, and feeds the same dashboard. The exact line-camera source still needs to be identified and latency-tested on the physical CSI system. An adapter will be required if the camera provides RTSP or `/dev/video` instead of JPEG files.

## Reproducing the model

Model weights are excluded in accordance with the hackathon repository guidance. The committed training code, frozen configuration, source/artifact checksums, metrics, and prediction files document the experiment. Run `dataset2/model/train.py` against the generated v4.0.0 corpus to reproduce the weight artifacts.

## Safety boundary

- Advisory only; no PLC write or equipment-control endpoint exists.
- Operators remain responsible for flush, divert, and scrap decisions.
- Real-machine calibration is required before deployment.
- Alerts from synthetic training alone must not be treated as validated production defects.

## Hackathon

This work addresses the [Fluid Mixing Station 3 challenge](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing) in the Rockwell Automation + NVIDIA Innovation Lab Fall 2026 hackathon.


