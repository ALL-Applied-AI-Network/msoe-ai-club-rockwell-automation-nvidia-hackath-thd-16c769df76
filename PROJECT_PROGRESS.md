# Project progress

## Problem framing

The team selected Fluid Mixing Station 3 and focused on carryover: residual liquid from the previous recipe can make the first vial after a transition visibly inconsistent. The objective became an operator-facing, advisory system that warns before the transition and checks the result during filling.

## Data and alignment

- Audited the supplied Dataset 5 video, frame labels, tank levels, and recipe queue.
- Found timestamp drift in the original frame labels and produced corrected fill windows.
- Confirmed 24 complete scored vials plus one cut-off final vial.
- Built hue/color baselines and an interpretable first-vial-after-change rule.
- Kept the unrelated tray-inspection data out of this task after verifying that its camera view and labels do not match vial filling.

## Synthetic data

- Built a deterministic 30 fps vial-fill renderer using the reference station geometry.
- Added liquid palette variation, recipe mixes, clean/mild/medium/severe carryover, lighting/camera nuisance factors, and reproducible seeds.
- Froze v4.0.0 at 480 scenarios: 320 train, 80 validation, and 80 test.
- Split by transition and palette families to prevent leakage.
- Generated checksums, manifests, per-frame labels, per-vial labels, and QA summaries.
- Kept the multi-gigabyte raw corpus outside Git; this repository contains the plan, tools, metadata, and representative samples.

## Model and rules

- Trained a compact multi-input neural network on image crops plus recipe context.
- Exported PyTorch, TorchScript, and ONNX artifacts locally; weights are excluded from Git per repository guidance.
- Added a neural shadow mode alongside the original recipe/color rules.
- Kept neural alerts advisory and recorded per-frame probabilities for audit.
- Evaluated against corrected Dataset 5 without modifying the frozen model weights.

## Evaluation

- The interpretable batch rule found all seven expected first-vial carryover cases in the corrected replay.
- The frozen neural model ranked Dataset 5 cases with ROC AUC 0.839.
- Leave-one-recipe-out thresholding achieved 71.4% recall, 75.0% specificity, and 73.2% balanced accuracy on 23 vials.
- Mean inference latency was 4.1 ms with 4.9 ms p95 on the Spark.
- The fixed synthetic threshold did not transfer directly to real Dataset 5, demonstrating the need for real-data calibration.

## Real-time operator interface

- Built a token-protected dashboard that binds to localhost.
- Added the entire recipe queue, active/next recipe, live model probability, event timeline, and distinct future-risk versus observed-defect pop-ups.
- Added a live `latest_frame` camera panel with model and color-rule overlays.
- Added 20 ms folder polling with file-stability checks for a 30 fps camera folder.
- Replayed the corrected Dataset 5 recording through the actual frozen model at original timing and verified the live dashboard end to end.
- Split the operator interface into a stable `/demo` saved-run presentation and a separate `/live` CSI testing workspace.
- Added dashboard recipe entry/CSV import, expected-color calculation, validation, timestamped queue snapshots, and guarded start/pause/resume/stop/realign controls for the advisory monitor only.

## Deployment boundary and next steps

- The CSI machine was unavailable during the planned physical test, so no production claim is made.
- Identify the line-attached Spark's camera source and add an RTSP or V4L2 adapter if it is not a JPEG folder.
- Collect labeled real fills across recipes and lighting conditions.
- Calibrate thresholds on a separate real validation set and retain a final untouched test set.
- Measure camera-to-alert latency and operator usefulness at CSI.
- Replace the manual queue with a read-only PLC/MES recipe feed for production deployment.


