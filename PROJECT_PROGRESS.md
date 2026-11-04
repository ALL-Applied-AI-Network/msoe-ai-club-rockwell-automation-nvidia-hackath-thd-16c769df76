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
- Added 20 ms folder polling with file-stability checks for unique-frame camera folders; the fixed-path V4L2 adapter uses a bounded 5 fps stream.
- Replayed the corrected Dataset 5 recording through the actual frozen model at original timing and verified the live dashboard end to end.
- Split the operator interface into a stable `/demo` saved-run presentation and a separate `/live` CSI testing workspace.
- Added dashboard recipe entry/CSV import, expected-color calculation, validation, timestamped queue snapshots, and guarded start/pause/resume/stop/realign controls for the advisory monitor only.
- Deployed the split-view dashboard beside v2 on Spark localhost port 20401 and verified both views through an independent SSH tunnel; the preserved port-20400 presentation was not changed.
- Added an on-demand Dataset 5 control panel to `/demo` that can launch the frozen model at 1x live speed, stream fresh frames/results into the dashboard, and pause, resume, restart, or stop the isolated replay while preserving every run.
- Reduced operator interruption by presenting neural `MODEL` events as purple non-blocking watches and recipe `PREDICT` events as amber banners; only color `DETECT` and `CONFIRM` evidence opens an acknowledgement modal. The demo and live launchers now use the recall-prioritized `0.0071` threshold with three-of-five persistence, intentionally accepting more false alarms to reduce missed defects.
- Identified the local USB camera's stable V4L2 link and verified the token-protected 640x480 raw preview end to end; the authenticated live-state API reported a fresh frame about 52 ms old.
- Cached the NVIDIA PyTorch runtime and completed a local camera-plus-model readiness run on 2026-11-03. Artifact hashes, CUDA model loading, and a real TorchScript forward pass succeeded on the NVIDIA GB10; the live dashboard then started a nine-vial advisory run and reported a monitor-owned frame about 120 ms old while its frame and event-heartbeat files continued refreshing.
- Fixed GB10 unified-memory startup reliability by loading the small TorchScript model on CPU before moving it to CUDA. The live run was stopped through the authenticated API, and the container, camera process, and dashboard listener all shut down cleanly. No colored fill was performed, so this proves runtime readiness and frame ingestion, not a physical carryover alert or production behavior.

## Deployment boundary and next steps

- The CSI machine was unavailable during the planned physical test, so no production claim is made.
- The local vial-fill source was identified as a USB V4L2 camera. A supervised FFmpeg adapter now publishes a token-protected raw preview and an atomically replaced JPEG that the folder watcher can consume. Camera-to-container ingestion and an independent GPU model forward pass are verified; full camera-to-alert latency still needs to be measured during a labeled colored fill.
- Collect labeled real fills across recipes and lighting conditions.
- Calibrate thresholds on a separate real validation set and retain a final untouched test set.
- Measure camera-to-alert latency and operator usefulness at CSI.
- Replace the manual queue with a read-only PLC/MES recipe feed for production deployment.

## Final submission video

- Recorded nine narration sections totaling about 5 minutes 9 seconds.
- Preserved four CSI line videos, one CSI tank photograph, and a 29-minute 1080p/60 fps dashboard recording under the final-submission source-media folder.
- Built an editable nine-slide visual package covering the architecture, testing data, access limitations, recipe-queue boundary, measured results, manufacturing value, and next steps.
- Selected a dashboard excerpt showing the Recipe 6-to-7 transition, orange future-risk warning, queue movement, live camera view, and red confirmed-carryover alert.
- Rendered a 1920x1080, 30 fps first cut with the narration, CSI footage, synthetic/digital-twin footage, dashboard demonstration, and metric slides.
- Normalized the narration to approximately -16.6 LUFS for reliable presentation playback.
- The video states the physical-line access limitations directly and does not imply an untouched live production trial.

## Executive summary

- Produced a final one-page executive summary for the judges.
- The summary directly answers the required questions about the manufacturing problem, physical AI, CSI equipment, data, evidence, manufacturing value, limitations, and deployment path.
- It reports the balanced-threshold Dataset 5 results separately from the recall-prioritized dashboard setting.
- It states that the team did not complete an untouched live production trial and identifies authorized live-camera and read-only PLC or MES integration as future work.
- The final submission copy is available as a visually verified PDF in the final-submission executive-summary folder.

