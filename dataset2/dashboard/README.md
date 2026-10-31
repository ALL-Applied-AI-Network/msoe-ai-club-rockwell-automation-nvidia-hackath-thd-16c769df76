# Carryover operator dashboard

This is a token-protected browser interface for `live_monitor.py`. It has two deliberately separate views:

- `/demo` is the stable saved-run presentation view.
- `/live` is the CSI test view for entering/importing a recipe queue and managing the advisory camera/model monitor.

Both views show the latest camera frame with the model and color-analysis regions, recipe queue, active vial and recipe, latest neural probability, and recent event timeline. New events produce operator pop-ups:

- `PREDICT`: amber future carryover-risk warning
- `MODEL`: red neural shadow warning
- `DETECT`: red unexpected-color warning during the pour
- `CONFIRM`: red confirmed carryover warning

Acknowledging a pop-up only closes it in that browser. Live controls can start, pause, resume, realign, and stop the monitor container, but never connect to or operate the PLC or physical line. The physical machine must be controlled separately under CSI procedures.

## Run it

Start `live_monitor.py` first so it writes `events.jsonl` and `neural_predictions.jsonl`. In another terminal, run:

```bash
python3 dashboard/dashboard_server.py \
  --queue /path/to/output/queue.csv \
  --events /path/to/output/events.jsonl \
  --neural-log /path/to/output/neural_predictions.jsonl \
  --latest-frame /path/to/output/latest_frame.jpg
```

The server prints a newly generated login token. It binds to `127.0.0.1:20400` by default.

## Real-time camera mode

Production and CSI testing should use the monitor's `watch` source, not the
recorded-log replay helper. The watcher processes each new JPEG as the camera
writes it and updates the dashboard JSONL files immediately. Before starting,
prepare the manually entered queue CSV and identify the live camera's JPEG
folder on the line-attached Spark.

The included launcher runs the frozen neural model in the NVIDIA PyTorch
container and watches for complete files every 20 ms:

```bash
dashboard/start_live_monitor.sh \
  /path/to/live/vial_fill/images \
  /path/to/manual_queue.csv \
  /path/to/live/carryover-monitor
```

Point `dashboard_server.py` at the same queue and the live output files:

```bash
python3 dashboard/dashboard_server.py \
  --queue /path/to/manual_queue.csv \
  --events /path/to/live/carryover-monitor/events.jsonl \
  --neural-log /path/to/live/carryover-monitor/neural_predictions.jsonl \
  --latest-frame /path/to/live/carryover-monitor/latest_frame.jpg
```

To enable the `/live` controls, provide one fixed, allowlisted live profile:

```bash
python3 dashboard/dashboard_server.py \
  --queue /path/to/saved-demo/queue.csv \
  --events /path/to/saved-demo/events.jsonl \
  --neural-log /path/to/saved-demo/neural_predictions.jsonl \
  --latest-frame /path/to/saved-demo/latest_frame.jpg \
  --live-root /home/USER/workspace/live/operator-runs \
  --live-workspace /home/USER/workspace \
  --live-code-dir /home/USER/workspace/code/PROJECT/dataset2/code \
  --live-camera-dir /home/USER/workspace/live/camera-jpegs \
  --live-model /home/USER/workspace/models/carryover/model.ts \
  --live-tank-colors /home/USER/workspace/models/carryover/tank_colors.npy \
  --live-image nvcr.io/nvidia/pytorch:25.10-py3
```

The live page validates vial order, volumes, totals, and recipe IDs; computes expected RGB/hue from the tank calibration; and saves an immutable queue snapshot into each timestamped run folder. `Realign next vial` stops the advisory monitor and starts a new preserved segment aligned to the selected vial.

At 30 fps, three-frame persistence adds roughly 0.1 seconds. Folder ingest,
model inference, and the dashboard's 750 ms refresh should normally keep an
alert within about one second of the qualifying camera frames. Actual latency
must be measured on the CSI camera path.

To view a Spark-hosted dashboard safely from a PC, create an SSH tunnel from the PC:

```powershell
ssh -L 20400:127.0.0.1:20400 USER@SPARK_HOST
```

Then open `http://127.0.0.1:20400` on the PC and enter the printed dashboard token. Keep the SSH session open while using the dashboard.

For a different machine, substitute its SSH login. Do not bind the dashboard to a shared network interface unless CSI has approved the exposure and an appropriate protected access method is in place.

## Inputs

The queue needs one row per vial and should include:

```text
vial,recipe_id,t100_ml,t200_ml,t400_ml,total_ml
```

Optional expected-color columns from the monitor are also accepted. The monitor atomically publishes `latest_frame.jpg` and its metadata beside the JSONL files. The dashboard tolerates missing or partially written inputs and updates every 750 ms.

## Security and operating boundary

- Token login is required.
- State-changing live requests require a same-session control token.
- The default bind address is localhost.
- Monitor commands use a fixed container profile; arbitrary shell commands are not accepted.
- No PLC endpoints or physical equipment-control actions exist in this server.
- Model warnings remain advisory and require operator judgment.

