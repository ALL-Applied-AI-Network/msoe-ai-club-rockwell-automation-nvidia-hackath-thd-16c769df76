# Carryover operator dashboard

This is a token-protected browser interface for `live_monitor.py`. It has two deliberately separate views:

- `/demo` is the stable saved-run presentation view.
- `/live` is the CSI test view for entering/importing a recipe queue and managing the advisory camera/model monitor.

Both views show the latest camera frame with the model and color-analysis regions, recipe queue, active vial and recipe, latest neural probability, and recent event timeline. New events produce operator pop-ups:

- `PREDICT`: amber, non-blocking future carryover-risk banner
- `MODEL`: purple, non-blocking neural model watch
- `DETECT`: red unexpected-color warning during the pour
- `CONFIRM`: red confirmed carryover warning

Only `DETECT` and `CONFIRM` open acknowledgement modals. Acknowledging a modal only closes it in that browser. Live controls can start, pause, resume, realign, and stop the monitor container, but never connect to or operate the PLC or physical line. The physical machine must be controlled separately under CSI procedures. The neural shadow uses a recall-prioritized threshold of `0.0071` with three qualifying frames in the latest five-frame window. This intentionally favors catching possible defects over minimizing false alarms.

## On-demand Dataset 5 demo

The saved-run page can optionally start Dataset 5 as a fresh, real-time (1x)
replay on the Spark. This is intended for demonstrations away from CSI. Every
start creates a timestamped output folder and streams its current frame,
predictions, detections, and recipe position into `/demo`. The fixed profile can
be paused, resumed, restarted, or stopped from the page; it cannot run arbitrary
commands or interact with the PLC.

Enable it by adding the frozen replay inputs when starting the server:

```bash
python3 dashboard/dashboard_server.py \
  ...saved-run arguments... \
  --demo-root /home/USER/workspace/live/on-demand-dataset5 \
  --demo-workspace /home/USER/workspace \
  --demo-code-dir /home/USER/workspace/code/PROJECT/dataset2/code \
  --demo-data-root /home/USER/workspace/data/csi-vial-fill-dataset/vial_fill \
  --demo-model /home/USER/workspace/models/carryover/model.ts \
  --demo-tank-colors /home/USER/workspace/models/carryover/tank_colors.npy
```

The demo uses the server's saved-run queue unless `--demo-queue` is supplied.
The page disables Start and identifies any missing fixed input. Dataset 5 takes
about 28 minutes at live speed; navigation away from the page does not stop it.

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

### Raw USB preview

The dashboard can show the local vial-fill USB camera before the neural model
and calibration artifacts are available. The adapter alone is capture/display
only: it does not make carryover decisions, write to the PLC, change camera
controls, or enable the analysis controls. Its bounded output can separately
feed the monitor when the same directory is selected as `--live-camera-dir`.

```bash
python3 dashboard/dashboard_server.py \
  --queue /path/to/saved-demo/queue.csv \
  --events /path/to/saved-demo/events.jsonl \
  --live-root /path/to/runtime/live-control \
  --live-camera-device /dev/v4l/by-id/usb-CAMERA-video-index0
```

The server uses FFmpeg to read MJPEG at 640x480 and publishes a five-frame-per-second
preview atomically. It restarts FFmpeg if the capture process exits. Keep the
dashboard bound to `127.0.0.1`; the frame endpoint remains behind the dashboard
login.

The **Start live monitoring** button still requires the model, tank-color calibration,
camera JPEG folder, and container settings described below. This keeps a raw camera
preview from being mistaken for an active carryover analysis.

### Live-monitor readiness

Every runtime path must be inside the single directory passed as
`--live-workspace`, because that is the directory mounted into the analysis
container. A local checkout can use this ignored layout:

```text
REPOSITORY/
  dataset2/code/
  live/
    control/camera_preview/latest_frame.jpg
    models/carryover/model.ts
    models/carryover/tank_colors.npy
```

Start the dashboard with `REPOSITORY` as `--live-workspace`, `live/control` as
`--live-root`, `dataset2/code` as `--live-code-dir`, and
`live/control/camera_preview` as `--live-camera-dir`. Point `--live-model` and
`--live-tank-colors` at the two files above. The camera watcher supports the
atomically replaced `latest_frame.jpg`, so a separate unbounded JPEG spool is
not required.

For the original THD v0.1.0 experiment, verify recovered artifacts before use:

```text
model.ts        f492763a91b9ca76bf03164d5ac8b678294e83513f77b03c5b2e591f0c76b2f5
tank_colors.npy 2cf67e8e5d09a1a7b4e79f9724292dfefdcdcd692eeedd5ff87b72db17829e24
```

The Docker check also must be green. `docker info` must work for the account
running the dashboard, and the NVIDIA runtime must be available. Do not make
the Docker socket world-writable; Docker access is effectively root-level and
should be granted by the machine administrator.

Production and CSI testing should use the monitor's `watch` source, not the
recorded-log replay helper. The watcher processes each new JPEG as the camera
writes it and updates the dashboard JSONL files immediately. Before starting,
prepare the manually entered queue CSV and identify the live camera's JPEG
folder on the line-attached Spark.

The included launcher runs the frozen neural model in the NVIDIA PyTorch
container and watches for complete files every 20 ms. It defaults to
`$CARRYOVER_WORKSPACE/live/models/carryover/{model.ts,tank_colors.npy}`;
override those paths with `CARRYOVER_MODEL` and `CARRYOVER_TANK_COLORS` when
needed. The code, camera folder, queue, artifacts, and output must all be under
the mounted `CARRYOVER_WORKSPACE`.

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
  --live-root /home/USER/workspace/live/control \
  --live-workspace /home/USER/workspace \
  --live-code-dir /home/USER/workspace/code/PROJECT/dataset2/code \
  --live-camera-dir /home/USER/workspace/live/control/camera_preview \
  --live-camera-device /dev/v4l/by-id/usb-CAMERA-video-index0 \
  --live-model /home/USER/workspace/live/models/carryover/model.ts \
  --live-tank-colors /home/USER/workspace/live/models/carryover/tank_colors.npy \
  --live-image nvcr.io/nvidia/pytorch:25.10-py3
```

The live page validates vial order, volumes, totals, recipe IDs, and the station's
documented operating limits: no more than 80 mL total, no more than 50 mL from
T-200, at least 10 mL from each tank that is used, and no T-300. It computes
expected RGB/hue from the tank calibration and saves an immutable queue
snapshot into each timestamped run folder. `Realign next vial` stops the
advisory monitor and starts a new preserved segment aligned to the selected
vial.

The raw-camera adapter publishes five frames per second by default. When its
preview directory is also used as `--live-camera-dir`, the separate monitor
analyzes that bounded stream; the earliest three threshold hits in a five-frame
window span roughly 0.4 seconds, and the full window spans roughly 0.8 seconds.
Folder ingest, model inference, and the dashboard's 750 ms refresh add to that
time. The adapter itself remains capture/display only and does not make
carryover decisions. Actual end-to-end latency must be measured on the CSI
camera path.

The monitor copies each accepted version into a bounded three-frame temporary
ring before measurement. Color rules, dashboard publishing, and neural
inference therefore see the same immutable bytes even when the camera adapter
replaces one filename. The snapshots are removed automatically and do not form
an unbounded image archive.

Keep the 5 fps default when one atomically replaced `latest_frame.jpg` feeds
the monitor. The watcher deliberately waits for an unchanged file signature
across two polls; higher fixed-path rates can replace a version before it is
observed twice. A higher-rate integration should use immutable filenames or a
bounded ring/spool instead.

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
