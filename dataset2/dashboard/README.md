# Carryover operator dashboard

This is a read-only browser interface for `live_monitor.py`. It shows the latest camera frame with the model and color-analysis regions, recipe queue, active vial and recipe, latest neural probability, and recent event timeline. New events produce operator pop-ups:

- `PREDICT`: amber future carryover-risk warning
- `MODEL`: red neural shadow warning
- `DETECT`: red unexpected-color warning during the pour
- `CONFIRM`: red confirmed carryover warning

Acknowledging a pop-up only closes it in that browser. The dashboard never writes to the PLC, changes the queue, or changes a model decision.

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
- The default bind address is localhost.
- No PLC endpoints or equipment-control actions exist in this server.
- Model warnings remain advisory and require operator judgment.
