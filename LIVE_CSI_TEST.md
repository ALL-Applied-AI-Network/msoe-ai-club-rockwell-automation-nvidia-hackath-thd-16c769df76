# Live CSI Test

## Scope

On 2026-11-03 UTC, the team completed a local live-readiness test of the
carryover dashboard, USB camera adapter, containerized GPU runtime, and advisory
model pipeline. This was a preflight test for CSI use, not an untouched
production-line trial. No PLC connection was made, no machine command was sent,
and no recipe was changed by the software.

This record is intentionally sanitized. It contains no account names,
passwords, authentication tokens, host IP addresses, private workspace paths,
or camera serial identifiers.

## Safety and data boundaries

- The dashboard and monitor remained advisory-only.
- The web server bound to `127.0.0.1` and required a freshly generated login
  token. The token was supplied only at runtime and was not written to Git.
- The recipe queue was validated before startup against the documented station
  limits: no more than 80 mL total, no more than 50 mL from T-200, at least
  10 mL from every tank used, and no T-300.
- Model artifacts, calibration files, captured frames, run outputs, and control
  audit logs stayed under the ignored local `live/` directory.

## Preparation

1. Confirmed the USB vial-fill camera through its stable V4L2 by-id device path.
   Documentation and source use a placeholder such as
   `/dev/v4l/by-id/<camera-device>` rather than a host-specific identifier.
2. Verified that the frozen TorchScript model and tank-color calibration
   matched their documented SHA-256 checksums without adding either artifact to
   Git.
3. Pulled and cached `nvcr.io/nvidia/pytorch:25.10-py3` locally.
4. Confirmed Docker GPU passthrough on the NVIDIA GB10.
5. Verified that the saved nine-vial queue passed all dashboard safety checks.

## GPU and model validation

The first direct CUDA load exposed a unified-memory startup issue: the JIT
loader reported an out-of-memory error even though the model itself was small.
The loader was changed to deserialize the model on CPU and then transfer only
the loaded model to the selected CUDA device.

After that change, an offline container smoke test passed with:

- CUDA available on the NVIDIA GB10;
- the model and calibration checksums unchanged;
- a finite `3 x 3` tank-color calibration;
- image input shape `1 x 3 x 160 x 160`;
- recipe input shape `1 x 10`; and
- output shapes `1 x 3` for risk, `1` for observed carryover, and `1 x 4` for
  severity.

The container used no network access for this model check, and the repository
was mounted read-only.

## Live dashboard run

The dashboard was started with:

- a localhost-only listener and runtime-generated login token;
- the USB camera adapter at 640 x 480;
- a bounded five-frame-per-second atomic JPEG preview;
- the validated nine-vial queue;
- the ignored local model and calibration artifacts; and
- the cached NVIDIA PyTorch container image.

The authenticated readiness endpoint reported that the camera, model,
calibration, code, workspace, Docker runtime, and queue were configured. The
camera preview was running without restarts and published a fresh frame.

After a confirmed advisory start from vial 1:

- the monitor container remained running with a GPU device request;
- all nine queue rows loaded;
- neural shadow mode initialized successfully;
- the monitor watched the camera folder every 20 ms;
- the monitor-owned frame was about 120 ms old at the sampled dashboard update;
  and
- `latest_frame.jpg`, `latest_frame.json`, and the event heartbeat continued to
  refresh inside the run directory.

An empty neural-prediction log was expected because no colored vial fill was
performed during this readiness test. Independent GPU inference had already
been verified by the offline model smoke test.

## Saved local evidence

Every confirmed start creates a unique ignored directory at:

```text
live/control/runs/<run-id>/
```

The run directory preserves:

- `queue.csv`, the immutable validated queue snapshot;
- `events.jsonl`, including its frame-ingest heartbeat;
- `neural_predictions.jsonl`;
- `latest_frame.jpg`; and
- `latest_frame.json`.

Start, pause, resume, realign, and stop actions are also recorded in the ignored
local control audit. The current implementation keeps the latest analyzed frame
and structured event records; it does not record a full session video or retain
every camera frame.

## Shutdown verification

The monitor was stopped through the authenticated dashboard API before the
dashboard process was interrupted. Final checks confirmed that:

- the temporary monitor container was removed;
- no FFmpeg camera process remained; and
- the localhost dashboard port was closed.

The preserved run directory was not deleted. The dashboard was later restarted
for operator use without automatically starting the advisory monitor.

## Software validation

After the live test and CUDA loader fix:

- 32 dashboard and live-control unit tests passed;
- 10 camera-watcher and model-loader unit tests passed;
- Python compilation checks passed;
- the launcher passed `bash -n`; and
- `git diff --check` passed.

## Limitations and next test

- No colored liquid was dispensed, so this test did not measure a physical
  carryover alert or camera-to-alert latency.
- No production claim is made.
- The next supervised CSI test should use labeled fills, record operator and
  Teledyne observations, and measure end-to-end alert timing.
- A separate approved recording workflow is required if full-session video is
  needed.

## Explicitly excluded from Git

The following remained local and are not part of this commit:

- model weights and exports (`*.pt`, `*.pth`, `*.onnx`, and `*.ts`);
- tank-color calibration artifacts;
- raw CSI datasets, camera images, and video;
- CSI digital-twin source and generated synthetic corpora;
- `live/`, `runs/`, `outputs/`, and dashboard archives;
- rendered submission media and narration sources;
- login and CSRF tokens, passwords, SSH keys, and service credentials; and
- usernames, private host paths, IP addresses, and camera serial identifiers.
