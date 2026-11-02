#!/usr/bin/env python3
"""
live_monitor.py - live carryover monitor for the CSI fluid-mixing station (Dataset 2).

Looks at each vial_fill frame the moment it arrives, tracks each vial through
EMPTY -> FILLING -> LEFT STATION, and fires alerts:

  PREDICT  (before the fill)  next recipe in the queue is very different from the
                              current one -> carryover risk for its first vial
  DETECT   (during the fill)  a color shows up that this recipe cannot make
                              (orange with no T-200, blue with no T-400)
  CONFIRM  (~65 s later)      when the 2nd vial of a recipe finishes, compare the
                              1st vial to it -> carryover confirmed, divert/scrap
                              the 1st vial before capping

Frame sources:
  replay : plays video 5's saved frames in time order, as if they were live
  watch  : watches a folder and processes new .jpg files as a camera saves them

Examples:
  python3 live_monitor.py                        # replay video 5 at 20x speed
  python3 live_monitor.py --speed 1              # replay at real speed
  python3 live_monitor.py --speed 0              # replay as fast as possible
  python3 live_monitor.py --source watch --dir /path/to/camera/folder --queue queue.csv

Every alert is also written to out/events.jsonl (one JSON object per line),
which a web dashboard can read.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import time
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image

BASE = os.environ.get("CARRYOVER_WORKSPACE", os.path.expanduser("~/workspace"))
DATA = f"{BASE}/data/csi-vial-fill-dataset/vial_fill"
OUT = f"{BASE}/out"
LOG = f"{OUT}/events.jsonl"

ROI = (230, 330, 380, 440)   # box around the vial bottom (left, top, right, bottom)
THRESH = 0.015               # colored fraction above this = liquid present
MIN_FILL_FRAMES = 3          # color must stay this many frames in a row to count as a fill
                             # (ignores short flickers like drips or reflections)
UNEXPECTED_MIN_FRAC = 0.03   # need at least this much color before judging its hue
CONFIRM_T = 15               # hue difference (0-255 scale) that counts as carryover
RISK_HIGH, RISK_MED = 0.40, 0.25
MODEL_ROI = (200, 190, 420, 480)
LIVE_FRAME_INTERVAL = 0.10   # publish at most 10 dashboard frames per second

ANSI = {"PREDICT": "\033[93m", "DETECT": "\033[91m", "CONFIRM": "\033[91m",
        "MODEL": "\033[95m",
        "OK": "\033[92m", "INFO": "\033[96m"}
RESET = "\033[0m"


# ---------------------------------------------------------------- helpers
def frame_time(path):
    """Capture time from a name like vial_fill_20260923_195325_005000.jpg.
    Falls back to the current time if the name has another format."""
    try:
        d, hms, us = os.path.basename(path).split("_")[-3:]
        return datetime.strptime(d + hms + us.split(".")[0], "%Y%m%d%H%M%S%f")
    except ValueError:
        return datetime.now()


def measure(path):
    """Colored fraction and median hue inside the ROI of one frame."""
    full = Image.open(path).convert("RGB")
    size = full.size
    hsv = np.asarray(full.crop(ROI).convert("HSV")).astype(float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (s > 90) & (v > 40)
    frac = float(colored.mean())
    hue = float(np.median(h[colored])) if colored.any() else float("nan")
    return frac, hue, size


def publish_latest_frame(path, capture_time, size):
    """Atomically publish a dashboard preview without changing the source."""
    latest = os.path.join(OUT, "latest_frame.jpg")
    metadata = os.path.join(OUT, "latest_frame.json")
    suffix = f".{os.getpid()}.tmp"
    image_tmp = latest + suffix
    metadata_tmp = metadata + suffix
    shutil.copyfile(path, image_tmp)
    os.replace(image_tmp, latest)
    payload = {
        "capture_time": capture_time.isoformat(),
        "published_at": datetime.now().isoformat(),
        "source_name": os.path.basename(path),
        "width": size[0],
        "height": size[1],
        "rule_roi": ROI,
        "model_roi": MODEL_ROI,
    }
    with open(metadata_tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(metadata_tmp, metadata)


def hue_diff(a, b):
    """Hue is a circle (0-255 wraps), so take the shorter way round."""
    x = abs(a - b)
    return min(x, 256 - x)


def color_name(h):
    if h is None or np.isnan(h):
        return "clear"
    if h <= 30 or h >= 245:
        return "orange"
    if h <= 60:
        return "olive/yellow"
    if h <= 100:
        return "green"
    if h <= 130:
        return "teal"
    if h <= 170:
        return "blue"
    return "purple"


def mix(r):
    """Recipe as fractions [orange T200, blue T400, clear T100]."""
    return np.array([r["t200_ml"], r["t400_ml"], r["t100_ml"]], float) / float(r["total_ml"])


def change_score(a, b):
    """0 = same mix, 1 = completely different mix."""
    return 0.5 * float(np.abs(mix(a) - mix(b)).sum())


def emit(kind, t, vial, msg, **extra):
    print(f"{ANSI.get(kind, '')}[{t:%H:%M:%S}] {kind:8s} vial {vial}: {msg}{RESET}", flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": t.isoformat(), "kind": kind, "vial": vial,
                            "message": msg, **extra}) + "\n")


# ---------------------------------------------------------------- inputs
def load_queue(path):
    """Recipe queue: one row per vial, in order. Columns:
    vial, recipe_id, t100_ml, t200_ml, t400_ml, total_ml.
    If the file doesn't exist, build it from the corrected video-5 labels."""
    if not os.path.exists(path):
        corrected = f"{OUT}/video5_labels_corrected.csv"
        with open(corrected, newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        columns = ["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml"]
        with open(path, "w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            writer.writerows({name: row[name] for name in columns} for row in rows)
        print(f"(created {path} from the corrected video-5 labels)")
    with open(path, newline="", encoding="utf-8") as handle:
        queue = list(csv.DictReader(handle))
    numeric = {"vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml",
               "expected_r", "expected_g", "expected_b", "pred_R", "pred_G", "pred_B"}
    for row in queue:
        for name in numeric.intersection(row):
            value = row[name]
            row[name] = 0.0 if value is None or value.strip().lower() in {"", "nan"} else float(value)
    return sorted(queue, key=lambda row: float(row["vial"]))


def replay_source(speed):
    """Yield video 5's frames in time order, waiting between them like a live camera."""
    with open(f"{DATA}/image_index.csv", newline="", encoding="utf-8") as handle:
        idx = [row for row in csv.DictReader(handle) if int(float(row["video_id"])) == 5]
    idx.sort(key=lambda row: row["filename"])
    prev = None
    for row in idx:
        f = row["filename"]
        t = frame_time(f)
        if prev is not None and speed > 0:
            time.sleep(max(0.0, (t - prev).total_seconds() / speed))
        prev = t
        yield f"{DATA}/images/{f}", t


def watch_source(folder, poll=0.02):
    """Yield complete new JPEGs without throttling a 30 fps camera.

    A file is considered complete after its size and modification time remain
    unchanged across two polls. This avoids reading a partially written frame
    without adding the old fixed 0.2-second delay to every image.
    """
    seen = set(glob.glob(os.path.join(folder, "*.jpg")))   # ignore files already there
    pending = {}
    print(f"(watching {folder} for new images every {poll * 1000:.0f} ms; Ctrl+C to stop)")
    while True:
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            if p in seen:
                continue
            try:
                stat = os.stat(p)
            except FileNotFoundError:
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            if stat.st_size > 0 and pending.get(p) == signature:
                seen.add(p)
                pending.pop(p, None)
                yield p, frame_time(p)
            else:
                pending[p] = signature
        time.sleep(poll)


# ---------------------------------------------------------------- main loop
def run(source, queue, neural=None, start_index=0):
    # Position of each vial inside its recipe batch (1 = first vial after a change)
    pos = []
    for i, q in enumerate(queue):
        if i == 0 or q["recipe_id"] != queue[i - 1]["recipe_id"]:
            pos.append(1)
        else:
            pos.append(pos[-1] + 1)

    state, i = "EMPTY", start_index - 1
    recent = deque(maxlen=5)          # last few hues of the current vial
    pending = []                      # colored frames seen while EMPTY, not yet a confirmed fill
    end_hue, fill_start, left_at, first_alert = {}, {}, {}, {}
    predicted, detected, confirmed, model_alerted = {}, set(), set(), set()
    cur, unexpected_run, did_detect = None, 0, False
    last_frame_publish = 0.0

    def filling_frame(path, frac, hue, now):
        """One frame of the vial being filled: remember its hue, and DETECT
        colors this recipe cannot make. `now` = time the alert is raised."""
        nonlocal unexpected_run, did_detect
        recent.append(hue)
        if neural is not None:
            previous_recipe = queue[i - 1] if i > 0 else cur
            result = neural.predict(path, now, int(cur["vial"]), previous_recipe, cur)
            if result["new_shadow_alert"]:
                v = int(cur["vial"])
                model_alerted.add(v)
                first_alert.setdefault(v, now)
                emit("MODEL", now, v,
                     f"shadow alert: {result['threshold_hits']}/{result['window_frames']} recent frames "
                     f">= {result['threshold']:.4f}; smoothed probability "
                     f"{result['smoothed_probability']:.4f}; {result['severity_prediction']} indicated. "
                     "Advisory only—no PLC action.",
                     probability=result["smoothed_probability"], threshold=result["threshold"],
                     severity=result["severity_prediction"], recipe_id=int(cur["recipe_id"]),
                     alert_scope="observed", advisory_only=True)
        if did_detect or frac <= UNEXPECTED_MIN_FRAC or np.isnan(hue):
            return
        v = int(cur["vial"])
        c = color_name(hue)
        bad = (c == "orange" and cur["t200_ml"] == 0) or (c == "blue" and cur["t400_ml"] == 0)
        unexpected_run = unexpected_run + 1 if bad else 0
        if unexpected_run >= 2:              # 2 frames in a row, to ignore a single glitch
            did_detect = True
            detected.add(v)
            first_alert.setdefault(v, now)
            emit("DETECT", now, v,
                 f"{c} liquid seen, but recipe {int(cur['recipe_id'])} uses no "
                 f"{'T-200 (orange)' if c == 'orange' else 'T-400 (blue)'}: carryover in progress",
                 hue=hue, observed_color=c, recipe_id=int(cur["recipe_id"]),
                 alert_scope="observed")

    for path, t in source:
        # Keep the dashboard's activity indicator alive even during the normal
        # quiet gap between one vial leaving and the next fill starting.
        os.utime(LOG, None)
        frac, hue, size = measure(path)
        now_monotonic = time.monotonic()
        if now_monotonic - last_frame_publish >= LIVE_FRAME_INTERVAL:
            publish_latest_frame(path, t, size)
            last_frame_publish = now_monotonic

        # ---- EMPTY: wait until color stays for MIN_FILL_FRAMES frames in a row
        if state == "EMPTY":
            if frac <= THRESH:
                pending.clear()              # color vanished again: it was only a flicker
                continue
            pending.append((path, t, frac, hue))
            if len(pending) < MIN_FILL_FRAMES:
                continue

            # a real fill started (at the time of the first colored frame)
            i += 1
            if i >= len(queue):
                emit("INFO", t, "?", "a fill started but the queue has no more vials; stopping")
                break
            state, cur = "FILLING", queue[i]
            recent.clear()
            unexpected_run, did_detect = 0, False
            v = int(cur["vial"])
            t0 = pending[0][1]
            fill_start[v] = t0
            if neural is not None:
                neural.start_vial(v)
            lead = ""
            if v in first_alert:
                lead = f" (warned {(t0 - first_alert[v]).total_seconds():.0f} s before the fill started)"
            emit("INFO", t, v, f"fill started at {t0:%H:%M:%S}, recipe {int(cur['recipe_id'])}{lead}",
                 state="FILLING", recipe_id=int(cur["recipe_id"]))
            for pp, _, pf, ph in pending:        # catch up on the frames we waited for
                filling_frame(pp, pf, ph, t)
            pending.clear()
            continue

        # ---- FILLING
        v = int(cur["vial"])
        if frac > THRESH:
            filling_frame(path, frac, hue, t)
            continue

        # ---- color dropped to ~0: the vial left the station
        state = "EMPTY"
        left_at[v] = t
        hs = [h for h in list(recent)[:-1] if not np.isnan(h)] or [h for h in recent if not np.isnan(h)]
        eh = float(np.median(hs)) if hs else float("nan")
        end_hue[v] = eh
        emit("INFO", t, v, f"left the station, final color {color_name(eh)} (hue {eh:.0f})",
             state="COMPLETED", recipe_id=int(cur["recipe_id"]),
             final_color=color_name(eh), final_hue=eh)

        # CONFIRM: this is the 2nd vial of a recipe -> judge the 1st vial against it
        if pos[i] == 2 and int(queue[i - 1]["vial"]) in end_hue:
            v1 = int(queue[i - 1]["vial"])
            h1 = end_hue.get(v1, float("nan"))
            d = hue_diff(h1, eh)
            if d > CONFIRM_T:
                confirmed.add(v1)
                first_alert.setdefault(v1, t)
                emit("CONFIRM", t, v1,
                     f"carryover confirmed: {color_name(h1)} vs batch color {color_name(eh)} "
                     f"(diff {d:.0f}). Divert/scrap vial {v1} before capping.", diff=d,
                     recipe_id=int(queue[i - 1]["recipe_id"]), alert_scope="observed")
            else:
                emit("OK", t, v1, f"matches its batch (diff {d:.0f}), no carryover", diff=d,
                     recipe_id=int(queue[i - 1]["recipe_id"]))

        # PREDICT: is the NEXT vial the first of a new recipe?
        if i + 1 < len(queue):
            nxt = queue[i + 1]
            if nxt["recipe_id"] != cur["recipe_id"]:
                sc = change_score(cur, nxt)
                level = "HIGH" if sc >= RISK_HIGH else "MEDIUM" if sc >= RISK_MED else "LOW"
                nv = int(nxt["vial"])
                predicted[nv] = (sc, level)
                first_alert.setdefault(nv, t)
                advice = {"HIGH": f"Flush first, or plan to divert vial {nv}.",
                          "MEDIUM": f"Consider a flush; vial {nv} will be checked closely.",
                          "LOW": f"Small change; vial {nv} will still be checked."}[level]
                emit("PREDICT", t, nv,
                     f"recipe {int(cur['recipe_id'])} -> {int(nxt['recipe_id'])}, change score {sc:.2f}: "
                     f"{level} carryover risk. {advice}",
                     score=sc, level=level, from_recipe=int(cur["recipe_id"]),
                     to_recipe=int(nxt["recipe_id"]), target_vial=nv,
                     alert_scope="future")

    if state == "FILLING" and cur is not None:
        print(f"(stream ended while vial {int(cur['vial'])} was still filling)")

    # ---- summary: how did each first-vial-of-a-recipe go?
    print("\n=== Summary: first vial after each recipe change ===")
    print(f"{'vial':>4} {'predict':>14} {'model':>7} {'detect':>7} {'confirm':>8}  "
          f"{'first alert vs vial leaving':>28}")
    caught = total = 0
    for k, q in enumerate(queue):
        v = int(q["vial"])
        if pos[k] != 1 or k == 0 or v not in end_hue:
            continue
        if k + 1 >= len(queue) or int(queue[k + 1]["vial"]) not in end_hue:
            continue                                # no 2nd vial seen, can't confirm
        total += 1
        caught += v in confirmed
        p = f"{predicted[v][1]} {predicted[v][0]:.2f}" if v in predicted else "-"
        lead = ""
        if v in first_alert and v in left_at:
            secs = (left_at[v] - first_alert[v]).total_seconds()
            lead = f"{secs:.0f} s before" if secs >= 0 else f"{-secs:.0f} s after"
        print(f"{v:>4} {p:>14} {('yes' if v in model_alerted else '-'):>7} "
              f"{('yes' if v in detected else '-'):>7} "
              f"{('yes' if v in confirmed else 'no'):>8}  {lead:>28}")
    print(f"\nCONFIRM caught {caught}/{total} first vials. Events saved to {LOG}")


def main():
    global DATA, OUT, LOG
    ap = argparse.ArgumentParser(description="Live carryover monitor")
    ap.add_argument("--source", choices=["replay", "watch"], default="replay")
    ap.add_argument("--dir", help="folder to watch (for --source watch)")
    ap.add_argument("--watch-poll-ms", type=float, default=20,
                    help="live-folder polling interval in milliseconds (default: 20)")
    ap.add_argument("--speed", type=float, default=20,
                    help="replay speed: 1 = real time, 20 = 20x faster, 0 = no waiting")
    ap.add_argument("--queue", default=f"{OUT}/queue.csv", help="recipe queue CSV")
    ap.add_argument("--data-root", default=DATA, help="vial_fill folder containing image_index.csv and images/")
    ap.add_argument("--out-dir", default=OUT, help="folder for event and neural prediction logs")
    ap.add_argument("--model", help="TorchScript model.ts; enables advisory-only neural shadow mode")
    ap.add_argument("--model-threshold", type=float, default=0.0071,
                    help="recall-prioritized provisional shadow threshold")
    ap.add_argument("--model-window", type=int, default=5, help="rolling visible-frame window")
    ap.add_argument("--model-min-frames", type=int, default=3,
                    help="recent frames that must meet threshold before a model alert")
    ap.add_argument("--model-log", help="JSONL path for per-frame neural predictions")
    ap.add_argument("--tank-colors", default=f"{OUT}/tank_colors.npy",
                    help="optional 3x3 expected-color calibration")
    ap.add_argument("--start-vial", type=int,
                    help="assign the next detected fill to this vial (for operator realignment)")
    a = ap.parse_args()

    DATA, OUT = a.data_root, a.out_dir
    LOG = f"{OUT}/events.jsonl"
    os.makedirs(OUT, exist_ok=True)
    open(LOG, "w").close()                      # start a fresh event log
    queue = load_queue(a.queue)
    start_index = 0
    if a.start_vial is not None:
        matches = [index for index, row in enumerate(queue) if int(row["vial"]) == a.start_vial]
        if not matches:
            ap.error(f"--start-vial {a.start_vial} is not present in the queue")
        start_index = matches[0]
    print(f"Loaded {len(queue)} vials from the queue. Source: {a.source}\n")
    if start_index:
        print(f"Operator alignment: next detected fill is vial {int(queue[start_index]['vial'])}.\n")

    neural = None
    if a.model:
        from neural_shadow import NeuralShadowMonitor
        neural = NeuralShadowMonitor(
            a.model,
            a.model_log or f"{OUT}/neural_predictions.jsonl",
            threshold=a.model_threshold,
            window=a.model_window,
            min_frames=a.model_min_frames,
            tank_colors_path=a.tank_colors,
        )
        print(f"NEURAL SHADOW MODE: threshold={a.model_threshold:.4f}, "
              f"window={a.model_window}, minimum frames={a.model_min_frames}. "
              "Advisory only; no PLC action.\n")

    if a.source == "replay":
        src = replay_source(a.speed)
    else:
        if not a.dir:
            ap.error("--source watch needs --dir")
        if a.watch_poll_ms <= 0:
            ap.error("--watch-poll-ms must be greater than zero")
        src = watch_source(a.dir, poll=a.watch_poll_ms / 1000.0)
    try:
        run(src, queue, neural=neural, start_index=start_index)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
