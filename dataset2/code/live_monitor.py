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
import glob
import json
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

BASE = "/home/yt-thd0812/workspace"
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

ANSI = {"PREDICT": "\033[93m", "DETECT": "\033[91m", "CONFIRM": "\033[91m",
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
    im = Image.open(path).convert("RGB").crop(ROI)
    hsv = np.asarray(im.convert("HSV")).astype(float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colored = (s > 90) & (v > 40)
    frac = float(colored.mean())
    hue = float(np.median(h[colored])) if colored.any() else float("nan")
    return frac, hue


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
        lab = pd.read_csv(f"{OUT}/video5_labels_corrected.csv")
        lab[["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml"]].to_csv(path, index=False)
        print(f"(created {path} from the corrected video-5 labels)")
    q = pd.read_csv(path).fillna(0).sort_values("vial").reset_index(drop=True)
    return q.to_dict("records")


def replay_source(speed):
    """Yield video 5's frames in time order, waiting between them like a live camera."""
    idx = pd.read_csv(f"{DATA}/image_index.csv")
    idx = idx[idx["video_id"].astype(str) == "5"].sort_values("filename")
    prev = None
    for f in idx["filename"]:
        t = frame_time(f)
        if prev is not None and speed > 0:
            time.sleep(max(0.0, (t - prev).total_seconds() / speed))
        prev = t
        yield f"{DATA}/images/{f}", t


def watch_source(folder, poll=0.5):
    """Yield each new .jpg that appears in a folder (for a live camera)."""
    seen = set(glob.glob(os.path.join(folder, "*.jpg")))   # ignore files already there
    print(f"(watching {folder} for new images; Ctrl+C to stop)")
    while True:
        for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            if p not in seen:
                seen.add(p)
                time.sleep(0.2)              # let the camera finish writing the file
                yield p, frame_time(p)
        time.sleep(poll)


# ---------------------------------------------------------------- main loop
def run(source, queue):
    # Position of each vial inside its recipe batch (1 = first vial after a change)
    pos = []
    for i, q in enumerate(queue):
        if i == 0 or q["recipe_id"] != queue[i - 1]["recipe_id"]:
            pos.append(1)
        else:
            pos.append(pos[-1] + 1)

    state, i = "EMPTY", -1
    recent = deque(maxlen=5)          # last few hues of the current vial
    pending = []                      # colored frames seen while EMPTY, not yet a confirmed fill
    end_hue, fill_start, left_at, first_alert = {}, {}, {}, {}
    predicted, detected, confirmed = {}, set(), set()
    cur, unexpected_run, did_detect = None, 0, False

    def filling_frame(frac, hue, now):
        """One frame of the vial being filled: remember its hue, and DETECT
        colors this recipe cannot make. `now` = time the alert is raised."""
        nonlocal unexpected_run, did_detect
        recent.append(hue)
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
                 hue=hue)

    for path, t in source:
        frac, hue = measure(path)

        # ---- EMPTY: wait until color stays for MIN_FILL_FRAMES frames in a row
        if state == "EMPTY":
            if frac <= THRESH:
                pending.clear()              # color vanished again: it was only a flicker
                continue
            pending.append((t, frac, hue))
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
            t0 = pending[0][0]
            fill_start[v] = t0
            lead = ""
            if v in first_alert:
                lead = f" (warned {(t0 - first_alert[v]).total_seconds():.0f} s before the fill started)"
            emit("INFO", t, v, f"fill started at {t0:%H:%M:%S}, recipe {int(cur['recipe_id'])}{lead}")
            for _, pf, ph in pending:            # catch up on the frames we waited for
                filling_frame(pf, ph, t)
            pending.clear()
            continue

        # ---- FILLING
        v = int(cur["vial"])
        if frac > THRESH:
            filling_frame(frac, hue, t)
            continue

        # ---- color dropped to ~0: the vial left the station
        state = "EMPTY"
        left_at[v] = t
        hs = [h for h in list(recent)[:-1] if not np.isnan(h)] or [h for h in recent if not np.isnan(h)]
        eh = float(np.median(hs)) if hs else float("nan")
        end_hue[v] = eh
        emit("INFO", t, v, f"left the station, final color {color_name(eh)} (hue {eh:.0f})")

        # CONFIRM: this is the 2nd vial of a recipe -> judge the 1st vial against it
        if pos[i] == 2:
            v1 = int(queue[i - 1]["vial"])
            h1 = end_hue.get(v1, float("nan"))
            d = hue_diff(h1, eh)
            if d > CONFIRM_T:
                confirmed.add(v1)
                first_alert.setdefault(v1, t)
                emit("CONFIRM", t, v1,
                     f"carryover confirmed: {color_name(h1)} vs batch color {color_name(eh)} "
                     f"(diff {d:.0f}). Divert/scrap vial {v1} before capping.", diff=d)
            else:
                emit("OK", t, v1, f"matches its batch (diff {d:.0f}), no carryover", diff=d)

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
                     score=sc, level=level)

    if state == "FILLING" and cur is not None:
        print(f"(stream ended while vial {int(cur['vial'])} was still filling)")

    # ---- summary: how did each first-vial-of-a-recipe go?
    print("\n=== Summary: first vial after each recipe change ===")
    print(f"{'vial':>4} {'predict':>14} {'detect':>7} {'confirm':>8}  {'first alert vs vial leaving':>28}")
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
        print(f"{v:>4} {p:>14} {('yes' if v in detected else '-'):>7} "
              f"{('yes' if v in confirmed else 'no'):>8}  {lead:>28}")
    print(f"\nCONFIRM caught {caught}/{total} first vials. Events saved to {LOG}")


def main():
    ap = argparse.ArgumentParser(description="Live carryover monitor")
    ap.add_argument("--source", choices=["replay", "watch"], default="replay")
    ap.add_argument("--dir", help="folder to watch (for --source watch)")
    ap.add_argument("--speed", type=float, default=20,
                    help="replay speed: 1 = real time, 20 = 20x faster, 0 = no waiting")
    ap.add_argument("--queue", default=f"{OUT}/queue.csv", help="recipe queue CSV")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    open(LOG, "w").close()                      # start a fresh event log
    queue = load_queue(a.queue)
    print(f"Loaded {len(queue)} vials from the queue. Source: {a.source}\n")

    if a.source == "replay":
        src = replay_source(a.speed)
    else:
        if not a.dir:
            ap.error("--source watch needs --dir")
        src = watch_source(a.dir)
    try:
        run(src, queue)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
