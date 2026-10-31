#!/usr/bin/env python3
"""Build metadata and assemble a video-5-shaped synthetic carryover dataset.

Run inside an Isaac Sim job in two phases:

  prepare  -> write render_plan.csv, queue.csv, and per-vial ground truth
  finalize -> expand rendered phase images into timestamped JPG frames and CSVs

The renderer is intentionally separate.  render_robust.py consumes the plan made
by ``prepare`` and writes one image for each unique vial phase.  ``finalize`` then
reuses those phase images to create a two-second frame cadence like the real data.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path


RECIPES = [
    {"recipe_id": 5, "type": "Triple", "t100_ml": 10, "t200_ml": 10, "t400_ml": 10,
     "dispense_order": "T200>T400>T100", "color_purpose": "mixed - low even"},
    {"recipe_id": 6, "type": "Pair 100+200", "t100_ml": 30, "t200_ml": 10, "t400_ml": 0,
     "dispense_order": "T200>T100", "color_purpose": "pale orange"},
    {"recipe_id": 7, "type": "Pair 100+400", "t100_ml": 20, "t200_ml": 0, "t400_ml": 50,
     "dispense_order": "T400>T100", "color_purpose": "pale blue"},
    {"recipe_id": 8, "type": "Triple", "t100_ml": 20, "t200_ml": 20, "t400_ml": 20,
     "dispense_order": "T200>T400>T100", "color_purpose": "even"},
    {"recipe_id": 9, "type": "Triple", "t100_ml": 10, "t200_ml": 35, "t400_ml": 35,
     "dispense_order": "T200>T400>T100", "color_purpose": "dark saturated"},
    {"recipe_id": 10, "type": "Triple", "t100_ml": 45, "t200_ml": 15, "t400_ml": 15,
     "dispense_order": "T200>T400>T100", "color_purpose": "pale wash"},
    {"recipe_id": 11, "type": "Triple", "t100_ml": 10, "t200_ml": 10, "t400_ml": 50,
     "dispense_order": "T200>T400>T100", "color_purpose": "blue dominant"},
    {"recipe_id": 12, "type": "Triple", "t100_ml": 10, "t200_ml": 50, "t400_ml": 10,
     "dispense_order": "T200>T400>T100", "color_purpose": "orange dominant"},
    {"recipe_id": 13, "type": "Pair 200+400", "t100_ml": 0, "t200_ml": 50, "t400_ml": 20,
     "dispense_order": "T200>T400", "color_purpose": "orange-blue pair"},
    {"recipe_id": 14, "type": "Pair 100+400", "t100_ml": 60, "t200_ml": 0, "t400_ml": 10,
     "dispense_order": "T400>T100", "color_purpose": "very pale blue"},
]

# Approximate USD diffuse colors. These are synthetic rendering parameters, not
# calibrated physical color measurements. T100 is a pale cyan-gray diluent.
TANK_RGB = {
    "t100_ml": (0.12, 0.22, 0.24),
    "t200_ml": (0.45, 0.12, 0.01),
    "t400_ml": (0.01, 0.20, 0.36),
}

PHASES = [
    (60, 0.10),
    (180, 0.35),
    (300, 0.60),
    (500, 0.82),
    (686, 1.00),
]


def clamp(x: float, lo: float = 0.002, hi: float = 0.80) -> float:
    return max(lo, min(hi, x))


def blend(a, b, amount: float):
    return tuple(clamp((1.0 - amount) * x + amount * y) for x, y in zip(a, b))


def recipe_rgb(recipe):
    total = float(recipe["t100_ml"] + recipe["t200_ml"] + recipe["t400_ml"])
    # Colored tanks dominate perceived color; clear tank mostly dilutes it.
    colored = recipe["t200_ml"] + recipe["t400_ml"]
    if colored:
        dye = tuple(
            (recipe["t200_ml"] * TANK_RGB["t200_ml"][i]
             + recipe["t400_ml"] * TANK_RGB["t400_ml"][i]) / colored
            for i in range(3)
        )
    else:
        dye = TANK_RGB["t100_ml"]
    dilution = recipe["t100_ml"] / total
    return blend(dye, TANK_RGB["t100_ml"], 0.70 * dilution)


def color_metadata(recipe, rgb):
    """Human-readable expected color fields derived from the synthetic recipe mix."""
    display = tuple(max(0, min(255, round(v * 255))) for v in rgb)
    hue = (colorsys.rgb_to_hsv(*rgb)[0] * 360.0) % 360.0
    if recipe.get("color_purpose") and recipe["color_purpose"] != "—":
        name = recipe["color_purpose"]
    elif recipe["t200_ml"] and recipe["t400_ml"]:
        name = "orange-blue mix"
    elif recipe["t200_ml"]:
        name = "pale orange"
    elif recipe["t400_ml"]:
        name = "pale blue"
    else:
        name = "clear"
    return {
        "expected_color": name,
        "expected_rgb": f"{display[0]},{display[1]},{display[2]}",
        "expected_hex": "#" + "".join(f"{v:02X}" for v in display),
        "expected_hue_deg": f"{hue:.2f}",
    }


def jitter(rgb, rng, scale=0.012):
    return tuple(clamp(v + rng.uniform(-scale, scale)) for v in rgb)


def rows_for_vials(seed: int):
    rng = random.Random(seed)
    vials = []
    vial = 0
    previous = None
    for recipe in RECIPES:
        intended = recipe_rgb(recipe)
        total = recipe["t100_ml"] + recipe["t200_ml"] + recipe["t400_ml"]
        for position in (1, 2, 3):
            vial += 1
            is_carryover = previous is not None and position == 1
            alpha = rng.uniform(0.22, 0.42) if is_carryover else 0.0
            final = blend(intended, previous, alpha) if is_carryover else intended
            final = jitter(final, rng)
            vials.append({
                **recipe,
                **color_metadata(recipe, intended),
                "vial": vial,
                "pos_in_batch": position,
                "total_ml": total,
                "expected_carryover": is_carryover,
                "carryover_alpha": alpha,
                "intended_rgb": intended,
                "actual_final_rgb": final,
                "previous_rgb": previous,
                "opacity": rng.uniform(0.55, 0.72),
            })
        previous = intended
    return vials


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def prepare(out_dir: Path, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    vials = rows_for_vials(seed)
    render_rows = [{"out": "unique/empty.png", "frame": 1}]
    for v in vials:
        intended = v["intended_rgb"]
        previous = v["previous_rgb"] or intended
        alpha = v["carryover_alpha"]
        for phase_idx, (frame, progress) in enumerate(PHASES):
            # Residue is strongest at the start of the first vial and decays.
            liquid_alpha = alpha * (1.35 - 0.35 * progress)
            stream_alpha = alpha * max(0.0, 1.25 - 1.20 * progress)
            liquid = blend(intended, previous, min(0.75, liquid_alpha))
            stream = blend(intended, previous, min(0.90, stream_alpha))
            liquid = jitter(liquid, random.Random(seed + v["vial"] * 100 + phase_idx), 0.006)
            render_rows.append({
                "out": f"unique/vial_{v['vial']:03d}_phase_{phase_idx}.png",
                "frame": frame,
                "liquid_r": f"{liquid[0]:.6f}",
                "liquid_g": f"{liquid[1]:.6f}",
                "liquid_b": f"{liquid[2]:.6f}",
                "stream_r": f"{stream[0]:.6f}",
                "stream_g": f"{stream[1]:.6f}",
                "stream_b": f"{stream[2]:.6f}",
                "opacity": f"{v['opacity']:.4f}",
            })

    plan_fields = ["out", "frame", "liquid_r", "liquid_g", "liquid_b",
                   "stream_r", "stream_g", "stream_b", "opacity"]
    write_csv(out_dir / "render_plan.csv", plan_fields, render_rows)

    queue_fields = ["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                    "expected_color", "expected_rgb", "expected_hex", "expected_hue_deg"]
    write_csv(out_dir / "queue.csv", queue_fields, vials)

    truth_rows = []
    for v in vials:
        truth_rows.append({
            **v,
            "intended_r": v["intended_rgb"][0], "intended_g": v["intended_rgb"][1],
            "intended_b": v["intended_rgb"][2], "actual_r": v["actual_final_rgb"][0],
            "actual_g": v["actual_final_rgb"][1], "actual_b": v["actual_final_rgb"][2],
        })
    truth_fields = ["vial", "recipe_id", "pos_in_batch", "expected_carryover", "carryover_alpha",
                    "t100_ml", "t200_ml", "t400_ml", "total_ml",
                    "expected_color", "expected_rgb", "expected_hex", "expected_hue_deg",
                    "intended_r", "intended_g", "intended_b", "actual_r", "actual_g", "actual_b"]
    write_csv(out_dir / "synthetic_vial_ground_truth.csv", truth_fields, truth_rows)

    manifest = {
        "name": "synthetic_video5_carryover_v2",
        "synthetic": True,
        "seed": seed,
        "vials": len(vials),
        "recipes": len(RECIPES),
        "rendered_images": len(render_rows),
        "frame_interval_seconds": 2.0,
        "fill_frames_per_vial": 15,
        "idle_frames_after_each_vial": 15,
        "video_fps": 30.0,
        "tank_start_ml": {"T100": 1000, "T200": 1000, "T300": 0, "T400": 1000},
        "limitations": [
            "USD colors are approximate and are not calibrated physical measurements.",
            "Five unique rendered fill phases are repeated to create the two-second cadence.",
            "Carryover is an explicit color blend, not a fluid-dynamics simulation.",
            "The supplied twin has no tank geometry; tank_level is a synchronized drawdown visualization.",
            "Use this dataset for development and robustness testing, not final real-line performance claims.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared {len(render_rows)} unique renders for {len(vials)} vials")


def finalize(out_dir: Path, start_time: str, video_fps: float):
    import cv2
    import numpy as np

    manifest = json.loads((out_dir / "manifest.json").read_text())
    vials = rows_for_vials(manifest["seed"])
    unique = out_dir / "unique"
    vf_images = out_dir / "vial_fill" / "images"
    vf_labels = out_dir / "vial_fill" / "labels"
    vf_videos = out_dir / "vial_fill" / "videos"
    tl_images = out_dir / "tank_level" / "images"
    tl_videos = out_dir / "tank_level" / "videos"
    for d in (vf_images, vf_labels, vf_videos, tl_images, tl_videos):
        d.mkdir(parents=True, exist_ok=True)

    empty = cv2.imread(str(unique / "empty.png"), cv2.IMREAD_COLOR)
    if empty is None:
        raise FileNotFoundError(unique / "empty.png")

    current = datetime.fromisoformat(start_time)
    interval = timedelta(seconds=2)
    vf_index, tl_index, frame_labels, vial_recipes, tank_rows = [], [], [], [], []
    vf_paths, tl_paths = [], []
    video_s = 0.0
    tank_levels = {"T100": 1000.0, "T200": 1000.0, "T300": 0.0, "T400": 1000.0}

    def tank_frame(levels, vial, in_fill):
        canvas = np.full((480, 640, 3), (228, 231, 234), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (639, 62), (40, 48, 58), -1)
        cv2.putText(canvas, "SYNTHETIC TANK LEVELS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (255, 255, 255), 2, cv2.LINE_AA)
        status = "IDLE" if not in_fill else f"VIAL {vial['vial']:02d}  RECIPE {vial['recipe_id']}"
        cv2.putText(canvas, status, (20, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (185, 220, 255), 1, cv2.LINE_AA)
        colors = {"T100": (190, 205, 190), "T200": (20, 105, 235),
                  "T300": (130, 130, 130), "T400": (215, 105, 20)}
        for i, tank in enumerate(("T100", "T200", "T300", "T400")):
            x0, y0, width, height = 48 + i * 150, 104, 92, 276
            cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (45, 52, 60), 3)
            cv2.ellipse(canvas, (x0 + width // 2, y0), (width // 2, 12), 0, 0, 360, (45, 52, 60), 3)
            value = levels[tank]
            pct = max(0.0, min(1.0, value / 1000.0))
            liquid_h = int((height - 8) * pct)
            if liquid_h:
                cv2.rectangle(canvas, (x0 + 4, y0 + height - liquid_h),
                              (x0 + width - 4, y0 + height - 4), colors[tank], -1)
            cv2.putText(canvas, tank, (x0 + 14, 414), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (35, 40, 45), 2, cv2.LINE_AA)
            label = "DISABLED" if tank == "T300" else f"{value:6.1f} mL"
            cv2.putText(canvas, label, (x0 - 4, 446), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (35, 40, 45), 1, cv2.LINE_AA)
        cv2.putText(canvas, current.strftime("%Y-%m-%d %H:%M:%S"), (402, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas

    def emit_frame(img, levels, in_fill, vial=None):
        nonlocal current, video_s
        timestamp = current.isoformat(sep=" ")
        vf_name = f"vial_fill_{current:%Y%m%d_%H%M%S}_{current.microsecond:06d}.jpg"
        tl_name = f"tank_level_{current:%Y%m%d_%H%M%S}_{current.microsecond:06d}.jpg"
        vf_path, tl_path = vf_images / vf_name, tl_images / tl_name
        tank_img = tank_frame(levels, vial, in_fill)
        if not cv2.imwrite(str(vf_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise OSError(f"failed to write {vf_path}")
        if not cv2.imwrite(str(tl_path), tank_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise OSError(f"failed to write {tl_path}")
        vf_paths.append(vf_path); tl_paths.append(tl_path)
        vf_index.append({"filename": vf_name, "timestamp": timestamp, "video_id": 5})
        tl_index.append({"filename": tl_name, "timestamp": timestamp, "video_id": 5})
        tank_rows.append({"filename": tl_name, "timestamp": timestamp, "video_s": f"{video_s:.3f}",
                          "vial": vial["vial"] if vial else "",
                          "recipe_id": vial["recipe_id"] if vial else "",
                          "t100_level_ml": f"{levels['T100']:.3f}",
                          "t200_level_ml": f"{levels['T200']:.3f}",
                          "t300_level_ml": f"{levels['T300']:.3f}",
                          "t400_level_ml": f"{levels['T400']:.3f}"})
        if in_fill:
            frame_labels.append({
                "filename": vf_name, "timestamp": timestamp, "video_s": f"{video_s:.3f}",
                "in_fill": 1, "vial": vial["vial"], "recipe_id": vial["recipe_id"],
                "type": vial["type"], "t100_ml": vial["t100_ml"], "t200_ml": vial["t200_ml"],
                "t400_ml": vial["t400_ml"], "total_ml": vial["total_ml"],
                "dispense_order": vial["dispense_order"],
                "color_purpose": "synthetic carryover" if vial["expected_carryover"] else "synthetic clean",
            })
        else:
            frame_labels.append({"filename": vf_name, "timestamp": timestamp,
                                 "video_s": f"{video_s:.3f}", "in_fill": 0})
        current += interval
        video_s += interval.total_seconds()

    for v in vials:
        vial_start = video_s
        before = tank_levels.copy()
        frame_no = 0
        for phase_idx in range(len(PHASES)):
            img = cv2.imread(str(unique / f"vial_{v['vial']:03d}_phase_{phase_idx}.png"), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(unique / f"vial_{v['vial']:03d}_phase_{phase_idx}.png")
            for _ in range(3):
                frame_no += 1
                progress = frame_no / 15.0
                levels = before.copy()
                for tank, field in (("T100", "t100_ml"), ("T200", "t200_ml"), ("T400", "t400_ml")):
                    levels[tank] = before[tank] - v[field] * progress
                emit_frame(img, levels, True, v)
        tank_levels = levels
        vial_end = video_s - interval.total_seconds()
        vial_recipes.append({
            **v, "start_time": str(timedelta(seconds=vial_start)),
            "end_time": str(timedelta(seconds=vial_end)),
            "start_s": f"{vial_start:.3f}", "end_s": f"{vial_end:.3f}",
            "duration_s": f"{vial_end - vial_start:.3f}",
            "start_frame": int(vial_start * video_fps),
            "end_frame": int(vial_end * video_fps),
        })
        for _ in range(15):
            emit_frame(empty, tank_levels, False)

    write_csv(out_dir / "vial_fill" / "image_index.csv", ["filename", "timestamp", "video_id"], vf_index)
    write_csv(out_dir / "tank_level" / "image_index.csv", ["filename", "timestamp", "video_id"], tl_index)
    write_csv(out_dir / "tank_level" / "tank_levels.csv",
              ["filename", "timestamp", "video_s", "vial", "recipe_id", "t100_level_ml",
               "t200_level_ml", "t300_level_ml", "t400_level_ml"], tank_rows)
    frame_fields = ["filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id", "type",
                    "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order", "color_purpose"]
    write_csv(vf_labels / "video5_frame_labels.csv", frame_fields, frame_labels)
    recipe_fields = ["vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                     "dispense_order", "color_purpose", "expected_color", "expected_rgb", "expected_hex",
                     "expected_hue_deg", "start_time", "end_time", "start_s", "end_s", "duration_s",
                     "start_frame", "end_frame"]
    write_csv(vf_labels / "video5_vial_recipes.csv", recipe_fields, vial_recipes)
    write_csv(out_dir / "videos.csv", ["video_id", "duration_s", "cameras", "n_vials", "labeled", "notes"], [{
        "video_id": 5, "duration_s": f"{video_s:.0f}", "cameras": "tank_level;vial_fill",
        "n_vials": len(vials), "labeled": "vial_fill;tank_level",
        "notes": "synthetic FillCam twin plus synchronized tank-level visualization",
    }])

    def write_realtime_video(paths, destination):
        h, w = empty.shape[:2]
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"could not create {destination}")
        repeat = int(round(interval.total_seconds() * video_fps))
        for n, path in enumerate(paths, 1):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(path)
            for _ in range(repeat):
                writer.write(frame)
            if n % 100 == 0:
                print(f"video {destination.name}: {n}/{len(paths)} samples", flush=True)
        writer.release()
        return len(paths) * repeat

    vf_video = vf_videos / "vial_fill_video_5.mp4"
    tl_video = tl_videos / "tank_level_video_5.mp4"
    vf_count = write_realtime_video(vf_paths, vf_video)
    tl_count = write_realtime_video(tl_paths, tl_video)
    metadata = {
        "fps": video_fps, "duration_s": video_s, "resolution": [640, 480],
        "vial_fill_frames": vf_count, "tank_level_frames": tl_count,
        "codec_fourcc": "mp4v", "sampled_still_interval_s": 2,
    }
    (out_dir / "video_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = (
        "# Synthetic video-5-shaped carryover dataset v2\n\n"
        f"Contains {len(vials)} vials, {len(vf_index)} synchronized sampled frames per camera, "
        f"and two {video_fps:g} fps videos lasting {video_s:.0f} seconds.\n\n"
        "- `vial_fill/`: FillCam digital-twin images, labels, and 30 fps video\n"
        "- `tank_level/`: synchronized synthetic tank drawdown images, values, and 30 fps video\n"
        "- `queue.csv`: recipe queue with expected color, RGB/hex, and hue\n"
        "- `vial_fill/labels/video5_vial_recipes.csv`: per-vial recipe timing and expected color\n\n"
        "The tank-level view is a labeled visualization because the supplied twin has no tank geometry or tank camera. "
        "T300 is disabled. Colors and carryover are synthetic and must not be presented as real-line measurements.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"finalized {len(vf_index)} synchronized samples and two {video_fps:g} fps videos", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "finalize"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-time", default="2026-10-27T12:00:00")
    ap.add_argument("--video-fps", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    if args.mode == "prepare":
        prepare(out, args.seed)
    else:
        finalize(out, args.start_time, args.video_fps)


if __name__ == "__main__":
    main()
