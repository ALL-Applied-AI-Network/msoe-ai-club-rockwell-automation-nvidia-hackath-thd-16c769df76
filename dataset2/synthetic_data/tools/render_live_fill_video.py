#!/usr/bin/env python3
"""Render the FillCam animation at one unique Isaac Sim frame per video frame."""

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from isaacsim import SimulationApp


ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="stations/fill_station/fill_station.usda")
ap.add_argument("--camera-config", default="cameras/FillCam.json")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--idle-seconds", type=float, default=2.0)
ap.add_argument("--start-time", default="2026-10-27T12:00:00")
ap.add_argument("--renderer", default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
ap.add_argument("--subframes", type=int, default=1)
ap.add_argument("--warmup", type=int, default=90)
ap.add_argument("--max-tries", type=int, default=20)
ap.add_argument("--first-vial", type=int, default=1)
ap.add_argument("--max-vials", type=int, default=30)
ap.add_argument("--max-animation-frames", type=int, default=686)
a = ap.parse_args()

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
from make_synthetic_video5 import blend, rows_for_vials, write_csv  # noqa: E402


def resolve(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else HERE / p).resolve()


stage_path = resolve(a.stage)
camera_config = json.loads(resolve(a.camera_config).read_text())
out_dir = Path(a.out_dir).expanduser().resolve()
images_dir = out_dir / "vial_fill" / "images"
labels_dir = out_dir / "vial_fill" / "labels"
videos_dir = out_dir / "vial_fill" / "videos"
clips_dir = videos_dir / "vials"
for directory in (images_dir, labels_dir, clips_dir):
    directory.mkdir(parents=True, exist_ok=True)

all_vials = rows_for_vials(42)
start_index = max(0, a.first_vial - 1)
vials = all_vials[start_index:start_index + a.max_vials]
animation_frames = min(686, max(1, a.max_animation_frames))
rw, rh = camera_config["render_px"]

kit = SimulationApp({"headless": True, "renderer": a.renderer, "width": rw, "height": rh})
try:
    import cv2
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(stage_path)):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    camera_prim = next((p for p in (camera_config["prim_in_station_file"],
                                    camera_config["prim_in_csi_line"])
                        if stage.GetPrimAtPath(p).IsValid()), None)
    if camera_prim is None:
        raise RuntimeError("FillCam not found")

    render_product = rep.create.render_product(camera_prim, (rw, rh))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    clip = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update()
    clip.Set(near_far); kit.update()

    def find_shader(material_name):
        for prim in stage.Traverse():
            if str(prim.GetPath()).endswith(f"{material_name}/Principled_BSDF"):
                return UsdShade.Shader(prim)
        return None

    liquid_shader = find_shader("USD_blue_liquid_approximation")
    stream_shader = find_shader("USD_jet_approximation")
    if not liquid_shader or not stream_shader:
        raise RuntimeError("liquid or stream shader not found")

    timeline = omni.timeline.get_timeline_interface()
    tcps = stage.GetTimeCodesPerSecond()
    for _ in range(a.warmup):
        kit.update()

    def capture():
        for attempt in range(1, a.max_tries + 1):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            data = np.asarray(annotator.get_data())
            if data.ndim == 3 and data.shape[:2] == (rh, rw) and data.shape[2] >= 3:
                return data[..., :3].copy()
            print(f"capture attempt {attempt}: shape={data.shape}", flush=True)
            for _ in range(5):
                kit.update()
        raise RuntimeError("RGB capture remained empty")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_path = videos_dir / "vial_fill_video_5.mp4"
    combined = cv2.VideoWriter(str(combined_path), fourcc, a.fps, (rw, rh))
    if not combined.isOpened():
        raise RuntimeError(f"could not create {combined_path}")

    current = datetime.fromisoformat(a.start_time)
    combined_frame = 0
    sample_every = max(1, int(round(2.0 * a.fps)))
    idle_frames = max(0, int(round(a.idle_seconds * a.fps)))
    image_index, frame_labels, vial_recipes = [], [], []
    last_bgr = None

    def sample_still(bgr, vial=None, in_fill=False):
        filename = f"vial_fill_{current:%Y%m%d_%H%M%S}_{current.microsecond:06d}.jpg"
        path = images_dir / filename
        if not cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise OSError(f"failed to write {path}")
        timestamp = current.isoformat(sep=" ")
        image_index.append({"filename": filename, "timestamp": timestamp, "video_id": 5})
        row = {"filename": filename, "timestamp": timestamp,
               "video_s": f"{combined_frame / a.fps:.6f}", "in_fill": int(in_fill)}
        if vial:
            row.update({"vial": vial["vial"], "recipe_id": vial["recipe_id"], "type": vial["type"],
                        "t100_ml": vial["t100_ml"], "t200_ml": vial["t200_ml"],
                        "t400_ml": vial["t400_ml"], "total_ml": vial["total_ml"],
                        "dispense_order": vial["dispense_order"],
                        "color_purpose": "synthetic carryover" if vial["expected_carryover"] else "synthetic clean"})
        frame_labels.append(row)

    for vial_number, vial in enumerate(vials, 1):
        clip_path = clips_dir / f"vial_{vial['vial']:03d}.mp4"
        clip_writer = cv2.VideoWriter(str(clip_path), fourcc, a.fps, (rw, rh))
        if not clip_writer.isOpened():
            raise RuntimeError(f"could not create {clip_path}")
        vial_start_frame = combined_frame
        intended = vial["intended_rgb"]
        previous = vial["previous_rgb"] or intended
        alpha = vial["carryover_alpha"]
        for frame in range(1, animation_frames + 1):
            progress = (frame - 1) / max(1, animation_frames - 1)
            liquid_alpha = alpha * (1.35 - 0.35 * progress)
            stream_alpha = alpha * max(0.0, 1.25 - 1.20 * progress)
            liquid = blend(intended, previous, min(0.75, liquid_alpha))
            stream = blend(intended, previous, min(0.90, stream_alpha))
            liquid_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*liquid))
            liquid_shader.GetInput("opacity").Set(float(vial["opacity"]))
            stream_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*stream))
            timeline.set_current_time(frame / tcps)
            kit.update()
            rgb = capture()
            bgr = rgb[..., ::-1]
            combined.write(bgr); clip_writer.write(bgr)
            last_bgr = bgr
            if combined_frame % sample_every == 0:
                sample_still(bgr, vial, True)
            combined_frame += 1
            current += timedelta(seconds=1.0 / a.fps)
            if frame % 100 == 0:
                print(f"vial {vial['vial']:02d}: {frame}/{animation_frames}", flush=True)
        clip_writer.release()
        vial_end_frame = combined_frame - 1
        vial_recipes.append({
            **vial,
            "start_time": str(timedelta(seconds=vial_start_frame / a.fps)),
            "end_time": str(timedelta(seconds=vial_end_frame / a.fps)),
            "start_s": f"{vial_start_frame / a.fps:.6f}",
            "end_s": f"{vial_end_frame / a.fps:.6f}",
            "duration_s": f"{animation_frames / a.fps:.6f}",
            "start_frame": vial_start_frame, "end_frame": vial_end_frame,
        })
        for _ in range(idle_frames):
            combined.write(last_bgr)
            if combined_frame % sample_every == 0:
                sample_still(last_bgr, None, False)
            combined_frame += 1
            current += timedelta(seconds=1.0 / a.fps)
        print(f"completed vial {vial['vial']} ({vial_number}/{len(vials)})", flush=True)

    combined.release()
    write_csv(out_dir / "vial_fill" / "image_index.csv", ["filename", "timestamp", "video_id"], image_index)
    frame_fields = ["filename", "timestamp", "video_s", "in_fill", "vial", "recipe_id", "type",
                    "t100_ml", "t200_ml", "t400_ml", "total_ml", "dispense_order", "color_purpose"]
    write_csv(labels_dir / "video5_frame_labels.csv", frame_fields, frame_labels)
    recipe_fields = ["vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                     "dispense_order", "color_purpose", "expected_color", "expected_rgb", "expected_hex",
                     "expected_hue_deg", "start_time", "end_time", "start_s", "end_s", "duration_s",
                     "start_frame", "end_frame"]
    write_csv(labels_dir / "video5_vial_recipes.csv", recipe_fields, vial_recipes)
    queue_fields = ["vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                    "expected_color", "expected_rgb", "expected_hex", "expected_hue_deg"]
    write_csv(out_dir / "queue.csv", queue_fields, vials)
    duration_s = combined_frame / a.fps
    write_csv(out_dir / "videos.csv", ["video_id", "duration_s", "cameras", "n_vials", "labeled", "notes"], [{
        "video_id": 5, "duration_s": f"{duration_s:.6f}", "cameras": "vial_fill",
        "n_vials": len(vials), "labeled": "vial_fill",
        "notes": "one unique Isaac Sim render per 30 fps frame; live-speed fill animation",
    }])
    metadata = {"fps": a.fps, "resolution": [rw, rh], "combined_frames": combined_frame,
                "duration_s": duration_s, "unique_rendered_frames": len(vials) * animation_frames,
                "animation_frames_per_vial": animation_frames, "idle_frames_per_vial": idle_frames,
                "per_vial_clips": len(vials), "renderer": a.renderer, "subframes": a.subframes}
    (out_dir / "video_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Live-speed synthetic FillCam dataset\n\n"
        "Every non-idle video frame is independently rendered from the Isaac Sim fill animation at 30 fps. "
        "No two-second sampled stills are repeated during a fill. Recipe colors and carryover remain synthetic; "
        "the nozzle, stream, liquid level, vial, lighting, and camera motion come from the digital twin.\n",
        encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()
