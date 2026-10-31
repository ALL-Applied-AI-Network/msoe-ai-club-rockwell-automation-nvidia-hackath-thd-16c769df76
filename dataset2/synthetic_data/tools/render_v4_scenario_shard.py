#!/usr/bin/env python3
"""Render one v4 scenario shard with residual-volume labels and QA."""

import argparse
import csv
import hashlib
import json
import math
import traceback
from pathlib import Path

from isaacsim import SimulationApp


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--stage", default="stations/fill_station/fill_station.usda")
ap.add_argument("--camera-config", default="cameras/FillCam.json")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--warmup", type=int, default=90)
ap.add_argument("--subframes", type=int, default=1)
ap.add_argument("--max-tries", type=int, default=20)
a = ap.parse_args()

HERE = Path(__file__).resolve().parent.parent


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else HERE / path).resolve()


def clamp(value, low=0.002, high=0.80):
    return max(low, min(high, value))


def mix_rgb(previous, expected, residual_ml, current_ml):
    if residual_ml <= 0 or current_ml <= 0:
        return expected if residual_ml <= 0 else previous
    total = residual_ml + current_ml
    return tuple(clamp((previous[i] * residual_ml + expected[i] * current_ml) / total) for i in range(3))


def color_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = resolve(a.manifest)
with manifest_path.open(newline="", encoding="utf-8") as handle:
    scenarios = list(csv.DictReader(handle))
if not scenarios:
    raise ValueError(f"empty manifest: {manifest_path}")

camera = json.loads(resolve(a.camera_config).read_text())
rw, rh = camera["render_px"]
out = Path(a.out_dir).expanduser().resolve()
clips_dir, samples_dir, qa_dir = out / "clips", out / "samples", out / "qa"
for directory in (clips_dir, samples_dir, qa_dir):
    directory.mkdir(parents=True, exist_ok=True)

renderer = scenarios[0]["renderer"]
kit = SimulationApp({"headless": True, "renderer": renderer, "width": rw, "height": rh})
try:
    import cv2
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade

    ctx = omni.usd.get_context()
    stage_path = resolve(a.stage)
    if not ctx.open_stage(str(stage_path)):
        raise RuntimeError(f"could not open {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    camera_prim = next((path for path in (camera["prim_in_station_file"], camera["prim_in_csi_line"])
                        if stage.GetPrimAtPath(path).IsValid()), None)
    if camera_prim is None:
        raise RuntimeError("FillCam not found")
    render_product = rep.create.render_product(camera_prim, (rw, rh))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    clip_range = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim)).GetClippingRangeAttr()
    near_far = clip_range.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip_range.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update()
    clip_range.Set(near_far); kit.update()

    def shader(name):
        for prim in stage.Traverse():
            if str(prim.GetPath()).endswith(f"{name}/Principled_BSDF"):
                return UsdShade.Shader(prim)
        return None

    liquid_shader = shader("USD_blue_liquid_approximation")
    stream_shader = shader("USD_jet_approximation")
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
            print(f"capture retry {attempt}: shape={data.shape}", flush=True)
            for _ in range(5):
                kit.update()
        raise RuntimeError("RGB capture remained empty")

    def apply_domain_variation(rgb, scenario, frame_index):
        """Apply mild, label-independent camera variation recorded in the manifest."""
        image = rgb.astype(np.float32)
        exposure = 2.0 ** float(scenario.get("exposure_ev", 0.0))
        contrast = float(scenario.get("contrast", 1.0))
        white_balance = np.array([
            float(scenario.get("wb_r", 1.0)),
            float(scenario.get("wb_g", 1.0)),
            float(scenario.get("wb_b", 1.0)),
        ], dtype=np.float32).reshape(1, 1, 3)
        image = ((image * white_balance * exposure - 127.5) * contrast) + 127.5
        noise_std = float(scenario.get("sensor_noise_std", 0.0))
        if noise_std > 0:
            seed = int(scenario.get("seed", 0)) * 1000 + frame_index
            noise = np.random.default_rng(seed).normal(0.0, noise_std, image.shape)
            image += noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        blur_sigma = float(scenario.get("blur_sigma", 0.0))
        if blur_sigma > 0:
            image = cv2.GaussianBlur(image, (3, 3), blur_sigma)
        dx = int(scenario.get("camera_dx_px", 0))
        dy = int(scenario.get("camera_dy_px", 0))
        if dx or dy:
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            image = cv2.warpAffine(image, transform, (rw, rh), borderMode=cv2.BORDER_REFLECT_101)
        return image

    vial_rows, frame_rows, sample_records, failures = [], [], [], []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for scenario_number, scenario in enumerate(scenarios, 1):
        scenario_id = scenario["scenario_id"]
        fps, frame_count = float(scenario["fps"]), int(scenario["frame_count"])
        residual_ml = float(scenario["residual_ml"])
        recipe_total = float(scenario["current_total_ml"])
        previous = tuple(float(scenario[f"previous_{channel}"]) for channel in "rgb")
        expected = tuple(float(scenario[f"expected_{channel}"]) for channel in "rgb")
        opacity = float(scenario["opacity"])
        clip_path = clips_dir / f"{scenario_id}.mp4"
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (rw, rh))
        if not writer.isOpened():
            raise RuntimeError(f"could not create {clip_path}")
        sample_hashes, sample_images = [], []
        for frame_index in range(frame_count):
            progress = frame_index / max(1, frame_count - 1)
            current_ml = recipe_total * progress
            actual = mix_rgb(previous, expected, residual_ml, current_ml)
            fraction = residual_ml / (residual_ml + current_ml) if residual_ml + current_ml > 0 else 0.0
            distance = color_distance(actual, expected)
            liquid_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*actual))
            liquid_shader.GetInput("opacity").Set(opacity)
            stream_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*actual))
            timeline.set_current_time((frame_index + 1) / tcps)
            kit.update()
            rgb = apply_domain_variation(capture(), scenario, frame_index)
            bgr = rgb[..., ::-1]
            writer.write(bgr)
            frame_rows.append({
                "scenario_id": scenario_id, "frame_index": frame_index,
                "video_s": f"{frame_index / fps:.6f}", "fill_progress": f"{progress:.8f}",
                "current_dispensed_ml": f"{current_ml:.8f}", "residual_ml": f"{residual_ml:.8f}",
                "carryover_fraction": f"{fraction:.8f}",
                "actual_r": f"{actual[0]:.8f}", "actual_g": f"{actual[1]:.8f}",
                "actual_b": f"{actual[2]:.8f}", "expected_r": f"{expected[0]:.8f}",
                "expected_g": f"{expected[1]:.8f}", "expected_b": f"{expected[2]:.8f}",
                "color_error_distance": f"{distance:.8f}",
                "stream_visible": int(frame_index >= 45), "liquid_visible": int(frame_index >= 45),
                "error_visible": int(residual_ml > 0 and distance >= 0.015),
            })
            if frame_index in (0, 137, 274, 411, 548, frame_count - 1):
                sample_path = samples_dir / f"{scenario_id}_f{frame_index:03d}.jpg"
                cv2.imwrite(str(sample_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                digest = hashlib.sha256(bgr.tobytes()).hexdigest()
                sample_hashes.append(digest); sample_images.append((sample_path, frame_index))
                sample_records.append((scenario_id, sample_path, frame_index))
            if (frame_index + 1) % 200 == 0:
                print(f"{scenario_id}: {frame_index + 1}/{frame_count}", flush=True)
        writer.release()
        check = cv2.VideoCapture(str(clip_path))
        actual_frames = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT)))
        actual_fps = float(check.get(cv2.CAP_PROP_FPS))
        actual_w = int(round(check.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(check.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        check.release()
        passed = actual_frames == frame_count and abs(actual_fps - fps) < 0.01 and (actual_w, actual_h) == (rw, rh)
        passed = passed and len(set(sample_hashes)) >= 5 and clip_path.stat().st_size > 10000
        if not passed:
            failures.append({"scenario_id": scenario_id, "reason": "video metadata or uniqueness QA failed"})
        vial_rows.append({**scenario, "clip_path": f"clips/{clip_path.name}",
                          "clip_sha256": file_sha256(clip_path), "actual_frame_count": actual_frames,
                          "actual_fps": f"{actual_fps:.6f}", "width": actual_w, "height": actual_h,
                          "unique_sample_hashes": len(set(sample_hashes)), "qa_passed": int(passed)})
        print(f"completed {scenario_id} ({scenario_number}/{len(scenarios)}) qa={passed}", flush=True)

    vial_fields = list(vial_rows[0].keys())
    frame_fields = list(frame_rows[0].keys())
    write_csv(out / "vial_manifest.csv", vial_fields, vial_rows)
    write_csv(out / "frame_labels.csv", frame_fields, frame_rows)
    write_csv(qa_dir / "failures.csv", ["scenario_id", "reason"], failures)

    thumb_w, thumb_h = 240, 180
    sheet = np.full((math.ceil(len(sample_records) / 6) * (thumb_h + 28), 6 * thumb_w, 3), 245, np.uint8)
    for index, (scenario_id, path, frame_index) in enumerate(sample_records):
        image = cv2.imread(str(path)); image = cv2.resize(image, (thumb_w, thumb_h))
        x, y = (index % 6) * thumb_w, (index // 6) * (thumb_h + 28)
        sheet[y:y + thumb_h, x:x + thumb_w] = image
        cv2.putText(sheet, f"{scenario_id} f{frame_index}", (x + 4, y + thumb_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(qa_dir / "contact_sheet.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    report = {"shard": scenarios[0]["shard_id"], "scenarios": len(scenarios),
              "frame_rows": len(frame_rows), "failures": len(failures),
              "passed": len(failures) == 0, "fps": float(scenarios[0]["fps"]),
              "resolution": [rw, rh], "frames_per_scenario": int(scenarios[0]["frame_count"]),
              "renderer": renderer, "subframes": a.subframes}
    (qa_dir / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    config = {"input_manifest": str(manifest_path), "stage": str(resolve(a.stage)),
              "camera_config": str(resolve(a.camera_config)), "warmup": a.warmup,
              "subframes": a.subframes, "renderer": renderer}
    (out / "render_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checksum_lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("checksums.sha256", "READY_SHARD.json"):
            checksum_lines.append(f"{file_sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    if report["passed"]:
        ready = {**report, "manifest_sha256": file_sha256(out / "vial_manifest.csv"),
                 "frame_labels_sha256": file_sha256(out / "frame_labels.csv")}
        (out / "READY_SHARD.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} scenario(s) failed QA")
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()
