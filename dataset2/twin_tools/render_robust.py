"""Robust renderer + simple synthetic-data generator for the CSI vial-line digital twin (Isaac Sim, headless).

Based on tools/render_example.py, with:
  - warm-up updates before the first capture (fixes "RGB capture returned shape (0,)")
  - retry if a capture comes back empty
  - optional liquid / stream color and liquid opacity (set in the session layer; the USD files are not changed)
  - batch mode (--plan CSV): many images from ONE Isaac Sim start, plus a labels.csv describing each image
  - --renderer choice: PathTracing (quality, default) or RayTracedLighting (faster real-time RTX)

Put this file in csi_digital_twin_v1/tools/ (it uses that folder layout).

Single image:
  /isaac-sim/python.sh tools/render_robust.py --camera FillCam --frame 686 \
      --stage stations/fill_station/fill_station.usda --out /job/out/fill.png
Single image, orange liquid:
  ... --liquid-rgb 0.45 0.12 0.01 --out /job/out/orange.png
Batch:
  /isaac-sim/python.sh tools/render_robust.py --camera FillCam \
      --stage stations/fill_station/fill_station.usda --plan test_plan.csv --out-dir /job/out

Plan CSV columns (blank = keep the twin's original value):
  out,frame,liquid_r,liquid_g,liquid_b,stream_r,stream_g,stream_b,opacity
Colors are USD diffuseColor values, 0-1 per channel (linear, not 0-255).
"""
import argparse, csv, json, os, subprocess, sys, time, traceback
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--camera", required=True, choices=["CapSealCam", "FillCam"])
ap.add_argument("--stage", default=os.path.join(HERE, "csi_line.usda"), help="csi_line.usda or one station file")
ap.add_argument("--frame", type=int, default=1, help="time code (fill animation: 1-686 at 30 fps)")
ap.add_argument("--out", default="render.png", help="output image (single-image mode)")
ap.add_argument("--plan", help="CSV with one row per image (batch mode)")
ap.add_argument("--out-dir", default=".", help="folder for batch-mode images and labels.csv")
ap.add_argument("--liquid-rgb", type=float, nargs=3, metavar=("R", "G", "B"))
ap.add_argument("--stream-rgb", type=float, nargs=3, metavar=("R", "G", "B"))
ap.add_argument("--opacity", type=float, help="liquid opacity 0-1")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing subframes (quality)")
ap.add_argument("--renderer", default="PathTracing", choices=["PathTracing", "RayTracedLighting"])
ap.add_argument("--warmup", type=int, default=60, help="renderer updates before the first capture")
ap.add_argument("--max-tries", type=int, default=20, help="capture attempts before giving up")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim / Kit argument (repeatable)")
a = ap.parse_args()

cam = json.load(open(os.path.join(HERE, "cameras", a.camera + ".json")))


def resolve(p):
    p = Path(p).expanduser()
    return str((p if p.is_absolute() else Path(HERE) / p).resolve())


stage_path = resolve(a.stage)
if not os.path.isfile(stage_path):
    ap.error(f"stage does not exist: {stage_path}")


def rgb3(row, prefix):
    vals = [(row.get(f"{prefix}_{c}") or "").strip() for c in "rgb"]
    return tuple(float(v) for v in vals) if all(vals) else None


# ---- the list of images to render
jobs = []
if a.plan:
    with open(resolve(a.plan), newline="") as f:
        for row in csv.DictReader(f):
            op = (row.get("opacity") or "").strip()
            jobs.append({
                "out": str(Path(a.out_dir).expanduser().resolve() / row["out"].strip()),
                "frame": int((row.get("frame") or "").strip() or a.frame),
                "liquid": rgb3(row, "liquid"),
                "stream": rgb3(row, "stream"),
                "opacity": float(op) if op else None,
            })
else:
    jobs.append({"out": str(Path(a.out).expanduser().resolve()), "frame": a.frame,
                 "liquid": tuple(a.liquid_rgb) if a.liquid_rgb else None,
                 "stream": tuple(a.stream_rgb) if a.stream_rgb else None,
                 "opacity": a.opacity})
if not jobs:
    ap.error("nothing to render")
for j in jobs:
    if j["frame"] < 1 or (a.camera == "FillCam" and j["frame"] > 686):
        ap.error(f"bad frame {j['frame']}: the fill animation covers 1-686")
    Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
rw, rh = cam["render_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": a.renderer, "width": rw, "height": rh, "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom, UsdShade

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_path):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    prim = next((p for p in [cam["prim_in_csi_line"], cam["prim_in_station_file"]]
                 if stage.GetPrimAtPath(p).IsValid()), None)
    if prim is None:
        raise RuntimeError(f"camera {a.camera} not found in {stage_path}")

    rp = rep.create.render_product(prim, (rw, rh))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    # Same camera re-sync trick as render_example.py (all edits go to the session layer).
    clip = UsdGeom.Camera(stage.GetPrimAtPath(prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    stage.SetEditTarget(stage.GetSessionLayer())
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1]))
    kit.update()
    clip.Set(near_far)
    kit.update()

    # ---- find the liquid and stream materials (works in the station file and the full line)
    def find_shader(material_name):
        for p in stage.Traverse():
            if p.GetPath().pathString.endswith(f"{material_name}/Principled_BSDF"):
                return UsdShade.Shader(p)
        return None

    liquid_sh = find_shader("USD_blue_liquid_approximation")
    stream_sh = find_shader("USD_jet_approximation")
    print("liquid shader:", liquid_sh.GetPath() if liquid_sh else "NOT FOUND", flush=True)
    print("stream shader:", stream_sh.GetPath() if stream_sh else "NOT FOUND", flush=True)
    orig = {
        "liquid": tuple(liquid_sh.GetInput("diffuseColor").Get()) if liquid_sh else None,
        "opacity": liquid_sh.GetInput("opacity").Get() if liquid_sh else None,
        "stream": tuple(stream_sh.GetInput("diffuseColor").Get()) if stream_sh else None,
    }

    # ---- warm-up: let RTX load the scene and compile shaders before we capture anything
    t0 = time.time()
    for _ in range(a.warmup):
        kit.update()
    print(f"warm-up done ({a.warmup} updates, {time.time() - t0:.1f} s)", flush=True)

    def capture():
        rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)   # discard one, settle changes
        for attempt in range(1, a.max_tries + 1):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            data = np.asarray(ann.get_data())
            if data.ndim == 3 and data.shape[:2] == (rh, rw) and data.shape[2] >= 3:
                return data[..., :3].copy()
            print(f"  capture attempt {attempt}: got shape {data.shape}, retrying", flush=True)
            for _ in range(10):
                kit.update()
        raise RuntimeError(f"no RGB image after {a.max_tries} attempts; check GPU/RTX startup errors")

    def save(img, out):
        if cam.get("overscan", 1.0) != 1.0:      # CapSealCam: add the real lens distortion
            tmp = out + ".pinhole.png"
            if not cv2.imwrite(tmp, img[..., ::-1]):
                raise OSError(f"could not write {tmp}")
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "tools", "apply_lens.py"), tmp,
                                os.path.join(HERE, "cameras", a.camera + ".json"), out], check=True)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        elif not cv2.imwrite(out, img[..., ::-1]):
            raise OSError(f"could not write {out}")

    tl = omni.timeline.get_timeline_interface()
    tcps = stage.GetTimeCodesPerSecond()
    labels = []
    for n, j in enumerate(jobs, 1):
        t1 = time.time()
        liq = j["liquid"] or orig["liquid"]
        stm = j["stream"] or orig["stream"]
        opa = j["opacity"] if j["opacity"] is not None else orig["opacity"]
        if liquid_sh:
            liquid_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*liq))
            liquid_sh.GetInput("opacity").Set(float(opa))
        if stream_sh:
            stream_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*stm))
        tl.set_current_time(j["frame"] / tcps)
        for _ in range(5):
            kit.update()
        save(capture(), j["out"])
        labels.append({"file": os.path.basename(j["out"]), "frame": j["frame"],
                       "liquid_r": liq[0], "liquid_g": liq[1], "liquid_b": liq[2],
                       "stream_r": stm[0], "stream_g": stm[1], "stream_b": stm[2],
                       "opacity": opa, "renderer": a.renderer})
        print(f"[{n}/{len(jobs)}] wrote {j['out']} ({time.time() - t1:.1f} s)", flush=True)

    if a.plan:
        lab_path = Path(a.out_dir).expanduser().resolve() / "labels.csv"
        with open(lab_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
            w.writeheader(); w.writerows(labels)
        print("wrote", lab_path, flush=True)
except BaseException:
    traceback.print_exc()
    # Preserve failure status even when Kit's fast shutdown terminates Python.
    kit.close(exit_code=1)
    raise
kit.close()
