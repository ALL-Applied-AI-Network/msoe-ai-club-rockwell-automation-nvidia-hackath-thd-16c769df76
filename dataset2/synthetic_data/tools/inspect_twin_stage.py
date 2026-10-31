#!/usr/bin/env python3
"""List cameras and tank-like prims in the CSI digital twin."""

import argparse

from isaacsim import SimulationApp


ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
args = ap.parse_args()

kit = SimulationApp({"headless": True})
try:
    import omni.usd
    from pxr import UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(args.stage):
        raise RuntimeError(f"could not open {args.stage}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    stage = ctx.get_stage()
    print("CAMERAS", flush=True)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            print(prim.GetPath(), flush=True)
    print("TANK_LIKE_PRIMS", flush=True)
    needles = ("tank", "reservoir", "reagent", "liquid", "fluid", "level")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(word in path.lower() for word in needles):
            print(prim.GetTypeName(), path, flush=True)
finally:
    kit.close()
