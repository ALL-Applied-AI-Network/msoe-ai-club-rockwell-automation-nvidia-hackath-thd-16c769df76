#!/usr/bin/env python3
"""Create the deterministic 48-scenario v4 carryover pilot plan."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


RECIPES = {
    5:  {"t100_ml": 10, "t200_ml": 10, "t400_ml": 10, "total_ml": 30, "name": "mixed low even"},
    6:  {"t100_ml": 30, "t200_ml": 10, "t400_ml": 0,  "total_ml": 40, "name": "pale orange"},
    7:  {"t100_ml": 20, "t200_ml": 0,  "t400_ml": 50, "total_ml": 70, "name": "pale blue"},
    8:  {"t100_ml": 20, "t200_ml": 20, "t400_ml": 20, "total_ml": 60, "name": "even mixture"},
    9:  {"t100_ml": 10, "t200_ml": 35, "t400_ml": 35, "total_ml": 80, "name": "dark saturated"},
    10: {"t100_ml": 45, "t200_ml": 15, "t400_ml": 15, "total_ml": 75, "name": "pale wash"},
    11: {"t100_ml": 10, "t200_ml": 10, "t400_ml": 50, "total_ml": 70, "name": "blue dominant"},
    12: {"t100_ml": 10, "t200_ml": 50, "t400_ml": 10, "total_ml": 70, "name": "orange dominant"},
    13: {"t100_ml": 0,  "t200_ml": 50, "t400_ml": 20, "total_ml": 70, "name": "orange blue pair"},
    14: {"t100_ml": 60, "t200_ml": 0,  "t400_ml": 10, "total_ml": 70, "name": "very pale blue"},
}

PALETTES = {
    "pilot_base": {
        "T100": [0.12, 0.22, 0.24],
        "T200": [0.45, 0.12, 0.01],
        "T400": [0.01, 0.20, 0.36],
        "description": "canonical clear-orange-blue approximation",
    },
    "pilot_alt": {
        "T100": [0.16, 0.24, 0.22],
        "T200": [0.55, 0.09, 0.015],
        "T400": [0.015, 0.30, 0.30],
        "description": "red-orange and teal-blue generalization palette",
    },
}

TRANSITIONS = [(5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 14)]
RESIDUALS = [(0.0, "clean", "full"), (2.0, "medium", "partial"), (6.0, "severe", "none")]


def clamp(value, low=0.002, high=0.80):
    return max(low, min(high, value))


def recipe_rgb(recipe, palette):
    colored = recipe["t200_ml"] + recipe["t400_ml"]
    if colored:
        dye = [
            (recipe["t200_ml"] * palette["T200"][i] + recipe["t400_ml"] * palette["T400"][i]) / colored
            for i in range(3)
        ]
    else:
        dye = palette["T100"]
    dilution = recipe["t100_ml"] / recipe["total_ml"]
    return [clamp((1.0 - 0.70 * dilution) * dye[i] + 0.70 * dilution * palette["T100"][i])
            for i in range(3)]


def change_score(previous, current):
    prev = [previous["t200_ml"] / previous["total_ml"], previous["t400_ml"] / previous["total_ml"],
            previous["t100_ml"] / previous["total_ml"]]
    curr = [current["t200_ml"] / current["total_ml"], current["t400_ml"] / current["total_ml"],
            current["t100_ml"] / current["total_ml"]]
    return 0.5 * sum(abs(a - b) for a, b in zip(prev, curr))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {out}")
    (out / "shards").mkdir(parents=True)

    recipe_rows = [{"recipe_id": key, **value} for key, value in RECIPES.items()]
    write_csv(out / "recipes.csv", list(recipe_rows[0].keys()), recipe_rows)
    (out / "palettes.json").write_text(json.dumps(PALETTES, indent=2), encoding="utf-8")

    rows = []
    counter = 0
    for previous_id, current_id in TRANSITIONS:
        for residual_ml, carryover_class, flush_state in RESIDUALS:
            for palette_id, palette in PALETTES.items():
                counter += 1
                previous, current = RECIPES[previous_id], RECIPES[current_id]
                prev_rgb = recipe_rgb(previous, palette)
                expected_rgb = recipe_rgb(current, palette)
                rows.append({
                    "scenario_id": f"pilot_{counter:04d}", "split": "pilot", "seed": args.seed + counter,
                    "previous_recipe_id": previous_id, "current_recipe_id": current_id,
                    "previous_t100_ml": previous["t100_ml"], "previous_t200_ml": previous["t200_ml"],
                    "previous_t400_ml": previous["t400_ml"], "previous_total_ml": previous["total_ml"],
                    "current_t100_ml": current["t100_ml"], "current_t200_ml": current["t200_ml"],
                    "current_t400_ml": current["t400_ml"], "current_total_ml": current["total_ml"],
                    "recipe_change_score": f"{change_score(previous, current):.8f}",
                    "flush_state": flush_state, "residual_ml": f"{residual_ml:.3f}",
                    "carryover_class": carryover_class, "palette_id": palette_id,
                    "previous_r": f"{prev_rgb[0]:.8f}", "previous_g": f"{prev_rgb[1]:.8f}",
                    "previous_b": f"{prev_rgb[2]:.8f}", "expected_r": f"{expected_rgb[0]:.8f}",
                    "expected_g": f"{expected_rgb[1]:.8f}", "expected_b": f"{expected_rgb[2]:.8f}",
                    "opacity": "0.6400", "renderer": "RayTracedLighting", "fps": "30",
                    "frame_count": "686", "shard_id": "",
                })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["shard_id"] = f"shard_{index % args.shards:02d}"
    rows.sort(key=lambda row: row["scenario_id"])
    fields = list(rows[0].keys())
    write_csv(out / "master_manifest.csv", fields, rows)
    for shard_index in range(args.shards):
        shard_id = f"shard_{shard_index:02d}"
        shard_rows = [row for row in rows if row["shard_id"] == shard_id]
        write_csv(out / "shards" / f"{shard_id}.csv", fields, shard_rows)

    split = {
        "dataset_version": "v4.0.0-pilot", "status": "planned", "seed": args.seed,
        "scenarios": len(rows), "shards": args.shards, "transitions": TRANSITIONS,
        "residual_conditions_ml": [item[0] for item in RESIDUALS],
        "palette_ids": list(PALETTES), "split": "pilot",
    }
    (out / "split_definition.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    readme = (
        "# v4.0.0 pilot plan\n\n"
        "Forty-eight controlled FillCam scenarios: eight ordered recipe transitions, three residual "
        "conditions, and two palettes. This plan is deterministic and must not be edited after rendering begins.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    checksums = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "scenarios": len(rows), "shards": args.shards,
                      "per_shard": [sum(r["shard_id"] == f"shard_{i:02d}" for r in rows)
                                    for i in range(args.shards)]}, indent=2))


if __name__ == "__main__":
    main()
