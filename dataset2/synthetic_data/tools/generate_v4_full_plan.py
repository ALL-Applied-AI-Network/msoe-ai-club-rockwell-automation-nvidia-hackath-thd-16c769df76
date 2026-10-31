#!/usr/bin/env python3
"""Create the deterministic, leakage-safe 480-scenario v4 production plan."""

import argparse
import colorsys
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def recipe(t100, t200, t400, name):
    total = t100 + t200 + t400
    assert total <= 80 and t200 <= 50
    assert all(value == 0 or value >= 10 for value in (t100, t200, t400))
    return {"t100_ml": t100, "t200_ml": t200, "t400_ml": t400,
            "total_ml": total, "name": name}


RECIPES = {
    5: recipe(10, 10, 10, "mixed low even"),
    6: recipe(30, 10, 0, "pale orange"),
    7: recipe(20, 0, 50, "pale blue"),
    8: recipe(20, 20, 20, "even mixture"),
    9: recipe(10, 35, 35, "dark saturated"),
    10: recipe(45, 15, 15, "pale wash"),
    11: recipe(10, 10, 50, "blue dominant"),
    12: recipe(10, 50, 10, "orange dominant"),
    13: recipe(0, 50, 20, "orange blue pair"),
    14: recipe(60, 0, 10, "very pale blue"),
    15: recipe(0, 10, 10, "small orange blue"),
    16: recipe(10, 0, 10, "small clear blue"),
    17: recipe(10, 10, 0, "small clear orange"),
    18: recipe(40, 10, 10, "dilute balanced"),
    19: recipe(10, 40, 10, "strong orange"),
    20: recipe(10, 10, 40, "strong blue"),
    21: recipe(0, 30, 30, "saturated balanced"),
    22: recipe(30, 0, 30, "clear blue balanced"),
    23: recipe(30, 30, 0, "clear orange balanced"),
    24: recipe(20, 40, 20, "large orange mix"),
    25: recipe(20, 20, 40, "large blue mix"),
    26: recipe(40, 20, 20, "large dilute mix"),
    27: recipe(0, 50, 30, "maximum orange saturated"),
    28: recipe(30, 0, 50, "maximum blue dilute"),
}


PALETTES = {
    "train_base": ([0.12, 0.22, 0.24], [0.45, 0.12, 0.01], [0.01, 0.20, 0.36]),
    "train_teal": ([0.16, 0.24, 0.22], [0.55, 0.09, 0.015], [0.015, 0.30, 0.30]),
    "train_amber": ([0.14, 0.23, 0.23], [0.50, 0.16, 0.015], [0.012, 0.18, 0.34]),
    "train_coral": ([0.13, 0.21, 0.24], [0.57, 0.11, 0.025], [0.018, 0.23, 0.37]),
    "train_cyan": ([0.15, 0.23, 0.25], [0.47, 0.10, 0.012], [0.008, 0.29, 0.39]),
    "train_soft": ([0.18, 0.25, 0.25], [0.43, 0.14, 0.025], [0.025, 0.24, 0.34]),
    "train_rust": ([0.13, 0.22, 0.23], [0.52, 0.105, 0.008], [0.014, 0.19, 0.33]),
    "train_ocean": ([0.145, 0.235, 0.24], [0.46, 0.135, 0.018], [0.008, 0.26, 0.42]),
    "validation_gold": ([0.17, 0.25, 0.24], [0.58, 0.18, 0.02], [0.01, 0.22, 0.38]),
    "validation_aqua": ([0.14, 0.23, 0.26], [0.49, 0.115, 0.01], [0.01, 0.33, 0.36]),
    "test_copper": ([0.155, 0.24, 0.235], [0.60, 0.13, 0.018], [0.016, 0.21, 0.40]),
    "test_lagoon": ([0.165, 0.245, 0.25], [0.48, 0.15, 0.022], [0.006, 0.35, 0.34]),
}

SPLIT_PALETTES = {
    "train": [name for name in PALETTES if name.startswith("train_")],
    "validation": [name for name in PALETTES if name.startswith("validation_")],
    "test": [name for name in PALETTES if name.startswith("test_")],
}

CLASS_RANGES = {
    "clean": (0.0, 0.0, "full"),
    "mild": (0.25, 1.5, "partial"),
    "medium": (1.5, 4.0, "partial"),
    "severe": (4.0, 8.0, "none"),
}


def clamp(value, low=0.002, high=0.80):
    return max(low, min(high, value))


def recipe_rgb(value, palette):
    t100, t200, t400 = palette
    colored = value["t200_ml"] + value["t400_ml"]
    dye = [
        (value["t200_ml"] * t200[i] + value["t400_ml"] * t400[i]) / colored
        for i in range(3)
    ] if colored else list(t100)
    dilution = value["t100_ml"] / value["total_ml"]
    return [clamp((1.0 - 0.70 * dilution) * dye[i] + 0.70 * dilution * t100[i])
            for i in range(3)]


def change_score(previous, current):
    def ratios(item):
        return [item["t200_ml"] / item["total_ml"], item["t400_ml"] / item["total_ml"],
                item["t100_ml"] / item["total_ml"]]
    return 0.5 * sum(abs(a - b) for a, b in zip(ratios(previous), ratios(current)))


def rgb_details(rgb):
    hsv = colorsys.rgb_to_hsv(*rgb)
    rgb255 = [max(0, min(255, round(channel * 255))) for channel in rgb]
    return {
        "expected_h": f"{hsv[0]:.8f}", "expected_s": f"{hsv[1]:.8f}",
        "expected_v": f"{hsv[2]:.8f}", "expected_hue_deg": f"{hsv[0] * 360:.4f}",
        "expected_hex": "#" + "".join(f"{channel:02X}" for channel in rgb255),
    }


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_transitions(rng):
    all_pairs = [(a, b, change_score(RECIPES[a], RECIPES[b]))
                 for a in RECIPES for b in RECIPES]
    same = [(a, b, score) for a, b, score in all_pairs if a == b]
    small = [(a, b, score) for a, b, score in all_pairs if a != b and score <= 0.20]
    medium = [(a, b, score) for a, b, score in all_pairs if 0.20 < score <= 0.50]
    large = [(a, b, score) for a, b, score in all_pairs if score > 0.50]
    for group in (same, small, medium, large):
        rng.shuffle(group)
    selected = same[:6] + small[:9] + medium[:25] + large[:20]
    if len({(a, b) for a, b, _ in selected}) != 60:
        raise RuntimeError("transition selection was not unique")
    rng.shuffle(selected)
    return selected


def domain_variants():
    variants = []
    for index in range(16):
        variants.append({
            "domain_variant_id": f"dv_{index:02d}",
            "exposure_ev": f"{(-0.10 + (index % 5) * 0.05):.3f}",
            "contrast": f"{(0.96 + (index % 3) * 0.04):.3f}",
            "wb_r": f"{(0.97 + (index % 4) * 0.02):.3f}",
            "wb_g": "1.000",
            "wb_b": f"{(1.03 - (index % 4) * 0.02):.3f}",
            "sensor_noise_std": f"{((index % 4) * 0.5):.3f}",
            "blur_sigma": f"{((index % 3) * 0.20):.3f}",
            "camera_dx_px": str((index % 5) - 2),
            "camera_dy_px": str(((index * 2) % 5) - 2),
        })
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.shards != 8:
        raise ValueError("the frozen production design requires exactly eight shards")
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {out}")
    (out / "shards").mkdir(parents=True)
    rng = random.Random(args.seed)

    recipe_rows = [{"recipe_id": key, "t300_ml": 0, **value} for key, value in RECIPES.items()]
    write_csv(out / "recipes.csv", list(recipe_rows[0].keys()), recipe_rows)
    palette_records = {}
    for name, (t100, t200, t400) in PALETTES.items():
        palette_records[name] = {"T100": t100, "T200": t200, "T400": t400,
                                 "split": name.split("_", 1)[0]}
    (out / "palettes.json").write_text(json.dumps(palette_records, indent=2), encoding="utf-8")

    transitions = select_transitions(rng)
    split_for_index = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
    rows = []
    variants = domain_variants()
    scenario_number = 0
    extra_classes = ["mild", "medium", "severe"] * 20
    for transition_index, ((previous_id, current_id, score), split) in enumerate(zip(transitions, split_for_index)):
        classes = ["clean"] * 4 + ["mild", "medium", "severe", extra_classes[transition_index]]
        rng.shuffle(classes)
        palettes = SPLIT_PALETTES[split]
        palette_sequence = ([palettes[(transition_index + i) % len(palettes)] for i in range(8)])
        rng.shuffle(palette_sequence)
        for local_index, (carryover_class, palette_id) in enumerate(zip(classes, palette_sequence)):
            scenario_number += 1
            previous, current = RECIPES[previous_id], RECIPES[current_id]
            palette = PALETTES[palette_id]
            previous_rgb = recipe_rgb(previous, palette)
            expected_rgb = recipe_rgb(current, palette)
            low, high, flush_state = CLASS_RANGES[carryover_class]
            residual_ml = 0.0 if carryover_class == "clean" else rng.uniform(low, high)
            opacity = rng.uniform(0.60, 0.68)
            variant = variants[(scenario_number + transition_index + local_index) % len(variants)]
            row = {
                "dataset_version": "v4.0.0", "scenario_id": f"prod_{scenario_number:04d}",
                "split": split, "batch_id": f"{split}_batch_{transition_index:02d}",
                "seed": args.seed + scenario_number, "transition_id": f"tr_{transition_index:02d}",
                "previous_recipe_id": previous_id, "current_recipe_id": current_id,
                "previous_t100_ml": previous["t100_ml"], "previous_t200_ml": previous["t200_ml"],
                "previous_t400_ml": previous["t400_ml"], "previous_total_ml": previous["total_ml"],
                "current_t100_ml": current["t100_ml"], "current_t200_ml": current["t200_ml"],
                "current_t400_ml": current["t400_ml"], "current_total_ml": current["total_ml"],
                "recipe_change_score": f"{score:.8f}", "flush_state": flush_state,
                "residual_ml": f"{residual_ml:.6f}", "carryover_class": carryover_class,
                "palette_id": palette_id,
                "previous_r": f"{previous_rgb[0]:.8f}", "previous_g": f"{previous_rgb[1]:.8f}",
                "previous_b": f"{previous_rgb[2]:.8f}", "expected_r": f"{expected_rgb[0]:.8f}",
                "expected_g": f"{expected_rgb[1]:.8f}", "expected_b": f"{expected_rgb[2]:.8f}",
                **rgb_details(expected_rgb), "opacity": f"{opacity:.4f}", **variant,
                "renderer": "RayTracedLighting", "fps": "30", "frame_count": "686",
                "shard_id": "",
            }
            rows.append(row)

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["shard_id"] = f"shard_{index % args.shards:02d}"
    rows.sort(key=lambda row: row["scenario_id"])
    fields = list(rows[0].keys())
    write_csv(out / "master_manifest.csv", fields, rows)
    for shard_index in range(args.shards):
        shard_id = f"shard_{shard_index:02d}"
        write_csv(out / "shards" / f"{shard_id}.csv", fields,
                  [row for row in rows if row["shard_id"] == shard_id])

    transition_splits = defaultdict(list)
    for index, (previous_id, current_id, score) in enumerate(transitions):
        transition_splits[split_for_index[index]].append({
            "transition_id": f"tr_{index:02d}", "previous_recipe_id": previous_id,
            "current_recipe_id": current_id, "change_score": round(score, 8),
        })
    split_definition = {
        "dataset_version": "v4.0.0", "status": "frozen", "seed": args.seed,
        "scenarios": len(rows), "shards": args.shards,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "class_counts": dict(Counter(row["carryover_class"] for row in rows)),
        "transitions": transition_splits, "palette_splits": SPLIT_PALETTES,
        "residual_ranges_ml": {name: [low, high] for name, (low, high, _) in CLASS_RANGES.items()},
    }
    (out / "split_definition.json").write_text(json.dumps(split_definition, indent=2), encoding="utf-8")

    pairs_by_split = {split: {(r["previous_recipe_id"], r["current_recipe_id"])
                              for r in rows if r["split"] == split}
                      for split in SPLIT_PALETTES}
    if any(pairs_by_split[a] & pairs_by_split[b] for a, b in (("train", "validation"),
                                                               ("train", "test"),
                                                               ("validation", "test"))):
        raise RuntimeError("transition leakage detected")
    used_palettes = {split: {r["palette_id"] for r in rows if r["split"] == split}
                     for split in SPLIT_PALETTES}
    if any(used_palettes[a] & used_palettes[b] for a, b in (("train", "validation"),
                                                             ("train", "test"),
                                                             ("validation", "test"))):
        raise RuntimeError("palette leakage detected")
    expected_classes = {"clean": 240, "mild": 80, "medium": 80, "severe": 80}
    if dict(Counter(row["carryover_class"] for row in rows)) != expected_classes:
        raise RuntimeError("class distribution does not match frozen target")
    if any(sum(row["shard_id"] == f"shard_{index:02d}" for row in rows) != 60
           for index in range(args.shards)):
        raise RuntimeError("shards are not exactly balanced")

    summary = {
        "dataset_version": "v4.0.0", "status": "frozen", "scenarios": len(rows),
        "classes": expected_classes, "splits": dict(Counter(row["split"] for row in rows)),
        "palettes": len(PALETTES), "recipes": len(RECIPES), "transitions": len(transitions),
        "shards": {f"shard_{i:02d}": 60 for i in range(args.shards)},
        "transition_leakage": False, "palette_leakage": False,
    }
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# v4.0.0 production collection plan\n\n"
        "Frozen deterministic plan for 480 FillCam scenarios. Ordered transitions and palettes are "
        "held out by split. Do not edit after rendering begins; corrections require a new version.\n",
        encoding="utf-8")
    checksum_lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), **summary}, indent=2))


if __name__ == "__main__":
    main()
