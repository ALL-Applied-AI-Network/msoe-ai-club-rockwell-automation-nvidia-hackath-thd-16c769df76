"""Leakage-safe frame cache and PyTorch dataset for synthetic carryover v4."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FRAME_INDICES = (45, 60, 75, 90, 120, 150, 200, 274, 350, 450, 550, 685)
SEVERITY_TO_INDEX = {"clean": 0, "mild": 1, "medium": 2, "severe": 3}
RISK_TO_INDEX = {"low": 0, "medium": 1, "high": 2}
DEFAULT_ROI = (200, 190, 420, 480)  # left, top, right, bottom in 640x480 FillCam frames


def risk_tier(change_score: float) -> str:
    if change_score >= 0.40:
        return "high"
    if change_score >= 0.25:
        return "medium"
    return "low"


def recipe_features(row: dict[str, str]) -> list[float]:
    previous_total = max(float(row["previous_total_ml"]), 1.0)
    current_total = max(float(row["current_total_ml"]), 1.0)
    return [
        float(row["previous_t100_ml"]) / previous_total,
        float(row["previous_t200_ml"]) / previous_total,
        float(row["previous_t400_ml"]) / previous_total,
        float(row["current_t100_ml"]) / current_total,
        float(row["current_t200_ml"]) / current_total,
        float(row["current_t400_ml"]) / current_total,
        float(row["expected_r"]),
        float(row["expected_g"]),
        float(row["expected_b"]),
        1.0,
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_frame_cache(
    dataset_root: str | Path,
    cache_dir: str | Path,
    image_size: int = 160,
    roi: tuple[int, int, int, int] = DEFAULT_ROI,
    max_scenarios_per_split: int | None = None,
) -> Path:
    """Decode selected frames once; never writes inside the immutable dataset."""
    dataset_root = Path(dataset_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode the MP4 training clips")
    ready = dataset_root / "READY.json"
    if not ready.is_file():
        raise FileNotFoundError(f"dataset is not published (missing READY.json): {dataset_root}")
    metadata = json.loads(ready.read_text(encoding="utf-8"))
    if metadata.get("version") != "v4.0.0" or metadata.get("status") != "ready":
        raise ValueError(f"expected ready dataset v4.0.0, got {metadata}")
    manifest_path = cache_dir / "samples.csv"
    if manifest_path.exists():
        return manifest_path
    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    vial_rows = read_rows(dataset_root / "manifests" / "master_vial_manifest.csv")
    selected_vials = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for row in vial_rows:
        split = row["split"]
        if max_scenarios_per_split is not None and split_counts[split] >= max_scenarios_per_split:
            continue
        split_counts[split] += 1
        selected_vials.append(row)
    selected_ids = {row["scenario_id"] for row in selected_vials}

    labels = {}
    frame_label_path = dataset_root / "manifests" / "master_frame_labels.csv"
    with frame_label_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame_index"])
            if row["scenario_id"] in selected_ids and frame_index in FRAME_INDICES:
                labels[(row["scenario_id"], frame_index)] = row

    output_rows: list[dict[str, object]] = []
    target_indices = set(FRAME_INDICES)
    select_filter = "+".join(f"eq(n\\,{index})" for index in FRAME_INDICES)
    for scenario_number, vial in enumerate(selected_vials, 1):
        scenario_id = vial["scenario_id"]
        clip_path = dataset_root / "raw" / "shards" / vial["shard_id"] / vial["clip_path"]
        if not clip_path.is_file():
            raise FileNotFoundError(clip_path)
        cached: dict[int, str] = {}
        with tempfile.TemporaryDirectory(prefix=f"{scenario_id}_", dir=cache_dir) as temporary:
            decoded_pattern = str(Path(temporary) / "frame_%03d.png")
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(clip_path),
                 "-vf", f"select={select_filter}", "-vsync", "0", decoded_pattern],
                text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {clip_path}: {completed.stderr.strip()}")
            decoded = sorted(Path(temporary).glob("frame_*.png"))
            if len(decoded) != len(FRAME_INDICES):
                raise RuntimeError(
                    f"ffmpeg returned {len(decoded)} of {len(FRAME_INDICES)} frames for {scenario_id}"
                )
            left, top, right, bottom = roi
            for frame_index, decoded_path in zip(FRAME_INDICES, decoded):
                with Image.open(decoded_path) as frame:
                    if right > frame.width or bottom > frame.height:
                        raise RuntimeError(f"ROI {roi} exceeds {frame.size} for {scenario_id}")
                    crop = frame.convert("RGB").crop((left, top, right, bottom))
                    crop = crop.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    image_name = f"{scenario_id}_f{frame_index:03d}.jpg"
                    crop.save(images_dir / image_name, format="JPEG", quality=94)
                    cached[frame_index] = image_name
        if set(cached) != target_indices:
            raise RuntimeError(f"missing cached frames for {scenario_id}: {sorted(target_indices - set(cached))}")

        features = recipe_features(vial)
        severity = SEVERITY_TO_INDEX[vial["carryover_class"]]
        tier = risk_tier(float(vial["recipe_change_score"]))
        for selected_index in FRAME_INDICES:
            label = labels[(scenario_id, selected_index)]
            # The immutable source label marks color error before liquid is visible.
            # The deployable target requires both conditions, so empty frames are negative.
            observed = int(label["error_visible"] == "1" and label["liquid_visible"] == "1")
            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "split": vial["split"],
                "frame_index": selected_index,
                "image": f"images/{cached[selected_index]}",
                "liquid_visible": int(label["liquid_visible"]),
                "observed_target": observed,
                "severity_target": severity,
                "risk_target": RISK_TO_INDEX[tier],
                "carryover_class": vial["carryover_class"],
                "recipe_change_score": vial["recipe_change_score"],
            }
            row.update({f"recipe_{index}": value for index, value in enumerate(features)})
            output_rows.append(row)
        print(f"cached {scenario_number}/{len(selected_vials)} {scenario_id}", flush=True)

    write_rows(manifest_path, output_rows)
    cache_metadata = {
        "dataset_version": metadata["version"],
        "dataset_ready": metadata,
        "image_size": image_size,
        "roi": list(roi),
        "frame_indices": list(FRAME_INDICES),
        "scenario_counts": split_counts,
        "sample_rows": len(output_rows),
        "observed_target_rule": "error_visible AND liquid_visible",
    }
    (cache_dir / "cache_metadata.json").write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
    return manifest_path


class CarryoverFrameDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir).resolve()
        self.rows = [row for row in read_rows(self.cache_dir / "samples.csv") if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = np.asarray(Image.open(self.cache_dir / row["image"]).convert("RGB"), dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        image = (image - 0.5) / 0.5
        recipe = torch.tensor([float(row[f"recipe_{i}"]) for i in range(10)], dtype=torch.float32)
        return {
            "image": image,
            "recipe": recipe,
            "risk_target": torch.tensor(int(row["risk_target"]), dtype=torch.long),
            "observed_target": torch.tensor(float(row["observed_target"]), dtype=torch.float32),
            "severity_target": torch.tensor(int(row["severity_target"]), dtype=torch.long),
            "liquid_visible": torch.tensor(int(row["liquid_visible"]), dtype=torch.bool),
            "scenario_id": row["scenario_id"],
            "frame_index": torch.tensor(int(row["frame_index"]), dtype=torch.long),
        }

