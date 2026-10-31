#!/usr/bin/env python3
"""Verify and publish eight production shards as immutable dataset v4.0.0."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CLASSES = Counter({"clean": 240, "mild": 80, "medium": 80, "severe": 80})
EXPECTED_SPLITS = Counter({"train": 320, "validation": 80, "test": 80})
RESIDUAL_RANGES = {"clean": (0.0, 0.0), "mild": (0.25, 1.5),
                   "medium": (1.5, 4.0), "severe": (4.0, 8.0)}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_checksums(root):
    lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"checksum failed: {path}")
    return len(lines)


def assert_no_split_leakage(rows):
    transitions = {}
    palettes = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        transitions[split] = {(row["previous_recipe_id"], row["current_recipe_id"])
                              for row in split_rows}
        palettes[split] = {row["palette_id"] for row in split_rows}
    comparisons = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for first, second in comparisons:
        if transitions[first] & transitions[second]:
            raise RuntimeError(f"transition leakage between {first} and {second}")
        if palettes[first] & palettes[second]:
            raise RuntimeError(f"palette leakage between {first} and {second}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--version", default="v4.0.0")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--shard", action="append", required=True,
                        help="SHARD_ID=/absolute/result/out")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    target = dataset_root / args.version
    if target.exists():
        raise FileExistsError(f"refusing to overwrite version: {target}")
    plan_dir = Path(args.plan_dir).expanduser().resolve()
    verify_checksums(plan_dir)
    plan_rows = read_csv(plan_dir / "master_manifest.csv")
    if len(plan_rows) != 480:
        raise RuntimeError("frozen plan does not contain 480 scenarios")

    mappings = {}
    for value in args.shard:
        shard_id, path = value.split("=", 1)
        mappings[shard_id] = Path(path).expanduser().resolve()
    expected_shards = [f"shard_{index:02d}" for index in range(8)]
    if sorted(mappings) != expected_shards:
        raise ValueError(f"expected {expected_shards}, got {sorted(mappings)}")

    raw_dir = target / "raw" / "shards"
    manifests_dir = target / "manifests"
    qa_contacts = target / "qa" / "contact_sheets"
    raw_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    qa_contacts.mkdir(parents=True)
    shutil.copytree(plan_dir, manifests_dir / "collection_plan")

    vial_rows, frame_rows, qa_reports, source_jobs = [], [], [], []
    for shard_id in expected_shards:
        source = mappings[shard_id]
        ready_path = source / "READY_SHARD.json"
        if not ready_path.is_file():
            raise RuntimeError(f"missing READY_SHARD.json: {source}")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if not ready.get("passed") or ready.get("failures") or ready.get("scenarios") != 60:
            raise RuntimeError(f"shard did not pass production QA: {shard_id}")
        checksum_files = verify_checksums(source)
        copied = raw_dir / shard_id
        shutil.copytree(source, copied)
        if verify_checksums(copied) != checksum_files:
            raise RuntimeError(f"checksum count changed after copy: {shard_id}")
        shard_vials = read_csv(copied / "vial_manifest.csv")
        shard_frames = read_csv(copied / "frame_labels.csv")
        if len(shard_vials) != 60 or len(shard_frames) != 60 * 686:
            raise RuntimeError(f"wrong row counts in {shard_id}")
        vial_rows.extend(shard_vials)
        frame_rows.extend(shard_frames)
        report = json.loads((copied / "qa" / "qa_report.json").read_text(encoding="utf-8"))
        qa_reports.append(report)
        shutil.copy2(copied / "qa" / "contact_sheet.jpg", qa_contacts / f"{shard_id}.jpg")
        source_jobs.append({"shard_id": shard_id, "source_result": str(source),
                            "job_id": source.parent.name,
                            "ready_sha256": file_sha256(ready_path)})

    scenario_ids = [row["scenario_id"] for row in vial_rows]
    if len(vial_rows) != 480 or len(set(scenario_ids)) != 480:
        raise RuntimeError("expected 480 unique vial scenarios")
    if set(scenario_ids) != {row["scenario_id"] for row in plan_rows}:
        raise RuntimeError("rendered scenarios do not match the frozen plan")
    if len(frame_rows) != 480 * 686:
        raise RuntimeError(f"expected {480 * 686} frame rows, got {len(frame_rows)}")
    frame_counts = Counter(row["scenario_id"] for row in frame_rows)
    if set(frame_counts) != set(scenario_ids) or set(frame_counts.values()) != {686}:
        raise RuntimeError("frame rows are not exactly 686 per scenario")
    class_counts = Counter(row["carryover_class"] for row in vial_rows)
    split_counts = Counter(row["split"] for row in vial_rows)
    if class_counts != EXPECTED_CLASSES:
        raise RuntimeError(f"unexpected class balance: {class_counts}")
    if split_counts != EXPECTED_SPLITS:
        raise RuntimeError(f"unexpected split balance: {split_counts}")
    transition_counts = Counter((row["previous_recipe_id"], row["current_recipe_id"])
                                for row in vial_rows)
    if len(transition_counts) != 60 or set(transition_counts.values()) != {8}:
        raise RuntimeError(f"unexpected transition balance: {transition_counts}")
    if len({row["palette_id"] for row in vial_rows}) != 12:
        raise RuntimeError("expected all 12 palettes")
    if any(row.get("qa_passed") != "1" for row in vial_rows):
        raise RuntimeError("one or more scenario rows did not pass QA")
    for row in vial_rows:
        low, high = RESIDUAL_RANGES[row["carryover_class"]]
        value = float(row["residual_ml"])
        if not low <= value <= high:
            raise RuntimeError(f"residual out of range for {row['scenario_id']}: {value}")
    assert_no_split_leakage(vial_rows)

    vial_rows.sort(key=lambda row: row["scenario_id"])
    frame_rows.sort(key=lambda row: (row["scenario_id"], int(row["frame_index"])))
    write_csv(manifests_dir / "master_vial_manifest.csv", list(vial_rows[0].keys()), vial_rows)
    write_csv(manifests_dir / "master_frame_labels.csv", list(frame_rows[0].keys()), frame_rows)
    (manifests_dir / "source_jobs.json").write_text(json.dumps(source_jobs, indent=2), encoding="utf-8")
    palette_counts = Counter(row["palette_id"] for row in vial_rows)
    summary = {
        "version": args.version, "scenarios": 480, "frame_rows": len(frame_rows),
        "classes": dict(class_counts), "splits": dict(split_counts),
        "palettes": dict(palette_counts), "transitions": len(transition_counts),
        "transition_leakage": False, "palette_leakage": False,
        "shards": qa_reports, "all_shards_passed": all(row["passed"] for row in qa_reports),
    }
    (target / "qa" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "README.md").write_text(
        "# Synthetic carryover v4.0.0\n\n"
        "Immutable 480-scenario production collection with leakage-safe transition and palette splits. "
        "Every clip contains 686 independently rendered Isaac Sim FillCam frames at 30 fps and 640x480. "
        "Use manifests/master_vial_manifest.csv and manifests/master_frame_labels.csv.\n",
        encoding="utf-8")

    checksum_lines = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name not in ("checksums.sha256", "READY.json"):
            checksum_lines.append(f"{file_sha256(path)}  {path.relative_to(target).as_posix()}")
    (target / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    total_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    ready = {
        "version": args.version, "status": "ready", "created_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": 480, "frame_rows": len(frame_rows), "total_bytes": total_bytes,
        "vial_manifest_sha256": file_sha256(manifests_dir / "master_vial_manifest.csv"),
        "frame_labels_sha256": file_sha256(manifests_dir / "master_frame_labels.csv"),
        "checksums_sha256": file_sha256(target / "checksums.sha256"),
    }
    (target / "READY.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")

    registry_path = dataset_root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"versions": []}
    if any(item.get("version") == args.version for item in registry["versions"]):
        raise RuntimeError(f"registry already contains {args.version}")
    registry["versions"].append({**ready, "path": str(target),
                                 "change": "480-scenario leakage-safe production collection"})
    temp_registry = dataset_root / "registry.json.tmp"
    temp_registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    os.replace(temp_registry, registry_path)
    print(json.dumps({"target": str(target), **ready, "class_counts": dict(class_counts),
                      "split_counts": dict(split_counts)}, indent=2))


if __name__ == "__main__":
    main()
