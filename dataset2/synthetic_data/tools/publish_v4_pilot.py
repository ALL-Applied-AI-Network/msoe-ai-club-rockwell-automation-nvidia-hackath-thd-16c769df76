#!/usr/bin/env python3
"""Verify and publish the eight v4 pilot shards as an immutable dataset version."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


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
        writer.writeheader(); writer.writerows(rows)


def verify_checksums(root):
    lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"checksum failed: {path}")
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--version", default="v4.0.0-pilot")
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--shard", action="append", required=True,
                        help="SHARD_ID=/absolute/result/out")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    target = dataset_root / args.version
    if target.exists():
        raise FileExistsError(f"refusing to overwrite published/staged version: {target}")
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
    plan_dir = Path(args.plan_dir).expanduser().resolve()
    shutil.copytree(plan_dir, manifests_dir / "collection_plan")

    vial_rows, frame_rows, qa_reports, source_jobs = [], [], [], []
    for shard_id in expected_shards:
        source = mappings[shard_id]
        ready_path = source / "READY_SHARD.json"
        if not ready_path.is_file():
            raise RuntimeError(f"missing READY_SHARD.json: {source}")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if not ready.get("passed") or ready.get("failures"):
            raise RuntimeError(f"shard did not pass QA: {shard_id}")
        checksum_files = verify_checksums(source)
        copied = raw_dir / shard_id
        shutil.copytree(source, copied)
        copied_checks = verify_checksums(copied)
        if copied_checks != checksum_files:
            raise RuntimeError(f"checksum count changed after copy: {shard_id}")
        vial_rows.extend(read_csv(copied / "vial_manifest.csv"))
        frame_rows.extend(read_csv(copied / "frame_labels.csv"))
        report = json.loads((copied / "qa" / "qa_report.json").read_text(encoding="utf-8"))
        qa_reports.append(report)
        shutil.copy2(copied / "qa" / "contact_sheet.jpg", qa_contacts / f"{shard_id}.jpg")
        source_jobs.append({"shard_id": shard_id, "source_result": str(source),
                            "ready_sha256": file_sha256(ready_path)})

    scenario_ids = [row["scenario_id"] for row in vial_rows]
    if len(vial_rows) != 48 or len(set(scenario_ids)) != 48:
        raise RuntimeError("expected 48 unique vial scenarios")
    if len(frame_rows) != 48 * 686:
        raise RuntimeError(f"expected {48 * 686} frame rows, got {len(frame_rows)}")
    frame_counts = Counter(row["scenario_id"] for row in frame_rows)
    if set(frame_counts) != set(scenario_ids) or set(frame_counts.values()) != {686}:
        raise RuntimeError("frame rows are not exactly 686 per scenario")
    class_counts = Counter(row["carryover_class"] for row in vial_rows)
    palette_counts = Counter(row["palette_id"] for row in vial_rows)
    transition_counts = Counter((row["previous_recipe_id"], row["current_recipe_id"]) for row in vial_rows)
    if class_counts != Counter({"clean": 16, "medium": 16, "severe": 16}):
        raise RuntimeError(f"unexpected class balance: {class_counts}")
    if palette_counts != Counter({"pilot_base": 24, "pilot_alt": 24}):
        raise RuntimeError(f"unexpected palette balance: {palette_counts}")
    if set(transition_counts.values()) != {6} or len(transition_counts) != 8:
        raise RuntimeError(f"unexpected transition balance: {transition_counts}")
    if any(row.get("qa_passed") != "1" for row in vial_rows):
        raise RuntimeError("one or more scenario rows did not pass QA")

    vial_rows.sort(key=lambda row: row["scenario_id"])
    frame_rows.sort(key=lambda row: (row["scenario_id"], int(row["frame_index"])))
    write_csv(manifests_dir / "master_vial_manifest.csv", list(vial_rows[0].keys()), vial_rows)
    write_csv(manifests_dir / "master_frame_labels.csv", list(frame_rows[0].keys()), frame_rows)
    (manifests_dir / "source_jobs.json").write_text(json.dumps(source_jobs, indent=2), encoding="utf-8")
    summary = {
        "version": args.version, "scenarios": 48, "frame_rows": len(frame_rows),
        "classes": dict(class_counts), "palettes": dict(palette_counts),
        "transitions": {f"{a}->{b}": count for (a, b), count in transition_counts.items()},
        "shards": qa_reports, "all_shards_passed": all(row["passed"] for row in qa_reports),
    }
    (target / "qa" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "README.md").write_text(
        "# Synthetic carryover v4.0.0 pilot\n\n"
        "Immutable 48-scenario pilot: eight recipe transitions × three residual conditions × two palettes. "
        "Every clip contains 686 unique Isaac Sim FillCam frames at 30 fps and 640×480. "
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
        "scenarios": 48, "frame_rows": len(frame_rows), "total_bytes": total_bytes,
        "vial_manifest_sha256": file_sha256(manifests_dir / "master_vial_manifest.csv"),
        "frame_labels_sha256": file_sha256(manifests_dir / "master_frame_labels.csv"),
        "checksums_sha256": file_sha256(target / "checksums.sha256"),
    }
    (target / "READY.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")

    dataset_root.mkdir(parents=True, exist_ok=True)
    registry_path = dataset_root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"versions": []}
    if any(item.get("version") == args.version for item in registry["versions"]):
        raise RuntimeError(f"registry already contains {args.version}")
    registry["versions"].append({**ready, "path": str(target),
                                 "change": "48-scenario residual-volume pilot"})
    temp_registry = dataset_root / "registry.json.tmp"
    temp_registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    os.replace(temp_registry, registry_path)
    print(json.dumps({"target": str(target), **ready, "class_counts": dict(class_counts),
                      "palette_counts": dict(palette_counts)}, indent=2))


if __name__ == "__main__":
    main()
