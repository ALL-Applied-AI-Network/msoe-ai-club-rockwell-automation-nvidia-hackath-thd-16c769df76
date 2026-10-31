#!/usr/bin/env python3
"""Replay completed monitor JSONL files into the operator dashboard inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path


def load_records(path: Path, stream: str) -> list[tuple[datetime, str, str]]:
    records: list[tuple[datetime, str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            timestamp = datetime.fromisoformat(payload["time"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {path} line {line_number}: {exc}") from exc
        records.append((timestamp, stream, raw_line + "\n"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-source", type=Path, required=True)
    parser.add_argument("--neural-source", type=Path, required=True)
    parser.add_argument("--events-dest", type=Path, required=True)
    parser.add_argument("--neural-dest", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120.0,
                        help="target wall-clock replay duration in seconds")
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()

    timeline = load_records(args.events_source, "events")
    timeline += load_records(args.neural_source, "neural")
    timeline.sort(key=lambda item: (item[0], 0 if item[1] == "neural" else 1))
    if not timeline:
        raise SystemExit("source logs are empty")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = args.archive_root / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for destination in (args.events_dest, args.neural_dest):
        if destination.exists():
            shutil.copy2(destination, archive / destination.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")

    source_span = max((timeline[-1][0] - timeline[0][0]).total_seconds(), 1.0)
    speed = source_span / max(args.duration, 1.0)
    started = time.monotonic()
    base_timestamp = timeline[0][0]

    handles = {
        "events": args.events_dest.open("a", encoding="utf-8", buffering=1),
        "neural": args.neural_dest.open("a", encoding="utf-8", buffering=1),
    }
    try:
        for timestamp, stream, line in timeline:
            target = (timestamp - base_timestamp).total_seconds() / speed
            delay = target - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
            handles[stream].write(line)
            handles[stream].flush()
    finally:
        for handle in handles.values():
            handle.close()

    elapsed = time.monotonic() - started
    print(f"replayed {len(timeline)} records in {elapsed:.1f}s; previous logs archived at {archive}")


if __name__ == "__main__":
    main()
