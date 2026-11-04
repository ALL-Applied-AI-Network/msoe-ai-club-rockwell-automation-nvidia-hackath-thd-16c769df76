#!/usr/bin/env bash
set -euo pipefail

workspace="${CARRYOVER_WORKSPACE:-$HOME/workspace}"
workspace=$(realpath -m "$workspace")
code_dir="$(cd "$(dirname "$0")/../code" && pwd)"
model=$(realpath -m "${CARRYOVER_MODEL:-$workspace/live/models/carryover/model.ts}")
tank_colors=$(realpath -m "${CARRYOVER_TANK_COLORS:-$workspace/live/models/carryover/tank_colors.npy}")
image="${CARRYOVER_IMAGE:-nvcr.io/nvidia/pytorch:25.10-py3}"
container_name="${CARRYOVER_CONTAINER:-carryover-live-monitor}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CAMERA_JPEG_FOLDER QUEUE_CSV [LIVE_OUTPUT_FOLDER]" >&2
  exit 2
fi

camera_dir=$(realpath "$1")
queue_csv=$(realpath "$2")
live_dir=$(realpath -m "${3:-$workspace/live/carryover-monitor}")

[[ -d "$workspace" ]] || { echo "Workspace not found: $workspace" >&2; exit 1; }
[[ -d "$camera_dir" ]] || { echo "Camera folder not found: $camera_dir" >&2; exit 1; }
[[ -f "$queue_csv" ]] || { echo "Queue CSV not found: $queue_csv" >&2; exit 1; }
[[ -f "$code_dir/live_monitor.py" ]] || { echo "Monitor code not found: $code_dir/live_monitor.py" >&2; exit 1; }
[[ -f "$model" ]] || { echo "Model not found: $model" >&2; exit 1; }
[[ -f "$tank_colors" ]] || { echo "Tank calibration not found: $tank_colors" >&2; exit 1; }
[[ $(wc -l < "$queue_csv") -gt 1 ]] || { echo "Queue CSV has no recipe rows" >&2; exit 1; }

for path in "$code_dir" "$camera_dir" "$queue_csv" "$live_dir" "$model" "$tank_colors"; do
  if [[ "$path" != "$workspace" && "$path" != "$workspace/"* ]]; then
    echo "Path must be inside the mounted workspace $workspace: $path" >&2
    exit 1
  fi
done

command -v docker >/dev/null 2>&1 || { echo "Docker is not installed" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is unavailable to this account" >&2; exit 1; }

if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists. Stop or remove that specific container before starting another live monitor." >&2
  exit 1
fi

mkdir -p "$live_dir/archive"
stamp=$(date +%Y%m%d-%H%M%S)
for name in events.jsonl neural_predictions.jsonl; do
  if [[ -s "$live_dir/$name" ]]; then
    cp -p "$live_dir/$name" "$live_dir/archive/${stamp}-$name"
  fi
done

docker run -d --rm \
  --name "$container_name" \
  --gpus all \
  --pull missing \
  -v "$workspace:$workspace" \
  -w "$code_dir" \
  "$image" \
  python3 -u live_monitor.py \
    --source watch \
    --dir "$camera_dir" \
    --watch-poll-ms 20 \
    --queue "$queue_csv" \
    --out-dir "$live_dir" \
    --model "$model" \
    --model-threshold 0.0071 \
    --model-window 5 \
    --model-min-frames 3 \
    --tank-colors "$tank_colors"

echo "Live monitor started in container: $container_name"
echo "Camera: $camera_dir"
echo "Queue:  $queue_csv"
echo "Logs:   $live_dir"
echo "Follow its output with: docker logs -f $container_name"
