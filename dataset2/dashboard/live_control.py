"""Allowlisted live-monitor run control for the operator dashboard.

This module starts and stops only the advisory carryover monitor container. It
does not connect to, write to, or control the PLC or physical equipment.
"""

from __future__ import annotations

import ast
import colorsys
import csv
import json
import math
import os
import secrets
import struct
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path


QUEUE_FIELDS = [
    "vial", "recipe_id", "t100_ml", "t200_ml", "t400_ml", "total_ml",
    "expected_r", "expected_g", "expected_b", "expected_hue", "expected_hex",
]
MIN_ACTIVE_TANK_ML = 10.0
MAX_T200_ML = 50.0
MAX_TOTAL_ML = 80.0


class LiveControlError(RuntimeError):
    """A safe, operator-displayable live-control error."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_npy_3x3(path):
    """Read the small numeric 3x3 tank calibration without requiring NumPy."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise LiveControlError("tank calibration is not a NumPy array")
        major, _minor = handle.read(2)
        if major == 1:
            header_length = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise LiveControlError("unsupported tank calibration format")
        header = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
        if tuple(header.get("shape", ())) != (3, 3) or header.get("fortran_order"):
            raise LiveControlError("tank calibration must be a row-major 3x3 array")
        formats = {"<f4": ("<9f", 36), "<f8": ("<9d", 72), "|f4": ("<9f", 36), "|f8": ("<9d", 72)}
        if header.get("descr") not in formats:
            raise LiveControlError("tank calibration must contain float32 or float64 values")
        unpack_format, byte_count = formats[header["descr"]]
        values = struct.unpack(unpack_format, handle.read(byte_count))
    return [list(values[index:index + 3]) for index in range(0, 9, 3)]


def _as_float(value, field, row_number, minimum=0.0):
    if isinstance(value, bool):
        raise LiveControlError(f"row {row_number}: {field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiveControlError(f"row {row_number}: {field} must be a number") from None
    if not math.isfinite(number):
        raise LiveControlError(f"row {row_number}: {field} must be finite")
    if number < minimum:
        raise LiveControlError(f"row {row_number}: {field} must be at least {minimum:g}")
    return number


def _clean_number(value):
    return int(value) if float(value).is_integer() else round(float(value), 4)


def _whole_vial_number(value, field="start_vial"):
    if isinstance(value, bool):
        raise LiveControlError(f"{field} must be a whole vial number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiveControlError(f"{field} must be a whole vial number") from None
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise LiveControlError(f"{field} must be a whole vial number")
    return int(number)


def _expected_color(row, tank_colors):
    supplied = [row.get("expected_r"), row.get("expected_g"), row.get("expected_b")]
    if all(value not in (None, "") for value in supplied):
        rgb = [max(0.0, min(255.0, float(value))) for value in supplied]
    elif tank_colors is not None:
        total = max(float(row["total_ml"]), 1.0)
        # Calibration rows are [T-200 orange, T-400 blue, T-100 clear].
        weights = [float(row["t200_ml"]) / total, float(row["t400_ml"]) / total,
                   float(row["t100_ml"]) / total]
        rgb = [sum(weights[index] * tank_colors[index][channel] for index in range(3))
               for channel in range(3)]
        rgb = [max(0.0, min(255.0, value)) for value in rgb]
    else:
        return None
    hue = colorsys.rgb_to_hsv(*(value / 255.0 for value in rgb))[0] * 255.0
    hex_color = "#" + "".join(f"{int(round(value)):02x}" for value in rgb)
    return rgb, hue, hex_color


def validate_queue(rows, tank_colors=None):
    if not isinstance(rows, list) or not rows:
        raise LiveControlError("add at least one recipe row")
    if len(rows) > 500:
        raise LiveControlError("recipe queue is limited to 500 vials")
    normalized = []
    seen_vials = set()
    previous_vial = None
    previous_recipe = None
    for row_number, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise LiveControlError(f"row {row_number}: invalid recipe row")
        vial_number = _as_float(raw.get("vial"), "vial", row_number, 1)
        recipe_number = _as_float(raw.get("recipe_id"), "recipe_id", row_number, 0)
        if not vial_number.is_integer() or not recipe_number.is_integer():
            raise LiveControlError(f"row {row_number}: vial and recipe_id must be whole numbers")
        vial = int(vial_number)
        recipe_id = int(recipe_number)
        if vial in seen_vials:
            raise LiveControlError(f"row {row_number}: vial {vial} is duplicated")
        if previous_vial is not None and vial <= previous_vial:
            raise LiveControlError(f"row {row_number}: vial numbers must increase")
        if raw.get("t300_ml") not in (None, ""):
            t300_ml = _as_float(raw["t300_ml"], "t300_ml", row_number, 0)
            if t300_ml != 0:
                raise LiveControlError(
                    f"row {row_number}: T-300 is out of service and must remain at 0 mL"
                )
        volumes = {
            name: _as_float(raw.get(name, 0), name, row_number, 0)
            for name in ("t100_ml", "t200_ml", "t400_ml")
        }
        for name, value in volumes.items():
            if 0 < value < MIN_ACTIVE_TANK_ML:
                raise LiveControlError(
                    f"row {row_number}: {name} must be 0 or at least "
                    f"{MIN_ACTIVE_TANK_ML:g} mL"
                )
        if volumes["t200_ml"] > MAX_T200_ML:
            raise LiveControlError(
                f"row {row_number}: t200_ml must not exceed {MAX_T200_ML:g} mL"
            )
        calculated_total = sum(volumes.values())
        if calculated_total > MAX_TOTAL_ML:
            raise LiveControlError(
                f"row {row_number}: total volume must not exceed {MAX_TOTAL_ML:g} mL"
            )
        supplied_total = raw.get("total_ml")
        reported_total = (
            calculated_total
            if supplied_total in (None, "")
            else _as_float(supplied_total, "total_ml", row_number, 0.001)
        )
        if calculated_total <= 0:
            raise LiveControlError(f"row {row_number}: at least one tank volume must be positive")
        if abs(reported_total - calculated_total) > 0.05:
            raise LiveControlError(
                f"row {row_number}: total_ml {reported_total:g} does not equal "
                f"tank volume sum {calculated_total:g}"
            )
        # The component sum is authoritative. A reported total may differ by a
        # small CSV-rounding amount, but it must not skew color/risk fractions.
        total = calculated_total
        row = {
            "vial": vial,
            "recipe_id": recipe_id,
            **{name: _clean_number(value) for name, value in volumes.items()},
            "total_ml": _clean_number(total),
        }
        for name in ("expected_r", "expected_g", "expected_b"):
            if raw.get(name) not in (None, ""):
                row[name] = _as_float(raw[name], name, row_number, 0)
        expected = _expected_color(row, tank_colors)
        if expected:
            rgb, hue, hex_color = expected
            row.update({
                "expected_r": round(rgb[0], 3), "expected_g": round(rgb[1], 3),
                "expected_b": round(rgb[2], 3), "expected_hue": round(hue, 3),
                "expected_hex": hex_color,
            })
        mix = [volumes["t100_ml"] / total, volumes["t200_ml"] / total,
               volumes["t400_ml"] / total]
        if previous_recipe is None or previous_recipe[0] == recipe_id:
            risk_score = 0.0
        else:
            risk_score = 0.5 * sum(abs(a - b) for a, b in zip(previous_recipe[1], mix))
        row["transition_risk"] = (
            "HIGH" if risk_score >= 0.40 else "MEDIUM" if risk_score >= 0.25 else "LOW"
        ) if previous_recipe is not None and previous_recipe[0] != recipe_id else "—"
        row["transition_score"] = round(risk_score, 4)
        normalized.append(row)
        seen_vials.add(vial)
        previous_vial = vial
        previous_recipe = (recipe_id, mix)
    return normalized


class LiveRunController:
    """Starts only allowlisted carryover-monitor container commands."""

    def __init__(self, *, live_root, workspace, code_dir, camera_dir, model_path,
                 tank_colors_path, image, container_name="carryover-live-monitor",
                 preview_frame=None):
        self.live_root = Path(live_root).resolve()
        self.workspace = Path(workspace).resolve()
        self.code_dir = Path(code_dir).resolve()
        self.camera_dir = Path(camera_dir).resolve() if camera_dir else None
        self.model_path = Path(model_path).resolve() if model_path else None
        self.tank_colors_path = Path(tank_colors_path).resolve() if tank_colors_path else None
        self.image = image
        self.container_name = container_name
        self.preview_frame = Path(preview_frame).resolve() if preview_frame else None
        self.live_root.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.live_root / "queue_drafts" / "current_queue.csv"
        self.state_path = self.live_root / "controller_state.json"
        self.audit_path = self.live_root / "control_audit.jsonl"
        self.lock = threading.RLock()
        self.tank_colors = None
        if self.tank_colors_path:
            try:
                self.tank_colors = _read_npy_3x3(self.tank_colors_path)
            except (OSError, ValueError, SyntaxError, struct.error, LiveControlError):
                # Keep the dashboard available so its readiness check can show
                # the invalid calibration instead of failing at startup.
                pass

    def _docker(self, *arguments, check=True, timeout=30):
        try:
            result = subprocess.run(
                ["docker", *map(str, arguments)], check=False, capture_output=True,
                text=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LiveControlError(f"Docker command unavailable: {error}") from None
        if check and result.returncode:
            message = (result.stderr or result.stdout or "Docker command failed").strip()
            raise LiveControlError(message[:500])
        return result

    def _container_status(self):
        try:
            result = self._docker(
                "inspect", "--format", "{{.State.Status}}", self.container_name,
                check=False, timeout=10,
            )
        except LiveControlError:
            return "unavailable"
        if result.returncode == 0:
            return result.stdout.strip()
        message = (result.stderr or result.stdout or "").lower()
        if "no such object" in message or "no such container" in message:
            return "stopped"
        return "unavailable"

    def _read_state(self):
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, state):
        _atomic_text(self.state_path, json.dumps(state, indent=2) + "\n")

    def _audit(self, action, **details):
        record = {"time": _utc_now(), "action": action, **details, "advisory_only": True}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_queue(self, path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(5)}.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=QUEUE_FIELDS, extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def configured(self, container_status=None):
        def within_workspace(path):
            return bool(path and (path == self.workspace or self.workspace in path.parents))

        tank_colors_ready = bool(
            self.tank_colors_path and self.tank_colors_path.is_file()
            and within_workspace(self.tank_colors_path)
        )
        if tank_colors_ready:
            try:
                self.tank_colors = _read_npy_3x3(self.tank_colors_path)
                tank_colors_ready = self.tank_colors is not None
            except (OSError, ValueError, SyntaxError, struct.error, LiveControlError):
                self.tank_colors = None
                tank_colors_ready = False
        if container_status is None:
            container_status = self._container_status()
        checks = {
            "camera_dir": bool(self.camera_dir and self.camera_dir.is_dir() and within_workspace(self.camera_dir)),
            "model": bool(self.model_path and self.model_path.is_file() and within_workspace(self.model_path)),
            "tank_colors": tank_colors_ready,
            "code": (self.code_dir / "live_monitor.py").is_file() and within_workspace(self.code_dir),
            "workspace": self.workspace.is_dir(),
            "live_root": within_workspace(self.live_root),
            "docker": container_status != "unavailable",
        }
        return checks, all(checks.values())

    def save_queue(self, rows):
        with self.lock:
            normalized = validate_queue(rows, self.tank_colors)
            self._write_queue(self.queue_path, normalized)
            self._audit("queue_saved", rows=len(normalized))
            return normalized

    def load_queue(self):
        if not self.queue_path.is_file():
            return []
        with self.queue_path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def paths(self):
        state = self._read_state()
        run_dir = Path(state["run_dir"]) if state.get("run_dir") else self.live_root / "no-run"
        queue_path = Path(state.get("queue_path", self.queue_path))
        monitor_frame = run_dir / "latest_frame.jpg"
        frame = monitor_frame
        candidates = []
        for candidate in (monitor_frame, self.preview_frame):
            if candidate is None:
                continue
            try:
                if candidate.is_file():
                    candidates.append((candidate.stat().st_mtime_ns, candidate))
            except OSError:
                continue
        if candidates:
            # Prefer the freshest available source so a stale raw preview does
            # not hide frames from an active analysis run.
            frame = max(candidates, key=lambda item: item[0])[1]
        return {
            "queue": queue_path,
            "events": run_dir / "events.jsonl",
            "neural": run_dir / "neural_predictions.jsonl",
            "frame": frame,
        }

    def status(self):
        with self.lock:
            state = self._read_state()
            container_status = self._container_status()
            checks, configured = self.configured(container_status)
            state.update({
                "container_status": container_status,
                "active": container_status in {"running", "paused"},
                "paused": container_status == "paused",
                "queue_saved": self.queue_path.is_file(),
                "queue_rows": len(self.load_queue()),
                "configured": configured,
                "configuration": checks,
                "preview_frame_available": bool(
                    self.preview_frame and self.preview_frame.is_file()
                ),
                "advisory_only": True,
            })
            return state

    def start(self, *, start_vial=None, reason="start"):
        with self.lock:
            checks, ready = self.configured()
            if not ready:
                missing = ", ".join(name for name, present in checks.items() if not present)
                raise LiveControlError(f"live profile is not configured: {missing}")
            if not self.queue_path.is_file():
                raise LiveControlError("save and validate a recipe queue before starting")
            if self._container_status() in {"running", "paused"}:
                raise LiveControlError("a live monitor run is already active")
            # Revalidate the persisted file at the execution boundary. This
            # prevents an older or externally edited draft from bypassing the
            # current physical-station limits.
            queue = validate_queue(self.load_queue(), self.tank_colors)
            vials = [int(row["vial"]) for row in queue]
            if start_vial is None:
                start_vial = vials[0]
            else:
                start_vial = _whole_vial_number(start_vial)
            if start_vial not in vials:
                raise LiveControlError(f"vial {start_vial} is not in the saved queue")
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
            run_dir = self.live_root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            run_queue = run_dir / "queue.csv"
            self._write_queue(run_queue, queue)
            command = [
                "run", "-d", "--rm", "--name", self.container_name, "--gpus", "all",
                "--pull", "missing", "-v", f"{self.workspace}:{self.workspace}",
                "-w", str(self.code_dir), self.image,
                "python3", "-u", "live_monitor.py", "--source", "watch",
                "--dir", str(self.camera_dir), "--watch-poll-ms", "20",
                "--queue", str(run_queue), "--out-dir", str(run_dir),
                "--start-vial", str(start_vial), "--model", str(self.model_path),
                "--model-threshold", "0.0071", "--model-window", "5",
                "--model-min-frames", "3", "--tank-colors", str(self.tank_colors_path),
            ]
            result = self._docker(*command, timeout=120)
            state = {
                "run_id": run_id, "run_dir": str(run_dir), "queue_path": str(run_queue),
                "started_at": _utc_now(), "start_vial": start_vial,
                "container_id": result.stdout.strip(), "reason": reason,
            }
            self._write_state(state)
            self._audit("run_started", run_id=run_id, start_vial=start_vial, reason=reason)
            return self.status()

    def stop(self, reason="operator_stop"):
        with self.lock:
            status = self._container_status()
            if status in {"running", "paused", "created", "restarting"}:
                self._docker("stop", "--time", "10", self.container_name, timeout=30)
            state = self._read_state()
            state.update({"stopped_at": _utc_now(), "stop_reason": reason})
            self._write_state(state)
            self._audit("run_stopped", run_id=state.get("run_id"), reason=reason)
            return self.status()

    def pause(self):
        with self.lock:
            if self._container_status() != "running":
                raise LiveControlError("the live monitor is not running")
            self._docker("pause", self.container_name)
            self._audit("run_paused", run_id=self._read_state().get("run_id"))
            return self.status()

    def resume(self):
        with self.lock:
            if self._container_status() != "paused":
                raise LiveControlError("the live monitor is not paused")
            self._docker("unpause", self.container_name)
            self._audit("run_resumed", run_id=self._read_state().get("run_id"))
            return self.status()

    def realign(self, start_vial):
        with self.lock:
            start_vial = _whole_vial_number(start_vial)
            if not self.queue_path.is_file():
                raise LiveControlError("save and validate a recipe queue before realigning")
            queue = validate_queue(self.load_queue(), self.tank_colors)
            if start_vial not in {int(row["vial"]) for row in queue}:
                raise LiveControlError(f"vial {start_vial} is not in the saved queue")
            previous = self._read_state().get("run_id")
            self.stop(reason=f"realign_to_vial_{start_vial}")
            status = self.start(start_vial=start_vial, reason=f"realign_from_{previous}")
            self._audit("run_realigned", previous_run=previous, start_vial=start_vial)
            return status

