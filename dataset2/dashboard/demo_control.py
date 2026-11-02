"""Allowlisted Dataset 5 replay control for the saved-run dashboard."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from live_control import LiveControlError


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class DemoRunController:
    """Runs only the frozen Dataset 5 replay profile in Docker."""

    def __init__(self, *, demo_root, workspace, code_dir, data_root, queue_path,
                 model_path, tank_colors_path, image,
                 container_name="carryover-dataset5-on-demand"):
        self.demo_root = Path(demo_root).resolve()
        self.workspace = Path(workspace).resolve()
        self.code_dir = Path(code_dir).resolve()
        self.data_root = Path(data_root).resolve() if data_root else None
        self.queue_path = Path(queue_path).resolve() if queue_path else None
        self.model_path = Path(model_path).resolve() if model_path else None
        self.tank_colors_path = Path(tank_colors_path).resolve() if tank_colors_path else None
        self.image = image
        self.container_name = container_name
        self.state_path = self.demo_root / "controller_state.json"
        self.audit_path = self.demo_root / "control_audit.jsonl"
        self.lock = threading.RLock()
        self.demo_root.mkdir(parents=True, exist_ok=True)

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
        return result.stdout.strip() if result.returncode == 0 else "stopped"

    def _read_state(self):
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, state):
        _atomic_json(self.state_path, state)

    def _audit(self, action, **details):
        record = {"time": _utc_now(), "action": action, **details, "dataset": "Dataset 5"}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _within_workspace(self, path):
        return bool(path and (path == self.workspace or self.workspace in path.parents))

    def configured(self):
        checks = {
            "dataset_5": bool(
                self.data_root and (self.data_root / "image_index.csv").is_file()
                and self._within_workspace(self.data_root)
            ),
            "queue": bool(self.queue_path and self.queue_path.is_file() and self._within_workspace(self.queue_path)),
            "model": bool(self.model_path and self.model_path.is_file() and self._within_workspace(self.model_path)),
            "tank_colors": bool(
                self.tank_colors_path and self.tank_colors_path.is_file()
                and self._within_workspace(self.tank_colors_path)
            ),
            "code": bool(
                (self.code_dir / "live_monitor.py").is_file()
                and (self.code_dir / "neural_shadow.py").is_file()
                and self._within_workspace(self.code_dir)
            ),
            "workspace": self.workspace.is_dir(),
        }
        return checks, all(checks.values())

    def paths(self):
        state = self._read_state()
        if not state.get("run_dir") or not state.get("queue_path"):
            return None
        run_dir = Path(state["run_dir"])
        return {
            "queue": Path(state["queue_path"]),
            "events": run_dir / "events.jsonl",
            "neural": run_dir / "neural_predictions.jsonl",
            "frame": run_dir / "latest_frame.jpg",
        }

    def status(self):
        with self.lock:
            state = self._read_state()
            container_status = self._container_status()
            checks, ready = self.configured()
            state.update({
                "container_status": container_status,
                "active": container_status in {"running", "paused"},
                "paused": container_status == "paused",
                "configured": ready,
                "configuration": checks,
                "profile": "dataset5_real_time",
                "speed": 1,
                "estimated_minutes": 28,
                "advisory_only": True,
            })
            return state

    def start(self, reason="start"):
        with self.lock:
            checks, ready = self.configured()
            if not ready:
                missing = ", ".join(name for name, present in checks.items() if not present)
                raise LiveControlError(f"Dataset 5 demo is not configured: {missing}")
            if self._container_status() in {"running", "paused"}:
                raise LiveControlError("the Dataset 5 demo is already active")
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
            run_dir = self.demo_root / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            run_queue = run_dir / "queue.csv"
            shutil.copy2(self.queue_path, run_queue)
            command = [
                "run", "-d", "--rm", "--name", self.container_name, "--gpus", "all",
                "--pull", "missing", "-v", f"{self.workspace}:{self.workspace}",
                "-w", str(self.code_dir), self.image,
                "python3", "-u", "live_monitor.py", "--source", "replay", "--speed", "1",
                "--data-root", str(self.data_root), "--queue", str(run_queue),
                "--out-dir", str(run_dir), "--model", str(self.model_path),
                "--model-threshold", "0.0071", "--model-window", "5",
                "--model-min-frames", "3", "--tank-colors", str(self.tank_colors_path),
            ]
            result = self._docker(*command, timeout=120)
            state = {
                "run_id": run_id, "run_dir": str(run_dir), "queue_path": str(run_queue),
                "started_at": _utc_now(), "container_id": result.stdout.strip(),
                "reason": reason,
            }
            self._write_state(state)
            self._audit("demo_started", run_id=run_id, reason=reason)
            return self.status()

    def stop(self, reason="operator_stop"):
        with self.lock:
            status = self._container_status()
            if status in {"running", "paused", "created", "restarting"}:
                self._docker("stop", "--time", "10", self.container_name, timeout=30)
            state = self._read_state()
            state.update({"stopped_at": _utc_now(), "stop_reason": reason})
            self._write_state(state)
            self._audit("demo_stopped", run_id=state.get("run_id"), reason=reason)
            return self.status()

    def pause(self):
        with self.lock:
            if self._container_status() != "running":
                raise LiveControlError("the Dataset 5 demo is not running")
            self._docker("pause", self.container_name)
            self._audit("demo_paused", run_id=self._read_state().get("run_id"))
            return self.status()

    def resume(self):
        with self.lock:
            if self._container_status() != "paused":
                raise LiveControlError("the Dataset 5 demo is not paused")
            self._docker("unpause", self.container_name)
            self._audit("demo_resumed", run_id=self._read_state().get("run_id"))
            return self.status()

    def restart(self):
        with self.lock:
            self.stop(reason="operator_restart")
            return self.start(reason="restart")
