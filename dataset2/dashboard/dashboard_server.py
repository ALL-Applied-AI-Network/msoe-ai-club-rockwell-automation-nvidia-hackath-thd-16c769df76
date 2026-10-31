#!/usr/bin/env python3
"""Operator dashboard for the advisory carryover shadow monitor.

The server reads queue.csv, events.jsonl, neural_predictions.jsonl, and the
monitor's latest-frame preview. Its allowlisted live controls can manage only
the advisory monitor container and its recipe queue. It never writes to the
PLC or controls physical equipment. Bind to localhost and use an SSH tunnel.
"""

import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from live_control import LiveControlError, LiveRunController, validate_queue


HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
COOKIE_NAME = "carryover_dashboard_session"
ACTUAL_KINDS = {"MODEL", "DETECT", "CONFIRM"}


def read_jsonl(path):
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["id"] = line_number
            records.append(record)
    return records


def number(value):
    try:
        parsed = float(value)
        return int(parsed) if parsed.is_integer() else parsed
    except (TypeError, ValueError):
        return value


def read_queue(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for index, row in enumerate(rows):
        normalized = {key: number(value) for key, value in row.items()}
        normalized["queue_index"] = index
        result.append(normalized)
    return sorted(result, key=lambda row: float(row.get("vial", row["queue_index"])))


def file_age(path):
    if not path.is_file():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def read_frame_state(frame_path, frame_url="/api/latest-frame"):
    metadata_path = frame_path.with_suffix(".json")
    if not frame_path.is_file():
        return {"available": False}
    metadata = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    metadata.update({
        "available": True,
        "age_seconds": file_age(frame_path),
        "version": frame_path.stat().st_mtime_ns,
        "url": frame_url,
    })
    return metadata


def build_state(queue_path, events_path, neural_path, frame_path, frame_url="/api/latest-frame"):
    queue = read_queue(queue_path)
    events = read_jsonl(events_path)
    neural = read_jsonl(neural_path)
    completed = set()
    active_vial = None
    predictions = {}
    defects = {}

    for event in events:
        vial = number(event.get("vial"))
        kind = str(event.get("kind", "INFO")).upper()
        message = str(event.get("message", ""))
        if kind == "INFO" and (event.get("state") == "FILLING" or "fill started" in message):
            active_vial = vial
        if kind == "INFO" and (event.get("state") == "COMPLETED" or "left the station" in message):
            completed.add(vial)
            if active_vial == vial:
                active_vial = None
        if kind == "PREDICT":
            predictions[vial] = event
        if kind in ACTUAL_KINDS:
            defects[vial] = event

    current_index = None
    if active_vial is not None:
        current_index = next(
            (index for index, row in enumerate(queue) if number(row.get("vial")) == active_vial), None
        )
    if current_index is None:
        current_index = next(
            (index for index, row in enumerate(queue) if number(row.get("vial")) not in completed), None
        )

    queue_view = []
    for index, row in enumerate(queue):
        vial = number(row.get("vial"))
        item = dict(row)
        if vial in completed:
            item["status"] = "completed"
        elif index == current_index:
            item["status"] = "filling" if active_vial == vial else "next"
        else:
            item["status"] = "upcoming"
        item["prediction"] = predictions.get(vial)
        item["defect"] = defects.get(vial)
        queue_view.append(item)

    current = queue_view[current_index] if current_index is not None else None
    next_recipe = queue_view[current_index + 1] if current_index is not None and current_index + 1 < len(queue_view) else None
    last_neural = neural[-1] if neural else None
    event_age = file_age(events_path)
    neural_age = file_age(neural_path)
    activity_ages = [age for age in (event_age, neural_age) if age is not None]
    activity_age = min(activity_ages) if activity_ages else None
    return {
        "server_time": time.time(),
        "monitor": {
            "has_events": bool(events),
            "activity_age_seconds": activity_age,
            "status": "active" if activity_age is not None and activity_age < 10 else "waiting",
        },
        "current": current,
        "next": next_recipe,
        "queue": queue_view,
        "events": events[-100:],
        "latest_neural": last_neural,
        "frame": read_frame_state(frame_path, frame_url),
        "counts": {
            "total": len(queue_view),
            "completed": len(completed),
            "predictions": len(predictions),
            "defects": len(defects),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CarryoverDashboard/2.0"

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def authenticated(self):
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and secrets.compare_digest(header[7:], self.server.token):
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        value = jar.get(COOKIE_NAME)
        return value is not None and hmac.compare_digest(value.value, self.server.session_key)

    def send_bytes(self, status, body, content_type, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Browsers routinely cancel an in-flight poll while navigating or closing.
            pass

    def send_json(self, status, payload):
        self.send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def serve_file(self, path, content_type):
        if not path.is_file():
            self.send_error(404)
            return
        self.send_bytes(200, path.read_bytes(), content_type)

    def read_json_body(self, limit=1_000_000):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise LiveControlError("invalid request size") from None
        if size < 0 or size > limit:
            raise LiveControlError("request is too large")
        try:
            return json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise LiveControlError("request body must be valid JSON") from None

    def csrf_authorized(self):
        supplied = self.headers.get("X-CSRF-Token", "")
        return bool(supplied) and secrets.compare_digest(supplied, self.server.csrf_token)

    def require_live_write_access(self):
        if not self.authenticated():
            self.send_json(401, {"error": "authentication required"})
            return False
        if not self.csrf_authorized():
            self.send_json(403, {"error": "invalid control token"})
            return False
        return True

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            if not self.authenticated():
                self.send_json(401, {"error": "authentication required"})
                return
            self.send_json(
                200,
                build_state(
                    self.server.queue_path,
                    self.server.events_path,
                    self.server.neural_path,
                    self.server.frame_path,
                ),
            )
            return
        if path == "/api/live/config":
            if not self.authenticated():
                self.send_json(401, {"error": "authentication required"})
                return
            self.send_json(200, {
                "csrf_token": self.server.csrf_token,
                "control": self.server.live_controller.status(),
                "queue": self.server.live_controller.load_queue(),
                "profiles": [{
                    "id": "live_camera", "name": "CSI live camera",
                    "description": "Watch the configured camera folder and run the advisory model.",
                }],
                "safety": "Starts monitoring only. Does not start or control the PLC or machine.",
            })
            return
        if path == "/api/live/state":
            if not self.authenticated():
                self.send_json(401, {"error": "authentication required"})
                return
            paths = self.server.live_controller.paths()
            state = build_state(
                paths["queue"], paths["events"], paths["neural"], paths["frame"],
                "/api/live/latest-frame",
            )
            state["control"] = self.server.live_controller.status()
            self.send_json(200, state)
            return
        if path == "/api/latest-frame":
            if not self.authenticated():
                self.send_json(401, {"error": "authentication required"})
                return
            self.serve_file(self.server.frame_path, "image/jpeg")
            return
        if path == "/api/live/latest-frame":
            if not self.authenticated():
                self.send_json(401, {"error": "authentication required"})
                return
            self.serve_file(self.server.live_controller.paths()["frame"], "image/jpeg")
            return
        if path == "/healthz":
            self.send_json(200, {"status": "ok"})
            return
        if path == "/app.js":
            self.serve_file(STATIC / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/live.js":
            self.serve_file(STATIC / "live.js", "text/javascript; charset=utf-8")
            return
        if path == "/style.css":
            self.serve_file(STATIC / "style.css", "text/css; charset=utf-8")
            return
        if path in {"/", "/demo"} and self.authenticated():
            self.serve_file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        if path == "/live" and self.authenticated():
            self.serve_file(STATIC / "live.html", "text/html; charset=utf-8")
            return
        if path in {"/", "/login", "/demo", "/live"}:
            self.serve_file(STATIC / "login.html", "text/html; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/login":
            try:
                size = min(int(self.headers.get("Content-Length", "0")), 4096)
            except ValueError:
                size = 0
            form = parse_qs(self.rfile.read(size).decode("utf-8", errors="replace"))
            supplied = form.get("token", [""])[0]
            if not secrets.compare_digest(supplied, self.server.token):
                self.send_bytes(401, b"Invalid dashboard token", "text/plain; charset=utf-8")
                return
            cookie = (
                f"{COOKIE_NAME}={self.server.session_key}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200"
            )
            self.send_bytes(303, b"", "text/plain", {"Location": "/demo", "Set-Cookie": cookie})
            return
        if not path.startswith("/api/live/"):
            self.send_error(404)
            return
        if not self.require_live_write_access():
            return
        try:
            payload = self.read_json_body()
            controller = self.server.live_controller
            if path == "/api/live/queue/validate":
                result = validate_queue(payload.get("rows"), controller.tank_colors)
                self.send_json(200, {"valid": True, "rows": result})
            elif path == "/api/live/queue":
                result = controller.save_queue(payload.get("rows"))
                self.send_json(200, {"saved": True, "rows": result, "control": controller.status()})
            elif path == "/api/live/start":
                if payload.get("profile") != "live_camera":
                    raise LiveControlError("only the predefined live_camera profile is allowed")
                if payload.get("confirmed") is not True:
                    raise LiveControlError("operator confirmation is required")
                self.send_json(200, controller.start(start_vial=payload.get("start_vial")))
            elif path == "/api/live/stop":
                if payload.get("confirmed") is not True:
                    raise LiveControlError("operator confirmation is required")
                self.send_json(200, controller.stop())
            elif path == "/api/live/pause":
                self.send_json(200, controller.pause())
            elif path == "/api/live/resume":
                self.send_json(200, controller.resume())
            elif path == "/api/live/realign":
                if payload.get("confirmed") is not True:
                    raise LiveControlError("operator confirmation is required")
                self.send_json(200, controller.realign(payload.get("start_vial")))
            else:
                self.send_error(404)
        except LiveControlError as error:
            self.send_json(409, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description="Token-protected carryover operator dashboard")
    parser.add_argument("--queue", required=True, help="recipe queue CSV")
    parser.add_argument("--events", required=True, help="events.jsonl written by live_monitor.py")
    parser.add_argument("--neural-log", help="optional neural_predictions.jsonl")
    parser.add_argument(
        "--latest-frame",
        help="latest_frame.jpg written by live_monitor.py; defaults beside --events",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address; localhost is safest")
    parser.add_argument("--port", type=int, default=20400)
    parser.add_argument("--token", default=os.environ.get("CARRYOVER_DASH_TOKEN"))
    parser.add_argument("--live-root", help="root for queue drafts, run outputs, and audit logs")
    parser.add_argument("--live-workspace", default=os.environ.get("CARRYOVER_WORKSPACE", str(Path.home() / "workspace")))
    parser.add_argument("--live-code-dir", help="folder containing live_monitor.py")
    parser.add_argument("--live-camera-dir", help="allowlisted CSI camera JPEG folder")
    parser.add_argument("--live-model", help="allowlisted TorchScript model.ts")
    parser.add_argument("--live-tank-colors", help="allowlisted 3x3 tank_colors.npy")
    parser.add_argument("--live-image", default=os.environ.get("CARRYOVER_IMAGE", "nvcr.io/nvidia/pytorch:25.10-py3"))
    parser.add_argument("--live-container-name", default="carryover-live-monitor")
    args = parser.parse_args()

    token = args.token or secrets.token_urlsafe(18)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.token = token
    server.session_key = hashlib.sha256((token + secrets.token_hex(16)).encode()).hexdigest()
    server.csrf_token = secrets.token_urlsafe(24)
    server.queue_path = Path(args.queue).resolve()
    server.events_path = Path(args.events).resolve()
    server.neural_path = Path(args.neural_log).resolve() if args.neural_log else Path("__missing__")
    server.frame_path = (
        Path(args.latest_frame).resolve()
        if args.latest_frame else server.events_path.with_name("latest_frame.jpg")
    )
    live_root = Path(args.live_root).resolve() if args.live_root else server.events_path.parent / "live_control"
    server.live_controller = LiveRunController(
        live_root=live_root,
        workspace=args.live_workspace,
        code_dir=args.live_code_dir or HERE.parent / "code",
        camera_dir=args.live_camera_dir,
        model_path=args.live_model,
        tank_colors_path=args.live_tank_colors,
        image=args.live_image,
        container_name=args.live_container_name,
    )
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Login token: {token}")
    print("Advisory monitor only. Live controls do not connect to or control the PLC.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

