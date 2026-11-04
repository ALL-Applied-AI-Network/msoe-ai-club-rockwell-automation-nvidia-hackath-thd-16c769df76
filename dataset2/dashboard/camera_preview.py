"""Supervised raw-frame capture from a local V4L2 camera.

The adapter is intentionally separate from the advisory model.  It opens a
local video device through FFmpeg, receives a bounded MJPEG stream, and
atomically publishes the newest frame for the token-protected dashboard.  It
does not make carryover decisions, change camera controls, or connect to the
PLC or other equipment. The monitor may separately watch the output folder.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


RULE_ROI = (230, 330, 380, 440)
MODEL_ROI = (200, 190, 420, 480)
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
MAX_FRAME_BYTES = 8 * 1024 * 1024
_VIDEO_DEVICE = re.compile(r"^/dev/video\d+$")


class CameraPreviewError(RuntimeError):
    """A safe, operator-displayable camera preview error."""


def _validate_device_path(device):
    """Accept only local V4L2 device nodes or stable udev by-id links."""
    path = Path(device).expanduser()
    value = str(path)
    if not path.is_absolute():
        raise CameraPreviewError("camera device must be an absolute path")
    if _VIDEO_DEVICE.fullmatch(value):
        return path
    by_id = Path("/dev/v4l/by-id")
    if path.parent == by_id and path.name not in {"", ".", ".."}:
        return path
    raise CameraPreviewError("camera device must be /dev/videoN or /dev/v4l/by-id/<device>")


def _positive_number(value, name, *, integer=False):
    if isinstance(value, bool):
        raise CameraPreviewError(f"{name} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise CameraPreviewError(f"{name} must be a positive number") from None
    if not math.isfinite(parsed) or parsed <= 0 or (integer and not parsed.is_integer()):
        raise CameraPreviewError(f"{name} must be a positive number")
    return int(parsed) if integer else parsed


def build_ffmpeg_command(device, *, width=640, height=480, input_fps=30,
                         preview_fps=5, ffmpeg="ffmpeg"):
    """Build the fixed local-device capture command as an argv list."""
    device = _validate_device_path(device)
    width = _positive_number(width, "width", integer=True)
    height = _positive_number(height, "height", integer=True)
    input_fps = _positive_number(input_fps, "input_fps")
    preview_fps = _positive_number(preview_fps, "preview_fps")
    if preview_fps > input_fps:
        raise CameraPreviewError("preview_fps cannot exceed input_fps")
    if not str(ffmpeg).strip():
        raise CameraPreviewError("ffmpeg executable is required")
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{width}x{height}", "-framerate", f"{input_fps:g}",
        "-i", str(device), "-an", "-vf", f"fps={preview_fps:g}",
        "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]


def iter_jpeg_frames(stream, *, chunk_size=64 * 1024,
                     max_frame_bytes=MAX_FRAME_BYTES):
    """Yield complete JPEGs from a chunked MJPEG byte stream.

    Noise before a start marker and oversized/incomplete frames are discarded
    without publishing partial camera data.
    """
    chunk_size = _positive_number(chunk_size, "chunk_size", integer=True)
    max_frame_bytes = _positive_number(
        max_frame_bytes, "max_frame_bytes", integer=True,
    )
    buffer = bytearray()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(JPEG_START)
            if start < 0:
                # Keep a trailing 0xff in case the next chunk starts with 0xd8.
                buffer[:] = buffer[-1:] if buffer.endswith(b"\xff") else b""
                break
            if start:
                del buffer[:start]
            end = buffer.find(JPEG_END, len(JPEG_START))
            if end >= 0:
                end += len(JPEG_END)
                if end > max_frame_bytes:
                    next_start = buffer.find(JPEG_START, len(JPEG_START), end)
                    if next_start >= 0:
                        del buffer[:next_start]
                        continue
                frame = bytes(buffer[:end])
                del buffer[:end]
                if len(frame) <= max_frame_bytes:
                    yield frame
                continue
            if len(buffer) > max_frame_bytes:
                # Resynchronize to a later start marker when possible.
                next_start = buffer.find(JPEG_START, len(JPEG_START))
                if next_start >= 0:
                    del buffer[:next_start]
                    continue
                buffer[:] = buffer[-1:] if buffer.endswith(b"\xff") else b""
            break


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path, payload):
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    _atomic_bytes(path, encoded)


class CameraPreview:
    """Publish a bounded, low-rate JPEG stream from one fixed V4L2 device."""

    def __init__(self, device, output_path, *, width=640, height=480,
                 input_fps=30, preview_fps=5,
                 source_name="CSI vial-fill camera", ffmpeg="ffmpeg"):
        self.device = _validate_device_path(device)
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise CameraPreviewError("camera preview output must be a JPEG path")
        self.width = _positive_number(width, "width", integer=True)
        self.height = _positive_number(height, "height", integer=True)
        self.input_fps = _positive_number(input_fps, "input_fps")
        self.preview_fps = _positive_number(preview_fps, "preview_fps")
        if self.preview_fps > self.input_fps:
            raise CameraPreviewError("preview_fps cannot exceed input_fps")
        self.source_name = str(source_name).strip()
        if not self.source_name:
            raise CameraPreviewError("camera source name is required")
        self.ffmpeg = str(ffmpeg).strip()
        if not self.ffmpeg:
            raise CameraPreviewError("ffmpeg executable is required")

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._last_frame_at = None
        self._last_error = None
        self._restart_count = 0

    @property
    def metadata_path(self):
        return self.output_path.with_suffix(".json")

    def command(self):
        return build_ffmpeg_command(
            self.device, width=self.width, height=self.height,
            input_fps=self.input_fps, preview_fps=self.preview_fps,
            ffmpeg=self.ffmpeg,
        )

    def _validate_runtime(self):
        if os.path.sep not in self.ffmpeg:
            executable = shutil.which(self.ffmpeg)
        else:
            executable = self.ffmpeg if os.access(self.ffmpeg, os.X_OK) else None
        if not executable:
            raise CameraPreviewError("ffmpeg is not installed or executable")
        try:
            mode = self.device.stat().st_mode
        except OSError as error:
            raise CameraPreviewError(f"camera device is unavailable: {error.strerror}") from None
        if not stat.S_ISCHR(mode):
            raise CameraPreviewError("camera device is not a character device")

    def _publish(self, frame):
        now = datetime.now(timezone.utc)
        _atomic_bytes(self.output_path, frame)
        _atomic_json(self.metadata_path, {
            "capture_time": now.isoformat(),
            "published_at": now.isoformat(),
            "source_name": self.source_name,
            "width": self.width,
            "height": self.height,
            "rule_roi": RULE_ROI,
            "model_roi": MODEL_ROI,
        })
        with self._lock:
            self._last_frame_at = now.timestamp()
            self._last_error = None

    def _open_process(self):
        return subprocess.Popen(
            self.command(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0, close_fds=True,
        )

    def _supervise(self):
        attempt = 0
        while not self._stop_event.is_set():
            process = None
            try:
                process = self._open_process()
                with self._lock:
                    self._process = process
                    if attempt:
                        self._restart_count += 1
                if process.stdout is None:
                    raise CameraPreviewError("ffmpeg preview pipe is unavailable")
                for frame in iter_jpeg_frames(process.stdout):
                    if self._stop_event.is_set():
                        break
                    self._publish(frame)
                if not self._stop_event.is_set():
                    return_code = process.poll()
                    if return_code is None:
                        return_code = process.wait(timeout=1)
                    with self._lock:
                        self._last_error = f"camera capture exited with status {return_code}"
            except (OSError, subprocess.SubprocessError, CameraPreviewError) as error:
                if not self._stop_event.is_set():
                    with self._lock:
                        self._last_error = str(error)[:300]
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    if process.stdout is not None:
                        process.stdout.close()
                with self._lock:
                    if self._process is process:
                        self._process = None
            attempt += 1
            self._stop_event.wait(1.0)

    def start(self):
        """Start capture in a supervised background thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._validate_runtime()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._stop_event.clear()
            self._last_error = None
            self._thread = threading.Thread(
                target=self._supervise, name="vial-fill-camera-preview", daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self):
        """Stop capture without deleting the last successfully published frame."""
        with self._lock:
            self._stop_event.set()
            process = self._process
            thread = self._thread
        if process is not None and process.poll() is None:
            process.terminate()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        with self._lock:
            if thread is None or not thread.is_alive():
                self._thread = None
                self._process = None
        return self.status()

    def status(self):
        """Return non-sensitive state suitable for the authenticated API."""
        with self._lock:
            process_running = self._process is not None and self._process.poll() is None
            thread_running = self._thread is not None and self._thread.is_alive()
            frame_available = self.output_path.is_file()
            age = None
            if frame_available:
                try:
                    age = max(0.0, time.time() - self.output_path.stat().st_mtime)
                except OSError:
                    frame_available = False
            return {
                "configured": True,
                "running": bool(process_running and thread_running),
                "supervising": bool(thread_running),
                "frame_available": frame_available,
                "frame_age_seconds": age,
                "source_name": self.source_name,
                "width": self.width,
                "height": self.height,
                "preview_fps": self.preview_fps,
                "restart_count": self._restart_count,
                "last_error": self._last_error,
            }
