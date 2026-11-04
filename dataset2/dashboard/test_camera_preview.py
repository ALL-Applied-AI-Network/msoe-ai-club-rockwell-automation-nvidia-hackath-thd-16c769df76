import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera_preview import (
    CameraPreview,
    CameraPreviewError,
    MODEL_ROI,
    RULE_ROI,
    build_ffmpeg_command,
    iter_jpeg_frames,
)


class ChunkedStream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read(self, _size):
        return next(self.chunks, b"")


class CameraPreviewTests(unittest.TestCase):
    def test_command_is_fixed_argv_for_local_v4l2_device(self):
        command = build_ffmpeg_command(
            "/dev/v4l/by-id/usb-test-video-index0",
            width=640, height=480, input_fps=30, preview_fps=5,
        )
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-i") + 1], "/dev/v4l/by-id/usb-test-video-index0")
        self.assertIn("640x480", command)
        self.assertIn("fps=5", command)
        self.assertEqual(command[-1], "pipe:1")

    def test_rejects_network_and_relative_sources(self):
        for source in ("video4", "rtsp://camera/live", "/tmp/video4", "/dev/v4l/by-path/device"):
            with self.subTest(source=source), self.assertRaises(CameraPreviewError):
                build_ffmpeg_command(source)
        with self.assertRaisesRegex(CameraPreviewError, "cannot exceed"):
            build_ffmpeg_command("/dev/video4", input_fps=5, preview_fps=10)

    def test_rejects_nonfinite_or_fractional_capture_settings(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(CameraPreviewError):
                build_ffmpeg_command("/dev/video4", preview_fps=value)
        with self.assertRaises(CameraPreviewError):
            build_ffmpeg_command("/dev/video4", width=640.5)
        with self.assertRaises(CameraPreviewError):
            build_ffmpeg_command("/dev/video4", height=True)

    def test_parser_handles_noise_and_markers_split_across_chunks(self):
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        stream = ChunkedStream([
            b"noise\xff", b"\xd8fir", b"st\xff", b"\xd9junk\xff\xd8sec", b"ond\xff\xd9",
        ])
        self.assertEqual(list(iter_jpeg_frames(stream, chunk_size=4)), [first, second])

    def test_parser_discards_oversized_incomplete_frame_and_resynchronizes(self):
        valid = b"\xff\xd8ok\xff\xd9"
        stream = io.BytesIO(b"\xff\xd8" + b"x" * 30 + valid)
        self.assertEqual(list(iter_jpeg_frames(stream, max_frame_bytes=16)), [valid])

    def test_publish_atomically_writes_frame_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "latest_frame.jpg"
            preview = CameraPreview("/dev/video4", output, source_name="Vial fill")
            frame = b"\xff\xd8test-frame\xff\xd9"
            preview._publish(frame)

            self.assertEqual(output.read_bytes(), frame)
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_name"], "Vial fill")
            self.assertEqual(metadata["width"], 640)
            self.assertEqual(metadata["height"], 480)
            self.assertEqual(tuple(metadata["rule_roi"]), RULE_ROI)
            self.assertEqual(tuple(metadata["model_roi"]), MODEL_ROI)
            self.assertTrue(preview.status()["frame_available"])
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_output_must_be_jpeg(self):
        with self.assertRaisesRegex(CameraPreviewError, "JPEG"):
            CameraPreview("/dev/video4", "/tmp/preview.png")


if __name__ == "__main__":
    unittest.main()
