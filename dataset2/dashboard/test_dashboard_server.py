import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_server import (
    camera_preview_is_analysis_source,
    camera_preview_ready,
    dashboard_destination,
    live_control_status,
)


class DashboardDestinationTests(unittest.TestCase):
    def test_camera_preview_defaults_to_live_dashboard(self):
        server = SimpleNamespace(camera_preview=object())
        self.assertEqual(dashboard_destination(server), "/live")

    def test_no_camera_preview_defaults_to_saved_demo(self):
        server = SimpleNamespace(camera_preview=None)
        self.assertEqual(dashboard_destination(server), "/demo")

    def test_only_allowlisted_requested_destination_is_preserved(self):
        server = SimpleNamespace(camera_preview=object())
        self.assertEqual(dashboard_destination(server, "/demo"), "/demo")
        self.assertEqual(dashboard_destination(server, "/live"), "/live")
        self.assertEqual(
            dashboard_destination(server, "https://example.invalid"), "/live",
        )


class LiveControlStatusTests(unittest.TestCase):
    def test_unconfigured_preview_does_not_add_a_failed_readiness_check(self):
        controller = SimpleNamespace(status=lambda: {
            "configured": True, "configuration": {"camera_dir": True},
        })
        server = SimpleNamespace(live_controller=controller, camera_preview=None)

        status = live_control_status(server)

        self.assertNotIn("camera_preview", status["configuration"])
        self.assertFalse(status["camera_preview"]["configured"])

    def test_camera_check_requires_running_fresh_preview(self):
        controller = SimpleNamespace(status=lambda: {
            "configured": False, "configuration": {},
        })
        stale_preview = SimpleNamespace(status=lambda: {
            "configured": True, "running": True, "frame_available": True,
            "frame_age_seconds": 4,
        })
        server = SimpleNamespace(
            live_controller=controller, camera_preview=stale_preview,
        )
        status = live_control_status(server)
        self.assertFalse(status["configuration"]["camera_preview"])

        server.camera_preview = SimpleNamespace(status=lambda: {
            "configured": True, "running": True, "frame_available": True,
            "frame_age_seconds": 0.1,
        })
        status = live_control_status(server)
        self.assertTrue(status["configuration"]["camera_preview"])

    def test_required_preview_readiness_controls_monitor_configuration(self):
        frame = Path("/workspace/live/camera/latest_frame.jpg")
        controller = SimpleNamespace(
            preview_frame=frame,
            camera_dir=frame.parent,
            status=lambda: {"configured": True, "configuration": {}},
        )
        preview = SimpleNamespace(status=lambda: {
            "configured": True, "running": True, "frame_available": True,
            "frame_age_seconds": 4,
        })
        server = SimpleNamespace(
            live_controller=controller, camera_preview=preview,
        )

        self.assertTrue(camera_preview_is_analysis_source(server))
        self.assertFalse(camera_preview_ready(preview.status()))
        self.assertFalse(live_control_status(server)["configured"])

        server.live_controller.camera_dir = Path("/workspace/other-camera")
        self.assertFalse(camera_preview_is_analysis_source(server))
        self.assertTrue(live_control_status(server)["configured"])


if __name__ == "__main__":
    unittest.main()
