import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from live_control import LiveControlError, LiveRunController, validate_queue


class QueueValidationTests(unittest.TestCase):
    def test_computes_expected_color_and_transition_risk(self):
        rows = validate_queue([
            {"vial": 1, "recipe_id": 10, "t100_ml": 30, "t200_ml": 0, "t400_ml": 0},
            {"vial": 2, "recipe_id": 20, "t100_ml": 0, "t200_ml": 30, "t400_ml": 0},
        ], [[240, 120, 30], [30, 90, 220], [245, 245, 245]])
        self.assertEqual(rows[0]["expected_hex"], "#f5f5f5")
        self.assertEqual(rows[1]["expected_hex"], "#f0781e")
        self.assertEqual(rows[1]["transition_risk"], "HIGH")

    def test_rejects_fractional_ids_and_bad_totals(self):
        with self.assertRaisesRegex(LiveControlError, "whole numbers"):
            validate_queue([{"vial": 1.5, "recipe_id": 1, "t100_ml": 10}])
        with self.assertRaisesRegex(LiveControlError, "does not equal"):
            validate_queue([{
                "vial": 1, "recipe_id": 1, "t100_ml": 10,
                "t200_ml": 10, "t400_ml": 10, "total_ml": 25,
            }])

    def test_rejects_duplicate_or_empty_queue(self):
        with self.assertRaisesRegex(LiveControlError, "at least one"):
            validate_queue([])
        with self.assertRaisesRegex(LiveControlError, "duplicated"):
            validate_queue([
                {"vial": 1, "recipe_id": 1, "t100_ml": 10},
                {"vial": 1, "recipe_id": 2, "t200_ml": 10},
            ])

    def test_accepts_documented_machine_limit_boundaries(self):
        rows = validate_queue([{
            "vial": 1, "recipe_id": 1, "t100_ml": 10,
            "t200_ml": 50, "t300_ml": 0, "t400_ml": 20,
            "total_ml": 80,
        }])

        self.assertEqual(rows[0]["total_ml"], 80)
        self.assertEqual(rows[0]["t200_ml"], 50)

        minimum = validate_queue([{
            "vial": 1, "recipe_id": 1, "t100_ml": 0,
            "t200_ml": 0, "t400_ml": 10,
        }])
        self.assertEqual(minimum[0]["t400_ml"], 10)

    def test_rejects_recipes_outside_documented_machine_limits(self):
        invalid_cases = [
            (
                {"vial": 1, "recipe_id": 1, "t100_ml": 10, "t200_ml": 50.01},
                "must not exceed 50 mL",
            ),
            (
                {
                    "vial": 1, "recipe_id": 1, "t100_ml": 30.04,
                    "t200_ml": 50, "total_ml": 80,
                },
                "total volume must not exceed 80 mL",
            ),
            (
                {"vial": 1, "recipe_id": 1, "t100_ml": 10, "t300_ml": 10},
                "T-300 is out of service",
            ),
        ]
        for row, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                LiveControlError, message,
            ):
                validate_queue([row])

        for field in ("t100_ml", "t200_ml", "t400_ml"):
            row = {"vial": 1, "recipe_id": 1, field: 9.99}
            with self.subTest(field=field), self.assertRaisesRegex(
                LiveControlError, rf"{field} must be 0 or at least 10 mL",
            ):
                validate_queue([row])

    def test_uses_calculated_total_after_accepting_small_rounding_difference(self):
        rows = validate_queue([{
            "vial": 1, "recipe_id": 1, "t100_ml": 10,
            "t200_ml": 10, "total_ml": 19.99,
        }])
        self.assertEqual(rows[0]["total_ml"], 20)

    def test_rejects_boolean_numeric_input(self):
        with self.assertRaisesRegex(LiveControlError, "vial must be a number"):
            validate_queue([{
                "vial": True, "recipe_id": 1, "t100_ml": 10,
            }])


class LivePreviewPathTests(unittest.TestCase):
    def make_controller(self, root):
        root = Path(root)
        code = root / "code"
        code.mkdir()
        return LiveRunController(
            live_root=root / "live",
            workspace=root,
            code_dir=code,
            camera_dir=None,
            model_path=None,
            tank_colors_path=None,
            image="unused",
        )

    def test_preview_frame_is_used_without_a_monitor_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "code"
            code.mkdir()
            preview = root / "preview" / "latest_frame.jpg"
            preview.parent.mkdir()
            preview.write_bytes(b"jpeg")
            controller = LiveRunController(
                live_root=root / "live",
                workspace=root,
                code_dir=code,
                camera_dir=None,
                model_path=None,
                tank_colors_path=None,
                image="unused",
                preview_frame=preview,
            )

            self.assertEqual(controller.paths()["frame"], preview.resolve())

    def test_freshest_frame_wins_when_preview_and_monitor_both_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "code"
            code.mkdir()
            preview = root / "preview" / "latest_frame.jpg"
            preview.parent.mkdir()
            preview.write_bytes(b"preview")
            run_dir = root / "run"
            run_dir.mkdir()
            monitor = run_dir / "latest_frame.jpg"
            monitor.write_bytes(b"monitor")
            controller = LiveRunController(
                live_root=root / "live",
                workspace=root,
                code_dir=code,
                camera_dir=None,
                model_path=None,
                tank_colors_path=None,
                image="unused",
                preview_frame=preview,
            )
            controller.state_path.write_text(
                json.dumps({"run_dir": str(run_dir)}), encoding="utf-8",
            )
            os.utime(preview, ns=(1_000_000_000, 1_000_000_000))
            os.utime(monitor, ns=(2_000_000_000, 2_000_000_000))
            self.assertEqual(controller.paths()["frame"], monitor.resolve())

            os.utime(preview, ns=(3_000_000_000, 3_000_000_000))
            self.assertEqual(controller.paths()["frame"], preview.resolve())

    def test_live_root_must_be_inside_mounted_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            code = workspace / "code"
            code.mkdir(parents=True)
            controller = LiveRunController(
                live_root=root / "outside",
                workspace=workspace,
                code_dir=code,
                camera_dir=None,
                model_path=None,
                tank_colors_path=None,
                image="unused",
            )
            with patch.object(controller, "_container_status", return_value="stopped"):
                checks, ready = controller.configured()
            self.assertFalse(checks["live_root"])
            self.assertFalse(ready)

    def test_docker_permission_error_is_reported_as_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            result = SimpleNamespace(
                returncode=1, stdout="", stderr="permission denied: docker.sock",
            )
            with patch.object(controller, "_docker", return_value=result):
                self.assertEqual(controller._container_status(), "unavailable")
                checks, ready = controller.configured()
            self.assertFalse(checks["docker"])
            self.assertFalse(ready)

    def test_missing_container_still_means_docker_is_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            result = SimpleNamespace(
                returncode=1, stdout="", stderr="Error: No such object: monitor",
            )
            with patch.object(controller, "_docker", return_value=result):
                self.assertEqual(controller._container_status(), "stopped")
                checks, _ready = controller.configured()
            self.assertTrue(checks["docker"])

    def test_start_revalidates_a_persisted_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            controller.queue_path.parent.mkdir(parents=True)
            controller.queue_path.write_text(
                "vial,recipe_id,t100_ml,t200_ml,t400_ml,total_ml\n"
                "1,1,40,50,10,100\n",
                encoding="utf-8",
            )
            with (
                patch.object(controller, "configured", return_value=({}, True)),
                patch.object(controller, "_container_status", return_value="stopped"),
            ):
                with self.assertRaisesRegex(
                    LiveControlError, "total volume must not exceed 80 mL",
                ):
                    controller.start()

    def test_start_rejects_invalid_alignment_without_truncating(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            controller.queue_path.parent.mkdir(parents=True)
            controller.queue_path.write_text(
                "vial,recipe_id,t100_ml,t200_ml,t400_ml,total_ml\n"
                "1,1,10,10,10,30\n",
                encoding="utf-8",
            )
            with (
                patch.object(controller, "configured", return_value=({}, True)),
                patch.object(controller, "_container_status", return_value="stopped"),
            ):
                for value in (1.9, "x", True, float("inf"), 0):
                    with self.subTest(value=value), self.assertRaisesRegex(
                        LiveControlError, "whole vial number",
                    ):
                        controller.start(start_vial=value)

    def test_realign_validates_before_stopping_active_monitor(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            with patch.object(controller, "stop") as stop:
                with self.assertRaisesRegex(LiveControlError, "whole vial number"):
                    controller.realign("not-a-vial")
            stop.assert_not_called()

    def test_realign_rejects_vial_outside_queue_before_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            controller.queue_path.parent.mkdir(parents=True)
            controller.queue_path.write_text(
                "vial,recipe_id,t100_ml,t200_ml,t400_ml,total_ml\n"
                "1,1,10,10,10,30\n",
                encoding="utf-8",
            )
            with patch.object(controller, "stop") as stop:
                with self.assertRaisesRegex(LiveControlError, "not in the saved queue"):
                    controller.realign(999)
            stop.assert_not_called()

    def test_run_snapshot_uses_revalidated_canonical_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            controller.queue_path.parent.mkdir(parents=True)
            controller.queue_path.write_text(
                "vial,recipe_id,t100_ml,t200_ml,t400_ml,total_ml\n"
                "1,1,10,10,0,19.99\n",
                encoding="utf-8",
            )
            docker_result = SimpleNamespace(stdout="container-id\n")
            with (
                patch.object(controller, "configured", return_value=({}, True)),
                patch.object(controller, "_container_status", return_value="stopped"),
                patch.object(controller, "_docker", return_value=docker_result),
            ):
                controller.start(start_vial="1")

            run_queue = Path(controller._read_state()["queue_path"])
            with run_queue.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["total_ml"], "20")


if __name__ == "__main__":
    unittest.main()
