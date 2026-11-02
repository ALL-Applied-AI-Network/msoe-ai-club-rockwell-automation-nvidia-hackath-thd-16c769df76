import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_control import DemoRunController


class DemoRunControllerTests(unittest.TestCase):
    def make_controller(self, root):
        workspace = Path(root)
        code = workspace / "code"
        data = workspace / "dataset5"
        model = workspace / "model.ts"
        tanks = workspace / "tank_colors.npy"
        queue = workspace / "queue.csv"
        code.mkdir()
        data.mkdir()
        (code / "live_monitor.py").write_text("", encoding="utf-8")
        (code / "neural_shadow.py").write_text("", encoding="utf-8")
        (data / "image_index.csv").write_text("filename\n", encoding="utf-8")
        model.write_bytes(b"model")
        tanks.write_bytes(b"colors")
        queue.write_text("vial,recipe_id\n1,5\n", encoding="utf-8")
        return DemoRunController(
            demo_root=workspace / "runs", workspace=workspace, code_dir=code,
            data_root=data, queue_path=queue, model_path=model,
            tank_colors_path=tanks, image="fixed-image:1",
        )

    def test_configuration_requires_both_runtime_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            checks, ready = controller.configured()
            self.assertTrue(ready)
            (controller.code_dir / "neural_shadow.py").unlink()
            checks, ready = controller.configured()
            self.assertFalse(ready)
            self.assertFalse(checks["code"])

    def test_start_uses_only_fixed_realtime_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(temporary)
            calls = []

            def fake_docker(*arguments, **_kwargs):
                calls.append(arguments)
                if arguments[0] == "inspect":
                    return SimpleNamespace(returncode=1, stdout="", stderr="")
                return SimpleNamespace(returncode=0, stdout="container-id\n", stderr="")

            with patch.object(controller, "_docker", side_effect=fake_docker):
                status = controller.start()

            run_call = next(call for call in calls if call[0] == "run")
            self.assertIn("--gpus", run_call)
            self.assertIn("replay", run_call)
            self.assertIn("--speed", run_call)
            self.assertEqual(run_call[run_call.index("--speed") + 1], "1")
            self.assertEqual(run_call[run_call.index("--model-threshold") + 1], "0.0071")
            self.assertEqual(status["profile"], "dataset5_real_time")
            self.assertTrue(Path(status["queue_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
