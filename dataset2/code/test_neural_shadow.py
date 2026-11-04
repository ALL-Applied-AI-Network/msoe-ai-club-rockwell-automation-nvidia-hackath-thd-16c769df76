import importlib
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class FakeModel:
    def __init__(self):
        self.moved_to = []
        self.eval_calls = 0
        self.warmup_calls = 0

    def to(self, device):
        self.moved_to.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, _image, _recipe):
        self.warmup_calls += 1


class FakeJit:
    def __init__(self, model):
        self.model = model
        self.load_calls = []

    def load(self, path, *, map_location):
        self.load_calls.append((path, map_location))
        return self.model


class NeuralShadowMonitorLoadTest(unittest.TestCase):
    def test_loads_on_cpu_before_moving_model_to_selected_cuda_device(self):
        device = SimpleNamespace(type="cuda")
        model = FakeModel()
        jit = FakeJit(model)
        cuda = SimpleNamespace(
            is_available=lambda: True,
            synchronize=Mock(),
        )
        fake_torch = SimpleNamespace(
            cuda=cuda,
            device=lambda _name: device,
            jit=jit,
            no_grad=nullcontext,
            zeros=lambda *args, **kwargs: (args, kwargs),
        )
        fake_numpy = ModuleType("numpy")
        fake_pil = ModuleType("PIL")
        fake_pil.Image = SimpleNamespace()

        with tempfile.TemporaryDirectory() as temporary:
            prediction_log = Path(temporary) / "predictions.jsonl"
            modules = {"torch": fake_torch, "numpy": fake_numpy, "PIL": fake_pil}
            with patch.dict(sys.modules, modules):
                sys.modules.pop("neural_shadow", None)
                NeuralShadowMonitor = importlib.import_module(
                    "neural_shadow"
                ).NeuralShadowMonitor
                monitor = NeuralShadowMonitor("model.ts", prediction_log)
            sys.modules.pop("neural_shadow", None)

        self.assertIs(monitor.device, device)
        self.assertEqual(jit.load_calls, [("model.ts", "cpu")])
        self.assertEqual(model.moved_to, [device])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.warmup_calls, 3)
        cuda.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
