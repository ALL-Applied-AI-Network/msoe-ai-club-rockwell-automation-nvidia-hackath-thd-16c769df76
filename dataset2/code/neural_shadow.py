"""Advisory-only neural shadow inference for the live carryover monitor."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_ROI = (200, 190, 420, 480)
IMAGE_SIZE = 160
RISK_NAMES = ("LOW", "MEDIUM", "HIGH")
SEVERITY_NAMES = ("clean", "mild", "medium", "severe")


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fractions(recipe):
    total = max(float(recipe.get("total_ml", 0)), 1.0)
    return np.asarray(
        [float(recipe.get("t100_ml", 0)), float(recipe.get("t200_ml", 0)),
         float(recipe.get("t400_ml", 0))],
        dtype=np.float32,
    ) / total


class NeuralShadowMonitor:
    """Runs the frozen model without controlling equipment or changing rules."""

    def __init__(
        self,
        model_path,
        prediction_log,
        threshold=0.0071,
        window=5,
        min_frames=3,
        tank_colors_path=None,
    ):
        import torch

        if window < 1 or min_frames < 1 or min_frames > window:
            raise ValueError("require 1 <= min_frames <= window")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Loading directly onto CUDA can make the JIT loader request a large
        # contiguous allocation before unified-memory page cache is reclaimed.
        # Loading on CPU first keeps startup reliable, then transfers only the
        # model's actual parameters to the selected inference device.
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model = self.model.to(self.device).eval()
        self.threshold = float(threshold)
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.recent = deque(maxlen=self.window)
        self.current_vial = None
        self.alerted = False
        self.tank_colors = None
        if tank_colors_path and Path(tank_colors_path).is_file():
            self.tank_colors = np.load(tank_colors_path).astype(np.float32)
            if self.tank_colors.shape != (3, 3):
                raise ValueError(f"expected 3x3 tank colors, got {self.tank_colors.shape}")
        self.prediction_log = Path(prediction_log)
        self.prediction_log.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_log.write_text("", encoding="utf-8")
        dummy_image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        dummy_recipe = torch.zeros(1, 10, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy_image, dummy_recipe)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def start_vial(self, vial):
        self.current_vial = int(vial)
        self.recent.clear()
        self.alerted = False

    def _expected_color(self, recipe):
        for names, source in ((('expected_r', 'expected_g', 'expected_b'), 'queue'),
                              (('pred_R', 'pred_G', 'pred_B'), 'loro_queue')):
            if all(name in recipe and _finite(recipe[name]) for name in names):
                rgb = np.asarray([float(recipe[name]) for name in names], dtype=np.float32)
                if float(rgb.max()) > 1.5:
                    rgb /= 255.0
                return np.clip(rgb, 0.0, 1.0), source, 1.0
        if self.tank_colors is not None:
            total = max(float(recipe.get("total_ml", 0)), 1.0)
            # Saved tank-color rows are [T200 orange, T400 blue, T100 clear].
            mix = np.asarray(
                [float(recipe.get("t200_ml", 0)), float(recipe.get("t400_ml", 0)),
                 float(recipe.get("t100_ml", 0))], dtype=np.float32,
            ) / total
            return np.clip((mix @ self.tank_colors) / 255.0, 0.0, 1.0), "tank_calibration", 0.75
        return np.zeros(3, dtype=np.float32), "unavailable", 0.0

    def _recipe_tensor(self, previous, current):
        expected, source, confidence = self._expected_color(current)
        values = np.concatenate(
            (_fractions(previous), _fractions(current), expected,
             np.asarray([confidence], dtype=np.float32))
        )
        tensor = self.torch.from_numpy(values).unsqueeze(0).to(self.device)
        return tensor, source, confidence

    def _image_tensor(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB").crop(MODEL_ROI)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = self.torch.from_numpy(array).permute(2, 0, 1)
        return ((tensor - 0.5) / 0.5).unsqueeze(0).to(self.device)

    def predict(self, path, captured_at, vial, previous_recipe, current_recipe):
        started = time.perf_counter()
        image = self._image_tensor(path)
        recipe, expected_source, queue_confidence = self._recipe_tensor(
            previous_recipe, current_recipe
        )
        with self.torch.no_grad():
            risk_logits, observed_logit, severity_logits = self.model(image, recipe)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        risk_probability = self.torch.softmax(risk_logits, dim=1)[0].cpu().numpy()
        severity_probability = self.torch.softmax(severity_logits, dim=1)[0].cpu().numpy()
        observed_probability = float(self.torch.sigmoid(observed_logit)[0].cpu())
        self.recent.append(observed_probability)
        top = sorted(self.recent)[-min(3, len(self.recent)):]
        smoothed_probability = float(np.mean(top))
        threshold_hits = sum(value >= self.threshold for value in self.recent)
        ready = len(self.recent) >= self.min_frames
        new_alert = bool(ready and threshold_hits >= self.min_frames and not self.alerted)
        if new_alert:
            self.alerted = True
        record = {
            "time": captured_at.isoformat(),
            "vial": int(vial),
            "filename": Path(path).name,
            "observed_probability": observed_probability,
            "smoothed_probability": smoothed_probability,
            "threshold": self.threshold,
            "window_frames": len(self.recent),
            "threshold_hits": threshold_hits,
            "new_shadow_alert": new_alert,
            "shadow_alert_active": self.alerted,
            "risk_prediction": RISK_NAMES[int(risk_probability.argmax())],
            "risk_probability": float(risk_probability.max()),
            "severity_prediction": SEVERITY_NAMES[int(severity_probability.argmax())],
            "severity_probability": float(severity_probability.max()),
            "expected_color_source": expected_source,
            "queue_confidence": queue_confidence,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "advisory_only": True,
        }
        with self.prediction_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
