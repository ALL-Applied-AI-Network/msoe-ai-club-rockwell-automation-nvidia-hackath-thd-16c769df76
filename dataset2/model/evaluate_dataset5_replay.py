#!/usr/bin/env python3
"""Evaluate the frozen synthetic model on corrected CSI Dataset 5 replay frames.

This script never trains or changes the model. Dataset 5 supplies binary
carryover labels only, so severity predictions are recorded but not scored.
Expected RGB comes from the existing leave-one-recipe-out calibration: the
tested recipe was excluded when its expected color was estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODEL_ROI = (200, 190, 420, 480)
IMAGE_SIZE = 160
RISK_NAMES = ("low", "medium", "high")
SEVERITY_NAMES = ("clean", "mild", "medium", "severe")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True, help="realtime_loro_results.csv")
    parser.add_argument("--timeline", required=True, help="color_timeline.csv")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def fractions(row):
    total = max(float(row["total_ml"]), 1.0)
    return np.asarray(
        [float(row["t100_ml"]), float(row["t200_ml"]), float(row["t400_ml"])],
        dtype=np.float32,
    ) / total


def risk_tier(previous, current):
    score = 0.5 * float(np.abs(previous - current).sum())
    return score, 2 if score >= 0.40 else 1 if score >= 0.25 else 0


def binary_metrics(truth, probability, threshold):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(probability) >= threshold
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def metrics_from_predictions(truth, prediction):
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def threshold_candidates(probability):
    values = sorted(set(float(value) for value in probability))
    midpoints = [(left + right) / 2 for left, right in zip(values, values[1:])]
    return sorted(set([0.0, 0.5, 1.0, *values, *midpoints]))


def roc_auc(truth, probability):
    positives = [p for y, p in zip(truth, probability) if y]
    negatives = [p for y, p in zip(truth, probability) if not y]
    wins = sum(a > b for a in positives for b in negatives)
    ties = sum(a == b for a in positives for b in negatives)
    return (wins + 0.5 * ties) / max(len(positives) * len(negatives), 1)


def preprocess(path: Path):
    with Image.open(path) as image:
        image = image.convert("RGB").crop(MODEL_ROI)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    labels_path = Path(args.labels).resolve()
    timeline_path = Path(args.timeline).resolve()
    images_dir = Path(args.images).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv(labels_path)
    by_vial = {int(float(row["vial"])): row for row in labels}
    scored = [row for row in labels if as_bool(row["scored"]) and int(float(row["vial"])) <= 24]
    timeline = read_csv(timeline_path)
    frames_by_vial = defaultdict(list)
    # color_timeline's original in_fill/vial columns are known to drift and
    # must not be used. Assign frames only from the corrected per-vial windows.
    for label in scored:
        vial = int(float(label["vial"]))
        start_s, end_s = float(label["start_s"]), float(label["end_s"])
        frames_by_vial[vial] = [
            row for row in timeline
            if start_s <= float(row["video_s"]) <= end_s and float(row["color_frac"]) > 0.015
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    dummy_image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    dummy_recipe = torch.zeros(1, 10, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_image, dummy_recipe)
    if device.type == "cuda":
        torch.cuda.synchronize()

    frame_rows = []
    vial_rows = []
    all_latencies = []
    for label in scored:
        vial = int(float(label["vial"]))
        current = fractions(label)
        previous_label = by_vial.get(vial - 1, label)
        previous = fractions(previous_label)
        change_score, risk_target = risk_tier(previous, current)
        expected = np.asarray(
            [float(label["pred_R"]), float(label["pred_G"]), float(label["pred_B"])],
            dtype=np.float32,
        ) / 255.0
        recipe_values = np.concatenate((previous, current, expected, np.asarray([1.0], np.float32)))
        recipe = torch.from_numpy(recipe_values).unsqueeze(0).to(device)
        vial_predictions = []
        for frame in sorted(frames_by_vial[vial], key=lambda row: float(row["video_s"])):
            started = time.perf_counter()
            image_path = images_dir / frame["filename"]
            image = preprocess(image_path).unsqueeze(0).to(device)
            with torch.no_grad():
                risk_logits, observed_logit, severity_logits = model(image, recipe)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            all_latencies.append(latency_ms)
            risk_prob = torch.softmax(risk_logits, dim=1)[0].cpu().numpy()
            severity_prob = torch.softmax(severity_logits, dim=1)[0].cpu().numpy()
            observed_probability = float(torch.sigmoid(observed_logit)[0].cpu())
            vial_predictions.append((observed_probability, risk_prob, severity_prob))
            frame_rows.append({
                "vial": vial,
                "recipe_id": label["recipe_id"],
                "filename": frame["filename"],
                "video_s": frame["video_s"],
                "color_frac": frame["color_frac"],
                "observed_probability": observed_probability,
                "risk_prediction": RISK_NAMES[int(risk_prob.argmax())],
                "severity_prediction": SEVERITY_NAMES[int(severity_prob.argmax())],
                "latency_ms": latency_ms,
            })
        if not vial_predictions:
            raise RuntimeError(f"no visible replay frames for scored vial {vial}")
        observed = sorted(item[0] for item in vial_predictions)
        scenario_probability = float(np.mean(observed[-min(3, len(observed)) :]))
        risk_mean = np.mean([item[1] for item in vial_predictions], axis=0)
        severity_mean = np.mean([item[2] for item in vial_predictions], axis=0)
        vial_rows.append({
            "vial": vial,
            "recipe_id": label["recipe_id"],
            "pos_in_batch": label["pos_in_batch"],
            "expected_carryover": int(as_bool(label["expected_carryover"])),
            "frame_count": len(vial_predictions),
            "observed_probability": scenario_probability,
            "observed_prediction_at_0_5": int(scenario_probability >= 0.5),
            "recipe_change_score": change_score,
            "risk_target_from_recipe": RISK_NAMES[risk_target],
            "risk_prediction": RISK_NAMES[int(risk_mean.argmax())],
            "severity_prediction_unscored": SEVERITY_NAMES[int(severity_mean.argmax())],
            "expected_r_loro": float(expected[0]),
            "expected_g_loro": float(expected[1]),
            "expected_b_loro": float(expected[2]),
        })

    truth = [bool(row["expected_carryover"]) for row in vial_rows]
    probability = [float(row["observed_probability"]) for row in vial_rows]
    fixed = binary_metrics(truth, probability, 0.5)
    thresholds = threshold_candidates(probability)
    best = max(
        (binary_metrics(truth, probability, threshold) for threshold in thresholds),
        key=lambda item: (item["balanced_accuracy"], item["f1"]),
    )
    cv_predictions = {}
    cv_thresholds = {}
    recipe_ids = sorted(set(row["recipe_id"] for row in vial_rows), key=float)
    for held_recipe in recipe_ids:
        training = [row for row in vial_rows if row["recipe_id"] != held_recipe]
        held = [row for row in vial_rows if row["recipe_id"] == held_recipe]
        training_truth = [bool(row["expected_carryover"]) for row in training]
        training_probability = [float(row["observed_probability"]) for row in training]
        chosen = max(
            (binary_metrics(training_truth, training_probability, threshold)
             for threshold in threshold_candidates(training_probability)),
            key=lambda item: (item["balanced_accuracy"], item["f1"], item["accuracy"]),
        )
        cv_thresholds[str(held_recipe)] = chosen["threshold"]
        for row in held:
            prediction = int(float(row["observed_probability"]) >= chosen["threshold"])
            cv_predictions[int(row["vial"])] = prediction
            row["loro_calibrated_threshold"] = chosen["threshold"]
            row["loro_calibrated_prediction"] = prediction
    cv_prediction_list = [cv_predictions[int(row["vial"])] for row in vial_rows]
    cv_metrics = metrics_from_predictions(truth, cv_prediction_list)
    risk_consistency = sum(
        row["risk_target_from_recipe"] == row["risk_prediction"] for row in vial_rows
    ) / len(vial_rows)
    latency = np.asarray(all_latencies)
    metrics = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256(model_path),
        "labels_sha256": sha256(labels_path),
        "timeline_sha256": sha256(timeline_path),
        "device": str(device),
        "scored_vials": len(vial_rows),
        "positive_vials": int(sum(truth)),
        "negative_vials": int(len(truth) - sum(truth)),
        "replay_frames": len(frame_rows),
        "fixed_synthetic_threshold": fixed,
        "roc_auc": roc_auc(truth, probability),
        "post_hoc_best_threshold_do_not_treat_as_test_accuracy": best,
        "leave_one_recipe_out_threshold_calibration": {
            **cv_metrics,
            "thresholds_by_held_recipe": cv_thresholds,
            "method": "Each held recipe is scored using a threshold selected from all other recipes only.",
        },
        "risk_tier_consistency_with_recipe_rule": risk_consistency,
        "latency_ms": {
            "mean": float(latency.mean()),
            "median": float(np.median(latency)),
            "p95": float(np.percentile(latency, 95)),
            "max": float(latency.max()),
            "end_to_end_fps_from_mean": float(1000.0 / latency.mean()),
        },
        "limitations": [
            "Dataset 5 ground truth is expected carryover by batch position, not an independent laboratory assay.",
            "Expected RGB uses existing leave-one-recipe-out calibration from other Dataset 5 recipes.",
            "Dataset 5 has no reviewed severity labels, so severity predictions are not scored.",
            "The post-hoc threshold is descriptive only and must not be reported as unbiased test accuracy.",
            "Leave-one-recipe-out threshold results are cross-validated but remain uncertain because only 23 vials are scored.",
        ],
        "frame_assignment": "corrected start_s/end_s windows; original timeline in_fill/vial columns ignored",
    }
    write_csv(output_dir / "frame_predictions.csv", frame_rows)
    write_csv(output_dir / "vial_predictions.csv", vial_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Dataset 5 recorded-video replay evaluation\n\n"
        "Frozen model evaluation only; no training occurred. Binary carryover is scored per completed vial "
        "using corrected Dataset 5 labels. Expected color is leave-one-recipe-out calibrated. Severity "
        "is recorded but cannot be scored because Dataset 5 has no reviewed severity ground truth.\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
