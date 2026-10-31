#!/usr/bin/env python3
"""Train, evaluate, and export the practical v1 carryover model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CarryoverFrameDataset, build_frame_cache
from model import RECIPE_FEATURES, RecipeConditionedCarryoverNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(truth, probability, threshold=0.5):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(probability) >= threshold
    tp = int(((prediction == 1) & (truth == 1)).sum())
    fp = int(((prediction == 1) & (truth == 0)).sum())
    fn = int(((prediction == 0) & (truth == 1)).sum())
    tn = int(((prediction == 0) & (truth == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": (recall + specificity) / 2,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    frame_truth, frame_prob = [], []
    groups = defaultdict(lambda: {"observed": [], "severity": [], "risk": [], "target": None,
                                  "risk_target": None, "visible": []})
    risk_correct = severity_correct = samples = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        recipe = batch["recipe"].to(device, non_blocking=True)
        output = model(image, recipe)
        observed_prob = torch.sigmoid(output["observed_logit"]).cpu().numpy()
        severity_prob = torch.softmax(output["severity_logits"], dim=1).cpu().numpy()
        risk_prob = torch.softmax(output["risk_logits"], dim=1).cpu().numpy()
        observed_truth = batch["observed_target"].numpy().astype(int)
        severity_truth = batch["severity_target"].numpy()
        risk_truth = batch["risk_target"].numpy()
        frame_truth.extend(observed_truth.tolist())
        frame_prob.extend(observed_prob.tolist())
        risk_correct += int((risk_prob.argmax(1) == risk_truth).sum())
        severity_correct += int((severity_prob.argmax(1) == severity_truth).sum())
        samples += len(observed_truth)
        for i, scenario_id in enumerate(batch["scenario_id"]):
            group = groups[scenario_id]
            group["observed"].append(float(observed_prob[i]))
            group["severity"].append(severity_prob[i])
            group["risk"].append(risk_prob[i])
            group["target"] = int(severity_truth[i] != 0)
            group["severity_target"] = int(severity_truth[i])
            group["risk_target"] = int(risk_truth[i])
            group["visible"].append(bool(batch["liquid_visible"][i]))

    scenario_truth, scenario_probability, predictions = [], [], []
    scenario_severity_correct = scenario_risk_correct = 0
    for scenario_id, group in sorted(groups.items()):
        visible_observed = [p for p, visible in zip(group["observed"], group["visible"]) if visible]
        observed_values = sorted(visible_observed or group["observed"])
        observed_probability = float(np.mean(observed_values[-min(3, len(observed_values)) :]))
        severity_probability = np.mean(group["severity"], axis=0)
        risk_probability = np.mean(group["risk"], axis=0)
        scenario_truth.append(group["target"])
        scenario_probability.append(observed_probability)
        scenario_severity_correct += int(int(severity_probability.argmax()) == group["severity_target"])
        scenario_risk_correct += int(int(risk_probability.argmax()) == group["risk_target"])
        predictions.append({
            "scenario_id": scenario_id,
            "observed_target": group["target"],
            "observed_probability": observed_probability,
            "severity_target": group["severity_target"],
            "severity_prediction": int(severity_probability.argmax()),
            "risk_target": group["risk_target"],
            "risk_prediction": int(risk_probability.argmax()),
        })
    metrics = {
        "frame_observed": binary_metrics(frame_truth, frame_prob),
        "scenario_observed": binary_metrics(scenario_truth, scenario_probability),
        "frame_severity_accuracy": severity_correct / max(samples, 1),
        "frame_risk_accuracy": risk_correct / max(samples, 1),
        "scenario_severity_accuracy": scenario_severity_correct / max(len(groups), 1),
        "scenario_risk_accuracy": scenario_risk_correct / max(len(groups), 1),
        "scenarios": len(groups),
        "frames": samples,
    }
    return metrics, predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    # YAAD batch containers have a small /dev/shm; multiprocessing workers can
    # bus-error while exchanging image batches. Single-process loading is safe
    # and fast enough for this compact cached dataset.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--queue-dropout", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--max-scenarios-per-split", type=int)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    build_frame_cache(dataset_root, cache_dir, args.image_size,
                      max_scenarios_per_split=args.max_scenarios_per_split)
    train_set = CarryoverFrameDataset(cache_dir, "train")
    validation_set = CarryoverFrameDataset(cache_dir, "validation")
    test_set = CarryoverFrameDataset(cache_dir, "test")
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers,
                   "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = RecipeConditionedCarryoverNet().to(device)
    observed_targets = [int(row["observed_target"]) for row in train_set.rows]
    positives = sum(observed_targets)
    negatives = len(observed_targets) - positives
    observed_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1), device=device))
    severity_counts = np.bincount([int(row["severity_target"]) for row in train_set.rows], minlength=4)
    severity_weights = torch.tensor(len(train_set) / np.maximum(severity_counts, 1), dtype=torch.float32, device=device)
    severity_weights /= severity_weights.mean()
    severity_loss = nn.CrossEntropyLoss(weight=severity_weights)
    risk_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ready = json.loads((dataset_root / "READY.json").read_text(encoding="utf-8"))
    config = {**vars(args), "dataset_root": str(dataset_root), "output_dir": str(output_dir),
              "cache_dir": str(cache_dir), "dataset_ready": ready,
              "vial_manifest_sha256": sha256(dataset_root / "manifests" / "master_vial_manifest.csv"),
              "recipe_features": list(RECIPE_FEATURES), "device": str(device),
              "scenario_aggregation": "mean of top 3 visible-frame probabilities",
              "checkpoint_selection": "frame observed balanced accuracy + 0.50 scenario severity accuracy + 0.20 scenario risk accuracy",
              "created_at": datetime.now(timezone.utc).isoformat()}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_f1, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            recipe = batch["recipe"].to(device, non_blocking=True)
            risk_target = batch["risk_target"].to(device, non_blocking=True)
            observed_target = batch["observed_target"].to(device, non_blocking=True)
            severity_target = batch["severity_target"].to(device, non_blocking=True)
            queue_available = torch.rand(len(recipe), device=device) >= args.queue_dropout
            observed_recipe = recipe.clone()
            observed_recipe[~queue_available, :-1] = 0.0
            observed_recipe[~queue_available, -1] = 0.0
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(image, observed_recipe)
                visible = batch["liquid_visible"].to(device)
                loss_observed = observed_loss(output["observed_logit"], observed_target)
                loss_severity = severity_loss(output["severity_logits"][visible], severity_target[visible])
                if queue_available.any():
                    loss_risk = risk_loss(output["risk_logits"][queue_available], risk_target[queue_available])
                else:
                    loss_risk = output["risk_logits"].sum() * 0.0
                loss = loss_observed + 0.60 * loss_severity + 0.35 * loss_risk
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(image)
        scheduler.step()
        validation_metrics, _ = evaluate(model, validation_loader, device)
        score = (validation_metrics["frame_observed"]["balanced_accuracy"]
                 + 0.50 * validation_metrics["scenario_severity_accuracy"]
                 + 0.20 * validation_metrics["scenario_risk_accuracy"])
        epoch_record = {"epoch": epoch, "train_loss": running_loss / len(train_set),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "selection_score": score, "validation": validation_metrics}
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                    "validation_metrics": validation_metrics}, output_dir / "last.pt")
        if score > best_f1 + 1e-6:
            best_f1, stale = score, 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "validation_metrics": validation_metrics}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics, validation_predictions = evaluate(model, validation_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)
    metrics = {"best_epoch": checkpoint["epoch"], "validation": validation_metrics, "test": test_metrics,
               "history": history}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name, rows in (("validation_predictions.csv", validation_predictions),
                       ("test_predictions.csv", test_predictions)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    model.eval()
    example_image = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    example_recipe = torch.zeros(1, len(RECIPE_FEATURES), device=device)
    class ExportWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, image, recipe):
            output = self.inner(image, recipe)
            return output["risk_logits"], output["observed_logit"], output["severity_logits"]
    wrapper = ExportWrapper(model)
    traced = torch.jit.trace(wrapper, (example_image, example_recipe))
    traced.save(str(output_dir / "model.ts"))
    try:
        torch.onnx.export(wrapper, (example_image, example_recipe), output_dir / "model.onnx",
                          input_names=["image", "recipe"],
                          output_names=["risk_logits", "observed_logit", "severity_logits"],
                          dynamic_axes={"image": {0: "batch"}, "recipe": {0: "batch"},
                                        "risk_logits": {0: "batch"}, "observed_logit": {0: "batch"},
                                        "severity_logits": {0: "batch"}}, opset_version=17)
    except Exception as exc:
        (output_dir / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Carryover model v1 experiment\n\n"
        f"Dataset: v4.0.0 (`{ready['vial_manifest_sha256']}`)\n\n"
        "Outputs: `best.pt` (PyTorch), `model.ts` (TorchScript), optional `model.onnx`, "
        "`metrics.json`, and scenario predictions. Risk means recipe-transition risk; observed error "
        "means a visible mismatch during filling. Synthetic test results require real-data calibration.\n",
        encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                      "best_epoch": checkpoint["epoch"], "validation": validation_metrics,
                      "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

