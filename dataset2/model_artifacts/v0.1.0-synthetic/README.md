# Carryover model v1 experiment

Dataset: v4.0.0 (`84d513802332cd41a4b6ff72ad45164a798aefbe6cf1b67905c754ca4de714d1`)

The original experiment produced `best.pt` (PyTorch), `model.ts` (TorchScript), `model.onnx`, `metrics.json`, and scenario predictions. Weight files are intentionally excluded from Git; their checksums remain in `ARTIFACT_CHECKSUMS.sha256` and they can be reproduced with the committed training code and frozen dataset plan.

This directory publishes the configuration, source checksums, metrics, scenario predictions, corrected Dataset 5 evaluation, and shadow-monitor integration logs. Risk means recipe-transition risk; observed error means a visible mismatch during filling. Synthetic test results require real-data calibration.
