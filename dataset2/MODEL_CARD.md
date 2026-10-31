# Model card: carryover detector v0.1.0-synthetic

## Purpose

The model is an advisory visual classifier for vial filling. It combines a fixed image crop with recipe context and produces observed-defect probability, severity, and transition-risk outputs. It runs beside interpretable recipe/color rules rather than replacing them.

## Training

Training used the frozen synthetic v4.0.0 corpus. Configuration, source checksums, training history, validation/test predictions, and exported-artifact checksums are stored under `model_artifacts/v0.1.0-synthetic`. Weight files are intentionally excluded from Git.

## Corrected Dataset 5 evaluation

| Metric | Result |
|---|---:|
| Scored vials | 23 |
| Positive / negative | 7 / 16 |
| ROC AUC | 0.839 |
| Leave-one-recipe-out precision | 55.6% |
| Leave-one-recipe-out recall | 71.4% |
| Leave-one-recipe-out specificity | 75.0% |
| Leave-one-recipe-out F1 | 62.5% |
| Leave-one-recipe-out balanced accuracy | 73.2% |
| Mean / p95 latency | 4.1 / 4.9 ms |

The synthetic threshold of 0.5 found no real positives. This domain shift is a central result: thresholds must be calibrated on representative real fills. A post-hoc threshold reached higher descriptive performance but is not reported as unbiased test accuracy.

## Limitations

- Only 23 real vials are scored.
- Ground truth is inferred from batch position, not lab assay.
- Dataset 5 has no reviewed severity labels.
- Expected RGB calibration uses other Dataset 5 recipes.
- Camera position, lighting, recipe set, and liquid properties may change at CSI.

## Safety

The model is advisory only. It does not write to the PLC, stop equipment, divert product, or make a final quality disposition. Operators must verify alerts, and a separate real validation/test program is required before deployment.

