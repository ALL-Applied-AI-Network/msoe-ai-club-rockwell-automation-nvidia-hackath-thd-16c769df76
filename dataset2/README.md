# Dataset 2: Fluid Mixing Station 3

This folder contains the complete prototype pipeline for predicting recipe-transition carryover and checking vial color during filling.

## Pipeline

```text
manual recipe queue -> transition-risk warning
camera frames       -> color rules + frozen neural shadow model
                    -> JSONL events + probabilities + latest frame
                    -> read-only operator dashboard
```

The manually entered queue is the hackathon workaround for the unavailable live PLC queue. In production, the same schema can be populated automatically by the PLC/MES integration without changing the model interface.

## Corrected Dataset 5 alignment

The original video-5 frame-label windows drift from the vial-fill images. Image-based segmentation shows that the recording contains vials 1-25, with vial 25 cut off. Use `results/video5_labels_corrected.csv` for vials 1-24 when evaluating complete fills.

## Important files

- `code/live_monitor.py`: recorded replay and live-folder monitor
- `code/neural_shadow.py`: frozen model preprocessing and inference
- `dashboard/`: live operator interface
- `model/`: model, data loader, training, and replay evaluation
- `model_artifacts/v0.1.0-synthetic/`: auditable metrics and predictions
- `synthetic_data/production_plan_v4/`: frozen 480-scenario plan
- `synthetic_data/samples/`: compact visual and video examples
- `results/video5_labels_corrected.csv`: corrected real recording windows

See `DATASET_CARD.md` and `MODEL_CARD.md` before interpreting the results.

