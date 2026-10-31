# Synthetic carryover data

This folder publishes the reproducible parts of synthetic dataset v4.0.0 without committing the approximately 2.2 GiB rendered corpus.

- `production_plan_v4/`: frozen recipes, palettes, split definition, scenario manifest, and shard plans
- `tools/`: planning, rendering, validation, and publishing utilities
- `v4_metadata/`: readiness record, per-vial manifest, source-job record, and QA summary
- `samples/validation/`: representative clean and carryover frames used for visual review
- `samples/live_fill_smoke/`: compact 30 fps vial-fill smoke test with labels and metadata

The plan contains 480 scenarios across clean, mild, medium, and severe classes. Transition families and palette families do not cross train/validation/test boundaries. Regenerated data should be checked against the committed plan and metadata before training.

