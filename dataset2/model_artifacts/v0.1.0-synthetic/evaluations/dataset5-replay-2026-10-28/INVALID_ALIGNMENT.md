# Invalidated evaluation

Do not report the metrics in this directory.

This first Dataset 5 evaluation selected frames using the original `color_timeline.csv` `in_fill` and `vial` columns. Those columns inherit the known timestamp drift from the original labels. The end-to-end monitor replay exposed the frame-count discrepancy.

Use the corrected evaluation instead: `../dataset5-corrected-replay-2026-10-28`. It assigns frames exclusively from `video5_labels_corrected.csv` `start_s` and `end_s` windows and explicitly ignores the original timeline assignments.
