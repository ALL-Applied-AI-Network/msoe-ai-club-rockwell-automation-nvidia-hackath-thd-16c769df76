# Dataset card

## Supplied real data

The project uses the hackathon's Fluid Mixing Station 3 Dataset 5 for real-domain evaluation. The team corrected fill timing because the original frame-label windows drift relative to the vial-fill images. The recording contains 24 complete scored fills and one cut-off final fill.

Ground truth is operational rather than laboratory-confirmed: the first vial after each recipe change is treated as expected carryover. This creates seven positive and sixteen negative scored vials. Results must be interpreted with that limitation.

Raw real-machine media is not duplicated in this repository. Corrected windows and derived evaluation outputs are included.

## Synthetic v4.0.0

- 480 scenarios
- 240 clean, 80 mild, 80 medium, and 80 severe
- 320 train, 80 validation, and 80 test
- 12 palettes, 24 recipes, and 60 transition families
- deterministic seeds and frozen manifests
- transition and palette leakage checks passed

The full rendered corpus is approximately 2.2 GiB and remains outside Git. Generation code, the frozen production plan, metadata, checksums, and compact samples are included so the corpus can be regenerated and audited.

## Intended use

- prototype training and controlled validation of carryover detection
- dashboard and streaming integration tests
- testing robustness across known synthetic nuisance factors

## Out-of-scope use

- production acceptance without real-machine calibration
- final claims based only on synthetic validation
- tray/cap inspection or other camera stations
- autonomous PLC or equipment control

