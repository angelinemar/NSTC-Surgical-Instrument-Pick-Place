# Matched A/C presentation and inference-resolution correction (2026-10-19)

The presentation package is `/home/user/NSTC/presentation_RFDETR_20261019`. Its three sections are A tested on synthetic, A tested on DP, and C (mixed training) tested on both. A completed 100 epochs; C completed 70 with early stopping. B was interrupted during epoch 38, after 37 completed validation epochs; it has no final `result.json` and is excluded from final comparisons.

## Critical resolution correction

These exported `.pth` checkpoints omit training resolution from `args` and do not contain `model_config`. Installed RF-DETR 1.11.0 `from_checkpoint()` consequently instantiated Seg Small at its default 384 resolution. The standalone evaluator and old prediction galleries did not override that default. Old standalone results are therefore **384 inference results**, not native-768 results. They are preserved with correction notices and must not be mixed with the corrected comparison.

The evaluator now derives resolution from `trial_config.json`, passes it explicitly to `from_checkpoint`, and records `segmentation_native_resolution_v2` in its cache configuration. It rejects a conflicting resolution and refuses an unknown resolution without an explicit override. Training's best-checkpoint reload also explicitly passes its configured resolution. Existing cache/config mismatch prevents silently reusing old 384 results. Regression tests cover sidecar recovery and forbidden silent fallback.

The archived training `result.json` is a separate evaluation path with explicit input-resolution settings and different metric implementation. Use the new matched results—not a mixture of those historical summaries and standalone results—for all cross-domain/model comparisons.

## Matched benchmark

All eight evaluations use the same fixed C evaluation subsets: A/C × synthetic/DP × valid/test, 1,350 images each. Synthetic uses its entire split; DP uses the balanced selected subset (225 images per camera), not all original DP frames. Annotation SHA256 and checkpoint hashes are in `comparison_provenance.json`.

Inference: 768, FP16, no compile. Evaluator: COCO-style box and mask AP, score floor 0.001, 100 detections per category/image, fixed confidence 0.50 and IoU 0.50 for P/R/F1. Report percentages. No classification accuracy is invented. Validation is labelled separately from test.

| Model / test domain | Box AP50:95 | Mask AP50:95 |
|---|---:|---:|
| A / synthetic | 93.77% | 64.39% |
| C / synthetic | 93.38% | 62.87% |
| A / DP subset | 78.88% | 25.23% |
| C / DP subset | 90.10% | 53.72% |

C improves DP mask AP by about 28.49 percentage points with a 1.52-point synthetic mask decrease. This is a matched-data comparison, not equal-compute training: C uses twice the training images and a different validation source, and the models run for different epoch counts. DP labels still have known omissions. Reused test sets were previously inspected; obtain a new blinded holdout for final claims.

## Media scope

PNG contact sheets and page-cycle GIFs cover all 23,400 selected C images, with per-split source CSVs. A's synthetic collection is the same synthetic half. DP train imagery is explicitly labelled C-only. The original 1,000 H5 files are catalogued, but their every raw frame is not duplicated as separate presentation PNGs.

Prediction GIFs show every one of 157 recorded frames of one test pick episode, for all six cameras, comparing RGB/GT/A/C at 768. This is one full recorded pick H5, not a whole pick+place session. Synthetic GIFs show all nine captured viewpoints of one static scene. Actual camera/light metadata diagrams and RGB examples are included; lighting examples are not a controlled causal ablation.

Scripts for rebuilding charts, galleries, prediction movies and PPT are included in the presentation root. Dataset and model symlinks point to existing local sources; PNG, GIF, CSV and PowerPoint artifacts are standalone. The web index separates the three experiment/evaluation sections.
