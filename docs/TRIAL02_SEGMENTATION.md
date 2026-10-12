# Trial 02: standalone RF-DETR instance segmentation

Trial 01 is preserved at `/home/user/NSTC/datasets/Object datasets/trials/trial_01_rfdetr_small_bbox_ring448`: training/checkpoint, original dataset links, and DP evaluation. It trained RF-DETR Small **bounding boxes**, native 448, ring/top cameras. Its synthetic test AP50:95 was 78.13%, AP50 98.27%. These are not segmentation scores.

Trial 02 lives at `/home/user/NSTC/datasets/Object datasets/trials/trial_02_rfdetr_seg_hemisphere768`. Its large recording and full training are **pending user execution**. The old detector GUI is the Trial 01 workflow; use the new terminal launcher below for Trial 02.

## Run on this workstation

```bash
cd /home/user/NSTC
./start-segmentation6.sh record --train-scenes 1000 --valid-scenes 150 --test-scenes 150
```

This opens Isaac Sim with a viewport, generates 11,700 images (9 views × 1,300 scenes), then audits and exports the three COCO splits. Add `--headless` for recording without a window. There are no robot actions or successful-pick requirements. The robot may appear statically in 25% of scenes. Progress prints in the terminal; full logs are under the trial's `logs/` folder. On interruption, rerun the **same command** to resume committed scenes. Do not change protocol files in an existing trial.

The default 5–25 tools per scene are deliberately bounded by placement capacity. Classes are allocated cyclically for balanced requested counts, while visible class counts can differ with occlusion/cropping; inspect `dataset/manifest.json` after export. The recorder rejects invalid placement, inconsistent masks and blank views; blank camera poses are resampled within the same split and distance band.

After recording/export finishes:

```bash
./start-segmentation6.sh train --epochs 100 --batch-size 1 --grad-accum-steps 16 --num-workers 8
```

RF-DETR-Seg Small, native 768, effective batch 16, learning rate 0.0001, EMA, seed 42, early stopping patience 15. This is a tested conservative starting configuration, not an empirically optimized claim. The completed GPU smoke test used physical batch 1; batch 32 from the old box-only 448 run has not been validated for segmentation at 768. W&B is enabled under `nstc-surgical-segmentation`; training also saves local configuration, TensorBoard logs and checkpoints. Early stopping may finish before 100 epochs.

Results go into `trial_02_rfdetr_seg_hemisphere768/training/YYYYMMDD_HHMMSS/`. Evaluate the resulting best checkpoint:

```bash
TRIAL='/home/user/NSTC/datasets/Object datasets/trials/trial_02_rfdetr_seg_hemisphere768'
# Replace the timestamp with the actual training directory printed by training.
CKPT="$TRIAL/training/YYYYMMDD_HHMMSS/checkpoint_best_total.pth"
./start-segmentation6.sh evaluate --checkpoint "$CKPT"
./start-segmentation6.sh evaluate-dp --checkpoint "$CKPT" --split test
```

The first command evaluates every synthetic test image. The second evaluates every frame from all six cameras in the original DP test split. To inspect DP validation separately, use `--split valid`; never train on these recordings. All-frame DP evaluation is substantial. Each evaluation writes `metrics.json`, readable `README.md`, `cases.json`, per-camera/distance example images and best/worst examples. Precision/recall/F1 use confidence 0.5 and matching IoU 0.5; AP integrates over confidence scores, AP50:95 over IoU 0.50–0.95. Reports include box and mask AP, per-class scores, and camera/distance groups. Percentages are not a single classification “accuracy”: object detection has no useful count of true negatives.

For evaluation resume, rerun with `--output '/exact/previous/evaluation/folder'` and otherwise identical arguments. The cache avoids repeated model inference. For training resume, use the same `--output` and `--resume '/path/to/last.ckpt'` (a full Lightning checkpoint, not `.pth`); dataset hash, resolution and seed must match. Set the original batch/accumulation settings too.

## Generation protocol

- Cameras face the table center and sample continuous positions over elevation 15–85 degrees, at close 0.40–0.65 m, medium 0.70–1.10 m and far 1.20–1.85 m. Nine cameras are sampled per scene, three per distance range. Views may crop instruments; labels describe visible instances.
- Angular support is disjoint: within each 60-degree azimuth sector, train uses 1–35 degrees, validation 37–47, test 49–59. This covers the table from all sides without a predefined list of fixed poses.
- A frozen scan of all 1,000 DP H5 files found 690,018 calibration rows and 16 unique positions after 2 cm quantization. Cameras reject locations within 12 cm of reserved DP positions plus 2 cm quantization allowance. This reserves **position neighborhoods**, not every viewing direction at every distance. Reservation SHA is checked before recording.
- Four lights vary position, distance, height (0.7–2.2 m), intensity (1,000–10,000 when enabled), color and cone angle. Ambient illumination varies 120–500. Train uses combinations of at least two lights; validation reserves key-only/fill-only, test reserves rim-only/back-only. Metallic stainless PBR materials and DLAA remain enabled.
- Layouts vary class, quantity, position and yaw, with physics settling before rendering. Distinct seed namespaces per split and export audits check scene seeds, object poses, RGB hashes, angular support, lighting masks and DP clearance. Test configurations remain test-only.
- RGB, semantic PNG, instance PNG and per-view camera/light metadata are retained. Semantic IDs: 0 background, 1 robot, 2 tray, 3 scalpel, 4 scissor, 5 love_retractor, 6 kelly, 7 scalpel_type2, 8 table, 9 floor, 10 room. COCO categories are 1–5 in that instrument order. Each object body has its own instance ID even when classes repeat. COCO RLE masks preserve holes/disconnected parts; export checks exact agreement with the instance PNG and its semantic class, bounding box and area.
- Minimum labeled visible area is 8 pixels. Tiny and distant tools remain challenging; full ground truth visibility cannot be established solely by internal annotation consistency.

## Verification performed on 2026-10-11

- All three split captures completed: 2 scenes and 18 images each, **54 images at 768** total; 670 visible instance annotations. Export verified exact RLE/PNG masks and split separation.
- One complete RF-DETR-Seg Small training epoch, best-checkpoint reload and box/mask test evaluation completed on the RTX 4090. These tiny-sample metrics are plumbing checks, not a trained-model accuracy claim.
- Separate evaluator completed 18 synthetic test images and six DP camera examples (one frame per camera from one H5), including class, box and mask metrics and prediction overlays. Full DP segmentation evaluation awaits the production checkpoint.
- A controlled 384-vs-768 capture used the same seed, identical cameras and lighting. Median visible mask area grew approximately 3.91–4.04 times per view; some tiny instances crossed the 8-pixel threshold. This supports retaining more instrument detail at 768, but does not establish an optimal training resolution. See `resolution_study/comparison.png` and `report.json`.
- Fifteen targeted tests passed, including disjoint sampling, mask encoding, capture contracts, DP evaluation, and a regression showing that correct boxes can still have incorrect masks.

Existing DP ground truth has documented missing/partial instrument labels. Its evaluation deliberately uses the supplied labels unchanged, so apparent failures can include annotation errors. See Trial 01's `evaluation_dp_cameras/label_quality.json` and `label_quality_example.png`; do not interpret DP percentages as audited real-world accuracy.

## Reproducibility / fresh setup

The protocol JSON and reservation are immutable once a trial starts. Raw scene metadata stores actual cameras, light settings, poses and labels. Training stores dataset-manifest SHA and its settings. Keep raw captures: exported image paths are symlinks into them.

On this machine `start-segmentation6.sh` forwards to `versions/6.0/scripts/launchers/trial02.sh`. From another checkout use `./scripts/launchers/trial02.sh` directly with the documented Isaac Sim 6 runtime and RF-DETR environment. Before a fresh trial, run `prepare` to scan DP camera calibration and freeze protocols; `--trial-root /new/path` selects a separate trial directory. `smoke --headless` repeats the small end-to-end workflow (completed outputs are reused). Current implementation uses installed RF-DETR 1.11.0 and Isaac Sim 6.0.1. Original 5.1 source is preserved separately.

RF-DETR references: [segmentation training](https://rfdetr.roboflow.com/latest/cookbooks/fine-tune_segmentation/) and [Seg Small](https://rfdetr.roboflow.com/latest/reference/seg_small/).

## Two controlled training experiments (2026-10-12)

Production synthetic capture is complete: 9,000 train, 1,350 valid and 1,350 test images. Both experiments train from the same synthetic training set and pretrained Seg Small initialization, seed 42, 768 resolution, batch 1 × accumulation 16, 100-epoch cap and early-stopping patience 15.

From `/home/user/NSTC`:

```bash
./train-experiments6.sh both
```

This runs A, then exports DP validation/test, then runs B **sequentially**. A failure stops the queue. To run separately use `./train-experiments6.sh a`, then `./train-experiments6.sh b`; do not start them concurrently. These commands start fresh training runs. See the earlier resume instructions for continuing interrupted training.

- A: synthetic train / synthetic validation / synthetic test. W&B name `trial02A_seg-small_r768_train-SYN_val-SYN_test-SYN_seed42_TIMESTAMP`.
- B: synthetic train / original DP validation / original DP test. W&B name `trial02B_seg-small_r768_train-SYN_val-DP_test-DP_seed42_TIMESTAMP`.
- W&B project for both: `nstc-surgical-segmentation`. Checkpoint directories are `training/experiment_A_synthetic_validation_TIMESTAMP` and `training/experiment_B_dp_validation_TIMESTAMP`.

B preparation uses all frames from all six cameras in the original DP valid/test files, preserving their original split. It writes RGB PNGs and visible-instance COCO masks to `dataset_dp_validation`; train is a symlink to the exact synthetic training split. This can take substantial disk space and time: the prior DP inventory contained 136,404 validation and 68,322 test images. Validation on B will consequently be much slower than A. Completed split exports are reused; an interrupted partial split is rebuilt. Export was smoke-checked on six images per split and passed the training dataset preflight; full export and B training remain user-run.

Each training evaluates its designated test source. For a fair final comparison, also evaluate **each checkpoint on both** test sources using the earlier `evaluate` (synthetic) and `evaluate-dp` commands. DP test is never used for checkpoint selection. DP annotation limitations still apply. A and B may stop at different epochs because their validation sources differ.
