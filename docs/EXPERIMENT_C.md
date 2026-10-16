# Experiment C: mixed synthetic and DP training

Experiment C learns from both sources. A and B trained only on synthetic images. C starts from pretrained RF-DETR-Seg Small, not from either finished experiment checkpoint.

## Dataset

Location: `/home/user/NSTC/datasets/Object datasets/trials/trial_03C_rfdetr_seg_syn_dp_balanced/dataset`.

| Split | Synthetic | DP selected frames | Total |
|---|---:|---:|---:|
| train | 9,000 | 9,000 | 18,000 |
| valid | 1,350 | 1,350 | 2,700 |
| test | 1,350 | 1,350 | 2,700 |

Each split has 50/50 source balance. DP samples have equal quotas across five target episode classes and six cameras. Visible annotation counts are not forced equal: other instruments, cropping, and occlusion naturally change them. DP frames are selected deterministically across episodes and temporally spread positions (midpoint, quarters, eighths, ...), avoiding domination by long nearly static episodes. Original RGB resolution is retained; training resizes to 768. Synthetic files are linked, so keep the source dataset and raw captures.

## Leakage checks

Preparation checks committed DP file checksums and original split metadata; the same pair transaction and requested layout (rounded to five decimal places) cannot cross splits. All cameras and frames from an episode inherit that episode's split. Existing pick/place pairs stay together. Synthetic scene seeds and pose layouts are rechecked. Every selected RGB is hashed after decoding, across both domains and all splits; duplicate DP candidates are excluded rather than moved into train. Synthetic cross-split duplicates cause an error. The manifest is written only when every quota and audit completes.

`source_episodes.json` records all 1,000 source DP files. `selection.json` records the selected H5, frame, camera, target, pair, and RGB hash. `manifest.json` records counts and audits. COCO IDs are unique within each split and masks remain visible instance RLE, preserving holes and disconnected parts.

These are held-out episodes/layouts, **not unseen camera identities**: six DP cameras occur in all splits. Exact duplicate and metadata checks do not prove visual dissimilarity. DP labels have known missing/partial instrument regions and are preserved unchanged; structural consistency does not prove annotation completeness. Original DP valid/test performance has already been inspected in A/B, so these reused splits are not a new blinded benchmark. Obtain a newly recorded holdout for a final blind study.

## Run

The dataset has to finish preparation and pass the check before training:

```bash
cd /home/user/NSTC
./train-experiment-c6.sh check
./train-experiment-c6.sh train
```

Do not run C concurrently with B on the same GPU. This command starts a fresh training run; it does not stop, resume or modify B.

Settings: RF-DETR-Seg Small, resolution 768, seed 42, 100-epoch cap, early stopping patience 15, physical batch 1, accumulation 16, 8 workers, learning rate 0.0001, EMA, progress bar and per-step loss. Same settings as A/B; C contains twice as many training images, so equal epochs do not mean equal optimizer steps.

W&B project: `nstc-surgical-segmentation`.
Run name: `trial03C_seg-small_r768_train-SYN50-DP50_val-MIX_test-BOTH_seed42_TIMESTAMP`.
Outputs: `trial_03C_rfdetr_seg_syn_dp_balanced/training/experiment_C_mixed_TIMESTAMP`.

The training wrapper first evaluates the mixed held-out test. Then the launcher evaluates the best checkpoint separately on `dataset/evaluation/synthetic/test` and `dataset/evaluation/dp/test` using the **same evaluator** and confidence 0.5 / IoU 0.5 for P/R/F1. Separate outputs are `evaluations/synthetic` and `evaluations/dp` in that run. Both report bbox/mask AP, P/R/F1, per-class scores and camera/distance groups. These DP results cover the fixed balanced subset (1,350 images), not all 68,322 DP test frames. For a fair A-vs-C comparison, evaluate A on these same domain subsets with the same evaluator too.

For an interrupted training, use `start-segmentation6.sh train --dataset '/path/to/C/dataset' --output '/original/C/run' --resume '/original/C/run/last.ckpt' --run-name 'original-name' --epochs 100 --batch-size 1 --grad-accum-steps 16 --num-workers 8 --seed 42`. Then run the domain evaluations separately if needed. The C launcher itself always starts a fresh run.

## Rebuild in a new destination

From the 6.0 repository with the RF-DETR Python environment and recorder import paths:

```bash
python -m training.prepare_experiment_c \
  --synthetic '/home/user/NSTC/datasets/Object datasets/trials/trial_02_rfdetr_seg_hemisphere768/dataset' \
  --dp-root '/home/user/NSTC/datasets/DP datasets/NSTC' \
  --output '/new/empty/dataset/location'
```

Preparation refuses to overwrite existing outputs, and never changes the original DP recordings or A/B datasets.
