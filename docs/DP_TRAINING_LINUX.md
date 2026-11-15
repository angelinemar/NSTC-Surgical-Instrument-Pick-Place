# Multi-instrument Diffusion Policy training on this workstation

This launcher trains **two skill policies**, Pick then Place. Each policy learns all five instrument targets using six RGB cameras at 448px, two observation frames, 16-D proprioception, and a five-class requested-target command. The model predicts 16 steps of 8-D actions. A custom CNN/ConditionalUnet1D is trained jointly with an auxiliary semantic head. It does **not** consume or retrain the RF-DETR-Seg B checkpoint.

## Commands

```bash
cd /home/user/NSTC
./train-dp6.sh prepare
./train-dp6.sh check
./train-dp6.sh both
```

`prepare` exports original full demonstration episodes (not the selected object-detection image subset). It preserves the frozen source inventory's 350 train / 100 valid / 50 test pick–place pairs: 350/100/50 episodes per skill. All five instruments are included. Copying native pixels and auditing all 1,000 files takes time; progress is printed per pair. Rerun `prepare` to resume completed shards. Original recordings are unchanged. Keep the complete export folder together: its index files use relative HDF5 external links to per-pair shards.

`both` runs **Pick then Place sequentially**, so each has the GPU available. `pick` or `place` trains only that skill. Do not rerun `both` to resume a previous run: that creates new timestamped runs.

## Defaults

| Setting | Value |
|---|---|
| Batch size | 24, actual batch per optimizer step |
| Maximum epochs | 100 |
| Early stopping | 15 consecutive epochs without validation diffusion-loss improvement greater than 0.0001 |
| Precision | BF16 mixed |
| Optimizer | AdamW, learning rate 0.0001 |
| Gradient clipping | 1.0 |
| DataLoader workers | 4 |
| Validation | Original valid episodes, deterministic windows at stride 16, fixed evaluation noise |
| Checkpoint selection | Lowest validation diffusion loss, subject to min-delta |
| Logging | W&B online plus terminal and metrics.csv |

On this RTX 4090 24GB, a 448px three-step memory probe measured batch 24 at 12.02 GiB allocated / 16.33 GiB reserved. Batch 32 also ran (15.98 / 21.74 GiB), but the default is 24 for headroom. Real exported-data smoke training also passed for both Pick and Place (two training batches and one validation batch each), with 12.02 GiB allocated / 16.24 GiB reserved. Native RGB and masks were checked against original files across all six cameras. These are computation tests, not evidence of better robot success. Batch 128 was tested only at 224px.

The default export is `datasets/DP_training/all5_rgb448_v1`. Native RGB and semantic masks are copied unchanged, without cropping or upscaling. Existing 224 exports are preserved separately. New checkpoints store their resolution; live inference must call `read_live_sensors(env, image_size=runtime.image_size)`. Legacy checkpoints default to 224.

The dedicated `.venv-dp` environment has W&B installed. Existing credentials were verified; if authentication is needed later, run `./train-dp6.sh login` locally. Never paste API keys into chat.

Project: https://forge.coreweave.com/wandb/angelinemarcellinachandrawijaya/nstc-surgical-diffusion-policy

Run names distinguish skill, ALL5, rgb448, batch size, seed, and timestamp. Small verification runs use an explicit `SMOKE_` prefix and are not production models.

## Outputs and resume

Runs are saved under `/home/user/NSTC/datasets/DP_training/runs/`.

- `checkpoint_best.pt`: portable inference weights + normalization + controller/input contract.
- `best.ckpt`: best full training state.
- `last.ckpt`: latest completed epoch, including optimizer, RNG, early-stop state and W&B run ID.
- `perception_best.pt`: auxiliary semantic model, **not RF-DETR B**.
- `training_config.json`, `metrics.csv`, `result.json`: configuration, validation history and completion status.

```bash
./train-dp6.sh resume '/home/user/NSTC/datasets/DP_training/runs/EXACT_RUN_FOLDER'
```

Resume restarts at the next epoch after the last complete epoch. Interrupted partial-epoch work is repeated. Dataset manifest, skill, batch, seed, learning rate and stopping settings must match. If you overrode settings originally, supply the same flags on resume. Completed/early-stopped runs are not silently restarted. Resuming Pick alone does not automatically queue Place afterward; use `./train-dp6.sh place` when appropriate.

No test examples are used for training, normalization, checkpoint selection or early stopping. Validation denoising loss and auxiliary segmentation loss are diagnostic; they are **not robot success rates**. The exported test episodes remain for a later held-out evaluation. Physical rollout must be validated separately before deployment. Prior analysis of DP test images means this is not a fresh blind benchmark.
