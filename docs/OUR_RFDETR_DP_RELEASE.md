# RF-DETR and Diffusion Policy work

This branch contains our RF-DETR-Seg detection and target-relative Diffusion Policy work.

## What is included

- RF-DETR-Seg A and B checkpoints under `models/object_detection/` (Git LFS), the live detection bridge, data preparation, and native-768 evaluation code.
- Twelve best target-relative DP checkpoints under `models/dp/target_relative/` (Git LFS): Pick and Place specialists for five instruments plus one five-class joint Pick and one five-class joint Place policy. Each run includes its training configuration, result, and metrics CSV.
- `experiments/dp_target_relative/` for training, live RGB-D target estimation, held-out Pick/Place trials, and the individual-versus-joint benchmark.
- `evaluation/` and the Isaac Sim 6 training/recorder code needed by those experiments.
- Saved result summaries and eight matched A/B validation and test metric files in `docs/results/`. The DP checkpoint inventory is in `docs/DP_CHECKPOINT_REGISTRY.md`.

## Workspace setup

The large raw H5 demonstrations and Isaac Sim installation are **not in GitHub**. To run the benchmarks, place this repository at `NSTC/versions/6.0` or link it there, and provide the original `NSTC/datasets/DP datasets/NSTC` tree. Set `NSTC_WORKSPACE_ROOT` if the workspace directory is not discovered automatically. The existing Python environments and Isaac Sim 6 setup remain machine-specific.

The published DP checkpoints can be linked into the expected workspace run paths without overwriting local files:

```bash
python3 scripts/launchers/install_dp_model_links.py --workspace /home/user/NSTC
```

Do not rerun training merely to use the released checkpoints. Check Git LFS after cloning with `git lfs pull`; pointer files alone cannot run inference. For this workstation, the original H5 data and checkpoints already exist in `/home/user/NSTC/datasets`, so linking is optional.

## Results and boundaries

The matched RF-DETR benchmark uses the same 1,350-image test subset per model and domain at 768-pixel inference. On DP-camera test images, A scores 78.88% box and 25.23% mask mAP50:95; B scores 90.10% box and 53.72% mask mAP50:95. The eight metric JSON files and comparison provenance are in `docs/results/rfdetr_768/`; raw test images are not copied into this Git branch.

The trained Scissor Pick policy achieved 92/100 stable lifts in each of two local placement protocols. In the matched five-class local-skill benchmark, individual policies achieved 61/100 Pick and 95/100 Place, versus 42/100 Pick and 89/100 Place for joint policies. Place begins with a held instrument. These are simulated local-skill outcomes, not end-to-end transfer or clinical safety results. See `docs/results/` for the saved summaries.

The live Scissor Pick uses the front RGB-D camera, RF-DETR B's mask, depth and calibration to estimate an initial target pose. The target is locked for a trial; the learned DP predicts the local TCP motion and gripper decisions. The simulator supplies the trial reset and success gate. Other instruments' pose-estimation performance has not been established by the Scissor result.

The scripts preserve the original workspace data paths for reproducibility. A fresh machine needs the original datasets and compatible Isaac Sim/Python environments before running the experiments.
