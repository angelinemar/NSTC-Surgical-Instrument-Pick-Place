# Scissor Pick: RF-DETR B + trained Diffusion Policy

This experiment answers whether a *trained* Scissor Pick policy can lift an instrument using a target estimated from the live camera. It is separate from the earlier camera-guided trajectory-retrieval diagnostic. The latter's 45/50 result must not be reported as policy success.

## What controls the robot

1. Restore a held-out DP validation scene at the start of Pick. The recorded state is used for reset and physical scoring; recorded actions are never replayed.
2. Capture the live front RGB-D frame. RF-DETR B predicts the Scissor instance mask. Registered depth points give a target pose in the robot base frame. The pose is locked for this Pick attempt.
3. The trained target-relative Diffusion Policy receives measured 16-value robot state and the locked camera-estimated target pose. It generates target-relative TCP position and orientation actions. A separately trained head predicts OPEN/CLOSE. Replan every four control steps. No expert waypoint, saved action, or simulator object pose enters the policy.
4. Isaac Sim executes the predicted actions. The physical success gate requires a stable held Scissor and at least 10 cm lift.

The policy itself does not process raw pixels; RF-DETR B and the RGB-D pose estimator are its perception frontend. This version evaluates **Pick only**, not Place or a whole transfer. The previously trained Place checkpoint is loaded to satisfy the actor interface but is not used in a Pick-only trial.

## Data and training

The Scissor Pick corpus has 70 train, 20 validation, and 10 test episodes. Training uses only the train split for weights and normalization. Validation selects the checkpoint; the test split remains untouched by this candidate's validation. The model uses a conditional diffusion network for a 16-action motion window and a separate gripper decision head. Saved training configuration and metrics are in `../../datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/`. A successful validation loss does not establish closed-loop success.

## Additional instrument and Place training (3 November 2026)

`train_family.py` generalizes the target-relative representation to one checkpoint
per instrument and skill. Each run uses its own 70 training episodes and 20
validation episodes; its 10 test episodes are reserved for evaluation. Training
labels supply the object pose for Pick. Place uses the assigned tray-slot center
as the target frame and learns the OPEN transition separately from TCP motion.
The sensor contract at deployment is therefore different:

- Pick: a **camera-estimated** 7D target pose for the selected instrument plus
  measured robot state. The Scissor front RGB-D estimator is evaluated in the
  100-trial Scissor study. A train-only reference shape can now be built for
  the other classes, but its physical accuracy must be measured separately.
- Place: a calibrated/known tray-slot center plus measured robot state. This
  is a task goal, not a simulator object-pose oracle.

The nine additional runs are queued sequentially by
`train_remaining_20261103.sh`; the original Scissor Pick checkpoint is frozen.
Each run writes `training_config.json`, `metrics.csv`, `checkpoint_best.pt`, and
`result.json` under `../../datasets/DP_training/runs/`. The queue log is
`../../datasets/DP_training/runs/target_relative_family_20261103_queue.log`.
`audit_family_offline.py --all` reports held-out recorded-test OPEN/CLOSE
classification. This teacher-forced audit cannot prove physical Pick or Place
success. In particular, training all five Place models does **not** establish
that any can release reliably in closed loop. The exact checkpoint identities,
training status, and hashes are in
`../../competition_report/DP_CHECKPOINT_REGISTRY.md`.

Small closed-loop runtime checks are available with:

```bash
cd /home/user/NSTC
.venv-dp/bin/python experiments/dp_target_relative/benchmark_family_place_smoke.py
.venv-dp/bin/python experiments/dp_target_relative/benchmark_family_pick_smoke.py
```

Each script executes **one held-out test episode per class** in Isaac Sim and
stores a result JSON, a GIF, and a compact summary under `datasets/DP_evaluation`.
The Place goal comes from the calibrated class-designated tray slot. These
pilots check runtime wiring and reveal failures; a reliable success-rate claim
requires a larger, predeclared per-class benchmark with independent scenes.

## Reproduce validation

From `/home/user/NSTC`:

```bash
.venv-dp/bin/python experiments/dp_target_relative/benchmark_learned.py --limit 5
```

Runs are sequential. The first five distinct validation scenes completed with 5/5 physical lifts. All ten original test scenes completed with 10/10 lifts. One preset XY/yaw shift on each test scene completed with 9/10 lifts. The larger frozen evaluation of 10 held-out test scenes × 10 target shifts yielded **92/100 physical Scissor Picks**; see [100-trial report](../../datasets/DP_evaluation/scissor_trained_dp_live_100_20261103/RESULTS.md) and [100-trial GIF gallery](../../datasets/DP_evaluation/scissor_trained_dp_live_100_20261103/GIF_INDEX.html). The 100 attempts share ten base scenes, so they are not 100 independent scenes. To extend the validation to all 20 scenes, rerun with `--limit 20`. For the original test split, use `--split test --limit 10`; for the shifted test, add `--perturb`. Existing completed trials are read from their respective output folders; each run contains `result.json`, actions, camera pose evidence, and a GIF. These results do not establish a success rate for all placements or instruments.
