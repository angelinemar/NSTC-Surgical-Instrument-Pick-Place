# Five-instrument DP comparison

## Models

The **individual** approach has ten trained checkpoints: one Pick and one
Place policy for each of scalpel, scissor, Love retractor, Kelly, and type 2
scalpel. The **joint** approach has two newly trained checkpoints in total:
one Pick and one Place. Each joint policy receives the target class as a
five-element one-hot condition. The separate Pick/Place models are necessary
because their start states and gripper events differ; all five instruments
still contribute demonstrations to each joint model.

Both approaches predict target-relative TCP motion and a separate binary
gripper event. The policy uses the last two target-relative robot states,
not raw camera pixels. During Pick evaluation, RF-DETR-Seg B estimates and
locks the target pose from a live front RGB-D capture. Place uses the
calibrated assigned tray slot. This is a **local skill** test, not an entire
home-to-tray transfer or a six-camera fusion test.

## Frozen evaluation

The benchmark schedules 100 Pick and 100 Place trials for **each** approach:
20 trials per instrument and skill. Each class contributes ten held-out test
episodes and two predeclared variations/seeds per episode. The same snapshot,
variation, seed, 300-step timeout, and physical success gate are used for
individual and joint policies. In total there are 400 Isaac Sim rollouts.
No expert trajectory, training template, or ground-truth object pose is sent
to the policy during the Pick trial. The simulator uses ground truth only to
score the result. The five-class selector is supplied as a known task command.

Every scheduled attempt remains in the denominator. A detector failure after
a valid reset is a pipeline failure. Invalid technical resets are counted
separately. Success is a stable 10 cm lift for Pick and the existing strict
slot, settled-object, open-finger, and clear-TCP gate for Place.

`technical_invalid` checks whether the recorded robot start was restored
within 10 mm, whether the Pick target drifted no more than 10 mm during
warm-up (5 mm for a held Place object), and the existing tray-state checks.
It does **not** certify that every valid spawn is easy to see, collision-free,
or reachable. A difficult but successfully restored spawn remains in the
100-trial denominator; so does a later RF-DETR pose failure or policy timeout.
The live status shows these outcomes separately. Any future exclusion rule
for impossible scenes must be fixed before a new matched rerun, rather than
chosen after observing which models failed.

**Interpretation limits:** the 20 trials/class share ten underlying scenes,
so 100 trials are not 100 independent scenes. The pose selection threshold
for non-Scissor classes was developed after inspecting two type 2 scalpel
test scenes. Type 2 scalpel results in this comparison are therefore a
development-set estimate, not a blind test. The benchmark does not establish
clinical safety or whole-cycle success.

## Files and commands

- Joint Pick: `datasets/DP_training/runs/target_relative_dp_joint_pick_v1_20261103/checkpoint_best.pt`
- Joint Place: `datasets/DP_training/runs/target_relative_dp_joint_place_v1_20261103/checkpoint_best.pt`
- Plan and results: `datasets/DP_evaluation/target_relative_individual_vs_joint_100_20261103/`
- Training: `bash experiments/dp_target_relative/train_joint_two_20261103.sh`
- Resume benchmark with two independent Isaac Sim workers after the four-worker run stalled:
  `.venv-dp/bin/python experiments/dp_target_relative/benchmark_joint_vs_individual_parallel.py --limit 100 --workers 2`
- See progress: `python3 experiments/dp_target_relative/status_joint_vs_individual.py`
- Live terminal dashboard (refreshes every two seconds):
  `cd /home/user/NSTC && ./dp-joint-live.sh`

The live dashboard shows total progress, the four approach/skill groups, all
five instrument classes, active trial steps, and recent success/failure
reasons. It also shows PAUSED when the saved pause note is newer than the last
result. Press Ctrl+C to close the dashboard; this does not stop evaluation.

The runner freezes checkpoint hashes in `plan.json`. It refuses a mixed run
if a checkpoint changes. Trial logs, `result.json`, `actions.jsonl`, and GIFs
are saved under each approach and skill. Do not call partial counts a final
success rate.

## Failure audit

Each completed trial is classified from its saved result and executed actions.
`failure_audit.json` retains the raw outcome, evidence, and every unmet final
Place gate; `failure_audit.csv` is a compact spreadsheet; `FAILURE_RESULTS.md`
summarizes the mutually exclusive primary categories. Separate Pick and Place
failure charts are written to `competition_report/figures/` automatically
after all 400 trials finish and included in the refreshed report. The labels
describe measured endpoints (for example, CLOSE was commanded but the tool
did not rise 35 mm), **not proven root causes**.

While the benchmark is running, generate a clearly marked preview with
`.venv-dp/bin/python experiments/dp_target_relative/build_joint_failure_report.py --allow-partial`.
It writes only to `preliminary_failure_audit/`; the final report does not use
this partial preview.
