# Current P4 status: 2026-09-17

## Current scope (supersedes historical plans below)

The user requested 50 combinations: five instruments x ten grid centers, yaw 0,
with at most two attempts per case. Do not resume the historical 200-case plans.
Current frozen cohort: `debug/test_runs/benchmark_grid_baseline_20260917_v11`.
Its `summary.json` is authoritative for live counts; `TRAINING_AND_GRID_AUDIT.md`
provides the generated human-readable matrix. Scissor cell 9 and scalpel cell 0
have passed under v11. Pending cases are not validated coverage.

Failures are acceptable only when discarded. A recovered case requires a complete,
audited successful pick/place pair; failure and retry counters remain visible.
After this cohort completes, audit and export this same version, then verify both
training pipelines. The existing sensor-only pilot proves pipeline functionality,
not final-v11 data completeness or learned-policy competence. No production-500 run.

Sections below retain the investigation history. References to older active cohorts
or broader matrices describe their status at the time, not current instructions.

An automatic completion worker now waits for this baseline before content/FK audit,
sensor-only export and both policy smoke/runtime checks. Read
`debug/validation/baseline_v11_finalization/status.json` for its current state and
per-step logs in that directory. Four finalization tests reject partial coverage,
duplicate/missing combinations, bad storage/accounting and accept discarded retries.
Do not run a duplicate export while the worker is active.

## User requirements

- Expert generation and QC may use simulator ground truth.
- Training/inference inputs must not contain world/object pose, grid, target-slot or expert-state leakage.
- Keep P3, assets, scene placement, camera geometry/resolution and semantic IDs unchanged.
- Failed physical attempts are discarded; production coverage counts only complete successes.
- Organize sources/debug/environment/training and preserve working imports/entry points.

## Completed changes and evidence

### Recorder precision

`src/recorder/phase4_approach.py` v6 uses a place-skill-only bounded position integral from measured robot TCP
error to compensate small servo residuals. It activates only near a stationary pose and resets
for moving/far/rotationally misaligned commands. The correction is capped at 4 mm and the joint
solver still respects joint limits. Object pose is not used by this correction. Actual pose,
slip, clearance and final placement gates remain unchanged.

The formerly failing scalpel cell 0/yaw 0 completed a full success in
`debug/test_runs/precision_v5_scalpel_c0`: lower-place residual 1.8 mm versus the same 2.5 mm gate,
371 recorded steps, approximately 3.41 minutes total wall time. Both policy files passed the
pre-commit validator and a commit manifest was written. A subsequent full-matrix run repeated
this case successfully after source organization. This is not all-grid proof.

The broader v5 activation caused love-retractor cell 0 to stall at LOWER_GRASP. That cohort
was stopped after 4 cases (3 PASS, 1 FAIL). In v6, pick uses the previous non-integral bounded
controller; switching skills clears accumulated correction. Live policy code selects this
controller mode from its explicit pick/place skill, not a simulator/expert-stage input.

### Interrupted-save protection

`src/recorder/phase4_storage.py` stages selected policy segments under `.pending`, audits them,
publishes them, and writes `.commits/<object>_<episode>.json` only after all selected segments
are present. V6 also flushes H5 bytes before publication and stores SHA-256 checksums in the
commit manifest; resume/export reject modified committed files. Resume and coverage checks require the manifest for this storage contract.
Interrupted uncommitted writes are moved to `.quarantine`; they never advance coverage.
Single-policy modes remain supported. The full physical scalpel run exercised the real writer.

### Sensor-only learning

`training/export_sensor_only.py` creates policy files with exactly seven observation keys:
six RGB streams and `robot_proprio`. All simulator object/debug metadata is excluded from
these inputs. The separate perception file holds class-mask labels only.
`sensor_policy.py` implements shared RGB perception and conditional DDPM action learning.
`runtime.py` accepts only sensor observations and rejects injected GT keys.

Both pick and place completed optimizer/validation/inference smoke tests. The runtime was
then exercised on held-out recorded sensor frames: finite 16x8 actions, unit quaternions,
binary gripper, and six 224x224 predicted class masks. GT injection and deployment of smoke
checkpoints were rejected. These are pipeline checks, not successful learned-policy rollouts.
The tiny smoke checkpoints have poor recognition metrics and are not deployable.

Reports:
- `debug/validation/sensor_only_pick_runtime.json`
- `debug/validation/sensor_only_place_runtime.json`
- `training/runs/pick_smoke_v1/report.json`
- `training/runs/place_smoke_v1/report.json`

### Organization and tests

69 source/document files were moved into their functional folders; old import/command paths
use compatibility entries. Debug directories were moved under `debug` with legacy junctions.
Environment JSONs are in `env` with root hard-link aliases. The current file mapping and
`scripts/check_structure.py` verify that implementations and aliases still resolve.

Verified after organization:
- 47 recorder/controller tests pass (`debug/validation/approach_matrix/v6_unit_tests.log`).
- 14 panel tests pass.
- 11 new storage/anti-leakage/integral/history tests pass, including publication interruption and checksum mismatch.
- Structure/dependency checks and recorder CLI dry runs pass.
- Read-only P3 migration checksums and seven table-geometry dependency checks pass after organization.

## Active benchmark and remaining work

The old v2 matrix was stopped for fixes after 8 cases: 5 PASS, 2 physical FAIL, 1 startup ERROR.
The two scalpel failures were stalled lower-place residuals. Love cell 1 was interrupted during
scene initialization after it stopped producing logs; it is not a measured motion failure.

The v6 cohort `debug/test_runs/benchmark_all_grid_20260917_precision_v6` stopped after
21 cases: 19 PASS and 2 FAIL, with zero H5 for either failed case. Kelly cell 3/yaw 0
timed out during settle; scalpel cell 4/yaw 0 diverged during LOWER_PRE (77.5 mm error).
This cohort must not resume after runtime changes. The intended matrix tests
5 instruments x 10 centers x 4 yaws = 200 first attempts,
four other instruments in the tray. Cases have a 20-minute process cap plus graceful stop/
termination handling so an initialization hang cannot block the matrix indefinitely.

Still required before claiming every blocker resolved:
1. Finish the full matrix; investigate any remaining failures/errors and retest fixes.
2. Validate repeated trials and intended production jitter/roll/occupancy, not only centers.
3. Collect sufficient homogeneous data; the new v6 pilot has only 12 train pairs and 5 val pairs.
4. Train/evaluate recognition and policies meaningfully, then measure actual closed-loop rollouts.
5. Resolve target-command ambiguity if multiple eligible instruments are on the table: use an
   explicit user command or image selection, never hidden simulator object ID/pose.

No production-500 run or learned-policy readiness is claimed. Do not modify the frozen
runtime sources or launch a competing Isaac process while the matrix is active.

## V6 evidence added

Scalpel cell 0/yaw 0 passed under v6. Its two H5 files independently passed the full topic
and pair/commit/checksum audit in `debug/validation/consistency_20260917_131954`.
The six-camera GIF under that case directory has 129 decoded frames; its contact sheet was
visually checked for approach, grasp/carry, tray placement and retreat. Scissor cell 0 also
passed. Later results remain authoritative in the live matrix summary; these examples do
not certify every instrument/grid or a trained policy.

## Follow-up: bounded settle continuation

`src/recorder/phase4_runtime.py` now wraps the P4 backend settle call with one continuation
of at most 300 steps, preserving the hold action and every stable-poll gate. It does not
reset or record during this continuation. Missing/nonfinite states are rejected, including
a legacy success report containing NaNs. The copied P3 settle implementation is unchanged.
Four tests in `tests/test_settle_continuation.py` verify immediate success, damping success,
permanent-instability timeout and invalid-state rejection. All 11 scale-contract tests and
the structure check also pass.

Physical regression output: `debug/test_runs/settle_continuation_v7_kelly_c3`.
The formerly failing second settle passed after 661 steps (450 + 211). The full episode
then PASSED: 370 recorded steps, both H5 independently audited, one successful attempt.
This is not a completed matrix or production authorization.

Both v6 pilot runtime checks pass: `debug/validation/sensor_only_pick_v6_runtime.json`
and `sensor_only_place_v6_runtime.json`. Each accepts only robot proprioception and six RGB
images, rejects injected object GT, emits finite normalized actions and blocks smoke
checkpoint deployment. Physical learned-policy rollout is still unverified.

The frozen v6 cohort also passed a complete content re-audit of all its successful files:
38/38 H5 and 19/19 paired episodes, one schema/controller/timing/storage contract.
Report: `debug/validation/consistency_20260917_151137/summary.json`.
Recorded data total 3,153,729,505 bytes; a linear projection for 500 pairs is about 83 GB
before exports, logs, GIFs and other overhead. Coverage remains only the tested cases.

## Automatic retry and cleanup follow-up

Scalpel cell 4/yaw 0 passed using the alternate branch in
`debug/test_runs/scalpel_c4_alternate_v7_retry`: 245 pick + 207 place steps;
both H5 have 76 topics and independently pass the validator.
The sandbox attempt without the `_retry` suffix stopped during remote-asset loading;
it is not a measured robot-motion result.

The production retry mechanism was then tested without forcing the initial branch:
`debug/test_runs/scalpel_c4_auto_retry_v7`. Attempt 1 failed LOWER_PRE and was discarded;
attempt 2 automatically switched branch and succeeded (454 recorded steps, exactly two
audited H5). Summary retains 1 failure + 1 success, `first_attempt_success=false`,
`recovered_after_failure=true`. Four accounting tests verify recovered cases, audit failure,
missing pairs and inconsistent attempt counts. No stage/pose success gate was relaxed.

The full v7 matrix was paused after scalpel cell 0 passed so obsolete migration scripts
could be deleted at the user's request. That stopped cohort is
`debug/test_runs/benchmark_all_grid_20260917_settle_v7`; do not resume it after cleanup.
The next cohort was `debug/test_runs/benchmark_all_grid_20260917_settle_v7_clean`:
200 cases, up to two automatic attempts each, separate first-attempt/recovery counters.
The live Markdown report follows this cohort. No result is inherited from previous cohorts.

Removed only `prepare_p4.py`, `scripts/tools/prepare_p4.py` and
`scripts/maintenance/organize_workspace.py`: completed migrations with no active callers.
The other 49 root wrappers still support imports, tools and test commands.
See `ROOT_FILE_AUDIT.md` and `ROOT_FILE_AUDIT.json` for evidence and retained-entry reasons.
After deletion: structure/import/CLI checks, 14 panel tests and read-only P3/geometry
migration checks passed. P3, layout, cameras, semantic IDs, assets and data were preserved.

## V8 transfer follow-up (historical)

V7 clean stopped after 59/200 cases: 51 PASS, 8 FAIL. One Windows sharing/access error
interrupted atomic summary publication after case 58; unchanged code resumed case 59,
then stopped safely for fixes. All eight failures were MOVE_TO_TARGET on yaw-zero cells
6-9 (scissor, love_retractor, scalpel_type2). The repeated residuals coincided with wrist
joint limits; they were not evidence that more timeout alone would solve the motion.

V8 adds an expert waypoint when the direct far-side-to-tray segment crosses within
0.35 m of the robot base: route through base-frame [0.45, -0.15, >=0.30] m, hold grasp
orientation on the first leg, then align for placement on the second. Each leg has a
0.6 m/s command-speed cap; existing slip, pose, release and placement gates remain.
No object/world pose is added to model observations. The trajectory metadata contract
is `base_corridor_detour_v8`; do not mix v7 and v8 exports silently.

First physical regression: `debug/test_runs/transfer_detour_v8_love_c7` PASS, one attempt,
both policy H5 audited. Transfer reached its gate in 182 steps with 1.085 rad nearest
joint margin, versus the previous failure's 0.005 rad margin. Placement and retreat pass.
Its six-camera GIF has 143 decoded frames; contact sheet visually checked.

V8 cohort: `debug/test_runs/benchmark_all_grid_20260917_detour_v8`.
It prioritizes the eight prior failures, with no inherited outcomes. Summary publication
retries transient Windows locks atomically, keeping the prior complete summary intact.
47 recorder/controller tests, 3 route tests and 6 benchmark accounting/atomic-write tests
pass. Remaining matrix cases and learned-policy performance are not yet certified.

V8 stopped after scissor cell 6 failed both attempts: the direct-path wrist-arc choice
was stale after the first detour leg, reaching joint 7's limit. Offline URDF lookahead
at the lift reproduced costs short=0.277, other=0.299; at the waypoint the costs changed
to short=0.013, other=0.0. V9 reassesses with the actual robot Jacobian/joints at that
waypoint and reserves enough interpolation time for either near-half-turn arc.
The flow regression test verifies both evaluations and reaching the final target.
V9 cohort: `debug/test_runs/benchmark_all_grid_20260917_waypoint_v9`;
contract `base_corridor_waypoint_arc_v9`. Scissor cell 6/yaw 0 PASSED on its first attempt,
including both H5 audits. It stopped safely before the final exact-half-turn fix below.

Independent FK audit of all 102 successful v7 H5 passed: maximum position discrepancy
1.45 micrometers and orientation discrepancy 0.000129 degrees. The robot-proprioception
channel agrees with robot joints/URDF, not hidden instrument state (`robot_fk_v7.json`).

The 102-file content audit also passed all 51 v7 pairs, with zero failed-case H5 and
one schema/metadata contract: `debug/validation/consistency_20260917_200421/summary.json`.

V10 closes the exact-180-degree quaternion tie: the explicitly requested other arc now
uses the opposite axis even when the quaternion dot product is exactly zero, reaching
the same final orientation. The new regression failed before the correction and passes
after it. 49 recorder/controller tests, 3 route tests and 4 settle tests pass;
the 6 atomic-publication/accounting tests also pass. Log: `approach_matrix/v10_unit_tests.log`.

V10 matrix: `debug/test_runs/benchmark_all_grid_20260917_waypoint_v10`.
Contract: `base_corridor_waypoint_arc_v10`. The user reduced scope to 50 cases:
five instruments x ten cell centers, baseline yaw 0. It prioritizes the eight
old failures, at most two attempts per case. No production coverage is advanced and
no pending case is counted as success. Do not edit runtime/harness sources while it runs.
Failures followed by successful retries are acceptable; failed data must still be discarded.
This baseline does not certify untested yaws, every within-cell position or learned-policy accuracy.

V10 stopped after 6 cases: 5 PASS, scissor cell 9 FAIL after both attempts. The 0.35 m
straight-line radius trigger missed this outer-slot transfer, so it still used the old
path. V11 consistently routes all far-side starts (base Y > 0.10 m) to tray-side goals
(base Y < -0.30 m), retaining the same waypoint, arc reassessment and physical gates.
The route regression now covers all five slot X coordinates from all four far-side centers;
it reproduced four missed cases before the fix and passes after it.
Current frozen baseline: `debug/test_runs/benchmark_grid_baseline_20260917_v11`, exactly
50 combinations, yaw 0, up to two attempts each, starting with scissor cell 9.
Contract: `far_side_waypoint_arc_v11`. Previous failed files are not exported or counted.
