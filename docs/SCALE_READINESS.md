# P4 data consistency and scale readiness

Historical pre-fix snapshot. See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the new storage,
sensor-only training boundary, organized source tree and active v6 benchmark.

Snapshot: 2026-09-17 00:52 local. This is a readiness audit, not production authorization.

## Decision

- Current successful recordings: consistent in the audited snapshot.
- Small training pipeline experiments: reasonable after loader/config conversion and curated splitting.
- Unattended large-scale recording and claims of reliable all-grid operation: not ready.
- Final model training/generalization: not demonstrated.

## New data actually inspected

Evidence: `validation/consistency_20260917_005128/summary.json` and eight per-file reports.
The audit freezes completed benchmark results before opening files; it does not inspect files being written.

All eight H5 files from four successful pick/place pairs passed the full content audit.
They have one identical 76-topic schema, matching pair metadata, target object, skill pairing,
and identical grasp/place evidence strings within each pair. All 13 inspected contract fields
have one value across the cohort, including control_dt_s=0.02, robot-base absolute 8D actions,
quaternion wxyz, semantic mapping, controller version, pre-step observation alignment,
and stage-local step IDs. RGB/depth/semantic contents, aliases, calibration, temporal lengths,
finite values and gripper commands are checked by `audit_training_topics.inspect`.
The independent benchmark validator also audited these successful pairs.

This validates the saved data contract, not sensor-to-action latency measured independently
of the recorder or a working training/inference deployment.

The two completed failed cases have zero policy H5. No completed case in this snapshot has
an orphan pair or a topic audit failure.

## Physical readiness blocker

Benchmark snapshot: 6/200 complete, 4 PASS and 2 FAIL. Both failures are scalpel,
cell 0 and cell 1 at yaw 0, at LOWER_PLACE. Residual errors are approximately 2.8 and
2.9 mm against the unchanged 2.5 mm gate. Logs classify both as stalled outside tolerance,
not still-progressing timeouts. Failures are discarded. The prior progressing-hover timeout
fix therefore does not resolve all placement failures.

The 200-case test is still running with frozen source. It covers five instruments, ten
cell centers and four cardinal yaws with four other objects in the tray. It does not prove
continuous position, within-cell jitter, roll, occupancy or repeated-trial reliability.
Do not modify the active recorder while this version's benchmark is running.

## Historical data and training blockers

The older audit covers 115 H5: 98 pass the current topic contract; 37 complete pairs pass
both topic checks and physical-QC metadata checks. Versions are mixed. Keep them separate
from the new cohort until deliberately curated. Four new consistent pairs are not enough
to establish generalization, and the current cohort has no successful scalpel pair yet.

`C:/IsaacLab/configs_scissor_dp.json` uses state-only input with empty RGB/depth lists,
validation disabled, rollout disabled, and hdf5_cache_mode=all. It is not a validated
visual P4 training configuration. A loader/export adapter and a small end-to-end training
and rollout check are still needed. RAM use of full caching must be measured for visual data.

Use episode/session splits, keep each pick/place pair in one split, recompute normalization
from the training split, canonicalize quaternion signs, mask invalid depth, and prevent
action chunks crossing policy/episode boundaries. Place teacher-grasp NaNs are intentional;
do not include them in a finite-valued policy input or unmasked grasp loss. Simulator GT
semantic/supervision can label perception, but deployment inputs must be available from
real observations or a trained prediction model. Target object ID is task conditioning,
not a substitute for recognition.

## Persistence risk before unattended collection

Code inspection shows sequential pick and place saves, each directly opening its final
episode filename with h5py.File(..., 'w'). Example: backends/phase3_grid_split_scissor_recorder.py
at the writer near line 2480 and pair calls near line 4001. phase4_feedback.py guards success,
RGB quality and overwriting, but delegates directly to the writer. The two files are not
committed as a single transaction. A crash/disk error during saving can leave a partial file
or orphan pair. This was not observed in the completed snapshot and is distinct from
physical failure discard, which worked for both failed cases.

Before unattended scale, implement and test temporary-file validation, pair commit/manifest,
and recovery/quarantine of incomplete writes. Do this in a subsequent recorder version after
the frozen benchmark, then revalidate. Coverage must continue to count only audited complete
successes; never count files solely because a filename exists.

## Storage sizing

Four new successful pairs total 640,171,639 bytes: approximately 160 MB per pair.
At that observed mean, 500 successful paired episodes (1,000 H5 files) require approximately
80 GB / 74.5 GiB for H5 alone. If the goal is 500 per instrument, the corresponding estimate
is approximately 400 GB / 372.6 GiB. Available space at audit time was approximately
354 GB / 330 GiB. Episode lengths vary; estimates exclude ongoing benchmark outputs, GIF,
logs, training exports, cache, checkpoints and reserve capacity.

## Exit criteria for scale

1. Finish the matrix; explain/fix systematic failures and rerun affected cases plus regressions.
2. Verify repeated successes and production-like jitter/occupancy for intended collection scope.
3. Harden interrupted-save recovery without weakening physical success gates.
4. Validate the visual loader, sample batches, action convention/timing, train/validation split,
   and a small training plus rollout experiment.
5. Freeze a recorder/data version and collect a monitored pilot before expanding volume.

None of these checks guarantees an arbitrary fixed episode count is sufficient for learning.
Use held-out recognition and policy rollout metrics to decide whether more or different data is needed.
