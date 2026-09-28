# P4 feedback validation — 2026-09-15

## Scope and interpretation

P3 is preserved. The five P4 object handlers retain their individual grasp geometry, asset conventions, camera resolution and H5 topic contract. Shared feedback controls stage transitions, unsuccessful-attempt rejection and saving.

**A completed collection is not a 100% success-rate guarantee.** Report attempts and failures separately from saved successes. Do not train from diagnostic failure previews or mix experimental datasets into production without auditing their metadata.

## Same-code GUI baseline

Run: `test_runs/feedback_v5_consistent_retest`.

Each handler requested one successful episode, maximum three attempts. No source was edited during this five-handler run. All five succeeded on their first attempt: **5 attempts, 5 successes, 0 failures, 10 completed H5 segments**.

| Object | Attempts | Success | Fail | Pick samples | Place samples | Sim seconds | Run wall minutes* |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalpel | 1 | 1 | 0 | 145 | 116 | 5.22 | 4.06 |
| scissor | 1 | 1 | 0 | 143 | 128 | 5.42 | 4.07 |
| love_retractor | 1 | 1 | 0 | 154 | 157 | 6.22 | 4.43 |
| kelly | 1 | 1 | 0 | 145 | 145 | 5.80 | 4.29 |
| scalpel_type2 | 1 | 1 | 0 | 147 | 131 | 5.56 | 4.17 |

*Batch-measured launch-to-exit wall time, excluding separately generated GIFs. Includes startup, settling, rendering, data collection and writing; not just robot motion. Simulation step is 0.02 s.

### Measured stage sample counts

| Stage | Scalpel | Scissor | Love | Kelly | Type2 |
|---|---:|---:|---:|---:|---:|
| OPEN_HOVER | 39 | 35 | 37 | 32 | 35 |
| LOWER_PRE | 30 | 31 | 34 | 31 | 30 |
| LOWER_GRASP | 42 | 44 | 46 | 47 | 45 |
| CLOSE | 13 | 12 | 14 | 12 | 15 |
| LIFT_CLEAR | 21 | 21 | 23 | 23 | 22 |
| MOVE_TO_TARGET | 38 | 51 | 76 | 67 | 52 |
| LOWER_PLACE | 40 | 41 | 42 | 40 | 40 |
| OPEN | 16 | 14 | 17 | 16 | 17 |
| RETREAT | 22 | 22 | 22 | 22 | 22 |

These are observed counts, not fixed budgets. Different spawn locations require different travel distances. No separate MICRO_LIFT, LOWER_EXTRA or HOLD_AFTER_CLOSE appeared in this run. CLOSE was 0.24–0.30 simulated seconds; measured EE drift during CLOSE was only 0.070–0.133 mm. Both finger joints receive the close command before lifting.

### Data and visual checks

- `validate_feedback_dataset.py` passed on all ten H5 files, including cross-file topic/shape/dtype agreement.
- Each file has 76 datasets, six RGB streams (`224x224x3`, uint8) and six semantic streams (`224x224`, uint16); image/calibration/proprio sample counts align.
- Absolute actions have shape `(T,8)`, finite values, unit quaternion and binary gripper commands. CLOSE pose commands are stationary.
- Every saved episode contains verified lift and final-placement evidence, and the aligned 0.1034 m observed/control TCP contract.
- Native-resolution front/wrist/tray end-of-stage sheets were inspected for all five objects. The objects were held during transfer and visibly left in the tray after release.
- RGB and semantic GIFs were generated from the recorded H5 images, not recreated motion. Stride 3 / 60 ms is nominal simulation-time playback. Boundary-frame retention can add a small extra pause; GIF viewer timing also varies. Scalpel RGB uses stride 1 / 20 ms.

Evidence: `review/audit_all.json`, `review/alignment_audit.log`, `review/source_sha256.json`, `review/*_stages_*.png`, `gifs/`, `gifs_semantic/` under the run folder. The original batch `SUMMARY.md` shows zero GIFs because the batch used `-SkipGif`; previews were generated separately afterwards.

## Far-side regression and final checks

The baseline was followed by a deliberate fixed-pose regression at type2 X=0.22247 m, Y=0.19999 m, yaw=256.3 degrees. The original 1 m/s transfer slipped at approximately 48 mm relative displacement; it aborted with exit 1, counted one placement failure, and saved zero H5. Folder: `test_runs/feedback_v5_far_manual`.

A shared command-speed cap of 0.6 m/s for transfers longer than 0.8 m was tested at the same type2 pose. Short transfers and CLOSE are unchanged. The repeat (`test_runs/feedback_v5_far_slow`) **passed on attempt 1**, exit 0, 367 total samples, 4.734 wall minutes. Transfer took 138 samples / 2.76 simulated seconds; CLOSE remained 15 samples / 0.30 s. Both H5 segments passed audit and visual stage review. Sixteen unit/regression tests passed after the speed change (`validation/feedback_v5_far_speed_units.log`). This change was made **after** the baseline above; do not misrepresent the baseline as a final-version stress test.

A second far-side test with love (`test_runs/feedback_v5_far_love`) exposed a separate thin-shaft grasp: the object began dropping during the first five transfer samples, before meaningful translation. It aborted with exit 1 and no H5. CLOSE TCP was approximately 10.0 mm while the object was unusually low. Love's safe target is unchanged, but its position tolerance is now 0.25 mm instead of 1 mm, avoiding an early stop on the fingertip edge. Seventeen tests pass (`validation/feedback_v5_love_contact_units.log`), including a test that the requested TCP never goes below the same safe floor.

The same love pose then **passed on attempt 1**, exit 0 (`test_runs/feedback_v5_far_love_contact`): 160 pick + 216 place samples, 4.663 wall minutes. Actual pre-close TCP reached 9.4 mm. LOWER_GRASP used 56 samples / 1.12 s and CLOSE remained 14 samples / 0.28 s. Both H5 files passed audit, and the front/wrist/tray stage sheets confirm held transfer and the object left on the tray after opening. The one-repeat comparison supports this correction, but is not a statistical guarantee over every pose.

The ordinary GUI baseline therefore remains a 5/5 smoke test; the difficult-pose regressions include two deliberately reproduced failures followed by two successful repeats after their respective fixes. Do not erase those failures or report all historical attempts as zero-fail.

### Forced-empty-grasp integration test

Final run: `test_runs/feedback_v5_empty_fault_checked`; console: `validation/feedback_v5_empty_fault_checked.log`.

The test-only harness moved the target away immediately before LIFT_CLEAR. The ordinary production feedback then rejected the attempt after **6 lift samples / 0.12 simulated seconds**: EE rose 30.7 mm, object rose -0.1 mm. It never entered MOVE_TO_TARGET. Result: **attempts=1, success=0, pick_fail=1, H5=0, process exit=1**, as expected for this negative test. Stage telemetry was present without `P4 STAGE METRICS ERROR`.

An earlier injection run (`feedback_v5_empty_fault`) also rejected the empty grasp with zero H5, but its test wrapper did not preserve the wrapped function signature, causing stage-telemetry warnings. The harness was corrected with `functools.wraps` and the clean test above was repeated; no production telemetry relaxation was introduced.

### Limits and reproducibility

These are a five-handler smoke test, two difficult-pose regressions and negative integration checks, **not a 500-episode reliability evaluation**. Natural failures may still occur at other poses; they must be rejected and counted, not converted into PASS. A larger bounded random-pose collection remains the next validation step. No simulator run is left active by this validation.

`validation/harnesses/README.md` describes the fixed-pose tests and clearly labels the deliberate fault injector as diagnostic-only. It is never imported by normal recording. `validation/inspect_completed.py` reproduces the read-only H5/image audit. The P3 migration/hash audit passed again (`validation/feedback_v5_final_migration.log`).

## Retained safeguards

Actual mesh vertices in the PhysX rigid-link frame replace stale authored USD extents. Franka instance proxies are traversed for finger geometry. Both finger joints must stabilize before lift. After approximately 20 mm EE lift, an object that does not rise aborts the attempt before transport. Slip, release readiness and stable placement remain mandatory. Failed samples are reset, not stored as successful H5. Success counters increment only after saving.

Simulator object geometry is expert/QC metadata, not an additional policy observation. Training still needs an explicit observation allowlist rather than loading every metadata field as an input.

The updated `run_all_v2_test.ps1` filename is retained for compatibility, but headings now say P4. It displays attempt/success/failure counts, failure categories and saved outputs in separate tables. PASS requires the requested success count and exact expected H5 count. GIF duration follows the requested stride rather than accidentally slowing stride-1 playback.

See `FEEDBACK_V5_GUIDE.md` for the GUI panel and automatic recording commands.
