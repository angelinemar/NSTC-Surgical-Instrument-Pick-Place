# P4 recorder feedback validation — 2026-09-15

## Scope

P4 only: `C:\IsaacLab\scripts\custom\i4h_project\p4`.
P3 sources were verified unchanged. The hospital, tables, camera poses,
224x224 recording crop, and instrument-specific XY/yaw grasp construction
were retained. Originals are in `archive/before_feedback_20260915`.

## Confirmed defects and corrections

1. The IK controlled a tool point 0.107 m from `panda_hand`, while observations
   measured a point 0.1034 m away. This 3.6 mm mismatch moved the observed EE
   even when commanding a hold. IK now uses the observed TCP offset and rotation.
2. Gripper action is binary: negative closes, nonnegative opens. The previous
   ramp spent its early CLOSE samples still commanding OPEN. CLOSE/OPEN now
   switch immediately and hold the measured EE position.
3. CLOSE completion checks both measured finger positions and their stability
   window, not just a step count or a single noisy contact velocity. Asymmetric
   contact positions are allowed. A successful finger gate is not a successful
   grasp: the physical lift check must also pass.
4. Lower-grasp Z is checked against live finger geometry above the table.
   It must reach within 1 mm; an unreachable/stalled target fails explicitly.
   No timeouts silently advance to CLOSE. No micro-lift or extra HOLD stage.
5. Transfer slip is detected from object-follow consistency by the scripted
   expert/QC path. Failures stop the attempt and do not save policy H5 data.
6. Quality checks now require measured completion proof with sample counts
   matching the actual stage. The obsolete 20-frame CLOSE minimum was removed.
7. Logs separate stage completion from episode success and include failure
   stage/category, attempt counts, simulation time, and wall time. Attempts can
   be bounded. Resume validates segmented files; overwrite is rejected.

Simulator object poses remain available to the demonstration expert and QC;
these fixes do not require feeding those privileged poses to a trained policy.

## Five-recorder GUI smoke test

Artifacts: `test_runs/feedback_v4_smoke_20260915`.
One successful saved episode requested per object, maximum two attempts.

| Object | Attempts | Saved success | Failed attempts | Pick steps | Place steps | CLOSE steps | CLOSE simulation seconds | Recorder wall minutes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalpel | 1 | 1 | 0 | 161 | 87 | 13 | 0.26 | 4.015 |
| scissor | 1 | 1 | 0 | 147 | 91 | 12 | 0.24 | 3.686 |
| love_retractor | 1 | 1 | 0 | 150 | 89 | 14 | 0.28 | 3.698 |
| kelly | 1 | 1 | 0 | 162 | 99 | 12 | 0.24 | 4.107 |
| scalpel_type2 | 2 | 1 | 1 place failure | 150 | 93 | 15 | 0.30 | 5.816 |

All five processes completed with exit 0, producing 10 H5 files, 5 RGB GIFs,
and 5 semantic GIFs. This means all requested successes were collected;
**it does not mean every attempt succeeded** (5 successes / 6 attempts).
Wall times above come from `run_metrics.json`; batch summary times additionally
include launcher/process overhead. Simulation `dt` is 0.02 s.

The type 2 failure was detected during a ~0.905 m MOVE_TO_TARGET after 11 steps:
object-to-gripper relative shift reached 0.1067 m. It produced diagnostic PNGs,
not an H5 episode. The next attempt succeeded. This motivated the additional
distance-aware long-transfer profile documented in `FEEDBACK_FIXES.md`.

## Additional type 2 GUI test after long-transfer smoothing

Artifacts: `validation/feedback_type2_distance_v4` and its adjacent `.log`.
Requested two saved episodes with maximum three attempts: **2 successes,
1 hover-orientation timeout**, exit 0. No spawn or place failures in this run.

| Saved episode | Pick steps | Place steps | Transfer span | Transfer steps | CLOSE steps | CLOSE EE drift |
|---|---:|---:|---:|---:|---:|---:|
| 000000 | 160 | 93 | ~0.42 m | 30 | 15 | 0.079 mm |
| 000001 | 156 | 142 | 0.903 m | 77 | 15 | 0.087 mm |

The long transfer used 68 interpolated command steps plus convergence checks,
and passed the object-follow check, release, and final placement check.
The rejected attempt ended at OPEN_HOVER with 21.4 mm position error but 31.0
degrees orientation error; it did not descend or save an H5 demonstration.
Total run wall time was 8.535 min: successful attempts 5.994 min, failed attempt
1.155 min, startup/save/shutdown 1.386 min. All four additional H5s passed the
same audit. RGB and semantic previews are exported alongside these episodes.

The far successful spawn had a different yaw from the earlier slipping spawn.
This validates one long transfer, not a controlled proof that speed alone
caused the earlier failure. Large-yaw reachability/convergence remains a
limitation, safely rejected by the current feedback gates.

## Data audit

`dataset_audit.json` in the batch folder lists every topic, shape, dtype and
stage count. All 10 H5s passed the same schema audit:

- 76 datasets/topics per H5; leading time dimensions aligned.
- Actions `(T,8)` float32; finite values, unit quaternions, binary gripper.
- State `(T,18)` and robot proprioception `(T,16)`.
- Six unique RGB streams `(T,224,224,3)` uint8.
- Six semantic streams `(T,224,224)` uint16; sampled canonical IDs within 0..7.
- Calibration arrays and stage names aligned with observations/actions.
- CLOSE commands start negative and have constant target translation.
- Maximum measured CLOSE EE drift was 0.091–0.120 mm in these saved episodes.
- No MICRO or HOLD_AFTER_CLOSE stage in these trajectories.
- RGB first/middle/last samples were non-flat; final RGB GIF frames for all
  five objects were inspected and show the object on the tray.

The GIF contains seven panels because gripper/wrist alias the same sensor;
there are six unique camera streams. GIF display enlargement is not the H5
training resolution. Semantic ID-range checks are not a pixel-perfect labeling
accuracy evaluation, and sampled RGB checks are not exhaustive image QA.

## Compatibility and limitations

New episodes carry `p4_feedback_version=20260915-v4` and explicit TCP metadata.
Old labels made with the mismatched TCP must not be blindly mixed with these
labels; observation shape equality does not imply the same action tool point.

Eight CPU-tensor regression tests passed, including failure propagation across
all five backends, finger gates, geometry floor, TCP alignment, transfer speed
profile, attempt budget, quality proof, and safe resume. Camera contract and
P3 migration checks passed. Reading all five new output folders returned one
completed episode each; attempting a non-resume overwrite was rejected before
simulation launch.
The separate cell-spawn check also passed 5,000 generated spawn sets; this
checks geometry bounds and sampling, not 5,000 physical pick executions.

This is a bounded smoke test, not proof of 100% success for every random pose.
Earlier development candidates failed physical tests and were revised; their
logs remain under `validation`. One earlier headless candidate was stopped by
the existing black-RGB guard, so this report does not certify headless rendering.
Use separate fresh output directories for further robustness tests.

## Run and inspect

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\scripts\launchers\run_all_v2_test.ps1 -Episodes 1 -MaxAttempts 3 -Gui -GifStride 3
ii .\test_runs\feedback_v4_smoke_20260915\gifs
```

`STAGE_COMPLETE` means only a stage passed. `--episodes N` requests N saved
successes; retries continue until that goal or the attempt limit. Final metrics
must be read alongside the batch PASS status.
