# P4 feedback FSM — 2026-09-15

P3 is unchanged. Camera layout, native/cropped image dimensions, H5 topic names,
semantic IDs, robot-base action frame and object-specific grasp XY are unchanged.
The five P4 backends use `phase4_feedback.py` through `phase4_fsm.py`.

**Control TCP correction:** upstream IK controlled panda_hand +107 mm while
the recorded EE used +103.4 mm. P4 now sets the IK offset to the observed EE
offset before environment construction. This eliminates the 3.6 mm hold bias.
New H5 metadata identifies `p4_feedback_version=20260915-v4` and the TCP offset.
Do not mix older action labels with v4 without reconciling their tool-point
definitions. No old dataset is automatically modified.

## Transition rules

| Stage | Completion / failure rule |
|---|---|
| Hover, lower, lift, transfer, retreat | Consecutive pose checks; timeout aborts the attempt instead of continuing. Up to 12 additional steps only when close to target and still converging. |
| Lower grasp | Contact target is clamped to finger-geometry/table clearance (+1 mm), preserving XY. Position tolerance at most 1 mm, orientation within 10 degrees; no fixed long idle after reaching. |
| Close | Binary close immediately; stationary EE command. Both finger positions stable over 5 samples, each below 35 mm, filtered velocity below 10 mm/s, EE hold error below 8 mm. Minimum 0.24 s; timeout 0.90 s. |
| Lift | Only after close gate; original object-specific physical lift check remains required. Finger closure alone is not success. |
| Transfer / lower place | Abort if object-to-EE attachment drifts more than 50 mm for 3 samples (expert validation only, not policy input). |
| Open | Binary open immediately; both fingers above 37 mm and stable. Minimum 0.16 s; timeout 0.70 s. |

All seconds above are **simulation seconds**, not real elapsed time. Raw contact
velocities are logged, but a single solver impulse is not the close gate.
No micro-lift or extra HOLD stage was added. Failed attempts are discarded by
the existing backend; detailed stage/attempt logs remain available.

Scalpel type 2 transfers longer than 0.6 m use distance-aware cubic easing
with commanded peak translation speed <= 1 m/s (rather than a fixed 16-step
linear target sweep). Short transfers and the other four objects keep their
validated profiles. This is a command-profile limit, not a measured robot
speed guarantee. The transfer remains bounded and object-follow validation
is never bypassed.

`STAGE_COMPLETE` is not episode success. `--episodes N` requests N **successful
saved episodes**, so a success at `2/3` correctly starts another attempt.

## Bounded tests

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\run_all_v2_test.ps1 -Episodes 1 -MaxAttempts 3 -Gui -GifStride 1
```

Individual:

```powershell
C:\IsaacLab\_isaac_sim\python.bat .\record.py --object scissor --episodes 1 --max-attempts 3 --out_dir .\test_scissor_feedback
```

Omit `--headless` for GUI. `MaxAttempts 0` / `--max-attempts 0` means unlimited
attempts (legacy behavior); use a finite budget when diagnosing.

Logs: `stage_metrics.jsonl` includes attempt, stage, step count, simulation/wall
time, EE error, quaternion error, finger positions/velocities and precise failure.
`run_metrics.json` contains separate spawn/pick/place counters and attempt times.
`scene_manifest.json` records source hashes and attempt budget.
`failure_previews/` contains the last recorded camera PNGs for rejected grasps
and stage failures; these are diagnostics, not failed H5 training episodes.
These PNGs use the raw sensor buffer; the saved training H5 still uses the
configured symmetric 224x224 crop.

The quality validator requires measured stage-completion proof and matching
sample counts. Its close/open minima match the feedback timing, rather than
the obsolete fixed 20-frame minimum. Resume counts the segmented H5 folders,
rejects holes/unmatched pick-place pairs and incompatible TCP versions, and
new runs refuse to overwrite existing episode files.

An equivalent-jaw 180-degree yaw candidate is experimental and opt-in via
`P4_SYMMETRIC_GRASP=1`; it does not change the object's spawn yaw or grasp point.
Do not interpret passing unit tests or one successful episode as proof of
robustness across all spawn poses. Physical validation results are recorded
separately in `validation`.
