# P4 control panel: custom setups, retries and grid coverage

Run with ordinary Python (Tkinter), NOT Isaac's Python:

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\RUNME.ps1 -Mode panel
```

Close an old control panel and simulator before launching the updated version.
P3 files, camera sizes, instrument meshes and fixed tray-slot IDs are unchanged.

## Choose the collection mode

| Collection | Count field | Target spawn | Completion |
|---|---|---|---|
| `single` + `manual` | Successful episodes | Your selected X/Y/yaw, restored on every retry | Requested saved successes |
| `single` + `auto` | Successful episodes | Random bounded cell, conservative near-base exclusion | Requested saved successes |
| `grid_cycles` | Grid rounds | Scheduled cell, random bounded offset and yaw | Every cell succeeds once per round |

The current layout has 2 world-X columns x 5 world-Y rows = **10 cells**,
each 0.20 x 0.20 m. Three grid rounds request **30 successful episodes**.
With Save skill=`both`, that means 30 pick H5 + 30 place H5, not 60 attempts.

The cell order is row-major IDs 0..9. A failed attempt does NOT advance the
cell or round. Distractor assignments, target jitter/yaw and random tray
occupancy can change on a retry in automatic mode. Instruments remain inside
the table/cell footprint safety bounds; distractors do not share target cells.

Grid cycles do not silently exclude difficult near-base cells. The amber circle
is a conservative 0.32 m warning, not a full inverse-kinematics reachability
certificate. Some cells/rotations may remain difficult. Stop and inspect the
reason rather than interpreting a partial run as complete coverage.

## Buttons and previews

- Select an instrument by clicking its footprint or using the dropdown.
  `Use as target` selects the recorder class. Drag or edit X/Y/yaw in manual mode.
- `Launch + record`: starts Isaac, applies setup, settles, then records automatically.
- `Prepare preview only`: starts without recording; `Record prepared scene`
  then enables recording and automatic retries.
- `Discard this attempt`: rejects the current attempt; automatic recording retries.
- `Stop run`: requests a normal stop at a recorder check. An unfinished goal is
  reported as incomplete, not successful. No forced process kill is used.
- The right-hand log shows attempts, stage sim/wall times, failure reasons,
  resets, completed saves and process exit. `console.log` retains the full log.
- During a run the diagram displays the requested spawn setup from the backend,
  NOT a live physics animation. It distinguishes table and tray occupancy.
- Tray counts remain 0/1 per class, with target absent from the initial tray.
  `random` samples 0..4 other types; `manual` preserves your selection on retry.

There is no Max Attempts setting in the panel: it passes `--max-attempts 0`.
Fatal configuration, file I/O or simulator errors still stop with an error;
they are not hidden inside an endless retry loop.

## What was fixed

1. Removed the legacy second `force_extra_distractors` spawn pass. All five
   bodies are already positioned by the shared force/reset path. Repeating
   the old pass moved tray bodies back onto the table with incompatible poses.
2. Reset robot joints/actions and all object velocities before settling.
   Repeated attempts restore settled PhysX-link templates, then tray occupancy.
   Requested custom positions are not replaced by automatic sampled positions.
3. Keep the initial settled scalpel pose mode for subsequent resets of the run;
   this avoids mixing a BROAD_FLAT template with an EDGE_SIDE request. Yaw
   still randomizes in auto mode. This is an intentional sampling restriction.
4. Do not flip the wrist branch on every even attempt. For a custom pose, a
   failed motion tries the equivalent 180-degree branch; a working branch is
   retained. Automatic mode uses the place-compatible branch for each new pose.
5. Print the failure and DISCARD before resetting. Failed trajectories never
   become policy H5; diagnostic last-frame PNGs can remain in failure_previews.
6. Responsive dark panel, centered world-coordinate drawing and bounded log view.

## Coverage evidence and training limits

Each saved segment in grid mode includes `coverage_contract`,
`coverage_cell_id`, `coverage_cycle`, requested cell assignments and settled
spawn metadata. On normal completion or handled interruption,
`coverage_report.json` independently reads saved H5 metadata and lists counts
per cell. A successful grid run requires complete, matching pick/place segments
(or the requested single skill), the expected cell/round, and no episode holes.

Coverage means sampled locations were represented in successful demonstrations.
It does **not** establish that 30 episodes are sufficient to train a reliable
policy, nor prove performance on unseen yaw, clutter or lighting. Training and
held-out closed-loop evaluation have not been performed by this change.

Keep using GUI recording for now: an earlier headless run produced flat camera
buffers. Every recorded RGB frame is checked before saving; the headless render
problem itself is not claimed fixed.

## Validation

- CPU regression suite: shared bindings for all five recorders, grasp/lift/place
  rejection, two-finger close gate, save proof, cell bounds, coverage and logs.
- GUI construction tests: dark theme, coordinate roundtrip/centering at multiple
  window sizes, and actual Launch command for 3 rounds => 30 saved episodes,
  unlimited attempts. These are programmatic tests, not screenshot-based QA.
- `test_runs/panel_retry_validation_v2`: Love custom pose, four preloaded tray
  instruments, 3 attempts => 2 saved successes + 1 placement failure. No spawn
  failures. The failure exposed the old even-attempt wrist flip, since fixed.
- `test_runs/panel_retry_validation_v3`: final Love custom-pose regression,
  four other instruments preloaded in the tray. **2 attempts, 2 saved successes,
  0 spawn/pick/place failures**, exit 0, total 8.395 minutes. Both episodes retain
  the same requested custom position across reset. No injected discard.
  All four H5 segments passed the all-frame RGB/data audit: 76 dataset topics
  per segment, six RGB and six semantic streams, 224x224 recording images.
  Pick = 155 frames and place = 158 frames in each episode. CLOSE = 14 control
  steps / 0.28 simulated seconds; measured EE drift during close <0.1 mm.
  This proves this tested pose, not universal reliability for random poses.
- All cells/all five instruments have NOT yet passed a complete physical grid
  round with the new scheduler. Do not label them universally validated.

Backups of files replaced in this update: `archive/pre_grid_coverage_20260916`.
