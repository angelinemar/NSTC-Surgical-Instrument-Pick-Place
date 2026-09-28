# P4 approach correction — 2026-09-16

## Scope

Only P4 shared recorder/controller files are changed. P3, robot placement,
table/grid/tray layout, camera poses, 224-pixel crop, object assets, collision
settings, friction, semantic IDs and action schema are not changed.

## Confirmed controller defect

The installed differential IK implementation applies a body-local TCP offset
directly to a base-frame Jacobian. In this installation the raw PhysX Jacobian
is at the hand center of mass. Both conversions are required:

```
J_base = rotate_world_jacobian_to_robot_base(J_PhysX)
r_base = R_hand_in_base * (TCP_offset_in_hand - COM_offset_in_hand)
J_TCP_linear = J_base_linear - skew(r_base) * J_base_angular
```

`phase4_approach.py` supplies a local action subclass; IsaacLab itself is not
patched. The observed TCP and commanded TCP remain at local `(0,0,0.1034)`.
The raw PhysX tensor is cloned rather than modified in place.

Live validation against finite differences of the installed official Franka
URDF gave a maximum component difference of **7.05e-7** at the reset pose.
The same FK matched the live TCP to below 1 micrometer. Unit tests also cover
arbitrary hand rotations and the COM shift. This verifies kinematics, not
collision-free reachability or grasp success for all poses.

## Motion and telemetry changes

- All expert episodes use bounded incremental least-squares
  IK: actual joint limits with a 0.005-rad margin and 0.20-rad maximum command
  increment. The far-target regression exposed the original unconstrained
  DLS requesting joint 7 at 3.824 rad despite its 2.897-rad hard limit, leaving
  a 55-degree placement orientation error. Bounds therefore apply throughout
  pick and place, not only to near-base approach.
- Near-base hover/descent have a 120-step **maximum**, not a fixed wait.
  Existing position/orientation gates and early stall discard remain intact.
- Near-base lift maximum is 80 steps, not the old 50: the regression held the
  instrument but timed out at 31.5-mm residual versus a 30-mm tolerance while
  still converging after the old 12-step grace. No fixed hold is added.
- Preserve the original place-compatible 180-degree-equivalent wrist choice.
  The attempted shortest-wrist override was tested and removed: it stalled
  during descent at the central near-base test point.
- Calculate grasp-floor clearance at the destination wrist orientation,
  rather than freezing clearance from the slightly tilted pregrasp pose.
  The same finger mesh points and 0.2-mm table clearance are used. Placement
  and live release clearance checks still use actual current geometry.
- A slower smooth-lift experiment did not prevent the scissor slipping and
  was reverted. No extra hold stage or changed lift profile is retained.
- Stage JSON now contains actual arm joint positions, targets, velocities,
  torques, joint limits and nearest-limit margin when available.
- Near-half-turn tray transfers compare both rotation directions against a
  local joint-limit lookahead. The final orientation is identical, including
  instrument tip direction. This is not a global collision planner; physical
  gates still reject stalled motion, slip or bad placement.
- Failed placement does not flip an already-working manual grasp branch.
  Only failed hover/descent tries the equivalent alternate wrist orientation.
- LOWER_PLACE now requires 2.5-mm TCP residual and 2-degree orientation error
  before release (previously 3.5 mm/3 degrees). The final tray gate is still
  8 mm/8 degrees. This reserves room for physical settling without accepting
  off-center demonstrations or raising the final tolerance.
  The finger clearance margin is 4.5 mm: subtracting the 2.5-mm pose budget
  leaves 2 mm, with the independent OPEN gate still requiring 1.5 mm. This
  also reduces unnecessary release height. A 1-mm approach gate was tested
  and reverted because measured contact/control residual stalled at 2 mm.
- New H5 metadata identifies `approach_controller`, `near_base_approach`, and
  `tcp_jacobian_contract`. The run manifest hashes the new controller module.

## Acceptance rules remain strict

No physics teleport or force-success is introduced. A stage completion is
not an episode success. Close requires the existing two-finger/EE gate; lift,
continued object following, placement geometry and RGB quality must pass.
Failed trials save diagnostic previews but no successful policy episode H5.
Grid coverage advances only after a saved success; no cell is silently skipped.

## Validation status

CPU: **45 recorder/geometry/coverage/controller tests passed**, plus
**14 panel tests passed** using the Tk-capable Python runtime. Isaac Python
does not include Tk; UI tests must not be run in that runtime.

Recorder tests:
`C:\IsaacLab\_isaac_sim\python.bat -m unittest test_near_base_approach test_phase4_fsm test_grid_coverage test_grasp_evidence test_tray_slots test_camera_names`

Panel tests:
`python -m unittest test_panel_tabs test_panel_output test_panel_dashboard test_panel_retry.PanelTests`

Physical regression history (one attempt per configuration, GUI rendering):

| Case | Observed result |
| --- | --- |
| Near-base v6, shortest wrist branch | Hover passed; descent stalled 86 mm away |
| Near-base v7, place-compatible branch | Hover/descent passed; scissor lifted 21 mm then slipped |
| Near-base v8, slower smooth lift | Still slipped; smooth-lift experiment reverted |
| Far-base v9, unbounded DLS | Pick verified; placement yaw stuck at joint-7 limit; failed and discarded |
| Near-base v10, corrected grasp floor | Grasp held; lift still converging at 31.5 mm when old 50+12 step cap expired |
| Near-base v11, 80-step lift maximum | Pick verified, lift completed in 66 steps; transfer reached joint-1/joint-7 bounds and was discarded |
| Near-base v12, alternative wrist arc | Pick, transfer, lower and release passed; final center error 8.086 mm exceeded the unchanged 8-mm gate, so discarded |
| Near-base v13, 1-mm release pose gate | Rejected at 2.03-mm steady residual; this experimental gate was replaced, not kept |
| Near-base v14, retained final configuration | **PASS: 1 attempt, 1 saved success, 0 failures**; both H5 files independently audited PASS |

No failed case above produced a successful training episode H5.

### Successful physical regression

Output: `test_runs/bounded_scissor_near_v14`.
Fixed target scissor: requested table XY `(0.22,-0.1924)`, yaw 0 degrees,
with all four other instruments preloaded in their tray slots. This is a
single targeted regression, not a full-grid or random-yaw success-rate study.

| Segment/stage | Recorded steps |
| --- | ---: |
| Pick: OPEN_HOVER | 62 |
| Pick: LOWER_PRE | 46 |
| Pick: LOWER_GRASP | 58 |
| Pick: CLOSE | 12 |
| Pick: LIFT_CLEAR | 66 |
| **Pick total** | **244** |
| Place: MOVE_TO_TARGET | 156 |
| Place: LOWER_PLACE | 52 |
| Place: OPEN | 15 |
| Place: RETREAT | 22 |
| **Place total** | **245** |

Control interval 0.02 s; total 489 steps = 9.78 s of recorded simulation.
Total wall time was 7.235 minutes including setup, capture and saving.
Each H5 contains 76 dataset topics; action shape `(T,8)`, six valid RGB streams
`(T,224,224,3)` uint8 and six semantic streams `(T,224,224)` uint16. All
recorded RGB frames passed the spatial-detail/blank-frame check, sample counts
matched, semantic IDs were within 0..7, and quaternion actions were normalized.
The cross-segment topic/shape/dtype schema comparison passed.

Measured EE drift during CLOSE: 0.075 mm. Lift and placement evidence both
verified. Final target center error: 7.132 mm (limit 8 mm); axis error:
4.032 degrees (limit 8 degrees); footprint inside assigned slot; settled.
All four preloaded tray objects also passed final geometry checks.

Saved controller metadata: `tcp_frame_correct_bounded_v4` and
`world_COM_to_base_observed_TCP_v1`.

**Remaining validation:** other instruments, other yaw angles, and every grid
cell still require a physical run matrix. The shared fixes are installed for
all five backends, but this one passing regression is not an "always pass"
guarantee or approval for an unattended 500-episode production run.

See the accompanying dated test logs/results. Exploratory runs under
`test_runs/near_base_*` are diagnostic data, not a production training dataset.
Do not interpret passing CPU tests or one physical episode as an all-grid,
all-yaw success guarantee.

The isolated near-base harness bypasses ONLY the conservative manual-mode
radius rejection for a fixed regression pose. Production manual validation
is not bypassed. Grid-cycle mode already allows that central cell.
The harness limits each diagnostic configuration to one attempt. This does
not change the production panel's retry-until-saved-goal/coverage behavior.

## Inference compatibility

Use the P4 environment configuration to obtain the corrected action class.
If reproducing the bounded expert controller at inference, reuse
`phase4_approach.configure(env, True)` before controlling the robot;
loading only legacy P3 IK would not reproduce this controller. The controller
uses robot proprioception, Jacobians and commanded TCP goals, not object
ground-truth positions. Ground-truth grasp evidence remains expert QC only.

## Restart

An already-running Isaac process retains its imported code. Stop the run and
launch a fresh recorder after updating. Restart the panel if needed:

```powershell
Set-Location 'C:\IsaacLab\scripts\custom\i4h_project\p4'
python .\control_panel.py
```

## Continuation: cross-instrument near-base checks, 2026-09-16

The retained v14 controller was tested with the other four instruments at
requested XY `(0.22,-0.1924)`, yaw 0, four other objects preloaded in the tray,
one attempt per instrument, and a fresh Isaac process per case. **All four
new attempts failed and were discarded; each output contains zero policy H5
files.** No controller/gate/layout/camera or P3 change was made during these
comparisons. This materially limits the earlier scissor-only result.

| Instrument | Physical result | Last failed stage | TCP residual | Angle residual | Wall minutes |
| --- | --- | --- | ---: | ---: | ---: |
| scalpel | Failed pick | LOWER_PRE | 77.512 mm | 8.067 deg | 2.888 |
| love_retractor | Pick/lift passed, failed transfer | MOVE_TO_TARGET | 187.615 mm | 14.919 deg | 5.734 |
| kelly | Failed pick | LOWER_PRE | 42.305 mm | 9.050 deg | 2.858 |
| scalpel_type2 | Pick/lift passed, failed transfer | MOVE_TO_TARGET | 175.403 mm | 22.068 deg | 5.133 |

These errors are stage TCP errors, not final object-placement errors. The
nearby joint limits and recorded torques are diagnostic evidence, not proof
of a specific collision. Do not solve these failures by relaxing discard,
placement tolerances or changing layout. The full physical grid/yaw matrix
is still outstanding; production 500 is not approved by these results.

Outputs: `test_runs/bounded_<instrument>_near_followup_v14`.
Logs: `validation/approach_fix_20260916_v14/<instrument>_followup.log`.
Machine-readable results: `validation/approach_matrix/near_base_followup_results.json`.
The scissor row in that JSON is historical v14 data, not a new physical run.

### Rechecked artifacts and tooling

- The 45 recorder/controller tests and 14 panel tests passed again. Initial
  sandbox attempts encountered Windows temporary-directory permissions;
  reruns outside that restriction passed.
- Both historical scissor H5 files passed a fresh independent audit; report:
  `validation/approach_fix_20260916_v14/scissor_followup_audit.json`.
- RGB GIF created and inspected:
  `test_runs/bounded_scissor_near_v14/episode_000000_six_cameras.gif`.
  All 165 frames decode, 672x532 pixels; selected pick/transfer/release/retreat
  frames are in the adjacent `.contact_sheet.jpg`.
- `make_all_camera_gif.py` now displays the wrist sensor once when both its
  canonical and legacy dataset names exist, reports the actual camera count,
  and sizes the layout to the available cameras. H5 data is unchanged.
- `validation/approach_matrix/run_case.py` supports an explicit instrument,
  cell center and yaw, with one attempt and strict physical gates. Its
  diagnostic `--alternate-wrist` option tests the existing retry branch on
  the first attempt without alternating scalpel roll mode.
- All 200 combinations of five instruments, ten cell centers, and yaw
  0/90/180/270 passed offline footprint/spacing prechecks. These are not
  physical successes or production coverage. See the diagnostic README.

### Alternate scalpel wrist and transfer budget follow-up

The v14 alternate-wrist diagnostic at cell 4/yaw 0 passed approach, grasp and
lift, unlike the baseline branch. It then timed out at MOVE_TO_TARGET after
84 steps: 165.470-mm TCP residual, 12.001-degree angle residual. It was still
converging (182.4 mm at step 80), with the nearest joint limit 1.102 rad away.
The failed attempt saved no policy H5. Output:
`test_runs/matrix_scalpel_c4_y0_alternate_v14`.

After these baseline tests, `phase4_feedback.py` was changed so that
MOVE_TO_TARGET following a near-base approach has a maximum budget of at
least 160 steps. Longer existing budgets are preserved. Interpolation,
velocity limits, pose tolerances, stall/slip discard and early completion
are unchanged. This is a maximum, not a fixed wait. Ordinary non-near-base
transfers retain their previous budgets. Scissor v14's 183-step maximum was
already above 160, so this change does not alter that budget.

A CPU motion regression verifies that a progressing speed-limited transfer
finishes in 143 steps, the same non-near-base case still times out at the
previous cap, a stalled case still fails early, and a fast case finishes in
55 steps. The complete recorder/controller suite now passes **46 tests**.
The previously completed 14 panel tests are unaffected by this motion edit.

Physical validation of this budget change is recorded under
`test_runs/matrix_scalpel_c4_y0_alternate_v15`, with log
`validation/approach_matrix/scalpel_c4_y0_alternate_v15.log`.
The fixed alternate branch is diagnostic only; production wrist selection
has not been changed by this transfer-budget fix.

**v15 physical result (finished 2026-09-17 00:11 local):** transfer now PASSED
in 114 steps (45.816-mm TCP residual, 4.973-degree angle, within the transfer
gate). All five preceding pick stages had exactly the same step counts and
final position residuals as the alternate-wrist v14 comparison. The episode
then FAILED at LOWER_PLACE after 78 steps: 2.925-mm position residual and
0.172-degree angle residual, versus the unchanged 2.5-mm position gate. It
was discarded before release; zero policy H5 files. Total wall time 3.766
minutes. The budget fix is physically validated for this transfer, **not for
a complete successful scalpel episode**. Do not loosen the release gate to
convert this failure into a success.

Across this continuation there were six new physical attempts (four baseline
instruments plus two alternate-wrist scalpel comparisons), all discarded,
with no new successful training episodes. The only complete success listed
remains historical scissor v14. Next work must resolve approach-branch
selection and remaining transfer/release failures, then execute the physical
all-grid/yaw matrix. Offline prechecks are not a substitute for that matrix.

## Full-grid benchmark and progressing-timeout correction, 2026-09-17

The initial full-grid benchmark exposed a different timeout at scalpel cell 0,
yaw 0: OPEN_HOVER hit its 50-step nominal cap at 134.996 mm and 37.118 degrees.
Its last 12 samples improved by 59.390 mm and 9.905 degrees. The old grace
condition rejected it solely because it was not already close to the goal.
The initial cohort was paused after two cases (scalpel failed; scissor passed
all physical gates and independent H5 audit). Those results remain under
`test_runs/benchmark_all_grid_20260917` and are not mixed with the new cohort.

`motion_still_converging` now compares net position/orientation error scaled
by their existing tolerances. After a nominal stage budget, a progressing
stage may use up to 120 extra steps. Stall/slip/contact/release checks and
early completion remain unchanged, and the hard cap prevents indefinite
waiting. Stage JSON logs include the last 12 position/angular errors,
improvement, actual tolerances, nominal/hard caps, and actual EE position.

The re-run of scalpel cell 0 passed hover, grasp, lift and transfer. It still
failed LOWER_PLACE at approximately 2.8 mm against the unchanged 2.5-mm gate;
the failed episode saved no policy H5. This confirms that the earlier hover
failure was timing-related without turning the remaining placement failure
into an accepted episode. The complete CPU recorder suite passes 47 tests,
including progressing-far, stalled, fast-success, divergent and hard-cap
cases. Production wrist selection remains unchanged.

New H5 metadata records `motion_timing_contract=progress_extension_120_v1`,
the actual `control_dt_s`, pre-action observation alignment, and the
stage-local meaning of `step_ids`. Stale legacy distractor descriptions are
overridden with all four actual non-target class names. Dataset topics,
camera geometry, semantic IDs and action dimensions are unchanged.

The active benchmark is `test_runs/benchmark_all_grid_20260917_progress_v2`:
five instruments, ten cell centers, yaw 0/90/180/270, first attempt, four
preloaded tray objects. Source fingerprints are frozen across its 200 cases.
Its summary explicitly separates PASS/FAIL/ERROR and untested cases. A PASS
requires a successful pair and H5 audit. This is a sampled matrix, not proof
for every continuous pose/jitter/roll/occupancy. `TRAINING_AND_GRID_AUDIT.md`
is refreshed by the watcher as the benchmark proceeds. Consult its live
summary before claiming completion or all-grid success.
