# P4 tray placement validation - 2026-09-16

Bounded engineering validation, not a 100% success guarantee. P3 was not edited.
No failed attempt below produced a successful policy H5 episode.

## Regression and data checks

- 26 CPU regression tests PASS, including bad placement rejection, target
  exclusion, occupancy counts 0..4, and explicit pick/place failure counters.
- New all-frame spatial RGB QC rejects flat gray/black/solid-color buffers
  before H5 writing (and counts sensor failures separately under other_fail).
  Save entry points recheck it; resume requires the new RGB-QC metadata marker.
- Tkinter panel construction PASS, including five tray controls and slot preview.
- Migration audit PASS: existing backend grasp geometry/schema preserved.
- Completed H5 audits check 76 topics per segment, frame alignment, binary
  gripper actions, physical success evidence, tray metadata and camera calibration.
- RGB remains uint8 224x224; semantic remains uint16 224x224. Camera poses and
  dimensions unchanged. Six standard RGB/semantic streams plus the existing
  extra gripper image are retained.
- GIFs use recorded frames at stride 3 and 60 ms/frame (simulation dt=20 ms).
  They show simulated-time motion, not slower wall-clock rendering speed.

## Physical runs

Initial random-table-pose batch, four preloaded tray neighbors:
`test_runs/tray_slots_full_validation`.

| Object | Attempts | Saved success | Failures | Finding |
|---|---:|---:|---:|---|
| scalpel | 2 | 0 | 2 | Lip contact / transfer timeout, before final clearance correction |
| scissor | 1 | 1 | 0 | Placement and H5 audit PASS |
| love_retractor | 2 | 1 | 1 | Transfer orientation timeout; next placement/H5 PASS |
| kelly | 1 | 1 | 0 | Placement and H5 audit PASS |
| scalpel_type2 | 2 | 0 | 2 | Pick slip, then unreachable hover; never reached placement |

Historical type2 metrics classified the slip as place_fail because
`displacement` contains `place`. Its raw log identifies PICK. The P4 parser now
prioritizes explicit phase categories; historical files have not been rewritten.

Scalpel fixed-pose retests (not random-grid coverage):

- `tray_slots_scalpel_retest`: 1 attempt, 0 saved. Finger clearance 1.1 mm was
  below the 1.5 mm OPEN gate; rejected before opening.
- `tray_slots_scalpel_retest_v2`: 1 attempt, 1 saved, 0 failures after commanded
  lip margin increased from 3 to 6 mm to accommodate tracking residual. Both
  H5 segments audit PASS; RGB/semantic GIFs and review sheets generated.
- `tray_slots_type2_retest`: fixed reachable target pose, four tray neighbors,
  1 attempt, 1 saved, 0 failures. Both H5 segments audit PASS. This isolates
  placement; it does not establish reliability across all random pick poses.
- `tray_slots_empty_headless`: scissor, 1 attempt, 0 saved. Cameras and empty
  occupancy initialized correctly; transfer reached 2 mm positional residual
  but remained 51.3 degrees from required orientation and was rejected. The
  equivalent-grasp heuristic is therefore not sufficient for every wrist path.
- `tray_slots_empty_headless_scalpel`: 1 attempt, 1 saved, 0 failures using the
  same fixed scalpel pose as the GUI full-tray test. PHYSICAL SUCCESS ONLY:
  the subsequent image review found blank gray sensor frames despite correct
  topic sizes. The old sparse image audit had falsely passed this test.
  **Both H5 files are INVALID FOR TRAINING.** New all-frame audit rejects them.
  Front camera: 60/159 pick frames and 100/112 place frames are flat; other
  cameras are also affected. Files retained as diagnostics, not deleted or
  silently repaired. See its `INVALID_FOR_TRAINING.md` marker.

| Successful run/object | Pick frames | Place frames | Final center error | Final yaw error | Release gap |
|---|---:|---:|---:|---:|---:|
| full batch / scissor | 144 | 185 | 6.89 mm | 3.90 deg | 12.35 mm |
| full batch / love_retractor | 171 | 117 | 1.71 mm | 0.08 deg | 11.49 mm |
| full batch / kelly | 149 | 174 | 1.43 mm | 0.19 deg | 8.96 mm |
| scalpel retest v2 | 159 | 112 | 2.26 mm | 0.69 deg | 13.39 mm |
| type2 fixed-pose retest | 147 | 154 | 3.23 mm | 0.47 deg | 9.01 mm |

## Corrections during validation

- Enable PhysX scene queries and recorder cameras in GUI and headless modes.
- Align preloaded bodies after initial settling resolves physical link frames;
  settle and validate again before recording.
- Keep table/tray region metadata separate, excluding tray bodies from table cells.
- Use measured held-object offsets for placement, not the EE origin.
- Use a 3.5 mm EE waypoint tolerance with independent object checks: center
  <=8 mm, planar yaw <=8 deg, full slot containment, release gap 3..18 mm,
  final support within 5 mm and stable pose; other tray objects must remain valid.
- Measure lip (~13.21 mm world Z) separately from support floor (~3.19 mm).
  OPEN still requires measured finger clearance, regardless of waypoint.
- Consider final place orientation when choosing equivalent grasp yaw. This
  reduces an avoidable branch problem, not a universal IK reachability guarantee.

## Limitations

- Headless RGB rendering still intermittently produces flat buffers on this
  installation. Cause is not yet resolved. Use GUI for now; the new pre-save
  guard rejects bad images rather than claiming these recordings are valid.
- The final RGB guard was regression-tested and replay-audited against the
  completed good/bad H5 files; a fresh live collection with this last guard
  has not yet been run. Prior GUI physical results remain evidence, not a
  full post-change end-to-end production qualification.
- Randomized poses can still fail grasp/transfer. PASS means saved successes,
  not zero failed attempts. Use bounded tests before large dataset collection.
- GUI supports one physical instance per type, routed between table and tray.
  Duplicate bodies, arbitrary omission of table bodies and stacking are not
  implemented. The current target is always absent from the initial tray.
- Canonical view: left-to-right = tray +Y; down = tray +X. Existing oblique tray
  camera sees this rotated; its pose is unchanged.
- Direction uses the positive longest physical-link axis, not verified
  anatomical blade/tip versus handle annotations.
- Simulator geometry is expert QC/metadata, not a learned-policy input.
