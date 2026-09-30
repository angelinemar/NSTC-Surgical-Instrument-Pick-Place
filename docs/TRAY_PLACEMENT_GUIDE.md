# P4 fixed-slot placement contract

P3 is unchanged. P4 uses its existing five physical instrument bodies and grasp
handlers. Camera poses, resolution and 224x224 recording crop are unchanged.

| Existing object type ID / slot ID | Instrument |
|---|---|
| 0 | scalpel |
| 1 | scissor |
| 2 | love_retractor |
| 3 | kelly |
| 4 | scalpel_type2 |

These are spatial positions, **not a pick sequence**. Semantic pixel IDs remain
unchanged and are not interchangeable with object type/slot IDs.

## Coordinate convention

Canonical tray top view: left-to-right = tray local +Y (the 550 mm side);
top-to-bottom = tray local +X (the 220 mm side). All centers share local X=0.
There are five 104 mm lanes, with 15 mm reserved at either long end and 10 mm
reserved at either short end. This convention is displayed in the control
panel; an oblique camera can show the same slots rotated in its image.

The positive longest PhysX-link axis points down in this view. This is a fixed
per-asset geometric direction, **not a verified anatomical tip/handle label**.

## Initial occupancy and GUI

Default: choose a count uniformly from 0..4, then sample that many different
non-target classes without replacement. The target slot is ALWAYS empty.
Selected existing bodies are moved from the table to their slots after initial
asset settling, then settle again before recording. Remaining bodies stay in
their randomized table cells. No extra duplicate assets are silently added.

`.\RUNME.ps1 -Mode panel` provides `random`, `empty`, `full`, and `manual` tray
occupancy. In manual mode enter 0 or 1 per class; target must have count 0.
Prepare applies the choices before Start. Multiple same-class instances,
arbitrary omission of table bodies, and stacked targets are not implemented.

## Physical controls and success

- Query the actual collision surface at every slot, including headless mode.
- Use the held object's measured offset, rotated into the desired EE frame,
  rather than treating the EE origin as the instrument center.
- Rotate smoothly during transfer while retaining measured grasp/slip checks.
- Nominal release gap: 8 mm above tray support. Finger clearance may raise it;
  release is rejected outside 3..18 mm or if alignment is invalid.
- Both fingers must clear the measured tray lip before opening, including at
  the two end slots. The grasp's equivalent yaw branch considers the final
  aligned wrist pose. An unreachable transfer is rejected, not called success.
- Final center error <=8 mm, signed long-axis planar yaw error <=8 degrees, full geometry
  inside its lane, bottom within 5 mm of support, and stable pose required.
- Existing preloaded instruments must remain supported and aligned in their
  own lanes. Failed attempts never become policy H5 episodes.

QC reads simulator geometry as expert validation, not as a learned-policy
observation. A tight expert placement target does not itself prove the future
policy generalizes.

## Data compatibility

New feedback version: `20260916-tray-slots-v1`. Use a fresh output directory;
old center-of-tray demonstrations cannot be resumed into this contract.
Resume also requires `rgb_frame_qc=all_frames_spatial_v1`. All six camera crops
must have spatial image detail in every frame before either segment is saved.
Flat-buffer failures are reported as sensor failures, not successful motion data.
Existing action/state/image/stage topics remain. Added metadata includes
`tray_contract`, `tray_slot_id`, `tray_slot_order`, `tray_slot_target`,
`initial_tray_occupancy`, `initial_tray_geometry`, and `tray_support_z_m`.
The occupancy vector follows slot IDs 0..4. `expert_grasp_evidence` includes
the target's final center/angle/containment measurements and initial occupancy.

## Commands (PowerShell)

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\RUNME.ps1 -Mode panel
```

One object, GUI, place segment only:

```powershell
C:\IsaacLab\_isaac_sim\python.bat .\record.py --object scalpel --episodes 1 --max-attempts 3 --enable_cameras --tray-occupancy random --record_mode place --out_dir .\datasets\tray_slots_test\scalpel
```

All five, GUI, RGB and semantic GIFs:

```powershell
.\scripts\launchers\run_all_v2_test.ps1 -Episodes 1 -MaxAttempts 3 -Gui -TrayOccupancy random
```

Use `-TrayOccupancy full` to stress-test four neighboring instruments; `empty`
tests a completely empty tray. The requested episode count counts successful
saves, not attempts. There is no guarantee of 100% physical success.

Use GUI for now: the headless validation exposed intermittent flat gray RGB
buffers on this installation. The cause is unresolved; a new pre-save guard
rejects them. Inspect representative recordings before collecting at scale.
