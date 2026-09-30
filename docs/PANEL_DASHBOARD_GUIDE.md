# P4 live dashboard and measured scene report

Launch from PowerShell:

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\RUNME.ps1 -Mode panel
```

Restart an already-open panel to load the new interface. P3 is unchanged.

## Counters and lights

- Top right: next successful episode being attempted, saved/goal, attempt number, failures and active stage.
- Saved increments on `SAVE SPLIT`, not just a positive physical result. Repeated failure lines count once per attempt. Final counters reconcile with run_metrics.json.
- Bottom right has three red indicators: RESET / WAIT, RECORDING, END / SAVING. One indicator is bright; the others are dim. End alone does not mean success: read Complete or Stopped / incomplete.
- Single mode uses the requested saved episode count. Grid cycles uses rounds times all 10 cells. Failed attempts do not advance the cell.

## Instrument rotation display

The old generic yaw line did not represent every instrument's asset axis. The panel now draws class-specific outer hulls and a positive local-axis arrow. This is an envelope, not detailed ring/hole geometry and not an anatomical-tip marker.

Before Prepare: calibrated prediction based on scaled USD vertices and a settled reference pose for that class. Different contact dynamics can change the actual orientation.

After Prepare / settle: measured world-space geometry is delivered by the simulator. This is a snapshot, not a continuously animated moving-object feed. It is refreshed on preparation/reset. If measurement fails, a warning is logged and prediction remains; display telemetry cannot abort recording.

No object pose from this display is added to policy observations. Scene, camera size, physics masses and motion timing are unchanged by this dashboard update.

## Dimensioned report

`output/pdf/p4_scene_dimensions_and_recorder_report_20260916.pdf` contains 10 pages: table, grid, tray slots, instrument envelopes and configured masses, camera poses/crop/calibration, object distances, recorded EE trajectory/timing, dataset schema and validation limits.

Historical report scripts and measurements are retained locally, outside the published source tree. The PDF is a dated snapshot, not live documentation after layout edits.

Configured simulation mass is not a manufacturer's instrument weight. Instrument L/W/T values describe a bounding envelope, not steel thickness. Settled height can differ with pose. Pre-run camera positions are world/env_0 except the wrist mount-local position; recorded extrinsics are robot-base relative.

Full-grid/all-yaw physical success and downstream training quality are not guaranteed by the dashboard or report.
