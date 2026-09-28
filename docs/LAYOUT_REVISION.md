# Annotated-image layout revision (2026-09-14)

P4 only. P3 is unchanged. Previous P4 scene/camera files are preserved in
`archive/layout_before_green_center_20260914`.

| Component | Current layout |
| --- | --- |
| Selected furniture | Native Table_04, unchanged scale/pose |
| Exposed green-area Y bounds | -0.744513 .. +0.359677 m, measured from blue-pad material subsets |
| Green-area lateral midpoint | -0.192418 m |
| Robot base XYZ | (0, -0.1924, 0.003) m; orientation unchanged |
| Spawn X bounds | 0.20 .. 0.50 m |
| Spawn Y bounds | -0.6424 .. +0.2576 m |
| Grid | 3 x 9 = 27 square 10cm cells, total 30 x 90cm |
| Green side margins to grid | About 10.2cm on both lateral sides |
| Tray center XY | (0.15, -0.89) m, on left narrow blue pad |
| Tray physical footprint | 55cm along world X, 22cm along world Y |
| Tray height/thickness | Unchanged Z scale and pivot height (0.008m) |
| Robot-to-tray center XY distance | About 0.714m |
| Furthest grid corner XY distance from robot | About 0.673m |

The grid defines permissible object centers, with an additional 8.5cm footprint
clearance check. Tray checks use its rotated rectangle, not a fixed circular
margin. These checks do not guarantee every randomized grasp/orientation is
reachable without collisions. No instrument mesh, grasp offset, stage, action
schema or timing is modified.

Static cameras have been reframed to cover the enlarged grid; wrist pose is
unchanged. Image dimensions/intrinsics remain the P3 settings (448x336 native,
224x224 centered saved crop). The grid is drawn only in preview, not recording.

## Verification

- `verify_migration.py`: PASS for all five source backends and shared contracts.
- `validation/green_center_preview`: successful scene spawn/settle and camera
  snapshots; tray position and elongated footprint visually checked.
- `validation/wide_cameras_diagnostic`: final wider camera views rendered;
  top/front views cover the enlarged grid and tray camera covers the tray.
- `validation/green_center_reach_only`: PASS for one unchanged scalpel
  pick/place routine executed through preview. RGB was ready in this run
  (25 warm-up render frames, no warning bypass used). Total 298 control steps;
  XY placement error 0.006753m; final center (0.14634,-0.88432,0.01024)m.
  The move-to-target began 1.1139m from its goal and reached it after 69 steps.
  Final tray-camera snapshot visibly shows the scalpel inside the elongated tray.
  This diagnostic does not save a training episode or certify the normal
  recording entry point; only one randomized scalpel case was tested.
- Headless recorder attempts `green_center_motion`, `green_center_motion_retry`
  and `green_center_direct` stopped at RGB readiness before pick motion. Camera
  output was constant white/black although the separate preview rendered well.
  No successful training episode was saved by these tests. The direct-backend
  process also crashed during shutdown following its RGB error.

The headless recorder rendering problem remains unresolved. Do not interpret
geometric reach or a preview-motion check as a successful production recorder
run. Inspect the scene with `scripts/launchers/view_env.ps1` before further recording.
