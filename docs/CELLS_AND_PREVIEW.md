# Instrument-sized cells and exact camera preview

- Grid: 2 x 5, square 0.20m cells; total X=0.12..0.52, Y=-0.6924..0.3076m.
- Measured longest dimension: scalpel_type2 0.150215m, love_retractor 0.150188m,
  scalpel 0.150050m, scissor 0.133942m, kelly 0.131280m (asset-root local bounds
  with configured scale; orientations are applied by existing object handlers).
- The sampler considers the entire root-relative envelope, not just length:
  scissor's off-center pivot requires a conservative radius of 0.085295m.
- Five distinct cells are sampled without replacement. Yaw is preserved from
  the original randomized sampler. In-cell jitter reserves 1cm envelope clearance.
- Following settling, rotated asset bounds are checked for 2mm line clearance.
  Escaped footprints are translated in XY only and settled again (at most two
  correction passes). Recording is rejected if containment still fails.
- No pick/place stage, instrument scale, grasp offsets or camera resolution changed.

## Verified

`verify_cell_spawn.py`: 5000 randomized spawn sets passed for all five targets.
`validation/cells_tuner.log`: all five settled footprints passed after recenter;
RGB readiness passed; six live 224x224 RGB uploads completed without resize.
Test camera pose save/load comparison passed (max translation error <5e-8m,
quaternion agreement >0.99999). The active camera_layout.json was NOT overwritten
by this test; the output was validation/cells_tuner/camera_layout_test.json.
This is not a full recording/pick-place qualification of the expanded cells.
The previously documented intermittent headless RGB startup issue remains open.

## Use

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\scripts\launchers\view_env.ps1
# Camera editing + clean recorder-framing preview + pose autosave:
.\scripts\launchers\view_env.ps1 -Cameras
```

The floating **P4 Recording RGB | 224 x 224 | LIVE** panel displays actual native
camera RGB buffers through the same symmetric 224 crop as the H5 saver. Main
viewport resolution/aspect does not set H5 image size. All six unique cameras
appear; wrist and gripper remain the same underlying camera. Edit camera prim
XYZ/rotation and watch the panel. Wait for `[P4 CAMERA SAVED]`; Ctrl+C makes a
final save. Ctrl+S is not needed for camera_layout.json.

The image upload API follows NVIDIA's ByteImageProvider documentation:
https://docs.omniverse.nvidia.com/kit/docs/omni.ui/2.26.8/omni.ui/omni.ui.ByteImageProvider.html
