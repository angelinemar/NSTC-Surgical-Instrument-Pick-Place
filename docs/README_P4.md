# P4 - new hospital, copied P3 object handlers

Start here: `C:\IsaacLab\scripts\custom\i4h_project\p4`.
P3 is preserved. Production datasets, old test output and previews were not copied.

Latest layout and test status: see `LAYOUT_REVISION.md`. The expanded-grid /
left-tray layout passed one scalpel motion check, but repeated normal headless
recorder starts encountered RGB readiness failures. Do not start production
collection until that rendering issue is resolved. Recording guards remain on.

## Preview

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\view_env.ps1
```

Camera tuner: `.\view_env.ps1 -Cameras` (same native 448x336 -> 224x224 crop as P3).
Use the **P4 Recording RGB | 224 x 224 | LIVE** panel to judge recorded framing.
It shows six real sensor crops with no resize. The main viewport's window size
does not determine recording resolution. Normal preview shows this panel too,
but includes the diagnostic grid; camera-tuning mode does not draw that grid.
Camera edits autosave to P4's `camera_layout.json`.
Close the normal scene preview without saving: it never writes layout changes.
The camera tuner does save camera edits; wait for its save message before exiting.

## Scene selection

Edit `scene_layout.json` before launch. `selected_table` accepts Table_01 to Table_07.
The whole hospital is aligned to that table, so instrument handlers always see
the familiar Z=0 tabletop and an unrotated robot frame. Asset sizes remain 1:1.
The table's original colliders are used, not P3's invisible Franka table.
The bundled PhysicsScene is disabled in favor of IsaacLab's PhysicsScene.

Current arrangement (annotated-image revision): Table_04; robot centered laterally
in the exposed green area at Y=-0.1924m; a 2x5 grid of 20cm square cells directly
in front, spanning 40x100cm. Each of the five instruments is assigned a distinct
random cell; yaw remains random and jitter accounts for the asset's full envelope.
The longest instrument is approximately 15.02cm. After settling, any escaped
footprint is recentered in XY and settled/validated again before recording.
A 55x22cm tray sits on the left narrow blue pad at XY=(0.15,-0.89).
Tray dimensions change in-plane only, not height. Geometric reach checks are preliminary; run physical
tests before collecting production data. Static camera positions are adjusted;
the tray camera also rotates toward the tray from an oblique viewpoint to reduce
robot occlusion. Wrist camera attachment/pose, focal lengths and image dimensions
are preserved.

P4 scene transforms use `scene_layout.json` as authority. Do not use the old
workspace/layout GUI to persist robot/table/hospital transformations: inherited
P3 `shared_layout.json` is compatibility data and its workspace values are
overridden by P4. Use the preview for inspection and `scene_layout.json` for edits.

## Record / test

```powershell
.\run_all_v2_test.ps1 -Episodes 1 -Gui
# Each recorder, headless:
C:\IsaacLab\_isaac_sim\python.bat .\record.py --object love_retractor --episodes 1 --record_mode pick --headless --out_dir .\test_love
# Check recorded images and shape/calibration contracts before training:
C:\IsaacLab\_isaac_sim\python.bat .\validate_rgb_h5.py .\test_love
```

All five object backends, motion timings, gripper logic, stage IDs, state/action
schema, crop settings and instrumentation are inherited from P3. The only backend
edits relocate absolute asset paths to P4. See `P3_COPY_MANIFEST.json` for hashes.
Assets have relative internal dependencies, including their texture packages.
P4's shared `phase4_cell_spawn.py` intentionally replaces the final initial
placement coordinates for all backends. It does not modify grasp handling.
Requested cell coordinates and final settled poses are separate metadata;
settle_steps includes all recenter/settle passes.
Do not mix P3/P4 datasets without recording the scene/camera version in a manifest.

P4 centers the offset tray asset on the logical placement target and seats its
underside at the new cloth surface. `tray_root_z=0.008` is an asset-pivot height,
not an 8mm air gap. After physical settling, P4 warms up the renderer without
advancing physics and checks every RGB camera before recording. This changes
startup/render preparation only, not motion stages or their step counts.
The large shared soft lights keep their illumination but their emitters are
invisible to primary camera rays, preventing cameras from viewing the inside
of a light after relocation. This is a P4-only scene/light setting.
