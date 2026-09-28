# P4 migration validation - 2026-09-14

> Historical migration test below predates the annotated-image layout revision.
> Current robot/grid/tray positions are documented in README_P4.md and
> scene_layout.json. Do not treat the old sensor_safe episode as validation of
> the expanded grid or relocated/resized tray. Previous layout files are retained
> in archive/layout_before_green_center_20260914.

## Scope and preservation

P4 is an independent runtime copy of P3's active `with_env_cfg/refactored_v2`.
No P3 recorder, configuration, asset or dataset was edited or removed.
`verify_migration.py` verifies the recorded P3 source hashes and compares all
five backends: their only edits are relocation of asset paths to P4.
Shared settle/camera/metrics contracts, object registry and motion timing AST
match P3. All active Python files parse and all five registry entries resolve.

The new hospital USDZ was copied from Downloads with matching SHA256.
Old assets retained for compatibility are not the active hospital/table scene.

## Scene

- Native Table_04, approximately 2.2 x 0.9 m, native furniture collision.
- Robot near table center; 3 x 4 square spawn cells (10 cm per side), tray on
  the opposite side of the same table. Grid does not fill unreachable edges.
- All seven table selections pass preliminary geometric footprint/reach checks.
  This is not proof of collision-free motion for every randomized episode.
- Hospital transform aligns its native tabletop to task Z=0; instrument and
  grasp offsets remain inherited from P3. Tray's offset asset pivot is centered
  on the logical placement target; its underside aligns with the cloth.
- Static camera poses are adjusted. Camera crop/resolution/intrinsics are
  unchanged: native 448x336, centered 224x224 saved output. Wrist pose unchanged.
- Shared soft-light emitters are hidden from primary camera rays without
  disabling their illumination. A post-settle RGB readiness check renders
  without advancing physics and rejects persistently black/constant streams.

## Completed physical recording

`validation/sensor_safe` completed one scalpel pick/place episode:

| Item | Result |
| --- | --- |
| Attempts / successes | 1 / 1 |
| Spawn / pick / place failures | 0 / 0 / 0 |
| Pick frames | 161 |
| Place frames | 118 |
| Total control steps | 279 |
| Successful-attempt wall time | 4.416 minutes |

Both H5s passed `validate_rgb_h5.py`: six unique RGB streams, 224x224x3
uint8, no all-black frames, finite 8D actions and calibration arrays, aligned
uint16 semantic image shapes and IDs within the existing 0..7 range.
Semantic ID range checks do not prove every pixel is correctly classified.
RGB GIF contains seven panels because gripper/wrist retain P3's shared view.

The full-run GIF was reviewed. The old vertical tray-camera viewpoint was
occluded by the retreating arm, so the final tray camera was moved to an oblique
view at (0.85, 0.65, 0.65), aimed at the tray center. Resolution is unchanged.
The full `sensor_safe` H5/GIF predates that final camera adjustment. The separate
`validation/oblique_retreat_camera` capture checks the updated view by replaying
the final robot joint state from the completed place episode; it is not a new
recorded demonstration or a complete trajectory replay.

## Remaining qualification

Only scalpel has completed a physical P4 recording in this migration check.
Scissor, love_retractor, kelly and scalpel_type2 are source/configuration-audited,
not yet physically validated in the new scene. Run all five with GUI before
collecting production data; one success does not establish randomized reliability.

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\run_all_v2_test.ps1 -Episodes 1 -Gui
```

All `validation/` outputs are diagnostics, not production data. Earlier diagnostic
runs include invalid images and must not be used for training. Each new recorder
run writes `scene_manifest.json` alongside its output to identify its scene and
camera version without changing the inherited H5 schema.
