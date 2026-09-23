# P4 hospital recorder workspace

Start with [README_P4.md](README_P4.md). P3 remains unchanged.

For the current dark control panel, automatic retry and successful grid-cycle
collection, see [PANEL_COVERAGE_GUIDE.md](PANEL_COVERAGE_GUIDE.md).
Launch it from this folder with `python .\control_panel.py` (ordinary Python).

For current fixed-slot placement, see [TRAY_PLACEMENT_GUIDE.md](TRAY_PLACEMENT_GUIDE.md)
and measured results/limitations in [TRAY_PLACEMENT_VALIDATION.md](TRAY_PLACEMENT_VALIDATION.md).
Use a fresh dataset folder for the new tray-slots contract.

- `view_env.ps1`: inspect native hospital/table/grid/tray without recording or saving layout.
- `view_env.ps1 -Cameras`: shared camera tuner; autosaves P4 camera poses.
- `scene_layout.json`: hospital/table selection, robot, grid and tray coordinates.
- `camera_layout.json`: camera poses; image dimensions remain inherited from P3.
- `record.py`: one entry point for all five object handlers.
- `run_all_v2_test.ps1 -Episodes 1 -Gui`: test all five with RGB and semantic GIFs.
- `backends/`: P3 instrument handlers; asset paths relocated, behavior preserved.
- `phase3_*.py`: actively used shared recording modules; names retained for compatibility.
- `phase4_scene.py`: new hospital placement and native table collider integration.
- `phase4_runtime.py`: shared spawn/reset ownership and RGB readiness after settle.
- `phase4_feedback.py`: measured close/lift/place gates; immediate failed-attempt discard.
- `phase4_coverage.py`: successful-cell scheduling and saved-H5 coverage audit.
- `phase4_tray_slots.py`: fixed class slots, randomized non-target tray occupancy, physical placement QC.
- `control_panel.py`: manual/auto recording, target selection and tray occupancy controls (0-1 per type).
- `validate_rgb_h5.py`: check H5 dimensions, finite actions/calibration, semantic IDs and black frames.
- `assets/`: local asset packages. Old hospital/table files are retained dependencies,
  but are not spawned in P4.
- `validation/`: diagnostic and smoke-test outputs, including failed tests; NEVER use as a production dataset.
- `P3_COPY_MANIFEST.json`: migration checksums.

Do not run `prepare_p4.py` again on a tuned workspace: it is the migration utility,
not a recorder. Scene/camera changes require revalidation before bulk recording.
