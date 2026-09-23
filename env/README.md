# Environment

Canonical environment/camera source and configuration live here. The active implementation
is loaded through root compatibility modules so existing Isaac/backend imports keep working.

- `phase4_scene.py`: hospital/table, transforms, collision geometry, manifest.
- `phase3_shared_env_cfg.py`: shared Isaac environment builder; historical name is active.
- `phase3_camera_tuning.py`: camera definitions and persistent layout handling.
- `phase3_recorder_camera_patch.py`: camera and recording hooks.
- `phase4_camera_names.py`: public/sensor camera-name mapping.
- `scene_layout.json`: scene/table/robot/grid/tray configuration.
- `camera_layout.json`: active camera poses.
- `shared_layout.json`: legacy compatibility configuration.
- `instrument_preview_geometry.json`: panel preview geometry.
- `camera_layout_front_tray_candidate.json`: historical candidate, not the active layout.

Root JSON aliases are hard links to these files, not independent copies. Keep the links intact
and run `python scripts/check_structure.py` after any layout-file maintenance. Do not alter
layouts during a frozen benchmark. Organization changed file locations only, not values.
