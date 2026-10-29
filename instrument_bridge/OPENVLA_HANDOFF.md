# Target handoff for OpenVLA, DP, and other action policies

## What this branch actually provides

- RF-DETR-Seg Small checkpoints **A** (synthetic training) and **B** (synthetic + DP training), both in Git LFS.
- Local GUI `./instrument-studio6.sh` for saved synthetic scenes and DP HDF5 frames. It scores **each camera separately** and shows predicted and ground-truth overlays. The existing Isaac Sim recorder panel can create new randomized scenes and lighting; the GUI can then load the latest completed scene.
- One-click target selection from a model-predicted box, a list, or a simple class instruction. Multiple same-class instances require an explicit choice. This is a **single-camera 2D instance lock**; it is not a cross-camera 3D track.
- A JSON handoff file plus a clean RGB PNG. External action code can consume these without importing RF-DETR or Isaac Sim into OpenVLA's Python environment.

## Working handoff demo

1. Run `./instrument-studio6.sh`; choose A/B and a saved scene or DP episode. Click **Run detection on every camera**.
2. Select a camera. Type `pick up a scalpel`, then choose the intended detection by clicking it or selecting it in the list. With exactly one scalpel, **Select by command** chooses it automatically.
3. Click **Lock selected target**. The selected report folder now contains `target_request.json` and `<camera>_rgb.png`.
4. Set **Adapter Python** to a Python interpreter and **Adapter script** to `instrument_bridge/examples/dry_run_action_adapter.py`. Click **Run adapter**. `action_result.json` records the dry-run output and exit status. No robot action occurs.

## Request contract (`nstc_target_request_v1`)

```json
{
  "contract": "nstc_target_request_v1",
  "task": "pick",
  "instruction": "pick up a scalpel",
  "perception_model": "B",
  "camera": "front",
  "image_path": "/absolute/path/front_rgb.png",
  "instance_index": 2,
  "selected_by": "click",
  "target": {
    "class_name": "scalpel",
    "score": 0.92,
    "bbox_xyxy": [110, 90, 220, 180],
    "center_xy": [165, 135],
    "mask_rle": {"size": [448, 448], "counts": "COCO compressed RLE string"}
  },
  "coordinate_frame": "camera_pixels",
  "pose_3d": null,
  "camera_observations": [
    {"camera": "front", "image_path": "/absolute/path/front_rgb.png",
     "predicted_counts": {"scalpel": 2},
     "instances_2d": [{"class_name": "scalpel", "score": 0.92,
       "bbox_xyxy": [110, 90, 220, 180], "center_xy": [165, 135]}]}
  ]
}
```

The example values above are illustrative and abbreviate the remaining classes/cameras. The real file also contains the threshold, recorded source path/frame, backend label and a note. `camera_observations` lists **separate 2D observations per camera**; repeated views of one object are not deduplicated. Ground-truth counts and spawn poses are not included. The adapter is called as:

```bash
<adapter-python> <adapter-script.py> <absolute-path-to-target_request.json>
```

Exit code, stdout and stderr are saved to `action_result.json`. **Exit code 0 means the adapter process finished; it does not prove a successful grasp.** Your adapter must write or log its own scene-level success criteria.

## How to connect your OpenVLA policy

Read `target_request.json`, load `image_path`, and use the `instruction` together with the selected `target` box/mask. Your policy wrapper may crop or highlight the selected instance if its input contract supports it. Retain the full original image and the target metadata in logs. Check that the live camera still observes the same object before executing actions: this GUI currently exports a **saved frame**, not a synchronized live camera stream. If the robot or object moved, refresh the observation and detect/track again.

Do not assume OpenVLA accepts this JSON natively. The partner's adapter must translate it into that model's actual image, language, and action API, and implement robot/simulator stepping, gripper control, stop conditions, and success measurement. Switching **OpenVLA adapter** to **DP adapter** in the GUI only changes the label in the handoff. The current DP models cover trained Pick/Place stages, so a DP adapter also needs stage preparation and transitions outside those learned stages.

The detector outputs **camera pixels**, not a 3D robot target or grasp orientation. If your action controller requires XYZ, add calibrated camera intrinsics/extrinsics plus depth or a verified multi-view estimator; report localization error separately. The simulator's exact spawn pose is evaluation ground truth and is deliberately absent from `target_request.json`. Using it for robot control would be an oracle controller, not a test of detection-based localization.

## Evaluation boundaries

- `report.json` contains per-camera precision, recall, F1, TP/FP/FN at mask IoU 0.50. One scene/frame is a diagnostic, not dataset mAP.
- `spawn_on_table` includes hidden objects. `visible_ground_truth` and predictions are camera-specific. DP labels may be incomplete.
- No result is merged across the six DP cameras or deduplicated into a world inventory. A robust multi-camera tracker needs camera calibration and instance association.
- For live performance, record each attempted target, selected camera, detector model, action policy/version, action steps, grasp/release result, and failure reason. Report detection metrics and pick/place success separately.
