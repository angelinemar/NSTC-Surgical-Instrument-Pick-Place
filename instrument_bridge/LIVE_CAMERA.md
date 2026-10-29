# Live camera feed for the Instrument Vision Studio

## Isaac synthetic recorder: already connected

The `detection.record` process now publishes every rendered RGB view **and its instance mask** to `<collection>/live/` immediately after camera capture. The feed is atomic: a manifest points to complete PNGs, so the GUI does not read half-written images.

1. Start the detection recorder panel with `./scripts/launchers/run_detection6.sh panel` and start a collection.
2. Start `./instrument-studio6.sh` in another terminal. Choose **live** or click **Use recorder live feed**. Click **Analyze cameras / Start live**.
3. As the recorder moves its camera around the table, the GUI lists each new view and updates its RF-DETR box/mask overlay, ground-truth mask, detected class counts, and mask IoU≥0.50 precision/recall/F1 for that camera. It loads model A or B once for the live session. Click **Stop live detection** to stop inference.
4. When a scene is committed, choose its saved `scene_XXXXXX` folder for a reproducible recorded-scene report with the full metadata.

The recorder is a sequence of camera snapshots around a static randomized scene, not a six-camera robot video stream. Its camera names are `hemisphere_XX` or `ring_XX` and `top`. It does not run a robot action policy. GPU contention may slow captures and live inference when both processes run on one GPU.

## Feed six DP/OpenVLA cameras from a runtime

The robot runtime must call this helper each time it has a fresh H×W×3 `uint8` RGB camera image:

```python
from instrument_bridge.live_feed import publish_frame

for camera_name, rgb in camera_frames.items():
    publish_frame('/absolute/path/to/live_feed', camera_name, rgb)
```

If Isaac Sim also provides an instance-ID image and a mapping, publish them with the same frame:

```python
publish_frame('/absolute/path/to/live_feed', 'front', front_rgb,
              scene_id=scene_number,
              instance_mask=front_instance_ids,
              instance_classes={'1': {'name': 'scalpel'}, '2': {'name': 'kelly'}},
              spawn_on_table={'scalpel': 1, 'kelly': 1})
```

Only include these labels when they are genuine simulator ground truth. An RGB-only publisher must omit them; the GUI will show metrics as unavailable.

Use names such as `front`, `wrist`, `cam_top`, `cam_left`, `cam_right`, `cam_tray`. Point the GUI's **Source folder** to `/absolute/path/to/live_feed`, choose **live**, and start detection. The helper depends only on NumPy and Pillow, so it can be called from the Isaac Sim process. It neither loads RF-DETR nor blocks on model inference. The GUI loads RF-DETR separately.

Each `<camera>_latest.json` contains the latest immutable PNG filename and timestamp. The GUI watches those manifests, so changes to a frame do not require a restart. It saves `live_report.json`, camera RGB snapshots, and prediction overlays under the chosen report directory. If the publisher supplies an instance mask and class mapping, it also saves a ground-truth overlay and per-camera metrics. Each camera's counts are independent; the same physical instrument may appear in several cameras. Views from different scene IDs are labelled and are not pooled. A simple same-class box-IoU tracker adds `track_id` **within one camera** and resets at a new `scene_id`; it is not robust to long occlusions, viewpoint changes or cross-camera identity.

## What can be scored

For a bare live RGB feed, **ground truth is absent**. The GUI shows predictions, confidence, boxes, masks and counts, but no accuracy percentage. If the live publisher includes instance masks and class names, it computes class-correct mask IoU≥0.50 precision, recall and F1 separately for each camera frame. Recorded-scene mode is still recommended for a reproducible report. Never label a confidence score as accuracy. DP annotations can be partial.

A target locked from a live frame becomes stale when that camera receives a newer frame; the GUI clears the lock and requires a new selection. An action adapter must still verify target identity, robot state and 3D localization before commanding motion.
