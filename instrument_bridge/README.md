# RF-DETR-Seg bridge for OpenVLA and other policies

This branch packages **two trained instance-segmentation checkpoints** and a small inference API. It accepts RGB frames and returns instrument classes, scores, boxes, masks, and per-class counts. It does not command a robot. Your OpenVLA code decides how to select and lock one detected instance and how to use that result.

| Choice | Training source | Checkpoint |
|---|---|---|
| A | Synthetic instrument scenes | `models/object_detection/A/checkpoint_best_total.pth` |
| B | Synthetic + Diffusion Policy recordings | `models/object_detection/B/checkpoint_best_total.pth` |

Both are RF-DETR-Seg Small at 768 input resolution and predict `scalpel`, `scissor`, `love_retractor`, `kelly`, `scalpel_type2`. Choose B as the starting point for DP-style camera views; compare both on your own held-out images.

## Install after cloning

```bash
git clone --branch angel/openvla-object-detection-bridge https://github.com/angelinemar/NSTC-Surgical-Instrument-Pick-Place.git
cd NSTC-Surgical-Instrument-Pick-Place
git lfs pull
python3.11 -m venv .venv-rfdetr
.venv-rfdetr/bin/python -m pip install --upgrade pip
.venv-rfdetr/bin/python -m pip install 'rfdetr==1.11.0' pycocotools h5py scipy pillow numpy
export NSTC_RFDETR_PYTHON="$PWD/.venv-rfdetr/bin/python"
```

Verify `git lfs pull` downloaded the weights: both checkpoint files should be about 133 MB rather than small text pointers. A CUDA-capable PyTorch installation is recommended; use your machine's supported PyTorch/CUDA setup.

## Graphical camera audit

Start the local GUI with `./instrument-studio6.sh` after installing the RF-DETR environment and running `git lfs pull`. Choose **A** or **B**, set a score threshold, and load either:

The GUI uses Tkinter, which must be available in the Python 3.11 installation. On this NSTC workstation it is available in the RF-DETR virtual environment. If `import tkinter` fails on another computer, install that system's Python Tk package before launching the GUI.

- a synthetic recorder's `train/scenes/scene_XXXXXX` folder containing `scene.json`, or
- a recorded DP `.h5` episode and a frame number.

Click **Run detection on every camera**. The left side lists every camera separately with mask IoU≥0.50 TP, FP, FN and F1. Select a camera to see its prediction overlay, ground-truth overlay, per-class visible counts, and its own conclusion. Toggle **Image overlay** between model and ground truth. Synthetic scenes show every recorded hemisphere view; DP episodes show all six DP cameras. The GUI intentionally does **not** combine the six views into a single accuracy score, because they observe different parts of the scene. Reports and preview PNGs are saved under the chosen **Save reports to** folder.

**Open scene recorder** starts the existing Isaac Sim detection recorder panel. There you can set instrument count, camera views, randomization mode and a custom wide lighting profile. Record a new scene, then click **Use latest recorded scene** here, or browse to its `scene_XXXXXX` folder. A live Isaac Sim scene is not controlled from the audit GUI itself; the recorder saves the ground truth used for evaluation. Do not use a scene from the model's training split as evidence of generalization. Prefer an independent test split or a new held-out capture.

The `Spawn` column counts all instruments placed on the table, including hidden ones. `Visible` is the camera's annotated instance count; `Pred` is the model output. Precision/recall/F1 compare predicted **masks** with visible instance masks, class-correct at mask IoU≥0.50. They are diagnostic values for one scene/frame, not mAP or whole-table object-count accuracy. DP labels may be incomplete.

After detection, pick one **predicted instance** by clicking its box or selecting it from the list. Type `pick up the scalpel` (or another supported class) and click **Select by command** to select it automatically only when exactly one instance of that class appears in the selected camera. Duplicates require an explicit click/list selection. Click **Lock selected target** to write `target_request.json`, which contains the instruction, class, confidence, box, mask, pixel center, camera, model A/B, and a clean RGB image path. Simulator spawn pose and ground-truth annotations are excluded from this action request. If you change camera or run detection again, the old lock is cleared.

The optional **Run adapter** button launches a user-supplied Python script with the request JSON as its sole argument. Choose **OpenVLA adapter** or **DP adapter** only to label the intended backend; neither model is packaged in this GUI. For a safe end-to-end handoff check, select `instrument_bridge/examples/dry_run_action_adapter.py` and the desired Python interpreter. It prints the selected instance without moving a robot. See [OPENVLA_HANDOFF.md](OPENVLA_HANDOFF.md) for the partner integration contract and the steps required before live action.

## Run on a camera image

```bash
./detect-instruments6.sh --model B --image /absolute/path/frame.png --output /tmp/detect_B
./detect-instruments6.sh --model A --image /absolute/path/frame.png --output /tmp/detect_A
```

Each output folder contains `detections.json` and `preview.png`. The JSON includes an `objects` list and `counts` mapping. Every object has `class_name`, `score`, `bbox_xyxy`, `center_xy`, `mask_area_px`, and a COCO `mask_rle`. Set `--threshold 0.5` or another score threshold explicitly if needed. The model loads once for each CLI invocation.

## Python API: load once in your robot process

```python
from instrument_bridge import InstrumentDetector

model = InstrumentDetector(model='B', threshold=0.5)
result, masks = model.predict(front_rgb_uint8_hwc)
print(result['counts'])          # visible predictions by instrument class
print(result['objects'])         # every instance, with box, score and mask RLE
# masks[i] is a boolean H×W numpy array for result['objects'][i]
```

`front_rgb_uint8_hwc` must be RGB, H×W×3, `uint8`. Keep the model in memory; don't construct it each robot tick. If multiple instances share a class, choose a specific instance using the box/mask and **lock its identity** in your policy logic. A class label alone cannot tell OpenVLA which duplicate instrument to pick.

For **table-only visible count estimates**, pass `table_roi_xyxy=[x1,y1,x2,y2]` in pixel coordinates to `predict`; or `--table-roi X1 Y1 X2 Y2` in the CLI. It counts predicted object centers inside that 2D region. Calibrate the ROI for each camera; hidden objects remain invisible.

## OpenVLA in a different Python environment

Use the [example subprocess client](examples/openvla_client.py). It starts `serve-instruments6.sh` once, then sends RGB frames as lossless PNG through JSON lines. The service returns one JSON response per frame. Set `NSTC_RFDETR_PYTHON` to the RF-DETR environment before starting OpenVLA. OpenVLA's own model checkpoint stays in its own environment; this package provides detections to it and does not assume an OpenVLA API or overwrite its weights.

Minimal service request:

```json
{"request_id": 1, "image_path": "/absolute/path/frame.png"}
```

A request may instead contain `rgb_png_base64` and optional `table_roi_xyxy`. The response has `ok`, the matching `request_id`, `counts`, and `objects`. The example client implements that protocol from an in-memory RGB array.

## Check counts against recorded ground truth

With a DP recording available locally:

```bash
./detect-instruments6.sh --model B \
  --episode /absolute/path/episode_000001.h5 \
  --camera all --frame 0 --output /tmp/B_six_cameras
```

This creates `detections_all_cameras.json` and one preview per camera. It reports class-correct **mask IoU≥0.5** TP/FP/FN, precision, recall and F1 per class and camera, plus an aggregate over the six camera images. The count fields mean different things:

- `counts`: model-predicted instances visible in one image.
- `visible_ground_truth_counts`: labelled visible instances in that camera frame.
- `spawned_on_table_at_episode_start`: instrument instances in scene spawn metadata; includes occluded objects and is not restricted to what this camera sees.

The dataset's DP instance labels are partial in some recordings. A one-frame check is useful for integration, **not a dataset-wide accuracy claim or mAP**. Use the existing `training.evaluate_segmentation` pipeline for full-dataset AP metrics. Compare A and B with the same episode, camera, frame, score threshold and evaluation settings.

## Scope and provenance

The checkpoint files are tracked with Git LFS. Model A SHA-256: `6c15f47d239c783e6109ba6cf6bb45a7f4d4b5f45eb89251aa7cac58a7d3db15`. Model B SHA-256: `7d97fad639a711e75418d92b5142adf0512f842e8cef8a94d9e8ea068d2bece9`. The detector performs perception only. It does not combine multiple cameras into a deduplicated world inventory or prove robotic grasp success.
