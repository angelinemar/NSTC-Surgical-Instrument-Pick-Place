
import json
from pathlib import Path
from phase4_camera_names import sensor_names


CAMERA_WIDTH  = 448
CAMERA_HEIGHT = 336
CAMERA_OUTPUT_CROP_SIZE = 224
CAMERA_DATA_TYPES = ["rgb", "distance_to_image_plane", "semantic_segmentation"]

PHASE3_CAMERAS = {
    "camera": {
        "prim_path": "{ENV_REGEX_NS}/cam_front",
        "focal_length": 16.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.01, 10.0),
        "pos": (1.256523, 0.000000, 0.297649),
        "rot": (-0.39384708, 0.57984880, 0.58908579, -0.40204201),
    },

    "grip_cam_b": {
        "prim_path": "{ENV_REGEX_NS}/Robot/panda_hand/GripCamB_Final",
        "focal_length": 14.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.001, 2.0),
        "pos": (0.061365, 0.017054, 0.040467),
        "rot": (-0.64681794, 0.15352391, 0.25956301, -0.70048840),
    },

    "cam_top": {
        "prim_path": "{ENV_REGEX_NS}/CamTop",
        "focal_length": 16.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.01, 10.0),
        "pos": (0.510759, -0.024618, 1.243130),
        "rot": (-0.00625446, 0.70707912, 0.70707912, -0.00625446),
    },

    # From your GUI CamLeft, converted to IsaacLab CameraCfg rot convention
    "cam_left": {
        "prim_path": "{ENV_REGEX_NS}/CamLeft",
        "focal_length": 16.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.01, 10.0),
        "pos": (0.349634, 0.638153, 0.779947),
        "rot": (-0.00414037, 0.00127212, -0.95588456, 0.29371064),
    },

    # From your GUI CamRight, converted to IsaacLab CameraCfg rot convention
    "cam_right": {
        "prim_path": "{ENV_REGEX_NS}/CamRight",
        "focal_length": 16.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.01, 10.0),
        "pos": (0.301181, -0.669862, 0.779950),
        "rot": (0.29903149, -0.95411975, -0.00113561, -0.01530929),
    },

    "cam_tray": {
        "prim_path": "{ENV_REGEX_NS}/CamTray",
        "focal_length": 16.0,
        "horizontal_aperture": 20.955,
        "clipping_range": (0.01, 10.0),
        # Centered above the shared tray on surgical table 2. Static-camera
        # coordinates are workspace-local and receive WORKSPACE.offset later.
        "pos": (0.3680401892, 0.5927763309, 0.882957),
        "rot": (0.00503358, 0.70555958, 0.70859880, -0.00693478),
    },
}


# GUI camera tuner writes only poses here. Keeping it separate from this file
# makes camera changes recoverable and avoids rewriting executable Python.
CAMERA_LAYOUT_FILE = Path(__file__).parent / 'env' / "camera_layout.json"
if CAMERA_LAYOUT_FILE.is_file():
    try:
        _saved = json.loads(CAMERA_LAYOUT_FILE.read_text(encoding="utf-8"))
        for _name, _pose in sensor_names(_saved.get("cameras", {})).items():
            if _name in PHASE3_CAMERAS:
                if "pos" in _pose:
                    PHASE3_CAMERAS[_name]["pos"] = tuple(float(v) for v in _pose["pos"])
                if "rot" in _pose:
                    PHASE3_CAMERAS[_name]["rot"] = tuple(float(v) for v in _pose["rot"])
        print(f"[CAMERA LAYOUT LOADED] {CAMERA_LAYOUT_FILE}")
    except Exception as _camera_layout_error:
        raise RuntimeError(f"Invalid camera layout: {CAMERA_LAYOUT_FILE}") from _camera_layout_error
