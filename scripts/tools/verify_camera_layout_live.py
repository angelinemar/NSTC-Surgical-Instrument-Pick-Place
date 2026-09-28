"""Read-only live-vs-saved camera pose audit for Isaac Sim Script Editor."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import omni.usd
import torch
from pxr import UsdGeom

from isaaclab.utils.math import convert_camera_frame_orientation_convention


ROOT = Path(r"C:\IsaacLab\scripts\custom\i4h_project\p4")
CAMERA_LAYOUT = ROOT/'env'/"camera_layout.json"
SHARED_LAYOUT = ROOT/'env'/"shared_layout.json"

CAMERA_PATHS = {
    "cam_front": "/World/envs/env_0/cam_front",
    "grip_cam_b": "/World/envs/env_0/Robot/panda_hand/GripCamB_Final",
    "cam_top": "/World/envs/env_0/CamTop",
    "cam_left": "/World/envs/env_0/CamLeft",
    "cam_right": "/World/envs/env_0/CamRight",
    "cam_tray": "/World/envs/env_0/CamTray",
}


def _quat_angle_deg(a, b):
    """Sign-invariant angular difference between wxyz quaternions."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= max(np.linalg.norm(qa), 1.0e-12)
    qb /= max(np.linalg.norm(qb), 1.0e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def main():
    if not CAMERA_LAYOUT.is_file():
        print("[CAMERA VERIFY FAIL] camera_layout.json does not exist yet:", CAMERA_LAYOUT)
        print("Wait for [CAMERA AUTOSAVED], then run this verifier again.")
        return

    saved = json.loads(CAMERA_LAYOUT.read_text(encoding="utf-8"))
    # P4 aligns the hospital to tabletop Z=0, with zero workspace translation.
    # shared_layout.json retains P3 archival values; do not use that offset.
    offset = np.zeros(3,dtype=np.float64)
    stage = omni.usd.get_context().get_stage()

    print("\n" + "=" * 100)
    print("CAMERA LIVE VS camera_layout.json")
    print("saved file =", CAMERA_LAYOUT)
    print("workspace offset =", offset.tolist())
    print("PASS tolerance: position <= 0.1 mm, rotation <= 0.05 deg")
    print("=" * 100)

    all_ok = True
    for name, path in CAMERA_PATHS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[MISSING] {name:12s} {path}")
            all_ok = False
            continue

        local = UsdGeom.Xformable(prim).GetLocalTransformation()
        t = local.ExtractTranslation()
        q = local.ExtractRotationQuat()
        qi = q.GetImaginary()
        live_pos = np.asarray([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
        if name != "grip_cam_b":
            live_pos -= offset

        quat_gl = torch.tensor(
            [[float(q.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])]],
            dtype=torch.float32,
        )
        live_rot = convert_camera_frame_orientation_convention(
            quat_gl, origin="opengl", target="ros"
        )[0].cpu().numpy().astype(np.float64)

        entry = saved.get("cameras", {}).get(name)
        if entry is None and name == 'cam_front':
            entry = saved.get('cameras', {}).get('camera')
        if entry is None:
            print(f"[NO JSON] {name:12s}")
            all_ok = False
            continue
        saved_pos = np.asarray(entry["pos"], dtype=np.float64)
        saved_rot = np.asarray(entry["rot"], dtype=np.float64)
        pos_mm = float(np.linalg.norm(live_pos - saved_pos) * 1000.0)
        rot_deg = _quat_angle_deg(live_rot, saved_rot)
        ok = pos_mm <= 0.1 and rot_deg <= 0.05
        all_ok &= ok

        print(f"\n[{'PASS' if ok else 'DIFF'}] {name:12s} pos_delta={pos_mm:.4f} mm rot_delta={rot_deg:.5f} deg")
        print("  LIVE/CFG pos =", [round(v, 9) for v in live_pos.tolist()])
        print("  JSON     pos =", [round(v, 9) for v in saved_pos.tolist()])
        print("  LIVE/CFG rot =", [round(v, 9) for v in live_rot.tolist()])
        print("  JSON     rot =", [round(v, 9) for v in saved_rot.tolist()])

    print("\n" + ("[ALL CAMERAS MATCH SAVED JSON]" if all_ok else "[CAMERA DIFFERENCE FOUND - DO NOT QUIT YET]"))


main()
