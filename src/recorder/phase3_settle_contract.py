"""Shared spawn/settle validation and metadata for all Phase-3 recorders."""

from __future__ import annotations

import json
import math

import torch


SETTLE_OBJECTS = ("scalpel", "scissor", "love_retractor", "kelly", "scalpel_type2")


def _as_list(value):
    return [float(x) for x in value.detach().cpu().tolist()]


def capture_object_states(env, scene_key_for_object, names=SETTLE_OBJECTS):
    """Return JSON-safe root pose and velocity data for every required object."""
    states = {}
    for name in names:
        key = scene_key_for_object(name)
        if key not in env.scene.keys():
            states[name] = {"scene_key": key, "missing": True}
            continue
        obj = env.scene[key]
        states[name] = {
            "scene_key": key,
            "position_w": _as_list(obj.data.root_pos_w[0]),
            "quaternion_wxyz": _as_list(obj.data.root_quat_w[0]),
            "linear_velocity_w": _as_list(obj.data.root_lin_vel_w[0]),
            "angular_velocity_w": _as_list(obj.data.root_ang_vel_w[0]),
            "linear_speed": float(torch.linalg.norm(obj.data.root_lin_vel_w[0]).item()),
            "angular_speed": float(torch.linalg.norm(obj.data.root_ang_vel_w[0]).item()),
        }
    return states


def settle_all_objects(
    env,
    hold_action,
    scene_key_for_object,
    *,
    min_steps,
    poll_every,
    consecutive_ok,
    linear_speed_threshold,
    angular_speed_threshold=0.25,
    position_delta_threshold=0.005,
    rotation_delta_threshold=0.10,
    names=SETTLE_OBJECTS,
):
    """Step physics without recording until every required object is stationary."""
    # Some imported USDs update their root pose to the rigid-body/COM frame on
    # the first physics steps. Give them time to damp after that one-time
    # canonicalization instead of allowing only exactly three polls.
    max_steps = int(min_steps + max(300, consecutive_ok * poll_every))
    consecutive = 0
    last_states = None
    previous_states = None
    pose_deltas = {}
    for step in range(max_steps):
        env.step(hold_action)
        if step < min_steps or step % poll_every:
            continue
        last_states = capture_object_states(env, scene_key_for_object, names)
        all_ok = previous_states is not None
        pose_deltas = {}
        if previous_states is not None:
            for name, state in last_states.items():
                previous = previous_states.get(name, {})
                if state.get("missing", False) or previous.get("missing", False):
                    all_ok = False
                    continue
                p0 = previous["position_w"]
                p1 = state["position_w"]
                position_delta = math.sqrt(sum((p1[i] - p0[i]) ** 2 for i in range(3)))
                q0 = previous["quaternion_wxyz"]
                q1 = state["quaternion_wxyz"]
                dot = min(1.0, max(-1.0, abs(sum(q0[i] * q1[i] for i in range(4)))))
                rotation_delta = 2.0 * math.acos(dot)
                pose_deltas[name] = {
                    "position_m": float(position_delta),
                    "rotation_rad": float(rotation_delta),
                }
                if (
                    state["linear_speed"] > linear_speed_threshold
                    or position_delta > position_delta_threshold
                    or rotation_delta > rotation_delta_threshold
                ):
                    all_ok = False
        previous_states = last_states
        consecutive = consecutive + 1 if all_ok else 0
        if consecutive >= consecutive_ok:
            return {
                "success": True,
                "steps": step + 1,
                "reason": "all_objects_stationary",
                "states": last_states,
                "pose_deltas": pose_deltas,
                "linear_speed_threshold": float(linear_speed_threshold),
                "angular_speed_threshold": float(angular_speed_threshold),
                "position_delta_threshold": float(position_delta_threshold),
                "rotation_delta_threshold": float(rotation_delta_threshold),
            }

    if last_states is None:
        last_states = capture_object_states(env, scene_key_for_object, names)
    return {
        "success": False,
        "steps": max_steps,
        "reason": "settle_timeout_or_missing_object",
        "states": last_states,
        "pose_deltas": pose_deltas,
        "linear_speed_threshold": float(linear_speed_threshold),
        "angular_speed_threshold": float(angular_speed_threshold),
        "position_delta_threshold": float(position_delta_threshold),
        "rotation_delta_threshold": float(rotation_delta_threshold),
    }


def validate_settled_spawn(before, report, *, max_xy_drift=0.120, max_z_change=0.080):
    """Reject missing, non-finite, or truly displaced objects after settling.

    Imported instrument USDs can shift their reported root by 6--8 cm when
    PhysX resolves the authored root to the rigid-body/COM frame.  That stable,
    yaw-dependent offset is preserved in metadata and must not be mistaken for
    objects scattering across the table.
    """
    errors = []
    drift = {}
    after = report.get("states", {})
    if not report.get("success", False):
        errors.append(str(report.get("reason", "settle_failed")))

    for name in SETTLE_OBJECTS:
        pre = before.get(name, {})
        post = after.get(name, {})
        if pre.get("missing") or post.get("missing") or "position_w" not in pre or "position_w" not in post:
            errors.append(f"{name}:missing")
            continue
        p0, p1 = pre["position_w"], post["position_w"]
        values = p0 + p1
        if not all(math.isfinite(float(v)) for v in values):
            errors.append(f"{name}:non_finite_pose")
            continue
        dxy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        dz = p1[2] - p0[2]
        drift[name] = {"xy_m": float(dxy), "z_m": float(dz)}
        if dxy > max_xy_drift:
            errors.append(f"{name}:xy_drift={dxy:.4f}m")
        if abs(dz) > max_z_change:
            errors.append(f"{name}:z_change={dz:.4f}m")

    return len(errors) == 0, errors, drift


def build_settle_metadata(spawn_params, before, report, drift):
    """Create H5-attribute-safe metadata preserving requested and physical poses."""
    requested = {name: spawn_params.get(name) for name in SETTLE_OBJECTS}
    return {
        "spawn_requested_all": json.dumps(requested, sort_keys=True),
        "spawn_before_settle_all": json.dumps(before, sort_keys=True),
        "spawn_settled_all": json.dumps(report.get("states", {}), sort_keys=True),
        "spawn_settle_drift_all": json.dumps(drift, sort_keys=True),
        "settle_success": bool(report.get("success", False)),
        "settle_steps": int(report.get("steps", 0)),
        "settle_reason": str(report.get("reason", "unknown")),
        "settle_linear_speed_threshold": float(report.get("linear_speed_threshold", 0.0)),
        "settle_angular_speed_threshold": float(report.get("angular_speed_threshold", 0.0)),
        "settle_pose_deltas": json.dumps(report.get("pose_deltas", {}), sort_keys=True),
        "settle_position_delta_threshold": float(report.get("position_delta_threshold", 0.0)),
        "settle_rotation_delta_threshold": float(report.get("rotation_delta_threshold", 0.0)),
        "settle_rule": "linear_speed_and_observed_pose_delta; raw_angular_speed_is_diagnostic_only",
        "recording_starts_after_settle": True,
    }
