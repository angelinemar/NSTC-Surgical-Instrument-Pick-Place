"""Validate Phase-3 recorder H5 schemas across objects and pick/place segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


OBJECT_IDS = {
    "scalpel": 0,
    "scissor": 1,
    "love_retractor": 2,
    "kelly": 3,
    "scalpel_type2": 4,
}
EXPECTED_STATE_DIM = 18
EXPECTED_ACTION_DIM = 8
EXPECTED_RESOLUTION = (224, 224)
REQUIRED = {
    "actions",
    "dones",
    "rewards",
    "stage_names",
    "stage_suffixes",
    "step_ids",
    "observations/state",
    "observations/robot_proprio",
    "observations/object_type_id",
    "observations/skill_id",
    "observations/stage_id",
    "observations/front_rgb",
    "observations/front_depth",
    "observations/front_semantic",
    "observations/wrist_rgb",
    "observations/wrist_depth",
    "observations/grip_b_semantic",
    "observations/cam_top_rgb",
    "observations/cam_left_rgb",
    "observations/cam_right_rgb",
    "observations/cam_tray_rgb",
}


def text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def datasets(h5: h5py.File):
    result = {}

    def collect(name, item):
        if isinstance(item, h5py.Dataset):
            result[name] = item

    h5.visititems(collect)
    return result


def infer_object(path: Path, h5: h5py.File) -> str:
    target = text(h5.attrs.get("target_object", ""))
    if target in OBJECT_IDS:
        return target
    for name in OBJECT_IDS:
        if name in path.parts:
            return name
    return "unknown"


def infer_segment(path: Path, h5: h5py.File) -> str:
    skill = text(h5.attrs.get("policy_skill", ""))
    if skill in ("pick", "place"):
        return skill
    return "pick" if "pick_policy" in path.parts else "place"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Dataset root or test run folder")
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(root.rglob("episode_*.h5"))
    if not files:
        raise SystemExit(f"No episode H5 files under: {root}")

    failures = []
    signatures = {}
    print("object          segment steps datasets size_MB state action RGB")
    print("--------------- ------- ----- -------- -------- ----- ------ --------")

    for path in files:
        with h5py.File(path, "r") as h5:
            ds = datasets(h5)
            obj = infer_object(path, h5)
            segment = infer_segment(path, h5)
            steps = int(h5.attrs.get("num_samples", ds.get("actions").shape[0]))
            missing = sorted(REQUIRED - set(ds))
            if missing:
                failures.append(f"{path}: missing topics {missing}")

            for name, item in ds.items():
                per_episode_topics = {
                    "norm_stats/action_mean", "norm_stats/action_std",
                    "norm_stats/state_mean", "norm_stats/state_std",
                    "supervision_gt/teacher_grasp_pose_b",
                }
                if item.ndim and name not in per_episode_topics:
                    if item.shape[0] != steps:
                        failures.append(f"{path}: {name} has {item.shape[0]} rows, expected {steps}")

            state_shape = ds["observations/state"].shape if "observations/state" in ds else ()
            action_shape = ds["actions"].shape if "actions" in ds else ()
            if state_shape != (steps, EXPECTED_STATE_DIM):
                failures.append(f"{path}: state shape {state_shape}, expected {(steps, EXPECTED_STATE_DIM)}")
            if action_shape != (steps, EXPECTED_ACTION_DIM):
                failures.append(f"{path}: action shape {action_shape}, expected {(steps, EXPECTED_ACTION_DIM)}")

            rgb_shape = ds["observations/front_rgb"].shape if "observations/front_rgb" in ds else ()
            if rgb_shape[1:3] not in ((224,224),(448,448)):
                failures.append(f"{path}: unsupported native RGB resolution {rgb_shape[1:3]}")

            if obj in OBJECT_IDS and int(h5.attrs.get("object_type_id", -1)) != OBJECT_IDS[obj]:
                failures.append(f"{path}: wrong object_type_id for {obj}")
            expected_skill = 0 if segment == "pick" else 1
            if int(h5.attrs.get("skill_id", -1)) != expected_skill:
                failures.append(f"{path}: wrong skill_id for {segment}")

            if "stage_suffixes" in ds and steps:
                first = text(ds["stage_suffixes"][0])
                last = text(ds["stage_suffixes"][-1])
                expected_last = "LIFT_CLEAR" if segment == "pick" else "RETREAT"
                expected_first = "OPEN_HOVER" if segment == "pick" else "MOVE_TO_TARGET"
                if first != expected_first or last != expected_last:
                    failures.append(f"{path}: {segment} stages {first} -> {last}, expected {expected_first} -> {expected_last}")

            signature = {name: (item.shape[1:], str(item.dtype)) for name, item in ds.items()}
            signatures.setdefault(segment, []).append((path, signature))
            print(f"{obj:<15} {segment:<7} {steps:>5} {len(ds):>8} {path.stat().st_size / 1048576:>8.1f} "
                  f"{state_shape!s:<5} {action_shape!s:<6} {rgb_shape[1:3]!s}")

    for segment, entries in signatures.items():
        reference_path, reference = entries[0]
        for path, signature in entries[1:]:
            if signature != reference:
                only_ref = sorted(reference.keys() - signature.keys())
                only_cur = sorted(signature.keys() - reference.keys())
                changed = sorted(name for name in reference.keys() & signature.keys() if reference[name] != signature[name])
                failures.append(
                    f"{path}: {segment} schema differs from {reference_path}; "
                    f"missing={only_ref}, extra={only_cur}, changed={changed}"
                )

    print()
    if failures:
        print(f"CONSISTENCY FAIL ({len(failures)} issue(s))")
        for issue in failures:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"CONSISTENCY PASS: {len(files)} H5 file(s), topics/dtypes/shapes/metadata/stages aligned")


if __name__ == "__main__":
    main()
