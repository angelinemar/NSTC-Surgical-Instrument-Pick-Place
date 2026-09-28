"""Small, explicit registry for the five verified object-specific backends.

The motion implementation remains frozen in the proven recorder files during
the compatibility phase.  This registry makes target selection declarative and
prevents one object's handler from being selected for another object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectHandler:
    name: str
    object_type_id: int
    backend: str
    stage_prefix: str
    notes: str


HANDLERS = {
    "scalpel": ObjectHandler(
        "scalpel", 0, "phase3_grid_split_scalpel_recorder.py", "SCALPEL",
        "Preserves broad-flat/edge-side pose selection and scalpel center offset."),
    "scissor": ObjectHandler(
        "scissor", 1, "phase3_grid_split_scissor_recorder.py", "SCISSOR",
        "Preserves scissor root pose, grasp orientation, close and lift sequence."),
    "love_retractor": ObjectHandler(
        "love_retractor", 2, "phase3_grid_split_love_retractor_recorder.py", "LOVE",
        "Preserves verified Love body grasp and yaw-dependent orientation handling."),
    "kelly": ObjectHandler(
        "kelly", 3, "phase3_grid_split_kelly_recorder.py", "KELLY",
        "Preserves Kelly spawn height, grasp pose and placement checks."),
    "scalpel_type2": ObjectHandler(
        "scalpel_type2", 4, "phase3_grid_split_scalpel_type2_recorder.py", "SCALPEL_TYPE2",
        "Preserves type-2 spawn root, body alignment and grasp yaw offset."),
}


def get_handler(name: str) -> ObjectHandler:
    key = name.strip().lower()
    try:
        return HANDLERS[key]
    except KeyError as exc:
        choices = ", ".join(HANDLERS)
        raise ValueError(f"Unknown object {name!r}; choose one of: {choices}") from exc

