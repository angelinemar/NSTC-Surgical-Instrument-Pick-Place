"""Independent 100-trial Scissor Pick replication with new XY/yaw positions.

This reuses the frozen RF-DETR B and trained Scissor DP checkpoints, the same
ten held-out base scenes, and the same inference seeds by attempt index.
The only planned change is target placement: new offsets are at least 6 mm
away from every original offset for that base scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import benchmark_learned100 as original


OUTPUT = original.ROOT / "datasets/DP_evaluation/scissor_trained_dp_live_100_new_positions_20261104"
PRIOR_PLAN = original.OUTPUT / "plan.json"
SEED = 20261104


def plan() -> dict:
    files = sorted(original.DATA.glob("episode_*.h5"))
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 distinct held-out Scissor scenes, found {len(files)}")
    previous = json.loads(PRIOR_PLAN.read_text())
    if len(previous["attempts"]) != 100:
        raise RuntimeError("Original 100-trial plan is incomplete")
    old_xy: dict[int, list[np.ndarray]] = {scene: [] for scene in range(1, 11)}
    for old in previous["attempts"]:
        old_xy[int(old["scene"])].append(np.array([old["dx_m"], old["dy_m"]]))

    rng = np.random.default_rng(SEED)
    attempts = []
    for scene, snapshot in enumerate(files, 1):
        fresh_xy: list[np.ndarray] = []
        for variation in range(1, 11):
            for _ in range(10000):
                offset = rng.uniform(-0.022, 0.022, size=2)
                if np.linalg.norm(offset) < 0.010:
                    continue
                if any(np.linalg.norm(offset - prior) < 0.006 for prior in old_xy[scene]):
                    continue
                if any(np.linalg.norm(offset - prior) < 0.004 for prior in fresh_xy):
                    continue
                break
            else:
                raise RuntimeError(f"Could not generate distinct offsets for scene {scene}")
            fresh_xy.append(offset)
            attempts.append(dict(
                scene=scene,
                variation=variation,
                snapshot=str(snapshot.resolve()),
                dx_m=float(offset[0]),
                dy_m=float(offset[1]),
                yaw_deg=float(rng.uniform(-28, 28)),
            ))
    return dict(
        contract="trained_dp_rfdetr_b_scissor_pick_100_v1",
        replication="new_positions_20261104",
        seed=SEED,
        base_test_scenes=10,
        variations_per_scene=10,
        planned_attempts=100,
        perturbation="New XY in +/-2.2 cm, shift >=1.0 cm from base; yaw +/-28 deg; >=6 mm XY from all original same-scene offsets",
        matched_inference_seeds="Same per-attempt seeds as original run, to isolate position change",
        previous_plan=str(PRIOR_PLAN),
        pick_checkpoint=str(original.PICK),
        pick_sha256=original.digest(original.PICK),
        detector_checkpoint=str(original.DETECTOR),
        detector_sha256=original.digest(original.DETECTOR),
        attempts=attempts,
    )


if __name__ == "__main__":
    original.plan = plan
    original.OUTPUT = OUTPUT
    original.main()
