# Target-relative family: first closed-loop checks (3 November 2026)

These are **single-episode smoke tests**, not final success rates. All runs use
trained checkpoint weights, held-out test snapshots for the skill start, a
300-step limit, and the normal low-level robot controller. Expert skill actions
are not replayed. The Place goal is the class-designated calibrated tray slot;
Pick uses RF-DETR B and live front RGB-D. Snapshot state is used for reset and
simulator ground truth for scoring only.

| Instrument | Pick | Place |
| --- | --- | --- |
| Scalpel | [Success, 138 steps](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/scalpel/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/scalpel/rollout_SUCCESS.gif) | [Success, 145 steps](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel/rollout_SUCCESS.gif) |
| Scissor | [92/100 measured separately](../../datasets/DP_evaluation/scissor_trained_dp_live_100_20261103/RESULTS.md) | [Success, 94 steps](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scissor/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scissor/rollout_SUCCESS.gif) |
| Love retractor | [Success, 124 steps](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/love_retractor/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/love_retractor/rollout_SUCCESS.gif) | [Success, 111 steps](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/love_retractor/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/love_retractor/rollout_SUCCESS.gif) |
| Kelly | [Success, 134 steps](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/kelly/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/kelly/rollout_SUCCESS.gif) | [Success, 90 steps](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/kelly/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/kelly/rollout_SUCCESS.gif) |
| Type 2 scalpel | **Not established.** Initial highest-score mask selected a wrong location; a separate held-out episode had no class detection above the frozen 0.1 threshold. [Initial failed GIF](../../datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/scalpel_type2_highest_confidence_fail/rollout_FAIL.gif). | [Success, 85 steps](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel_type2/result.json) · [GIF](../../datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel_type2/rollout_SUCCESS.gif) |

The Love retractor's first setup attempt failed while building a reference
shape from too-small training masks; it was archived and rerun after the shape
builder skipped those unusable training masks. No expert action was used.

The first Type 2 scalpel Pick failure was a **perception error**: the selected
pose was 45.44 cm from the target object pose used solely for retrospective
audit, and the model issued CLOSE while the TCP was still far away. The target
mask was present at a lower score. A measured-TCP proximity selector reduced
the saved-frame pose error to 2.74 cm in a diagnostic, but the rule was tuned
after inspecting this test case. The case is therefore development evidence,
not an unbiased final test. On another held-out episode, the frozen detector
threshold yielded no Type 2 scalpel candidate, so the DP never acted. We do
not claim Type 2 Pick success. A multi-camera perception study and an untouched
test set are required before reporting its physical success rate.

Place begins with the instrument already held at the recorded Place start.
Five successes in five different classes show that the new runtime wiring can
work; one trial per class is too small to estimate reliability or to claim an
improvement over the earlier five-start Place pilot. Pick and Place were not
connected into a continuous transfer in these runs.
