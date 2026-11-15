# DP checkpoint registry

Checkpoint names here refer to distinct trained weights. Training loss is not a physical success rate.
The Scissor Pick checkpoint scored 92/100 in each of two position sets. Its evaluation protocol is not a matched comparison with earlier checkpoints.

| ID | Model and measured status | Best epoch | SHA-256 |
| --- | --- | ---: | --- |
| DP-I-P | Original coupled, five-class Pick | 28 | `4fa474e476f41dfb` |
| DP-I-L | Original coupled, five-class Place | 46 | `7f03e922612002d0` |
| DP-II-P | Transition-aware, five-class Pick | 4 | `836dc0740663065f` |
| DP-II-L | Transition-aware, five-class Place | 6 | `e8ba967f59beda4d` |
| DP-III-S-P | Target-relative Scissor Pick; 92/100 in each of two position sets | 98 | `98b20559162f1447` |
| DP-IV-scissor-place | Target-relative scissor place; closed loop pending | 69 | `a7063508fb9768c9` |
| DP-IV-love_retractor-pick | Target-relative love_retractor pick; closed loop pending | 79 | `5cec8f303ea9a8c1` |
| DP-IV-love_retractor-place | Target-relative love_retractor place; closed loop pending | 84 | `09abf7f0e4deef86` |
| DP-IV-scalpel-pick | Target-relative scalpel pick; closed loop pending | 60 | `83d420ccd24120fc` |
| DP-IV-scalpel-place | Target-relative scalpel place; closed loop pending | 90 | `bdf8774ffb34a9e6` |
| DP-IV-kelly-pick | Target-relative kelly pick; closed loop pending | 75 | `829f8987cf78702d` |
| DP-IV-kelly-place | Target-relative kelly place; closed loop pending | 81 | `3fe6851d4abb1b53` |
| DP-IV-scalpel_type2-pick | Target-relative scalpel_type2 pick; closed loop pending | 41 | `967cd5385f01dff3` |
| DP-IV-scalpel_type2-place | Target-relative scalpel_type2 place; closed loop pending | 69 | `97453b7cc037d78b` |
| DP-V-JOINT-PICK | Five-class, class-conditioned target-relative pick; 100-trial benchmark pending | 97 | `b3033c75e7ebfd2d` |
| DP-V-JOINT-PLACE | Five-class, class-conditioned target-relative place; 100-trial benchmark pending | 79 | `16dc9090af3dfc5d` |

## RF-DETR B perception frontend

- Checkpoint: `/home/user/NSTC/worktrees/openvla-object-detection-bridge/models/object_detection/B/checkpoint_best_total.pth`
- SHA-256: `7d97fad639a711e75418d92b5142adf0512f842e8cef8a94d9e8ea068d2bece9`
- Used with live front RGB-D for DP-III-S-P target pose estimation.
- Six-camera detector scores are a separate experiment; the DP-III policy consumes a target pose and robot state.

## Full paths and status

## DP-I-P

- Meaning: Original coupled, five-class Pick
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/dp_pick_all5_rgb448_b24_20261025_025052/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/dp_pick_all5_rgb448_b24_20261025_025052/result.json`
- SHA-256: `4fa474e476f41dfb6329361c6dbd34c2445e5c784ade43e09c450a807a61dc5d`
- Training complete: True
- Closed-loop verified: False

## DP-I-L

- Meaning: Original coupled, five-class Place
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/dp_place_all5_rgb448_b24_20261025_025052/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/dp_place_all5_rgb448_b24_20261025_025052/result.json`
- SHA-256: `7f03e922612002d0659a4bd3ecac460f5da20b139defc96b83c0319f274ada58`
- Training complete: True
- Closed-loop verified: False

## DP-II-P

- Meaning: Transition-aware, five-class Pick
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/gripper_v2_pick_rgb448_b24_20261028_223111/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/gripper_v2_pick_rgb448_b24_20261028_223111/result.json`
- SHA-256: `836dc0740663065f46e5935984f422a5018f183c1fcde8cd9d779fc2d450bd92`
- Training complete: True
- Closed-loop verified: False

## DP-II-L

- Meaning: Transition-aware, five-class Place
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/gripper_v2_place_rgb448_b24_20261028_223111/result.json`
- SHA-256: `e8ba967f59beda4db3fd852a96a84e1983f77a8ead240087b5d8af1a7c4ec2b1`
- Training complete: True
- Closed-loop verified: False

## DP-III-S-P

- Meaning: Target-relative Scissor Pick; 92/100 in each of two position sets
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/result.json`
- SHA-256: `98b20559162f1447b59b3e42bce253c58702589d51bd444bb1de0a60319d1a15`
- Training complete: True
- Closed-loop verified: True
- First position set: `datasets/DP_evaluation/scissor_trained_dp_live_100_20261103/summary.json` (92/100; eight timeouts).
- New position set: `datasets/DP_evaluation/scissor_trained_dp_live_100_new_positions_20261104/summary.json` (92/100; eight timeouts).
- Both use the same ten held-out base scenes and familiar grid cells; the second run changes XY/yaw, not camera viewpoint.

## DP-IV-scissor-place

- Meaning: Target-relative scissor place; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scissor_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scissor_place_v1_20261103/result.json`
- SHA-256: `a7063508fb9768c9e88e6538b0ce7a6ca6a41b9b66536d9c99d7e6678a471633`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scissor/result.json`. This is not a measured class success rate.

## DP-IV-love_retractor-pick

- Meaning: Target-relative love_retractor pick; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_love_retractor_pick_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_love_retractor_pick_v1_20261103/result.json`
- SHA-256: `5cec8f303ea9a8c1e4df3dbb24ded6a78da820944454b95d44c783865a39d6cb`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/love_retractor/result.json`. This is not a measured class success rate.

## DP-IV-love_retractor-place

- Meaning: Target-relative love_retractor place; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_love_retractor_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_love_retractor_place_v1_20261103/result.json`
- SHA-256: `09abf7f0e4deef8681f582de385ec2697087cf7fbc036847c610b862508c348b`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_place_smoke_20261103/love_retractor/result.json`. This is not a measured class success rate.

## DP-IV-scalpel-pick

- Meaning: Target-relative scalpel pick; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_pick_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_pick_v1_20261103/result.json`
- SHA-256: `83d420ccd24120fce2918ad4907ee089b459cb4d83dc95c7550dd3327ef2cd5c`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/scalpel/result.json`. This is not a measured class success rate.

## DP-IV-scalpel-place

- Meaning: Target-relative scalpel place; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_place_v1_20261103/result.json`
- SHA-256: `bdf8774ffb34a9e6efcc1913ce3f1d819ede61f63f48da7d18b67c99dc340323`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel/result.json`. This is not a measured class success rate.

## DP-IV-kelly-pick

- Meaning: Target-relative kelly pick; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_kelly_pick_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_kelly_pick_v1_20261103/result.json`
- SHA-256: `829f8987cf78702db74b9a0498b2db4d60ae6fe018567700e82bd7ecf5196d46`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/kelly/result.json`. This is not a measured class success rate.

## DP-IV-kelly-place

- Meaning: Target-relative kelly place; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_kelly_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_kelly_place_v1_20261103/result.json`
- SHA-256: `3fe6851d4abb1b53a391af41e8a45515b741b0454801641b448d57d0f1ab856a`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_place_smoke_20261103/kelly/result.json`. This is not a measured class success rate.

## DP-IV-scalpel_type2-pick

- Meaning: Target-relative scalpel_type2 pick; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_type2_pick_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_type2_pick_v1_20261103/result.json`
- SHA-256: `967cd5385f01dff377a21f2ce0f3cfb80fbcc1efc2e3b0849735eb67469f235e`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: did not pass; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_pick_smoke_20261103/scalpel_type2/result.json`. This is not a measured class success rate.

## DP-IV-scalpel_type2-place

- Meaning: Target-relative scalpel_type2 place; closed loop pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_type2_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_scalpel_type2_place_v1_20261103/result.json`
- SHA-256: `97453b7cc037d78b2ed53a3ba650c79b9009c446ac26cee2772f7649c4d7fb49`
- Training complete: True
- Closed-loop verified: False
- Single-episode smoke: success; `/home/user/NSTC/datasets/DP_evaluation/target_relative_family_place_smoke_20261103/scalpel_type2/result.json`. This is not a measured class success rate.

## DP-V-JOINT-PICK

- Meaning: Five-class, class-conditioned target-relative pick; 100-trial benchmark pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_joint_pick_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_joint_pick_v1_20261103/result.json`
- SHA-256: `b3033c75e7ebfd2d9e85d8e582d2ad2451d7e181cacc4968b4aafc0fc6b4b173`
- Training complete: True
- Closed-loop verified: False

## DP-V-JOINT-PLACE

- Meaning: Five-class, class-conditioned target-relative place; 100-trial benchmark pending
- Checkpoint: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_joint_place_v1_20261103/checkpoint_best.pt`
- Training result: `/home/user/NSTC/datasets/DP_training/runs/target_relative_dp_joint_place_v1_20261103/result.json`
- SHA-256: `16dc9090af3dfc5d81259e6e1ae31ea836394e67e20da85fa7601b71593417f1`
- Training complete: True
- Closed-loop verified: False
