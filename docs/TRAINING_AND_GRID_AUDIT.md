# P4 training-topic audit and grid benchmark

Updated: 2026-09-17T23:47:50

> Historical audit. Counts and architecture descriptions below reflect the audit date, not the current training contract. See [Training](../training/README.md) for the current workflow.

## Interim findings

The complete schema provides data for perception/object recognition and diffusion policy. It does not prove sufficient dataset diversity or success at every robot pose. Physical benchmarks and topic audits are evaluated separately.

Initial inventory: 115 H5 files and 57 split pairs; 98 files passed the topic audit at the time, and 37 pairs passed both segment topic audits and physical-QC metadata checks.

## Model requirements

| Requirement | Topic / contract | Status and usage |
| --- | --- | --- |
| Image input | 6 RGB uint8, T x 224 x 224 x 3 | Select relevant views; wrist aliases are not additional cameras |
| Recognition labels | 6 semantic uint16, T x 224 x 224 | IDs: 0 background, 1 robot, 2 tray, 3..7 instruments |
| RGB-D geometry | depth + intrinsics_output + position_b + quaternion_b_ros | Intrinsics match the crop; Z depth comes from distance_to_image_plane, not radial distance |
| Proprioception | robot_proprio T x 16 / state T x 18 | 7 joints, 2 fingers, EE XYZ+quat; state additionally contains object and skill IDs |
| Target policy | actions T x 8 | absolute robot-base XYZ, quaternion wxyz, gripper -1/+1 |
| Task condition | object_type_id 0..4; skill_id pick=0/place=1 | Not the class-mask labels 3..7 |
| Additional supervision | conditioning_mask_*; teacher_grasp_pose_b | Pick masks identify instruments; place masks identify the tray. Place teacher grasps are intentionally NaN |
| Timing | Row sequence; stage-local step_ids | Older files lack explicit control_dt_s; verify dt from the run/config before conversion |

Do not use debug_gt, semantic GT, supervision_gt, or expert stage_id as policy inputs if unavailable at deployment. Semantic and supervision data are valid perception-loss labels. At the time of this audit, the pipeline excluded simulator target IDs and state18, accepting only RGB + robot_proprio. The exporter required one actionable instrument; explicit target commands were not yet enabled.

## Training compatibility

The inspected `C:/IsaacLab/configs_scissor_dp.json` used only low_dim=[state], rgb=[], depth=[]. This older configuration does not train an image-conditioned diffusion policy. P4 H5 files also need an adapter/export matching the training loader's structure, not just a different dataset path.

The audit found 68 quaternion sign changes between samples. q and -q represent the same orientation; align signs sequentially during export, then normalize predicted quaternions before control. Recorder actions retain their 8-D definition.

Six files had flat RGB across all cameras. Continuous gripper data that fails the binary policy contract may still be useful for perception training: assess perception suitability from RGB/semantics and policy suitability from actions/proprioception/physical QC. Do not use target-object status as the sole class label for multi-object images.

Split train/validation/test by episode/session; paired pick and place segments must remain in the same split. Do not randomly split neighboring frames. Compute normalization statistics from training data only, preserve quaternion sign continuity, and do not let action chunks cross episode or pick/place policy boundaries. This data volume does not prove generalization; evaluate held-out poses, grids, yaws, and visual conditions.

## Historical inventory and coverage

| Instrument | Pairs passing topic + QC metadata checks | Historically recorded grids (IDs 0..9) |
| --- | ---: | --- |
| scalpel | 8 | 0, 1, 2, 3, 5 |
| scissor | 12 | 0, 1, 2, 3 |
| love_retractor | 8 | 1, 6, 7 |
| kelly | 4 | 0, 5, 8 |
| scalpel_type2 | 5 | 1, 3 |

Historical counts mix recorder versions and must not be treated as coverage of the active controller. Initial controller metadata inventory: {'missing': 113, 'tcp_frame_correct_bounded_v4': 2}. No instrument had complete historical evidence across all 10 grids in this inventory.

## Controller benchmark at the time of the audit

Plan: 50 cases; yaw [0.0]. Up to 2 attempts per combination, the default initial wrist pose, and four other objects in the tray. Failures are retained in reports; only successful audited pairs are saved. This does not cover every point within a cell, jitter, scalpel roll, or occupancy.

Completed 50/50; results {'PASS': 50}; active None. All cases complete: True.

First-attempt successes: 48; successes after retry: 2.

Each cell reports PASS/FAIL/ERROR/PENDING counts according to the plan. PASS requires an H5 pair and an audit. PENDING is not success.

| Instrument | Grid 0 | Grid 1 | Grid 2 | Grid 3 | Grid 4 | Grid 5 | Grid 6 | Grid 7 | Grid 8 | Grid 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scalpel | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| scissor | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| love_retractor | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| kelly | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |
| scalpel_type2 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 | 1/0/0/0 |

Machine-readable details: `debug/test_runs/benchmark_grid_baseline_20260917_v11/summary.json`. Do not change the controller/layout during a benchmark; source fingerprints are checked between cases.

## Complete topic audit

The following catalog covers all 76 topics in the scissor v14 schema. T is the segment length; pick and place do not need the same T. Historical schemas, nonfinite values, RGB, depth, semantics, aliases, and calibration are recorded in `validation/training_topic_audit_20260917/file_*.json`.

| Topic | Example shape / dtype | Role |
| --- | --- | --- |
| `actions` | [244, 8] / float32 | Diffusion policy target: absolute XYZ + quaternion wxyz + gripper |
| `camera_calibration/cam_left/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_left/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_left/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_left/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_right/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_right/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_right/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_right/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_top/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_top/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_top/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_top/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_tray/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_tray/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_tray/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/cam_tray/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/front/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/front/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/front/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/front/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/grip_b/intrinsics_output` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/grip_b/intrinsics_raw` | [244, 3, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/grip_b/position_b` | [244, 3] / float32 | RGB-D geometry calibration; not class labels |
| `camera_calibration/grip_b/quaternion_b_ros` | [244, 4] / float32 | RGB-D geometry calibration; not class labels |
| `debug_gt/legacy_state_34d` | [244, 34] / float32 | Simulator diagnostics; must not become policy inputs |
| `debug_gt/object_a_pose_b` | [244, 7] / float32 | Simulator diagnostics; must not become policy inputs |
| `debug_gt/scalpel_pose_b` | [244, 7] / float32 | Simulator diagnostics; must not become policy inputs |
| `debug_gt/selected_object_pose_b` | [244, 7] / float32 | Simulator diagnostics; must not become policy inputs |
| `debug_gt/target_slot_b` | [244, 3] / float32 | Simulator diagnostics; must not become policy inputs |
| `dones` | [244] / bool | Episode boundary / reward placeholder; not success evidence |
| `norm_stats/action_mean` | [8] / float32 | Per-episode statistics; recompute normalization on the training split |
| `norm_stats/action_std` | [8] / float32 | Per-episode statistics; recompute normalization on the training split |
| `norm_stats/state_mean` | [18] / float32 | Per-episode statistics; recompute normalization on the training split |
| `norm_stats/state_std` | [18] / float32 | Per-episode statistics; recompute normalization on the training split |
| `observations/cam_left_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/cam_left_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `observations/cam_left_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/cam_right_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/cam_right_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `observations/cam_right_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/cam_top_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/cam_top_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `observations/cam_top_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/cam_tray_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/cam_tray_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `observations/cam_tray_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/depth_front` | [244, 224, 224] / float16 | Compatibility alias; do not duplicate channels |
| `observations/depth_grip_b` | [244, 224, 224] / float16 | Compatibility alias; do not duplicate channels |
| `observations/depth_wrist` | [244, 224, 224] / float16 | Compatibility alias; do not duplicate channels |
| `observations/front_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/front_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `observations/front_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/grip_b_semantic` | [244, 224, 224] / uint16 | Simulator segmentation / perception classification labels |
| `observations/images_front` | [244, 224, 224, 3] / uint8 | Compatibility alias; do not duplicate channels |
| `observations/images_grip_b` | [244, 224, 224, 3] / uint8 | Compatibility alias; do not duplicate channels |
| `observations/images_wrist` | [244, 224, 224, 3] / uint8 | Compatibility alias; do not duplicate channels |
| `observations/object_type_id` | [244, 1] / float32 | Expert metadata; excluded from sensor-only model inputs |
| `observations/robot_proprio` | [244, 16] / float32 | Measured robot proprioception input |
| `observations/segmentation_map` | [244, 224, 224] / uint16 | Compatibility alias; do not duplicate channels |
| `observations/skill_id` | [244, 1] / float32 | Expert metadata; excluded from sensor-only model inputs |
| `observations/stage_id` | [244, 1] / int32 | Stage analysis; step_ids are local to each stage |
| `observations/state` | [244, 18] / float32 | Legacy state + object/skill IDs; excluded from model inputs |
| `observations/wrist_depth` | [244, 224, 224] / float16 | Optional RGB-D input; mask invalid depth |
| `observations/wrist_rgb` | [244, 224, 224, 3] / uint8 | Policy/perception visual input |
| `rewards` | [244] / float32 | Episode boundary / reward placeholder; not success evidence |
| `stage_names` | [244] / object | Stage analysis; step_ids are local to each stage |
| `stage_suffixes` | [244] / object | Stage analysis; step_ids are local to each stage |
| `step_ids` | [244] / int32 | Stage analysis; step_ids are local to each stage |
| `supervision_gt/conditioning_mask_cam_left` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/conditioning_mask_cam_right` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/conditioning_mask_cam_top` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/conditioning_mask_cam_tray` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/conditioning_mask_front` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/conditioning_mask_grip_b` | [244, 224, 224] / uint8 | Perception/grasp training labels; not GT inputs at inference |
| `supervision_gt/teacher_grasp_pose_b` | [7] / float32 | Perception/grasp training labels; not GT inputs at inference |

## Issues in the initial inventory

Topic-contract failures: {"flat_rgb:cam_left": 6, "flat_rgb:cam_right": 6, "flat_rgb:cam_top": 6, "flat_rgb:cam_tray": 6, "flat_rgb:front": 6, "flat_rgb:grip_b": 6, "gripper_not_binary": 14, "nonfinite:supervision_gt/teacher_grasp_pose_b:7": 1}.

Continuous gripper commands in older data are not file corruption, but differ from the active controller's binary contract. Do not mix them without an explicit conversion/versioning decision. Passing a topic audit does not automatically qualify a file for a production dataset.
