# Trained Scissor Pick policy with RF-DETR B

**Protocol:** restore a held-out Scissor Pick scene, detect Scissor from the live front camera with RF-DETR B, estimate and lock its pose from RGB-D, then let the *trained* target-relative Diffusion Policy generate all Pick motion and gripper commands. The simulator object pose and expert actions are not policy inputs. Success requires a stable physical hold and a lift of at least 10 cm.

| Split | Distinct scenes evaluated | Physical Pick success | Control steps to success | Smallest measured peak lift |
| --- | ---: | ---: | ---: | ---: |
| Validation | 5 of 20 | 5/5 | 110–147 | 10.3 cm |
| Held-out test | All 10 | 10/10 | 98–231 | 10.1 cm |
| Held-out test with one preset XY/yaw shift per scene | All 10 | 9/10 | 85–170 for successful trials | See trial files |

The gripper's first CLOSE command occurred at steps 75–94 on these validation scenes and 69–170 on the ten test scenes. These values come from the emitted model actions, not expert labels. All 15 result files assert `model_inference_used=true`, `detector_inference_used=true`, `oracle_object_pose_used=false`, and no expert/template replay. Each has a rollout GIF.

**Evidence:** [validation summary](../../datasets/DP_evaluation/scissor_trained_dp_live_valid_20261103/summary.json), [test summary](../../datasets/DP_evaluation/scissor_trained_dp_live_test_20261103/summary.json), [perturbed test summary](../../datasets/DP_evaluation/scissor_trained_dp_live_test_perturbed_20261103/summary.json), [trained Pick checkpoint](../../datasets/DP_training/runs/target_relative_dp_scissor_v1_20261103/checkpoint_best.pt), [RF-DETR B checkpoint](../../worktrees/openvla-object-detection-bridge/models/object_detection/B/checkpoint_best_total.pth), [test GIF gallery](../../datasets/DP_evaluation/scissor_trained_dp_live_test_20261103/GIF_INDEX.html).

The earlier **45/50** result used camera-guided retrieval and transformation of training trajectories; it is a separate diagnostic and must not be counted as success of a trained action model. Its 50 target perturbations are also a different evaluation condition, so its percentage is not directly comparable to the unperturbed 10-scene test above.

The perturbation plan was fixed before these runs: each original test scene received one target shift with X/Y sampled independently from ±1.8 cm and yaw sampled from ±25°. Trial 10 failed at the 300-step timeout. Its target position estimate differed from the simulator scoring pose by about 2.3 mm, but the policy first commanded CLOSE while TCP was about 10 cm from the Scissor. The fingers then closed without grasping and the robot moved farther away. [Failure GIF](../../datasets/DP_evaluation/scissor_trained_dp_live_test_perturbed_20261103/trial_10/rollout_FAIL.gif), [failure result](../../datasets/DP_evaluation/scissor_trained_dp_live_test_perturbed_20261103/trial_10/result.json). This diagnosis uses the simulator pose only after the run for analysis; it was never a policy input.

This result applies to **Scissor Pick at saved held-out Pick starts in Isaac Sim**, with the stated small perturbations. It does not establish Place success, autonomous approach/transport, success for other instruments, or performance in a physical hospital setting.
