# RF-DETR B + trained DP: 100 Scissor Pick attempts

**Result: 92/100 successful physical Picks (92%).** 8 did not pass the 10 cm stable-lift gate. 0 had a technically invalid reset.

The frozen plan uses 10 held-out Scissor test scenes × 10 preset XY/yaw target perturbations. XY offsets were sampled independently within ±1.8 cm, with at least 0.5 cm total XY shift; yaw was sampled within ±25°. These are 100 attempts but only 10 independent base scenes.

RF-DETR B detected the target from live front RGB-D; the trained target-relative DP checkpoint generated all Pick motion and OPEN/CLOSE actions. Expert trajectories were not replayed. The simulator object pose was used for reset and scoring, never as a policy input.

Median successful episode length: 130 control steps.

## Results by original held-out scene

| Test scene | Successes / 10 |
| --- | ---: |
| 01 | 6/10 |
| 02 | 10/10 |
| 03 | 10/10 |
| 04 | 10/10 |
| 05 | 10/10 |
| 06 | 10/10 |
| 07 | 10/10 |
| 08 | 8/10 |
| 09 | 10/10 |
| 10 | 8/10 |

## Failed attempts

All eight failures reached the 300-step timeout. The detector returned a Scissor prediction in each; the model also issued a CLOSE command. Peak lift below is the largest observed Scissor rise, which did not satisfy the 10 cm stable-lift gate. These observations do not identify one unique cause.

| Trial | Base scene | First CLOSE step | Peak lift | GIF |
| ---: | ---: | ---: | ---: | --- |
| 006 | 01 | 141 | 7.6 cm | [View](trial_006/rollout_FAIL.gif) |
| 008 | 01 | 147 | 7.0 cm | [View](trial_008/rollout_FAIL.gif) |
| 009 | 01 | 171 | 7.5 cm | [View](trial_009/rollout_FAIL.gif) |
| 010 | 01 | 143 | 8.4 cm | [View](trial_010/rollout_FAIL.gif) |
| 073 | 08 | 84 | 0.5 cm | [View](trial_073/rollout_FAIL.gif) |
| 075 | 08 | 81 | 0.1 cm | [View](trial_075/rollout_FAIL.gif) |
| 092 | 10 | 274 | 4.5 cm | [View](trial_092/rollout_FAIL.gif) |
| 094 | 10 | 251 | 6.7 cm | [View](trial_094/rollout_FAIL.gif) |

## Evidence

- [Fixed plan](plan.json) (including checkpoint hashes and all perturbations)
- [Machine-readable results](summary.json)
- [GIF gallery](GIF_INDEX.html) with every success and failure

## Scope

This is Scissor **Pick only** from held-out recorded Pick starts in Isaac Sim. It does not measure Place, whole-task transfer, other instruments, arbitrary table positions, or physical hospital operation.
