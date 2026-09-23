# P4 · Surgical Instrument Recording & Sensor-Only DP

<p align="center"><img src="assets/readme/pipeline.svg" alt="Six camera recording pipeline" width="960"></p>
<p align="center"><img src="assets/readme/camera-contract.svg" alt="Six synchronized camera streams" width="960"></p>

## Run

> Replace every `<PLACEHOLDER>` before running it. Do not copy the angle brackets literally.

```powershell
.\RUNME.ps1 -Mode check
.\RUNME.ps1 -Mode objects
.\RUNME.ps1 -Mode record-dry-run -Object <OBJECT_NAME>
.\RUNME.ps1 -Mode panel
.\RUNME.ps1 -Mode validate -RunDirectory <RUN_DIRECTORY>
.\RUNME.ps1 -Mode export -Source <COMPLETED_RUN_DIRECTORY> -Output <NEW_DATASET_DIRECTORY>
.\RUNME.ps1 -Mode train-smoke -Dataset <EXPORTED_DATASET_DIRECTORY> -Skill <pick|place>
```

## What is recorded

| Six synchronized views | Per-view supervision | Robot / task evidence | DP uses |
| --- | --- | --- | --- |
| front · wrist · top · left · right · tray | RGB · depth · semantic class mask · camera calibration | 16-D robot proprioception · 8-D action · success/physical gates · episode commit | RGB + proprioception + robot-base action target |

Simulator object pose, grid cell, target slot, teacher grasp, stage ID, automatic class ID, depth and semantic GT are not policy inputs. Semantic masks are separate labels for the recognition head.

## Dataset gate

<p align="center"><img src="assets/readme/readiness.svg" alt="Dataset readiness" width="960"></p>

The checked `baseline_v11` export contains **50 pick + 50 place** demos, split 40/10 by episode. It is valid for export/training smoke tests, not a production detector or deployable DP policy: it still needs independent sessions, lighting, occlusions, within-cell jitter, held-out class IoU, and closed-loop learned-policy evaluation.

## Repository boundary

Source, configurations, tests and visual docs go to GitHub. Raw H5 recordings, exports, logs, checkpoints and debug output stay local and are ignored.

## Layout

| Folder | Role |
| --- | --- |
| `src/` | active recorder, runtime, UI |
| `backends/` | instrument-specific expert recorders |
| `env/` | scene and camera configuration |
| `training/` | sensor-only exporter, model, runtime |
| `tests/` | contract and regression tests |
| `scripts/` | stable CLI and inspection tools |
| `debug/`, `datasets/` | local only; ignored by Git |

Detailed operational evidence remains under `docs/`; this is the visual GitHub entry point.
