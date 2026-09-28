# Surgical Instrument Pick & Place

An IsaacLab recorder for five surgical instruments, standalone detector datasets, and Diffusion Policy (DP).

[Recorder](docs/RECORDER.md) · [Training](training/README.md) · [Inference](docs/INFERENCE.md) · [Debug](debug/README.md) · [Branch workflow](docs/BRANCHES.md)

## Pipeline

```mermaid
flowchart LR
    A["Control panel"] --> B["Scene + expert motion"]
    B --> C{"Physical + RGB checks"}
    C -->|Fail| D["Discard + log + retry"]
    D --> B
    C -->|Pass| E["Committed raw H5"]
    E --> F["Detector export"]
    E --> G["DP export"]
    F --> H["448 RGB + COCO boxes"]
    H --> L["RF-DETR standalone detector"]
    G --> I["RGB 224 + state + target command"]
    I --> J["Pick / Place models"]
    J --> K["Held-out tests + closed-loop rollout"]
```

**Recording is not training. A successful export does not prove model accuracy.**

## Open the control panel

Replace the placeholder with your checkout location and run from the project root.

```powershell
cd "<PROJECT_DIRECTORY>"
.\RUNME.ps1 -Mode check
.\RUNME.ps1 -Mode panel
```

| Setting | New collection | Meaning |
| --- | --- | --- |
| Save skill | `both` | Save Pick and Place segments |
| Dataset purpose | `Both: DP + detector (recommended)` | One raw recording, two export targets |
| Recorded image size | `448` | More source detail; DP input remains 224 |
| Save raw to | New folder | Do not mix older calibration or resolution |
| Distractors | Selected range | One target; extra duplicates appear only on the table |
| Tray | Random / full / manual | Initial occupants use fixed slots |

Start with a few episodes. Inspect all six cameras and the **exported DP 224 images** before collecting at scale.

## After recording: DP + RF-DETR

Choose **Both: DP + detector** and **448** in the panel. The same raw sessions are exported into two independent training inputs; this does not merge the two models.

```mermaid
flowchart LR
    A["Raw sessions: native 448"] --> B["Audit + session split"]
    B --> C["DP export: Lanczos to 224"]
    B --> D["Detection export: retain 448"]
    C --> E["Separate Pick / Place DP models"]
    D --> F["RF-DETR standalone detector"]
    F --> G["Held-out test mAP / mAR / F1"]
```

```powershell
# 1. Export both products after train, valid, and test sessions are complete.
.\RUNME.ps1 -Mode export -Source "<RAW_COLLECTION>" -Output "<NEW_EXPORT_DIRECTORY>"

# 2. Install RF-DETR once in an isolated Python 3.11 environment.
.\RUNME.ps1 -Mode rfdetr-setup

# 3. Refuse malformed, non-448, or incomplete detector data.
.\RUNME.ps1 -Mode rfdetr-check -Dataset "<NEW_EXPORT_DIRECTORY>\detection"

# 4. Train RF-DETR Small, then evaluate the held-out test split.
.\RUNME.ps1 -Mode rfdetr-train `
  -Dataset "<NEW_EXPORT_DIRECTORY>\detection" `
  -Output "<NEW_RFDETR_RUN_DIRECTORY>" `
  -RFDetrModel small -Epochs 50 -BatchSize 4 -GradAccumSteps 4
```

The final checkpoint is `<NEW_RFDETR_RUN_DIRECTORY>\checkpoint_best_total.pth`; held-out results are saved in `p4_rfdetr_result.json`. Full instructions and GPU notes are in [Training](training/README.md).

## Cameras: raw recordings and model inputs

```mermaid
flowchart LR
    A["6 native 448 cameras"] --> B["Raw H5 without cropping"]
    B --> C["Detector: retain 448"]
    B --> D["Lanczos resize"]
    D --> E["DP: 224 x 224"]
    F["Native 224 option"] --> G["Raw and DP remain 224"]
```

| Camera | RGB dataset in H5 | Purpose |
| --- | --- | --- |
| Front | `observations/front_rgb` | Main work area |
| Wrist / grip | `observations/wrist_rgb` | Detail near the gripper |
| Top | `observations/cam_top_rgb` | Overhead table context |
| Left | `observations/cam_left_rgb` | Left-side view |
| Right | `observations/cam_right_rgb` | Right-side view |
| Tray | `observations/cam_tray_rgb` | Tray area |

PNG previews contain sampled frames; H5 stores every frame within the selected segments.
448 + FXAA improves sampling and edges, but does not guarantee that small objects remain clear at the final 224 resolution.
Resume preserves the original session's camera layout; start a new session to use new framing.

## Policy segments

```mermaid
flowchart LR
    A["OPEN_HOVER: not saved"] --> B["Pick: LOWER_PRE through LIFT_CLEAR"]
    B --> C["MOVE_TO_TARGET: not saved"]
    C --> D["Place: LOWER_PLACE through RETREAT"]
```

| Segment | Saved stages |
| --- | --- |
| Pick | LOWER_PRE, LOWER_GRASP, optional LOWER_EXTRA, CLOSE, LIFT_CLEAR |
| Place | LOWER_PLACE, OPEN, RETREAT |
| Preparation / transfer | Still executed by the controller, excluded from policy data |

## Choose a workflow

| Goal | Guide | Output |
| --- | --- | --- |
| Record / resume | [Recorder](docs/RECORDER.md) | H5, commits, coverage |
| Export / train | [Training](training/README.md) | Datasets and checkpoints |
| Run a model | [Inference](docs/INFERENCE.md) | Action predictions; controller integration required |
| Investigate a problem | [Debug](debug/README.md) | Logs, audits, failure previews |
| Understand the modules | [Source](src/README.md) / [Environment](env/README.md) | Dependencies and configuration |

## Quality status

| Check | What is established | What it does not establish |
| --- | --- | --- |
| H5 / commit / resume | Checksum and consistency checks are implemented | Every attempt succeeds |
| Six cameras / new front view | Scene previews and projection tests were inspected | Every object stays visible during robot motion |
| Older 224 data | Sampled front-view targets were too small | Every dataset is unusable |
| DP 224 export | The resize path is tested | The entire collection has passed visual review |
| Training smoke test | The computation path executes | Recognition accuracy or manipulation success |

Evidence: [training v2 contract](docs/TRAINING_V2.md). Deployment readiness has not been established.

## Structure and publication

| Location | Contents |
| --- | --- |
| `src/`, `backends/`, `env/` | Recorder, panel, configuration |
| `training/` | Exporters, models, runtime |
| `scripts/`, `tests/` | CLI, audits, regression tests |
| `assets/` | Scene and instrument dependencies |
| `docs/` | Guides and historical evidence |
| `compat/` | Legacy flat import names; canonical code remains in the folders above |
| `datasets/`, `debug/`, `training/runs/` | Local output, not source for GitHub |

The root now contains only public launch and repository files. Compatibility
shims are grouped under `compat/`; edit the canonical source instead.
Do not publish raw H5, checkpoints, logs, or credentials. Run `python scripts/check_publish.py`; the scanner cannot guarantee that all secrets are detected.

**Integration branch: `angel/main`.** Develop changes on the appropriate functional branch and review them before merging.
The legacy `main` retains the pre-reorganization code baseline; documentation may receive maintenance updates.
`jordan` has independent history. See the [branch workflow](docs/BRANCHES.md).
