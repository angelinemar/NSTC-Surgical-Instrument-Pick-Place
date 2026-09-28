# Angel / training

Dataset export and training. Detector export, DP training, semantic perception and held-out evaluation.

## Workflow

```mermaid
flowchart LR
    A["angel/training"] --> B["Edit + test"]
    B --> C["Review"]
    C --> D["angel/main"]
```

| Item | Rule |
| --- | --- |
| Scope | Detector export, DP training, semantic perception and held-out evaluation. |
| Guide | [Open workflow guide](training/README.md) |
| Integration | Merge reviewed changes into angel/main |
| Shared source | Baseline dependencies remain available; this is a development branch, not an isolated package |
| Data and secrets | Raw recordings, logs, checkpoints and credentials stay local |
| Runtime safety | Do not switch branches while the recorder is running |

[All branches](docs/BRANCHES.md) | [Recorder](docs/RECORDER.md) | [Training](training/README.md) | [Inference](docs/INFERENCE.md) | [Debug](debug/README.md)

No model accuracy or deployment readiness is implied by this branch name.
