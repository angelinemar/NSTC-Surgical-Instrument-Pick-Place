# Angel / inference

Policy inference. Checkpoint loading, sensor inputs, action prediction and controller integration.

## Workflow

```mermaid
flowchart LR
    A["angel/inference"] --> B["Edit + test"]
    B --> C["Review"]
    C --> D["angel/main"]
```

| Item | Rule |
| --- | --- |
| Scope | Checkpoint loading, sensor inputs, action prediction and controller integration. |
| Guide | [Open workflow guide](docs/INFERENCE.md) |
| Integration | Merge reviewed changes into angel/main |
| Shared source | Baseline dependencies remain available; this is a development branch, not an isolated package |
| Data and secrets | Raw recordings, logs, checkpoints and credentials stay local |
| Runtime safety | Do not switch branches while the recorder is running |

[All branches](docs/BRANCHES.md) | [Recorder](docs/RECORDER.md) | [Training](training/README.md) | [Inference](docs/INFERENCE.md) | [Debug](debug/README.md)

No model accuracy or deployment readiness is implied by this branch name.
