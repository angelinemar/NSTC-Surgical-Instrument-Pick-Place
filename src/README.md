# Source map

[Overview](../README.md) · [Recorder](../docs/RECORDER.md)

```mermaid
flowchart LR
    A["Panel / CLI"] --> B["entry"]
    B --> C["Instrument backend"]
    C --> D["Environment + assets"]
    C --> E["Recorder hooks + physical checks"]
    E --> F["Storage transaction"]
    F --> G["Committed H5"]
```

| Folder | Responsibility |
| --- | --- |
| `entry/` | Target-instrument dispatch, preflight, runner |
| `recorder/` | Reset, spawn, control feedback, coverage, storage, status |
| `ui/` | Panel, geometry preview, log display, commands |

Legacy flat module names are grouped under `compat/` and loaded through
`compat/_p4_compat.py`; they are not disposable source duplicates.
Shims preserve import identity and resource roots; tracebacks point to the implementation.

| Boundary | Allowed access |
| --- | --- |
| Expert recorder / QC | Simulator geometry and state for verification |
| Policy inputs | RGB, robot proprioception, operator target command |
| Training labels | Semantic GT, separate from policy inputs |

Run `python scripts/p4.py panel` or `python scripts/p4.py record ...` from the project root.
