# Root layout audit

The repository root is intentionally limited to four public files:

| File | Purpose |
| --- | --- |
| `README.md` | Project overview and navigation |
| `RUNME.ps1` | Supported user launcher |
| `record.py` | Stable direct recorder entry point |
| `.gitignore` | Publication and local-output exclusions |

## Where the previous root files went

| Previous root content | Canonical location |
| --- | --- |
| Recorder and panel implementations | `src/` |
| Scene, camera, and layout configuration | `env/` |
| Legacy flat Python import names | `compat/` |
| Validation and visualization tools | `scripts/tools/` |
| Optional PowerShell launchers | `scripts/launchers/` |
| Unit-test compatibility modules | `tests/legacy/` through `compat/` |
| Guides and historical manifests | `docs/` and `docs/history/` |
| Generated reports | `reporting/` |

The files in `compat/` are small adapters, not second implementations. They
preserve legacy imports used by the five recorder backends while canonical code
stays organized. `scripts/check_structure.py` verifies the mapping and rejects
new loose root files.

Environment JSON files exist only under `env/`; the former root hard links were
removed. Local datasets, logs, checkpoints, and generated debug artifacts are
not publication source.
