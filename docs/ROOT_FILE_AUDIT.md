# Root file audit

The root Python files mostly contain three-line compatibility wrappers. The actual
implementations live in `src/`, `env/`, `scripts/tools/` and `tests/legacy/`.
They are not duplicate implementations and should not be edited directly.

## Why the retained entries are needed

| Entry type | Reason to retain |
| --- | --- |
| `phase4_*.py`, shared `phase3_*.py`, `runner.py`, `object_handlers.py` | Active source imports these module names. The wrappers preserve module identity for runtime hooks and test patches. |
| `record.py`, `control_panel.py` | Existing command entry points; `scripts/p4.py` also dispatches through them. |
| H5/GIF/preview/validation tool wrappers | Dynamic `scripts/p4.py tool <name>` dispatch and documented PowerShell launchers resolve these paths. Lack of a static import does not make a CLI tool unused. |
| `test_*.py` | Existing documented `python -m unittest test_...` commands still use these module names. |
| `_p4_compat.py` | Executes the canonical implementation while retaining legacy resource roots. |
| Root PowerShell launchers | Documented recorder, five-instrument test and environment-view commands. |
| Layout JSONs | Hard links to `env/`, not independent configuration copies. |

`ROOT_FILE_AUDIT.json` contains the checked Python entries, canonical paths,
source references and SHA-256 identifiers of reviewed obsolete files.
The reference scan covers source, environment, backends, scripts, tests, training,
validation programs and root launchers. Dynamic CLI dispatch was reviewed manually.

## Obsolete migration files

The following have no runtime, test or validation callers and only performed
completed one-time migrations:

- `prepare_p4.py` and `scripts/tools/prepare_p4.py`: original P3-to-P4 relocation;
  rerunning would overwrite tuned backends and camera configuration.
- `scripts/maintenance/organize_workspace.py`: completed folder migration;
  current imports and commands do not depend on it.

Their deletion does not remove `P3_COPY_MANIFEST.json` or the read-only
`verify_migration` checker. Historical evidence, datasets, assets and regression
scripts are retained: being old or absent from runtime imports is insufficient
evidence that they are unused.
