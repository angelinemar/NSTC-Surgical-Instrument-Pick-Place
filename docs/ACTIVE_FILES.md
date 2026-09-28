# Active P4 files

See [README.md](README.md) and [README_P4.md](README_P4.md).
The phase3-prefixed Python modules and five files under `backends/` ARE active:
their names preserve imports, schemas and instrument-specific behavior.
P4 does not import recorder code from the sibling P3 directory.

Only the new hospital scene is spawned. `scene_layout.json` selects its native table.
The legacy `shared_layout.json` is read for compatibility, but P4 overrides the
workspace transforms using `scene_layout.json`.
