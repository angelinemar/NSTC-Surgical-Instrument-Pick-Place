# Assets

This folder contains the local USD, mesh, texture and material dependencies used by P4.
Asset filenames and internal references were not relocated or changed by source organization.
The five instrument backends resolve assets relative to the P4 root. Scene configuration is
under `env`; the active hospital/table choice is defined by `env/scene_layout.json`.

Some historical hospital/table packages remain as dependencies or provenance. Their presence
does not mean they are spawned. Do not remove or rename asset files without validating USD
references and rerunning the environment checks. P3 assets/code were not edited.
