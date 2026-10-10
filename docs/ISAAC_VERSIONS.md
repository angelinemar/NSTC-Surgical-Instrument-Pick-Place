# Isaac Sim versions and the separate detector recorder

The unchanged detector source is commit
`a1ea326e73546af4af2eeca66cf2328c074a1f1f` on `angel/detection-recorder`.
Use the explicit version branches to keep runtime changes separate:

| Version | Branch | Workstation checkout |
| --- | --- | --- |
| Original Isaac Sim 5.1 source | `angel/detection-recorder-sim5.1` | `/home/user/NSTC/versions/5.1` |
| Isaac Sim 6.0.1 compatibility port | `angel/detection-recorder-sim6` | `/home/user/NSTC/versions/6.0` |

Both contain the complete project. The detector workflow is `detection/`; it
captures static RGB/masks/COCO, without DP actions, episodes, or H5 recording.
Existing DP code remains a separate entry point. Do not collect both processes
simultaneously on the same GPU.

## Run the detector on this PC

```bash
/home/user/NSTC/start-detection6.sh
```

Or from the Sim 6 checkout:

```bash
./scripts/launchers/run_detection6.sh panel
./scripts/launchers/run_detection6.sh record \
  --output datasets/detection_runs/first_collection \
  --scenes 2 --min-objects 8 --max-objects 18 --ring-cameras 8 --headless
```

Use a new output directory. To continue the same collection, retain its seed and
capture parameters, add `--resume`, and increase `--scenes` to the desired total.
The GUI's **Stop after scene** finishes the current scene before stopping.

For the separate DP panel, run `./scripts/launchers/run_linux6.sh panel`.
`run_linux6.sh train` uses the separate `.venv-dp`; `run_linux6.sh rfdetr` uses
the RF-DETR environment. `P4_RFDETR_PYTHON` can select a different detector Python.
The original workstation checkout is retained with its prior local changes.

## Original mode randomization

| Item | Actual behavior |
| --- | --- |
| Image quality | Native 448×448 DLAA; capture fails if runtime AA mode is not DLAA (`aa_op=4`). |
| Surrounding cameras | 8 ring viewpoints + 1 top by default; configurable 3–16 ring viewpoints. One camera moves sequentially. |
| Camera coverage | Framing includes the spawn workspace and tray; azimuth covers 360°, with random phase/jitter and 40–65° elevation. Pose metadata is checked against the requested location. |
| Table objects | Random count between the configured bounds, currently 5–30. All five classes occur, with random duplicates. |
| Object position | Random continuous XY inside the table workspace and yaw from −180° to 180°; footprint/non-overlap checks remain active. |
| Tray | Random, empty, or full. Tray tools stay in their designated slots. |
| Key/fill lights | Random azimuth, radius 0.25–0.65 m around the aim region, height 1.25–1.75 m; intensity ±25%, color variation, target jitter ±8 cm. |
| Ambient/background | Ambient intensity 320–480; randomized green/teal drape color and roughness. |
| Robot | Visible in 25% of scenes by default, configurable; stationary home pose. |

This is bounded randomization, not unlimited object counts or arbitrary lighting.
It does not stack tools, randomize their physical scale, or guarantee that every
dense arrangement fits. Random seed and scene metadata make each arrangement
reproducible. More viewpoints do not create more independent scenes.

Output is training-only PNG + COCO. `detection/assemble.py` combines it with
separate held-out `valid/` and `test/` detector exports; it does not train a DP
model or silently split neighboring views into evaluation data. See
[the detector documentation](../detection/README.md).

## Wide mode: configurable, with finite capacity

Select **wide** in the detector GUI, or run:

```bash
./scripts/launchers/run_detection6.sh record \
  --output datasets/detection_runs/wide_001 --scenes 100 \
  --randomization wide --min-objects 8 --max-objects 30 \
  --ring-cameras 16 --seed 123 --headless
```

Wide mode permits **1–64 requested table objects** and **3–32 ring views plus
one top view**. The pool allocates five times the requested maximum, so start
with 18–30 maximum objects on this GPU. 64 is a configured safety ceiling, not
a guarantee of capacity. Each scene samples a requested count; if non-overlap
placement runs out of room after reaching the minimum, it records the fitted
count and both requested/actual counts in metadata. If even the minimum cannot
fit, capture fails. It never silently drops the minimum. Counts below five
sample a subset of classes; counts of five or more start with all five classes.
No stacking or physical scale changes are introduced.

Default wide ranges (metres for position, degrees for angles):

```json
{
  "light_radius": [0.1, 1.4],
  "light_height": [0.65, 2.4],
  "light_intensity": [800, 12000],
  "ambient_intensity": [100, 900],
  "light_color": [0.55, 1.0],
  "aim_jitter": 0.25,
  "cone_angle": [50, 90],
  "camera_elevation": [25, 75],
  "camera_distance": [1.0, 1.2]
}
```

Copy `detection/wide.example.json` or save any subset to a JSON file and pass `--randomization-config /path/wide.json`,
or its path in the GUI. Light intensity uses log-uniform sampling; position,
color channels, and other ranges use uniform sampling. Key/fill remain two
lights, now with independent wider positions and intensities. Camera distance
is a multiplier of the conservative full-workspace framing distance; wide
mode still circles 360°. Geometry/room occlusion can hide tools from some views;
COCO boxes describe visible pixels only. Full settings and sampled lighting
are saved. GUI resume reuses the stored profile; CLI resume must pass the same
profile and parameters. Original collections remain resumable unchanged.

## Verification on this PC

Original mode captured and resumed 2 scenes: 18 RGB images and 189 COCO boxes.
The renderer reported DLAA (`aa_op=4`); moving-camera pose checks passed.
Wide mode also completed a GUI-started capture (9 images, 90 boxes), and a
capacity test requested 34 tools and fitted 31 (17 images, 472 boxes). All PNGs
decoded, COCO bounds/class validation passed, and RGB views had distinct hashes.
GUI resume completed a second wide scene and preserved every hash in the first
scene. Six detector unit tests passed. The existing regression suite passed 81 tests
and 53 subtests, excluding
`tests/legacy` and two tests requiring unpublished debug modules
(`test_baseline_finalization.py`, `test_benchmark_accounting.py`).
These are pipeline checks, not a claim of trained RF-DETR accuracy.

## Runtime compatibility

The PC uses RTX 4090, NVIDIA 595.99.02, Ubuntu 26.04, Python 3.12.15,
Isaac Sim 6.0.1.0, and Torch 2.11.0. Driver and boot settings were not changed.
Isaac Sim 5.1 crashes with this driver on this machine; its source is preserved
for a compatible 5.1 workstation, not claimed to run here.

The Sim 6 environment uses a **local compatibility port of Isaac Lab 2.3.2**,
not unmodified Isaac Lab 3. Lab 3 changes the tensor/quaternion APIs used by this
repository. This port and Ubuntu 26.04 are outside the official support matrix.
It is a tested local configuration, not a claim of general production support.

The complete Lab patch is in `scripts/setup/isaaclab-sim6-compat.patch`. Apply it
only to a separate Lab checkout at `37ddf626871758333d6ed89cf64ad702aef127d0`:

```bash
./scripts/setup/apply_isaaclab_sim6_patch.sh /path/to/IsaacLab-v2.3.2
```

It adapts moved PhysX APIs, material commands, sensor scheduling, and camera pose
reads to Kit 110. Do not replace this Lab checkout with Lab 3 using pip. Keep the
editable source at `/home/user/NSTC/IsaacLab-compat6` on this workstation.

The launcher finds `.venv-isaacsim6` in the checkout or either of its two parent
directories. Else set `P4_ISAAC_PYTHON` explicitly. Ubuntu 26.04 additionally uses
the locally extracted libxml2/ICU compatibility libraries under `.tools/compat`;
system libraries were not replaced. Other machines must install the simulator
and its platform prerequisites; copying source alone does not install the runtime.

The detector explicitly requests RGBA semantic palettes on Sim 6, then decodes
them to canonical class IDs before writing masks. Instance/class agreement is
still mandatory. Successful captures exit only after PNGs and atomic JSON
indexes are committed. Failed Sim 6 captures exit nonzero with diagnostics,
preserving pending scenes instead of hanging during native cleanup.


## Installed dependencies and separate training environments

Package inventories are versioned under `scripts/setup/environments/`. These
record the working installations; they are not standalone installers. Simulator
packages use NVIDIA's package index, Torch uses the appropriate CUDA wheels,
and Isaac Lab is the patched editable checkout described above. The DP runtime
uses editable robomimic at `d309eaecc18acf4152a830a895a6984b8ac71b05`.

| Workflow | Existing Python on this PC |
| --- | --- |
| Capture / simulator / export | `/home/user/NSTC/.venv-isaacsim6/bin/python` |
| DP training / inference | `/home/user/NSTC/.venv-dp/bin/python` |
| RF-DETR training | `/home/user/NSTC/NSTC-Surgical-Instrument-Pick-Place/.venv-rfdetr/bin/python` |

Keep these environments separate. DP and RF-DETR use Python 3.11 and Torch
2.7.0/CUDA 12.6; the simulator uses Python 3.12 and Torch 2.11/CUDA 13.
The workstation's existing environments have passed synthetic GPU training
checks. No trained production checkpoint is provided by these setup checks.

For the static recorder's train-only capture, use independently held-out
validation/test exports before training (see `detection/README.md`):

```bash
./scripts/launchers/run_detection6.sh assemble \
  --capture /path/to/static_collection \
  --evaluation /path/to/heldout_detection_export \
  --output /path/to/new_combined_dataset
./scripts/launchers/run_linux6.sh rfdetr \
  --dataset /path/to/new_combined_dataset --check-only
./scripts/launchers/run_linux6.sh rfdetr \
  --dataset /path/to/new_combined_dataset \
  --output training/runs/rfdetr_wide --model small --epochs 50
```
