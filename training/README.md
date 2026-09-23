# Sensor-only DP and object recognition

## Inputs and labels

The only policy inputs are `robot_proprio` (7 arm joints, 2 finger joints, measured robot-base
EE XYZ and quaternion) and six uint8 224x224 RGB streams: front, wrist, top, left, right, tray.
Object/world pose, grid ID, target slot, expert stage and automatic object ID are excluded.
No GT semantic, conditioning mask or teacher grasp is a policy input.

Eight-class segmentation labels are stored separately in `perception_labels.hdf5`.
Shared RGB features feed the segmentation head and the conditional diffusion network;
labels are used only by cross-entropy. A test changes labels while fixing images/noise and
verifies that diffusion loss is unchanged. Simulator GT remains available in raw diagnostic
files for expert/QC, but the exporter copies only the explicit sensor input allowlist.

## Commands

The current baseline has an automatic completion worker:
`../debug/validation/approach_matrix/finalize_baseline.py`. Its live status is
`../debug/validation/baseline_v11_finalization/status.json`. Do not launch a duplicate
export while this worker is waiting or running. It waits for the benchmark process
to exit, requires all 50 unique combinations to pass (retries allowed), checks the
frozen recorder fingerprint, audits full contents/commits and independent robot FK,
then exports `datasets/baseline_v11` and verifies pick/place with CPU smoke runs.
The expected split is 40 train and 10 validation pairs, with cells 1 and 8 held out.
Each step has its own log under the completion directory. A failed gate stops the
worker with `status=blocked`; it never converts a failed case into success.

`dataset_pipeline_verified=true` means the completed data and computation checks
passed. It does not mean the smoke checkpoints recognize objects or manipulate
reliably. These remain blocked for deployment.

Run from P4 after checking the final 50-case baseline, with a new output directory.
These commands export and smoke-test the same recorder version; do not combine older
cohorts with the v11 demonstrations:

```powershell
& C:\IsaacLab\_isaac_sim\python.bat training\export_sensor_only.py `
  --source debug\test_runs\benchmark_grid_baseline_20260917_v11 `
  --output training\datasets\baseline_v11 --validation-cells 1 8

& C:\IsaacLab\_isaac_sim\python.bat training\train_sensor_policy.py `
  --dataset training\datasets\baseline_v11 `
  --output training\runs\pick_new_smoke --skill pick `
  --steps 3 --batch-size 1 --smoke
```

Repeat the trainer with `--skill place` and a separate output folder. The three-step
commands verify the pipeline only. For actual training, omit `--smoke`, set the training
budget and device explicitly, and assess held-out per-class IoU and policy rollout success.
The 50-case baseline covers instrument/grid combinations, not recognition generalization
over all yaws, lighting, occlusions or within-cell jitter.

Both pick/place segments of an episode receive the same split. Validation cells are held out
before fitting normalization. Chunks never cross policy or episode boundaries. The loader
streams HDF5; it does not cache all images in RAM. Interrupted exports without a complete
manifest are rejected. New recorder files require the episode commit manifest.

The pilot has four training pairs and one validation pair from the older controller cohort.
Keep it separate from v6. The aborted first export was moved to
`../debug/tmp/aborted_sensor_only_export`; the working export is `datasets/sensor_only_v2_pilot_checked`.

## Architecture and timing

RGB CNN + GroupNorm + spatial features, an eight-class segmentation head, and conditional
temporal UNet with DDPM noise prediction. Two observation frames condition 16 future actions.
Train pick/place separately. Actions are absolute robot-base
`[x,y,z,qw,qx,qy,qz,gripper]` at 0.02 seconds per step. Normalization uses training data only.
Quaternion signs are canonicalized in export; decoded predictions are unit-normalized and
gripper commands become -1/+1. Degenerate/nonfinite predictions are rejected.

`runtime.PolicyRuntime.predict()` accepts only the allowlisted sensor dictionary.
`read_live_sensors()` accesses robot FK/encoders and RGB sensors, never instrument prims.
World robot pose is used internally only to express measured robot FK in the base frame;
world coordinates are not neural inputs. Match TCP offset, crop, base frame and control dt
at deployment. Call `observe()` on every 0.02-second control tick, including intermediate
ticks of a cached action prefix, then call `predict()` when replanning. This preserves the
adjacent-frame observation history used in training. Reset history between episodes/skills.
`attach_controller(env)` selects the explicit pick/place controller mode and rejects a
checkpoint trained with an incompatible controller contract. Keep physical QC outside inputs.

## Verified and pending

Pick/place each passed three optimizer steps, held-out batch evaluation, diffusion sampling,
and finite/quaternion/gripper checks. Reports are in `runs/pick_smoke_v1` and
`runs/place_smoke_v1`. The segmentation IoUs are still low. These smoke checkpoints do not
recognize or manipulate reliably. No successful learned-policy physical rollout is claimed.
The runtime rejects smoke checkpoints for deployment unless explicitly allowed for testing.

The current command-free policy assumes one actionable instrument, with the others in the
tray. The exporter enforces this condition from QC metadata, which is not a model input.
If multiple tabletop instruments are eligible and the user requests one in particular,
add an explicit user command or user-selected image crop. Never silently select it using
simulator object ID/pose. Resolve that ambiguity before mixing such episodes into training.

Before scale, finish recorder tests, collect independent examples per class, measure held-out
per-class recognition metrics and closed-loop pick/place success, and check additional
sessions/yaws/occupancies. Consistent files and a finite smoke loss do not establish readiness.
