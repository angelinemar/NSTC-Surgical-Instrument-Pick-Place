# P4 feedback-gated recording and control panel

P3 is untouched. P4 uses the existing five object handlers, with shared feedback, camera/data and session hooks. Do not merge old experimental H5 files into a new collection.

## GUI

Run in PowerShell using ordinary Python with Tkinter, not Isaac's Python:

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
python .\control_panel.py
```

1. Choose target instrument, manual/auto, saved skill and requested successful episodes. Maximum attempts must be at least the requested successes.
2. Manual: drag footprints or enter X/Y/yaw in meters/degrees. All five instruments must stay inside the spawn area without overlapping footprint circles.
3. Click **Launch / Prepare next**. Isaac GUI starts, spawns the instruments and waits for settling/camera readiness. No policy samples are collected yet.
4. Inspect the scene. To change positions, edit them and click **Prepare** again; wait for READY again.
5. Click **Start recording**. Target/mode cannot change during a running backend; finish/stop and relaunch to switch handlers.
6. **Discard this attempt** stops the current trajectory, resets it, and excludes it from H5. Prepare the next attempt to continue manual mode.
7. Auto mode samples bounded random placements and continues automatically until the saved success count or attempt limit.

Each launch has a unique folder under `datasets/panel_runs`, containing `console.log`, session/status JSON, metrics, diagnostic failure PNGs, and successful H5 segments. `Save skill=pick` saves only the pick segment; the existing expert still completes physical pick-and-place validation. It does not mean a separate pick-only physical execution mode.

## Success and failure rules

| Check | Behavior |
|---|---|
| LOWER | Must reach the measured pose; stall/timeout is failure, not success. Finger geometry limits the requested depth. |
| CLOSE | Both fingers receive CLOSE immediately. Hold EE position; wait for measured joint-position stability before lifting. Minimum 0.24 s simulation, maximum 0.90 s. |
| Empty lift | After EE rises 20 mm, inadequate object rise for two samples aborts immediately. No travel to tray. |
| Slip | More than 25 mm displacement within the gripper for two samples aborts during lift/transport/lowering. |
| Release | Requires verified lift, held object, footprint inside tray, and valid release height. |
| Final placement | Object must be released, inside tray, at tray height, and stable for six samples. Being at the retreat waypoint alone is not success. |
| Saving | H5 writer requires lift AND final-placement evidence. Success counter increments after `[SAVE SPLIT]`, not after a stage-complete message. |
| Run incomplete | Nonzero process exit. Read the categorized reason in `run_metrics_summary.txt`. |

Mesh measurements use actual mesh vertices in the PhysX link frame, not potentially stale USD `extent` attributes. Franka instance proxies are included. Simulator object poses are expert quality-control evidence only, not additional policy input channels. Training must continue to use an explicit observation allowlist.

No extra micro-lift or hold-after-close stage was added. The lift check runs inside `LIFT_CLEAR`.

Stage names and data formats are shared; contact tolerances need not be identical for different geometries. Love's thin shaft now requires LOWER_GRASP position error below 0.25 mm (others 1 mm) before closing. The tabletop safety height is unchanged. This can add a few convergence samples but is not an idle hold or downward pressing command.

## Automatic CLI test

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\run_all_v2_test.ps1 -Episodes 1 -MaxAttempts 3 -Gui
```

The batch also creates RGB and semantic GIFs unless `-SkipGif` is supplied. Default stride 3 with 60 ms/frame corresponds to the current 20 ms simulation step. Stride 1 with the same default GIF duration plays slower than simulation, not faster.

## Lighting and timing

Shared ambient/key/fill intensities are 400/5000/2500. Instrument materials and camera resolution are unchanged. Stored sensor crops remain 224x224.

Stage logs distinguish simulation seconds from wall-clock seconds and observation-collection time. GPU rendering/semantic/depth collection can make a 0.24-second simulated close take several real seconds. Shortening motion indiscriminately does not remove that rendering cost.

Transport uses cubic easing. Transfers longer than 0.8 m have a lower command-speed cap (0.6 m/s versus 1 m/s for shorter moves) following a reproduced far-side type2 slip. This does not change CLOSE duration or weaken grasp/placement checks. The cap applies to commanded Cartesian speed, not a guarantee of measured joint or EE speed.

## Validation status

See `FEEDBACK_V5_VALIDATION.md` for measured attempts/results. A finite test is not a guarantee of 100% success on arbitrary poses. Failed attempts must remain rejected even when this lowers the collection success rate.
