# Automatic dataset splits

[Recorder overview](../README.md#automatic-train--valid--test)

Automatic splitting operates on successful episodes within one target run.
A grid cell is a position stratum, not a split: the same cell can contribute
different independently randomized episodes to Train, Valid, and Test.

## Ten grids, ten cycles

One complete cycle saves one successful episode in every cell. Ten cycles
therefore produce 100 episodes (100 Pick/Place pairs when skill is Both).

| Grid cell | Train | Valid | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7 | 2 | 1 | 10 |
| 1 | 7 | 2 | 1 | 10 |
| 2 | 7 | 2 | 1 | 10 |
| 3 | 7 | 2 | 1 | 10 |
| 4 | 7 | 2 | 1 | 10 |
| 5 | 7 | 2 | 1 | 10 |
| 6 | 7 | 2 | 1 | 10 |
| 7 | 7 | 2 | 1 | 10 |
| 8 | 7 | 2 | 1 | 10 |
| 9 | 7 | 2 | 1 | 10 |
| **Total** | **70** | **20** | **10** | **100** |

## Allocation rules

| Situation | Behavior |
| --- | --- |
| First successful episode | Train |
| Second successful episode | Valid |
| Third successful episode | Test |
| Later successes | Fill the largest global 70/20/10 deficit, subject to the current cell's quota |
| Every ten successes in a cell | Exactly seven Train, two Valid, one Test |
| Equal allocation scores | Prefer the split deficient in the target's recorded 90-degree yaw bin |
| Failed attempt | No split quota consumed; retry the grid cell |
| Pick/Place pair | Published together under one split and one commit |
| Resume | Restore committed assignments and continue randomization attempt numbering |
| New target run | Start a new independent allocation ledger |

The allocator chooses among splits with remaining capacity in the cell's
current block of ten. Global deficit is `(successful episodes + 1) * ratio -
saved split count`. The first three successes use the explicit priority above.
Folder names and empty session contracts do not count as recorded examples.

## Small or interrupted collections

| Successfully saved | Train | Valid | Test |
| --- | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 |
| 2 | 1 | 1 | 0 |
| 3 | 1 | 1 | 1 |
| One complete 10-cell cycle | 7 | 2 | 1 |

One cycle has only one example from each grid, so it cannot put every grid in
all three splits. More cycles fill the missing coverage. Ratios between complete
ten-example blocks can be approximate; no existing episode is moved to force a
ratio. Export for the full DP/RF-DETR workflow requires all three splits and a
completed collection. A one-episode collection remains valid raw training data
but is insufficient for that train/validation/test workflow.

## Saved files and inspection

```text
sessions/<timestamp>/<target>/
  capture_contract.json             # split=auto
  session.json
  session.status.json               # dataset_splits.counts and per_grid
  train/
    .commits/
    pick_policy/<target>/episode_*.h5
    place_policy/<target>/episode_*.h5
  valid/                            # same layout
  test/                             # same layout
```

Episode numbers stay global within the run. Gaps within an individual split
are expected. H5 metadata and the checksum commit agree on `dataset_split`,
`split_contract`, `split_cell_id`, and `split_yaw_bin`. All RGB, semantic,
instance, action, and camera streams remain attached to the original episode.

Restart the panel and start a new session to activate this behavior. Existing
fixed-split sessions retain their original split on resume. Existing data is
not migrated or relabeled. Explicit CLI fixed splits remain supported.

## What balance does and does not establish

Grid proportions are guaranteed after complete ten-example blocks. Yaw remains
random, with a tie-break preference; exact yaw, lighting, clutter count, and
occlusion quotas are not guaranteed. Different target types require their own
recordings; this allocator does not automatically collect missing classes.

Automatic splits share a simulated environment and a base session seed, with
different episode randomizations. Export rejects episode-seed leakage across
splits; detector export also rejects identical cross-split images. This measures
generalization to new episodes in the same environment, not to a new hospital,
camera rig, or real-world deployment. Inspect those conditions separately.
