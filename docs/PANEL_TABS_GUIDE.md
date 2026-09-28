# P4 panel tabs and coverage markers

Restart the panel after safely stopping an old run to load the new layout:

```powershell
cd C:\IsaacLab\scripts\custom\i4h_project\p4
.\RUNME.ps1 -Mode panel
```

## Pages in one window

1. Workspace & live log: recording settings, the draggable top-down map, per-cell coverage and filtered logs. X/Y/yaw editing is for manual spawn, not grid cycles.
2. Tray & instructions: initial occupancy selection, manual 0/1 counts, fixed slots, workflow and map legend.
3. Recordings & files: open the active run's H5, GIF or diagnostic directories, or browse an older run when idle.

Run controls and counters remain visible when changing pages. Hover buttons for a longer explanation.

## Coverage contract

With the current 10-cell grid, one round requests 10 saved successes: one per cell. Two rounds requests 20: two per cell. The schedule is 0 through 9, repeated for each round. Randomness is within the selected cell (bounded position offset and yaw), plus distractor placement/occupancy. Failed attempts never advance the cell or count as successes. A run blocked at a difficult pose can keep retrying until stopped; a requested goal is not proof of complete coverage.

Blue CURRENT identifies the cell being attempted (or the next cell before starting). Green DONE means its requested round quota has been saved. Teal PARTIAL means some rounds have succeeded but more remain. Every label displays saved/quota, e.g. 1/2 saved. Final coverage is separately audited against the saved H5 metadata.

The markers are Tk panel graphics only, never new objects in the simulator or pixels added to the dataset. The yellow near-base circle has been removed from the display. The existing safety checks have not been removed or relaxed.

## Known motion limitation

Recent scissor and scalpel runs repeatedly failed OPEN_HOVER / LOWER_PRE in near-base cell 4. A timeout is a failed approach waypoint, even before touching the object. It does not establish that the gripper missed a grasp. Some orientations can pass the approach, so neither universal reachability nor universal impossibility is established. This UI update does not modify the arm controller, waypoint timing, scene placement or recording success criteria.
