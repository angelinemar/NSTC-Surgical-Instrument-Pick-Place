> Updated: use /home/user/NSTC/evaluate-dp-model-only6.sh for the requested
> model-only tests. Earlier hybrid commands below are retained for reference.
> Full DDPM sampling is now 100 steps, executing 8 actions per replan. The old
> 20-step sampler was invalid with the installed legacy scheduler.
> Snapshot protocol: 50 test starts per skill x 2 seeds, resets scored separately.

# Hybrid DP skill evaluation

The original FSM handles OPEN_HOVER approach and MOVE_TO_TARGET transfer.
DP Pick handles the whole LOWER_PRE–LIFT_CLEAR window.
DP Place handles the whole LOWER_PLACE–RETREAT window.
All expert actions inside these windows are removed in memory, for evaluation only.
Original recorder backends are unchanged. There is no expert fallback on model failure.
The viewport panel and terminal identify FSM versus DP control.

The same absolute TCP IK controller is retained. DP receives only six RGB images at
448, robot proprioception and a requested instrument class. Ground-truth object
geometry is available only to the original FSM and scorer, never as a DP input.
This is a hybrid skill evaluation, not full-DP autonomy or an RF-DETR benchmark.

## Verification status

Five backend boundaries and failure denominators passed CPU tests.
Pick and provisional Place weights passed actual inference through local pipes.
GUI physics execution is NOT yet validated: production Place training is running.
No training was interrupted and no simulator/driver/PyTorch packages were changed.
The actor uses the original DP environment. The launcher requires completed production
result.json files for both skills before starting any simulator.

## Commands after training finishes

    cd /home/user/NSTC
    ./evaluate-dp6.sh check
    ./evaluate-dp6.sh smoke

Smoke opens two GUI trials (scalpel, grid and free). Validate controller ownership,
physics and logs before interpreting any results. Two trials are not a benchmark.

After verifying that smoke test:

    ./evaluate-dp6.sh run --trials-per-object 10

This evaluates 100 trials: five instruments, two placement modes, ten trials each.
Each attempt uses a fresh simulator process and closes afterward. Failures are never
retried until success. Both modes use paired seeds but different sampled geometry.
Newly generated scenes are not replayed training frames; they are not certified as
a fresh blind benchmark, given earlier analysis of related data.

## Conditions and scoring

Grid uses the original recorder cell sampler, jitter and randomized yaw.
Free samples continuous non-overlapping positions inside the same workspace, with
the same reachable-target constraint and yaw randomization, without cell recentering.
Original lighting, tray occupancy and physical settling checks remain.

Every control tick (0.02 simulated seconds), DP predicts 16 actions using 20 denoising
steps, executes the first, then observes again. Each skill is capped at 300 steps.
Wall-clock playback can be slower than real time.

Lift: center rise >=0.10 m, object bottom >=0.025 m, and 12 stable held frames after
crossing 0.035 m, with <=0.025 m relative drift. This is an explicit evaluation
gate, not a claim of identical recorder waypoint termination.

Place: original slot center/yaw/containment tolerances, bottom within 0.005 m of
support, stable object, both fingers open >0.025 m, TCP >0.06 m above object,
12 consecutive passing frames, and preloaded objects still valid.
The scorer never corrects the model's lowering, release or retreat.

Reports separate:

- Lift successes / trials entering DP Pick.
- Place successes / trials entering DP Place (conditional on model lift and FSM transfer).
- Hybrid chain successes / all attempted scenes.
- Setup failures and transfer failures.
- The same counters per condition and instrument.

Place is not independently initialized: always report its denominator, especially
when Pick fails often. A zero denominator yields no percentage.

Outputs: datasets/DP_evaluation/hybrid_TIMESTAMP/.
Each trial records result.json, controller events.jsonl, model actions.jsonl,
scoring-only object state, checkpoint SHA-256 and logs. plan.json records trial
seeds/conditions, summary.json records aggregates. Infrastructure errors stop the
suite rather than silently counting as model failures.

## Independent 100-per-model protocol

    ./evaluate-dp6.sh run --independent --trials-per-object 10

Pick: original FSM approach, then model Pick; stop at verified lift.
Place: original FSM approach + expert Pick + transfer, then model Place only.
All original FSM preparation actions are labeled FSM and are not credited to the model.

The target is 100 actual model attempts each: 5 instruments × 2 conditions × 10.
Failed FSM preparation gets a new seeded setup, logged separately, with a global
three-times setup budget. A model failure is NEVER replaced. If setup capacity is
exhausted, the run stops and reports the actual denominator, not an invented /100.
Place performance here is conditional on expert-prepared starting states.
The independent summary deliberately does not report a chained DP success rate.

    ./evaluate-dp6.sh smoke --independent

Runs a small integration suite before the full experiment.
