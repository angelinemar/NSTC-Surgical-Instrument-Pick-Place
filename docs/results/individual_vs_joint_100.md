# Matched five-instrument local-skill DP comparison

All values below are physical Isaac Sim outcomes from trained checkpoints.
They are not teacher-forced F1, expert replay, or a full Pick-to-Place cycle.

| Policy | Skill | Success / scheduled | Technical invalid |
| --- | --- | ---: | ---: |
| Individual | Pick | 61/100 (61%) | 0 |
| Individual | Place | 95/100 (95%) | 0 |
| Joint | Pick | 42/100 (42%) | 0 |
| Joint | Place | 89/100 (89%) | 0 |

| Instrument | Individual Pick | Joint Pick | Individual Place | Joint Place |
| --- | ---: | ---: | ---: | ---: |
| scalpel | 12/20 | 11/20 | 19/20 | 20/20 |
| scissor | 20/20 | 14/20 | 20/20 | 13/20 |
| love_retractor | 2/20 | 2/20 | 20/20 | 19/20 |
| kelly | 16/20 | 13/20 | 20/20 | 18/20 |
| scalpel_type2 | 11/20 | 2/20 | 16/20 | 19/20 |

Pick uses a live front RGB-D frame and RF-DETR B target-pose estimate.
Place starts from a held-object test snapshot and uses a known calibrated tray slot.
Each class has ten held-out base episodes with two planned variations/seeds.
Two trials from one episode are correlated; this is not 100 independent scenes.
Non-Scissor pose selection was tuned after inspecting two Type 2 scalpel test scenes.
The Type 2 subgroup is developmental rather than blind evidence.
See plan.json, summary.json, individual trial logs, actions, and GIFs for audit.
