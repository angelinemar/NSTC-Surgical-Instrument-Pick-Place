# Observed failure endpoints

Categories identify measured end conditions, not proven root causes.
Each failed trial has exactly one plotted primary category; the JSON also retains all unmet final Place gates.
Invalid resets remain in the scheduled denominator and appear separately.

## Pick

| Observed endpoint | Individual | Joint |
| --- | ---: | ---: |
| CLOSE, lift <35 mm | 22 | 38 |
| No CLOSE command | 12 | 16 |
| Lift 35–100 mm | 3 | 2 |
| RF-DETR pose unavailable | 2 | 2 |

Individual: 61/100 successes; Joint: 42/100 successes.

## Place

| Observed endpoint | Individual | Joint |
| --- | ---: | ---: |
| Center error >8 mm | 4 | 7 |
| TCP not clear | 0 | 3 |
| No OPEN command | 1 | 0 |
| Outside assigned slot | 0 | 1 |

Individual: 95/100 successes; Joint: 89/100 successes.
