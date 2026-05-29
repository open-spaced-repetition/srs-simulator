# FSRS6 Cost-ADR Initialization Sensitivity

Date: 2026-05-29

## Question

Measure whether the current compressed Cost-ADR policy search is sensitive to the initialization point when every other experimental setting is held fixed and optimizer/simulator seeds are repeated as matched pairs.

## Design

All runs use users 1-8, FSRS6 environment, Markov off, batched engine, the 15-parameter `fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1` formula, desired-retention head, coefficient bounds `[-64, 64]`, retention bounds `[0.30, 0.995]`, `coefficient_preconditioning = none`, pop16/gen20 CMA-ES, `sigma0 = 1.0`, the same 16 Cost-ADR weights, and the same 16 fixed FSRS6 baseline DR manifest.

Simulator/baseline seed is fixed at 42. Optimizer seeds are matched across initialization conditions: [42, 43, 44].

| condition | initialization point |
| --- | --- |
| `constant_r90` | Hand-crafted constant desired-retention 0.90 starting point. |
| `first8_mean` | Current default built-in first-8 interval-implied-retention mean. |
| `zero` | All coefficients set to zero; generated as policy JSONs. |

The reproducible runner is:

```bash
uv run python experiments/rl_scheduler/run_fsrs6_cost_adr_init_sensitivity.py
```

The analysis summary JSON is:

`artifacts/rl_scheduler/fsrs6_cost_adr_init_sensitivity_users_1_8/analysis/init_sensitivity_summary.json`

## Aggregate Results

| condition | runs | FSRS6 HV delta mean +- std | rel time-save AUC mean +- std | target coverage mean +- std | train final HV mean +- std | gen0 train HV mean +- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `constant_r90` | 3 | 100,494 +- 790 | 12.910 +- 0.445% | 89.477 +- 1.190% | 100,622 +- 970 | -5,704 +- 42,920 |
| `first8_mean` | 3 | 106,543 +- 674 | 13.435 +- 0.237% | 91.965 +- 1.557% | 106,679 +- 337 | 27,018 +- 21,428 |
| `zero` | 3 | 101,427 +- 1,464 | 12.645 +- 0.408% | 88.829 +- 2.070% | 102,142 +- 2,269 | -220,572 +- 133,824 |

## Paired Deltas Versus `first8_mean`

| condition | seeds | HV delta diff mean +- std | rel time-save diff mean +- std | target coverage diff mean +- std | train final HV diff mean +- std | gen0 train HV diff mean +- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `constant_r90` | 3 | -6,048 +- 1,386 | -0.525 +- 0.653 pp | -2.488 +- 2.399 pp | -6,057 +- 1,223 | -32,723 +- 49,040 |
| `zero` | 3 | -5,116 +- 1,211 | -0.790 +- 0.214 pp | -3.136 +- 2.331 pp | -4,537 +- 1,935 | -247,590 +- 152,939 |

## Per-Run Results

| condition | seed | FSRS6 HV delta | rel time-save AUC | target coverage | budget coverage | train final HV | gen0 train HV | passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `constant_r90` | 42 | 100,481 | 12.524% | 89.854% | 81.991% | 100,340 | 16,987 | yes |
| `constant_r90` | 43 | 99,711 | 13.396% | 90.433% | 90.469% | 99,824 | 21,107 | yes |
| `constant_r90` | 44 | 101,291 | 12.810% | 88.144% | 83.862% | 101,701 | -55,207 | yes |
| `first8_mean` | 42 | 106,075 | 13.707% | 90.231% | 74.560% | 106,466 | 4,418 | yes |
| `first8_mean` | 43 | 107,315 | 13.273% | 92.424% | 81.218% | 107,068 | 47,041 | yes |
| `first8_mean` | 44 | 106,237 | 13.325% | 93.241% | 80.753% | 106,502 | 29,595 | yes |
| `zero` | 42 | 101,953 | 13.094% | 88.989% | 90.622% | 101,035 | -66,598 | yes |
| `zero` | 43 | 102,555 | 12.545% | 86.684% | 91.440% | 104,751 | -286,252 | yes |
| `zero` | 44 | 99,772 | 12.297% | 90.815% | 82.858% | 100,639 | -308,865 | yes |

## Interpretation

Initialization is materially relevant if the matched-seed spread is large relative to seed-to-seed noise. In this run, the best aggregate condition is `first8_mean` and the worst is `constant_r90`, with a mean FSRS6 HV spread of 6,048.
Against `first8_mean`, `constant_r90` changes mean HV by -6,048 and relative time-save AUC by -0.525 pp.
Against `first8_mean`, `zero` changes mean HV by -5,116 and relative time-save AUC by -0.790 pp.
Because all conditions use identical users, cost weights, objective, budget, bounds, preconditioning mode, and matched seeds, these deltas are attributable to the initialization point plus normal matched-seed optimizer noise.
