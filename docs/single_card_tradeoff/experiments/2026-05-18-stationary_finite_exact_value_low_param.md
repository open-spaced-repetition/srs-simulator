# Stationary Finite Exact-value Low-param Policies

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/reports/stationary_finite_exact_value_low_param.toml`

## Question

Can very low-parameter direct policies approach the stationary finite oracle while keeping high frontier coverage, once Monte Carlo rollout noise is removed?

## Evidence

Environment `fsrs6`, users 1 through 8, 1825 days, clipped 11-action retention grid, and the standard 17 evaluation cost weights. The exact-value evaluator converts each stationary policy to a table on the oracle `(cost_weight, stability, difficulty)` grid and computes finite-lifecycle objective, memory, minutes, reviews, and lapses with DP/occupancy, not rollout particles.

This evaluator is a teacher-gap diagnostic, not a claim that the grid exact policy is always the better deployed policy in the continuous simulator. The exact stationary finite teacher itself is a discrete-grid policy. Distill and direct policies are sampled back onto that same grid for deterministic evaluation, so this controls the objective and removes Monte Carlo noise, but it also removes any continuous-state smoothing benefit a neural policy may have during rollout.

Source artifacts:
- `stationary_finite_exact_value_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_value_first8_users/mean_summary.csv`
- `stationary_finite_exact_value_metadata`: `artifacts/single_card_tradeoff/stationary_finite_exact_value_first8_users/metadata.json`
- `first8_exact_vs_distill_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/mean_summary.csv`
- `first8_exact_vs_distill_time_saved_auc`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/regret_auc.csv`
- `low_param_direct_interaction15_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users_interaction15/metadata.json`
- `low_param_direct_basis32_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users_basis32/metadata.json`
- `low_param_direct_basis64_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users_basis64/metadata.json`

## Results

### Deterministic exact-value comparison

| policy | params/user | vs fsrs6 relative time saved | vs fsrs6 coverage | vs exact relative time saved | vs exact coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact stationary finite | table | +28.81% | 99.03% | 0.00% | 100.00% |
| 476-param distill | 476 | +19.61% | 98.18% | -15.37% | 98.50% |
| residual:4:1 e512 distill | 132 | +17.66% | 96.67% | -17.76% | 96.98% |
| direct 7 sparse | 7 | +9.02% | 71.29% | -20.71% | 66.69% |
| direct 7 dense | 7 | +12.99% | 68.66% | -18.55% | 63.72% |
| direct 15 interaction | 15 | +10.07% | 82.38% | -19.97% | 79.91% |
| direct 32 basis | 32 | +9.15% | 74.97% | -24.37% | 71.35% |
| direct 64 basis | 64 | +5.32% | 77.29% | -31.34% | 72.81% |

### Continuous rollout comparator

| policy | params/user | sampled rollout relative time saved vs fsrs6 | sampled rollout coverage vs fsrs6 |
| --- | ---: | ---: | ---: |
| exact stationary finite | table | +13.13% | 98.90% |
| 476-param distill | 476 | +8.76% | 97.20% |

On the exact-teacher shared span in the sampled rollout comparison, the 476-param distill has -5.12% relative time saved at 98.04% coverage versus exact. This rerun matches the deterministic teacher-gap direction after the stationary finite interpolation repair: the distill remains positive versus `fsrs6` on average, but it no longer appears deployment-competitive with the repaired exact stationary finite table on this sampled comparison.

### Direct-search rollout comparison

| policy | params/user | rollout relative time saved vs fsrs6 | rollout coverage vs fsrs6 |
| --- | ---: | ---: | ---: |
| direct 7 sparse | 7 | +7.50% | 72.37% |
| direct 7 dense | 7 | +9.01% | 66.45% |
| direct 15 interaction | 15 | +7.56% | 79.53% |
| direct 32 basis | 32 | +6.94% | 76.74% |
| direct 64 basis | 64 | +4.64% | 79.75% |

The direct-search rollout runs use continuous desired retention, while exact-value evaluation snaps direct policies to the nearest action in the clipped 11-action grid. The exact-value rows should be used for teacher-gap calibration; rollout rows remain the better deployment-style view for direct policies in the continuous simulator.

### Endpoint check

At `w=1024`, the exact stationary table averages 6762 expected memorized cards at 10.85 minutes/day. The 15-parameter direct policy averages 7220 memorized at 13.05 minutes/day, the 32-parameter policy averages 7342 memorized at 13.18 minutes/day, and the 64-parameter policy averages 7478 memorized at 13.58 minutes/day. The high-cost endpoint is not collapsing to too little memory; these low-param policies instead fail to move far enough toward the low-minute frontier.

### GPU monitor

| run | peak dedicated MiB | peak shared MiB | spill |
| --- | ---: | ---: | --- |
| exact-value evaluator | 23039 | 209.7 | no |
| direct 15 interaction | 2586 | 438.5 | no |
| direct 32 basis | 3048 | 368.2 | no |
| direct 64 basis | 3734 | 377.2 | no |

## Reproduction Profile

The TOML profile records the commands and expected outputs used to reproduce this report input. CUDA runs write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_stationary_finite_exact_value_first8 | 6/6 |
| train_low_param_direct_interaction15 | 5/5 |
| train_low_param_direct_basis32 | 5/5 |
| train_low_param_direct_basis64 | 5/5 |

## Conclusion

The exact-value evaluator resolves the sampled-evaluation ambiguity for the teacher's discrete DP problem: the distills do not exceed the exact stationary finite teacher under deterministic DP evaluation. The repaired sampled rollout comparison now points the same way. The 476-parameter distill remains positive versus `fsrs6`, but it trails the repaired exact stationary finite table on the shared span.

The direct-search capacity curve does not support the hypothesis that 64 or fewer parameters, in these monotone basis families, can closely match the oracle. The 15-parameter monotone interaction family improves coverage over the 7-parameter floor, but only to about 82% exact-value coverage versus `fsrs6` and about 80% versus the exact teacher. The 32- and 64-parameter basis families are not better in this run. Next work should either redesign the low-parameter family around the exact endpoint failures, or move the practical compression target back to the 96-172 parameter long-trained distill range.
