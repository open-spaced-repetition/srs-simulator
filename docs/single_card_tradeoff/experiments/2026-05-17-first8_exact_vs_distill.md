# First-eight Exact Stationary Finite vs Distill

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How does the evaluated tradeoff change when replacing exact stationary finite policy tables with per-user 476-parameter distills?

## Evidence

Environment `fsrs6`, users 1 through 8. The exact and distill policies are both evaluated against `fsrs6`; a direct same-target time saved row also compares the distill to the exact stationary finite teacher on their shared frontier span. This 2026-05-20 rerun uses the stationary finite four-corner and bilinear interpolation repair.

Source artifacts:
- `first8_exact_vs_distill_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/mean_summary.csv`
- `first8_exact_vs_distill_time_saved_auc`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/regret_auc.csv`

## Results

Exact stationary finite teacher versus per-user distill:

| scheduler | mean same-target time saved AUC vs fsrs6 | mean relative time saved vs fsrs6 | mean coverage vs fsrs6 |
| --- | --- | --- | --- |
| fsrs6_oracle_stationary_finite | 5.1976 | 13.13% | 98.90% |
| fsrs6_oracle_stationary_finite_distill_per_user | 3.1652 | 8.76% | 97.20% |

On the exact-teacher shared span, distill has -2.1209 deck-minutes/day same-target time saved AUC and -5.12% relative time saved at 98.04% coverage.

Visualization outputs:
- `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/same_target_time_saved_auc_by_user.png`
- `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/relative_time_saved_by_user.png`
- `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/span_coverage_by_user.png`
- `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/direct_distill_vs_exact_by_user.png`
- `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/tradeoff_frontiers_first8_users.png`

GPU monitor summary: training peak dedicated memory was 8,607 MiB and exact-vs-distill evaluation peak dedicated memory was 18,310 MiB. Both runs stayed below the 1 GiB shared-memory spill threshold.

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_first8_exact_vs_distill | 5/5 |

## Conclusion

The repaired exact stationary finite table is ahead in this sampled tradeoff evaluation. The per-user distill still improves on `fsrs6` on average, but direct distill-vs-exact relative time saved is negative on the shared span, so the previous non-dominance interpretation no longer holds.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-first8_exact_vs_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/first8_exact_vs_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
