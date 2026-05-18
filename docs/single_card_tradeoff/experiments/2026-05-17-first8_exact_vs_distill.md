# First-eight Exact Stationary Finite vs Distill

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How does the evaluated tradeoff change when replacing exact stationary finite policy tables with per-user 476-parameter distills?

## Evidence

Environment `fsrs6`, users 1 through 8. The exact and distill policies are both evaluated against `fsrs6`; a direct regret row also compares the distill to the exact stationary finite teacher on their shared frontier span.

Source artifacts:
- `first8_exact_vs_distill_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/mean_summary.csv`
- `first8_exact_vs_distill_regret_auc`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/regret_auc.csv`

## Results

Exact stationary finite teacher versus per-user distill:

| scheduler | mean time regret AUC vs fsrs6 | mean relative regret vs fsrs6 | mean coverage vs fsrs6 |
| --- | --- | --- | --- |
| fsrs6_oracle_stationary_finite | -3.2265 | -9.28% | 97.91% |
| fsrs6_oracle_stationary_finite_distill_per_user | -4.9501 | -12.36% | 97.55% |

On the exact-teacher shared span, distill has -1.5077 deck-minutes/day time regret AUC and -3.03% relative regret at 96.60% coverage.

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_first8_exact_vs_distill | 5/5 |

## Conclusion

The per-user distill is not dominated in this sampled tradeoff evaluation: direct distill-vs-exact relative regret is negative on the shared span. The exact table remains the teacher and diagnostic target; the distill is the compact deployable approximation.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-first8_exact_vs_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/first8_exact_vs_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
