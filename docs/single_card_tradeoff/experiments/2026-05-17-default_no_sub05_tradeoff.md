# Default No-sub-0.5 Tradeoff

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/stationary_finite_first8_report.toml`

## Question

How do the clipped compact single-card policies compare against `fsrs6_default` when action and target retentions below 0.5 are removed?

## Evidence

Environment `fsrs6_default`, 1825 days, 10,000 particles, `deck_scale=10000`, no sub-0.5 actions.

Source artifacts:
- `default_regret_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `default_results`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/results.csv`

## Results

| scheduler | params | time_regret_auc | relative_regret_auc_percent | coverage |
| --- | --- | --- | --- | --- |
| fsrs6_oracle_distill | 1,468 | -3.6263 | -24.18% | 86.29% |
| fsrs6_oracle_stationary_finite_distill | 1,452 | -3.4070 | -23.70% | 94.67% |
| fsrs6_oracle_retention_distill | 1,468 | -3.3248 | -21.79% | 83.51% |
| fsrs6_oracle_infinite_distill | 1,452 | -4.8242 | -14.66% | 9.01% |
| uvfa_ppo | 27,148 | -3.4886 | -24.89% | 99.16% |
| uvfa_ppo_rnn_interval | 87,559 | -3.0423 | -21.67% | 69.18% |

Direct comparisons against `fsrs6_oracle_distill`:

| scheduler | time_regret_auc | relative_regret_auc_percent | coverage |
| --- | --- | --- | --- |
| fsrs6_oracle_stationary_finite_distill | 0.0944 | 0.83% | 100.00% |
| fsrs6_oracle_retention_distill | 0.3542 | 3.07% | 96.71% |
| fsrs6_oracle_infinite_distill | 3.7877 | 15.59% | 10.44% |
| uvfa_ppo | -0.0639 | -0.56% | 99.91% |
| uvfa_ppo_rnn_interval | 0.1289 | 1.19% | 80.18% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_default_no_sub05_tradeoff | 5/5 |

## Conclusion

The stationary finite distill is slightly worse than unrestricted oracle distill on their shared span, but it covers more of the `fsrs6_default` frontier under the clipped action space.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-default_no_sub05_tradeoff.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/default_no_sub05_tradeoff.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
