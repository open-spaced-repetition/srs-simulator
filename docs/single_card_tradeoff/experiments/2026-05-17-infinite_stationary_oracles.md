# Infinite And Stationary Oracles

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Does the average-reward infinite oracle transfer well to a finite new-card lifecycle?

## Evidence

The comparison uses finite-lifecycle tradeoff artifacts for the infinite exact/distill and stationary finite exact policies.

Source artifacts:
- `stationary_finite_compare_time_saved_auc`: `artifacts/single_card_tradeoff/stationary_finite_compare/regret_auc.csv`
- `infinite_distill_compare_time_saved_auc`: `artifacts/single_card_tradeoff/infinite_distill_compare/regret_auc.csv`
- `infinite_distill_results`: `artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_results.csv`

## Results

Directly against `fsrs6_oracle_distill`, infinite distill has -13.15% relative time saved over only 6.80% coverage.

### Finite-lifecycle evaluation

| scheduler | same_target_time_saved_auc | relative_time_saved | coverage |
| --- | --- | --- | --- |
| fsrs6_oracle_infinite | 1.9666 | 7.08% | 7.58% |
| fsrs6_oracle_infinite_distill | 5.4734 | 16.42% | 5.34% |
| fsrs6_oracle_stationary_finite | 3.3900 | 27.86% | 78.05% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_infinite_stationary_oracles | 2/2 |

## Conclusion

The average-reward infinite objective is not aligned with the 1825-day new-card lifecycle in the current evaluation; its useful frontier span is narrow after the action floor.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-infinite_stationary_oracles.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/infinite_stationary_oracles.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
