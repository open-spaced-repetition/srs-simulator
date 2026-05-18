# Grid Oracle

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How do exact finite-horizon grid policies compare with the stationary finite policy class on the clipped default frontier?

## Evidence

This report uses the structured regret-AUC output from the exact oracle comparison run.

Source artifacts:
- `grid_oracle_regret_auc`: `artifacts/single_card_tradeoff/results_regret_auc.csv`

## Results

### Exact policy classes vs fsrs6_default

| scheduler | time_regret_auc | relative_regret | coverage |
| --- | --- | --- | --- |
| fsrs6_oracle | -3.4853 | -24.58% | 97.27% |
| fsrs6_oracle_stationary_finite | -3.2328 | -22.77% | 96.85% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_grid_oracle_compare | 2/2 |

## Conclusion

The stationary finite exact policy gives up a small amount of time-regret performance and coverage versus the unrestricted finite-horizon table, but removes the remaining-time policy input.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-grid_oracle.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/grid_oracle.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
