# Continuous Desired-Retention Distillation

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How does the continuous desired-retention distill compare with the clipped discrete-oracle distill baselines?

## Evidence

The deployment row comes from the current no-sub-0.5 default comparison; training/evaluation loss fields come from the structured retention-distill results CSV.

Source artifacts:
- `default_regret_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `retention_distill_results`: `artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_results.csv`

## Results

The structured result reports final loss 0.4610, eval retention MAE 0.0038, and eval log-interval MAE 1.1386.

### Current clipped tradeoff

| scheduler | params | time_regret_auc | relative_regret | coverage |
| --- | --- | --- | --- | --- |
| fsrs6_oracle_retention_distill | 1,468 | -3.3248 | -21.79% | 83.51% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_default_no_sub05_tradeoff | 5/5 |

## Conclusion

The retention-output student is compact and viable, but in the current clipped comparison it trails the discrete oracle distill and stationary finite distill on coverage.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-retention_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/retention_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
