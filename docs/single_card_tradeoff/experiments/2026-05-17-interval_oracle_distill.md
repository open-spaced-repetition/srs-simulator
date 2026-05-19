# Interval Oracle Distillation

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Can an integer-interval teacher and log-interval student improve the default FSRS-6 single-card frontier?

## Evidence

The interval distill training, tradeoff comparison, and model-size search were rerun to produce structured artifacts for this report.

Source artifacts:
- `interval_distill_results`: `artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill.csv`
- `interval_compare_time_saved_auc`: `artifacts/single_card_tradeoff/oracle_interval_compare/regret_auc.csv`
- `interval_hparam_summary`: `artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_hparam_summary.csv`

## Results

The rerun interval distill final loss is 0.05573, with eval log-interval MAE 0.2466.

### Interval policies vs fsrs6_default

| scheduler | same_target_time_saved_auc | relative_time_saved | coverage |
| --- | --- | --- | --- |
| fsrs6_oracle_interval | 6.3290 | 22.81% | 7.05% |
| fsrs6_oracle_interval_distill | 4.2179 | 30.03% | 100.00% |

### Interval model-scale search

| candidate | network | params | mean scalar | delta vs best | train_s |
| --- | --- | --- | --- | --- | --- |
| res96d3 | residual:96:3 | 57,217 | 0.6770 | -0.0006 | 60.14 |
| res32d2 | residual:32:2 | 4,609 | 0.6764 | -0.0012 | 41.35 |
| res80d2 | residual:80:2 | 26,881 | 0.6749 | -0.0028 | 39.63 |
| res64d2 | residual:64:2 | 17,409 | 0.6729 | -0.0047 | 44.51 |
| res64d1 | residual:64:1 | 8,961 | 0.6719 | -0.0058 | 37.25 |
| res48d1 | residual:48:1 | 5,185 | 0.6706 | -0.0071 | 41.30 |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| train_interval_distill | 2/2 |
| evaluate_interval_oracle_compare | 5/5 |
| search_interval_distill_hparams | 3/3 |

## Conclusion

The interval distill is strong in the default comparison, while the exact interval oracle row covers only the few cost weights solved in this comparison.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-interval_oracle_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/interval_oracle_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
