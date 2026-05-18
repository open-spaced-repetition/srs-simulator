# Oracle Policy Outputs

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Which desired-retention actions does the finite-horizon oracle choose under rollout-weighted and table-weighted state sampling?

## Evidence

The old output-distribution artifacts used actions below 0.5, so this report uses the rerun `no_sub05_*` structured CSVs.

Source artifacts:
- `oracle_policy_outputs_rollout`: `artifacts/single_card_tradeoff/analysis/no_sub05_fsrs6_oracle_policy_outputs_rollout_1825_p2048.csv`
- `oracle_policy_outputs_table`: `artifacts/single_card_tradeoff/analysis/no_sub05_fsrs6_oracle_policy_outputs_table_1825.csv`
- `oracle_policy_outputs_rollout_detail`: `artifacts/single_card_tradeoff/analysis/no_sub05_fsrs6_oracle_policy_outputs_rollout_1825_p2048_detail.csv`
- `oracle_policy_outputs_table_detail`: `artifacts/single_card_tradeoff/analysis/no_sub05_fsrs6_oracle_policy_outputs_table_1825_detail.csv`

## Results

### Rollout-weighted modal actions

| weight | modal retention | share | decisions |
| --- | --- | --- | --- |
| 0 | 0.98 | 72.73% | 150,914 |
| 16 | 0.96 | 21.62% | 58,664 |
| 64 | 0.90 | 25.07% | 35,992 |
| 256 | 0.90 | 26.08% | 17,602 |
| 1,024 | 0.50 | 26.44% | 9,594 |

### Table-weighted modal actions

| weight | modal retention | share | decisions |
| --- | --- | --- | --- |
| 0 | 0.98 | 54.25% | 3,735,552 |
| 16 | 0.93 | 19.22% | 3,735,552 |
| 64 | 0.90 | 20.41% | 3,735,552 |
| 256 | 0.50 | 44.13% | 3,735,552 |
| 1,024 | 0.50 | 62.55% | 3,735,552 |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| analyze_oracle_policy_outputs_rollout | 3/3 |
| analyze_oracle_policy_outputs_table | 3/3 |

## Conclusion

Rollout weighting and table weighting expose different parts of the finite oracle, but both show the cost-driven shift toward cheaper actions after the clipped action-space rerun.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-oracle_policy_outputs.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/oracle_policy_outputs.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
