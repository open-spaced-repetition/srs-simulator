# Low-parameter Direct Policy Search

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Can a 7-parameter direct desired-retention function optimized by evolutionary search replace the 476-parameter stationary finite distill?

## Evidence

Environment `fsrs6`, users 1 through 8. Both sparse and dense teacher-cost-weight direct-search runs are evaluated against `fsrs6` and against the per-user stationary finite distill.

Source artifacts:
- `low_param_sparse_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users/metadata.json`
- `low_param_dense_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users_dense_weights/metadata.json`

## Results

| run | params/user | total params | vs fsrs6 relative regret | vs fsrs6 coverage | vs distill relative regret | vs distill coverage |
| --- | --- | --- | --- | --- | --- | --- |
| sparse teacher weights | 7 | 56 | -7.50% | 72.37% | 4.43% | 68.48% |
| dense teacher weights | 7 | 56 | -9.01% | 66.45% | 3.12% | 62.74% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| train_low_param_direct_sparse | 5/5 |
| train_low_param_direct_dense | 5/5 |

## Conclusion

The 7-parameter family is informative as a capacity floor, but it does not currently replace the 476-parameter distill: both sparse and dense runs lose substantial frontier coverage.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-low_param_direct_search.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/low_param_direct_search.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
