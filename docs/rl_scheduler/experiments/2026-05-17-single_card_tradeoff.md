# Single-card stationary finite tradeoff report

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`

Config: `experiments/single_card_tradeoff/configs/stationary_finite_first8_report.toml`

## Question

Can stationary finite-lifecycle oracle distillation and a much smaller direct policy-search family improve the FSRS-6 memory-time frontier while keeping policy inputs limited to S, D, and cost weight?

## Evidence

This report is generated from the current machine-readable `artifacts/single_card_tradeoff` CSV and JSON outputs. It does not rely on hand-copied metrics.

Available inputs:
- `default_regret_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `default_results`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/results.csv`
- `first8_distill_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/summary.csv`
- `first8_distill_train_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/train_summary.csv`
- `first8_exact_vs_distill_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/mean_summary.csv`
- `first8_exact_vs_distill_regret_auc`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users/regret_auc.csv`
- `low_param_dense_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users_dense_weights/metadata.json`
- `low_param_sparse_metadata`: `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users/metadata.json`

## Default FSRS-6 Comparison

Environment `fsrs6_default`, 1825 days, 10,000 particles, `deck_scale=10000`, no sub-0.5 actions.

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

## First Eight Users

The per-user stationary finite distill trains eight independent 476-parameter students in one batched process (3,808 trainable parameters during training).

| metric | value |
| --- | --- |
| mean span coverage | 97.55% |
| mean time regret AUC | -4.9501 |
| mean relative regret AUC | -12.36% |
| teacher_s | 58.82 |
| train_s | 34.89 |
| eval_s | 66.94 |
| mean final CE | 0.69721 |
| mean table agreement | 72.39% |

Exact stationary finite teacher versus per-user distill:

| scheduler | mean time regret AUC vs fsrs6 | mean relative regret vs fsrs6 | mean coverage vs fsrs6 |
| --- | --- | --- | --- |
| fsrs6_oracle_stationary_finite | -3.2265 | -9.28% | 97.91% |
| fsrs6_oracle_stationary_finite_distill_per_user | -4.9501 | -12.36% | 97.55% |

On the exact-teacher shared span, distill has -1.5077 deck-minutes/day time regret AUC and -3.03% relative regret at 96.60% coverage.

## Low-Parameter Direct Search

| run | params/user | total params | vs fsrs6 relative regret | vs fsrs6 coverage | vs distill relative regret | vs distill coverage |
| --- | --- | --- | --- | --- | --- | --- |
| sparse teacher weights | 7 | 56 | -7.50% | 72.37% | 4.43% | 68.48% |
| dense teacher weights | 7 | 56 | -9.01% | 66.45% | 3.12% | 62.74% |

## Reproduction Profile

The TOML profile records the commands and expected outputs used to reproduce the current report inputs.
CUDA reruns of the tradeoff, multi-user distill, and low-parameter direct-search commands also write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_default_no_sub05_tradeoff | 5/5 |
| train_first8_stationary_finite_distill | 5/5 |
| evaluate_first8_stationary_finite_distill | 4/4 |
| evaluate_first8_exact_vs_distill | 5/5 |
| train_low_param_direct_sparse | 5/5 |
| train_low_param_direct_dense | 5/5 |
| generate_report | 3/3 |

## Conclusions

- The 476-parameter per-user stationary finite distill remains the best compact first-eight-user candidate in these artifacts: it beats fsrs6 on mean relative regret while preserving about 98% coverage.
- The 7-parameter direct policy-search family is useful as a lower-bound compression baseline, but it gives up roughly 28 to 39 coverage points versus the 476-parameter distill.
- For the default single-user FSRS-6 comparison, stationary finite distill trades a small shared-span loss versus unrestricted oracle distill for wider coverage against fsrs6_default.

## Artifacts

- Published report: `docs/rl_scheduler/experiments/2026-05-17-single_card_tradeoff.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
