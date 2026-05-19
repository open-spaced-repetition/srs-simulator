# First-eight Per-user Stationary Finite Distill

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Can independent 476-parameter stationary finite students trained in one batched process improve the first-eight-user FSRS-6 tradeoff?

## Evidence

Environment `fsrs6`, users 1 through 8, one independent student per user, uniform exact-table supervision over `(cost_weight, stability, difficulty)`, and `fsrs6` as the baseline.

Source artifacts:
- `first8_distill_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/summary.csv`
- `first8_distill_train_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/train_summary.csv`

## Results

The per-user stationary finite distill trains eight independent 476-parameter students in one batched process (3,808 trainable parameters during training).

| metric | value |
| --- | --- |
| mean span coverage | 97.55% |
| mean same-target time saved AUC | 4.9501 |
| mean relative time saved AUC | 12.36% |
| teacher_s | 58.82 |
| train_s | 34.89 |
| eval_s | 66.94 |
| mean final CE | 0.69721 |
| mean table agreement | 72.39% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| train_first8_stationary_finite_distill | 5/5 |
| evaluate_first8_stationary_finite_distill | 4/4 |

## Conclusion

Uniform exact-table supervision fixed the high-cost interpolation failure without adding teacher cost weights. The current first-eight artifact keeps about 98% coverage and improves mean relative time saved versus `fsrs6`.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-first8_stationary_finite_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/first8_stationary_finite_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
