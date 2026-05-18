# First-eight Residual:4:1 Epoch-512 Validation

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Does the 132-parameter `residual:4:1` stationary finite student trained for 512 epochs transfer to the first eight FSRS-6 users?

## Evidence

Environment `fsrs6`, users 1 through 8, one independent student per user, uniform exact-table supervision, 512 epochs, and the same sparse teacher weights, clipped action grid, evaluation weights, and eval particles as the current 476-parameter first-eight baseline.

Source artifacts:
- `first8_r4d1_e512_distill_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_r4d1_e512_uniform_table_supervision_fsrs6_baseline_gpu/summary.csv`
- `first8_r4d1_e512_distill_train_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_r4d1_e512_uniform_table_supervision_fsrs6_baseline_gpu/train_summary.csv`
- `first8_r4d1_e512_distill_gpu_monitor_summary`: `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_r4d1_e512_uniform_table_supervision_fsrs6_baseline_gpu/gpu_monitor/summary.json`
- `first8_r4d1_e512_exact_vs_distill_mean_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users_r4d1_e512/mean_summary.csv`
- `first8_r4d1_e512_exact_vs_distill_regret_auc`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users_r4d1_e512/regret_auc.csv`
- `first8_r4d1_e512_exact_vs_distill_gpu_monitor_summary`: `artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users_r4d1_e512/gpu_monitor/summary.json`

## Results

The validation trains eight independent 132-parameter students (1,056 total trainable parameters during batched training).

### Mean vs fsrs6

| model | params/user | epochs | mean coverage | mean time regret AUC | mean relative regret | mean agreement | mean CE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| residual:8:2 | 476 | 128 | 97.55% | -4.9501 | -12.36% | 72.39% | 0.69721 |
| residual:4:1 | 132 | 512 | 95.52% | -3.9894 | -10.67% | 68.48% | 0.81207 |
| delta r4d1 - r8d2 | -344 | +384 | -2.04% | +0.9607 | +1.68% | -3.92% | +0.11486 |

### Per-user deltas

| user | r4d1 coverage | r4d1 relative regret | coverage delta vs 476 | relative regret delta vs 476 |
| --- | --- | --- | --- | --- |
| 1 | 90.54% | -5.34% | -4.42% | +3.69% |
| 2 | 95.87% | -15.39% | -0.31% | -0.08% |
| 3 | 99.92% | -11.24% | -0.00% | +0.00% |
| 4 | 90.61% | -14.12% | -0.28% | +5.24% |
| 5 | 87.23% | -13.75% | -11.25% | +2.88% |
| 6 | 99.99% | -13.58% | +0.00% | +0.70% |
| 7 | 100.00% | -6.88% | +0.00% | +0.98% |
| 8 | 99.96% | -5.09% | -0.03% | +0.06% |

### Exact-vs-distill check

| scheduler | mean coverage vs fsrs6 | mean time regret AUC vs fsrs6 | mean relative regret vs fsrs6 |
| --- | --- | --- | --- |
| fsrs6_oracle_stationary_finite | 97.91% | -3.2265 | -9.28% |
| fsrs6_oracle_stationary_finite_distill_per_user | 95.52% | -3.9894 | -10.67% |

On the exact-teacher shared span, `residual:4:1` has -0.4721 time regret AUC and -0.91% relative regret at 97.27% coverage.

### GPU monitor

| stage | shared spill | peak shared memory | peak FB memory |
| --- | --- | --- | --- |
| train + fsrs6 eval | no | 179.8 MiB | 7509.0 MiB |
| exact-vs-distill eval | no | 232.5 MiB | 21275.0 MiB |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| train_first8_stationary_finite_r4d1_e512 | 7/7 |
| evaluate_first8_exact_vs_distill_r4d1_e512 | 5/5 |

## Conclusion

`residual:4:1` at 512 epochs does not validate as a drop-in replacement for the first-eight per-user default. It cuts parameters from 476 to 132 per user, but mean coverage changes by -2.04 points and mean relative regret changes by +1.68 points versus the 476-parameter baseline. The largest coverage loss is user 5, where the change is -11.25 points. Treat the 132-parameter result as promising for the single default-user sweep, but not yet robust across users.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-18-first8_stationary_finite_r4d1_e512.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/first8_stationary_finite_r4d1_e512.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
