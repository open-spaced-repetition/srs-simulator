# Stationary Finite Epoch Extension

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Can additional distillation epochs recover the relative time saved and span coverage of low-parameter stationary finite students?

## Evidence

The experiment reruns representative sub-216 architectures for 256 epochs and the full sub-216 set for 512 epochs.

All non-epoch variables stay aligned with the stationary finite distill recipe used for the ablation: five sparse teacher weights (`0,16,64,256,1024`), the clipped 11-action grid, `uniform_table` supervision, 64 steps per epoch, 10,000 evaluation particles, and eval seeds `42,43,44`.

Source artifacts:
- `stationary_finite_model_size_ablation`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/model_size_ablation_summary.csv`
- `stationary_finite_model_size_sub216_ablation`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/sub216_summary.csv`
- `stationary_finite_model_size_sub216_e256_ablation`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/sub216_e256_summary.csv`
- `stationary_finite_model_size_sub216_e512_ablation`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/sub216_e512_all_summary.csv`
- `stationary_finite_model_size_gpu_monitor_summary`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/gpu_monitor/summary.json`

## Results

### Epoch-extension rows

| variant | arch | params | epochs | agreement | relative_time_saved | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| sf_train5_r5d1_e128 | residual:5:1 | 172 | 128 | 71.41% | 20.00% +/- 0.34% | 95.30% +/- 0.29% |
| sf_train5_r5d1_e256 | residual:5:1 | 172 | 256 | 73.46% | 22.93% +/- 0.24% | 98.90% +/- 0.20% |
| sf_train5_r5d1_e512 | residual:5:1 | 172 | 512 | 74.50% | 21.04% +/- 0.17% | 99.27% +/- 0.23% |
| sf_train5_r4d1_e128 | residual:4:1 | 132 | 128 | 69.36% | 19.10% +/- 0.32% | 85.80% +/- 0.36% |
| sf_train5_r4d1_e256 | residual:4:1 | 132 | 256 | 71.67% | 21.45% +/- 0.46% | 92.47% +/- 0.27% |
| sf_train5_r4d1_e512 | residual:4:1 | 132 | 512 | 72.90% | 22.58% +/- 0.29% | 98.43% +/- 0.15% |
| sf_train5_r3d1_e128 | residual:3:1 | 96 | 128 | 51.90% | -74.46% +/- 10.13% | 81.88% +/- 0.30% |
| sf_train5_r3d1_e512 | residual:3:1 | 96 | 512 | 54.09% | -55.93% +/- 1.36% | 79.63% +/- 0.07% |
| sf_train5_mlp8_e128 | mlp:8 | 212 | 128 | 70.13% | 17.47% +/- 0.11% | 98.56% +/- 0.18% |
| sf_train5_mlp8_e256 | mlp:8 | 212 | 256 | 74.31% | 21.39% +/- 0.30% | 98.26% +/- 0.22% |
| sf_train5_mlp8_e512 | mlp:8 | 212 | 512 | 74.46% | 23.22% +/- 0.34% | 98.71% +/- 0.23% |
| sf_train5_mlp6_e128 | mlp:6 | 150 | 128 | 66.39% | 13.71% +/- 0.22% | 96.66% +/- 0.25% |
| sf_train5_mlp6_e512 | mlp:6 | 150 | 512 | 73.75% | 20.15% +/- 0.55% | 98.55% +/- 0.18% |
| sf_train5_mlp4_e128 | mlp:4 | 96 | 128 | 61.17% | 6.60% +/- 0.83% | 89.78% +/- 0.33% |
| sf_train5_mlp4_e512 | mlp:4 | 96 | 512 | 71.96% | 19.41% +/- 0.37% | 98.57% +/- 0.19% |
| sf_train5_linear_e128 | linear | 44 | 128 | 43.15% | -86.56% +/- 2.73% | 98.79% +/- 0.07% |
| sf_train5_linear_e512 | linear | 44 | 512 | 53.76% | 12.84% +/- 0.27% | 94.46% +/- 4.35% |
| sf_train5_quadratic_e128 | quadratic | 110 | 128 | 50.71% | 10.05% +/- 0.63% | 99.36% +/- 0.17% |
| sf_train5_quadratic_e256 | quadratic | 110 | 256 | 59.06% | 16.56% +/- 0.24% | 98.77% +/- 0.20% |
| sf_train5_quadratic_e512 | quadratic | 110 | 512 | 61.11% | 16.60% +/- 0.16% | 94.36% +/- 0.19% |

### Key comparators

| variant | arch | params | epochs | agreement | relative_time_saved | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| sf_train5_r6d1_e128 | residual:6:1 | 216 | 128 | 73.71% | 22.10% +/- 0.33% | 98.37% +/- 0.17% |
| sf_train5_r5d1_e256 | residual:5:1 | 172 | 256 | 73.46% | 22.93% +/- 0.24% | 98.90% +/- 0.20% |
| sf_train5_r4d1_e512 | residual:4:1 | 132 | 512 | 72.90% | 22.58% +/- 0.29% | 98.43% +/- 0.15% |
| sf_train5_mlp8_e512 | mlp:8 | 212 | 512 | 74.46% | 23.22% +/- 0.34% | 98.71% +/- 0.23% |
| sf_train5_r3d1_e512 | residual:3:1 | 96 | 512 | 54.09% | -55.93% +/- 1.36% | 79.63% +/- 0.07% |
| sf_train5_quadratic_e512 | quadratic | 110 | 512 | 61.11% | 16.60% +/- 0.16% | 94.36% +/- 0.19% |

### GPU monitor

| metric | value |
| --- | --- |
| shared memory spill | no |
| peak shared memory | 145.1 MiB |
| peak FB memory | 1631.0 MiB |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| rerun_stationary_finite_sub216_e256_epoch_extension | 7/7 |
| rerun_stationary_finite_sub216_e512_epoch_extension | 11/11 |

## Conclusion

Increasing epochs can recover low-parameter time saved and coverage, but not uniformly. `residual:5:1` reaches 22.93% +/- 0.24% relative time saved and 98.90% +/- 0.20% coverage at 256 epochs, then loses time saved at 512 epochs. `residual:4:1` needs 512 epochs to reach 22.58% +/- 0.29% relative time saved and 98.43% +/- 0.15% coverage, making it the smallest observed candidate that recovers both metrics in this single-train-seed sweep. The best long-epoch time-saved row is `mlp:8` at 212 parameters, 23.22% +/- 0.34% relative time saved, but `mlp:8` is only marginally smaller than the 216-parameter `residual:6:1` baseline. Capacity still matters: `residual:3:1` remains broken at -55.93% +/- 1.36% relative time saved, and `quadratic` loses coverage at 512 epochs. The 132-parameter row needs more train seeds and per-user validation before it should replace the 128-epoch defaults.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-18-stationary_finite_epoch_extension.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/stationary_finite_epoch_extension.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
