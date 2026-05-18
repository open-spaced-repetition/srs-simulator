# Stationary Finite Compression

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How far can the stationary finite distill be compressed while preserving relative regret and span coverage?

## Evidence

The report uses the multi-seed sparse-cost-weight ablation and the model-size ablation summaries from the current clipped action-space run.

Source artifacts:
- `stationary_finite_cost_weight_ablation`: `artifacts/single_card_tradeoff/stationary_finite_cost_weight_ablation/cost_weight_ablation_multiseed_summary.csv`
- `stationary_finite_model_size_ablation`: `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/model_size_ablation_summary.csv`

## Results

### Teacher cost-weight ablation

| variant | weights | params | relative_regret | coverage |
| --- | --- | --- | --- | --- |
| sf_distill_train12 | 0,1,2,4,8,16,32,64,128,256,512,1024 | 1,452 | -23.69% +/- 0.27% | 94.31% +/- 0.31% |
| sf_distill_train6 | 0,4,16,64,256,1024 | 1,452 | -23.53% +/- 0.22% | 96.88% +/- 0.17% |
| sf_distill_train5 | 0,16,64,256,1024 | 1,452 | -23.67% +/- 0.35% | 95.01% +/- 0.26% |

### Model-size ablation

| variant | arch | params | epochs | agreement | relative_regret | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| sf_train5_r16d2_e128 | residual:16:2 | 1,452 | 128 | 77.45% | -23.00% +/- 0.23% | 96.57% +/- 0.17% |
| sf_train5_r12d2_e128 | residual:12:2 | 900 | 128 | 77.00% | -23.30% +/- 0.33% | 97.60% +/- 0.26% |
| sf_train5_r10d2_e128 | residual:10:2 | 672 | 128 | 75.67% | -23.20% +/- 0.33% | 98.11% +/- 0.25% |
| sf_train5_r8d2_e128 | residual:8:2 | 476 | 128 | 74.98% | -22.15% +/- 0.28% | 98.41% +/- 0.18% |
| sf_train5_r8d1_e128 | residual:8:1 | 316 | 128 | 74.52% | -22.88% +/- 0.45% | 98.53% +/- 0.23% |
| sf_train5_r6d1_e128 | residual:6:1 | 216 | 128 | 73.71% | -22.10% +/- 0.33% | 98.37% +/- 0.17% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| rerun_stationary_finite_model_size_ablation | 6/6 |

## Conclusion

After aligning epochs and evaluation seeds, the 316-parameter `residual:8:1` student recovers frontier span and remains competitive with larger students. The 216-parameter `residual:6:1` student also keeps span coverage, but with weaker relative regret in this rerun.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-stationary_finite_compression.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/stationary_finite_compression.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
