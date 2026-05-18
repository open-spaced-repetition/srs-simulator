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
| sf_train5_r16d2_e64 | residual:16:2 | 1,452 | 64 | 73.76% | -23.67% +/- 0.35% | 95.01% +/- 0.26% |
| sf_train5_r12d2_e64 | residual:12:2 | 900 | 64 | 73.12% | -23.58% | 97.71% |
| sf_train5_r10d2_e64 | residual:10:2 | 672 | 64 | 71.45% | -22.68% | 85.66% |
| sf_train5_r8d2_e64 | residual:8:2 | 476 | 64 | 71.64% | -22.29% | 86.66% |
| sf_train5_r8d2_e128 | residual:8:2 | 476 | 128 | 73.01% | -23.50% +/- 0.25% | 95.27% +/- 0.27% |
| sf_train5_r8d1_e64 | residual:8:1 | 316 | 64 | 69.64% | -21.54% | 73.56% |
| sf_train5_r6d1_e64 | residual:6:1 | 216 | 64 | 69.45% | -19.68% | 73.68% |

## Conclusion

The current compression floor is 476 parameters with longer distillation. Smaller 216-316 parameter variants keep a favorable AUC only over a much narrower frontier span.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-stationary_finite_compression.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/stationary_finite_compression.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
