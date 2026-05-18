# Multiuser Evaluation Batching

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Does the multi-user stationary finite distill path have structured smoke evidence for single-process batched training/evaluation?

## Evidence

No full sequential-versus-batched benchmark artifact is available in the current tree. Per the report rule, this section is limited to the existing structured smoke CSVs and does not restate unstructured README timing claims.

Source artifacts:
- `multiuser_eval_batch_smoke_all_train`: `artifacts/single_card_tradeoff/eval_batch_smoke_all/train_summary.csv`
- `multiuser_eval_batch_smoke_group1_train`: `artifacts/single_card_tradeoff/eval_batch_smoke_group1/train_summary.csv`
- `multiuser_eval_batch_smoke_post_patch_train`: `artifacts/single_card_tradeoff/eval_batch_smoke_post_patch/train_summary.csv`

## Results

### Smoke runtime summaries

| artifact | users | teacher_s | train_s | eval_s | params/user |
| --- | --- | --- | --- | --- | --- |
| all | 2 | 0.095 | 0.070 | 0.025 | 395 |
| group1 | 2 | 0.070 | 0.041 | 0.043 | 395 |
| post_patch | 2 | 0.085 | 0.049 | 0.020 | 395 |

## Conclusion

The code path has structured smoke coverage. A full timing claim should be made only after writing a dedicated benchmark artifact.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-multiuser_eval_batching.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/multiuser_eval_batching.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
