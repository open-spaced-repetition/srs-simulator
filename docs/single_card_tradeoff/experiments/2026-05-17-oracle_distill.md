# Discrete Oracle Distillation

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How strong is the unrestricted finite-horizon FSRS-6 oracle distillation baseline under the clipped action space?

## Evidence

The tradeoff row is the current clipped default comparison. The model-size facts come from the rerun oracle-distill hparam-search summary.

Source artifacts:
- `default_time_saved_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `oracle_distill_results`: `artifacts/single_card_tradeoff/fsrs6_oracle_distill_results.csv`
- `oracle_distill_hparam_summary`: `artifacts/single_card_tradeoff/fsrs6_oracle_distill_hparam_summary.csv`

## Results

The current compact default is `res16d2` with 1,468 parameters, final CE 0.6043, and train teacher agreement 74.26%.

### Current clipped tradeoff

| scheduler | params | same_target_time_saved_auc | relative_time_saved | coverage |
| --- | --- | --- | --- | --- |
| fsrs6_oracle_distill | 1,468 | 3.6263 | 24.18% | 86.29% |

### Model-scale search

| candidate | network | params | mean scalar | delta vs best | train_s |
| --- | --- | --- | --- | --- | --- |
| res64d2 | residual:64:2 | 18,124 | 0.7848 | -0.0006 | 67.86 |
| res96d3 | residual:96:3 | 58,284 | 0.7838 | -0.0016 | 70.58 |
| res48d2 | residual:48:2 | 10,524 | 0.7836 | -0.0017 | 64.31 |
| res64d3 | residual:64:3 | 26,572 | 0.7835 | -0.0018 | 66.93 |
| res32d2 | residual:32:2 | 4,972 | 0.7831 | -0.0022 | 67.66 |
| res80d3 | residual:80:3 | 40,892 | 0.7826 | -0.0027 | 72.25 |
| res48d3 | residual:48:3 | 15,324 | 0.7821 | -0.0032 | 67.38 |
| res16d2 | residual:16:2 | 1,468 | 0.7816 | -0.0038 | 68.30 |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_default_no_sub05_tradeoff | 5/5 |
| search_oracle_distill_hparams | 3/3 |

## Conclusion

The unrestricted finite-oracle distill remains the compact baseline to beat on the default FSRS-6 single-card frontier.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-oracle_distill.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/oracle_distill.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
