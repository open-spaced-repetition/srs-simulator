# Recurrent Interval PPO

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Does the recurrent continuous-interval PPO policy improve the single-card memory-time frontier after the clipped action-space change?

## Evidence

The main row is from the current no-sub-0.5 default comparison. The guide-ablation rows are read from the structured PPO ablation CSVs.

Source artifacts:
- `default_regret_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `uvfa_ppo_rnn_interval_results`: `artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_results.csv`
- `ppo_ablation_summary`: `artifacts/single_card_tradeoff/ppo_ablation/ppo_oracle_warmup_ablation_summary.csv`
- `rnn_no_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_no_guide_regret_auc.csv`
- `rnn_static_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_static_guide_regret_auc.csv`
- `rnn_oracle_warmup_only_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_oracle_warmup_only_regret_auc.csv`

## Results

The default recurrent run used 2,359,296 training transitions and took 284.19s.

### Current clipped tradeoff

| scheduler | params | time_regret_auc | relative_regret | coverage |
| --- | --- | --- | --- | --- |
| uvfa_ppo_rnn_interval | 87,559 | -3.0423 | -21.67% | 69.18% |

### Recurrent guide ablations

| variant | params | time_regret_auc | relative_regret | coverage |
| --- | --- | --- | --- | --- |
| oracle guide | 87,559 | -3.0423 | -21.67% | 69.18% |
| no guide | 87,559 | 0.1070 | 0.93% | 87.79% |
| static guide | 87,559 | -0.3393 | -2.22% | 42.66% |
| oracle warmup only | 87,559 | -3.0183 | -21.31% | 72.55% |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| evaluate_default_no_sub05_tradeoff | 5/5 |

## Conclusion

The recurrent interval policy is useful as a continuous-action baseline, but the current artifact does not dominate the compact stationary finite distill on coverage per parameter.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-recurrent_interval_ppo.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/recurrent_interval_ppo.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
