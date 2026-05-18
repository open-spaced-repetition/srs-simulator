# PPO Guide Ablation

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

Which guide setup is responsible for the PPO policies' clipped single-card frontier performance?

## Evidence

The table combines current no-sub-0.5 default rows for the oracle-guided defaults with the six structured no-sub-0.5 ablation regret CSVs.

Source artifacts:
- `default_regret_auc`: `artifacts/single_card_tradeoff/no_sub05_distill_compare/regret_auc.csv`
- `ppo_ablation_summary`: `artifacts/single_card_tradeoff/ppo_ablation/ppo_oracle_warmup_ablation_summary.csv`
- `ppo_no_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_no_guide_regret_auc.csv`
- `ppo_static_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_static_guide_regret_auc.csv`
- `ppo_oracle_warmup_only_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_oracle_warmup_only_regret_auc.csv`
- `rnn_no_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_no_guide_regret_auc.csv`
- `rnn_static_guide_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_static_guide_regret_auc.csv`
- `rnn_oracle_warmup_only_regret_auc`: `artifacts/single_card_tradeoff/ppo_ablation/tradeoff_uvfa_ppo_rnn_interval_oracle_warmup_only_regret_auc.csv`

## Results

The default discrete PPO summary reports 2,359,296 training transitions; the default recurrent summary reports 2,359,296.

### Clipped regret-AUC ablations

| variant | params | time_regret_auc | relative_regret | coverage |
| --- | --- | --- | --- | --- |
| ppo oracle guide | 27,148 | -3.4886 | -24.89% | 99.16% |
| ppo no guide | 27,148 | -3.0500 | -22.74% | 85.89% |
| ppo static guide | 27,148 | -1.3626 | -8.52% | 53.95% |
| ppo oracle warmup only | 27,148 | -2.5476 | -16.40% | 64.18% |
| rnn oracle guide | 87,559 | -3.0423 | -21.67% | 69.18% |
| rnn no guide | 87,559 | 0.1070 | 0.93% | 87.79% |
| rnn static guide | 87,559 | -0.3393 | -2.22% | 42.66% |
| rnn oracle warmup only | 87,559 | -3.0183 | -21.31% | 72.55% |

## Conclusion

The oracle guide remains the safest PPO recipe in the current artifact set. Guide choice changes both regret and frontier span, so coverage must be reported with AUC.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-17-ppo_guide_ablation.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/ppo_guide_ablation.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
