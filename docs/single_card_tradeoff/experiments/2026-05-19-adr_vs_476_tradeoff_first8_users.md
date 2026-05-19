# ADR Versus 476-Parameter Distill, First Eight Users

Config: `experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml`

Command:

```bash
uv run python experiments/single_card_tradeoff/run_tradeoff_config.py \
  --config experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml \
  --force
```

## Inputs

- Environment: `fsrs6`
- Users: 1-8
- Baseline: `fsrs6` desired-retention grid
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1`
- Distill: per-user 476-parameter stationary finite checkpoints under `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu`
- Particles: 10,000
- Days: 1,825
- Button usage: `../Anki-button-usage/button_usage.jsonl`

## Mean Summary

| Scheduler | Users | Positive users | Mean time saved AUC | Mean relative time saved | Mean span coverage | Min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADR | 8 | 7 | 5.9704 | 11.26% | 77.59% | 63.19% |
| 476-param distill | 8 | 8 | 5.9087 | 14.57% | 97.42% | 90.55% |

ADR is slightly ahead on absolute same-target time saved AUC, but the advantage is small (`+0.0616` deck-minutes/day). The 476-parameter distill is clearly more stable: higher mean relative time saved, all eight users positive, and much better coverage of the FSRS6 baseline memory span.

## Per-User Comparison

| User | ADR AUC | Distill AUC | ADR - distill | ADR coverage | Distill coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.2408 | 3.1563 | 1.0845 | 85.91% | 95.30% |
| 2 | 12.5728 | 1.4065 | 11.1663 | 63.19% | 95.00% |
| 3 | 2.1320 | 1.6300 | 0.5020 | 64.61% | 100.00% |
| 4 | 19.5878 | 27.1440 | -7.5562 | 82.80% | 90.55% |
| 5 | 4.5918 | 6.6131 | -2.0213 | 74.58% | 98.54% |
| 6 | 4.2838 | 6.4492 | -2.1655 | 63.43% | 100.00% |
| 7 | -0.0332 | 0.5475 | -0.5807 | 89.51% | 100.00% |
| 8 | 0.3873 | 0.3232 | 0.0641 | 96.71% | 99.97% |

ADR wins users 1, 2, 3, and 8 by absolute AUC, with user 2 accounting for most of the aggregate advantage. Distill wins users 4, 5, 6, and 7, and has consistently broader coverage.

## Visuals

![Same-target time saved AUC by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/same_target_time_saved_auc_by_user.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/span_coverage_by_user.png)

Per-user Pareto plots are under `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/user_<id>/results.png`.

## Artifacts

- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/combined_results.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/combined_regret_auc.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/summary.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/mean_summary.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/same_target_time_saved_auc_by_user.png`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/relative_time_saved_by_user.png`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users/span_coverage_by_user.png`
