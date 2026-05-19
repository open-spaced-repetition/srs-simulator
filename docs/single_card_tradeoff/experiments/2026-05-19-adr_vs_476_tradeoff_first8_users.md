# ADR Versus 476-Parameter Distill, First Eight Users

Config: `experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml`

This report uses the Markov-off rerun under
`artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/`.
The older unsuffixed artifact root is legacy Markov-on evidence and should not
be mixed with the results below.

Command:

```bash
uv run python experiments/single_card_tradeoff/run_tradeoff_config.py \
  --config experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml \
  --force
```

## Inputs

- Environment: `fsrs6`
- Users: 1-8
- Review Markov transition: `false`
- Baseline: `fsrs6` desired-retention grid
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off`
- Distill: per-user 476-parameter stationary finite checkpoints under `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu`
- Particles: 10,000
- Days: 1,825
- Button usage: `../Anki-button-usage/button_usage.jsonl`

## Mean Summary

| Scheduler | Markov | Users | Positive users | Mean time saved AUC | Mean relative time saved | Mean span coverage | Min span coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADR | off | 8 | 8 | 6.2885 | 11.40% | 76.75% | 54.45% |
| 476-param distill | off | 8 | 8 | 5.0314 | 12.86% | 97.52% | 90.28% |

ADR is ahead on absolute same-target time saved AUC by `+1.2571`
deck-minutes/day on average, mostly from user 2. The 476-parameter distill has
the stronger coverage profile and higher mean relative time saved, with every
user covered and a minimum span coverage above 90%.

## Per-User Comparison

| User | ADR AUC | Distill AUC | ADR - distill | ADR coverage | Distill coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.9388 | 5.4396 | 0.4992 | 67.84% | 94.94% |
| 2 | 17.6828 | 7.1314 | 10.5515 | 54.45% | 96.12% |
| 3 | 2.0497 | 1.6797 | 0.3699 | 70.63% | 100.00% |
| 4 | 17.6686 | 18.2941 | -0.6255 | 81.73% | 90.28% |
| 5 | 3.7623 | 4.0948 | -0.3325 | 83.64% | 98.83% |
| 6 | 2.6678 | 2.6654 | 0.0024 | 67.58% | 100.00% |
| 7 | 0.1012 | 0.5605 | -0.4593 | 90.58% | 100.00% |
| 8 | 0.4364 | 0.3854 | 0.0510 | 97.52% | 99.96% |

ADR wins users 1, 2, 3, 6, and 8 by absolute AUC, but user 6 is effectively
tied and user 2 contributes most of the aggregate advantage. Distill wins users
4, 5, and 7 and maintains much broader memory-span coverage on every user
except the high-coverage edge of user 8 where both are broad.

The user-2 FSRS6 baseline scale is back to the historical Markov-off level:
DR=0.98 is `721.09` deck-minutes/day in this run, and the 476-parameter distill
returns `+15.03%` relative same-target time saved against FSRS6.

## Visuals

![Same-target time saved AUC by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/same_target_time_saved_auc_by_user.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/span_coverage_by_user.png)

Per-user Pareto plots are under
`artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/user_<id>/results.png`.

## GPU Monitor

Each per-user tradeoff run wrote `gpu_monitor/summary.json`. Shared-memory spill
was `false` for all eight users. The peak summed shared memory ranged from
`176.9` MiB to `185.5` MiB; the highest `nvidia-smi` FB memory peak was
`1,558` MiB.

## Artifacts

- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/combined_results.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/combined_regret_auc.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/summary.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/mean_summary.csv`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/same_target_time_saved_auc_by_user.png`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/relative_time_saved_by_user.png`
- `artifacts/single_card_tradeoff/adr_vs_476_tradeoff_first8_users_markov_off/span_coverage_by_user.png`
