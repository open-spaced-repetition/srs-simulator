# ADR Versus Exact Stationary Finite Teacher, First Eight Users

Config:
`experiments/single_card_tradeoff/configs/adr_vs_stationary_finite_exact_first8_users.toml`

This rerun checks whether the exact stationary finite teacher, after the
stationary finite corner and bilinear interpolation fixes, can beat the FSRS6 ADR
portfolio on the first eight users.

Formal rerun command:

```bash
uv run python experiments/single_card_tradeoff/run_tradeoff_config.py \
  --config experiments/single_card_tradeoff/configs/adr_vs_stationary_finite_exact_first8_users.toml \
  --force
```

Formal artifacts:
`artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/`

## Setup

- Environment: `fsrs6`
- Users: 1-8
- Review Markov transition: `false`
- Baseline: `fsrs6` desired-retention grid
- Candidate schedulers: `fsrs6_adr` and exact
  `fsrs6_oracle_stationary_finite`
- ADR source:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off`
- Days: 1,825
- Particles: 10,000
- Deck scale: 10,000
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Torch device: CUDA

## Formal Result

Against the FSRS6 desired-retention baseline, the exact stationary finite teacher
beats ADR on mean relative same-target time saved AUC and has much broader
coverage. ADR remains higher on absolute same-target time saved AUC in the
coarse formal run because user 2 is evaluated on a sparse exact scalarization
grid.

| Scheduler | Users | Positive users | Mean time saved AUC | Mean relative time saved | Mean span coverage | Min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADR | 8 | 8 | 6.2885 | 11.40% | 76.75% | 54.45% |
| Exact stationary finite | 8 | 8 | 5.1998 | 13.44% | 99.12% | 95.43% |

| User | ADR rel. AUC | Exact rel. AUC | Exact - ADR | ADR coverage | Exact coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.16% | 8.96% | -0.20 pp | 67.84% | 95.43% |
| 2 | 16.81% | 12.75% | -4.06 pp | 54.45% | 99.90% |
| 3 | 12.30% | 14.89% | +2.59 pp | 70.63% | 99.97% |
| 4 | 17.57% | 17.96% | +0.38 pp | 81.73% | 99.10% |
| 5 | 14.55% | 19.21% | +4.66 pp | 83.64% | 98.58% |
| 6 | 13.42% | 17.57% | +4.14 pp | 67.58% | 100.00% |
| 7 | 1.84% | 9.88% | +8.04 pp | 90.58% | 100.00% |
| 8 | 5.58% | 6.35% | +0.76 pp | 97.52% | 99.96% |

Exact stationary finite wins 6/8 users on relative time saved AUC and 8/8 users
on coverage.

## Direct Pairwise Check

The FSRS6-relative metric is not a scheduler-vs-scheduler ability test because
each scheduler is averaged over its own common span with FSRS6. ADR has narrow
coverage and concentrates in high-memory regions, while exact stationary finite
covers almost the whole FSRS6 span. The direct ADR-baseline rows in
`combined_regret_auc.csv` are more diagnostic.

| User | Exact vs ADR time saved AUC | Relative | Coverage of ADR span |
| ---: | ---: | ---: | ---: |
| 1 | +0.1882 | +0.32% | 100.00% |
| 2 | -4.0910 | -4.67% | 100.00% |
| 3 | +0.3423 | +2.34% | 100.00% |
| 4 | +1.3937 | +1.68% | 100.00% |
| 5 | +1.1714 | +5.30% | 100.00% |
| 6 | +0.6670 | +3.88% | 100.00% |
| 7 | +0.4281 | +7.93% | 100.00% |
| 8 | +0.0717 | +1.11% | 80.50% |

On direct pairwise comparison, exact stationary finite wins 7/8 users in the
formal coarse run. The only substantial miss is user 2.

## User 2 Diagnostic

User 2 is sensitive to scalarization grid density. The formal exact grid is
`0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`, while ADR has 16
portfolio policies spread across the high-memory region. Linear frontier
interpolation therefore penalizes the exact teacher when the frontier bends
sharply between low cost weights.

Two focused user 2 diagnostics were run under:

- `artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/diagnostics/user_2_dense_exact_cost_weights/`
- `artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/diagnostics/user_2_dense_loww_exact_cost_weights/`

The committed reproducibility config for the dense-low-weight diagnostic is:
`experiments/single_card_tradeoff/configs/adr_vs_stationary_finite_exact_user2_dense_loww.toml`

| User 2 run | Exact point count | Exact vs FSRS6 rel. AUC | Exact vs FSRS6 coverage | Exact vs ADR AUC |
| --- | ---: | ---: | ---: | ---: |
| Formal coarse grid | 17 | 12.75% | 99.90% | -4.0910 |
| Dense grid | 33 | 16.12% | 99.84% | -0.3168 |
| Dense low-weight grid | 42 | 17.77% | 99.77% | +1.2916 |

The dense low-weight rerun reverses the user 2 direct comparison. This is strong
evidence that the formal user 2 deficit was mostly an evaluation-grid artifact,
not a policy capability gap.

## Visuals

![Tradeoff frontiers, first eight users](../../../artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/tradeoff_frontiers_first8_users.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/span_coverage_by_user.png)

![User 2 scalarization-grid diagnostic](../../../artifacts/single_card_tradeoff/adr_vs_stationary_finite_exact_first8_users_markov_off/diagnostics/user_2_frontier_grid_diagnostic.png)

## GPU Monitor

All eight formal per-user runs wrote GPU monitor artifacts. Shared-memory spill
was `false` for every user. Across the formal rerun, peak summed shared GPU
memory was about 197 MiB and peak `nvidia-smi` FB memory was 4,598 MiB.

The dense user 2 diagnostics also reported no shared-memory spill:

- Dense grid: peak summed shared memory 216 MiB, peak FB memory 5,268 MiB.
- Dense low-weight grid: peak summed shared memory 200 MiB, peak FB memory
  3,345 MiB.

## Conclusion

After the stationary finite fixes, the exact stationary finite teacher does beat
ADR on the intended relative time saved AUC aggregate and on coverage. The
apparent deficits are not explained by weak teacher policy capacity. They are
mostly caused by evaluation details:

- FSRS6-relative AUC uses scheduler-specific covered spans, so ADR and exact are
  not averaged over the same memory interval.
- The formal exact scalarization grid is too sparse around user 2's high-memory
  frontier bend, while ADR's portfolio has many points in that region.

For future formal comparisons, prefer direct pairwise common-span metrics and use
a denser exact scalarization grid, especially below weight 2.
