# Oracle Four-Corner Grid Comparison, First Eight Users

## Question

Why did `fsrs6_oracle` previously underperform
`fsrs6_oracle_stationary_finite`, even though a finite-horizon non-stationary
policy should dominate a stationary policy in the same MDP?

This experiment reruns the first-eight-user oracle comparison after unifying the
finite grid oracle and the stationary finite seed on the same four-corner
transition kernel.

## Implementation Change

`fsrs6_oracle` now plans with the same four-corner transition approximation used
by `fsrs6_oracle_stationary_finite`:

- Transition backups interpolate next-state value over the four surrounding
  `(log stability, difficulty)` grid corners.
- Initial-state value extraction also uses the four-corner kernel.
- Grid oracle DP cache keys include
  `transition_kernel=four_corner_log_s_linear_d_v1`, so old nearest-grid cache
  entries are not reused.
- `fsrs6_oracle_stationary_finite` now reuses `FSRS6GridOracle.solve_policies()`
  for its finite seed instead of maintaining a separate duplicate finite-DP
  implementation.

The execution-time policy lookup is unchanged: rollout still maps the current
continuous `(S, D)` state to a policy-table action by nearest grid point.

## Configuration

Reproduction config:
`experiments/single_card_tradeoff/configs/oracle_interval_grid_stationary_first8_eval_weights_add_025_05.toml`

Forced rerun command:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_tradeoff_config \
  --config experiments/single_card_tradeoff/configs/oracle_interval_grid_stationary_first8_eval_weights_add_025_05.toml \
  --force
```

Setup:

- Environment: `fsrs6`
- Users: `1,2,3,4,5,6,7,8`
- Baseline: `fsrs6` desired-retention frontier
- Compared schedulers: `fsrs6_oracle_interval`, `fsrs6_oracle`,
  `fsrs6_oracle_stationary_finite`,
  `fsrs6_oracle_stationary_finite_distill`
- Distill policy:
  `artifacts/single_card_tradeoff/stationary_finite_distill_train_weights_add_4_only_first8_markov_off/user_{user_id}_policy.pt`
- Lifecycle: `1825` days
- Particles: `10000`
- Evaluation cost weights:
  `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Grid: `64` stability bins by `32` difficulty bins
- Interval chunk size: `64`
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Review Markov transitions: off
- Device: CUDA

## Results

Mean same-target time saved AUC vs `fsrs6`:

| scheduler | users | positive users | mean time saved AUC | mean relative time saved | mean span coverage | min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_interval` | 8 | 8 | 7.0926 | 16.97% | 99.99% | 99.93% |
| `fsrs6_oracle` | 8 | 8 | 6.3309 | 15.19% | 98.77% | 93.48% |
| `fsrs6_oracle_stationary_finite` | 8 | 8 | 6.0728 | 14.47% | 99.00% | 95.47% |
| `fsrs6_oracle_stationary_finite_distill` | 8 | 8 | 5.3481 | 13.06% | 99.09% | 96.82% |

Per-user relative time saved vs `fsrs6`:

| user | interval | finite grid oracle | stationary finite | distill |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.32% | 10.71% | 9.57% | 9.24% |
| 2 | 19.02% | 16.54% | 16.33% | 11.64% |
| 3 | 16.15% | 15.00% | 14.52% | 12.34% |
| 4 | 25.48% | 22.81% | 22.27% | 20.90% |
| 5 | 23.65% | 20.20% | 19.37% | 16.34% |
| 6 | 20.26% | 19.13% | 17.60% | 16.50% |
| 7 | 10.41% | 9.64% | 9.71% | 11.90% |
| 8 | 8.48% | 7.46% | 6.37% | 5.62% |

Direct oracle pairwise rows:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_stationary_finite` | `fsrs6_oracle` | +0.2476 | +0.83% | 7/8 | 99.19% |
| `fsrs6_oracle` | `fsrs6_oracle_stationary_finite` | -0.2476 | -0.84% | 1/8 | 99.93% |
| `fsrs6_oracle` | `fsrs6_oracle_interval` | +0.8028 | +2.11% | 8/8 | 99.84% |
| `fsrs6_oracle_stationary_finite` | `fsrs6_oracle_interval` | +1.0473 | +2.92% | 8/8 | 99.30% |

Direct distill pairwise rows from `combined_regret_auc.csv`:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle_interval` | +1.7611 | +4.50% | 7/8 | 96.59% |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle` | +0.9703 | +2.48% | 7/8 | 96.25% |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle_stationary_finite` | +0.7228 | +1.66% | 7/8 | 96.92% |

## Interpretation

The earlier anomaly was an implementation-comparison artifact, not evidence that
a stationary finite policy is theoretically stronger than the non-stationary
finite oracle. The old comparison was:

- `fsrs6_oracle`: finite non-stationary policy with nearest-grid transition
  backup.
- `fsrs6_oracle_stationary_finite`: stationary policy initialized and improved
  with a four-corner transition kernel.

After both policies use the same four-corner transition approximation for
finite-horizon planning, the expected ordering is restored in the aggregate:
`fsrs6_oracle` beats `fsrs6_oracle_stationary_finite` on mean AUC and mean
relative AUC vs `fsrs6`, and direct stationary-baseline pairwise comparison
shows `fsrs6_oracle` winning 7 of 8 users.

`fsrs6_oracle_interval` remains stronger than both retention-action grid
oracles. That is expected because it has a richer integer-interval action space.

The 476-parameter distill policy is below all exact oracle variants in the mean
frontier comparison, but it preserves broad coverage. It beats the exact grid
and interval variants only on user 7; this is consistent with the student being
a compact approximation of the stationary finite teacher rather than a planning
upper bound.

## Visuals

![Same-target time saved AUC by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/same_target_time_saved_auc_by_user.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/span_coverage_by_user.png)

Per-user frontier plots are under
`artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results_by_user/user_<id>.png`.

## Runtime And GPU Monitor

- Runtime: `463.38s`
- DP cache hits: `456`
- DP cache misses: `0`
- DP cache writes: `0`
- Peak `nvidia-smi` memory: `3270 MiB`
- Peak summed shared GPU memory: `242,528,256` bytes
- Shared-memory spill detected: `false`

## Artifacts

- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_regret_auc.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/mean_summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/oracle_pairwise_by_user.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/oracle_pairwise_mean_summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/performance_summary.json`

## Conclusion

The finite non-stationary grid oracle should be used as the fair retention-action
upper bound after this patch. `fsrs6_oracle_stationary_finite` remains useful as
a compact stationary teacher class, but it is no longer advantaged by a better
transition approximation than `fsrs6_oracle`.

The added `fsrs6_oracle_stationary_finite_distill` row lands where expected for
a 476-parameter student: lower mean AUC than the exact stationary finite policy
(`5.3481` vs `6.0728`), but high mean coverage (`99.09%`).
