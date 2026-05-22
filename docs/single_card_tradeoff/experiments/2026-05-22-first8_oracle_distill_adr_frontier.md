# First-Eight FSRS6 Oracle, Distill, And ADR Frontier Comparison

## Question

How does a native single-card `fsrs6_adr` policy, trained separately for each of
the first eight users and each evaluation cost weight, compare against the exact
oracle family and the compact stationary-finite distill baseline?

The experiment focuses on three practical questions:

- How much of the exact oracle frontier can the 1024-particle ADR policies
  recover?
- How much gap remains to the compact 476-parameter
  `fsrs6_oracle_stationary_finite_distill` policy?
- Does increasing ADR's training estimator from the earlier 64- and
  512-particle runs materially improve the frontier?

## Method

All schedulers are evaluated on the same first-eight-user `fsrs6` single-card
tradeoff setup. The metric is same-target time saved AUC relative to the
`fsrs6` desired-retention frontier, with coverage reporting how much of the
baseline memory span is shared by the compared frontier.

The oracle rows have different action-space and model-class meanings:

- `fsrs6_oracle_interval` is the richest exact baseline here because it chooses
  integer review intervals directly.
- `fsrs6_oracle` is the finite-horizon retention-action grid oracle.
- `fsrs6_oracle_stationary_finite` is the stationary retention-action policy
  class, seeded from the same finite grid oracle solve.
- `fsrs6_oracle_stationary_finite_distill` is a compact 476-parameter student
  of the stationary finite policy.
- `fsrs6_adr` is a native learned single-card policy trained by CEM, one policy
  per user and cost weight.

The exact retention-action grid oracles use the current four-corner transition
backup over `(log stability, difficulty)` grid corners. This keeps
`fsrs6_oracle` and `fsrs6_oracle_stationary_finite` on the same transition
approximation. Execution-time table lookup still maps each continuous `(S, D)`
state to the nearest policy grid point.

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
- Compared schedulers: `fsrs6_adr`, `fsrs6_oracle_interval`, `fsrs6_oracle`,
  `fsrs6_oracle_stationary_finite`,
  `fsrs6_oracle_stationary_finite_distill`
- ADR policies:
  `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off`
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

ADR training:

- Policies: `8 users * 19 cost weights = 152`
- CEM population: `32`
- Elite count: `8`
- Generations: `64`
- Train particles per candidate: `1024`
- Eval particles per policy: `10000`
- `job_batch_size`: `8192`

## Results

Mean same-target time saved AUC vs `fsrs6`:

| scheduler | users | positive users | mean time saved AUC | mean relative time saved | mean span coverage | min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_interval` | 8 | 8 | 7.0926 | 16.97% | 99.99% | 99.93% |
| `fsrs6_oracle` | 8 | 8 | 6.3309 | 15.19% | 98.77% | 93.48% |
| `fsrs6_oracle_stationary_finite` | 8 | 8 | 6.0728 | 14.47% | 99.00% | 95.47% |
| `fsrs6_oracle_stationary_finite_distill` | 8 | 8 | 5.3481 | 13.06% | 99.09% | 96.82% |
| `fsrs6_adr` | 8 | 8 | 4.9303 | 11.20% | 95.48% | 83.79% |

ADR train-particle progression from the reruns of this profile:

| train particles | mean time saved AUC | mean relative time saved | mean span coverage | training gate |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 4.3697 | 9.63% | 95.95% | 149/152 |
| 512 | 4.7020 | 10.64% | 94.94% | 150/152 |
| 1024 | 4.9303 | 11.20% | 95.48% | 152/152 |

Policy parameter / table-size comparison:

| scheduler | representation counted | per point | per-user evaluated frontier | first-eight total | note |
| --- | --- | ---: | ---: | ---: | --- |
| `fsrs6` | desired-retention scalar | 1 | 11 | 88 | Baseline frontier settings, not learned policy parameters. |
| `fsrs6_oracle_interval` | finite-horizon integer interval table | 3,739,648 | 71,053,312 | 568,426,496 | `(1825 + 1) * 64 * 32` table entries per cost weight. |
| `fsrs6_oracle` | finite-horizon retention-action table | 3,739,648 | 71,053,312 | 568,426,496 | Integer action ids over the retention action set. |
| `fsrs6_oracle_stationary_finite` | stationary retention-action table | 2,048 | 38,912 | 311,296 | `64 * 32` table entries per cost weight. |
| `fsrs6_oracle_stationary_finite_distill` | residual MLP | 476 | 476 | 3,808 | One per-user network conditions on goal cost weight. |
| `fsrs6_adr` | log-polynomial retention function | 6 | 114 | 912 | Six coefficients per cost weight, 19 policies per user. |

Per-user relative time saved vs `fsrs6`:

| user | interval | finite grid oracle | stationary finite | distill | ADR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.32% | 10.71% | 9.57% | 9.24% | 6.74% |
| 2 | 19.02% | 16.54% | 16.33% | 11.64% | 13.48% |
| 3 | 16.15% | 15.00% | 14.52% | 12.34% | 12.72% |
| 4 | 25.48% | 22.81% | 22.27% | 20.90% | 17.48% |
| 5 | 23.65% | 20.20% | 19.37% | 16.34% | 14.41% |
| 6 | 20.26% | 19.13% | 17.60% | 16.50% | 14.07% |
| 7 | 10.41% | 9.64% | 9.71% | 11.90% | 5.18% |
| 8 | 8.48% | 7.46% | 6.37% | 5.62% | 5.49% |

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

Direct ADR pairwise rows from `combined_regret_auc.csv`:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_adr` | `fsrs6_oracle_interval` | +2.3792 | +6.30% | 8/8 | 100.00% |
| `fsrs6_adr` | `fsrs6_oracle` | +1.6181 | +4.49% | 8/8 | 99.92% |
| `fsrs6_adr` | `fsrs6_oracle_stationary_finite` | +1.3568 | +3.66% | 8/8 | 99.98% |
| `fsrs6_adr` | `fsrs6_oracle_stationary_finite_distill` | +0.6069 | +1.97% | 6/8 | 99.96% |

## Interpretation

The frontier ordering is now stable and interpretable. `fsrs6_oracle_interval`
is the strongest row because its integer-interval action space is richer than
the retention-action grid. `fsrs6_oracle` is the best retention-action exact
grid policy in aggregate, followed by the stationary finite policy. The
stationary finite row remains useful as a teacher class, not as the planning
upper bound. The parameter table also makes the upper-bound caveat concrete:
the finite-horizon exact rows store about 71.05M action entries per user for
this 19-weight frontier.

The 476-parameter distill policy is below the exact oracle rows in the mean
frontier comparison, but it preserves broad coverage. Its `13.06%` mean relative
time saved is still above ADR's `11.20%`, so the compact supervised student
remains a stronger first-eight-user baseline than the native ADR family tested
here despite ADR using only 114 coefficients per user for the full 19-weight
frontier.

The native single-card ADR frontier is positive against `fsrs6` for all eight
users. Raising the training estimator from 64 to 512 to 1024 particles improves
ADR's mean relative AUC from 9.63% to 10.64% to 11.20%, and the training gate
improves from 149/152 to 152/152. That indicates the smaller training
estimators were materially noisy. The remaining gap is not just gate failure,
though: with 1024 particles, ADR still trails distill by 1.86 percentage points
of relative time saved and trails the interval oracle by 5.78 percentage points.

Coverage is the remaining ADR weakness. Its mean coverage is `95.48%`, but its
minimum user coverage is `83.79%`; the exact and distill rows all keep higher
mean coverage. Direct shared-span comparison shows distill ahead of ADR on 6 of
8 users; ADR is ahead on users 2 and 3.

## Visuals

![Same-target time saved AUC by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/same_target_time_saved_auc_by_user.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/span_coverage_by_user.png)

Per-user tradeoff plots are under
`artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results_by_user/user_<id>.png`.
These plots use a log-scaled y-axis for study minutes per day, omit the black
global Pareto frontier overlay, and use sparse point labels so low-workload
points remain readable alongside the high-workload `w=0` points.

## Runtime And GPU Monitor

ADR training:

- Runtime: `1252.41s`
- Policies: `152`
- Passed training objective gate: `152/152`
- Failed gates: none
- Peak `nvidia-smi` memory: `5020 MiB`
- Peak summed shared GPU memory: `240,205,824` bytes
- Shared-memory spill detected: `false`

Tradeoff evaluation:

- Runtime: `479.30s`
- DP cache hits: `456`
- DP cache misses: `0`
- DP cache writes: `0`
- Peak `nvidia-smi` memory: `3409 MiB`
- Peak summed shared GPU memory: `213,090,304` bytes
- Shared-memory spill detected: `false`

## Artifacts

- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_regret_auc.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/mean_summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/oracle_pairwise_by_user.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/oracle_pairwise_mean_summary.csv`
- `artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/performance_summary.json`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off/policy_manifest.toml`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off/summary.csv`
- `artifacts/single_card_tradeoff/native_adr_first8_eval_weights_add_025_05_markov_off/performance_summary.json`

## Conclusion

The current first-eight-user comparison should be read as a scheduler frontier
study rather than a four-corner implementation check. `fsrs6_oracle_interval`
is the interval-action upper bound in this setup, while `fsrs6_oracle` is the
retention-action finite-horizon upper bound.

The 476-parameter stationary finite distill policy remains the strongest compact
baseline in this comparison: `13.06%` mean relative time saved at `99.09%` mean
coverage.

The 1024-particle ADR run is clearly better than the 64- and 512-particle runs:
it reaches `11.20%` mean relative time saved, `95.48%` mean coverage, and passes
all 152 training gates. It is still below distill and all exact oracle rows, so
further progress likely needs changes to the ADR policy/search class rather than
only more particles.
