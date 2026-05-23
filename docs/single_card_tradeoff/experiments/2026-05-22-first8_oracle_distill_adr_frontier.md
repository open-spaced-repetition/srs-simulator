# First-Eight FSRS6 Oracle, Distill, And ADR Frontier Comparison

## Question

How does a native single-card `fsrs6_adr` policy, trained separately for each of
the first eight users and each evaluation cost weight, compare against the exact
oracle family, the continuous desired-retention oracle variants, and the compact
stationary-finite distill baselines?

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
- `fsrs6_oracle_continuous_retention` is the finite-horizon continuous
  desired-retention oracle; internally it still enumerates attainable rounded
  intervals.
- `fsrs6_oracle_continuous_stationary_finite` is the stationary continuous
  desired-retention policy class seeded from the continuous finite oracle.
- `fsrs6_oracle_continuous_stationary_finite_distill` is a compact
  476-parameter student of the continuous stationary finite teacher.
- `fsrs6_adr` is a native learned single-card policy trained by CEM, one policy
  per user and cost weight.

The exact retention-action grid oracles use the current four-corner transition
backup over `(log stability, difficulty)` grid corners. This keeps
`fsrs6_oracle` and `fsrs6_oracle_stationary_finite` on the same transition
approximation. Execution-time table lookup still maps each continuous `(S, D)`
state to the nearest policy grid point. The continuous desired-retention rows
use the same rounded-interval simulator semantics, but their policy tables store
retention values and execution uses bilinear retention-policy lookup.

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
  `fsrs6_oracle_stationary_finite_distill`,
  `fsrs6_oracle_continuous_retention`,
  `fsrs6_oracle_continuous_stationary_finite`,
  `fsrs6_oracle_continuous_stationary_finite_distill`
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
| `fsrs6_oracle_interval` | 8 | 8 | 7.0332 | 17.05% | 99.99% | 99.93% |
| `fsrs6_oracle_continuous_retention` | 8 | 8 | 6.7186 | 15.97% | 97.99% | 87.58% |
| `fsrs6_oracle_continuous_stationary_finite` | 8 | 8 | 6.4386 | 14.95% | 98.20% | 91.77% |
| `fsrs6_oracle` | 8 | 8 | 6.2762 | 15.27% | 98.78% | 93.38% |
| `fsrs6_oracle_stationary_finite` | 8 | 8 | 6.0174 | 14.55% | 99.01% | 95.37% |
| `fsrs6_oracle_continuous_stationary_finite_distill` | 8 | 8 | 5.4253 | 12.72% | 93.00% | 63.19% |
| `fsrs6_oracle_stationary_finite_distill` | 8 | 8 | 5.2921 | 13.14% | 99.10% | 96.72% |
| `fsrs6_adr` | 8 | 8 | 4.8727 | 11.29% | 95.52% | 84.15% |

ADR train-particle progression from the reruns of this profile:

| train particles | mean time saved AUC | mean relative time saved | mean span coverage | training gate |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 4.3697 | 9.63% | 95.95% | 149/152 |
| 512 | 4.7020 | 10.64% | 94.94% | 150/152 |
| 1024 | 4.8727 | 11.29% | 95.52% | 152/152 |

Policy parameter / table-size comparison:

| scheduler | representation counted | per point | per-user evaluated frontier | first-eight total | note |
| --- | --- | ---: | ---: | ---: | --- |
| `fsrs6` | desired-retention scalar | 1 | 11 | 88 | Baseline frontier settings, not learned policy parameters. |
| `fsrs6_oracle_interval` | finite-horizon integer interval table | 3,739,648 | 71,053,312 | 568,426,496 | `(1825 + 1) * 64 * 32` table entries per cost weight. |
| `fsrs6_oracle` | finite-horizon retention-action table | 3,739,648 | 71,053,312 | 568,426,496 | Integer action ids over the retention action set. |
| `fsrs6_oracle_continuous_retention` | finite-horizon continuous retention table | 3,739,648 | 71,053,312 | 568,426,496 | Continuous retention value per remaining-day and state cell. |
| `fsrs6_oracle_continuous_stationary_finite` | stationary continuous retention table | 2,048 | 38,912 | 311,296 | `64 * 32` retention values per cost weight. |
| `fsrs6_oracle_stationary_finite` | stationary retention-action table | 2,048 | 38,912 | 311,296 | `64 * 32` table entries per cost weight. |
| `fsrs6_oracle_continuous_stationary_finite_distill` | residual MLP | 476 | 476 | 3,808 | One per-user network conditions on goal cost weight. |
| `fsrs6_oracle_stationary_finite_distill` | residual MLP | 476 | 476 | 3,808 | One per-user network conditions on goal cost weight. |
| `fsrs6_adr` | log-polynomial retention function | 6 | 114 | 912 | Six coefficients per cost weight, 19 policies per user. |

Per-user relative time saved vs `fsrs6`:

| user | interval | cont finite | finite grid oracle | cont stationary | stationary finite | cont distill | distill | ADR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.44% | 12.17% | 10.86% | 11.26% | 9.70% | 10.03% | 9.38% | 6.89% |
| 2 | 18.59% | 18.03% | 16.10% | 17.10% | 15.89% | 16.24% | 11.17% | 13.03% |
| 3 | 16.57% | 15.89% | 15.42% | 14.88% | 14.94% | 11.73% | 12.77% | 13.15% |
| 4 | 25.30% | 23.52% | 22.61% | 23.15% | 22.06% | 17.65% | 20.69% | 17.26% |
| 5 | 23.24% | 19.82% | 19.81% | 18.62% | 18.98% | 12.91% | 15.93% | 14.04% |
| 6 | 20.44% | 19.58% | 19.31% | 17.67% | 17.79% | 14.98% | 16.69% | 14.27% |
| 7 | 11.11% | 10.22% | 10.35% | 10.28% | 10.43% | 12.26% | 12.59% | 5.92% |
| 8 | 8.71% | 8.49% | 7.69% | 6.65% | 6.61% | 5.96% | 5.86% | 5.73% |

Direct oracle pairwise rows:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_stationary_finite` | `fsrs6_oracle` | +0.2476 | +0.83% | 7/8 | 99.19% |
| `fsrs6_oracle` | `fsrs6_oracle_stationary_finite` | -0.2476 | -0.84% | 1/8 | 99.93% |
| `fsrs6_oracle` | `fsrs6_oracle_continuous_retention` | +0.3846 | +0.77% | 6/8 | 98.80% |
| `fsrs6_oracle_continuous_retention` | `fsrs6_oracle` | -0.3846 | -0.78% | 2/8 | 99.91% |
| `fsrs6_oracle_continuous_stationary_finite` | `fsrs6_oracle_continuous_retention` | +0.3107 | +1.20% | 7/8 | 99.38% |
| `fsrs6_oracle_continuous_retention` | `fsrs6_oracle_continuous_stationary_finite` | -0.3107 | -1.22% | 1/8 | 94.99% |
| `fsrs6_oracle` | `fsrs6_oracle_interval` | +0.8028 | +2.11% | 8/8 | 99.84% |
| `fsrs6_oracle_continuous_retention` | `fsrs6_oracle_interval` | +0.4196 | +1.34% | 8/8 | 100.00% |
| `fsrs6_oracle_stationary_finite` | `fsrs6_oracle_interval` | +1.0473 | +2.92% | 8/8 | 99.30% |
| `fsrs6_oracle_continuous_stationary_finite` | `fsrs6_oracle_interval` | +0.7212 | +2.50% | 8/8 | 100.00% |

Direct distill pairwise rows from `combined_regret_auc.csv`:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle_interval` | +1.7611 | +4.50% | 7/8 | 96.59% |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle` | +0.9703 | +2.48% | 7/8 | 96.25% |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle_stationary_finite` | +0.7228 | +1.66% | 7/8 | 96.92% |
| `fsrs6_oracle_stationary_finite_distill` | `fsrs6_oracle_continuous_stationary_finite_distill` | -0.0269 | -0.28% | 3/8 | 86.72% |
| `fsrs6_oracle_continuous_stationary_finite_distill` | `fsrs6_oracle_continuous_stationary_finite` | +1.1361 | +2.44% | 7/8 | 99.98% |
| `fsrs6_oracle_continuous_stationary_finite_distill` | `fsrs6_oracle_continuous_retention` | +1.4747 | +3.67% | 7/8 | 99.51% |
| `fsrs6_oracle_continuous_stationary_finite_distill` | `fsrs6_oracle_interval` | +1.7980 | +4.52% | 7/8 | 100.00% |

Direct ADR pairwise rows from `combined_regret_auc.csv`:

| baseline | scheduler | mean AUC delta | mean relative delta | positive users | mean coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_adr` | `fsrs6_oracle_interval` | +2.3792 | +6.30% | 8/8 | 100.00% |
| `fsrs6_adr` | `fsrs6_oracle_continuous_retention` | +2.0380 | +5.30% | 8/8 | 99.12% |
| `fsrs6_adr` | `fsrs6_oracle` | +1.6181 | +4.49% | 8/8 | 99.92% |
| `fsrs6_adr` | `fsrs6_oracle_stationary_finite` | +1.3568 | +3.66% | 8/8 | 99.98% |
| `fsrs6_adr` | `fsrs6_oracle_continuous_stationary_finite_distill` | +0.5436 | +1.59% | 6/8 | 94.63% |
| `fsrs6_adr` | `fsrs6_oracle_stationary_finite_distill` | +0.6069 | +1.97% | 6/8 | 99.96% |

## Interpretation

The frontier ordering is now cleaner. `fsrs6_oracle_interval` is still the
strongest row because its integer-interval action space is richer than the
retention-action grid. The continuous finite oracle is now positive for all
eight users and has the second highest absolute mean AUC (`6.7186`) and second
highest mean relative time saved (`15.97%`). The terminal no-review
canonicalization removed the earlier user 7 regression.

The stationary rows show a smaller but still visible approximation cost. The
continuous stationary finite row improves absolute mean AUC over discrete
stationary finite (`6.4386` vs `6.0174`) and is now slightly higher on relative
time saved (`14.95%` vs `14.55%`), while keeping lower coverage than the
discrete stationary row. The finite-horizon exact rows still store about
71.05M action entries or retention values per user for this 19-weight frontier,
so the stationary rows remain useful as teacher classes rather than planning
upper bounds.

The compact distill comparison is closer after retraining. The continuous
stationary distill is now positive for all eight users and has higher absolute
mean AUC than the discrete distill (`5.4253` vs `5.2921`). The original discrete
distill still has better mean relative time saved (`13.14%` vs `12.72%`) and
much better coverage (`99.10%` vs `93.00%`), mainly because continuous distill
still loses shared-span coverage on user 5. On this report's primary
relative/coverage reading, the original discrete distill remains the stronger
compact baseline, but the previous user 7 failure is gone.

The native single-card ADR frontier is positive against `fsrs6` for all eight
users. Raising the training estimator from 64 to 512 to 1024 particles improves
ADR's mean relative AUC from 9.63% to 10.64% to 11.29%, and the training gate
improves from 149/152 to 152/152. That indicates the smaller training
estimators were materially noisy. The remaining gap is not just gate failure:
with 1024 particles, ADR still trails the discrete distill by 1.85 percentage
points of relative time saved and trails the interval oracle by 5.76 percentage
points.

Coverage is still the main ADR weakness. Its mean coverage is `95.52%`, with a
minimum user coverage of `84.15%`. Direct shared-span comparison shows discrete
distill ahead of ADR on 6 of 8 users; ADR is ahead on users 2 and 3. The
continuous rows no longer have a user 7 sign problem, but continuous distill's
user 5 coverage remains the next diagnostic target.

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

Continuous stationary finite distill retrain:

- Runtime: `1288.7s` through checkpoint write
- Teacher cache hits: `48`
- Teacher cache misses: `0`
- Final train mean loss: `0.864017`
- Checkpoints:
  `artifacts/single_card_tradeoff/continuous_stationary_finite_distill_first8_eval_weights_add_025_05_markov_off/user_{1..8}_policy.pt`

Continuous refresh tradeoff evaluation:

- Runtime: `3243.51s`
- DP cache hits: `304`
- DP cache misses: `104`
- DP cache writes: `104`
- Peak `nvidia-smi` memory: `15982 MiB`
- Peak summed shared GPU memory: `196,243,456` bytes
- Shared-memory spill detected: `false`

Notes:

- The final refresh evaluated `fsrs6` plus the three continuous schedulers with
  `--target-batch-size 4`, then merged those rows back into the full comparison.
- An unchunked attempt hit shared GPU memory spill during continuous stationary
  finite cache-miss solving and was discarded.
- Continuous stationary finite cache-miss solving now writes cache per internal
  cost-weight block, so future runs of the same v3 policy-iteration key should
  hit cache.

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
is the interval-action upper bound in this setup. Among retention-action exact
rows, the continuous finite oracle is now the strongest mean row by both
absolute AUC and relative time saved, and it is positive for all eight users.

The 476-parameter stationary finite distill policy remains the strongest compact
baseline by relative time saved and coverage: `13.14%` mean relative time saved
at `99.10%` mean coverage. The continuous stationary distill is higher on
absolute AUC and now fixes user 7, but it remains weaker on relative time saved
and coverage because of user 5 shared-span loss.

The 1024-particle ADR run is clearly better than the 64- and 512-particle runs:
it reaches `11.29%` mean relative time saved, `95.52%` mean coverage, and passes
all 152 training gates. It is still below the interval oracle, the continuous
finite oracle, the discrete finite grid oracle, and the discrete distill row on
the primary relative comparison, so further progress likely needs changes to
the ADR policy/search class rather than only more particles.
