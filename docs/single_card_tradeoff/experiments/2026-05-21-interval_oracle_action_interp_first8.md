# Interval Oracle Action Interpolation, First Eight Users

## Question

When executing a precomputed `fsrs6_oracle_interval` policy, does bilinear
numeric interpolation of the output interval action improve coverage or relative
same-target time saved AUC compared with the current nearest-grid lookup?

## Configuration

Reproduction config:
`experiments/single_card_tradeoff/configs/interval_oracle_action_interp_first8_eval_weights.toml`

- Environment: `fsrs6`
- Users: `1,2,3,4,5,6,7,8`
- Baseline: `fsrs6` desired-retention frontier
- Lifecycle: `1825` days
- Particles: `10000`
- Evaluation cost weights: `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Interval oracle grid: `64` stability bins by `32` difficulty bins
- Interval chunk size: `64`
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Review Markov transitions: off
- Device: CUDA

Compared schedulers:

- `fsrs6_oracle_interval`: nearest-grid action lookup.
- `fsrs6_oracle_interval_bilinear_action`: bilinear numeric interpolation over
  the four surrounding `(log stability, difficulty)` action-table entries, then
  `round` and clamp to `[1, remaining + 1]`.

The policy table is unchanged between the two interval schedulers. Only the
execution-time lookup differs.

## Results

Mean vs `fsrs6`:

| scheduler | users | positive users | mean time-saved AUC | mean relative time saved | mean span coverage | min span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_interval` | 8 | 6 | 0.5471 | 8.65% | 99.99% | 99.95% |
| `fsrs6_oracle_interval_bilinear_action` | 8 | 6 | 0.6075 | 8.67% | 99.99% | 99.95% |

Bilinear action minus nearest-grid:

| user | AUC delta | relative delta | coverage delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.1124 | 0.19 pp | 0.00 pp |
| 2 | 0.1440 | 0.22 pp | 0.00 pp |
| 3 | -0.0018 | -0.01 pp | 0.00 pp |
| 4 | 0.3373 | 0.35 pp | 0.00 pp |
| 5 | -0.0507 | -0.21 pp | 0.00 pp |
| 6 | -0.0551 | -0.32 pp | 0.00 pp |
| 7 | -0.0050 | -0.09 pp | 0.00 pp |
| 8 | 0.0023 | 0.03 pp | 0.00 pp |

Aggregate delta:

- Mean same-target time-saved AUC delta: `+0.0604`
- Mean relative time-saved delta: `+0.0203` percentage points
- Mean coverage delta: `0.0000` percentage points
- Positive AUC delta users: `4/8`
- Negative AUC delta users: `4/8`

## Visuals

![Bilinear action deltas by user](../../../artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/action_interp_delta_by_user.png)

![Same-target time saved AUC by user](../../../artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/same_target_time_saved_auc_by_user.png)

![Relative time saved by user](../../../artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/relative_time_saved_by_user.png)

![Span coverage by user](../../../artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/span_coverage_by_user.png)

Per-user frontier plots are under
`artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/combined_results_by_user/user_<id>.png`.

## Runtime And GPU Monitor

- Runtime: `1306.79s`
- DP cache hits: `136`
- DP cache misses: `136`
- DP cache writes: `136`
- Peak `nvidia-smi` memory: `3880 MiB`
- Peak summed shared GPU memory: `245,338,112` bytes
- Shared-memory spill detected: `false`

The cache pattern is expected: the nearest-grid scheduler solved and cached the
136 user/weight interval policies, then the bilinear-action scheduler reused the
same policy tables and changed only the execution lookup.

## Artifacts

- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/combined_results.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/combined_regret_auc.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/summary.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/mean_summary.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/action_interp_delta_by_user.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/action_interp_delta_summary.csv`
- `artifacts/single_card_tradeoff/interval_oracle_action_interp_first8_eval_weights_markov_off/action_interp_delta_by_user.png`

## Conclusion

Bilinear numeric interpolation of the interval action does not materially change
coverage in this first-eight evaluation. Coverage is identical at the reported
precision for every user. It gives a very small average AUC lift
(`+0.0604`, or `+0.0203` relative percentage points), but the effect is mixed
across users: users 1, 2, 4, and 8 improve, while users 3, 5, 6, and 7 regress.

The practical takeaway is that action interpolation is not a clear fix for
frontier quality. It slightly smooths the execution policy, but most of the
observed behavior remains determined by the underlying interval policy table and
grid resolution.
