# Continuous Uniform-H Policy Analysis

## Question

This report tests how a hidden uniformly distributed lifecycle end date changes
the continuous-retention oracle policy. The comparison is between:

- fixed horizon: `fsrs6_oracle_continuous_retention` with `H = 1825`
- hidden random horizon: `H ~ Uniform{1, ..., 1825}`, where the policy only
  knows that the card has not terminated yet

The analysis is table-level, uniformly averaged over first-eight users and all
`64 x 32` stability/difficulty grid cells. It is not rollout-occupancy weighted.

## Source

Command:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/continuous_uniform_h_policy_analysis_first8.toml \
  --stage analyze
```

Configuration:

- Users: `1,2,3,4,5,6,7,8`
- Lifecycle upper bound: `1825` days
- Cost weights:
  `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Grid: `64` log-spaced stability values x `32` difficulty values
- Retention bounds: `[0.5, 0.98]`
- Uniform-H DP cache: `152` misses/writes; fixed-H policy cache: `152` hits
- Runtime: `1343.16s` on `cuda`
- GPU monitor: peak `nvidia-smi` memory `5765 MiB`, peak shared memory
  `163.16 MB`, `shared_memory_spill_detected=false`

The hidden Uniform-H Bellman backup uses remaining time `r` and terminal offset
`tau ~ Uniform{1, ..., r}`. Review cost is weighted by `P(tau >= interval)`,
future value by `P(tau > interval)`, and terminal/no-review still appears in
the retention table as `retention_min`.

## Artifacts

Artifact directory:
`artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/`

Key files:

- `uniform_remaining_summary.csv`: Uniform-H retention metrics by cost weight
  and remaining day.
- `fixed_vs_uniform_remaining_comparison.csv`: same-remaining fixed-H vs
  Uniform-H policy deltas.
- `uniform_start_vs_fixed_landmarks.csv`: which fixed-H remaining slice is
  closest to the Uniform-H start policy.
- `remaining_landmarks.csv`: selected remaining-day landmark rows.
- `metadata.json`, `performance_summary.json`, and `gpu_monitor/summary.json`.

Visuals:

![Uniform-H mean retention and min share](../../../artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/uniform_mean_retention_and_min_share_by_remaining.png)

![Fixed-H versus Uniform-H delta](../../../artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/fixed_vs_uniform_delta_by_remaining.png)

![Uniform-H start closest fixed-H remaining slice](../../../artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/uniform_start_closest_fixed_remaining.png)

![Uniform-H stability remaining heatmaps](../../../artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/stability_remaining_heatmaps.png)

![Uniform-H difficulty remaining heatmaps](../../../artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis/difficulty_remaining_heatmaps.png)

## Results

Uniform-H does not remove the remaining-time effect. It makes the start policy
more conservative than the fixed `1825`-day long-horizon policy, especially at
medium and high cost weights.

Mean desired retention:

| cost | 1d | 7d | 30d | 365d | 730d | 1125d | 1600d | 1824d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6519 | 0.7174 | 0.7765 | 0.8766 | 0.9065 | 0.9261 | 0.9335 | 0.9345 |
| 16 | 0.5003 | 0.5015 | 0.5260 | 0.7544 | 0.8203 | 0.8563 | 0.8805 | 0.8871 |
| 64 | 0.5003 | 0.5008 | 0.5136 | 0.6155 | 0.6986 | 0.7489 | 0.7871 | 0.8002 |
| 256 | 0.5003 | 0.5007 | 0.5122 | 0.5564 | 0.5831 | 0.6075 | 0.6362 | 0.6483 |
| 1024 | 0.5003 | 0.5007 | 0.5116 | 0.5525 | 0.5702 | 0.5815 | 0.5907 | 0.5942 |

Share of cells at `retention_min`:

| cost | 1d | 7d | 30d | 365d | 730d | 1125d | 1600d | 1824d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 59.0% | 45.3% | 33.0% | 12.1% | 5.9% | 1.8% | 0.2% | 0.0% |
| 16 | 99.4% | 96.4% | 87.6% | 26.6% | 13.6% | 6.6% | 2.0% | 0.8% |
| 64 | 99.4% | 96.6% | 90.6% | 57.1% | 33.1% | 21.7% | 14.0% | 11.5% |
| 256 | 99.4% | 96.6% | 90.8% | 76.1% | 66.3% | 59.1% | 49.9% | 45.8% |
| 1024 | 99.4% | 96.6% | 91.0% | 77.0% | 69.2% | 65.5% | 63.0% | 62.1% |

The Uniform-H start policy is closest to progressively shorter fixed-H slices
as review cost rises:

| cost | closest fixed remaining | closest abs delta | delta vs fixed 1824d | delta vs fixed 730d | delta vs fixed 365d |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1824 | 0.0006 | 0.0006 | 0.0284 | 0.0586 |
| 16 | 1595 | 0.0053 | 0.0082 | 0.0428 | 0.0963 |
| 64 | 1217 | 0.0111 | 0.0432 | 0.0517 | 0.1213 |
| 256 | 1010 | 0.0188 | 0.0682 | 0.0360 | 0.0710 |
| 1024 | 926 | 0.0194 | 0.0385 | 0.0221 | 0.0304 |

At the same remaining day, Uniform-H is generally lower than fixed-H for
nonzero costs. The largest observed mean absolute delta is `0.0689` at
`w=192`, `remaining=1437`, with `41.66%` of cells differing by more than `0.02`.

Selected same-remaining deltas:

| cost | rem | mean delta U-F | abs delta | share >0.02 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 365 | -0.0396 | 0.0404 | 25.2% |
| 16 | 1824 | -0.0059 | 0.0082 | 9.9% |
| 64 | 365 | -0.0663 | 0.0673 | 37.4% |
| 64 | 1824 | -0.0409 | 0.0432 | 33.7% |
| 256 | 365 | -0.0341 | 0.0354 | 17.0% |
| 256 | 1824 | -0.0665 | 0.0682 | 41.0% |
| 1024 | 365 | -0.0331 | 0.0354 | 16.0% |
| 1024 | 1824 | -0.0327 | 0.0385 | 21.9% |

## Interpretation

The fixed-horizon report showed a strong deadline mode. This experiment shows
that a hidden uniform terminal day does not simply smooth that mode away. It
turns the deadline into a survival-weighted gate that affects the policy even at
the start of the lifecycle.

Low review cost is nearly unchanged: `w=0` at the start is closest to the fixed
`1824`-day policy. As cost rises, long-term memory investments must survive the
random terminal date to pay back, so the Uniform-H oracle lowers target
retention and expands the lower-bound region. At `w=256`, the start policy is
closer to a fixed `1010`-day slice than to the fixed `1824`-day slice.

This supports the theoretical prediction: random finite lifecycle does not make
the optimal policy stationary. It changes the remaining-time input from a known
deadline into a time-varying survival distribution, and the resulting policy is
still materially nonstationary.

## Caveats

The analysis weights every user and every S/D grid cell uniformly. Rollout
state occupancy may emphasize a smaller subset of the policy table.

`retention_min_share` is a policy-surface proxy. It includes both terminal
no-review representation and ordinary low-retention intervals, so it should not
be read as an exact skip rate.
