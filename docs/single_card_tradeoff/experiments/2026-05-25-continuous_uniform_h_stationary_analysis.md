# Continuous Uniform-H Stationary Analysis

## Question

Can we implement a stationary approximation to the hidden
`H ~ Uniform{1..1825}` objective, and how does that stationary policy compare
with the existing `FSRS6ContinuousStationaryFiniteOracle` fixed-horizon
stationary policy?

## Method

This run adds `FSRS6ContinuousStationaryUniformTerminationOracle`, which keeps a
single policy table `pi(cost_weight, stability, difficulty)` but evaluates and
improves it under the hidden Uniform-H terminal model. The value recursion still
tracks remaining days, while review cost and future value are weighted by
`P(H >= interval)` and `P(H > interval)`.

The analysis CLI is:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/continuous_uniform_h_stationary_analysis_first8.toml \
  --stage analyze
```

The full first-eight CUDA run was attempted, but the cold run was interrupted
after `1546.71s`: it had finished user 1, entered user 2, and had written 19 new
DP cache entries. GPU monitor reported peak `7311 MiB` FB memory and no shared
memory spill. The complete numeric comparison below therefore uses the full-grid
user-1 diagnostic artifact at
`artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/`.

## Results

For user 1, all 19 cost weights converged. Policy iteration counts ranged from
1 to 8, and max residual was `8.95e-11`.

Selected stationary Uniform-H policy rows:

| cost weight | mean retention | retention-min share | Uniform-H objective | iterations |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9264 | 0.0000 | 0.4941 | 1 |
| 16 | 0.8243 | 0.0312 | 0.3934 | 4 |
| 64 | 0.7202 | 0.1543 | 0.2428 | 4 |
| 256 | 0.6420 | 0.2754 | -0.1885 | 8 |
| 1024 | 0.6124 | 0.3887 | -1.7919 | 3 |

Against `FSRS6ContinuousStationaryFiniteOracle`, evaluated under the Uniform-H
objective, the new stationary Uniform-H policy improves the objective at every
selected nonzero cost weight. The surface difference is largest around medium
cost:

| cost weight | mean abs retention delta | Uniform-H objective delta |
| ---: | ---: | ---: |
| 0 | 0.000009 | 0.0000000004 |
| 16 | 0.012429 | 0.000163 |
| 64 | 0.047194 | 0.000819 |
| 256 | 0.035542 | 0.012110 |
| 1024 | 0.023757 | 0.039710 |

Compared with the nonstationary Uniform-H oracle slices, the stationary
Uniform-H table stays close to the start/long-horizon slice at low and medium
costs, but diverges more at high cost. Closest selected slices:

| cost weight | closest selected remaining day | mean abs delta |
| ---: | ---: | ---: |
| 16 | 1824 | 0.003770 |
| 64 | 1824 | 0.009567 |
| 256 | 1600 | 0.053890 |

## Interpretation

The stationary Uniform-H approximation is not the same object as the
nonstationary hidden-H policy. It optimizes over the restricted class of
remaining-time-independent policies, so it smooths deadline behavior by
construction. Within that restricted class, it can outperform the fixed-H
stationary policy when both are judged under Uniform-H, especially at high cost
where survival-weighted review cost matters.

The main caveat is compute cost. A full first-eight cold solve is substantially
heavier than the previous nonstationary Uniform-H surface analysis. The
interrupted run and a 3-weight user-2 diagnostic both showed no GPU memory spill,
so the bottleneck is the exact stationary policy-improvement loop rather than
VRAM pressure.

## Visualizations

![Uniform-H stationary mean retention and min share](../../../artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/stationary_uniform_mean_retention_and_min_share.png)

![Uniform-H stationary vs fixed-H stationary objective](../../../artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/uniform_h_objective_comparison.png)

![Uniform-H stationary policy heatmaps](../../../artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/stationary_uniform_policy_heatmaps.png)

![Uniform-H stationary vs fixed-H stationary policy heatmaps](../../../artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/stationary_uniform_fixed_policy_comparison_heatmaps.png)

![Uniform-H stationary vs nonstationary Uniform-H slices](../../../artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/stationary_uniform_vs_nonstationary_uniform_delta.png)

## Artifacts

- User-1 diagnostic:
  `artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis_user1_diag/`
- Interrupted first-eight attempt:
  `artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis/`
- Config:
  `experiments/single_card_tradeoff/configs/continuous_uniform_h_stationary_analysis_first8.toml`
