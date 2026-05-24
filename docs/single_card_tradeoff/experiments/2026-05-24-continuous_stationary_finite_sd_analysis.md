# FSRS6 Continuous Stationary Finite S/D Policy Analysis

## Question

This report analyzes how `fsrs6_oracle_continuous_stationary_finite` changes
with FSRS stability `S` and difficulty `D`, then compares that stationary
surface against the finite-horizon
`fsrs6_oracle_continuous_retention` policy.

The comparison is table-level: each user, cost weight, stability grid cell, and
difficulty grid cell receives equal weight. It is a policy-geometry analysis,
not a rollout occupancy analysis.

## Source

Policy sources:

- `fsrs6_oracle_continuous_stationary_finite`
- `fsrs6_oracle_continuous_retention`

Configuration:

- Users: `1,2,3,4,5,6,7,8`
- Lifecycle: `1825` days
- State grid: `64` log-spaced stability values x `32` difficulty values
- Cost weights:
  `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Retention bounds: `[0.5, 0.98]`
- Stationary solve limit/tolerance: `128`, `1e-10`
- DP cache: loaded `152/152` finite continuous-retention entries and `152/152`
  stationary continuous-finite entries; no DP was rerun.

The stationary table shape is `[user, cost_weight, stability, difficulty]`.
The finite continuous-retention table shape is `[user, cost_weight, remaining,
stability, difficulty]`. Because the stationary policy has no remaining-time
dimension, comparisons use finite remaining days `30`, `365`, and `1824`.

## Artifacts

Artifact directory:
`artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/`

Key files:

- `stationary_sd_summary.csv`: aggregate stationary policy metrics by cost
  weight.
- `stationary_vs_finite_remaining_comparison.csv`: stationary-vs-finite deltas
  at remaining days `30`, `365`, and `1824`.
- `stationary_selected_stability_profiles.csv`: selected-weight profiles along
  `S`.
- `stationary_selected_difficulty_profiles.csv`: selected-weight profiles along
  `D`.
- `stationary_vs_finite_remaining_delta_curves.csv`: selected-weight deltas
  against all finite remaining-day slices.
- `policy_surfaces.npz`: plotted policy surfaces.

## Visuals

![Stationary S/D heatmaps](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_sd_heatmaps.png)

![Stationary S and D profiles](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_s_d_profiles.png)

![Stationary minus finite 1824 heatmaps](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_minus_finite_1824_heatmaps.png)

![Stationary-vs-finite delta by remaining](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_vs_finite_delta_by_remaining.png)

![Stationary summary by weight](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_summary_by_weight.png)

Selected per-weight S/D comparison panels:

- [`w=16`](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_vs_finite_sd_w16.png)
- [`w=64`](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_vs_finite_sd_w64.png)
- [`w=256`](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_vs_finite_sd_w256.png)
- [`w=1024`](../../../artifacts/single_card_tradeoff/continuous_stationary_finite_sd_analysis/stationary_vs_finite_sd_w1024.png)

## Stationary S/D Patterns

The stationary policy is mostly a stability policy. Across representative
weights, the mean-retention range over stability is about `0.31..0.34`, while
the range over difficulty is `0.002` at `w=0`, `0.030` at `w=16`, `0.070` at
`w=64`, and `0.131` at `w=256`. Difficulty matters most in the middle and high
cost region, but it is still smaller than the stability effect.

The low-stability edge is almost cost-invariant. At the lowest stability grid
cell, average stationary retention stays near `0.663..0.667` from `w=0` through
`w=1024`. This is not because all costs want the same memory target; it is the
rounded-interval geometry of very small `S`: the shortest physical interval is
one day, and its canonical retention sits above the configured lower bound.

The high-stability edge carries most of the cost tradeoff. At the highest
stability grid cell, average stationary retention is `0.9800` for `w=0` and
`w=4`, falls to `0.8600` at `w=16`, `0.6200` at `w=64`, and reaches `0.5000`
for `w=256` and `w=1024`. High-stability cards can tolerate long intervals, so
the cost-weighted policy spends the lower-bound action there first.

Difficulty bends the surface in the expected direction once review cost is
nontrivial. At `w=64`, the mean policy is `0.8540` at `D=1` and `0.7842` at
`D=10`. At `w=256`, it is `0.7686` at `D=1` and `0.6376` at `D=10`. Higher
difficulty makes successful reviews less valuable, so the policy accepts lower
retention at the same stability.

## Stationary Summary

| cost weight | mean retention | min share | max share | S range | D range | lowest-S mean | highest-S mean | D=1 mean | D=10 mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9340 | 0.0% | 36.5% | 0.3128 | 0.0019 | 0.6672 | 0.9800 | 0.9330 | 0.9348 |
| 4 | 0.9136 | 0.0% | 19.6% | 0.3128 | 0.0072 | 0.6672 | 0.9800 | 0.9120 | 0.9079 |
| 16 | 0.8917 | 0.4% | 11.9% | 0.3128 | 0.0298 | 0.6670 | 0.8600 | 0.8956 | 0.8674 |
| 64 | 0.8396 | 5.0% | 5.0% | 0.3440 | 0.0703 | 0.6667 | 0.6200 | 0.8540 | 0.7842 |
| 256 | 0.7277 | 23.2% | 2.5% | 0.3152 | 0.1310 | 0.6662 | 0.5000 | 0.7686 | 0.6376 |
| 1024 | 0.6428 | 46.2% | 1.0% | 0.3309 | 0.0769 | 0.6634 | 0.5000 | 0.6629 | 0.5867 |

## Comparison With Finite Continuous Retention

The stationary policy closely tracks the finite policy only at the longest
remaining time and only approximately at high cost. At `1824` remaining days,
mean absolute stationary-vs-finite delta is almost zero for `w=0..4`, `0.0038`
at `w=16`, `0.0072` at `w=64`, and about `0.021` at `w=256` and `w=1024`.
The stationary policy is therefore best interpreted as a long-horizon policy
iteration refinement of the finite oracle, not as an exact copy of the finite
`1824`-day slice.

At shorter horizons, the policies diverge sharply. At `30` remaining days, the
stationary policy is higher than the finite policy across most cells: mean
absolute delta is `0.1593` at `w=0`, `0.2514` at `w=4`, `0.3370` at `w=16`,
and `0.3125` at `w=64`. At `365` remaining days, the gap is smaller but still
large for mid/high costs: `0.1016` at `w=16`, `0.1607` at `w=64`, and `0.1434`
at `w=256`.

This is exactly the remaining-time effect seen in the finite
continuous-retention report. The finite oracle has a deadline mode: as the
horizon approaches, many cells collapse to the terminal/no-review lower-bound
representation. The stationary policy cannot express that deadline mode, so it
keeps applying a long-horizon review-value surface near the end of the episode.

Mean absolute stationary-vs-finite retention delta:

| cost weight | finite 30d | finite 365d | finite 1824d |
| ---: | ---: | ---: | ---: |
| 0 | 0.1593 | 0.0591 | 0.0001 |
| 4 | 0.2514 | 0.0691 | 0.0013 |
| 16 | 0.3370 | 0.1016 | 0.0038 |
| 64 | 0.3125 | 0.1607 | 0.0072 |
| 256 | 0.2033 | 0.1434 | 0.0217 |
| 1024 | 0.1185 | 0.0726 | 0.0214 |

Share of cells where `|stationary - finite| > 0.02`:

| cost weight | finite 30d | finite 365d | finite 1824d |
| ---: | ---: | ---: | ---: |
| 0 | 34.6% | 13.4% | 0.1% |
| 4 | 71.5% | 23.2% | 1.1% |
| 16 | 87.0% | 40.7% | 5.6% |
| 64 | 86.4% | 64.6% | 11.8% |
| 256 | 67.2% | 54.1% | 13.6% |
| 1024 | 42.0% | 33.8% | 10.3% |

## Interpretation

`fsrs6_oracle_continuous_stationary_finite` compresses away the remaining-time
axis by learning one S/D surface per cost weight. The resulting surface is
structured and interpretable: low `S` is pinned by one-day interval geometry,
high `S` carries most of the cost tradeoff, and high `D` lowers the value of
review at nontrivial costs.

Compared with `fsrs6_oracle_continuous_retention`, the stationary policy keeps
the long-horizon part of the policy but loses the terminal timing behavior. That
is the main behavioral approximation: it does not know whether there are 30
days or 1824 days left. This also explains why the stationary continuous finite
row is strong but below the finite continuous-retention row in the first-eight
frontier report. The missing dimension is not subtle noise; at mid costs,
hundreds of remaining days still leave large regions where the stationary
surface is materially higher than the finite policy.

## Caveats

These summaries average over the full S/D grid and over users. Rollout-weighted
state occupancy can emphasize a smaller subset of the surface. The charts are
therefore best used to understand policy geometry and approximation error, not
to estimate direct review counts.

The stored continuous-retention policy values are canonical desired retentions
for rounded intervals. At very low stability, the shortest attainable interval
can map to a retention above `retention_min`, which is why the low-S edge is not
equal to `0.5` even at high cost.
