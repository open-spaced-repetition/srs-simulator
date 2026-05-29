# FSRS6 Cost-ADR Output Distribution Analysis

Date: 2026-05-29

## Question

Analyze how the default Cost-ADR policy outputs vary across stability `S`,
difficulty `D`, and cost weight `w`; identify cross-user regularities; and
explain why Cost-ADR improves efficiency relative to the FSRS6 baseline.

## Scope

Primary run:

`artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off`

This is the current default compressed Cost-ADR profile:

- users: 1..128
- action head: `desired_retention`
- feature version:
  `fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1`
- parameter count: 15
- retention bounds: `[0.30, 0.995]`
- cost weights: `0,1,2,4,8,16,32,48,64,96,128,192,256,384,512,1024`

Analysis artifacts:

`artifacts/rl_scheduler/cost_adr_output_distribution_analysis_2026_05_29`

Key files:

- `summary.json`
- `retention_by_cost.csv`
- `interval_by_cost.csv`
- `state_bin_summaries.csv`
- `coefficient_summary.csv`
- `performance_by_cost_weight.csv`
- `cost_weight_baseline_nearest_comparison.csv`

## Policy Formula

For this compressed retention-head formula:

```text
x_s = normalized log stability in [0, 1]
x_d = normalized difficulty in [0, 1]
z   = log1p(w) / log1p(1024)
phi = [1, x_s, x_d, x_s * x_d, x_s^2]

logit_R(S,D,w) =
    beta_base dot phi
  - softplus(beta_z  dot phi) * z
  - softplus(beta_z2 dot phi) * z^2

R(S,D,w) = 0.30 + (0.995 - 0.30) * sigmoid(logit_R)
```

The subtraction of non-negative `softplus(...)` terms means the output
retention is monotone non-increasing in `w` by construction. The scheduler then
converts `R(S,D,w)` to an FSRS6 interval using the user's FSRS6 forgetting curve.

## Method

I evaluated the 128 learned policies on a structural grid:

- `S`: 64 log-spaced points from `0.1` to `9125`
- `D`: 32 linearly spaced points from `1` to `10`
- `w`: the formal 16 Cost-ADR weights
- total evaluated policy points: `128 * 64 * 32 * 16 = 4,194,304`

For interval conversion, I loaded each user's FSRS6 weights from benchmark
partition `0`, matching the scheduler visualization path. Pareto metrics come
from the run's existing `analysis_summary.json`.

Important caveat: the S/D grid is a structural policy probe, not the empirical
review-state distribution encountered during simulation. Runtime efficiency is
therefore interpreted from the sweep/Pareto outputs, while the grid explains the
shape of the learned policy.

## Retention Distribution By Cost Weight

Across the uniform S/D grid and all 128 users:

| w | mean R | q05 | q25 | median | q75 | q95 | R<0.50 | R>0.90 | user median R IQR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.925 | 0.719 | 0.915 | 0.963 | 0.978 | 0.990 | 0.78% | 78.62% | 0.958-0.974 |
| 1 | 0.920 | 0.714 | 0.908 | 0.958 | 0.976 | 0.989 | 0.82% | 76.99% | 0.954-0.970 |
| 16 | 0.878 | 0.615 | 0.848 | 0.919 | 0.956 | 0.984 | 2.42% | 59.10% | 0.905-0.942 |
| 64 | 0.813 | 0.438 | 0.755 | 0.858 | 0.924 | 0.978 | 6.43% | 34.50% | 0.814-0.893 |
| 128 | 0.767 | 0.355 | 0.683 | 0.813 | 0.898 | 0.974 | 10.48% | 24.35% | 0.741-0.857 |
| 384 | 0.681 | 0.307 | 0.530 | 0.716 | 0.849 | 0.967 | 22.27% | 15.56% | 0.585-0.779 |
| 1024 | 0.608 | 0.301 | 0.394 | 0.610 | 0.798 | 0.959 | 37.35% | 11.66% | 0.436-0.702 |

The main pattern is smooth triage. Low `w` keeps most of the grid near high
retention. As `w` rises, the policy opens a large low-retention region while
preserving a high-retention tail. The high-cost policy is not a global "set
retention to 0.5" rule; at `w=1024`, 11.66% of sampled states still output
`R>0.90`.

All 128 users were monotone non-increasing in `w` at every sampled grid point.

## Cross-User Regularities

The strongest regularity is in the coefficient signs:

| group | feature | mean | sign regularity |
|---|---|---:|---|
| base | `x_s` | 8.28 | positive in 100% of users |
| base | `x_s^2` | -5.85 | negative in 100% of users |
| `z` slope | `x_s` | 24.75 | positive in 100% of users |
| `z` slope | `x_s^2` | -22.47 | negative in 100% of users |
| `z^2` slope | `x_s` | 23.80 | positive in 100% of users |
| `z^2` slope | `x_d` | 3.64 | positive in 95.31% of users |
| `z^2` slope | `x_s^2` | -17.78 | negative in 100% of users |

No coefficient had material boundary clipping: the fraction with `abs(param)>60`
was 0% for all 15 parameters under the `[-64, 64]` bounds.

The generalizable shape is therefore:

- base retention rises strongly with log stability, but concavely;
- cost sensitivity also rises with log stability, again concavely;
- at high cost, difficulty contributes to a larger penalty for most users.

The absolute policy level remains personalized. Median cross-user retention
standard deviation at a fixed S/D grid point increases from `0.059` at `w=0` to
`0.143` at `w=128` and `0.155` at `w=1024`. High-cost behavior exposes more
user-specific tradeoff differences.

## State Effects

User-level finite-difference diagnostics:

| diagnostic | q25 | median | q75 |
|---|---:|---:|---:|
| median retention drop, `w=0 -> 1024` | 0.166 | 0.312 | 0.499 |
| median retention drop, `w=16 -> 128` | 0.041 | 0.070 | 0.134 |
| high-S minus low-S retention at `w=128` | -0.118 | 0.060 | 0.175 |
| hard-D minus easy-D retention at `w=128` | -0.290 | -0.185 | -0.101 |
| high-S cost sensitivity, `w=0 -> 1024` | 0.068 | 0.275 | 0.582 |
| low-S cost sensitivity, `w=0 -> 1024` | 0.046 | 0.082 | 0.136 |

The robust part is the cost response: mature/high-S states are adjusted much
more strongly as `w` increases than fragile/low-S states. This is exactly where
Cost-ADR has more freedom: high-S cards can tolerate long intervals and still
remain useful, while low-S cards often hit the one-day floor and are more easily
damaged by excessive relaxation.

The difficulty effect is also consistent: at `w=128`, hard states usually get
lower retention than easy states. This is a triage signal. Under time pressure,
hard cards have worse return per unit review time, so the policy stops trying to
protect every hard state equally.

Selected median retention slices:

| w | S bin | D bin | median R | q25 | q75 |
|---:|---|---|---:|---:|---:|
| 0 | low S | easy D | 0.863 | 0.773 | 0.920 |
| 0 | low S | hard D | 0.853 | 0.764 | 0.910 |
| 0 | high S | easy D | 0.975 | 0.962 | 0.983 |
| 0 | high S | hard D | 0.978 | 0.957 | 0.986 |
| 128 | low S | easy D | 0.842 | 0.760 | 0.899 |
| 128 | low S | hard D | 0.730 | 0.652 | 0.808 |
| 128 | high S | easy D | 0.923 | 0.831 | 0.961 |
| 128 | high S | hard D | 0.809 | 0.495 | 0.943 |
| 1024 | low S | easy D | 0.820 | 0.739 | 0.881 |
| 1024 | low S | hard D | 0.630 | 0.502 | 0.738 |
| 1024 | high S | easy D | 0.804 | 0.513 | 0.934 |
| 1024 | high S | hard D | 0.460 | 0.307 | 0.833 |

## Runtime Effect By Cost Weight

In the FSRS6 evaluation environment, increasing `w` moves smoothly along the
time-memory frontier:

| w | mean time | mean memorized | mean acc. memorized/hour | mean reviews |
|---:|---:|---:|---:|---:|
| 0 | 128.11 | 7073.09 | 14.00 | 521.27 |
| 1 | 119.65 | 7058.44 | 14.99 | 480.66 |
| 16 | 76.86 | 6865.76 | 21.06 | 288.06 |
| 64 | 47.83 | 6428.61 | 26.89 | 163.59 |
| 128 | 35.18 | 6069.98 | 29.82 | 112.88 |
| 384 | 20.88 | 5358.13 | 36.21 | 60.23 |
| 1024 | 14.75 | 4862.23 | 41.14 | 38.85 |

From `w=0` to `w=1024`, mean time falls by 88.5% and reviews fall by 92.5%,
while mean memorized cards fall by 31.3%. The accumulated-memorized-per-hour
metric rises from `14.00` to `41.14`.

The LSTM transfer environment shows the same direction:

| w | mean time | mean memorized | mean acc. memorized/hour | mean reviews |
|---:|---:|---:|---:|---:|
| 0 | 159.47 | 6958.11 | 12.63 | 684.62 |
| 16 | 80.53 | 6749.14 | 20.13 | 331.00 |
| 64 | 48.66 | 6356.69 | 26.88 | 182.21 |
| 128 | 36.42 | 6043.05 | 30.19 | 128.16 |
| 384 | 23.33 | 5461.95 | 37.84 | 73.43 |
| 1024 | 17.56 | 5044.33 | 43.36 | 50.27 |

## Pareto Results Relative To FSRS6

Authoritative Pareto metrics from the 128-user run:

| env | HV delta | HV delta % | rel same-target time-save AUC | target coverage | rel same-budget memory-lift AUC | budget coverage |
|---|---:|---:|---:|---:|---:|---:|
| FSRS6 | 1,432,968.83 | 4.168% | 11.447% | 90.746% | 1.898% | 83.644% |
| LSTM | 599,474.50 | 1.497% | 2.568% | 92.917% | 0.274% | 79.558% |

The FSRS6 gain is larger because the policy was trained in the FSRS6
environment. LSTM still gets positive transfer, but the same state-conditioned
retention surface is less perfectly aligned with the LSTM memory dynamics.

## Why It Beats A Fixed-Retention FSRS6 Baseline

The FSRS6 baseline frontier is built from scalar desired-retention schedules.
For a given baseline point, every state uses the same retention target, and the
interval can only vary through `S`. Difficulty affects future state updates but
not the current target retention.

Cost-ADR adds two degrees of control:

1. It conditions retention on state.
   Low-S and high-S cards do not need the same target. The learned policy keeps
   fragile regions relatively protected and spends less on regions with lower
   marginal payoff.

2. It conditions the whole surface on cost weight.
   The formal 16 weights do not merely pick 16 scalar retentions. They sweep a
   family of state-dependent retention surfaces. This gives denser and more
   useful frontier coverage than fixed FSRS6 DRs.

The efficiency mechanism is triage, not uniformly shorter intervals. At high
cost weight, the policy sharply reduces retention in parts of the state space
where reviews are expensive relative to memory payoff, but it keeps a
high-retention tail where the marginal value remains good. That combination
explains why time and reviews collapse much faster than memorized cards.

## Discussion

The strongest cross-user law is structural: the same coefficient signs and the
same monotone cost response appear across all 128 users. The personalized part
is the magnitude of the high-cost relaxation. That is why a compact 15-parameter
formula can generalize: it hard-codes the right monotonicity and uses enough
state features to express the common shape, while still letting each user choose
how aggressively to relax the surface.

The analysis also explains why the compressed formula worked better than the
earlier larger interval-head variants on the 128-user FSRS6 metric. The formula
does not waste parameters modeling weak directions (`sqrt_z`, `x_d^2`) and
keeps the search focused on the dominant pattern: a concave stability effect
plus a monotone, stability-sensitive cost penalty.

Remaining uncertainty: this report uses a uniform S/D probe. A stronger next
analysis would log or reconstruct the empirical distribution of scheduler
states visited during simulation and reweight the policy-output summaries by
actual state occupancy. That would make the interval distribution directly
match runtime behavior.
