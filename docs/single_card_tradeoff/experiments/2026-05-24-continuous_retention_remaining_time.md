# FSRS6 Continuous Retention Remaining-Time Policy Analysis

## Question

This report analyzes how `fsrs6_oracle_continuous_retention` changes its
finite-horizon desired-retention policy as remaining time changes. The goal is
to identify general rules in the policy surface, not just isolated user cases.

## Source

Policy source:
`fsrs6_oracle_continuous_retention`, first-eight FSRS6 benchmark users, same
profile as the first-eight oracle/distill/ADR frontier comparison.

Configuration:

- Users: `1,2,3,4,5,6,7,8`
- Lifecycle: `1825` days
- Remaining-day range analyzed: `1..1824`
- State grid: `64` log-spaced stability values x `32` difficulty values
- Cost weights:
  `0,0.25,0.5,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Retention bounds: `[0.5, 0.98]`
- DP cache: loaded `152/152` cached `continuous_retention` policy entries;
  no DP was rerun.

The policy table shape is `[user, cost_weight, remaining, stability,
difficulty]`. The analysis aggregates uniformly over users and grid cells. It
is intentionally table-level rather than rollout-occupancy-weighted.

Important semantic note: the DP represents the terminal no-review option
(`interval = remaining + 1`) as `retention_min`. Therefore
`retention_min_share` is a lower-bound/no-review signature, but it is not a
perfect count of terminal actions because an ordinary attainable low-retention
interval can also canonicalize to `0.5`.

## Artifacts

Artifact directory:
`artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/`

Key files:

- `remaining_summary.csv`: aggregate metrics for every cost weight and
  remaining day.
- `remaining_thresholds.csv`: stationary-like threshold summaries by cost
  weight.
- `remaining_landmarks.csv`: selected remaining-day landmarks.
- `selected_policy_surfaces.npz`: plotted aggregate surfaces.

## Visuals

![Mean retention and min-retention share by remaining time](../../../artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/mean_retention_and_min_share_by_remaining.png)

![Deadline sensitivity by remaining time](../../../artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/deadline_sensitivity_by_remaining.png)

![Stability remaining-time heatmaps](../../../artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/stability_remaining_heatmaps.png)

![Difficulty remaining-time heatmaps](../../../artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/difficulty_remaining_heatmaps.png)

![Landmark remaining days by cost weight](../../../artifacts/single_card_tradeoff/continuous_retention_remaining_time_analysis/landmark_remaining_by_weight.png)

## Main Patterns

The policy has a strong deadline mode. With one day remaining, every positive
cost weight has mean desired retention near the lower bound (`0.5003`) and
about `99.4%` of table cells at `retention_min`. Even with zero review cost,
`59.0%` of cells are at the lower bound with one day remaining. This is the
terminal option showing through the retention interface: if a review cannot
improve much before the horizon, the finite oracle often chooses not to review
again.

Longer remaining time raises the target retention, but the ceiling is set by
cost weight. At `1824` days remaining, mean desired retention ranges from
`0.9339` at `w=0` down to `0.6269` at `w=1024`. At `365` days remaining, the
same range is `0.8763` down to `0.5856`. The higher the cost weight, the more
the policy keeps high-stability states at the lower bound even far from the
deadline.

The finite policy is not stationary except very near the longest horizon. Using
the `1824`-day policy as the reference, the strict stationary-like condition
(`mean_abs_delta <= 0.005` and at most `1%` of cells differing by more than
`0.02`, for all later remaining days) is not reached until roughly
`1592..1813` remaining days depending on cost weight. Even the looser condition
(`mean_abs_delta <= 0.01` and at most `5%` of cells differing by more than
`0.02`) starts around `1125` days for low weights and around `1750` days for
the highest weights.

Stability is the dominant state axis. At `365` days remaining, the aggregate
retention range across stability is about `0.30..0.48` for representative
weights, while the range across difficulty is about `0.001..0.095`. Difficulty
still matters, especially at nonzero cost, but it mostly bends the bands already
set by stability and remaining time.

## Landmark Summary

Mean desired retention:

| cost weight | 1d | 7d | 30d | 365d | 1824d |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6519 | 0.7174 | 0.7764 | 0.8763 | 0.9339 |
| 4 | 0.5003 | 0.5322 | 0.6642 | 0.8484 | 0.9139 |
| 16 | 0.5003 | 0.5116 | 0.5549 | 0.7940 | 0.8931 |
| 64 | 0.5003 | 0.5085 | 0.5302 | 0.6818 | 0.8411 |
| 256 | 0.5003 | 0.5075 | 0.5282 | 0.5905 | 0.7148 |
| 1024 | 0.5003 | 0.5075 | 0.5282 | 0.5856 | 0.6269 |

Share of table cells at `retention_min`:

| cost weight | 1d | 7d | 30d | 365d | 1824d |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 59.0% | 45.3% | 33.0% | 12.1% | 0.0% |
| 4 | 99.4% | 87.8% | 51.7% | 13.2% | 0.0% |
| 16 | 99.4% | 93.4% | 79.8% | 19.7% | 0.4% |
| 64 | 99.4% | 94.3% | 87.3% | 39.5% | 5.0% |
| 256 | 99.4% | 94.4% | 87.8% | 66.5% | 26.8% |
| 1024 | 99.4% | 94.4% | 87.8% | 68.2% | 54.1% |

Share of cells differing from the `1824`-day policy by more than `0.02`:

| cost weight | 30d | 90d | 365d | 730d | 1125d | 1600d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 34.7% | 25.5% | 13.4% | 6.5% | 3.2% | 1.0% |
| 4 | 71.6% | 53.1% | 24.7% | 10.1% | 5.5% | 1.7% |
| 16 | 87.0% | 74.2% | 43.2% | 20.5% | 7.5% | 1.3% |
| 64 | 86.5% | 81.0% | 65.4% | 46.9% | 24.4% | 4.3% |
| 256 | 64.0% | 60.4% | 50.8% | 43.0% | 36.3% | 13.5% |
| 1024 | 35.3% | 33.5% | 27.0% | 24.2% | 22.0% | 16.0% |

## Interpretation

The finite continuous-retention oracle is best thought of as a retention policy
with an explicit deadline gate. When remaining time is short, the DP often
selects the terminal no-review option and exposes it as `retention_min`. As the
horizon opens up, target retention rises because earlier reviews have more time
to pay back future memory value.

Cost weight changes the whole surface rather than just shifting a scalar target.
Low weights push the long-horizon policy toward high retention, especially for
low-stability states. High weights keep a large lower-bound region even at the
start of the 1825-day lifecycle, because expensive reviews are only worthwhile
where the future memory payoff is large enough.

This explains why the finite continuous oracle can beat stationary continuous
finite policies: much of its advantage is a remaining-time-dependent terminal
and transition-value effect. A stationary policy can imitate the long-horizon
surface, but it cannot express the large short-horizon lower-bound region or the
gradual opening of higher-retention regions as remaining time grows.

## Caveats

The analysis weights every table cell equally. Rollout trajectories visit a
smaller and policy-dependent subset of `(S, D)` states, so these summaries are
about the policy table's geometry rather than operational state occupancy.

The lower-bound share should be read as a policy-surface signal, not an exact
review-skip rate. The simulator executes a retention by converting it to a
rounded integer interval, while the DP also uses `retention_min` to encode the
terminal no-review action.
