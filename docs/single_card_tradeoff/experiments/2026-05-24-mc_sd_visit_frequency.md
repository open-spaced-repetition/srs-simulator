# FSRS6 Monte Carlo S/D Visit Frequency

## Question

This report visualizes how often different `(S, D)` regions are visited during
Monte Carlo rollouts at different cost weights.

The visit definition is event-based: one visit is one active trajectory state
immediately before the scheduler chooses the next interval. The initial
learning state is included. This is not day-weighted occupancy.

## Source

Policies:

- `fsrs6_oracle_continuous_retention`
- `fsrs6_oracle_continuous_stationary_finite`

Configuration:

- Users: `1,2,3,4,5,6,7,8`
- Cost weights visualized: `0,4,16,64,256,1024`
- Particles: `2048` per user and cost weight
- Lifecycle: `1825` days
- State bins: nearest cell on the `64 x 32` oracle `(log S, D)` grid
- Retention bounds: `[0.5, 0.98]`
- Policy cache: `48/48` finite continuous-retention hits and `48/48`
  stationary continuous-finite hits; no DP was rerun.

The rollout covers `8 * 6 * 2048 = 98,304` trajectories per scheduler. Total
decision-state visits were `5,089,307` for `continuous_retention` and
`5,117,867` for `continuous_stationary_finite`.

## Artifacts

Artifact directory:
`artifacts/single_card_tradeoff/mc_sd_visit_frequency/`

Key files:

- `occupancy_counts.npz`: raw counts
  `[user, cost_weight, stability, difficulty]` plus decision counts.
- `occupancy_metrics_by_user_weight.csv`: rollout metrics and decision visits
  by user and cost weight.
- `occupancy_summary_by_scheduler_weight.csv`: aggregate occupancy summaries.
- `occupancy_nonzero_cells.csv`: nonzero aggregate visit cells.

## Visuals

![Continuous-retention visit heatmaps](../../../artifacts/single_card_tradeoff/mc_sd_visit_frequency/occupancy_heatmaps_continuous_retention.png)

![Continuous stationary finite visit heatmaps](../../../artifacts/single_card_tradeoff/mc_sd_visit_frequency/occupancy_heatmaps_continuous_stationary_finite.png)

![Stationary-vs-finite visit ratio](../../../artifacts/single_card_tradeoff/mc_sd_visit_frequency/occupancy_log_ratio_stationary_vs_finite.png)

![Occupancy marginals and visit counts](../../../artifacts/single_card_tradeoff/mc_sd_visit_frequency/occupancy_marginals_and_visit_counts.png)

![Occupancy concentration summary](../../../artifacts/single_card_tradeoff/mc_sd_visit_frequency/occupancy_concentration_summary.png)

## Main Patterns

Cost weight mainly changes how many decision states are visited. For
`continuous_retention`, mean decision visits per trajectory fall from `204.0`
at `w=0` to `6.0` at `w=1024`. The stationary policy is similar, falling from
`204.9` to `6.4`. This matches the intended tradeoff: high review cost makes
the scheduler choose much longer intervals and terminate with far fewer review
decisions.

The visited S/D region is broad at low and moderate cost but contracts at high
cost. Under `continuous_retention`, the occupied grid-cell share falls from
`66.2%` at `w=0` to `41.9%` at `w=1024`. The high-cost policy visits many fewer
states and concentrates around the state regions that remain reachable after
very sparse reviews.

The stability distribution is nonmonotonic in cost. The geometric mean visited
stability rises from `14.2d` at `w=0` to about `30.2d` at `w=16`, then falls to
`6.9d` at `w=1024` for `continuous_retention`. Moderate costs let cards climb
to higher stability while avoiding the very frequent low-cost review regime.
Very high costs produce few decision points; the distribution is then dominated
by early/low-stability decisions plus a small number of long-interval returns.

Difficulty shifts downward as cost rises through the middle of the range. For
`continuous_retention`, mean visited difficulty goes from `8.71` at `w=0` to
`5.44` at `w=256`, then rises to `6.11` at `w=1024`. Low-cost rollouts keep
reviewing and can repeatedly visit high-difficulty states. Mid/high cost
policies review less and avoid many of those repeated difficult-state visits.

The stationary and finite policies have very similar occupancy distributions at
low and middle costs, even though their full policy tables differ substantially
near the deadline. The main visible difference is at the highest cost: at
`w=1024`, stationary has more visits per trajectory (`6.38` vs `5.96`), higher
geometric mean stability (`7.93d` vs `6.88d`), and more visits with
`S >= 365d` (`2.61%` vs `0.02%`). This is consistent with the previous policy
surface analysis: stationary cannot express finite-horizon terminal timing, so
it keeps a slightly more long-horizon-like review pattern in sparse-review
regimes.

## Summary Table

`continuous_retention`:

| cost weight | visits / particle | occupied cells | top-cell share | geo mean S | S >= 30d | S >= 365d | mean D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 204.01 | 66.2% | 2.6% | 14.22 | 27.0% | 3.1% | 8.71 |
| 4 | 52.23 | 65.4% | 2.1% | 24.55 | 44.6% | 10.8% | 7.17 |
| 16 | 26.48 | 63.5% | 1.4% | 30.17 | 49.1% | 14.0% | 6.34 |
| 64 | 13.73 | 60.9% | 1.8% | 24.78 | 45.0% | 12.9% | 5.67 |
| 256 | 8.22 | 52.2% | 1.9% | 13.73 | 33.5% | 7.6% | 5.44 |
| 1024 | 5.96 | 41.9% | 1.9% | 6.88 | 18.8% | 0.0% | 6.11 |

`continuous_stationary_finite`:

| cost weight | visits / particle | occupied cells | top-cell share | geo mean S | S >= 30d | S >= 365d | mean D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 204.94 | 66.8% | 2.6% | 14.23 | 27.0% | 3.1% | 8.72 |
| 4 | 52.47 | 65.6% | 2.0% | 24.72 | 44.9% | 11.0% | 7.15 |
| 16 | 26.65 | 63.5% | 1.4% | 30.70 | 49.3% | 14.2% | 6.34 |
| 64 | 13.71 | 61.1% | 1.7% | 24.62 | 44.9% | 12.4% | 5.68 |
| 256 | 8.23 | 52.5% | 1.7% | 13.37 | 33.0% | 6.7% | 5.50 |
| 1024 | 6.38 | 43.7% | 2.3% | 7.93 | 21.6% | 2.6% | 5.90 |

## Interpretation

The earlier S/D policy-surface reports describe what the exact policy would do
over the whole grid. This MC occupancy view shows which parts of that grid are
actually exercised. Large table regions can be behaviorally less important if
rollouts rarely visit them.

The practical takeaway is that cost-weight changes affect both the action
surface and the state distribution. At low cost, the policy repeatedly reviews,
spreading visits across many S/D cells and accumulating many high-difficulty
decision states. At moderate cost, trajectories still get enough reviews to
reach higher stability, but they visit fewer states overall. At very high cost,
decision events become sparse and the visitation distribution contracts.

The stationary approximation looks closer under MC occupancy than under
full-grid table comparison for low and middle weights. Its biggest occupancy
difference appears in sparse-review high-cost settings, where the lack of a
finite-horizon deadline mode changes which high-stability states remain visited.

## Caveats

The heatmaps are normalized by decision visits within each scheduler and cost
weight. They show where visits happen, not how many visits happen in absolute
terms. Use the `visits / particle` metric alongside the heatmaps when comparing
cost weights.

The binning maps continuous simulator states to the nearest oracle grid cell.
That keeps the visualization aligned with the policy tables, but it is still a
grid approximation of continuous `(S, D)` trajectories.
