# FSRS6 ADR pop16 baseline DR selection report

Date: 2026-05-21

## Question

Select 16 per-user FSRS-6 baseline desired-retention values for the
`fsrs6_adr_portfolio_users_1_8_pop16_v1` portfolio config.

## Run

| config | selection env | target count | population | generations | reference | seed |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `fsrs6` | 16 | 16 | 5 | `uniform_anchor` | 42 |

Command:

```bash
uv run python experiments/rl_scheduler/select_fsrs6_baseline_drs.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml
```

## Summary

The selector started from the uniform 16-point anchor over `0.50..0.98` and
optimized per-user DR vectors with CMA-ES in the `fsrs6` environment.

- Total anchor hypervolume: `3,655,082.462`
- Total selected hypervolume: `3,687,030.135`
- Total gain: `31,947.674` (`+0.874%`)
- All 8 users improved over the uniform anchor
- Best absolute gain: user 4, `+14,124.970` (`+1.258%`)

Full precision values are in
`artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json`.

## Selected DRs

Values below are rounded to 3 decimals for readability.

| user | selected DR values | anchor HV | final HV | gain |
| ---: | --- | ---: | ---: | ---: |
| 1 | `0.515, 0.546, 0.572, 0.615, 0.630, 0.633, 0.668, 0.749, 0.783, 0.831, 0.870, 0.908, 0.930, 0.940, 0.963, 0.980` | 691,026.760 | 697,898.773 | +6,872.013 |
| 2 | `0.517, 0.523, 0.534, 0.563, 0.572, 0.681, 0.713, 0.730, 0.794, 0.831, 0.840, 0.891, 0.924, 0.944, 0.960, 0.973` | 1,380,059.257 | 1,387,521.037 | +7,461.781 |
| 3 | `0.506, 0.556, 0.587, 0.616, 0.707, 0.708, 0.743, 0.766, 0.806, 0.854, 0.864, 0.884, 0.897, 0.922, 0.960, 0.974` | 141,829.541 | 142,586.172 | +756.632 |
| 4 | `0.501, 0.543, 0.557, 0.586, 0.614, 0.744, 0.757, 0.812, 0.859, 0.894, 0.918, 0.922, 0.948, 0.951, 0.963, 0.976` | 1,122,590.431 | 1,136,715.401 | +14,124.970 |
| 5 | `0.500, 0.505, 0.576, 0.592, 0.631, 0.634, 0.673, 0.696, 0.785, 0.813, 0.870, 0.898, 0.922, 0.949, 0.962, 0.969` | 117,209.704 | 117,617.633 | +407.929 |
| 6 | `0.530, 0.542, 0.555, 0.560, 0.598, 0.644, 0.729, 0.764, 0.812, 0.865, 0.899, 0.903, 0.937, 0.953, 0.962, 0.977` | 162,259.353 | 164,022.188 | +1,762.834 |
| 7 | `0.505, 0.519, 0.559, 0.596, 0.606, 0.628, 0.650, 0.657, 0.697, 0.771, 0.795, 0.850, 0.885, 0.942, 0.968, 0.980` | 17,420.155 | 17,499.270 | +79.115 |
| 8 | `0.503, 0.504, 0.561, 0.567, 0.631, 0.684, 0.714, 0.747, 0.858, 0.888, 0.910, 0.934, 0.952, 0.957, 0.968, 0.975` | 22,687.261 | 23,169.662 | +482.400 |

## Provenance

- Config commit: `105a5351f8f9f15c805462d49076f59f20c9559f`
- Config sha256: `70e14044c7f49a7448c78122383bebdca9c70eb708b4014c27b3c8d99e029c8f`
- Manifest path: `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json`
- Progress log: `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.progress.jsonl`

