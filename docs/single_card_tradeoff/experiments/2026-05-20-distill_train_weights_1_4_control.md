# Stationary Finite Distill Train-Weight 1 and 4 Control

Config:
`experiments/single_card_tradeoff/configs/distill_train_weights_1_4_control_markov_off.toml`

Comparison artifacts:
`artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/`

## Question

Does adding low cost weights `1` and `4` to stationary finite distill training
repair the user-2 high-memory frontier regression without changing the
476-parameter per-user student?

## Setup

- Environment: `fsrs6`
- Users: 1-8
- Review Markov transition: `false`
- Teacher: repaired stationary finite, 64x32 table, clipped 11-action grid
- Student: per-user `residual:8:2`, 476 parameters, `uniform_table`
  supervision
- Epochs: 128
- Steps per epoch: 64
- Table samples per weight: 256
- Eval particles: 10,000
- Seed: 42
- Device: CUDA

## Primary Result

`add_4_only` is the selected default. It clears every gate and has the best mean
FSRS6-relative frontier metric among the passing rows, while avoiding the user-7
regression that blocks `add_1_4`.

| treatment | train weights | mean time saved AUC | mean relative AUC | mean coverage | min coverage | user 2 formal AUC | user 2 dense-loww AUC | passes all gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sparse baseline | `0,16,64,256,1024` | 3.165 | 8.76% | 97.20% | 89.78% | -3.554 | 1.123 | - |
| add 1 only | `0,1,16,64,256,1024` | 4.548 | 11.82% | 98.72% | 96.05% | 6.067 | 9.497 | yes |
| add 4 only | `0,4,16,64,256,1024` | 4.764 | 12.02% | 98.31% | 92.15% | 5.366 | 8.099 | yes |
| add 1,4 | `0,1,4,16,64,256,1024` | 4.753 | 11.79% | 98.19% | 91.96% | 6.254 | 9.893 | no |

Mean relative AUC improves by `+3.26 pp` over the sparse baseline. User 2 moves
from a negative formal FSRS6-relative AUC to a positive one under `add_4_only`
and gains `+6.98` deck-minutes/day on the dense low-weight grid.

## Gate Comparison

`add_1_4` still delivers the strongest user-2 low-weight repair, but it fails
the other-user gate. User 7's formal relative AUC drops from `9.71%` to
`7.33%`, a `-2.38 pp` change. Coverage is not the problem: user 7 stays at
`100%` coverage. The loss comes from slightly worse same-target time saved on
the same FSRS6 memory span.

At representative user-7 FSRS6 memory targets:

| target memorized | sparse saved | add 1,4 saved | delta |
| ---: | ---: | ---: | ---: |
| 8,878 | 0.471 | 0.397 | -0.074 |
| 9,595 | 0.202 | 0.153 | -0.050 |
| 9,827 | 1.112 | 0.948 | -0.164 |
| 9,899 | 1.326 | 1.259 | -0.067 |

The learned frontier shifts in the middle/high-memory region. This looks like an
optimization/interpolation interaction from supervising both `1` and `4`, not a
coverage failure.

## Optional Diagnostics

The optional `add_1_2_4` diagnostic was also run. It does not recover the
first-eight aggregate and remains the clearest fail of the extra-low-weight
variant.

| treatment | train weights | mean relative AUC | min coverage | user 2 dense-loww AUC | worst non-user2 delta | passes all gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| add 1,2,4 | `0,1,2,4,16,64,256,1024` | 10.72% | 95.22% | 9.225 | -6.55 pp | no |

## High-Memory Segment

For user 2 over the `9400-9750` memorized-card segment, `add_1_4` still has the
strongest local repair, but `add_4_only` is the selected aggregate winner
because it clears every gate and has the best mean relative AUC among the
passing rows.

| scheduler | segment AUC vs FSRS6 | relative |
| --- | ---: | ---: |
| exact stationary finite | 51.243 | 15.28% |
| ADR | 44.916 | 13.74% |
| sparse baseline | -17.724 | -5.28% |
| add 1,4 | 42.912 | 12.80% |
| add 1 only | 41.899 | 12.49% |
| add 4 only | 23.821 | 7.10% |
| add 1,2,4 | 38.128 | 11.37% |

## GPU Monitor

No training or exact-value evaluation run reported shared-memory spill.

| run | shared-memory spill | peak shared memory | peak FB memory |
| --- | --- | ---: | ---: |
| sparse baseline train | false | 646.0 MiB | 4,122 MiB |
| add 1,4 train | false | 508.0 MiB | 4,279 MiB |
| add 1 only train | false | 534.2 MiB | 4,158 MiB |
| add 4 only train | false | 479.0 MiB | 4,079 MiB |
| add 1,2,4 train | false | 447.6 MiB | 4,037 MiB |
| formal exact-value eval | false | 447.0 MiB | 9,711 MiB |
| dense-loww user-2 eval | false | 446.7 MiB | 4,249 MiB |

## Decision

The default stationary finite distill training weights are updated to
`0,4,16,64,256,1024`.

This is a clean promotion of `add_4_only`, not `add_1_4`, because the selected
row passes every gate while `add_1_4` still regresses user 7 by `2.38 pp`.
Future capacity or loss-shaping runs should continue to watch user 7 and
compare against the passing `add_1_only` diagnostic.

## Artifacts

- Summary CSV:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_summary.csv`
- By-user CSV:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_by_user.csv`
- Direct pairwise CSV:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_direct_vs_exact.csv`
- Dense-low-weight user-2 CSV:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_user2_dense_loww.csv`
- Segment AUC CSV:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_user2_segment_auc.csv`
- Frontier plots:
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_user2_frontier_formal.png`
  and
  `artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off/train_weight_user2_frontier_dense_loww.png`
