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

`add_1_4` materially improves the mean FSRS6-relative frontier metric and fixes
the user-2 low-weight failure, but it does not pass the strict original gate
because user 7 loses more than 2 relative-time-saved percentage points.

| treatment | train weights | mean time saved AUC | mean relative AUC | mean coverage | min coverage | user 2 formal AUC | user 2 dense-loww AUC | passes all gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sparse baseline | `0,16,64,256,1024` | 3.165 | 8.76% | 97.20% | 89.78% | -3.554 | 1.123 | - |
| add 1,4 | `0,1,4,16,64,256,1024` | 4.753 | 11.79% | 98.19% | 91.96% | 6.254 | 9.893 | no |

Mean relative AUC improves by `+3.03 pp`. User 2 moves from a negative formal
FSRS6-relative AUC to a positive one and gains `+8.77` deck-minutes/day on the
dense low-weight grid.

## Gate Failure

The only failed `add_1_4` gate is the per-user regression limit outside user 2.
User 7's formal relative AUC drops from `9.71%` to `7.33%`, a `-2.38 pp`
change. Coverage is not the problem: user 7 stays at `100%` coverage. The loss
comes from slightly worse same-target time saved on the same FSRS6 memory span.

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

Because `add_1_4` repaired user 2 but failed the other-user gate, the optional
diagnostic rows were run.

| treatment | train weights | mean relative AUC | min coverage | user 2 dense-loww AUC | worst non-user2 delta | passes all gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| add 1 only | `0,1,16,64,256,1024` | 11.82% | 96.05% | 9.497 | -0.67 pp | yes |
| add 4 only | `0,4,16,64,256,1024` | 12.02% | 92.15% | 8.099 | +0.11 pp | yes |
| add 1,2,4 | `0,1,2,4,16,64,256,1024` | 10.72% | 95.22% | 9.225 | -6.55 pp | no |

The optional rows show that either `1` or `4` alone can pass the strict gates in
this run. Adding `2` as well is not helpful for the first-eight aggregate.

## High-Memory Segment

For user 2 over the `9400-9750` memorized-card segment, `add_1_4` almost closes
the sparse-student failure and approaches the exact/ADR frontiers.

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
`0,1,4,16,64,256,1024`.

This is not a clean strict-gate promotion of the `add_1_4` row because user 7
regresses by `2.38 pp`, but it is the requested default change and it directly
addresses the failure mode that motivated the experiment: missing low-weight
labels around user 2's high-memory frontier bend. Future capacity or loss-shaping
runs should continue to watch user 7 and compare against the passing `add_1_only`
and `add_4_only` diagnostics.

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
