# Rollout-Distribution Supervised Distill, Four-Corner Teacher Rerun

## Question

After the stationary finite four-corner transition and bilinear value lookup
fixes, does the old conclusion still hold that pure rollout-distribution
supervision is the wrong distill target?

## Setup

- Environment: `fsrs6`
- Users: 1-8
- Review Markov transition: `false`
- Student: one residual 476-parameter policy per user
- Training: 128 epochs, 64 steps per epoch, 1,280 train envs per user
- Teacher solve: stationary finite, `--oracle-teacher-user-batch-size 2`
- Training cost weights: `0,16,64,256,1024`
- Evaluation cost weights:
  `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`
- Evaluation: TOML-configured `run_tradeoff_config.py --force`, 1,825 days,
  10,000 particles, deck scale 10,000, seed 42, ADR Markov-off portfolio
  baseline
- Torch device: CUDA

The four formal variants were:

- `64x32` teacher + uniform-table supervision
- `128x64` teacher + uniform-table supervision
- `64x32` teacher + rollout-distribution supervision
- `128x64` teacher + rollout-distribution supervision

## Result

| Variant | Mean AUC vs FSRS6 | Relative | Coverage | Min coverage | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform 64x32 | 3.2854 | 8.99% | 96.91% | 89.56% | 82.92% |
| Uniform 128x64 | 2.5949 | 7.81% | 97.21% | 91.01% | 81.91% |
| Rollout 64x32 | -11.1427 | -7.63% | 93.02% | 80.68% | 93.41% |
| Rollout 128x64 | -12.0906 | -10.82% | 95.14% | 85.82% | 92.44% |

Pure rollout supervision still fails the gate. It again improves teacher
agreement, but the deployed tradeoff frontier is much worse than uniform-table
supervision.

User 2 remains broken by the defined criterion:

| Variant | User 2 AUC vs FSRS6 | Coverage |
| --- | ---: | ---: |
| Uniform 64x32 | -3.8961 | 99.66% |
| Uniform 128x64 | -8.8529 | 99.72% |
| Rollout 64x32 | -67.0055 | 80.68% |
| Rollout 128x64 | -65.0515 | 85.82% |

Against ADR on the shared span, rollout supervision is also much worse:

| Variant | Mean direct AUC vs ADR |
| --- | ---: |
| Uniform 64x32 | -3.4992 |
| Uniform 128x64 | -4.7510 |
| Rollout 64x32 | -22.0191 |
| Rollout 128x64 | -24.4091 |

## Gate Check

| Gate | 64x32 rollout | 128x64 rollout |
| --- | ---: | ---: |
| Relative time saved worse than matching uniform by at least 3 pp | 16.62 pp | 18.63 pp |
| Coverage worse than matching uniform by at least 2 pp | 3.89 pp | 2.07 pp |
| User 2 negative AUC or coverage below 80% | pass: AUC -67.0055 | pass: AUC -65.0515 |
| Mean direct AUC vs ADR worse than -3.0 | -22.0191 | -24.4091 |

Conclusion: the old qualitative conclusion is consistent. The four-corner fix
does not rescue pure rollout-distribution supervision.

## Comparison To Old Report

The old rollout results were already bad:

| Variant | Old AUC | Old relative | Old coverage | Old user 2 AUC | Old user 2 coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rollout 64x32 | 0.5440 | 4.66% | 91.94% | -14.5230 | 62.28% |
| Rollout 128x64 | 0.8147 | 6.52% | 93.88% | -11.5810 | 65.86% |

The four-corner rerun improves rollout coverage but makes deployed AUC much
worse, especially for user 2. The failure mode therefore did not disappear; it
shifted from low coverage alone to severe bad frontier placement on the covered
span.

## GPU Monitor

All training and evaluation runs wrote GPU monitor artifacts. Shared-memory
spill was `false` for every run; peak summed shared GPU memory stayed below
1 GiB.

| Variant | Train peak FB MiB | Train peak shared MiB | Eval peak FB MiB | Eval peak shared MiB |
| --- | ---: | ---: | ---: | ---: |
| Uniform 64x32 | 2,137 | 366.68 | 2,110 | 231.24 |
| Uniform 128x64 | 7,487 | 241.69 | 2,102 | 226.35 |
| Rollout 64x32 | 2,318 | 390.79 | 2,156 | 404.50 |
| Rollout 128x64 | 2,438 | 233.02 | 2,156 | 238.41 |

## Artifacts

- Combined comparison:
  `artifacts/single_card_tradeoff/rollout_supervised_distill_first8_users_four_corner_markov_off/variant_summary.csv`
- Per-user comparison:
  `artifacts/single_card_tradeoff/rollout_supervised_distill_first8_users_four_corner_markov_off/per_user_auc_comparison.csv`
- Combined plots:
  `artifacts/single_card_tradeoff/rollout_supervised_distill_first8_users_four_corner_markov_off/distill_supervision_variant_auc_by_user.png`,
  `artifacts/single_card_tradeoff/rollout_supervised_distill_first8_users_four_corner_markov_off/distill_supervision_variant_coverage_by_user.png`
- Configs:
  `experiments/single_card_tradeoff/configs/adr_vs_64x32_uniform_table_supervised_distill_first8_users_four_corner.toml`,
  `experiments/single_card_tradeoff/configs/adr_vs_128x64_uniform_table_supervised_distill_first8_users_four_corner.toml`,
  `experiments/single_card_tradeoff/configs/adr_vs_64x32_rollout_supervised_distill_first8_users_four_corner.toml`,
  `experiments/single_card_tradeoff/configs/adr_vs_128x64_rollout_supervised_distill_first8_users_four_corner.toml`

## Note

The four-corner rerun also made the uniform-table user 2 result negative under
the same sparse eval grid. That is separate from the rollout question, but it
means follow-up distill work should keep using direct pairwise checks and denser
low-weight scalarization diagnostics for user 2.
