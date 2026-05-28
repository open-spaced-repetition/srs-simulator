# FSRS6 Cost-ADR Retention-head Structure Ablation

Date: 2026-05-28

## Question

Train compressed Cost-ADR desired-retention policies on the first eight users in
the FSRS6 environment and compare whether the 24-parameter structure can be
reduced without losing Pareto quality. A supplemental LSTM-environment sweep was
then run from the same trained policy artifacts to check transfer.

All training runs use users 1-8, the FSRS6 environment, pop16/gen20 CMA-ES,
`sigma0 = 1.0`, no coefficient preconditioning, the first-eight
interval-implied-R mean initializer projected into each compressed structure,
and the matched 16 cost weights. The main evaluation is FSRS6; the supplemental
section reruns the sweep in LSTM.

## Runs

| variant | config | params | removed structure |
| --- | --- | ---: | --- |
| full reference | previous `fsrs6_cost_adr_rethead_meaninit_nopre_ablation_users_1_8_pop16_gen20_v1_20260528T024902Z` | 24 | none |
| drop `sqrt_z` | `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1.toml` | 18 | `sqrt(z)` cost basis |
| drop `x_d^2` | `fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1.toml` | 20 | difficulty quadratic state feature |
| drop `sqrt_z + x_d^2` | `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1.toml` | 15 | both above |
| `z2` only | `fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1.toml` | 12 | `sqrt(z)` and `z` cost bases |

All four new FSRS6 ablation runs passed `dry-run`, `preflight`,
`stage-baseline`, `train-overfit`, `sweep`, `build-pareto`, and
`analyze-pareto`.

Supplemental LSTM eval runs were created under
`artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_lstm_eval_users_1_8`.
They reuse the FSRS6-trained `train-overfit` artifacts and rerun only
`stage-baseline`, `sweep`, `build-pareto`, and `analyze-pareto` with
`envs = ["lstm"]`.

## Results

Primary metric is scheduler-only hypervolume delta against the FSRS6 baseline
frontier. Time-save and memory-lift AUCs are interpolation diagnostics over the
common covered spans reported by `analysis_summary.json`.

| variant | params | HV delta | HV / baseline | time-save AUC | relative time-save AUC | target span coverage | memory-lift AUC | relative memory-lift AUC | budget span coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full reference | 24 | 102,728 | 3.323% | 4.405 | 13.065% | 94.818% | 89.41 | 1.364% | 86.932% |
| drop `sqrt_z` | 18 | 104,175 | 3.370% | 4.652 | 13.016% | 94.070% | 76.88 | 1.148% | 89.668% |
| drop `x_d^2` | 20 | 103,019 | 3.332% | 4.486 | 12.927% | 93.560% | 78.69 | 1.181% | 96.751% |
| drop `sqrt_z + x_d^2` | 15 | 106,075 | 3.431% | 4.639 | 13.707% | 90.231% | 98.52 | 1.507% | 74.560% |
| `z2` only | 12 | 99,496 | 3.218% | 5.304 | 13.456% | 88.203% | 91.69 | 1.389% | 78.552% |

Per-user FSRS6 HV delta:

| variant | u1 | u2 | u3 | u4 | u5 | u6 | u7 | u8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full reference | 15,410 | 29,858 | 3,161 | 37,354 | 9,024 | 4,027 | 1,374 | 2,521 |
| drop `sqrt_z` | 14,152 | 27,155 | 3,851 | 40,863 | 9,649 | 4,945 | 1,163 | 2,398 |
| drop `x_d^2` | 16,561 | 29,404 | 3,767 | 36,252 | 9,053 | 4,324 | 1,259 | 2,399 |
| drop `sqrt_z + x_d^2` | 14,708 | 30,097 | 3,659 | 40,098 | 9,286 | 4,415 | 1,302 | 2,511 |
| `z2` only | 15,084 | 25,948 | 3,951 | 38,071 | 9,419 | 3,308 | 1,291 | 2,424 |

## LSTM Eval Supplement

These rows evaluate the same FSRS6-trained policies in the LSTM environment
against the FSRS6 baseline scheduler. The 24-parameter full reference is the
existing no-preconditioning run with both FSRS6 and LSTM eval enabled.

| variant | params | LSTM HV delta | HV / baseline | time-save AUC | relative time-save AUC | target span coverage | memory-lift AUC | relative memory-lift AUC | budget span coverage | frontier points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full reference | 24 | 26,097 | 0.810% | -0.724 | 2.891% | 99.054% | 41.32 | 0.641% | 91.286% | 121 |
| drop `sqrt_z` | 18 | 47,522 | 1.476% | 1.102 | 5.293% | 98.588% | 42.61 | 0.644% | 90.224% | 125 |
| drop `x_d^2` | 20 | 51,870 | 1.611% | 1.782 | 5.312% | 97.349% | 40.66 | 0.615% | 94.015% | 123 |
| drop `sqrt_z + x_d^2` | 15 | 54,322 | 1.687% | 2.411 | 5.976% | 94.562% | 56.64 | 0.882% | 77.765% | 119 |
| `z2` only | 12 | 39,437 | 1.225% | -0.012 | 3.457% | 96.105% | 38.92 | 0.595% | 81.336% | 124 |

Per-user LSTM HV delta:

| variant | u1 | u2 | u3 | u4 | u5 | u6 | u7 | u8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full reference | 13,738 | -5,009 | 2,251 | 8,683 | 742 | 4,073 | 852 | 768 |
| drop `sqrt_z` | 13,179 | 1,585 | 2,976 | 18,966 | 2,602 | 6,843 | 751 | 618 |
| drop `x_d^2` | 17,186 | 7,928 | 2,833 | 14,748 | 3,603 | 4,385 | 569 | 618 |
| drop `sqrt_z + x_d^2` | 10,932 | 15,333 | 2,847 | 16,905 | 2,847 | 3,914 | 702 | 842 |
| `z2` only | 11,631 | 5,395 | 3,045 | 17,366 | 769 | -225 | 830 | 627 |

## Interpretation

The clearest result is that `sqrt_z` is not needed at this budget. Dropping it
reduces the policy from 24 to 18 parameters while slightly improving HV delta
and preserving target-span coverage. This matches the prior surface analysis:
`sqrt_z` contributed only about 3% of the high-cost decrement in the trained
24-parameter policies.

Dropping only `x_d^2` is also safe on HV and improves budget coverage, but it
does not improve the time-save objective. The combined 15-parameter variant has
the best HV delta and best relative time-save AUC in this run, but its budget
span coverage falls to 74.6%, so it is a more aggressive tradeoff rather than a
strict replacement.

The 12-parameter `z2`-only variant is surprisingly competitive but not a clean
default candidate: HV drops by about 3.1k versus the 24-parameter reference and
coverage is narrower. It is useful evidence that most cost conditioning lives
in the high-cost curvature term, but keeping the linear `z` basis is still
prudent.

Recommended next default candidate: `drop_sqrt_z` at 18 parameters. The
15-parameter `drop_sqrt_z + x_d^2` variant is worth a second-seed confirmation
because it won primary HV here but with lower coverage.

The LSTM supplement strengthens the compression result. Every compressed
variant beats the 24-parameter reference on LSTM HV, and the biggest gains come
from removing `sqrt_z`. The 15-parameter `drop_sqrt_z + x_d^2` variant again has
the best HV and time-save AUC, but its budget coverage is also the weakest, so
it remains the aggressive option. The 18-parameter `drop_sqrt_z` variant is the
best conservative default: it nearly doubles LSTM HV delta versus the full
reference while keeping target and budget coverage close to the full model.

## GPU Monitor

| variant | stage | shared-memory spill | peak shared memory | peak `nvidia-smi` memory |
| --- | --- | --- | ---: | ---: |
| drop `sqrt_z` | train-overfit | false | 436,244,480 bytes | 12,446 MiB |
| drop `sqrt_z` | sweep | false | 241,201,152 bytes | 12,442 MiB |
| drop `x_d^2` | train-overfit | false | 317,149,184 bytes | 12,408 MiB |
| drop `x_d^2` | sweep | false | 221,814,784 bytes | 12,392 MiB |
| drop `sqrt_z + x_d^2` | train-overfit | false | 241,356,800 bytes | 12,319 MiB |
| drop `sqrt_z + x_d^2` | sweep | false | 211,435,520 bytes | 12,319 MiB |
| `z2` only | train-overfit | false | 223,498,240 bytes | 12,439 MiB |
| `z2` only | sweep | false | 224,309,248 bytes | 12,439 MiB |

No run exceeded the 1 GiB shared-memory spill threshold.

Supplemental LSTM sweep GPU monitor:

| variant | stage | shared-memory spill | peak shared memory | peak `nvidia-smi` memory |
| --- | --- | --- | ---: | ---: |
| drop `sqrt_z` | sweep | false | 255,275,008 bytes | 10,938 MiB |
| drop `x_d^2` | sweep | false | 259,006,464 bytes | 10,935 MiB |
| drop `sqrt_z + x_d^2` | sweep | false | 258,465,792 bytes | 10,754 MiB |
| `z2` only | sweep | false | 259,215,360 bytes | 10,936 MiB |
