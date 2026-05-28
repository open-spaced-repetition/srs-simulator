# FSRS6 Cost-ADR Retention-head Structure Ablation

Date: 2026-05-28

## Question

Train compressed Cost-ADR desired-retention policies on the first eight users in
the FSRS6 environment and compare whether the 24-parameter structure can be
reduced without losing Pareto quality.

All runs use users 1-8, FSRS6 training and evaluation, pop16/gen20 CMA-ES,
`sigma0 = 1.0`, no coefficient preconditioning, the first-eight
interval-implied-R mean initializer projected into each compressed structure,
and the matched 16 cost weights.

## Runs

| variant | config | params | removed structure |
| --- | --- | ---: | --- |
| full reference | previous `fsrs6_cost_adr_rethead_meaninit_nopre_ablation_users_1_8_pop16_gen20_v1_20260528T024902Z` | 24 | none |
| drop `sqrt_z` | `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1.toml` | 18 | `sqrt(z)` cost basis |
| drop `x_d^2` | `fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1.toml` | 20 | difficulty quadratic state feature |
| drop `sqrt_z + x_d^2` | `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1.toml` | 15 | both above |
| `z2` only | `fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1.toml` | 12 | `sqrt(z)` and `z` cost bases |

All four new ablation runs passed `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

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
