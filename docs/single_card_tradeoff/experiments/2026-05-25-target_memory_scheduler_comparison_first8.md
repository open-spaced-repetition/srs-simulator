# First-Eight Fixed Target-Memory Scheduler Comparison

Date: 2026-05-25

This experiment compares single-card schedulers at fixed memory targets instead
of fixed scalarization weights. For each `(user_id, M0, scheduler)` the pipeline
selects the lowest-time deterministic policy observed with achieved memory
`M >= M0`, then reports achieved memory, achieved review time, target slack,
coverage, and extra time versus the exact stationary finite oracle target
answer.

## Setup

- Users: `1,2,3,4,5,6,7,8`
- Environment: `fsrs6`
- Button usage: `../Anki-button-usage/button_usage.jsonl`
- Review Markov transition: off
- Days: `1825`
- Seed: `42`
- Target memories: `0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.96`
- Output root:
  `artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8`

The primary oracle is the certified `fsrs6_oracle_stationary_finite` artifact at
`oracle_stationary_certified_supported`. The continuous oracle is reported
separately because it uses continuous desired-retention actions and a much more
expensive stationary solve.

## Commands

Exact stationary oracle:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --family fsrs6_oracle_stationary_finite \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --theta-grid 0,16,64,256,1024,4096,16384 \
  --theta-max 16384 \
  --init-points artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_user/points.csv \
  --max-refinement-rounds 100 --candidates-per-round 128 \
  --eval-group-batch-size 32 --oracle-refinement-scope user \
  --certificate-tolerance 1e-9 --no-plot \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported
```

This final pass uses supported-hull `lambda_AB` certificates and extends the
lambda range so low-memory user 8 targets are bracketed instead of pinned to the
old `lambda=1024` boundary. The resulting discrete stationary oracle has
`56/56` feasible and `56/56` certified target answers.

Continuous stationary oracle:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --family fsrs6_oracle_continuous_stationary_finite \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --theta-grid 0,16,64,256,1024 \
  --max-refinement-rounds 6 --candidates-per-round 4 \
  --certificate-tolerance 1e-9 --progress-log-interval-seconds 30 --no-plot \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_continuous_stationary
```

Rollout and direct-search baselines:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --family fsrs6 --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --explore-particles 2048 --confirm-particles 10000 \
  --max-refinement-rounds 4 --candidates-per-bracket 3 --no-plot \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fsrs6

uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --family fixed --theta-grid 4,8,16,32,64,128,256,512,1024 \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --explore-particles 2048 --confirm-particles 10000 \
  --max-refinement-rounds 2 --candidates-per-bracket 2 --no-plot \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fixed

uv run python -m experiments.single_card_tradeoff.cli.target_constrained_direct_policy_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --population-size 32 --elite-count 8 --generations 64 \
  --train-particles 64 --eval-particles 10000 \
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/direct

uv run python -m experiments.single_card_tradeoff.cli.target_conditioned_retention_distill \
  --env fsrs6 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --teacher-policy artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/direct/policy.pt \
  --epochs 64 --steps-per-epoch 32 --samples-per-job 256 \
  --eval-particles 10000 \
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/target_conditioned_distill
```

Gap reports for the rollout baselines:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_oracle_gap_report \
  --candidate-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fsrs6 \
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fsrs6

uv run python -m experiments.single_card_tradeoff.cli.target_oracle_gap_report \
  --candidate-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fixed \
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fixed
```

The comparison aggregator converted ADR and distill rows from:

```text
artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv
```

and wrote converted target-answer files under:

```text
artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/converted
```

The combined comparison was written under:

```text
artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/comparison
```

Aggregator command:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_memory_scheduler_compare \
  --oracle oracle_stationary=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary_certified_supported \
  --target-answer oracle_continuous_stationary=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_continuous_stationary \
  --target-answer fsrs6=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fsrs6 \
  --target-answer fixed=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/fixed \
  --target-answer direct=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/direct \
  --target-answer target_conditioned_distill=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/target_conditioned_distill \
  --tradeoff-result adr=artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv \
  --tradeoff-scheduler adr=fsrs6_adr \
  --tradeoff-result stationary_distill=artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv \
  --tradeoff-scheduler stationary_distill=fsrs6_oracle_stationary_finite_distill \
  --tradeoff-result continuous_stationary_distill=artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv \
  --tradeoff-scheduler continuous_stationary_distill=fsrs6_oracle_continuous_stationary_finite_distill \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --require-complete-target-grid \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/comparison
```

## Results

Primary scheduler summary, from `comparison/scheduler_summary.csv`:

| scheduler | coverage | feasible | mean T | mean extra T vs oracle | median extra T | p90 extra T | mean slack | negative gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_stationary | 1.000 | 56/56 | 0.003096 | +0.000000 | +0.000000 | +0.000000 | 0.003952 | 0 |
| oracle_continuous_stationary | 1.000 | 56/56 | 0.003698 | +0.000602 | +0.000079 | +0.001294 | 0.020558 | 6 |
| fsrs6 | 1.000 | 56/56 | 0.003691 | +0.000595 | +0.000236 | +0.001525 | 0.014814 | 0 |
| fixed | 0.982 | 55/56 | 0.005784 | +0.003132 | +0.001677 | +0.008158 | 0.012606 | 0 |
| adr | 1.000 | 56/56 | 0.003893 | +0.000797 | +0.000188 | +0.001407 | 0.035029 | 0 |
| stationary_distill | 1.000 | 56/56 | 0.003860 | +0.000764 | +0.000106 | +0.001381 | 0.021143 | 1 |
| continuous_stationary_distill | 1.000 | 56/56 | 0.003744 | +0.000648 | +0.000162 | +0.001329 | 0.031150 | 3 |
| direct | 0.643 | 36/56 | 0.002510 | +0.000243 | +0.000118 | +0.000685 | 0.017314 | 0 |
| target_conditioned_distill | 0.643 | 36/56 | 0.002152 | +0.000298 | +0.000226 | +0.000498 | 0.025867 | 0 |

Coverage by target:

| scheduler | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.93 | 0.96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_stationary | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| oracle_continuous_stationary | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| fsrs6 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| fixed | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 7/8 |
| adr | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| stationary_distill | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| continuous_stationary_distill | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| direct | 6/8 | 6/8 | 4/8 | 4/8 | 5/8 | 6/8 | 5/8 |
| target_conditioned_distill | 4/8 | 5/8 | 5/8 | 6/8 | 7/8 | 6/8 | 3/8 |

## Interpretation

The certified discrete stationary oracle is now the baseline: all `56/56`
target answers are feasible and certified. Extending the lambda range fixed the
old user 8 low-memory boundary case, so the oracle mean time dropped from the
earlier locally refined value.

Among full-coverage non-oracle schedulers, the plain FSRS6 desired-retention
baseline is the closest to the certified discrete oracle on mean extra time
(`+0.000595`). The continuous stationary finite distill is next (`+0.000648`),
followed by the discrete stationary distill (`+0.000764`) and ADR
(`+0.000797`).

The continuous stationary oracle and continuous stationary distill are not
strictly comparable to the discrete stationary oracle because they use a
continuous desired-retention action surface. Their few negative deterministic
gaps are therefore interpreted as action-space differences, not as failures of
the certified discrete oracle.

The fixed-interval baseline is much worse on time, especially at high-memory
targets, and misses one `M0=0.96` target. Direct target search and the
target-conditioned distill have lower mean time among their feasible rows, but
only cover `36/56` confirmed targets. Their train-time feasible-first signal was
optimistic under `64` train particles: after the `10000`-particle confirmation,
both had `20` infeasible rows. The largest memory constraint violation was
`0.02617` for direct search and `0.03816` for target-conditioned distill.

## Negative Gaps

There are still a few negative deterministic gaps, but they are no longer caused
by an uncertified discrete oracle. The largest are:

| gap | scheduler | user | target | candidate M/T | oracle M/T | oracle certified |
| ---: | --- | ---: | ---: | --- | --- | --- |
| -0.000045 | oracle_continuous_stationary | 1 | 0.96 | 0.960562 / 0.009900 | 0.960040 / 0.009946 | True |
| -0.000042 | oracle_continuous_stationary | 2 | 0.75 | 0.753125 / 0.000697 | 0.756297 / 0.000739 | True |
| -0.000039 | continuous_stationary_distill | 2 | 0.75 | 0.753882 / 0.000700 | 0.756297 / 0.000739 | True |
| -0.000016 | oracle_continuous_stationary | 8 | 0.90 | 0.900412 / 0.000558 | 0.914134 / 0.000574 | True |
| -0.000008 | continuous_stationary_distill | 3 | 0.70 | 0.708569 / 0.000613 | 0.715368 / 0.000621 | True |

The remaining negative rows are small. Most come from continuous-action
schedulers compared against a discrete-action oracle; the one discrete distill
negative row is below `1e-6` day-minutes/card and is consistent with rollout
conversion/evaluation noise. The mixed oracle target columns remain present in
the CSVs for diagnostics, while this report still ranks deterministic selected
policies.

## Performance

All GPU-monitored stages stayed below the shared-memory spill threshold.

| stage | runtime | peak VRAM MiB | shared spill | cache hits/misses/writes |
| --- | ---: | ---: | --- | ---: |
| oracle_stationary | 128.6s | 8438 | False | 200/32/32 |
| oracle_stationary_certified_user | 6434.1s | 2970 | False | 8/275/275 |
| oracle_stationary_certified_supported | 2953.4s | 3617 | False | 5/109/109 |
| oracle_continuous_stationary | 6826.0s | 17995 | False | 40/384/384 |
| fsrs6 | 264.2s | 19968 | False | 0/0/0 |
| fixed | 11.8s | 4926 | False | 0/0/0 |
| direct | 315.4s | 2930 | False | 0/0/0 |
| target_conditioned_distill | 29.8s | 2932 | False | 0/0/0 |

The certified discrete oracle required two warm-started passes after the
original local run. The first user-scoped pass avoided solving every new
`lambda_AB` for all users; the second supported-hull pass completed
certification and added the extended high-lambda user 8 bracket. Both stayed
well below the shared-memory spill threshold.

## Artifacts

- `oracle_stationary_certified_supported/target_answers.csv`
- `oracle_stationary_certified_supported/segments.csv`
- `oracle_stationary_certified_supported/performance_summary.json`
- `converted/adr_target_answers.csv`
- `converted/adr_gap/target_oracle_gaps.csv`
- `converted/stationary_distill_target_answers.csv`
- `converted/stationary_distill_gap/target_oracle_gaps.csv`
- `converted/continuous_stationary_distill_target_answers.csv`
- `converted/continuous_stationary_distill_gap/target_oracle_gaps.csv`
- `comparison/combined_target_answers.csv`
- `comparison/scheduler_target_matrix.csv`
- `comparison/scheduler_oracle_gaps.csv`
- `comparison/scheduler_summary.csv`
- `comparison/user_summary.csv`
- `comparison/target_summary.csv`
- `comparison/plots/mean_T_vs_target.png`
- `comparison/plots/mean_extra_T_vs_oracle.png`
- `comparison/plots/T_vs_target_by_user.png`
- `comparison/plots/extra_T_vs_oracle_by_user.png`
- `comparison/plots/mean_extra_T_by_scheduler.png`
- `comparison/plots/feasibility_heatmap.png`
- `comparison/plots/memory_slack_distribution.png`

The direct and target-conditioned distill output directories also contain
`policy.pt`, `target_answers.csv`, `target_oracle_gaps.csv`, and
`performance_summary.json`.
