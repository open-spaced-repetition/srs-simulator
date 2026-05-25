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
- Days: `1825`
- Seed: `42`
- Target memories: `0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.96`
- Output root:
  `artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8`

The primary oracle is `fsrs6_oracle_stationary_finite`. The continuous oracle is
reported separately because it uses continuous desired-retention actions and a
much more expensive stationary solve.

## Commands

Exact stationary oracle:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --family fsrs6_oracle_stationary_finite \
  --target-memories 0.70,0.75,0.80,0.85,0.90,0.93,0.96 \
  --theta-grid 0,16,64,256,1024 \
  --max-refinement-rounds 6 --candidates-per-round 4 \
  --certificate-tolerance 1e-9 --no-plot \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary
```

Continuous stationary oracle used the same target grid and refinement settings
with `--family fsrs6_oracle_continuous_stationary_finite` and
`--progress-log-interval-seconds 30`.

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
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/direct

uv run python -m experiments.single_card_tradeoff.cli.target_conditioned_retention_distill \
  --env fsrs6 \
  --button-usage ../Anki-button-usage/button_usage.jsonl \
  --teacher-policy artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/direct/policy.pt \
  --epochs 64 --steps-per-epoch 32 --samples-per-job 256 \
  --eval-particles 10000 \
  --oracle-target-answers artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/oracle_stationary \
  --out-dir artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/target_conditioned_distill
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

## Results

Primary scheduler summary, from `comparison/scheduler_summary.csv`:

| scheduler | coverage | feasible | mean T | mean extra T vs oracle | median extra T | p90 extra T | mean slack | negative gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fsrs6 | 1.000 | 56/56 | 0.003691 | -0.000020 | +0.000118 | +0.000600 | 0.014814 | 12 |
| oracle_continuous_stationary | 1.000 | 56/56 | 0.003698 | -0.000012 | -0.000001 | +0.000060 | 0.020558 | 31 |
| oracle_stationary | 1.000 | 56/56 | 0.003711 | +0.000000 | +0.000000 | +0.000000 | 0.019479 | 0 |
| continuous_stationary_distill | 1.000 | 56/56 | 0.003744 | +0.000033 | +0.000044 | +0.000303 | 0.031150 | 20 |
| stationary_distill | 1.000 | 56/56 | 0.003860 | +0.000149 | +0.000047 | +0.000516 | 0.021143 | 14 |
| adr | 1.000 | 56/56 | 0.003893 | +0.000182 | +0.000076 | +0.000335 | 0.035029 | 12 |
| fixed | 0.982 | 55/56 | 0.005784 | +0.002700 | +0.001410 | +0.007041 | 0.012606 | 0 |
| direct | 0.643 | 36/56 | 0.002510 | +0.000080 | +0.000042 | +0.000314 | 0.017314 | 8 |
| target_conditioned_distill | 0.643 | 36/56 | 0.002152 | +0.000169 | +0.000176 | +0.000422 | 0.025867 | 4 |

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

The safest full-coverage non-oracle answer in this run is the continuous
stationary finite distill: it covers all targets and has the smallest positive
mean extra time versus the discrete stationary oracle (`+0.000033`). The plain
FSRS6 desired-retention baseline has a slightly negative mean deterministic gap
(`-0.000020`), but this should not be read as a certified oracle win because the
oracle target answer is only locally refined and mostly uncertified.

The continuous stationary oracle is close to the discrete stationary oracle on
this target grid. It has full coverage and a small negative mean deterministic
gap (`-0.000012`), but again this is not a global certificate. It is useful as a
diagnostic for the value of continuous desired-retention actions, not as a
replacement for a certified constrained oracle.

The fixed-interval baseline is much worse on time, especially at high-memory
targets, and misses one `M0=0.96` target. Direct target search and the
target-conditioned distill have lower mean time among their feasible rows, but
only cover `36/56` confirmed targets. Their train-time feasible-first signal was
optimistic under `64` train particles: after the `10000`-particle confirmation,
both had `20` infeasible rows. The largest memory constraint violation was
`0.02617` for direct search and `0.03816` for target-conditioned distill.

## Negative Gaps

There are negative deterministic gaps in several schedulers. The largest are:

| gap | scheduler | user | target | candidate M/T | oracle M/T | oracle certified |
| ---: | --- | ---: | ---: | --- | --- | --- |
| -0.005459 | fsrs6 | 2 | 0.93 | 0.930002 / 0.014680 | 0.950079 / 0.020139 | False |
| -0.005131 | fsrs6 | 2 | 0.96 | 0.960182 / 0.033044 | 0.968334 / 0.038175 | False |
| -0.003830 | continuous_stationary_distill | 2 | 0.96 | 0.963885 / 0.034345 | 0.968334 / 0.038175 | False |
| -0.002072 | stationary_distill | 2 | 0.93 | 0.943877 / 0.018067 | 0.950079 / 0.020139 | False |
| -0.001776 | adr | 2 | 0.93 | 0.944792 / 0.018363 | 0.950079 / 0.020139 | False |

These rows compare deterministic selected points on a finite target-search grid.
They do not invalidate the oracle implementation by themselves. The discrete
oracle target answers were certified for only `2/56` targets, so many gaps are
best interpreted as target-grid, Monte Carlo, and local-refinement artifacts.
The mixed oracle target columns are present in the CSVs for additional
diagnostics, but this report ranks deterministic policies because the experiment
question was deterministic scheduler selection.

## Performance

All GPU-monitored stages stayed below the shared-memory spill threshold.

| stage | runtime | peak VRAM MiB | shared spill | cache hits/misses/writes |
| --- | ---: | ---: | --- | ---: |
| oracle_stationary | 128.6s | 8438 | False | 200/32/32 |
| oracle_continuous_stationary | 6826.0s | 17995 | False | 40/384/384 |
| fsrs6 | 264.2s | 19968 | False | 0/0/0 |
| fixed | 11.8s | 4926 | False | 0/0/0 |
| direct | 315.4s | 2930 | False | 0/0/0 |
| target_conditioned_distill | 29.8s | 2932 | False | 0/0/0 |

The continuous stationary target-search run was the dominant cost:
`6826.04s` with peak `17995 MiB` FB memory and no shared-memory spill. It wrote
`384` cache entries across finite and stationary solves; reruns of the same
lambda points should reuse those caches.

## Artifacts

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
