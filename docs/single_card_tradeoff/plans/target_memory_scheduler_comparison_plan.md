# Fixed Target-Memory Scheduler Comparison Plan

## 1. Goal

Compare different single-card schedulers at the same requested memory targets:

```text
given M0: select the lowest-time deterministic policy with M(policy) >= M0
```

The output should answer, for every `(user_id, M0, scheduler)`:

- whether the scheduler can reach `M0`;
- which scheduler parameter or policy was selected;
- achieved memory `M`;
- achieved review time `T`;
- target slack `M - M0`;
- extra time versus the oracle target frontier when an oracle answer is
  available.

This experiment is different from scalarized `lambda` frontier plots. It fixes
the same memory targets for every scheduler first, then compares required time
horizontally.

## 2. Primary Question

For a fixed target-memory grid, which scheduler reaches the target with the
lowest single-card review time?

Secondary questions:

- How much time does each non-oracle scheduler lose versus the exact stationary
  oracle deterministic target answer?
- Does continuous stationary finite improve the target answer over discrete
  stationary finite?
- Do distilled or direct-search policies preserve target feasibility after
  high-particle confirmation?
- Which memory targets are outside a scheduler's achievable span?
- How much deterministic overshoot is required at each target?

## 3. Scope

### Users

Primary run:

```text
user_ids = 1,2,3,4,5,6,7,8
```

Use the same FSRS-6 user weights and button-usage costs as the existing
first-eight single-card reports:

```text
env = fsrs6
button_usage = ../Anki-button-usage/button_usage.jsonl
review_markov_transition = off
days = 1825
seed = 42
```

### Target Grid

Primary target-memory grid:

```text
M0 = 0.70,0.75,0.80,0.85,0.90,0.93,0.96
```

Optional dense target-memory grid for final plots:

```text
M0 = 0.70,0.725,0.75,0.775,0.80,0.825,0.85,0.875,0.90,0.915,0.93,0.945,0.96
```

The primary grid is the acceptance set. The dense grid is for presentation only
after the pipeline is verified.

### Scheduler Families

Directly supported by `target_search`:

| scheduler label | runner family | theta |
| --- | --- | --- |
| `fsrs6` | `fsrs6` | desired retention |
| `fixed` | `fixed` | fixed interval days |
| `oracle_stationary_finite` | `fsrs6_oracle_stationary_finite` | cost weight `lambda` |
| `oracle_continuous_stationary_finite` | `fsrs6_oracle_continuous_stationary_finite` | cost weight `lambda` |

Target-answer-compatible trained policies:

| scheduler label | source |
| --- | --- |
| `target_constrained_direct` | `target_constrained_direct_policy_search.py` |
| `target_conditioned_retention_distill` | `target_conditioned_retention_distill.py` |

Schedulers requiring a conversion step from existing tradeoff frontier rows:

| scheduler label | source |
| --- | --- |
| `fsrs6_oracle_stationary_finite_distill` | `tradeoff.py` / existing result CSV |
| `fsrs6_oracle_continuous_stationary_finite_distill` | `tradeoff.py` / existing result CSV |
| `fsrs6_adr` | `tradeoff.py` / existing result CSV |

For converted schedulers, the deterministic target answer is:

```text
for each user_id, scheduler, M0:
  choose row with achieved M >= M0 and minimal achieved T
```

If no row reaches `M0`, mark the target infeasible.

## 4. Artifacts

Use one experiment root:

```text
artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8
```

Recommended layout:

```text
target_memory_scheduler_comparison_first8/
  oracle_stationary/
    points.csv
    frontier.csv
    target_answers.csv
    segments.csv
    metadata.json
  oracle_continuous_stationary/
    ...
  fsrs6/
    ...
  fixed/
    ...
  direct/
    policy.pt
    target_answers.csv
    target_oracle_gaps.csv
  target_conditioned_distill/
    policy.pt
    target_answers.csv
    target_oracle_gaps.csv
  converted/
    stationary_distill_target_answers.csv
    continuous_stationary_distill_target_answers.csv
    adr_target_answers.csv
  comparison/
    combined_target_answers.csv
    scheduler_target_matrix.csv
    scheduler_oracle_gaps.csv
    user_summary.csv
    scheduler_summary.csv
    plots/
```

## 5. Commands

Set shared shell variables:

```bash
ROOT=artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8
TARGETS=0.70,0.75,0.80,0.85,0.90,0.93,0.96
USERS=1,2,3,4,5,6,7,8
BUTTON_USAGE=../Anki-button-usage/button_usage.jsonl
```

### 5.1 Exact Stationary Oracle

This is the primary deterministic oracle baseline.

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 \
  --user-ids "$USERS" \
  --button-usage "$BUTTON_USAGE" \
  --family fsrs6_oracle_stationary_finite \
  --target-memories "$TARGETS" \
  --theta-grid 0,16,64,256,1024 \
  --max-refinement-rounds 6 \
  --candidates-per-round 4 \
  --certificate-tolerance 1e-9 \
  --out-dir "$ROOT/oracle_stationary"
```

Acceptance checks:

- `target_answers.csv` has `8 * 7 = 56` rows.
- `metadata.json` has `certification_scope = oracle_target_local`.
- Most target-local `certified` rows should be `True`. Any uncertified rows
  must be reported in the final analysis.

### 5.2 Continuous Stationary Oracle

This tests whether continuous desired-retention actions improve target-time
answers over the discrete stationary oracle.

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 \
  --user-ids "$USERS" \
  --button-usage "$BUTTON_USAGE" \
  --family fsrs6_oracle_continuous_stationary_finite \
  --target-memories "$TARGETS" \
  --theta-grid 0,16,64,256,1024 \
  --max-refinement-rounds 6 \
  --candidates-per-round 4 \
  --certificate-tolerance 1e-9 \
  --progress-log-interval-seconds 30 \
  --out-dir "$ROOT/oracle_continuous_stationary"
```

Acceptance checks:

- Same row count as the discrete oracle.
- Compare `achieved_T` against `oracle_stationary/target_answers.csv`.
- Continuous should not be interpreted as globally better unless target-local
  certificates are present.

### 5.3 FSRS6 Desired-Retention Baseline

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 \
  --user-ids "$USERS" \
  --button-usage "$BUTTON_USAGE" \
  --family fsrs6 \
  --target-memories "$TARGETS" \
  --explore-particles 2048 \
  --confirm-particles 10000 \
  --max-refinement-rounds 4 \
  --candidates-per-bracket 3 \
  --out-dir "$ROOT/fsrs6"
```

Then run the gap report:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_oracle_gap_report \
  --candidate-target-answers "$ROOT/fsrs6" \
  --oracle-target-answers "$ROOT/oracle_stationary" \
  --out-dir "$ROOT/fsrs6"
```

### 5.4 Fixed-Interval Baseline

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_search \
  --env fsrs6 \
  --user-ids "$USERS" \
  --button-usage "$BUTTON_USAGE" \
  --family fixed \
  --theta-grid 4,8,16,32,64,128,256,512,1024 \
  --target-memories "$TARGETS" \
  --explore-particles 2048 \
  --confirm-particles 10000 \
  --max-refinement-rounds 2 \
  --candidates-per-bracket 2 \
  --out-dir "$ROOT/fixed"
```

Then:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_oracle_gap_report \
  --candidate-target-answers "$ROOT/fixed" \
  --oracle-target-answers "$ROOT/oracle_stationary" \
  --out-dir "$ROOT/fixed"
```

### 5.5 Target-Constrained Direct Search

Run this after the oracle, using the oracle answers as the comparison target.

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_constrained_direct_policy_search \
  --env fsrs6 \
  --user-ids "$USERS" \
  --button-usage "$BUTTON_USAGE" \
  --target-memories "$TARGETS" \
  --population-size 32 \
  --elite-count 8 \
  --generations 64 \
  --train-particles 64 \
  --eval-particles 10000 \
  --oracle-target-answers "$ROOT/oracle_stationary" \
  --out-dir "$ROOT/direct"
```

Acceptance checks:

- `policy.pt` exists.
- `target_answers.csv` has 56 rows.
- `target_oracle_gaps.csv` exists.
- Any infeasible target is listed in the final report, not filtered out.

### 5.6 Target-Conditioned Distill

Distill the per-target direct policies into one target-conditioned retention
network.

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_conditioned_retention_distill \
  --env fsrs6 \
  --button-usage "$BUTTON_USAGE" \
  --teacher-policy "$ROOT/direct/policy.pt" \
  --epochs 64 \
  --steps-per-epoch 32 \
  --samples-per-job 256 \
  --eval-particles 10000 \
  --oracle-target-answers "$ROOT/oracle_stationary" \
  --out-dir "$ROOT/target_conditioned_distill"
```

Acceptance checks:

- `policy.pt` exists.
- `target_answers.csv` has 56 rows.
- `target_oracle_gaps.csv` exists.
- Compare feasibility and `achieved_T` against direct search to quantify
  distillation loss.

## 6. Scheduler Frontier Conversion

Some schedulers are evaluated through `tradeoff.py` rather than `target_search`.
For those, convert an existing result frontier into target answers.

Input columns expected from a tradeoff result CSV:

```text
user_id
scheduler
card_expected_retrievability
card_minutes_per_day
<scheduler-specific theta columns>
```

Conversion rule:

```text
for each (user_id, scheduler, M0):
  feasible_rows = rows where card_expected_retrievability >= M0
  if feasible_rows:
      selected = row with minimal card_minutes_per_day
      feasible = True
  else:
      selected = row with maximal card_expected_retrievability for diagnostics
      feasible = False
```

Output schema should match `target_answers.csv`:

```text
user_id,target_type,target_value,family,feasible,theta_name,theta_value,
achieved_M,achieved_T,memory_slack,time_slack,certified,policy_ref,cache_key,
neighbor_low_theta,neighbor_low_M,neighbor_low_T,
neighbor_high_theta,neighbor_high_M,neighbor_high_T,
mixed_available,mixed_probability_high,mixed_M,mixed_T
```

For converted rollout schedulers:

```text
certified = False
mixed_available = False
```

After conversion, run:

```bash
uv run python -m experiments.single_card_tradeoff.cli.target_oracle_gap_report \
  --candidate-target-answers "$ROOT/converted/adr_target_answers.csv" \
  --oracle-target-answers "$ROOT/oracle_stationary" \
  --out-dir "$ROOT/converted/adr_gap"
```

## 7. Aggregation

Build `comparison/combined_target_answers.csv` by concatenating:

- `oracle_stationary/target_answers.csv`
- `oracle_continuous_stationary/target_answers.csv`
- `fsrs6/target_answers.csv`
- `fixed/target_answers.csv`
- `direct/target_answers.csv`
- `target_conditioned_distill/target_answers.csv`
- converted target-answer files for ADR and ordinary distill schedulers

Build `comparison/scheduler_target_matrix.csv` with one row per
`(user_id, target_value)` and one group of columns per scheduler:

```text
<scheduler>_feasible
<scheduler>_theta
<scheduler>_M
<scheduler>_T
<scheduler>_memory_slack
<scheduler>_extra_T_vs_oracle
```

For memory targets:

```text
extra_T_vs_oracle = scheduler_achieved_T - oracle_stationary_achieved_T
```

Only compute `extra_T_vs_oracle` when both the scheduler and oracle rows are
feasible. Keep missing values empty rather than filling with zero.

## 8. Metrics

Per scheduler:

- `target_count`
- `feasible_count`
- `coverage = feasible_count / target_count`
- mean `achieved_T` on feasible rows
- mean `memory_slack` on feasible rows
- mean `extra_T_vs_oracle` on matched feasible rows
- median `extra_T_vs_oracle`
- p90 `extra_T_vs_oracle`
- worst positive `extra_T_vs_oracle`
- count of negative `extra_T_vs_oracle`

Negative `extra_T_vs_oracle` should be investigated. It can indicate Monte Carlo
noise, an uncertified oracle target segment, a mismatch in environment settings,
or a converted scheduler row evaluated under different settings.

Per target:

- number of feasible schedulers;
- best non-oracle scheduler by `achieved_T`;
- gap between best non-oracle and oracle;
- average deterministic overshoot.

Per user:

- scheduler ranking by mean `extra_T_vs_oracle`;
- infeasible targets by scheduler;
- target where each scheduler has the largest gap.

## 9. Plots

Generate these plots after aggregation:

1. `T` versus `M0`, one line per scheduler, faceted by user.
2. Extra time versus oracle by `M0`, one line per scheduler, faceted by user.
3. Mean extra time versus oracle by scheduler, aggregated over users.
4. Feasibility heatmap: scheduler by `M0`, cell value = feasible user count.
5. Memory slack distribution by scheduler.

All plots should distinguish:

- exact oracle;
- rollout-only family-constrained baselines;
- learned/distilled policies.

## 10. Validation

Before interpreting results:

- Confirm all runs use the same `env`, `button_usage`, `days`, `seed`, and
  target grid.
- Confirm every direct `target_search` output has the expected row count.
- Confirm oracle metadata has `certification_scope = oracle_target_local`.
- Check oracle `certified` counts. If a target is uncertified, label the gap as
  empirical rather than certified.
- Check GPU monitor summaries for CUDA runs. Shared-memory spill invalidates
  timing and may also make the run impractically slow.
- Re-run any surprising negative oracle gaps with a larger
  `--confirm-particles` value or exact oracle refinement.

## 11. Acceptance Criteria

The experiment is complete when:

- every scheduler has a `target_answers.csv`-compatible file for the fixed
  target grid;
- every non-oracle scheduler has a `target_oracle_gaps.csv` comparison against
  `oracle_stationary`;
- `combined_target_answers.csv` and `scheduler_target_matrix.csv` are written;
- the final report includes coverage, mean extra time versus oracle, and
  infeasible targets;
- any negative oracle gap or uncertified oracle segment is explicitly explained;
- all commands and artifact paths are recorded in the report metadata.

## 12. Recommended Report Tables

### Scheduler Summary

```text
scheduler
target_count
feasible_count
coverage
mean_extra_T_vs_oracle
median_extra_T_vs_oracle
p90_extra_T_vs_oracle
mean_memory_slack
negative_gap_count
```

### Target Summary

```text
target_value
best_non_oracle_scheduler
best_non_oracle_T
oracle_T
best_non_oracle_extra_T
feasible_scheduler_count
mean_memory_slack
```

### User Summary

```text
user_id
best_non_oracle_scheduler
mean_extra_T_vs_oracle
hardest_target
largest_extra_T_vs_oracle
infeasible_count
```

## 13. Follow-Up Implementation

The current runners already produce the target-answer files for `fsrs6`,
`fixed`, exact oracle families, direct constrained policies, and
target-conditioned distill. To make the experiment fully one-command,
add a small aggregation CLI:

```text
experiments.single_card_tradeoff.cli.target_memory_scheduler_compare
```

Suggested arguments:

```text
--target-answer NAME=PATH
--oracle NAME=PATH
--out-dir
--target-tolerance
--require-complete-target-grid
```

Suggested outputs:

```text
combined_target_answers.csv
scheduler_target_matrix.csv
scheduler_summary.csv
user_summary.csv
target_summary.csv
plots/*.png
metadata.json
```
