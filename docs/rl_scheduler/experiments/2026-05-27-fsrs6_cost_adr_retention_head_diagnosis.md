# FSRS6 Cost-ADR Retention-Head Diagnosis

Date: 2026-05-27

This note diagnoses why the Cost-ADR `desired_retention` action head underperforms the direct `interval` action head.

## Runs Compared

Candidate: Cost ADR schedHV stdpre

- Run root: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off`
- Config: `experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml`
- Report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.md`

Comparison: Cost ADR interval-head schedHV stdpre

- Run root: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off`
- Report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.md`

Both runs use users 1..8, 16 matched cost weights, pop16/gen20 CMA-ES, bounds `[-64, 64]`, scheduler-HV objective, diagonal preconditioning, Markov off, and the same FSRS6 baseline DR manifest.

## Artifact Validation

The retention-head run is internally consistent:

- `all/all_summary.json`: `passed = true`
- Train policies: 8/8 policies have `action_head = "desired_retention"`
- Feature version: 8/8 policies have `fsrs6_cost_adr_retention_mono_v1`
- Metadata action space: 8/8 policies have `sd_cost_retention_function`
- GPU monitor: train-overfit and sweep both report `shared_memory_spill_detected = false`

The interval-head comparison is also internally consistent:

- `all/all_summary.json`: `passed = true`
- Train policies: 8/8 policies have `action_head = "interval"`
- Feature version: 8/8 policies have `fsrs6_cost_adr_interval_mono_v1`
- Metadata action space: 8/8 policies have `sd_cost_interval_function`
- GPU monitor: train-overfit and sweep both report `shared_memory_spill_detected = false`

Conclusion: there is no evidence of a scheduler registration, artifact resolution, action-space metadata, stage execution, or GPU spill bug.

## Result Delta

The candidate below is retention-head; the comparison is interval-head.

| env | metric | retention head | interval head | delta |
|---|---:|---:|---:|---:|
| fsrs6 | HV delta sum | 72,550.561 | 97,711.064 | -25,160.503 |
| fsrs6 | relative same-budget memory lift AUC | 1.376% | 1.394% | -0.018 pp |
| fsrs6 | relative same-target time saved AUC | 10.741% | 12.793% | -2.051 pp |
| fsrs6 | budget span coverage | 79.410% | 87.137% | -7.727 pp |
| fsrs6 | target span coverage | 79.761% | 82.331% | -2.570 pp |
| lstm | HV delta sum | 8,906.798 | 40,051.846 | -31,145.048 |
| lstm | relative same-budget memory lift AUC | 0.730% | 0.814% | -0.084 pp |
| lstm | relative same-target time saved AUC | 1.918% | 4.590% | -2.672 pp |
| lstm | budget span coverage | 81.902% | 86.716% | -4.814 pp |
| lstm | target span coverage | 89.814% | 91.662% | -1.848 pp |

The retention head loses in the external report on both the training environment (`fsrs6`) and transfer environment (`lstm`).

## Training HV

The same gap appears during optimization, before external sweep/report effects:

| generation | retention best HV sum | interval best HV sum |
|---:|---:|---:|
| 0 | -57,503.320 | 16,912.685 |
| 1 | -2,927.810 | 35,913.622 |
| 2 | 38,005.500 | 43,886.163 |
| 5 | 55,737.786 | 69,660.163 |
| 10 | 62,266.456 | 79,170.169 |
| 15 | 65,362.090 | 85,928.843 |
| 19 | 72,673.199 | 97,905.235 |

Final training HV gain by run:

- Retention head: 72,673.199
- Interval head: 97,905.235
- Delta: -25,232.037

Conclusion: this is not primarily an LSTM transfer/reporting artifact. The retention-head objective is already harder or less expressive during FSRS6 training.

## Action-Space Diagnostic

Policy behavior was sampled over users 1..8, 32 log-spaced stability values, 32 difficulty values, and the same 16 cost weights, for 131,072 grid points per head.

For the interval head, implied retention was computed by passing the direct interval through the FSRS6 forgetting curve. For the retention head, output retention was converted to interval through `fsrs6_next_interval`.

Overall grid summary:

| quantity | q01 | q05 | q50 | q95 | q99 |
|---|---:|---:|---:|---:|---:|
| interval-head implied retention | 0.406 | 0.524 | 0.840 | 0.983 | 0.991 |
| interval-head interval days | 1.000 | 1.000 | 41.740 | 36,500.000 | 36,500.000 |
| retention-head output retention | 0.500 | 0.501 | 0.888 | 0.977 | 0.979 |
| retention-head converted interval days | 1.000 | 1.000 | 45.860 | 183,778.794 | 2,328,985.472 |

Important fractions:

- Interval-head implied retention below 0.50: 3.970%
- Interval-head implied retention above 0.98: 6.096%
- Interval-head implied retention inside `[0.50, 0.98]`: 89.934%
- Retention-head output near lower bound (`<= 0.505`): 6.438%
- Retention-head output near upper bound (`>= 0.975`): 6.411%

The retention head is not simply unable to create long intervals. It can create intervals much longer than the direct interval head because the FSRS conversion amplifies low desired-retention values at large stability.

## Cost-Weight Shape

Selected weight slices:

| cost weight | head | retention q05/q50/q95 | interval-days q05/q50/q95 | log(interval) vs log(S) slope | interval/S q05/q50/q95 |
|---:|---|---:|---:|---:|---:|
| 0 | interval implied | 0.700 / 0.954 / 0.991 | 1.000 / 8.478 / 1,140.114 | 0.636 | 0.075 / 0.366 / 9.075 |
| 0 | retention output | 0.898 / 0.964 / 0.979 | 1.000 / 8.540 / 1,057.041 | 0.699 | 0.147 / 0.273 / 6.918 |
| 128 | interval implied | 0.527 / 0.786 / 0.952 | 1.000 / 90.322 / 36,500.000 | 0.915 | 0.405 / 3.477 / 110.186 |
| 128 | retention output | 0.500 / 0.784 / 0.967 | 1.000 / 91.661 / 358,614.615 | 1.011 | 0.258 / 3.904 / 502.636 |
| 1024 | interval implied | 0.410 / 0.684 / 0.930 | 1.000 / 439.894 / 36,500.000 | 0.945 | 0.621 / 8.827 / 695.289 |
| 1024 | retention output | 0.500 / 0.588 / 0.945 | 1.000 / 457.883 / 820,156.468 | 1.065 | 0.512 / 11.288 / 547.400 |

The retention-head interval mapping is more tightly coupled to stability: its log-interval/log-stability slope is near or above 1 for medium/high cost weights. The interval head can learn a less FSRS-shaped schedule because it directly emits `log(interval)`.

## Mechanism

The source behavior is direct:

- `FSRS6CostConditionedADRPolicy.evaluate_action` returns `exp(value)` for the interval head, optionally clamped by `max_interval_days`.
- The retention head returns `retention_min + (retention_max - retention_min) * sigmoid(value)`.
- `FSRS6CostConditionedADRScheduler._interval_for_state` converts the retention head through `fsrs6_next_interval(params, stability, retention)`.

That means the retention head imposes an extra formula:

```text
interval = fsrs6_next_interval(fsrs6_params, stability, desired_retention)
```

The direct interval head optimizes the simulator action itself:

```text
interval = exp(policy_value)
```

This difference matters because Cost-ADR is optimizing an external Pareto objective over memorized cards and review time. The best action does not need to be representable as a clean desired-retention schedule under the FSRS6 curve.

## Diagnosis

Primary cause: action parameterization and policy-space geometry.

The retention head is a narrower and more distorted action space. It forces every action through the FSRS6 interval formula, coupling the learned schedule to stability and FSRS decay. The direct interval head can express schedules that only have an implied-retention interpretation after the fact, including about 10% of sampled grid points outside the retention head's `[0.50, 0.98]` bounds.

Secondary cause: optimization setup.

The retention-head initializer is weaker. Generation 0 starts at -57,503 HV, while the interval-head run starts at +16,913 HV. The retention run also uses the same first-8 distilled 24p standard-deviation preconditioner family, which was derived from interval-head policy coefficients, so the diagonal scale is likely mismatched for retention-head coefficients. This explains part of the early gap, but not all of it: after 20 generations, retention remains 25k HV behind.

Not supported as the main explanation:

- Artifact/plumbing bug: contradicted by policy metadata, stage summaries, and scheduler action-space validation.
- GPU spill or simulator slowdown: contradicted by GPU monitor summaries.
- LSTM-only generalization issue: contradicted by FSRS6 training HV and FSRS6 external Pareto loss.
- Simple inability to produce long intervals: contradicted by retention-head converted interval q95/q99 values of 183,779 and 2,328,985 days.

## Recommendation

Keep the formal Cost-ADR scheduler on the direct interval action head. Use implied retention as a diagnostic and visualization quantity, not as the primary action.

If the retention head is studied further, use it as a separate ablation with a retention-specific initialization and preconditioner. Even then, it should not replace the interval head unless it closes the training-HV gap under the same 16 cost weights and pop16/gen20 budget.
