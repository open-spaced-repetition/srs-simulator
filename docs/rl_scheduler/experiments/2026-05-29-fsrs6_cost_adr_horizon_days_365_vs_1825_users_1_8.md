# FSRS6 Cost-ADR Horizon Days: 365 vs 1825

Date: 2026-05-29

## Question

Compare Cost-ADR policies trained for a 365-day simulation horizon against
policies trained for a 1825-day horizon, then cross-evaluate both policy sets
on both horizons. The goal is to separate training-horizon fit from evaluation
horizon transfer and inspect how the learned policy distributions differ.

## Setup

Both policy sets use users 1-8, FSRS6 environment, Markov off, batched engine,
`new-first` behavior priority, `low_retrievability` scheduler priority,
deck size 10000, new limit 10, review limit 9999, daily cost limit 720 minutes,
seed 42, the same 16 FSRS6 baseline desired-retention points, and the same 16
Cost-ADR weights:

`0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024`.

The Cost-ADR formula is the current compressed default:
`fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1`, desired-retention head,
15 parameters, coefficient bounds `[-64, 64]`, `sigma0 = 1.0`, pop16/gen20,
no coefficient preconditioning, initialized from
`first8_interval_implied_r_mean_v1`.

Training artifacts:

| train horizon | run root |
| ---: | --- |
| 365 | `artifacts/rl_scheduler/fsrs6_cost_adr_horizon_days_users_1_8/fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1` |
| 1825 | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1` |

Cross-evaluation logs were written under
`logs/retention_sweep_cost_adr_horizon_days/<cell>`. Each cell has 256 JSONL
records: 8 users x 16 FSRS6 baseline DRs plus 8 users x 16 Cost-ADR weights.
Pareto and comparison outputs are under
`artifacts/rl_scheduler/cost_adr_horizon_days_users_1_8/<cell>`.

The reproducible analysis script is:

```bash
uv run python experiments/rl_scheduler/analyze_fsrs6_cost_adr_horizon_days.py
```

It reads the four `analysis_summary.json` files and both policy roots, then
writes:

- `artifacts/rl_scheduler/cost_adr_horizon_days_users_1_8/horizon_analysis/horizon_analysis_summary.json`
- `artifacts/rl_scheduler/cost_adr_horizon_days_users_1_8/horizon_analysis/horizon_analysis_summary.md`

## Performance Matrix

Primary metric is scheduler-only HV delta against the FSRS6 baseline frontier.
Time-save and memory-lift AUCs are interpolation diagnostics over common
covered spans.

| train days | eval days | HV delta | HV / baseline | relative time-save AUC | target coverage | relative memory-lift AUC | budget coverage | frontier points |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365 | 365 | 8,910.69 | 2.957% | 7.493% | 86.780% | 0.767% | 88.814% | 128 |
| 365 | 1825 | 84,943.16 | 2.750% | 11.658% | 87.945% | 1.309% | 97.737% | 125 |
| 1825 | 365 | 6,101.64 | 2.044% | 6.403% | 77.528% | 0.828% | 58.166% | 119 |
| 1825 | 1825 | 105,565.73 | 3.415% | 13.794% | 90.221% | 1.465% | 74.624% | 121 |

Per-user HV delta:

| train/eval | u1 | u2 | u3 | u4 | u5 | u6 | u7 | u8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365/365 | 1,887 | 1,834 | 518 | 2,708 | 1,013 | 496 | 244 | 212 |
| 365/1825 | 11,254 | 26,264 | 982 | 30,593 | 8,040 | 4,656 | 1,641 | 1,514 |
| 1825/365 | -9 | 1,782 | 438 | 2,209 | 1,090 | 437 | 124 | 33 |
| 1825/1825 | 14,467 | 30,640 | 3,517 | 39,009 | 9,294 | 4,748 | 1,353 | 2,537 |

## Policy Distribution

The analysis script evaluates both trained policy sets on a 41 x 41 structural
S/D grid for all 16 cost weights. The 365-day policy set is systematically
higher-retention:

| policy set | mean R | median R | q05 R | q95 R |
| --- | ---: | ---: | ---: | ---: |
| train365 | 0.8340 | 0.8960 | 0.4096 | 0.9913 |
| train1825 | 0.8081 | 0.8676 | 0.4058 | 0.9817 |

Retention delta by cost weight is `train365 - train1825`:

| cost weight | mean delta | median delta | q05 delta | q95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0330 | +0.0190 | +0.1054 | +0.0036 |
| 1 | +0.0345 | +0.0198 | +0.1088 | +0.0036 |
| 2 | +0.0361 | +0.0208 | +0.1162 | +0.0038 |
| 4 | +0.0389 | +0.0217 | +0.1328 | +0.0042 |
| 8 | +0.0424 | +0.0206 | +0.1666 | +0.0048 |
| 16 | +0.0433 | +0.0169 | +0.1946 | +0.0059 |
| 32 | +0.0370 | +0.0139 | +0.1704 | +0.0079 |
| 64 | +0.0266 | +0.0118 | +0.1022 | +0.0105 |
| 128 | +0.0175 | +0.0130 | +0.0140 | +0.0129 |
| 256 | +0.0114 | +0.0129 | -0.0022 | +0.0158 |
| 512 | +0.0086 | +0.0117 | -0.0016 | +0.0209 |
| 1024 | +0.0093 | +0.0173 | -0.0004 | +0.0277 |

Coefficient deltas are non-trivial even with the same initialization and
formula. Across the matched 8 users x 15 parameters, `abs(train365 -
train1825)` has mean 2.115, median 1.437, q95 6.491, and max 9.844. Users 7
and 8 changed the most by L2 norm, both about 18.2.

## Monotonicity Diagnostics

Both policies preserve the intended Cost-ADR guarantee: retention is
non-increasing in cost weight. There were zero `w` monotonicity violations in
the grid scan.

| policy set | axis | counts | mixed share |
| --- | --- | --- | ---: |
| train365 | w | `{'decreasing': 13274, 'flat': 174}` | 0.000% |
| train365 | S | `{'increasing': 1749, 'mixed': 3218, 'decreasing': 281}` | 61.319% |
| train365 | D | `{'decreasing': 3206, 'flat': 176, 'increasing': 1505, 'mixed': 361}` | 6.879% |
| train1825 | w | `{'decreasing': 13275, 'flat': 173}` | 0.000% |
| train1825 | S | `{'increasing': 1625, 'mixed': 3504, 'decreasing': 119}` | 66.768% |
| train1825 | D | `{'decreasing': 2913, 'mixed': 834, 'increasing': 1393, 'flat': 108}` | 15.892% |

The longer-horizon policy is more non-monotone in both S and D, especially D.
That is consistent with it learning more specialized tradeoffs across the
longer trajectory instead of a uniformly conservative surface.

## Interpretation

Matched horizon wins. The 365-day trained policy is better on 365-day eval:
HV delta is 8,911 versus 6,102 for the 1825-trained policy, relative
time-save AUC is 7.49% versus 6.40%, and target coverage is 86.8% versus
77.5%. The short-horizon policy is also much less coverage-fragile on the
short horizon: budget coverage is 88.8% versus 58.2%.

The 1825-day trained policy is better on 1825-day eval: HV delta is 105,566
versus 84,943, relative time-save AUC is 13.79% versus 11.66%, and target
coverage is 90.2% versus 87.9%. The exception is budget coverage: the
365-trained policy covers 97.7% of the baseline budget span on 1825-day eval,
while the 1825-trained policy covers 74.6%. The 1825 policy is more aggressive
and finds a better frontier, but it covers a narrower budget span.

The policy distribution explains the transfer behavior. The 365-day policy
keeps retention higher almost everywhere, with the largest mean increase around
low and mid cost weights. That conservatism helps the 365-day horizon, where
long-horizon deferred gains have less time to pay off. On the 1825-day horizon,
the lower-retention and more state-shaped 1825 policy reaches better
time-memory tradeoffs.

Cross-horizon transfer is asymmetric. The 365-day policy transfers reasonably
to 1825 days but leaves HV on the table. The 1825-day policy transfers worse to
365 days: it remains positive on aggregate HV, but user 1 is slightly negative
and coverage drops sharply.

## GPU Monitor

Formal training stages wrote GPU monitor artifacts. Neither training run
showed shared-memory spill:

| train horizon | shared-memory spill | peak shared memory | peak `nvidia-smi` memory |
| ---: | --- | ---: | ---: |
| 365 | false | 371,806,208 bytes | 3,846 MiB |
| 1825 | false | 241,356,800 bytes | 12,319 MiB |

The direct cross-evaluation sweeps completed successfully, but
`run_sweep_users_batched.py` does not emit the formal `gpu_monitor` artifact
outside `run_experiment.py` sweep stages, so spill claims are limited to the
training monitor evidence above.

## Limitations

This is users 1-8 only and FSRS6 environment only. The 365-day experiment reuses
the existing first-eight FSRS6 baseline DR manifest selected for the 1825-day
formal family. That keeps the baseline grid identical across cells, but it is
not an independently optimized 365-day baseline grid.
