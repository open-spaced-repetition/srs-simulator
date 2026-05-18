# ADR time vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/report/report_summary.json`

## Question

Evaluate whether an FSRS6 ADR policy with normalized remaining simulation time improves the matched-budget ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_adr_time_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_time` | `experiments/rl_scheduler/configs/fsrs6_adr_time_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- ADR time: `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| ADR time | analyze-pareto | yes | - |
| ADR time | build-pareto | yes | - |
| ADR time | preflight | yes | - |
| ADR time | stage-baseline | yes | - |
| ADR time | sweep | yes | - |
| ADR time | train-overfit | yes | - |
| ADR | analyze-pareto | yes | - |
| ADR | build-pareto | yes | - |
| ADR | preflight | yes | - |
| ADR | stage-baseline | yes | - |
| ADR | sweep | yes | - |
| ADR | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| ADR time | sweep | cuda | 52.0 | 280.6 | - |
| ADR time | train-overfit | cuda | 361.6 | 40.4 | 646.0 |
| ADR | sweep | cuda | 45.1 | 323.5 | - |
| ADR | train-overfit | cuda | 328.8 | 44.4 | 710.4 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| ADR time | 9fade5c657c4e19f26427a7537139258a0ed085a | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | c1a791b35e56ef777005d8329a11ac42e34ab707 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| ADR time | sweep | `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/sweep/gpu_monitor/summary.json` | 166.0 | 184.3 | False | 3,344.0 |
| ADR time | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/train-overfit/gpu_monitor/summary.json` | 372.0 | 390.2 | False | 2,180.0 |
| ADR | - | - | - | - | - | - |

## Conclusion

Do not promote `fsrs6_adr_time` over ordinary `fsrs6_adr` for the matched-budget portfolio.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 520 HV, 13.5 budget-memory gain AUC, -0.05 memory-target regret AUC versus comparison.
- lstm: -7,419 HV, 14.8 budget-memory gain AUC, 0.86 memory-target regret AUC versus comparison.
- The FSRS6 gain is marginal, while the LSTM regression is material on HV and absolute memory-target regret. The user-simple relative regret result is close and does not overturn that decision.
- Training HV is slightly higher for ADR time (`108,373` vs `107,746`), but that extra training objective value does not survive the external LSTM Pareto check.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and budget-memory gain are better. Negative memory-target regret is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | budget-memory gain AUC | budget-memory gain / baseline | budget coverage | memory-target regret AUC | memory-target regret / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | ADR time | 97,400 | +3.501% | 128 | 104.2 | +1.620% | 83/115, 79.851% span | -4.62 | -12.061% | 80/115, 77.083% span |
| fsrs6 | ADR | 96,880 | +3.482% | 128 | 90.7 | +1.400% | 88/115, 81.550% span | -4.57 | -11.555% | 81/115, 78.146% span |
| fsrs6 | ADR time - ADR | 520 | +0.019% | 0 | 13.5 | +0.220% | -5, -1.699 pp span | -0.05 | -0.506% | -1, -1.063 pp span |
| lstm | ADR time | 48,430 | +1.705% | 125 | 76.9 | +1.190% | 79/111, 82.098% span | -0.96 | -6.084% | 80/111, 82.908% span |
| lstm | ADR | 55,849 | +1.966% | 125 | 62.2 | +0.953% | 79/111, 82.466% span | -1.82 | -5.906% | 81/111, 83.775% span |
| lstm | ADR time - ADR | -7,419 | -0.261% | 0 | 14.8 | +0.237% | +0, -0.369 pp span | 0.86 | -0.177% | -1, -0.867 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 6/16 environment-user rows.

| user | fsrs6 ADR time HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm ADR time HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9,416 | 9,894 | -478 | 5,356 | 7,161 | -1,805 |
| 2 | 26,766 | 27,715 | -949 | 10,981 | 13,315 | -2,334 |
| 3 | 4,076 | 3,745 | 331 | 3,493 | 3,364 | 129 |
| 4 | 36,958 | 37,079 | -120 | 13,805 | 19,119 | -5,314 |
| 5 | 9,822 | 8,964 | 859 | 4,946 | 4,110 | 836 |
| 6 | 6,360 | 6,327 | 34 | 6,712 | 6,491 | 221 |
| 7 | 1,828 | 1,238 | 589 | 1,285 | 733 | 552 |
| 8 | 2,173 | 1,918 | 255 | 1,851 | 1,556 | 295 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | ADR time | 6,543.7 | 51.74 | 25.81 | 200.41 |
| fsrs6 | ADR | 6,553.0 | 50.42 | 25.42 | 195.90 |
| lstm | ADR time | 6,427.0 | 63.15 | 23.30 | 264.54 |
| lstm | ADR | 6,431.5 | 60.86 | 22.88 | 255.29 |

## Time-Feature Effect

`T_remaining_norm` is `(simulation_days - day) / simulation_days`, clipped to `[0, 1]`. In this 1825-day experiment, `T=0.2` means 365 days remain.

Across the 128 trained portfolio children, the time terms mostly act as an early-horizon retention lift that fades by the last simulation year.

| time comparison | average delta across a 3x3 normalized S/D grid | median delta across child policies |
| --- | --- | --- |
| `T=1.0` vs `T=0.0` | +4.64 pp retention | +1.33 pp |
| `T=1.0` vs `T=0.2` | +3.84 pp retention | +0.96 pp |
| `T=0.2` vs `T=0.0` | +0.80 pp retention | +0.33 pp |

State dependence is real:

- Weak, hard cards (`S≈1.0d`, `D≈8.2`) see a smaller average early-time lift: `+2.64 pp` from `T=0.0` to `T=1.0`.
- Mid-state cards (`S≈30.2d`, `D≈5.5`) see `+5.08 pp` on average, with a median `+1.50 pp`.
- Strong, easy cards (`S≈929d`, `D≈2.8`) see the largest average lift: `+5.76 pp`.

The raw time coefficients are mixed rather than uniformly positive:

- `T` is positive in `80/128` child policies.
- `S*T` is positive in `92/128`.
- `D*T` is positive in `70/128`.
- `T^2` is positive in `56/128`.

So the learned time behavior comes more from interactions and curvature than from a single positive offset. In practice it nudges many policies toward higher retention earlier in the run, which is consistent with the slightly higher policy-point review/time load seen above, but that extra effort does not convert into a better external LSTM Pareto frontier.

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| ADR time | 1 | 12,194 |
| ADR time | 2 | 30,559 |
| ADR time | 3 | 5,222 |
| ADR time | 4 | 37,989 |
| ADR time | 5 | 10,635 |
| ADR time | 6 | 7,447 |
| ADR time | 7 | 2,038 |
| ADR time | 8 | 2,290 |
| ADR | 1 | 12,771 |
| ADR | 2 | 31,067 |
| ADR | 3 | 4,980 |
| ADR | 4 | 38,251 |
| ADR | 5 | 9,555 |
| ADR | 6 | 7,609 |
| ADR | 7 | 1,542 |
| ADR | 8 | 1,971 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| ADR time | 8 | 108,373 |
| ADR | 8 | 107,746 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/fsrs6_adr_time_portfolio_users_1_8_pop16_v1/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-13-fsrs6_adr_time_portfolio_users_1_8_pop16_v1.md`
