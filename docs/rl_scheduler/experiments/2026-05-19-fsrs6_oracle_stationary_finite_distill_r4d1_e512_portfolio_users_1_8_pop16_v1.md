# Oracle stationary finite distill r4d1 e512 vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/report/report_summary.json`

## Question

Evaluate whether per-user FSRS6 oracle stationary finite distill r4d1/e512 policies with searched goal-cost weights can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1` | `fsrs6_oracle_stationary_finite_distill` | `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Oracle stationary finite distill r4d1 e512: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | analyze-pareto | yes | - |
| Oracle stationary finite distill r4d1 e512 | build-pareto | yes | - |
| Oracle stationary finite distill r4d1 e512 | preflight | yes | - |
| Oracle stationary finite distill r4d1 e512 | stage-baseline | yes | - |
| Oracle stationary finite distill r4d1 e512 | sweep | yes | - |
| Oracle stationary finite distill r4d1 e512 | train-overfit | yes | - |
| ADR | analyze-pareto | yes | - |
| ADR | build-pareto | yes | - |
| ADR | preflight | yes | - |
| ADR | stage-baseline | yes | - |
| ADR | sweep | yes | - |
| ADR | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | sweep | cuda | 95.6 | 152.7 | - |
| Oracle stationary finite distill r4d1 e512 | train-overfit | cuda | 950.9 | 15.4 | 245.7 |
| ADR | sweep | cuda | 45.1 | 323.5 | - |
| ADR | train-overfit | cuda | 328.8 | 44.4 | 710.4 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | d7155e700e86c522f50a35b159247459778fa432 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | c1a791b35e56ef777005d8329a11ac42e34ab707 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/sweep/gpu_monitor/summary.json` | 144.1 | 162.3 | False | 2,637.0 |
| Oracle stationary finite distill r4d1 e512 | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/train-overfit/gpu_monitor/summary.json` | 156.1 | 174.3 | False | 1,548.0 |
| ADR | - | - | - | - | - | - |

## Conclusion

Do not promote `fsrs6_oracle_stationary_finite_distill`.

With the matched portfolio budget, Oracle stationary finite distill r4d1 e512 remains behind ADR on HV; same-budget memory lift and same-target time saved deltas are:

- fsrs6: -50,029 HV, -7.5 same-budget memory lift AUC, -1.77 same-target time saved AUC versus comparison.
- lstm: -45,945 HV, -17.0 same-budget memory lift AUC, -1.94 same-target time saved AUC versus comparison.

Training HV gains and lower-time sampled policy points do not survive external Pareto evaluation.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill r4d1 e512 | 46,851 | +1.684% | 128 | 83.2 | +1.278% | 96/115, 73.835% span | 2.80 | +9.418% | 95/115, 90.504% span |
| fsrs6 | ADR | 96,880 | +3.482% | 128 | 90.7 | +1.400% | 88/115, 81.550% span | 4.57 | +11.555% | 81/115, 78.146% span |
| fsrs6 | Oracle stationary finite distill r4d1 e512 - ADR | -50,029 | -1.798% | 0 | -7.5 | -0.122% | +8, -7.716 pp span | -1.77 | -2.137% | +14, +12.358 pp span |
| lstm | Oracle stationary finite distill r4d1 e512 | 9,904 | +0.349% | 127 | 45.1 | +0.692% | 93/111, 70.615% span | -0.12 | +3.356% | 97/111, 95.914% span |
| lstm | ADR | 55,849 | +1.966% | 125 | 62.2 | +0.953% | 79/111, 82.466% span | 1.82 | +5.906% | 81/111, 83.775% span |
| lstm | Oracle stationary finite distill r4d1 e512 - ADR | -45,945 | -1.617% | 2 | -17.0 | -0.261% | +14, -11.852 pp span | -1.94 | -2.550% | +16, +12.139 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 10/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill r4d1 e512 HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill r4d1 e512 HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,044 | 9,894 | -1,851 | 3,351 | 7,161 | -3,810 |
| 2 | -16,273 | 27,715 | -43,987 | -11,804 | 13,315 | -25,119 |
| 3 | 4,026 | 3,745 | 281 | 2,492 | 3,364 | -873 |
| 4 | 32,792 | 37,079 | -4,286 | 6,011 | 19,119 | -13,108 |
| 5 | 9,332 | 8,964 | 368 | 2,407 | 4,110 | -1,703 |
| 6 | 7,687 | 6,327 | 1,360 | 7,485 | 6,491 | 993 |
| 7 | -1,366 | 1,238 | -2,604 | -1,739 | 733 | -2,473 |
| 8 | 2,607 | 1,918 | 689 | 1,703 | 1,556 | 147 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill r4d1 e512 | 6,539.4 | 52.42 | 24.82 | 201.29 |
| fsrs6 | ADR | 6,553.0 | 50.42 | 25.42 | 195.90 |
| lstm | Oracle stationary finite distill r4d1 e512 | 6,394.1 | 62.76 | 21.80 | 253.95 |
| lstm | ADR | 6,431.5 | 60.86 | 22.88 | 255.29 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | 1 | 12,268 |
| Oracle stationary finite distill r4d1 e512 | 2 | 22,975 |
| Oracle stationary finite distill r4d1 e512 | 3 | 5,113 |
| Oracle stationary finite distill r4d1 e512 | 4 | 38,323 |
| Oracle stationary finite distill r4d1 e512 | 5 | 10,250 |
| Oracle stationary finite distill r4d1 e512 | 6 | 8,019 |
| Oracle stationary finite distill r4d1 e512 | 7 | 485 |
| Oracle stationary finite distill r4d1 e512 | 8 | 3,018 |
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
| Oracle stationary finite distill r4d1 e512 | 8 | 100,452 |
| ADR | 8 | 107,746 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-19-fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.md`
