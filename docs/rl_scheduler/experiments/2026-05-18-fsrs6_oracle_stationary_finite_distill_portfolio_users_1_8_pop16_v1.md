# Oracle stationary finite distill vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether per-user FSRS6 oracle stationary finite distill policies with searched goal-cost weights can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_oracle_stationary_finite_distill` | `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Oracle stationary finite distill: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| Oracle stationary finite distill | analyze-pareto | yes | - |
| Oracle stationary finite distill | build-pareto | yes | - |
| Oracle stationary finite distill | preflight | yes | - |
| Oracle stationary finite distill | stage-baseline | yes | - |
| Oracle stationary finite distill | sweep | yes | - |
| Oracle stationary finite distill | train-overfit | yes | - |
| ADR | analyze-pareto | yes | - |
| ADR | build-pareto | yes | - |
| ADR | preflight | yes | - |
| ADR | stage-baseline | yes | - |
| ADR | sweep | yes | - |
| ADR | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | sweep | cuda | 94.9 | 153.9 | - |
| Oracle stationary finite distill | train-overfit | cuda | 1,022.8 | 14.3 | 228.4 |
| ADR | sweep | cuda | 44.8 | 325.9 | - |
| ADR | train-overfit | cuda | 302.2 | 48.3 | 773.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | 88de85231acfebb4e7826cfa726d26f9f6322e95 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | 88de85231acfebb4e7826cfa726d26f9f6322e95 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 194.9 | 213.2 | False | 3,406.0 |
| Oracle stationary finite distill | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 377.5 | 395.7 | False | 3,260.0 |
| ADR | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 216.6 | 234.8 | False | 3,478.0 |
| ADR | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 214.0 | 232.3 | False | 2,258.0 |

## Conclusion

Promotion decision for `fsrs6_oracle_stationary_finite_distill` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 6,990 HV, 30.6 same-budget memory lift AUC, -0.15 same-target time saved AUC versus comparison.
- lstm: -10,360 HV, 17.0 same-budget memory lift AUC, -1.54 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 103,060 | +3.334% | 128 | 116.2 | +1.788% | 96/112, 81.188% span | 4.41 | +13.457% | 91/112, 82.368% span |
| fsrs6 | ADR | 96,070 | +3.108% | 127 | 85.7 | +1.302% | 86/112, 83.234% span | 4.56 | +11.371% | 79/112, 74.096% span |
| fsrs6 | Oracle stationary finite distill - ADR | 6,990 | +0.226% | 1 | 30.6 | +0.486% | +10, -2.047 pp span | -0.15 | +2.086% | +12, +8.272 pp span |
| lstm | Oracle stationary finite distill | 42,521 | +1.321% | 126 | 67.2 | +1.031% | 96/114, 84.598% span | 0.62 | +5.575% | 96/114, 89.497% span |
| lstm | ADR | 52,881 | +1.642% | 121 | 50.2 | +0.764% | 81/114, 80.385% span | 2.16 | +5.359% | 85/114, 83.872% span |
| lstm | Oracle stationary finite distill - ADR | -10,360 | -0.322% | 5 | 17.0 | +0.267% | +15, +4.212 pp span | -1.54 | +0.217% | +11, +5.625 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 5/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13,325 | 12,081 | 1,243 | 8,395 | 9,639 | -1,245 |
| 2 | 26,501 | 29,679 | -3,178 | 10,611 | 15,362 | -4,751 |
| 3 | 4,667 | 3,835 | 832 | 3,225 | 3,144 | 81 |
| 4 | 39,078 | 34,918 | 4,160 | 9,248 | 16,379 | -7,131 |
| 5 | 9,006 | 7,564 | 1,442 | 1,605 | 1,933 | -328 |
| 6 | 5,842 | 4,745 | 1,097 | 6,514 | 4,578 | 1,936 |
| 7 | 2,435 | 1,222 | 1,213 | 1,697 | 640 | 1,057 |
| 8 | 2,207 | 2,026 | 181 | 1,227 | 1,206 | 21 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 6,530.0 | 52.84 | 26.37 | 215.48 |
| fsrs6 | ADR | 6,552.9 | 55.29 | 26.16 | 228.70 |
| lstm | Oracle stationary finite distill | 6,401.1 | 65.03 | 23.61 | 274.73 |
| lstm | ADR | 6,422.7 | 67.86 | 23.46 | 297.02 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill | 1 | 17,583 |
| Oracle stationary finite distill | 2 | 32,874 |
| Oracle stationary finite distill | 3 | 5,668 |
| Oracle stationary finite distill | 4 | 41,104 |
| Oracle stationary finite distill | 5 | 9,967 |
| Oracle stationary finite distill | 6 | 6,317 |
| Oracle stationary finite distill | 7 | 2,630 |
| Oracle stationary finite distill | 8 | 2,334 |
| ADR | 1 | 15,946 |
| ADR | 2 | 34,901 |
| ADR | 3 | 5,046 |
| ADR | 4 | 35,955 |
| ADR | 5 | 8,286 |
| ADR | 6 | 5,692 |
| ADR | 7 | 1,448 |
| ADR | 8 | 2,044 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| Oracle stationary finite distill | 8 | 118,478 |
| ADR | 8 | 109,319 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-18-fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.md`
