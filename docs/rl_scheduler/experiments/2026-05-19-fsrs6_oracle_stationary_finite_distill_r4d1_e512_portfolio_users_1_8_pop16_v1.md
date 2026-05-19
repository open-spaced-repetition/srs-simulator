# Oracle stationary finite distill r4d1 e512 vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether per-user FSRS6 oracle stationary finite distill r4d1/e512 policies with searched goal-cost weights can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_oracle_stationary_finite_distill` | `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Oracle stationary finite distill r4d1 e512: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

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
| Oracle stationary finite distill r4d1 e512 | sweep | cuda | 90.4 | 161.5 | - |
| Oracle stationary finite distill r4d1 e512 | train-overfit | cuda | 971.5 | 15.0 | 240.5 |
| ADR | sweep | cuda | 44.1 | 330.8 | - |
| ADR | train-overfit | cuda | 305.7 | 47.8 | 764.3 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | 54077ffe988aaf74eb90dfb52d3b2eae1300590d | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | 54077ffe988aaf74eb90dfb52d3b2eae1300590d | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 173.0 | 191.2 | False | 2,628.0 |
| Oracle stationary finite distill r4d1 e512 | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 143.0 | 161.3 | False | 1,527.0 |
| ADR | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 299.9 | 318.1 | False | 2,664.0 |
| ADR | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 158.3 | 176.5 | False | 1,488.0 |

## Conclusion

Do not promote `fsrs6_oracle_stationary_finite_distill`.

With the matched portfolio budget, Oracle stationary finite distill r4d1 e512 remains behind ADR on HV; same-budget memory lift and same-target time saved deltas are:

- fsrs6: -58,627 HV, 0.9 same-budget memory lift AUC, -2.19 same-target time saved AUC versus comparison.
- lstm: -41,066 HV, -6.8 same-budget memory lift AUC, -2.21 same-target time saved AUC versus comparison.

Training HV gains and lower-time sampled policy points do not survive external Pareto evaluation.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill r4d1 e512 | 37,238 | +1.244% | 128 | 82.4 | +1.266% | 90/110, 68.457% span | 2.58 | +8.810% | 91/110, 89.829% span |
| fsrs6 | ADR | 95,866 | +3.202% | 128 | 81.4 | +1.228% | 80/110, 81.322% span | 4.77 | +11.168% | 78/110, 76.609% span |
| fsrs6 | Oracle stationary finite distill r4d1 e512 - ADR | -58,627 | -1.958% | 0 | 0.9 | +0.038% | +10, -12.865 pp span | -2.19 | -2.358% | +13, +13.220 pp span |
| lstm | Oracle stationary finite distill r4d1 e512 | 13,149 | +0.502% | 127 | 42.4 | +0.652% | 95/117, 73.812% span | -0.52 | +2.810% | 98/117, 95.147% span |
| lstm | ADR | 54,215 | +2.068% | 124 | 49.3 | +0.746% | 83/117, 83.153% span | 1.69 | +4.927% | 79/117, 81.533% span |
| lstm | Oracle stationary finite distill r4d1 e512 - ADR | -41,066 | -1.567% | 3 | -6.8 | -0.094% | +12, -9.341 pp span | -2.21 | -2.117% | +19, +13.614 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 11/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill r4d1 e512 HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill r4d1 e512 HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,285 | 10,832 | -2,546 | 2,288 | 5,399 | -3,112 |
| 2 | -18,772 | 32,314 | -51,086 | -2,985 | 19,434 | -22,419 |
| 3 | 4,275 | 4,180 | 94 | 2,216 | 3,203 | -987 |
| 4 | 28,979 | 33,240 | -4,261 | 5,715 | 18,141 | -12,426 |
| 5 | 8,388 | 7,748 | 640 | 2,385 | 3,234 | -849 |
| 6 | 5,274 | 4,474 | 800 | 4,158 | 2,821 | 1,337 |
| 7 | -1,608 | 1,034 | -2,642 | -1,946 | 516 | -2,462 |
| 8 | 2,417 | 2,045 | 372 | 1,317 | 1,466 | -148 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill r4d1 e512 | 6,542.0 | 52.05 | 25.09 | 203.89 |
| fsrs6 | ADR | 6,548.2 | 51.96 | 26.14 | 211.25 |
| lstm | Oracle stationary finite distill r4d1 e512 | 6,390.7 | 63.43 | 22.00 | 262.42 |
| lstm | ADR | 6,419.7 | 62.39 | 23.37 | 269.98 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | 1 | 12,958 |
| Oracle stationary finite distill r4d1 e512 | 2 | 28,318 |
| Oracle stationary finite distill r4d1 e512 | 3 | 5,376 |
| Oracle stationary finite distill r4d1 e512 | 4 | 34,128 |
| Oracle stationary finite distill r4d1 e512 | 5 | 9,040 |
| Oracle stationary finite distill r4d1 e512 | 6 | 5,840 |
| Oracle stationary finite distill r4d1 e512 | 7 | 499 |
| Oracle stationary finite distill r4d1 e512 | 8 | 3,104 |
| ADR | 1 | 13,563 |
| ADR | 2 | 37,662 |
| ADR | 3 | 5,463 |
| ADR | 4 | 33,770 |
| ADR | 5 | 8,108 |
| ADR | 6 | 5,431 |
| ADR | 7 | 1,294 |
| ADR | 8 | 2,093 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| Oracle stationary finite distill r4d1 e512 | 8 | 99,263 |
| ADR | 8 | 107,385 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-19-fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.md`
