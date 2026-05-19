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
| Oracle stationary finite distill | sweep | cuda | 94.6 | 154.4 | - |
| Oracle stationary finite distill | train-overfit | cuda | 917.8 | 15.9 | 254.5 |
| ADR | sweep | cuda | 44.1 | 330.8 | - |
| ADR | train-overfit | cuda | 305.7 | 47.8 | 764.3 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | 54077ffe988aaf74eb90dfb52d3b2eae1300590d | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | 54077ffe988aaf74eb90dfb52d3b2eae1300590d | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 142.2 | 160.4 | False | 2,554.0 |
| Oracle stationary finite distill | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 142.2 | 160.4 | False | 1,526.0 |
| ADR | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 299.9 | 318.1 | False | 2,664.0 |
| ADR | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 158.3 | 176.5 | False | 1,488.0 |

## Conclusion

Promotion decision for `fsrs6_oracle_stationary_finite_distill` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 3,407 HV, 37.3 same-budget memory lift AUC, -0.30 same-target time saved AUC versus comparison.
- lstm: -16,347 HV, 16.0 same-budget memory lift AUC, -1.43 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 99,273 | +3.316% | 128 | 118.8 | +1.826% | 90/110, 66.331% span | 4.47 | +13.769% | 89/110, 87.316% span |
| fsrs6 | ADR | 95,866 | +3.202% | 128 | 81.4 | +1.228% | 80/110, 81.322% span | 4.77 | +11.168% | 78/110, 76.609% span |
| fsrs6 | Oracle stationary finite distill - ADR | 3,407 | +0.114% | 0 | 37.3 | +0.598% | +10, -14.991 pp span | -0.30 | +2.600% | +11, +10.707 pp span |
| lstm | Oracle stationary finite distill | 37,867 | +1.445% | 127 | 65.2 | +1.000% | 89/117, 68.771% span | 0.26 | +5.056% | 93/117, 88.296% span |
| lstm | ADR | 54,215 | +2.068% | 124 | 49.3 | +0.746% | 83/117, 83.153% span | 1.69 | +4.927% | 79/117, 81.533% span |
| lstm | Oracle stationary finite distill - ADR | -16,347 | -0.624% | 3 | 16.0 | +0.254% | +6, -14.382 pp span | -1.43 | +0.129% | +14, +6.763 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 7/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11,897 | 10,832 | 1,066 | 3,470 | 5,399 | -1,929 |
| 2 | 25,713 | 32,314 | -6,601 | 10,243 | 19,434 | -9,191 |
| 3 | 4,430 | 4,180 | 249 | 3,028 | 3,203 | -175 |
| 4 | 38,541 | 33,240 | 5,301 | 11,497 | 18,141 | -6,644 |
| 5 | 8,968 | 7,748 | 1,219 | 3,238 | 3,234 | 4 |
| 6 | 5,524 | 4,474 | 1,050 | 3,552 | 2,821 | 731 |
| 7 | 2,281 | 1,034 | 1,248 | 1,685 | 516 | 1,169 |
| 8 | 1,919 | 2,045 | -126 | 1,155 | 1,466 | -311 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 6,530.2 | 45.40 | 26.74 | 180.43 |
| fsrs6 | ADR | 6,548.2 | 51.96 | 26.14 | 211.25 |
| lstm | Oracle stationary finite distill | 6,389.8 | 54.65 | 24.00 | 228.22 |
| lstm | ADR | 6,419.7 | 62.39 | 23.37 | 269.98 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill | 1 | 14,621 |
| Oracle stationary finite distill | 2 | 34,368 |
| Oracle stationary finite distill | 3 | 5,634 |
| Oracle stationary finite distill | 4 | 40,676 |
| Oracle stationary finite distill | 5 | 9,708 |
| Oracle stationary finite distill | 6 | 6,234 |
| Oracle stationary finite distill | 7 | 2,431 |
| Oracle stationary finite distill | 8 | 2,094 |
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
| Oracle stationary finite distill | 8 | 115,766 |
| ADR | 8 | 107,385 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-18-fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.md`
