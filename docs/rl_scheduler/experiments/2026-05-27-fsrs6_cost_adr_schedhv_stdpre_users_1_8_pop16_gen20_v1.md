# FSRS6 Cost ADR schedHV stdpre vs ADR Portfolio experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether a 24-parameter Cost-ADR policy initialized from the first-eight single-card distill mean, trained with scheduler-only HV, and diagonal-preconditioned by the first-eight sample std reaches the matched-budget FSRS6 ADR portfolio on the matched 16 cost weights.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- FSRS6 Cost ADR schedHV stdpre: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR Portfolio: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | analyze-pareto | yes | - |
| FSRS6 Cost ADR schedHV stdpre | build-pareto | yes | - |
| FSRS6 Cost ADR schedHV stdpre | preflight | yes | - |
| FSRS6 Cost ADR schedHV stdpre | stage-baseline | yes | - |
| FSRS6 Cost ADR schedHV stdpre | sweep | yes | - |
| FSRS6 Cost ADR schedHV stdpre | train-overfit | yes | - |
| ADR Portfolio | analyze-pareto | yes | - |
| ADR Portfolio | build-pareto | yes | - |
| ADR Portfolio | preflight | yes | - |
| ADR Portfolio | stage-baseline | yes | - |
| ADR Portfolio | sweep | yes | - |
| ADR Portfolio | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | sweep | cuda | 50.9 | 286.7 | - |
| FSRS6 Cost ADR schedHV stdpre | train-overfit | cuda | 764.4 | 19.1 | 305.6 |
| ADR Portfolio | sweep | cuda | 44.8 | 325.9 | - |
| ADR Portfolio | train-overfit | cuda | 302.2 | 48.3 | 773.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | 6211afb0d88c48e24ff273271fded4ad939ed887 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR Portfolio | 88de85231acfebb4e7826cfa726d26f9f6322e95 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 171.3 | 189.5 | False | 5,390.0 |
| FSRS6 Cost ADR schedHV stdpre | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 376.4 | 394.6 | False | 5,380.0 |
| ADR Portfolio | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 216.6 | 234.8 | False | 3,478.0 |
| ADR Portfolio | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 214.0 | 232.3 | False | 2,258.0 |

## Conclusion

Promotion decision for `fsrs6_cost_adr` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 1,641 HV, 5.7 same-budget memory lift AUC, 0.13 same-target time saved AUC versus comparison.
- lstm: -12,829 HV, 2.7 same-budget memory lift AUC, -1.32 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre | 97,711 | +3.161% | 122 | 91.4 | +1.394% | 90/112, 87.137% span | 4.68 | +12.793% | 86/112, 82.331% span |
| fsrs6 | ADR Portfolio | 96,070 | +3.108% | 127 | 85.7 | +1.302% | 86/112, 83.234% span | 4.56 | +11.371% | 79/112, 74.096% span |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre - ADR Portfolio | 1,641 | +0.053% | -5 | 5.7 | +0.092% | +4, +3.903 pp span | 0.13 | +1.422% | +7, +8.235 pp span |
| lstm | FSRS6 Cost ADR schedHV stdpre | 40,052 | +1.244% | 118 | 52.9 | +0.814% | 89/114, 86.716% span | 0.85 | +4.590% | 91/114, 91.662% span |
| lstm | ADR Portfolio | 52,881 | +1.642% | 121 | 50.2 | +0.764% | 81/114, 80.385% span | 2.16 | +5.359% | 85/114, 83.872% span |
| lstm | FSRS6 Cost ADR schedHV stdpre - ADR Portfolio | -12,829 | -0.398% | -3 | 2.7 | +0.051% | +8, +6.330 pp span | -1.32 | -0.769% | +6, +7.790 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 8/16 environment-user rows.

| user | fsrs6 FSRS6 Cost ADR schedHV stdpre HV delta | fsrs6 ADR Portfolio HV delta | fsrs6 delta | lstm FSRS6 Cost ADR schedHV stdpre HV delta | lstm ADR Portfolio HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,820 | 12,081 | 2,738 | 14,509 | 9,639 | 4,869 |
| 2 | 29,774 | 29,679 | 95 | 4,740 | 15,362 | -10,622 |
| 3 | 3,571 | 3,835 | -264 | 2,843 | 3,144 | -301 |
| 4 | 31,571 | 34,918 | -3,347 | 10,231 | 16,379 | -6,148 |
| 5 | 9,243 | 7,564 | 1,679 | 1,292 | 1,933 | -641 |
| 6 | 4,955 | 4,745 | 210 | 4,533 | 4,578 | -45 |
| 7 | 1,186 | 1,222 | -35 | 650 | 640 | 10 |
| 8 | 2,591 | 2,026 | 565 | 1,253 | 1,206 | 48 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre | 6,398.3 | 58.47 | 27.25 | 229.47 |
| fsrs6 | ADR Portfolio | 6,552.9 | 55.29 | 26.16 | 228.70 |
| lstm | FSRS6 Cost ADR schedHV stdpre | 6,260.7 | 73.20 | 24.15 | 300.83 |
| lstm | ADR Portfolio | 6,422.7 | 67.86 | 23.46 | 297.02 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | 1 | 15,101 |
| FSRS6 Cost ADR schedHV stdpre | 2 | 29,853 |
| FSRS6 Cost ADR schedHV stdpre | 3 | 3,502 |
| FSRS6 Cost ADR schedHV stdpre | 4 | 31,194 |
| FSRS6 Cost ADR schedHV stdpre | 5 | 9,436 |
| FSRS6 Cost ADR schedHV stdpre | 6 | 5,019 |
| FSRS6 Cost ADR schedHV stdpre | 7 | 1,182 |
| FSRS6 Cost ADR schedHV stdpre | 8 | 2,619 |
| ADR Portfolio | 1 | 15,946 |
| ADR Portfolio | 2 | 34,901 |
| ADR Portfolio | 3 | 5,046 |
| ADR Portfolio | 4 | 35,955 |
| ADR Portfolio | 5 | 8,286 |
| ADR Portfolio | 6 | 5,692 |
| ADR Portfolio | 7 | 1,448 |
| ADR Portfolio | 8 | 2,044 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre | 8 | 97,905 |
| ADR Portfolio | 8 | 109,319 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.md`
