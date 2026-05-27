# FSRS6 Cost ADR meaninit16w vs ADR Portfolio experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether a 24-parameter Cost-ADR policy initialized from the first-eight single-card distill mean and tuned/evaluated on the matched 16 cost weights reaches the matched-budget FSRS6 ADR portfolio.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=6.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- FSRS6 Cost ADR meaninit16w: `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR Portfolio: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| FSRS6 Cost ADR meaninit16w | analyze-pareto | yes | - |
| FSRS6 Cost ADR meaninit16w | build-pareto | yes | - |
| FSRS6 Cost ADR meaninit16w | preflight | yes | - |
| FSRS6 Cost ADR meaninit16w | stage-baseline | yes | - |
| FSRS6 Cost ADR meaninit16w | sweep | yes | - |
| FSRS6 Cost ADR meaninit16w | train-overfit | yes | - |
| ADR Portfolio | analyze-pareto | yes | - |
| ADR Portfolio | build-pareto | yes | - |
| ADR Portfolio | preflight | yes | - |
| ADR Portfolio | stage-baseline | yes | - |
| ADR Portfolio | sweep | yes | - |
| ADR Portfolio | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR meaninit16w | sweep | cuda | 54.9 | 265.7 | - |
| FSRS6 Cost ADR meaninit16w | train-overfit | cuda | 833.5 | 17.5 | 280.3 |
| ADR Portfolio | sweep | cuda | 44.8 | 325.9 | - |
| ADR Portfolio | train-overfit | cuda | 302.2 | 48.3 | 773.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR meaninit16w | 164fe28ea730100eef07d01d66e4d6c414b42e4e | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR Portfolio | 88de85231acfebb4e7826cfa726d26f9f6322e95 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR meaninit16w | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 174.1 | 192.3 | False | 5,998.0 |
| FSRS6 Cost ADR meaninit16w | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 243.2 | 261.4 | False | 5,989.0 |
| ADR Portfolio | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 216.6 | 234.8 | False | 3,478.0 |
| ADR Portfolio | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 214.0 | 232.3 | False | 2,258.0 |

## Conclusion

Do not promote `fsrs6_cost_adr`.

With the matched training budget, FSRS6 Cost ADR meaninit16w remains behind ADR Portfolio on HV; same-budget memory lift and same-target time saved deltas are:

- fsrs6: -273,880 HV, -109.3 same-budget memory lift AUC, -6.33 same-target time saved AUC versus comparison.
- lstm: -176,419 HV, -101.7 same-budget memory lift AUC, -4.06 same-target time saved AUC versus comparison.

Training HV gains and lower-time sampled policy points do not survive external Pareto evaluation.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR meaninit16w | -177,810 | -5.752% | 112 | -23.7 | -0.357% | 80/112, 81.862% span | -1.78 | -0.510% | 73/112, 72.652% span |
| fsrs6 | ADR Portfolio | 96,070 | +3.108% | 127 | 85.7 | +1.302% | 86/112, 83.234% span | 4.56 | +11.371% | 79/112, 74.096% span |
| fsrs6 | FSRS6 Cost ADR meaninit16w - ADR Portfolio | -273,880 | -8.859% | -15 | -109.3 | -1.659% | -6, -1.372 pp span | -6.33 | -11.880% | -6, -1.444 pp span |
| lstm | FSRS6 Cost ADR meaninit16w | -123,537 | -3.837% | 109 | -51.5 | -0.668% | 84/114, 84.573% span | -1.89 | -5.644% | 76/114, 76.859% span |
| lstm | ADR Portfolio | 52,881 | +1.642% | 121 | 50.2 | +0.764% | 81/114, 80.385% span | 2.16 | +5.359% | 85/114, 83.872% span |
| lstm | FSRS6 Cost ADR meaninit16w - ADR Portfolio | -176,419 | -5.479% | -12 | -101.7 | -1.432% | +3, +4.187 pp span | -4.06 | -11.002% | -9, -7.013 pp span |

## Per-User HV Delta

FSRS6 Cost ADR meaninit16w is behind ADR Portfolio for every user in every formal environment.

| user | fsrs6 FSRS6 Cost ADR meaninit16w HV delta | fsrs6 ADR Portfolio HV delta | fsrs6 delta | lstm FSRS6 Cost ADR meaninit16w HV delta | lstm ADR Portfolio HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3,916 | 12,081 | -8,166 | 4,177 | 9,639 | -5,462 |
| 2 | -108,204 | 29,679 | -137,883 | -38,928 | 15,362 | -54,290 |
| 3 | -6,302 | 3,835 | -10,137 | -5,303 | 3,144 | -8,447 |
| 4 | -39,628 | 34,918 | -74,546 | -13,020 | 16,379 | -29,399 |
| 5 | 3,101 | 7,564 | -4,463 | -4,997 | 1,933 | -6,930 |
| 6 | -30,016 | 4,745 | -34,761 | -59,535 | 4,578 | -64,114 |
| 7 | -1,846 | 1,222 | -3,068 | -3,442 | 640 | -4,082 |
| 8 | 1,169 | 2,026 | -857 | -2,488 | 1,206 | -3,694 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR meaninit16w | 6,249.0 | 81.35 | 26.15 | 338.02 |
| fsrs6 | ADR Portfolio | 6,552.9 | 55.29 | 26.16 | 228.70 |
| lstm | FSRS6 Cost ADR meaninit16w | 6,033.4 | 104.22 | 21.86 | 462.99 |
| lstm | ADR Portfolio | 6,422.7 | 67.86 | 23.46 | 297.02 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| FSRS6 Cost ADR meaninit16w | 1 | 11,050 |
| FSRS6 Cost ADR meaninit16w | 2 | 14,393 |
| FSRS6 Cost ADR meaninit16w | 3 | 2,537 |
| FSRS6 Cost ADR meaninit16w | 4 | 12,179 |
| FSRS6 Cost ADR meaninit16w | 5 | 6,033 |
| FSRS6 Cost ADR meaninit16w | 6 | 1,932 |
| FSRS6 Cost ADR meaninit16w | 7 | 1,925 |
| FSRS6 Cost ADR meaninit16w | 8 | 2,452 |
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
| FSRS6 Cost ADR meaninit16w | 8 | 52,501 |
| ADR Portfolio | 8 | 109,319 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.md`
