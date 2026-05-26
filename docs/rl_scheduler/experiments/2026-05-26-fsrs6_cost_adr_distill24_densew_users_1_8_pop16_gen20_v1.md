# FSRS6 Cost ADR distill24 densew vs ADR Portfolio experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Does distill-initialized FSRS6 Cost-ADR match or exceed the ADR portfolio under the matched users 1..8 pop16/gen20 budget?

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=19 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- FSRS6 Cost ADR distill24 densew: `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR Portfolio: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| FSRS6 Cost ADR distill24 densew | analyze-pareto | yes | - |
| FSRS6 Cost ADR distill24 densew | build-pareto | yes | - |
| FSRS6 Cost ADR distill24 densew | preflight | yes | - |
| FSRS6 Cost ADR distill24 densew | stage-baseline | yes | - |
| FSRS6 Cost ADR distill24 densew | sweep | yes | - |
| FSRS6 Cost ADR distill24 densew | train-overfit | yes | - |
| ADR Portfolio | analyze-pareto | yes | - |
| ADR Portfolio | build-pareto | yes | - |
| ADR Portfolio | preflight | yes | - |
| ADR Portfolio | stage-baseline | yes | - |
| ADR Portfolio | sweep | yes | - |
| ADR Portfolio | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR distill24 densew | sweep | cuda | 60.4 | 241.7 | - |
| FSRS6 Cost ADR distill24 densew | train-overfit | cuda | 899.1 | 16.2 | 259.8 |
| ADR Portfolio | sweep | cuda | 44.8 | 325.9 | - |
| ADR Portfolio | train-overfit | cuda | 302.2 | 48.3 | 773.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR distill24 densew | 936380a0727fd8f73328176924074f7d4ddd0880 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR Portfolio | 88de85231acfebb4e7826cfa726d26f9f6322e95 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR distill24 densew | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 176.0 | 194.2 | False | 5,165.0 |
| FSRS6 Cost ADR distill24 densew | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 176.1 | 194.3 | False | 6,248.0 |
| ADR Portfolio | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 216.6 | 234.8 | False | 3,478.0 |
| ADR Portfolio | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 214.0 | 232.3 | False | 2,258.0 |

## Conclusion

Promote `fsrs6_cost_adr`.

With the matched training budget, FSRS6 Cost ADR distill24 densew exceeds ADR Portfolio on every reported external Pareto environment for HV, same-budget memory lift, and same-target time saved. Candidate-minus-comparison deltas are:

- fsrs6: 26,237 HV, 28.7 same-budget memory lift AUC, 0.97 same-target time saved AUC versus comparison.
- lstm: 22,901 HV, 67.6 same-budget memory lift AUC, 0.63 same-target time saved AUC versus comparison.

Coverage and sampled policy-point diagnostics remain secondary checks; the promotion decision is based on external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR distill24 densew | 122,307 | +3.956% | 277 | 114.3 | +1.768% | 78/112, 91.593% span | 5.52 | +14.151% | 73/112, 73.486% span |
| fsrs6 | ADR Portfolio | 96,070 | +3.108% | 127 | 85.7 | +1.302% | 86/112, 83.234% span | 4.56 | +11.371% | 79/112, 74.096% span |
| fsrs6 | FSRS6 Cost ADR distill24 densew - ADR Portfolio | 26,237 | +0.849% | 150 | 28.7 | +0.467% | -8, +8.359 pp span | 0.97 | +2.781% | -6, -0.611 pp span |
| lstm | FSRS6 Cost ADR distill24 densew | 75,782 | +2.354% | 271 | 117.7 | +2.102% | 77/114, 91.278% span | 2.79 | +6.652% | 84/114, 73.182% span |
| lstm | ADR Portfolio | 52,881 | +1.642% | 121 | 50.2 | +0.764% | 81/114, 80.385% span | 2.16 | +5.359% | 85/114, 83.872% span |
| lstm | FSRS6 Cost ADR distill24 densew - ADR Portfolio | 22,901 | +0.711% | 150 | 67.6 | +1.339% | -4, +10.893 pp span | 0.63 | +1.293% | -1, -10.690 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 4/16 environment-user rows.

| user | fsrs6 FSRS6 Cost ADR distill24 densew HV delta | fsrs6 ADR Portfolio HV delta | fsrs6 delta | lstm FSRS6 Cost ADR distill24 densew HV delta | lstm ADR Portfolio HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15,163 | 12,081 | 3,081 | 17,788 | 9,639 | 8,149 |
| 2 | 38,818 | 29,679 | 9,139 | 20,109 | 15,362 | 4,747 |
| 3 | 4,672 | 3,835 | 837 | 3,795 | 3,144 | 651 |
| 4 | 46,407 | 34,918 | 11,489 | 28,900 | 16,379 | 12,521 |
| 5 | 10,763 | 7,564 | 3,200 | 3,158 | 1,933 | 1,225 |
| 6 | 6,820 | 4,745 | 2,075 | 7,197 | 4,578 | 2,619 |
| 7 | 257 | 1,222 | -965 | 190 | 640 | -450 |
| 8 | -592 | 2,026 | -2,618 | -5,355 | 1,206 | -6,561 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR distill24 densew | 6,247.0 | 46.45 | 29.46 | 180.26 |
| fsrs6 | ADR Portfolio | 6,552.9 | 55.29 | 26.16 | 228.70 |
| lstm | FSRS6 Cost ADR distill24 densew | 6,108.3 | 56.23 | 26.41 | 229.85 |
| lstm | ADR Portfolio | 6,422.7 | 67.86 | 23.46 | 297.02 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| FSRS6 Cost ADR distill24 densew | 1 | 18,262 |
| FSRS6 Cost ADR distill24 densew | 2 | 32,881 |
| FSRS6 Cost ADR distill24 densew | 3 | 4,781 |
| FSRS6 Cost ADR distill24 densew | 4 | 39,725 |
| FSRS6 Cost ADR distill24 densew | 5 | 9,758 |
| FSRS6 Cost ADR distill24 densew | 6 | 5,690 |
| FSRS6 Cost ADR distill24 densew | 7 | 1,750 |
| FSRS6 Cost ADR distill24 densew | 8 | 2,708 |
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
| FSRS6 Cost ADR distill24 densew | 8 | 115,555 |
| ADR Portfolio | 8 | 109,319 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-26-fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.md`
