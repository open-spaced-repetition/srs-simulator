# Cost ADR schedHV stdpre vs Cost ADR interval-head schedHV stdpre experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether the Cost-ADR policy should return to a desired-retention action head instead of directly emitting intervals on the first 8 users with matched 16 cost weights, pop16/gen20 budget, scheduler-HV objective, and diagonal search preconditioning.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Cost ADR schedHV stdpre: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- Cost ADR interval-head schedHV stdpre: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| Cost ADR schedHV stdpre | analyze-pareto | yes | - |
| Cost ADR schedHV stdpre | build-pareto | yes | - |
| Cost ADR schedHV stdpre | preflight | yes | - |
| Cost ADR schedHV stdpre | stage-baseline | yes | - |
| Cost ADR schedHV stdpre | sweep | yes | - |
| Cost ADR schedHV stdpre | train-overfit | yes | - |
| Cost ADR interval-head schedHV stdpre | analyze-pareto | yes | - |
| Cost ADR interval-head schedHV stdpre | build-pareto | yes | - |
| Cost ADR interval-head schedHV stdpre | preflight | yes | - |
| Cost ADR interval-head schedHV stdpre | stage-baseline | yes | - |
| Cost ADR interval-head schedHV stdpre | sweep | yes | - |
| Cost ADR interval-head schedHV stdpre | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| Cost ADR schedHV stdpre | sweep | cuda | 46.6 | 313.3 | - |
| Cost ADR schedHV stdpre | train-overfit | cuda | 770.9 | 18.9 | 303.0 |
| Cost ADR interval-head schedHV stdpre | sweep | cuda | 50.9 | 286.7 | - |
| Cost ADR interval-head schedHV stdpre | train-overfit | cuda | 764.4 | 19.1 | 305.6 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Cost ADR schedHV stdpre | b7b3f77777dd4ad9df64d2fc6d7ca160a018e82b | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| Cost ADR interval-head schedHV stdpre | 6211afb0d88c48e24ff273271fded4ad939ed887 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Cost ADR schedHV stdpre | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 202.8 | 221.0 | False | 6,565.0 |
| Cost ADR schedHV stdpre | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 407.2 | 425.5 | False | 6,555.0 |
| Cost ADR interval-head schedHV stdpre | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 171.3 | 189.5 | False | 5,390.0 |
| Cost ADR interval-head schedHV stdpre | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 376.4 | 394.6 | False | 5,380.0 |

## Conclusion

Do not promote Cost ADR schedHV stdpre.

With the matched training budget, Cost ADR schedHV stdpre remains behind Cost ADR interval-head schedHV stdpre on HV; same-budget memory lift and same-target time saved deltas are:

- fsrs6: -25,161 HV, -1.2 same-budget memory lift AUC, -1.06 same-target time saved AUC versus comparison.
- lstm: -31,145 HV, -5.2 same-budget memory lift AUC, -3.01 same-target time saved AUC versus comparison.

Training HV gains and lower-time sampled policy points do not survive external Pareto evaluation.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Cost ADR schedHV stdpre | 72,551 | +2.347% | 118 | 90.2 | +1.376% | 88/112, 79.410% span | 3.62 | +10.741% | 85/112, 79.761% span |
| fsrs6 | Cost ADR interval-head schedHV stdpre | 97,711 | +3.161% | 122 | 91.4 | +1.394% | 90/112, 87.137% span | 4.68 | +12.793% | 86/112, 82.331% span |
| fsrs6 | Cost ADR schedHV stdpre - Cost ADR interval-head schedHV stdpre | -25,161 | -0.814% | -4 | -1.2 | -0.018% | -2, -7.727 pp span | -1.06 | -2.051% | -1, -2.570 pp span |
| lstm | Cost ADR schedHV stdpre | 8,907 | +0.277% | 120 | 47.7 | +0.730% | 87/114, 81.902% span | -2.17 | +1.918% | 91/114, 89.814% span |
| lstm | Cost ADR interval-head schedHV stdpre | 40,052 | +1.244% | 118 | 52.9 | +0.814% | 89/114, 86.716% span | 0.85 | +4.590% | 91/114, 91.662% span |
| lstm | Cost ADR schedHV stdpre - Cost ADR interval-head schedHV stdpre | -31,145 | -0.967% | 2 | -5.2 | -0.084% | -2, -4.814 pp span | -3.01 | -2.672% | +0, -1.848 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 12/16 environment-user rows.

| user | fsrs6 Cost ADR schedHV stdpre HV delta | fsrs6 Cost ADR interval-head schedHV stdpre HV delta | fsrs6 delta | lstm Cost ADR schedHV stdpre HV delta | lstm Cost ADR interval-head schedHV stdpre HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9,827 | 14,820 | -4,992 | 11,911 | 14,509 | -2,597 |
| 2 | 13,888 | 29,774 | -15,886 | -21,733 | 4,740 | -26,474 |
| 3 | 3,230 | 3,571 | -341 | 2,510 | 2,843 | -333 |
| 4 | 30,937 | 31,571 | -634 | 12,159 | 10,231 | 1,928 |
| 5 | 7,479 | 9,243 | -1,763 | 1,481 | 1,292 | 189 |
| 6 | 3,420 | 4,955 | -1,535 | 434 | 4,533 | -4,099 |
| 7 | 1,463 | 1,186 | 277 | 1,047 | 650 | 396 |
| 8 | 2,305 | 2,591 | -286 | 1,098 | 1,253 | -156 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Cost ADR schedHV stdpre | 6,352.2 | 53.14 | 27.25 | 193.08 |
| fsrs6 | Cost ADR interval-head schedHV stdpre | 6,398.3 | 58.47 | 27.25 | 229.47 |
| lstm | Cost ADR schedHV stdpre | 6,241.0 | 65.03 | 24.44 | 251.64 |
| lstm | Cost ADR interval-head schedHV stdpre | 6,260.7 | 73.20 | 24.15 | 300.83 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Cost ADR schedHV stdpre | 1 | 9,857 |
| Cost ADR schedHV stdpre | 2 | 14,022 |
| Cost ADR schedHV stdpre | 3 | 3,242 |
| Cost ADR schedHV stdpre | 4 | 31,194 |
| Cost ADR schedHV stdpre | 5 | 7,160 |
| Cost ADR schedHV stdpre | 6 | 3,382 |
| Cost ADR schedHV stdpre | 7 | 1,478 |
| Cost ADR schedHV stdpre | 8 | 2,338 |
| Cost ADR interval-head schedHV stdpre | 1 | 15,101 |
| Cost ADR interval-head schedHV stdpre | 2 | 29,853 |
| Cost ADR interval-head schedHV stdpre | 3 | 3,502 |
| Cost ADR interval-head schedHV stdpre | 4 | 31,194 |
| Cost ADR interval-head schedHV stdpre | 5 | 9,436 |
| Cost ADR interval-head schedHV stdpre | 6 | 5,019 |
| Cost ADR interval-head schedHV stdpre | 7 | 1,182 |
| Cost ADR interval-head schedHV stdpre | 8 | 2,619 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| Cost ADR schedHV stdpre | 8 | 72,673 |
| Cost ADR interval-head schedHV stdpre | 8 | 97,905 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.md`
