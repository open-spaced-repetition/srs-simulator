# Cost ADR retention-head interval-init wide schedHV stdpre vs Cost ADR interval-head schedHV stdpre experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether interval-implied-retention initialization plus wider retention bounds improves the Cost-ADR desired-retention action head on the first 8 users with matched 16 cost weights, pop16/gen20 budget, scheduler-HV objective, and diagonal search preconditioning.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Cost ADR retention-head interval-init wide schedHV stdpre: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- Cost ADR interval-head schedHV stdpre: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| Cost ADR retention-head interval-init wide schedHV stdpre | analyze-pareto | yes | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | build-pareto | yes | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | preflight | yes | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | stage-baseline | yes | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | sweep | yes | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | train-overfit | yes | - |
| Cost ADR interval-head schedHV stdpre | analyze-pareto | yes | - |
| Cost ADR interval-head schedHV stdpre | build-pareto | yes | - |
| Cost ADR interval-head schedHV stdpre | preflight | yes | - |
| Cost ADR interval-head schedHV stdpre | stage-baseline | yes | - |
| Cost ADR interval-head schedHV stdpre | sweep | yes | - |
| Cost ADR interval-head schedHV stdpre | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| Cost ADR retention-head interval-init wide schedHV stdpre | sweep | cuda | 52.1 | 280.3 | - |
| Cost ADR retention-head interval-init wide schedHV stdpre | train-overfit | cuda | 777.4 | 18.8 | 300.5 |
| Cost ADR interval-head schedHV stdpre | sweep | cuda | 50.9 | 286.7 | - |
| Cost ADR interval-head schedHV stdpre | train-overfit | cuda | 764.4 | 19.1 | 305.6 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Cost ADR retention-head interval-init wide schedHV stdpre | 01bac3badf019070099576717f0448ff470c722e | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| Cost ADR interval-head schedHV stdpre | 6211afb0d88c48e24ff273271fded4ad939ed887 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Cost ADR retention-head interval-init wide schedHV stdpre | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 209.5 | 227.7 | False | 5,808.0 |
| Cost ADR retention-head interval-init wide schedHV stdpre | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 297.8 | 316.0 | False | 5,855.0 |
| Cost ADR interval-head schedHV stdpre | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 171.3 | 189.5 | False | 5,390.0 |
| Cost ADR interval-head schedHV stdpre | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 376.4 | 394.6 | False | 5,380.0 |

## Conclusion

Promotion decision for Cost ADR retention-head interval-init wide schedHV stdpre is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 1,444 HV, 8.5 same-budget memory lift AUC, -0.13 same-target time saved AUC versus comparison.
- lstm: 6,694 HV, 21.8 same-budget memory lift AUC, 0.60 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Cost ADR retention-head interval-init wide schedHV stdpre | 99,155 | +3.207% | 123 | 99.9 | +1.511% | 93/112, 87.612% span | 4.55 | +13.217% | 85/112, 84.725% span |
| fsrs6 | Cost ADR interval-head schedHV stdpre | 97,711 | +3.161% | 122 | 91.4 | +1.394% | 90/112, 87.137% span | 4.68 | +12.793% | 86/112, 82.331% span |
| fsrs6 | Cost ADR retention-head interval-init wide schedHV stdpre - Cost ADR interval-head schedHV stdpre | 1,444 | +0.047% | 1 | 8.5 | +0.118% | +3, +0.475 pp span | -0.13 | +0.424% | -1, +2.394 pp span |
| lstm | Cost ADR retention-head interval-init wide schedHV stdpre | 46,745 | +1.452% | 121 | 74.8 | +1.190% | 90/114, 86.520% span | 1.45 | +6.052% | 94/114, 94.002% span |
| lstm | Cost ADR interval-head schedHV stdpre | 40,052 | +1.244% | 118 | 52.9 | +0.814% | 89/114, 86.716% span | 0.85 | +4.590% | 91/114, 91.662% span |
| lstm | Cost ADR retention-head interval-init wide schedHV stdpre - Cost ADR interval-head schedHV stdpre | 6,694 | +0.208% | 3 | 21.8 | +0.376% | +1, -0.196 pp span | 0.60 | +1.462% | +3, +2.340 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 4/16 environment-user rows.

| user | fsrs6 Cost ADR retention-head interval-init wide schedHV stdpre HV delta | fsrs6 Cost ADR interval-head schedHV stdpre HV delta | fsrs6 delta | lstm Cost ADR retention-head interval-init wide schedHV stdpre HV delta | lstm Cost ADR interval-head schedHV stdpre HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15,269 | 14,820 | 449 | 14,538 | 14,509 | 29 |
| 2 | 30,613 | 29,774 | 839 | 10,057 | 4,740 | 5,316 |
| 3 | 3,759 | 3,571 | 188 | 2,949 | 2,843 | 106 |
| 4 | 31,301 | 31,571 | -270 | 9,573 | 10,231 | -657 |
| 5 | 9,317 | 9,243 | 74 | 2,648 | 1,292 | 1,356 |
| 6 | 5,101 | 4,955 | 145 | 5,300 | 4,533 | 767 |
| 7 | 1,303 | 1,186 | 117 | 784 | 650 | 133 |
| 8 | 2,491 | 2,591 | -100 | 897 | 1,253 | -356 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Cost ADR retention-head interval-init wide schedHV stdpre | 6,375.8 | 60.46 | 27.37 | 237.15 |
| fsrs6 | Cost ADR interval-head schedHV stdpre | 6,398.3 | 58.47 | 27.25 | 229.47 |
| lstm | Cost ADR retention-head interval-init wide schedHV stdpre | 6,228.0 | 76.38 | 23.81 | 314.80 |
| lstm | Cost ADR interval-head schedHV stdpre | 6,260.7 | 73.20 | 24.15 | 300.83 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Cost ADR retention-head interval-init wide schedHV stdpre | 1 | 15,421 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 2 | 30,445 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 3 | 3,892 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 4 | 32,044 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 5 | 9,419 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 6 | 5,126 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 7 | 1,431 |
| Cost ADR retention-head interval-init wide schedHV stdpre | 8 | 2,487 |
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
| Cost ADR retention-head interval-init wide schedHV stdpre | 8 | 100,265 |
| Cost ADR interval-head schedHV stdpre | 8 | 97,905 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.md`
