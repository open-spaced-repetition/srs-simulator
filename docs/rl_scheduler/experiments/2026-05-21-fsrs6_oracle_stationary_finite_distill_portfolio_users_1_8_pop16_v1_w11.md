# Oracle stationary finite distill vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/report/report_summary.json`

## Question

Evaluate whether per-user FSRS6 oracle stationary finite distill policies with searched goal-cost weights can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11` | `fsrs6_oracle_stationary_finite_distill` | `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Oracle stationary finite distill: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
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
| Oracle stationary finite distill | sweep | cuda | 97.4 | 149.9 | - |
| Oracle stationary finite distill | train-overfit | cuda | 1,017.1 | 14.4 | 229.7 |
| ADR | sweep | cuda | 44.8 | 325.9 | - |
| ADR | train-overfit | cuda | 302.2 | 48.3 | 773.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | 026a5017d031b39dac1c11e6b3af67bcae493e3f | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | 88de85231acfebb4e7826cfa726d26f9f6322e95 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/sweep/gpu_monitor/summary.json` | 279.0 | 297.3 | False | 3,400.0 |
| Oracle stationary finite distill | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/train-overfit/gpu_monitor/summary.json` | 219.2 | 237.4 | False | 7,141.0 |
| ADR | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 216.6 | 234.8 | False | 3,478.0 |
| ADR | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 214.0 | 232.3 | False | 2,258.0 |

## Conclusion

Promotion decision for `fsrs6_oracle_stationary_finite_distill` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 9,591 HV, 28.1 same-budget memory lift AUC, 0.18 same-target time saved AUC versus comparison.
- lstm: -10,661 HV, 8.1 same-budget memory lift AUC, -1.74 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 105,660 | +3.418% | 128 | 113.7 | +1.741% | 95/112, 80.364% span | 4.74 | +13.588% | 86/112, 77.913% span |
| fsrs6 | ADR | 96,070 | +3.108% | 127 | 85.7 | +1.302% | 86/112, 83.234% span | 4.56 | +11.371% | 79/112, 74.096% span |
| fsrs6 | Oracle stationary finite distill - ADR | 9,591 | +0.310% | 1 | 28.1 | +0.440% | +9, -2.870 pp span | 0.18 | +2.218% | +7, +3.817 pp span |
| lstm | Oracle stationary finite distill | 42,220 | +1.311% | 127 | 58.3 | +0.890% | 92/114, 81.418% span | 0.42 | +5.111% | 92/114, 84.195% span |
| lstm | ADR | 52,881 | +1.642% | 121 | 50.2 | +0.764% | 81/114, 80.385% span | 2.16 | +5.359% | 85/114, 83.872% span |
| lstm | Oracle stationary finite distill - ADR | -10,661 | -0.331% | 6 | 8.1 | +0.126% | +11, +1.033 pp span | -1.74 | -0.248% | +7, +0.324 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 6/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,329 | 12,081 | 2,247 | 10,130 | 9,639 | 491 |
| 2 | 28,416 | 29,679 | -1,264 | 6,928 | 15,362 | -8,434 |
| 3 | 3,296 | 3,835 | -539 | 1,935 | 3,144 | -1,209 |
| 4 | 40,198 | 34,918 | 5,279 | 11,610 | 16,379 | -4,769 |
| 5 | 8,622 | 7,564 | 1,058 | 2,509 | 1,933 | 576 |
| 6 | 6,132 | 4,745 | 1,387 | 6,682 | 4,578 | 2,104 |
| 7 | 2,543 | 1,222 | 1,321 | 1,324 | 640 | 684 |
| 8 | 2,126 | 2,026 | 100 | 1,102 | 1,206 | -104 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 6,594.8 | 53.82 | 25.43 | 221.35 |
| fsrs6 | ADR | 6,552.9 | 55.29 | 26.16 | 228.70 |
| lstm | Oracle stationary finite distill | 6,454.4 | 65.64 | 22.61 | 279.14 |
| lstm | ADR | 6,422.7 | 67.86 | 23.46 | 297.02 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill | 1 | 17,735 |
| Oracle stationary finite distill | 2 | 33,324 |
| Oracle stationary finite distill | 3 | 4,961 |
| Oracle stationary finite distill | 4 | 41,677 |
| Oracle stationary finite distill | 5 | 9,548 |
| Oracle stationary finite distill | 6 | 6,553 |
| Oracle stationary finite distill | 7 | 2,718 |
| Oracle stationary finite distill | 8 | 2,287 |
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
| Oracle stationary finite distill | 8 | 118,802 |
| ADR | 8 | 109,319 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off_w11/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-21-fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_w11.md`
