# Anki SM2 AP vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/report/report_summary.json`

## Question

Evaluate whether a no-DR Anki SM2 adaptive-parameter portfolio can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `anki_sm2_ap_portfolio_users_1_8_pop16_20_v1` | `anki_sm2_ap` | `experiments/rl_scheduler/configs/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Anki SM2 AP: `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| Anki SM2 AP | analyze-pareto | yes | - |
| Anki SM2 AP | build-pareto | yes | - |
| Anki SM2 AP | preflight | yes | - |
| Anki SM2 AP | stage-baseline | yes | - |
| Anki SM2 AP | sweep | yes | - |
| Anki SM2 AP | train-overfit | yes | - |
| ADR | analyze-pareto | yes | - |
| ADR | build-pareto | yes | - |
| ADR | preflight | yes | - |
| ADR | stage-baseline | yes | - |
| ADR | sweep | yes | - |
| ADR | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| Anki SM2 AP | sweep | cuda | 45.6 | 320.1 | - |
| Anki SM2 AP | train-overfit | cuda | 330.8 | 44.1 | 706.3 |
| ADR | sweep | cuda | 45.1 | 323.5 | - |
| ADR | train-overfit | cuda | 328.8 | 44.4 | 710.4 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Anki SM2 AP | 275052e0d8f7eb3f9bd9f97b31ff128bf794de21 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | c1a791b35e56ef777005d8329a11ac42e34ab707 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

No automatic GPU monitor artifacts are present for these historical runs. The table below reports machine-readable Torch CUDA peak memory from `performance_summary.json`; shared-memory spill cannot be judged from these artifacts.

| run | stage | device | peak allocated MiB | peak reserved MiB | shared-memory spill |
| --- | --- | --- | --- | --- | --- |
| Anki SM2 AP | sweep | cuda | 1,045.8 | 1,080.0 | unavailable |
| Anki SM2 AP | train-overfit | cuda | 166.7 | 196.0 | unavailable |
| ADR | sweep | cuda | 1,274.1 | 1,302.0 | unavailable |
| ADR | train-overfit | cuda | 168.8 | 192.0 | unavailable |

## Conclusion

Do not promote `anki_sm2_ap`.

With the matched portfolio budget, Anki SM2 AP remains behind ADR on HV; same-budget memory lift and same-target time saved deltas are:

- fsrs6: -354,851 HV, -73.9 same-budget memory lift AUC, -4.09 same-target time saved AUC versus comparison.
- lstm: -400,193 HV, 8.5 same-budget memory lift AUC, -4.21 same-target time saved AUC versus comparison.

Training HV gains and lower-time sampled policy points do not survive external Pareto evaluation.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Anki SM2 AP | -257,971 | -9.272% | 121 | 16.8 | +0.247% | 34/115, 31.091% span | 0.48 | +1.266% | 32/115, 25.999% span |
| fsrs6 | ADR | 96,880 | +3.482% | 128 | 90.7 | +1.400% | 88/115, 81.550% span | 4.57 | +11.555% | 81/115, 78.146% span |
| fsrs6 | Anki SM2 AP - ADR | -354,851 | -12.754% | -7 | -73.9 | -1.153% | -54, -50.459 pp span | -4.09 | -10.289% | -49, -52.147 pp span |
| lstm | Anki SM2 AP | -344,345 | -12.120% | 107 | 70.6 | +1.071% | 35/111, 27.320% span | -2.39 | -6.331% | 34/111, 27.597% span |
| lstm | ADR | 55,849 | +1.966% | 125 | 62.2 | +0.953% | 79/111, 82.466% span | 1.82 | +5.906% | 81/111, 83.775% span |
| lstm | Anki SM2 AP - ADR | -400,193 | -14.085% | -18 | 8.5 | +0.118% | -44, -55.146 pp span | -4.21 | -12.237% | -47, -56.179 pp span |

## Per-User HV Delta

Anki SM2 AP is behind ADR for every user in every formal environment.

| user | fsrs6 Anki SM2 AP HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Anki SM2 AP HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | -59,270 | 9,894 | -69,164 | -60,225 | 7,161 | -67,385 |
| 2 | -102,383 | 27,715 | -130,098 | -152,432 | 13,315 | -165,747 |
| 3 | -7,074 | 3,745 | -10,820 | -8,201 | 3,364 | -11,565 |
| 4 | -72,924 | 37,079 | -110,002 | -88,553 | 19,119 | -107,671 |
| 5 | 216 | 8,964 | -8,747 | -5,331 | 4,110 | -9,441 |
| 6 | -8,504 | 6,327 | -14,831 | -19,056 | 6,491 | -25,547 |
| 7 | -6,827 | 1,238 | -8,065 | -6,997 | 733 | -7,731 |
| 8 | -1,206 | 1,918 | -3,124 | -3,550 | 1,556 | -5,105 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Anki SM2 AP | 6,709.1 | 41.90 | 21.77 | 137.54 |
| fsrs6 | ADR | 6,553.0 | 50.42 | 25.42 | 195.90 |
| lstm | Anki SM2 AP | 6,563.5 | 45.66 | 17.99 | 155.27 |
| lstm | ADR | 6,431.5 | 60.86 | 22.88 | 255.29 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Anki SM2 AP | 1 | 4,368 |
| Anki SM2 AP | 2 | 5,974 |
| Anki SM2 AP | 3 | 2,780 |
| Anki SM2 AP | 4 | 7,921 |
| Anki SM2 AP | 5 | 4,715 |
| Anki SM2 AP | 6 | 1,923 |
| Anki SM2 AP | 7 | 83 |
| Anki SM2 AP | 8 | 1,136 |
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
| Anki SM2 AP | 8 | 28,900 |
| ADR | 8 | 107,746 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-13-anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.md`
