# Oracle stationary finite distill vs ADR experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/report/report_summary.json`

## Question

Evaluate whether per-user FSRS6 oracle stationary finite distill policies with searched goal-cost weights can match or improve the matched-budget FSRS6 ADR portfolio on external Pareto metrics.

## Runs

| run | scheduler | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1` | `fsrs6_oracle_stationary_finite_distill` | `experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_8_16dr_pop16_gen5.json` |

Analysis summaries:

- Oracle stationary finite distill: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

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
| Oracle stationary finite distill | sweep | cuda | 100.9 | 144.6 | - |
| Oracle stationary finite distill | train-overfit | cuda | 1,019.5 | 14.3 | 229.1 |
| ADR | sweep | cuda | 45.1 | 323.5 | - |
| ADR | train-overfit | cuda | 328.8 | 44.4 | 710.4 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | a24826c6a07d400a45f0c9fc9715df5612993b78 | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR | c1a791b35e56ef777005d8329a11ac42e34ab707 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Oracle stationary finite distill | sweep | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/sweep/gpu_monitor/summary.json` | 186.3 | 204.5 | False | 2,773.0 |
| Oracle stationary finite distill | train-overfit | `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/train-overfit/gpu_monitor/summary.json` | 347.2 | 365.4 | False | 6,466.0 |
| ADR | - | - | - | - | - | - |

## Conclusion

Promotion decision for `fsrs6_oracle_stationary_finite_distill` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 7,737 HV, 29.4 budget-memory gain AUC, -0.05 memory-target regret AUC versus comparison.
- lstm: -18,265 HV, 7.7 budget-memory gain AUC, 1.06 memory-target regret AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## Interpretation

The train-overfit result confirms that the oracle stationary finite distill workflow can find competitive per-user policies under the matched search budget: final training HV gain is 117,610 versus ADR's 107,746. That is not sufficient evidence for promotion, because the external Pareto evaluation remains split by environment.

On the native FSRS6 environment, the candidate is stronger than ADR on aggregate HV (+7,737, +0.278 percentage points of baseline HV), budget-memory gain AUC (+29.4), memory-target time regret AUC (-0.05), and relative regret AUC (-14.172% versus -11.555%). With the user-simple relative regret aggregation, Oracle is also better for all 8 FSRS6 users.

On the LSTM environment, the relative regret result is much closer after switching to user-simple averaging: Oracle is -5.817% versus ADR's -5.906%, so ADR is better by only 0.090 percentage points. However, LSTM aggregate HV still favors ADR by 18,265, and absolute memory-target regret AUC favors ADR by 1.06. The policy-point diagnostics show the candidate uses less time and fewer reviews on average, but the LSTM Pareto frontier still does not consistently convert that lower workload into better robust memory outcomes.

The per-user breakdown shows the risk is not confined to one small corner case. The candidate is negative in 8/16 environment-user HV rows, including large LSTM losses for users 2 and 4 and an FSRS6 loss for user 2. Users 6 and 7 are consistent relative-regret wins across both environments, so the method may be useful for some user regimes, but the current portfolio is not a reliable ADR replacement.

GPU monitor artifacts do not indicate VRAM spill for the candidate run: train-overfit peaked at 347.2 MiB shared memory on one adapter and sweep peaked at 186.3 MiB, both with `shared_memory_spill_detected=false`. The quality result therefore should be read as an algorithmic robustness issue rather than an obvious GPU spill artifact. The candidate is also slower than ADR in this run: train-overfit took 1,019.5s versus 328.8s, and sweep took 100.9s versus 45.1s.

## Recommendation

Do not promote `fsrs6_oracle_stationary_finite_distill` as a replacement for `fsrs6_adr` based on this 8-user validation. Keep it as an experimental candidate and focus the next iteration on the LSTM transfer failures, especially users 2 and 4, before expanding to a larger user set. A useful follow-up is to compare the selected goal-cost weights and resulting review-load positions for the losing users, then test whether multi-environment selection, robustness regularization, or a constrained workload envelope reduces the LSTM HV/regret regression without giving up the FSRS6 gains.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and budget-memory gain are better. Negative memory-target regret is better.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | budget-memory gain AUC | budget-memory gain / baseline | budget coverage | memory-target regret AUC | memory-target regret / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 104,617 | +3.760% | 128 | 120.1 | +1.788% | 96/115, 70.232% span | -4.62 | -14.172% | 97/115, 88.859% span |
| fsrs6 | ADR | 96,880 | +3.482% | 128 | 90.7 | +1.349% | 88/115, 81.550% span | -4.57 | -11.555% | 81/115, 78.146% span |
| fsrs6 | Oracle stationary finite distill - ADR | 7,737 | +0.278% | 0 | 29.4 | - | +8, -11.319 pp span | -0.05 | - | +16, +10.713 pp span |
| lstm | Oracle stationary finite distill | 37,584 | +1.323% | 128 | 69.9 | +1.051% | 88/111, 66.183% span | -0.76 | -5.817% | 91/111, 89.146% span |
| lstm | ADR | 55,849 | +1.966% | 125 | 62.2 | +0.935% | 79/111, 82.466% span | -1.82 | -5.906% | 81/111, 83.775% span |
| lstm | Oracle stationary finite distill - ADR | -18,265 | -0.643% | 3 | 7.7 | - | +9, -16.283 pp span | 1.06 | - | +10, +5.371 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 8/16 environment-user rows.

| user | fsrs6 Oracle stationary finite distill HV delta | fsrs6 ADR HV delta | fsrs6 delta | lstm Oracle stationary finite distill HV delta | lstm ADR HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 11,968 | 9,894 | 2,073 | 5,210 | 7,161 | -1,951 |
| 2 | 24,008 | 27,715 | -3,707 | 3,761 | 13,315 | -9,554 |
| 3 | 4,201 | 3,745 | 456 | 3,207 | 3,364 | -157 |
| 4 | 42,269 | 37,079 | 5,191 | 11,319 | 19,119 | -7,800 |
| 5 | 10,319 | 8,964 | 1,355 | 3,243 | 4,110 | -866 |
| 6 | 7,866 | 6,327 | 1,539 | 7,772 | 6,491 | 1,281 |
| 7 | 2,198 | 1,238 | 959 | 1,569 | 733 | 836 |
| 8 | 1,788 | 1,918 | -130 | 1,503 | 1,556 | -53 |

## Per-User Relative Time Regret AUC

Relative time regret AUC is `time_regret_auc / baseline_time_auc * 100`, computed per user against the FSRS6 baseline frontier over the common covered memory-target interval. Negative values are better: the scheduler reaches the same memorized-card target faster than the FSRS6 baseline. The delta column is Oracle stationary finite distill minus ADR, so negative means Oracle is better than ADR.

| environment | user | Oracle stationary finite distill relative regret AUC | ADR relative regret AUC | Oracle - ADR |
| --- | --- | ---: | ---: | ---: |
| fsrs6 | 1 | -8.07% | -7.00% | -1.07 pp |
| fsrs6 | 2 | -17.87% | -16.26% | -1.61 pp |
| fsrs6 | 3 | -10.93% | -10.22% | -0.71 pp |
| fsrs6 | 4 | -19.47% | -16.86% | -2.61 pp |
| fsrs6 | 5 | -15.42% | -14.07% | -1.36 pp |
| fsrs6 | 6 | -15.85% | -14.83% | -1.02 pp |
| fsrs6 | 7 | -20.97% | -9.17% | -11.81 pp |
| fsrs6 | 8 | -4.79% | -4.02% | -0.77 pp |
| lstm | 1 | -0.21% | -2.16% | +1.95 pp |
| lstm | 2 | +2.76% | -0.85% | +3.61 pp |
| lstm | 3 | -8.78% | -9.19% | +0.41 pp |
| lstm | 4 | -1.96% | -6.42% | +4.46 pp |
| lstm | 5 | -3.22% | -5.10% | +1.88 pp |
| lstm | 6 | -11.84% | -10.28% | -1.56 pp |
| lstm | 7 | -19.34% | -7.57% | -11.77 pp |
| lstm | 8 | -3.94% | -5.68% | +1.74 pp |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | Oracle stationary finite distill | 6,552.2 | 44.84 | 25.93 | 171.36 |
| fsrs6 | ADR | 6,553.0 | 50.42 | 25.42 | 195.90 |
| lstm | Oracle stationary finite distill | 6,422.9 | 53.11 | 23.36 | 212.64 |
| lstm | ADR | 6,431.5 | 60.86 | 22.88 | 255.29 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| Oracle stationary finite distill | 1 | 14,451 |
| Oracle stationary finite distill | 2 | 29,493 |
| Oracle stationary finite distill | 3 | 5,272 |
| Oracle stationary finite distill | 4 | 44,538 |
| Oracle stationary finite distill | 5 | 10,856 |
| Oracle stationary finite distill | 6 | 8,589 |
| Oracle stationary finite distill | 7 | 2,398 |
| Oracle stationary finite distill | 8 | 2,012 |
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
| Oracle stationary finite distill | 8 | 117,610 |
| ADR | 8 | 107,746 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-18-fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.md`
