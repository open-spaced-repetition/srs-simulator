# FSRS6 Cost-ADR Retention Max 0.98 Posthoc Report

## Question

Evaluate whether limiting the default 15-parameter FSRS6 Cost-ADR
desired-retention policy from `retention_max = 0.995` to `retention_max = 0.98`
improves external LSTM validation without destroying the native FSRS6 Pareto
quality.

## Executive Answer

The `retention_max = 0.98` posthoc cap is not a promotion candidate.

On the first-8 user screening run, the cap improved LSTM relative same-target
time-saved AUC from `3.558%` to `4.501%`, but it also reduced LSTM scheduler-only
HV delta from `31,810` to `24,760`. That was enough to justify a full 128-user
diagnostic run, but not enough to treat the cap as a likely Pareto win.

On users 1-128, the same pattern held. The cap slightly improved reported LSTM
relative time-saved AUC from `2.568%` to `2.772%`, but it cut LSTM HV delta from
`599,474` to `396,312`. It also reduced native FSRS6 HV delta from `1,432,969`
to `1,299,762`. Against the matched FSRS6 ADR portfolio, the capped Cost-ADR
still wins native FSRS6 HV, but loses LSTM HV by `251,963` and loses LSTM
relative time-saved AUC by `2.124` percentage points.

The practical reading is that the high-retention tail was partly hurting the
reported LSTM time-save ratio, but posthoc capping it removes too much Pareto
area. Keep the current `0.995` cap for this policy family unless a future run
re-trains under `0.98` and recovers the HV loss.

## Mechanism

For the desired-retention head, `retention_max` is part of the policy output
mapping:

```text
retention = retention_min + (retention_max - retention_min) * sigmoid(logit)
```

The cap is therefore not a final display-time clamp. Lowering `retention_max`
from `0.995` to `0.98` rescales every output, not only outputs that would have
exceeded `0.98`. With `retention_min = 0.30`, the mapping changes from
`0.30 + 0.695 * sigmoid(logit)` to `0.30 + 0.680 * sigmoid(logit)`.

## Runs

| role | run | users | retention max | notes |
| --- | --- | ---: | ---: | --- |
| first-8 control | `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_from128_rmax0995_posthoc_v1_markov_off` | 1-8 | 0.995 | copied policies from the original 1-128 Cost-ADR run |
| first-8 cap | `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_from128_rmax098_posthoc_v1_markov_off` | 1-8 | 0.98 | copied the same policies and changed `policy.retention_max` |
| full cap | `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_rmax098_posthoc_v1_markov_off` | 1-128 | 0.98 | full FSRS6/LSTM sweep and Pareto analysis |
| original Cost-ADR | `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off` | 1-128 | 0.995 | original trained 15p Cost-ADR run |
| comparison | `fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off` | 1-128 | 0.98 | ordinary FSRS6 ADR portfolio, 16 child policies per user |

Analysis summary paths:

- `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_rmax_posthoc/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_from128_rmax0995_posthoc_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_rmax_posthoc/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_from128_rmax098_posthoc_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_rmax098_posthoc/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_rmax098_posthoc_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## First-8 Screening

The first-8 run was used as a gate before spending the full 128-user validation
budget. It uses the policies from the already trained 1-128 Cost-ADR run, so it
tests only the output-bound change.

| env | scheduler | retention max | HV delta sum | HV delta / baseline HV | relative memory-lift AUC | relative time-saved AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fsrs6 | Cost-ADR posthoc control | 0.995 | 100,301 | +3.244% | +1.477% | +13.078% |
| fsrs6 | Cost-ADR capped | 0.98 | 83,903 | +2.714% | +2.026% | +12.980% |
| fsrs6 | ADR portfolio | 0.98 | 96,070 | +3.108% | +1.302% | +11.371% |
| lstm | Cost-ADR posthoc control | 0.995 | 31,810 | +0.988% | +0.736% | +3.558% |
| lstm | Cost-ADR capped | 0.98 | 24,760 | +0.769% | +1.154% | +4.501% |
| lstm | ADR portfolio | 0.98 | 52,881 | +1.642% | +0.764% | +5.359% |

The first-8 evidence was mixed. The cap moved LSTM relative time-saved AUC in
the desired direction, but it worsened HV in both environments. It also did not
catch the ADR portfolio on LSTM HV or LSTM relative time-saved AUC.

## Full 1-128 Results

Scheduler-only hypervolume values are summed per-user HV delta against the same
staged FSRS6 baseline manifest. Positive same-budget memory-lift AUC means more
memorized cards at the same time budget. Positive same-target time-saved AUC
means less study time at the same memorized-card target. The relative AUC values
are user-simple averages from the analysis summaries.

| env | scheduler | retention max | HV delta sum | HV delta / baseline HV | frontier points | memory-lift AUC | relative memory lift | budget coverage | time-saved AUC | relative time saved | target coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| fsrs6 | Cost-ADR original | 0.995 | 1,432,969 | +4.168% | 1,940 | 121.7 | +1.898% | 1606/1801, 83.644% | 4.21 | +11.447% | 1518/1801, 90.746% |
| fsrs6 | Cost-ADR capped | 0.98 | 1,299,762 | +3.781% | 1,931 | 160.3 | +2.566% | 1483/1801, 63.874% | 3.99 | +11.735% | 1404/1801, 89.366% |
| fsrs6 | ADR portfolio | 0.98 | 1,224,940 | +3.563% | 2,031 | 108.2 | +1.662% | 1444/1801, 80.123% | 4.09 | +9.762% | 1390/1801, 83.013% |
| lstm | Cost-ADR original | 0.995 | 599,474 | +1.497% | 1,890 | 16.3 | +0.274% | 1539/1767, 79.558% | 0.91 | +2.568% | 1577/1767, 92.917% |
| lstm | Cost-ADR capped | 0.98 | 396,312 | +0.989% | 1,904 | 18.8 | +0.335% | 1432/1767, 56.891% | 0.74 | +2.772% | 1465/1767, 90.673% |
| lstm | ADR portfolio | 0.98 | 648,275 | +1.618% | 1,893 | 27.4 | +0.430% | 1363/1767, 73.461% | 2.20 | +4.895% | 1398/1767, 84.135% |

The LSTM time-save movement is only a relative-ratio improvement. Absolute
LSTM same-target time-saved AUC decreases from `0.91` to `0.74`, while the
reported relative value rises from `2.568%` to `2.772%` because the denominator
and covered target interval differ after the frontier shift.

## Deltas

Against the original `retention_max = 0.995` Cost-ADR run:

| env | HV delta change | HV ratio change | relative memory-lift change | relative time-saved change | budget span change | target span change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fsrs6 | -133,207 | -0.387 pp | +0.667 pp | +0.288 pp | -19.770 pp | -1.381 pp |
| lstm | -203,163 | -0.507 pp | +0.061 pp | +0.203 pp | -22.667 pp | -2.244 pp |

Against the ADR portfolio:

| env | HV delta change | HV ratio change | relative memory-lift change | relative time-saved change | budget span change | target span change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fsrs6 | +74,822 | +0.218 pp | +0.903 pp | +1.973 pp | -16.249 pp | +6.353 pp |
| lstm | -251,963 | -0.629 pp | -0.095 pp | -2.124 pp | -16.570 pp | +6.538 pp |

Per-user HV comparisons show that the HV loss is broad, not just a few outlier
users:

| comparison | env | capped wins | other wins | median capped-minus-other HV | delta sum |
| --- | --- | ---: | ---: | ---: | ---: |
| capped vs original Cost-ADR | fsrs6 | 34 | 94 | -250 | -133,207 |
| capped vs original Cost-ADR | lstm | 41 | 87 | -205 | -203,163 |
| capped vs ADR portfolio | fsrs6 | 80 | 48 | +321 | +74,822 |
| capped vs ADR portfolio | lstm | 30 | 98 | -809 | -251,963 |

## Stage And GPU Evidence

The full 1-128 posthoc run passed `stage-baseline`, synthetic `train-overfit`,
`sweep`, `build-pareto`, and `analyze-pareto`. The synthetic train summary
records that policies were copied from the original 1-128 Cost-ADR run and
`policy.retention_max` was set to `0.98`. There are 128 policy files in the
posthoc run, and all 128 report `retention_max = 0.98`.

The full sweep validated 4,096 logs in 301.3 seconds on CUDA at 775.4 user-days
per second. GPU monitor artifacts report no shared-memory spill:

| metric | value |
| --- | ---: |
| device | NVIDIA GeForce RTX 4090 D |
| elapsed seconds | 301.3 |
| validated logs | 4,096 |
| user-days/s | 775.4 |
| nvidia-smi peak memory | 5,486 MiB |
| shared memory peak | 356.8 MiB |
| shared-memory spill | false |

## Interpretation

The cap behaves like a frontier-shape tradeoff rather than a clean LSTM repair.
It moves some reported AUC ratios in the right direction, especially LSTM
relative time saved, but it also removes high-memory/high-retention frontier
area. That is why HV drops even when relative time saved improves.

The output-bound mechanism also matters. Because the policy uses a sigmoid range
mapping, lowering `retention_max` changes all desired-retention outputs. It is
not equivalent to clipping only rare actions above `0.98`.

## Conclusion

Do not replace the current `retention_max = 0.995` Cost-ADR policy with the
posthoc `0.98` cap. The cap is useful diagnostic evidence that the high-retention
tail contributes to the LSTM time-save weakness, but the posthoc fix gives up
too much HV and still fails to beat FSRS6 ADR on LSTM. A real follow-up would
need to retrain or regularize Cost-ADR under the capped range rather than only
rescaling already trained policies.
