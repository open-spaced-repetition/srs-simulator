# FSRS6 Cost-ADR 15p Retention Head vs FSRS6 ADR Portfolio 1-128 report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether the default 15-parameter FSRS6 Cost-ADR desired-retention policy beats the matched-budget ordinary FSRS6 ADR portfolio on users 1-128 with the same staged FSRS6 baseline manifest and matched 16-point comparison setup.

## Executive Answer

On the native FSRS6 evaluation, the 15-parameter Cost-ADR policy beats the ordinary FSRS6 ADR portfolio. It gains +208,029 scheduler-only HV delta, +0.605 percentage points of HV delta relative to baseline HV, +13.5 same-budget memory-lift AUC, +0.13 same-target time-save AUC, and wider covered budget and target spans. It also wins the per-user FSRS6 HV comparison on 99/128 users.

That result does not transfer cleanly to the LSTM external validation environment. On LSTM, ADR is ahead by 48,800 HV, 11.2 same-budget memory-lift AUC, and 1.29 same-target time-save AUC. Cost-ADR still covers more of the evaluated budget and target spans, but it loses the per-user LSTM HV comparison on 81/128 users and has more users with negative HV delta versus the FSRS6 baseline.

Operationally, Cost-ADR is the smaller deployment object: one 15-coefficient policy per user, 128 trained policy artifacts total. The ADR portfolio has 16 six-coefficient child policies per user, 2,048 trained child policy artifacts total, and 96 coefficients per user. The price is training cost: this Cost-ADR run was 11.7x slower than ADR in train-overfit and used much higher peak dedicated GPU memory. The practical reading is that this 15p Cost-ADR is a compact FSRS6-native improvement, but it is not yet a robust cross-model replacement for FSRS6 ADR.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_128_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_128_16dr_pop16_gen5.json` |

Analysis summaries:

- FSRS6 Cost ADR 15p rethead nopre 128: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR Portfolio pop16 128: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Strategy Interfaces

| dimension | FSRS6 Cost-ADR 15p rethead nopre 128 | ADR Portfolio pop16 128 |
| --- | --- | --- |
| scheduler | `fsrs6_cost_adr` | `fsrs6_adr` |
| learned object | one cost-conditioned desired-retention policy per user | 16 child dynamic-DR policies per user |
| policy artifacts | 128 metadata artifacts | 2,048 metadata artifacts |
| coefficients per user | 15 | 96 total, as 16 children x 6 coefficients |
| policy feature version | `fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1` | `fsrs6_adr_log_poly_v1` |
| action | emits desired retention conditioned on FSRS state and cost weight | emits desired retention from each portfolio child policy |
| retention bounds | `[0.30, 0.995]` | `[0.50, 0.98]` |
| optimizer | CMA-ES, pop16/gen20, `sigma0 = 1.0`, no coefficient preconditioning | SMS-EMOA portfolio, pop16/off16/gen20, mutation scale 0.35 |
| scalarization / portfolio budget | trains on cost weights `0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024` | no scalar cost weights; keeps a 16-child Pareto portfolio |

Both runs use users 1-128, train against the FSRS6 environment, evaluate in both FSRS6 and LSTM environments, use the same staged FSRS6 baseline DR manifest, use the batched engine for 1,825 days with a 10,000-card deck, and run with `review_markov_transition = false`.

Cost-ADR's DR is not a fixed baseline DR selected from the manifest. The trained 15-coefficient policy generates a desired retention at scheduling time from the current FSRS state and the cost weight. The ADR portfolio also contains trained DR behavior: each child policy is a trained six-coefficient dynamic desired-retention policy, and the portfolio exposes 16 such policies per user.

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | analyze-pareto | yes | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | build-pareto | yes | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | preflight | yes | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | stage-baseline | yes | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | sweep | yes | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | train-overfit | yes | - |
| ADR Portfolio pop16 128 | analyze-pareto | yes | - |
| ADR Portfolio pop16 128 | build-pareto | yes | - |
| ADR Portfolio pop16 128 | preflight | yes | - |
| ADR Portfolio pop16 128 | stage-baseline | yes | - |
| ADR Portfolio pop16 128 | sweep | yes | - |
| ADR Portfolio pop16 128 | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | sweep | cuda | 333.6 | 700.3 | - |
| FSRS6 Cost ADR 15p rethead nopre 128 | train-overfit | cuda | 9,750.2 | 24.0 | 383.3 |
| ADR Portfolio pop16 128 | sweep | cuda | 223.7 | 1,044.2 | - |
| ADR Portfolio pop16 128 | train-overfit | cuda | 834.3 | 280.0 | 4,480.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | d152dcb1de370305001c422864409e5e2e6b68c2 | true | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR Portfolio pop16 128 | 0de1a696a4a48a7ed4d5a39983baa2a0dd7758da | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 284.0 | 302.2 | False | 15,971.0 |
| FSRS6 Cost ADR 15p rethead nopre 128 | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 441.7 | 459.9 | False | 17,537.0 |
| ADR Portfolio pop16 128 | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 172.6 | 190.8 | False | 8,187.0 |
| ADR Portfolio pop16 128 | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 172.0 | 190.2 | False | 5,249.0 |

## Training and Deployment Cost

| metric | FSRS6 Cost-ADR 15p rethead nopre 128 | ADR Portfolio pop16 128 | reading |
| --- | ---: | ---: | --- |
| trained artifacts | 128 | 2,048 | Cost-ADR is 16x smaller by artifact count |
| coefficients per user | 15 | 96 | Cost-ADR is 6.4x smaller by coefficient count |
| train-overfit elapsed | 9,750.2 s | 834.3 s | Cost-ADR was 11.7x slower |
| train-overfit throughput | 24.0 user-days/s | 280.0 user-days/s | ADR was 11.7x faster |
| sweep elapsed | 333.6 s | 223.7 s | Cost-ADR was 1.5x slower |
| sweep throughput | 700.3 user-days/s | 1,044.2 user-days/s | ADR was 1.5x faster |
| train-overfit `nvidia-smi` peak | 17,537 MiB | 5,249 MiB | Cost-ADR used 3.3x more dedicated GPU memory |
| sweep `nvidia-smi` peak | 15,971 MiB | 8,187 MiB | Cost-ADR used 2.0x more dedicated GPU memory |
| shared-memory spill | false | false | neither run crossed the 1 GiB spill threshold |

This is the main engineering tradeoff. Cost-ADR compresses the serving-side policy representation, but the current CMA-ES training path is far more expensive than the ADR portfolio trainer on this 128-user run.

## Conclusion

Promotion depends on the acceptance criterion. If the criterion is native FSRS6 Pareto quality under the matched 16-point sweep and baseline setup, this 15p Cost-ADR run clears ADR: it improves HV, memory lift, time save, and coverage while using a much smaller deployed policy. If the criterion is robust external validation under LSTM, it does not clear ADR: ADR remains better on HV and both AUC metrics.

The most defensible status is therefore "FSRS6-native candidate, not cross-model default." Cost-ADR should not replace FSRS6 ADR globally until the LSTM transfer gap is reduced. The next useful experiment is not another report-only comparison; it is a Cost-ADR transfer repair run, for example mixed FSRS6/LSTM training, LSTM-calibrated initialization, or a regularizer that penalizes the users with large negative LSTM deltas.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR 15p rethead nopre 128 | 1,432,969 | +4.168% | 1,940 | 121.7 | +1.898% | 1606/1801, 83.644% span | 4.21 | +11.447% | 1518/1801, 90.746% span |
| fsrs6 | ADR Portfolio pop16 128 | 1,224,940 | +3.563% | 2,031 | 108.2 | +1.662% | 1444/1801, 80.123% span | 4.09 | +9.762% | 1390/1801, 83.013% span |
| fsrs6 | FSRS6 Cost ADR 15p rethead nopre 128 - ADR Portfolio pop16 128 | 208,029 | +0.605% | -91 | 13.5 | +0.236% | +162, +3.521 pp span | 0.13 | +1.685% | +128, +7.733 pp span |
| lstm | FSRS6 Cost ADR 15p rethead nopre 128 | 599,474 | +1.497% | 1,890 | 16.3 | +0.274% | 1539/1767, 79.558% span | 0.91 | +2.568% | 1577/1767, 92.917% span |
| lstm | ADR Portfolio pop16 128 | 648,275 | +1.618% | 1,893 | 27.4 | +0.430% | 1363/1767, 73.461% span | 2.20 | +4.895% | 1398/1767, 84.135% span |
| lstm | FSRS6 Cost ADR 15p rethead nopre 128 - ADR Portfolio pop16 128 | -48,800 | -0.122% | -3 | -11.2 | -0.156% | +176, +6.097 pp span | -1.29 | -2.327% | +179, +8.782 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 110/256 environment-user rows.

| environment | Cost-ADR wins | ADR wins | Cost-ADR negative vs baseline | ADR negative vs baseline | Cost-minus-ADR delta five-number |
| --- | ---: | ---: | ---: | ---: | --- |
| fsrs6 | 99 | 29 | 2 | 1 | min -26,197.9; p25 65.5; median 690.4; p75 2,605.3; max 30,947.6 |
| lstm | 47 | 81 | 38 | 22 | min -79,646.0; p25 -1,593.4; median -400.0; p75 965.3; max 138,745.5 |

| environment | largest Cost-ADR wins | largest Cost-ADR losses |
| --- | --- | --- |
| fsrs6 | u44 +30,947.6; u33 +16,249.5; u106 +14,272.3; u84 +13,773.1; u95 +12,038.6 | u59 -26,197.9; u101 -22,849.6; u119 -6,635.1; u2 -5,126.2; u27 -4,959.1 |
| lstm | u17 +138,745.5; u106 +16,488.8; u24 +11,325.6; u107 +10,031.6; u14 +8,706.6 | u123 -79,646.0; u59 -27,883.6; u66 -18,684.9; u100 -15,362.4; u34 -14,621.6 |

The FSRS6 per-user distribution is clearly shifted in Cost-ADR's favor: the median user gains +690 HV over ADR and the upper-quartile gain is +2,605 HV. The LSTM distribution is shifted the other way: the median user loses -400 HV to ADR, and 38 Cost-ADR users are below the FSRS6 baseline frontier on LSTM versus 22 ADR users. The very large LSTM win on user 17 is not enough to offset broad small and medium losses.

| user | fsrs6 FSRS6 Cost ADR 15p rethead nopre 128 HV delta | fsrs6 ADR Portfolio pop16 128 HV delta | fsrs6 delta | lstm FSRS6 Cost ADR 15p rethead nopre 128 HV delta | lstm ADR Portfolio pop16 128 HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,342 | 8,624 | 5,718 | 12,464 | 6,077 | 6,387 |
| 2 | 18,600 | 23,726 | -5,126 | -13,507 | 146 | -13,653 |
| 3 | 4,110 | 4,709 | -599 | 3,219 | 4,255 | -1,036 |
| 4 | 36,572 | 33,096 | 3,476 | 14,859 | 17,439 | -2,579 |
| 5 | 7,768 | 7,367 | 402 | 1,849 | 2,650 | -801 |
| 6 | 4,435 | 3,869 | 565 | 2,542 | 2,299 | 243 |
| 7 | 1,504 | 1,610 | -105 | 863 | 985 | -123 |
| 8 | 2,548 | 2,172 | 376 | 820 | 1,407 | -587 |
| 9 | 17,390 | 13,217 | 4,173 | -336 | 7,593 | -7,929 |
| 10 | 356 | 429 | -73 | 9 | -8 | 17 |
| 11 | 28,303 | 24,033 | 4,270 | 20,098 | 22,012 | -1,914 |
| 12 | 1,901 | 2,012 | -111 | -1,006 | -200 | -806 |
| 13 | 1,512 | 2,005 | -493 | 22 | 644 | -622 |
| 14 | 18,240 | 14,225 | 4,016 | 14,987 | 6,280 | 8,707 |
| 15 | 1,472 | 587 | 885 | 5,123 | 1,713 | 3,410 |
| 16 | 8,672 | 5,702 | 2,970 | 9,207 | 7,142 | 2,065 |
| 17 | 12,689 | 5,098 | 7,591 | 19,195 | -119,550 | 138,746 |
| 18 | 33,366 | 33,067 | 299 | 22,594 | 21,035 | 1,559 |
| 19 | 17,521 | 14,388 | 3,134 | 1,588 | 2,420 | -832 |
| 20 | 3,463 | 2,732 | 731 | 264 | 621 | -357 |
| 21 | 13,675 | 14,754 | -1,079 | 27,739 | 28,910 | -1,170 |
| 22 | 4,225 | 1,671 | 2,554 | 4,115 | -193 | 4,309 |
| 23 | 3,261 | 2,578 | 683 | 1,802 | 2,321 | -519 |
| 24 | 15,683 | 15,401 | 282 | 21,974 | 10,649 | 11,326 |
| 25 | 22,828 | 17,293 | 5,535 | 18,859 | 24,364 | -5,504 |
| 26 | 1,337 | 308 | 1,030 | 229 | 87 | 142 |
| 27 | 39,911 | 44,871 | -4,959 | 12,382 | 21,016 | -8,634 |
| 28 | 50,509 | 43,338 | 7,172 | 51,617 | 48,667 | 2,950 |
| 29 | 27,595 | 19,346 | 8,249 | 6,953 | 10,630 | -3,677 |
| 30 | 17,228 | 17,483 | -255 | 4,220 | 4,636 | -416 |
| 31 | 2,488 | 2,352 | 136 | -106 | 944 | -1,050 |
| 32 | 6,505 | 5,541 | 963 | 16,730 | 12,274 | 4,456 |
| 33 | 11,659 | -4,591 | 16,249 | -9,355 | -2,121 | -7,234 |
| 34 | 12,304 | 12,580 | -277 | 774 | 15,395 | -14,622 |
| 35 | 18,867 | 16,966 | 1,902 | -4,196 | -9,529 | 5,333 |
| 36 | 2,285 | 1,588 | 697 | 5,440 | 1,668 | 3,773 |
| 37 | 32,999 | 24,383 | 8,616 | 17,791 | 11,659 | 6,132 |
| 38 | 3,400 | 2,446 | 955 | 231 | 1,892 | -1,661 |
| 39 | 2,567 | 2,530 | 37 | 1,012 | 1,657 | -645 |
| 40 | 2,527 | 2,230 | 297 | 17,166 | 16,047 | 1,119 |
| 41 | 16,604 | 14,144 | 2,460 | 18,900 | 16,353 | 2,547 |
| 42 | 9,867 | 10,304 | -436 | 5,835 | 5,603 | 232 |
| 43 | 4,005 | 3,829 | 176 | 297 | 4,219 | -3,922 |
| 44 | 62,206 | 31,258 | 30,948 | -10,221 | 639 | -10,860 |
| 45 | 980 | 1,291 | -311 | 613 | 264 | 348 |
| 46 | 3,573 | 3,772 | -200 | 3,945 | 5,856 | -1,911 |
| 47 | 16,681 | 13,912 | 2,769 | 2,271 | 2,816 | -545 |
| 48 | 1,989 | 820 | 1,169 | 38 | 235 | -197 |
| 49 | 1,796 | 1,320 | 475 | 177 | 1,318 | -1,141 |
| 50 | 10,902 | 9,113 | 1,789 | 15,098 | 12,588 | 2,511 |
| 51 | 7,837 | 7,648 | 189 | -1,686 | 7,841 | -9,527 |
| 52 | 11,837 | 9,055 | 2,782 | 5,650 | 6,906 | -1,256 |
| 53 | 3,077 | 1,296 | 1,781 | 3,139 | 718 | 2,421 |
| 54 | 947 | 572 | 375 | -90 | 62 | -152 |
| 55 | 1,285 | 738 | 547 | -375 | -72 | -302 |
| 56 | 7,798 | 4,992 | 2,806 | -2,519 | -4,284 | 1,764 |
| 57 | 2,307 | 1,948 | 359 | -1,982 | -77 | -1,905 |
| 58 | 10,077 | 10,431 | -354 | 66,354 | 60,432 | 5,922 |
| 59 | -8,473 | 17,724 | -26,198 | -9,414 | 18,469 | -27,884 |
| 60 | 10,850 | 5,951 | 4,899 | -1,712 | -141 | -1,571 |
| 61 | 4,055 | 3,634 | 420 | 3,271 | 3,349 | -78 |
| 62 | 5,136 | 4,293 | 843 | 1,643 | 2,780 | -1,137 |
| 63 | 2,916 | 2,430 | 486 | 200 | 1,363 | -1,163 |
| 64 | 3,615 | 2,624 | 991 | 947 | 1,499 | -552 |
| 65 | 1,635 | 976 | 659 | -272 | 370 | -642 |
| 66 | 22,645 | 16,533 | 6,112 | -9,810 | 8,874 | -18,685 |
| 67 | 9,037 | 10,166 | -1,129 | -3,717 | 3,146 | -6,863 |
| 68 | 9,048 | 9,016 | 33 | 9,354 | 8,760 | 594 |
| 69 | 5,205 | 3,103 | 2,102 | 1,907 | 2,533 | -626 |
| 70 | 8,079 | 7,204 | 874 | 9,206 | 3,990 | 5,216 |
| 71 | 4,651 | 1,784 | 2,867 | 217 | 224 | -8 |
| 72 | 14,521 | 12,582 | 1,940 | 6,052 | 5,306 | 746 |
| 73 | 12,828 | 13,770 | -943 | 35,059 | 35,442 | -384 |
| 74 | 2,338 | 1,525 | 813 | 233 | 178 | 56 |
| 75 | 5,745 | 4,153 | 1,592 | 580 | 1,308 | -728 |
| 76 | 2,064 | 1,506 | 558 | 1,408 | 1,872 | -464 |
| 77 | 2,038 | 1,346 | 692 | 1,399 | 565 | 834 |
| 78 | 14,285 | 14,318 | -32 | 14,007 | 9,246 | 4,761 |
| 79 | 943 | 1,023 | -81 | 805 | 596 | 209 |
| 80 | 1,623 | 1,511 | 112 | -232 | 200 | -431 |
| 81 | 1,931 | 1,327 | 604 | 2,744 | 1,850 | 895 |
| 82 | 14,675 | 14,575 | 100 | 534 | 758 | -224 |
| 83 | 7,678 | 6,891 | 787 | 966 | 1,545 | -579 |
| 84 | 41,464 | 27,691 | 13,773 | -2,031 | -6,283 | 4,252 |
| 85 | 9,171 | 6,868 | 2,302 | -5,957 | -1,370 | -4,587 |
| 86 | 2,999 | 3,307 | -307 | -1,414 | 2,682 | -4,095 |
| 87 | 7,358 | 6,285 | 1,073 | 766 | 1,321 | -555 |
| 88 | 5,236 | 5,198 | 38 | 1,617 | 2,704 | -1,087 |
| 89 | 4,255 | 4,779 | -524 | 2,411 | 2,318 | 93 |
| 90 | 1,062 | 373 | 689 | 1,875 | 593 | 1,282 |
| 91 | 1,343 | 1,044 | 299 | 389 | 409 | -21 |
| 92 | 4,766 | 2,896 | 1,870 | -934 | 798 | -1,732 |
| 93 | 22,957 | 12,894 | 10,063 | 15,323 | 12,725 | 2,597 |
| 94 | 2,060 | 1,703 | 356 | 3,336 | 1,367 | 1,969 |
| 95 | 56,817 | 44,778 | 12,039 | -15,772 | -15,193 | -579 |
| 96 | 930 | 603 | 327 | 83 | 271 | -188 |
| 97 | 54,077 | 43,400 | 10,677 | -3,522 | -10,954 | 7,432 |
| 98 | 39,973 | 38,003 | 1,970 | 34,688 | 31,964 | 2,724 |
| 99 | 6,426 | 5,893 | 533 | -261 | 4,349 | -4,610 |
| 100 | 58,497 | 53,576 | 4,922 | 3,020 | 18,382 | -15,362 |
| 101 | -4,204 | 18,645 | -22,850 | -3,561 | 9,226 | -12,787 |
| 102 | 3,473 | 2,975 | 498 | -1,383 | 151 | -1,534 |
| 103 | 976 | 670 | 306 | -509 | -335 | -174 |
| 104 | 22,377 | 22,624 | -247 | 31,131 | 39,855 | -8,723 |
| 105 | 3,777 | 1,782 | 1,995 | -1,239 | 2,013 | -3,252 |
| 106 | 18,222 | 3,949 | 14,272 | 55,938 | 39,449 | 16,489 |
| 107 | 43,839 | 37,551 | 6,288 | 56,093 | 46,061 | 10,032 |
| 108 | 5,757 | 4,086 | 1,671 | 40,218 | 46,638 | -6,420 |
| 109 | 7,856 | 7,547 | 309 | 3,643 | 3,982 | -338 |
| 110 | 1,092 | 116 | 976 | -61 | -60 | -1 |
| 111 | 9,338 | 5,690 | 3,648 | -3,796 | -2,449 | -1,348 |
| 112 | 543 | 468 | 75 | 315 | -322 | 637 |
| 113 | 16,378 | 14,318 | 2,059 | 11,359 | 14,974 | -3,614 |
| 114 | 17,220 | 14,044 | 3,176 | 11,687 | 8,687 | 3,001 |
| 115 | 26,133 | 19,280 | 6,853 | 3,172 | 795 | 2,377 |
| 116 | 5,053 | 3,926 | 1,127 | -126 | 1,666 | -1,792 |
| 117 | 13,034 | 13,728 | -693 | 801 | 1,041 | -240 |
| 118 | 4,375 | 4,299 | 76 | 3,115 | 6,719 | -3,604 |
| 119 | 14,387 | 21,022 | -6,635 | -11,505 | -14,846 | 3,341 |
| 120 | 5,449 | 2,525 | 2,925 | 795 | 1,664 | -869 |
| 121 | 7,303 | 4,544 | 2,759 | -5,105 | 957 | -6,063 |
| 122 | 4,150 | 2,697 | 1,453 | -3,074 | -1,992 | -1,082 |
| 123 | 17,618 | 17,688 | -70 | -103,853 | -24,207 | -79,646 |
| 124 | 4,432 | 4,863 | -430 | -1,275 | -262 | -1,014 |
| 125 | 647 | 283 | 364 | -60 | 27 | -88 |
| 126 | 1,732 | 824 | 908 | 1,123 | 209 | 914 |
| 127 | 1,540 | 1,632 | -91 | 377 | 233 | 144 |
| 128 | 1,624 | 1,728 | -104 | 588 | 888 | -300 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR 15p rethead nopre 128 | 6,303.8 | 61.27 | 26.09 | 227.28 |
| fsrs6 | ADR Portfolio pop16 128 | 6,414.2 | 56.40 | 26.22 | 208.98 |
| lstm | FSRS6 Cost ADR 15p rethead nopre 128 | 6,268.3 | 68.92 | 26.09 | 276.73 |
| lstm | ADR Portfolio pop16 128 | 6,391.5 | 60.36 | 26.78 | 241.42 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | 1 | 14,722 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 2 | 19,436 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 3 | 3,819 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 4 | 37,202 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 5 | 7,967 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 6 | 4,644 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 7 | 1,420 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 8 | 2,526 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 9 | 16,974 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 10 | 377 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 11 | 29,224 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 12 | 1,864 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 13 | 1,548 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 14 | 18,225 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 15 | 1,597 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 16 | 8,533 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 17 | 11,349 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 18 | 32,161 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 19 | 17,419 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 20 | 3,471 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 21 | 13,962 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 22 | 4,042 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 23 | 3,310 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 24 | 16,264 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 25 | 21,971 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 26 | 1,256 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 27 | 46,996 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 28 | 49,264 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 29 | 28,603 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 30 | 17,493 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 31 | 2,632 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 32 | 6,670 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 33 | 10,295 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 34 | 13,121 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 35 | 19,148 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 36 | 2,230 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 37 | 33,616 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 38 | 3,448 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 39 | 2,526 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 40 | 2,566 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 41 | 16,227 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 42 | 9,754 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 43 | 3,794 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 44 | 61,655 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 45 | 928 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 46 | 3,374 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 47 | 16,896 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 48 | 2,163 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 49 | 2,006 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 50 | 11,148 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 51 | 8,207 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 52 | 11,770 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 53 | 3,224 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 54 | 936 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 55 | 1,379 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 56 | 7,893 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 57 | 2,315 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 58 | 10,255 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 59 | -9,191 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 60 | 10,536 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 61 | 3,945 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 62 | 5,348 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 63 | 2,631 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 64 | 3,706 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 65 | 1,811 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 66 | 22,309 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 67 | 9,072 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 68 | 8,877 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 69 | 5,082 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 70 | 9,779 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 71 | 4,501 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 72 | 15,425 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 73 | 12,356 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 74 | 2,344 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 75 | 5,594 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 76 | 2,142 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 77 | 1,989 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 78 | 14,195 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 79 | 1,083 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 80 | 1,615 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 81 | 1,854 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 82 | 15,087 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 83 | 7,800 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 84 | 41,462 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 85 | 9,396 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 86 | 3,187 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 87 | 7,530 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 88 | 5,565 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 89 | 4,523 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 90 | 1,160 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 91 | 1,381 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 92 | 4,685 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 93 | 23,385 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 94 | 2,061 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 95 | 58,932 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 96 | 919 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 97 | 56,318 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 98 | 39,571 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 99 | 6,396 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 100 | 56,742 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 101 | -4,111 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 102 | 3,548 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 103 | 1,002 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 104 | 21,960 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 105 | 3,906 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 106 | 18,596 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 107 | 43,622 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 108 | 5,996 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 109 | 7,288 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 110 | 1,040 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 111 | 9,449 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 112 | 641 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 113 | 16,695 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 114 | 16,368 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 115 | 26,968 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 116 | 5,232 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 117 | 12,980 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 118 | 4,555 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 119 | 13,653 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 120 | 5,862 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 121 | 7,819 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 122 | 4,105 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 123 | 17,277 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 124 | 4,501 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 125 | 718 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 126 | 1,824 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 127 | 1,510 |
| FSRS6 Cost ADR 15p rethead nopre 128 | 128 | 1,736 |
| ADR Portfolio pop16 128 | 1 | 14,226 |
| ADR Portfolio pop16 128 | 2 | 27,307 |
| ADR Portfolio pop16 128 | 3 | 5,387 |
| ADR Portfolio pop16 128 | 4 | 33,902 |
| ADR Portfolio pop16 128 | 5 | 7,541 |
| ADR Portfolio pop16 128 | 6 | 4,979 |
| ADR Portfolio pop16 128 | 7 | 1,707 |
| ADR Portfolio pop16 128 | 8 | 2,219 |
| ADR Portfolio pop16 128 | 9 | 13,727 |
| ADR Portfolio pop16 128 | 10 | 600 |
| ADR Portfolio pop16 128 | 11 | 25,749 |
| ADR Portfolio pop16 128 | 12 | 2,276 |
| ADR Portfolio pop16 128 | 13 | 2,302 |
| ADR Portfolio pop16 128 | 14 | 16,154 |
| ADR Portfolio pop16 128 | 15 | 681 |
| ADR Portfolio pop16 128 | 16 | 6,856 |
| ADR Portfolio pop16 128 | 17 | 24,283 |
| ADR Portfolio pop16 128 | 18 | 36,903 |
| ADR Portfolio pop16 128 | 19 | 15,298 |
| ADR Portfolio pop16 128 | 20 | 3,201 |
| ADR Portfolio pop16 128 | 21 | 17,621 |
| ADR Portfolio pop16 128 | 22 | 1,973 |
| ADR Portfolio pop16 128 | 23 | 2,926 |
| ADR Portfolio pop16 128 | 24 | 17,269 |
| ADR Portfolio pop16 128 | 25 | 21,304 |
| ADR Portfolio pop16 128 | 26 | 425 |
| ADR Portfolio pop16 128 | 27 | 60,084 |
| ADR Portfolio pop16 128 | 28 | 45,503 |
| ADR Portfolio pop16 128 | 29 | 23,856 |
| ADR Portfolio pop16 128 | 30 | 18,070 |
| ADR Portfolio pop16 128 | 31 | 2,464 |
| ADR Portfolio pop16 128 | 32 | 6,456 |
| ADR Portfolio pop16 128 | 33 | 30,819 |
| ADR Portfolio pop16 128 | 34 | 25,111 |
| ADR Portfolio pop16 128 | 35 | 20,073 |
| ADR Portfolio pop16 128 | 36 | 2,058 |
| ADR Portfolio pop16 128 | 37 | 25,282 |
| ADR Portfolio pop16 128 | 38 | 2,947 |
| ADR Portfolio pop16 128 | 39 | 2,629 |
| ADR Portfolio pop16 128 | 40 | 2,437 |
| ADR Portfolio pop16 128 | 41 | 15,735 |
| ADR Portfolio pop16 128 | 42 | 11,473 |
| ADR Portfolio pop16 128 | 43 | 4,952 |
| ADR Portfolio pop16 128 | 44 | 38,726 |
| ADR Portfolio pop16 128 | 45 | 1,845 |
| ADR Portfolio pop16 128 | 46 | 4,814 |
| ADR Portfolio pop16 128 | 47 | 15,415 |
| ADR Portfolio pop16 128 | 48 | 1,098 |
| ADR Portfolio pop16 128 | 49 | 1,495 |
| ADR Portfolio pop16 128 | 50 | 10,025 |
| ADR Portfolio pop16 128 | 51 | 8,702 |
| ADR Portfolio pop16 128 | 52 | 14,089 |
| ADR Portfolio pop16 128 | 53 | 2,118 |
| ADR Portfolio pop16 128 | 54 | 665 |
| ADR Portfolio pop16 128 | 55 | 1,737 |
| ADR Portfolio pop16 128 | 56 | 5,488 |
| ADR Portfolio pop16 128 | 57 | 2,407 |
| ADR Portfolio pop16 128 | 58 | 10,734 |
| ADR Portfolio pop16 128 | 59 | 25,728 |
| ADR Portfolio pop16 128 | 60 | 6,278 |
| ADR Portfolio pop16 128 | 61 | 3,949 |
| ADR Portfolio pop16 128 | 62 | 5,518 |
| ADR Portfolio pop16 128 | 63 | 4,535 |
| ADR Portfolio pop16 128 | 64 | 3,027 |
| ADR Portfolio pop16 128 | 65 | 1,009 |
| ADR Portfolio pop16 128 | 66 | 18,469 |
| ADR Portfolio pop16 128 | 67 | 12,966 |
| ADR Portfolio pop16 128 | 68 | 9,552 |
| ADR Portfolio pop16 128 | 69 | 3,632 |
| ADR Portfolio pop16 128 | 70 | 11,682 |
| ADR Portfolio pop16 128 | 71 | 2,238 |
| ADR Portfolio pop16 128 | 72 | 15,079 |
| ADR Portfolio pop16 128 | 73 | 17,614 |
| ADR Portfolio pop16 128 | 74 | 1,499 |
| ADR Portfolio pop16 128 | 75 | 5,082 |
| ADR Portfolio pop16 128 | 76 | 1,684 |
| ADR Portfolio pop16 128 | 77 | 2,381 |
| ADR Portfolio pop16 128 | 78 | 16,601 |
| ADR Portfolio pop16 128 | 79 | 1,092 |
| ADR Portfolio pop16 128 | 80 | 1,782 |
| ADR Portfolio pop16 128 | 81 | 1,664 |
| ADR Portfolio pop16 128 | 82 | 15,055 |
| ADR Portfolio pop16 128 | 83 | 7,325 |
| ADR Portfolio pop16 128 | 84 | 29,260 |
| ADR Portfolio pop16 128 | 85 | 7,073 |
| ADR Portfolio pop16 128 | 86 | 4,474 |
| ADR Portfolio pop16 128 | 87 | 6,599 |
| ADR Portfolio pop16 128 | 88 | 6,222 |
| ADR Portfolio pop16 128 | 89 | 5,229 |
| ADR Portfolio pop16 128 | 90 | 295 |
| ADR Portfolio pop16 128 | 91 | 1,087 |
| ADR Portfolio pop16 128 | 92 | 3,237 |
| ADR Portfolio pop16 128 | 93 | 26,719 |
| ADR Portfolio pop16 128 | 94 | 2,099 |
| ADR Portfolio pop16 128 | 95 | 46,357 |
| ADR Portfolio pop16 128 | 96 | 656 |
| ADR Portfolio pop16 128 | 97 | 53,245 |
| ADR Portfolio pop16 128 | 98 | 42,808 |
| ADR Portfolio pop16 128 | 99 | 7,469 |
| ADR Portfolio pop16 128 | 100 | 58,377 |
| ADR Portfolio pop16 128 | 101 | 36,206 |
| ADR Portfolio pop16 128 | 102 | 3,629 |
| ADR Portfolio pop16 128 | 103 | 885 |
| ADR Portfolio pop16 128 | 104 | 31,464 |
| ADR Portfolio pop16 128 | 105 | 2,551 |
| ADR Portfolio pop16 128 | 106 | 19,103 |
| ADR Portfolio pop16 128 | 107 | 44,537 |
| ADR Portfolio pop16 128 | 108 | 5,024 |
| ADR Portfolio pop16 128 | 109 | 7,575 |
| ADR Portfolio pop16 128 | 110 | 450 |
| ADR Portfolio pop16 128 | 111 | 18,460 |
| ADR Portfolio pop16 128 | 112 | 592 |
| ADR Portfolio pop16 128 | 113 | 16,340 |
| ADR Portfolio pop16 128 | 114 | 18,441 |
| ADR Portfolio pop16 128 | 115 | 21,132 |
| ADR Portfolio pop16 128 | 116 | 4,226 |
| ADR Portfolio pop16 128 | 117 | 14,379 |
| ADR Portfolio pop16 128 | 118 | 4,678 |
| ADR Portfolio pop16 128 | 119 | 22,776 |
| ADR Portfolio pop16 128 | 120 | 4,052 |
| ADR Portfolio pop16 128 | 121 | 4,904 |
| ADR Portfolio pop16 128 | 122 | 2,802 |
| ADR Portfolio pop16 128 | 123 | 20,648 |
| ADR Portfolio pop16 128 | 124 | 6,310 |
| ADR Portfolio pop16 128 | 125 | 327 |
| ADR Portfolio pop16 128 | 126 | 1,308 |
| ADR Portfolio pop16 128 | 127 | 2,243 |
| ADR Portfolio pop16 128 | 128 | 2,127 |

## Training HV

| run | users | final training HV gain sum |
| --- | --- | --- |
| FSRS6 Cost ADR 15p rethead nopre 128 | 128 | 1,445,659 |
| ADR Portfolio pop16 128 | 128 | 1,539,934 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-28-fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1.md`
