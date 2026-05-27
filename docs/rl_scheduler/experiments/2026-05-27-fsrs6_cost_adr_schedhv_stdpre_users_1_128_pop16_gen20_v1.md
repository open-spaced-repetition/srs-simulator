# FSRS6 Cost ADR schedHV stdpre 128 vs ADR Portfolio pop16 128 experiment report

Machine summary: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/report/report_summary.json`

## Question

Evaluate whether the scheduler-HV diagonal-preconditioned 24-parameter Cost-ADR policy reaches the matched-budget ordinary FSRS6 ADR portfolio on the first 128 users with the matched 16 cost weights.

## Runs

| run | scheduler | config | training budget | review Markov | baseline DR manifest |
| --- | --- | --- | --- | --- | --- |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off` | `fsrs6_cost_adr` | `experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml` | optimizer=cma_es, population=16, generations=20, sigma0=1.0, cost_weights=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_128_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off` | `fsrs6_adr` | `experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_pop16_v1.toml` | population=16, offspring=16, generations=20, portfolio=16 | off | `artifacts/rl_scheduler/baseline_dr_selection/fsrs6_users_1_128_16dr_pop16_gen5.json` |

Analysis summaries:

- FSRS6 Cost ADR schedHV stdpre 128: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`
- ADR Portfolio pop16 128: `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/analyze-pareto/analyze_pareto_outputs/analysis_summary.json`

## Stage Status

| run | stage | passed | failures |
| --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre 128 | analyze-pareto | yes | - |
| FSRS6 Cost ADR schedHV stdpre 128 | build-pareto | yes | - |
| FSRS6 Cost ADR schedHV stdpre 128 | preflight | yes | - |
| FSRS6 Cost ADR schedHV stdpre 128 | stage-baseline | yes | - |
| FSRS6 Cost ADR schedHV stdpre 128 | sweep | yes | - |
| FSRS6 Cost ADR schedHV stdpre 128 | train-overfit | yes | - |
| ADR Portfolio pop16 128 | analyze-pareto | yes | - |
| ADR Portfolio pop16 128 | build-pareto | yes | - |
| ADR Portfolio pop16 128 | preflight | yes | - |
| ADR Portfolio pop16 128 | stage-baseline | yes | - |
| ADR Portfolio pop16 128 | sweep | yes | - |
| ADR Portfolio pop16 128 | train-overfit | yes | - |

## Performance

| run | stage | device | elapsed seconds | user-days/s | candidate-days/s |
| --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre 128 | sweep | cuda | 228.9 | 1,020.4 | - |
| FSRS6 Cost ADR schedHV stdpre 128 | train-overfit | cuda | 9,944.2 | 23.5 | 375.9 |
| ADR Portfolio pop16 128 | sweep | cuda | 223.7 | 1,044.2 | - |
| ADR Portfolio pop16 128 | train-overfit | cuda | 834.3 | 280.0 | 4,480.0 |

## Provenance

| run | git commit | dirty | Python | PyTorch | CUDA | device |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre 128 | 0de1a696a4a48a7ed4d5a39983baa2a0dd7758da | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |
| ADR Portfolio pop16 128 | 0de1a696a4a48a7ed4d5a39983baa2a0dd7758da | false | 3.13.11 | 2.9.1+cu126 | 12.6 | NVIDIA GeForce RTX 4090 D |

## GPU Monitor

| run | stage | summary | shared peak MiB | summed peak MiB | spill | nvidia-smi peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre 128 | sweep | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/sweep/gpu_monitor/summary.json` | 175.3 | 193.5 | False | 8,163.0 |
| FSRS6 Cost ADR schedHV stdpre 128 | train-overfit | `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 377.9 | 396.1 | False | 12,014.0 |
| ADR Portfolio pop16 128 | sweep | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/sweep/gpu_monitor/summary.json` | 172.6 | 190.8 | False | 8,187.0 |
| ADR Portfolio pop16 128 | train-overfit | `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off/train-overfit/gpu_monitor/summary.json` | 172.0 | 190.2 | False | 5,249.0 |

## Conclusion

Promotion decision for `fsrs6_cost_adr` is inconclusive.

Candidate-minus-comparison deltas on the primary external Pareto metrics are:

- fsrs6: 143,532 HV, 26.6 same-budget memory lift AUC, -0.10 same-target time saved AUC versus comparison.
- lstm: -208,091 HV, -14.1 same-budget memory lift AUC, -1.57 same-target time saved AUC versus comparison.

Training HV and sampled policy-point diagnostics should be interpreted against the external Pareto metrics.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user HV delta against the same staged FSRS6 baseline manifest. Positive HV delta and same-budget memory lift are better. Positive same-target time saved is better. The two baseline-relative AUC columns use user-simple averages from the analysis summary.

| environment | scheduler | HV delta sum | HV delta / baseline HV | frontier points | same-budget memory lift AUC | same-budget memory lift / baseline | budget coverage | same-target time saved AUC | same-target time saved / baseline | target coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre 128 | 1,368,471 | +3.981% | 1,948 | 134.7 | +2.123% | 1571/1801, 81.608% span | 3.99 | +11.433% | 1480/1801, 88.749% span |
| fsrs6 | ADR Portfolio pop16 128 | 1,224,940 | +3.563% | 2,031 | 108.2 | +1.662% | 1444/1801, 80.123% span | 4.09 | +9.762% | 1390/1801, 83.013% span |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre 128 - ADR Portfolio pop16 128 | 143,532 | +0.418% | -83 | 26.6 | +0.460% | +127, +1.485 pp span | -0.10 | +1.671% | +90, +5.736 pp span |
| lstm | FSRS6 Cost ADR schedHV stdpre 128 | 440,183 | +1.099% | 1,892 | 13.3 | +0.212% | 1505/1767, 75.748% span | 0.63 | +2.842% | 1552/1767, 92.021% span |
| lstm | ADR Portfolio pop16 128 | 648,275 | +1.618% | 1,893 | 27.4 | +0.430% | 1363/1767, 73.461% span | 2.20 | +4.895% | 1398/1767, 84.135% span |
| lstm | FSRS6 Cost ADR schedHV stdpre 128 - ADR Portfolio pop16 128 | -208,091 | -0.519% | -1 | -14.1 | -0.218% | +142, +2.287 pp span | -1.57 | -2.053% | +154, +7.886 pp span |

## Per-User HV Delta

Candidate-minus-comparison per-user HV delta is negative for 119/256 environment-user rows.

| user | fsrs6 FSRS6 Cost ADR schedHV stdpre 128 HV delta | fsrs6 ADR Portfolio pop16 128 HV delta | fsrs6 delta | lstm FSRS6 Cost ADR schedHV stdpre 128 HV delta | lstm ADR Portfolio pop16 128 HV delta | lstm delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12,769 | 8,624 | 4,145 | 15,293 | 6,077 | 9,216 |
| 2 | 21,025 | 23,726 | -2,701 | 10,567 | 146 | 10,421 |
| 3 | 3,919 | 4,709 | -790 | 3,097 | 4,255 | -1,158 |
| 4 | 28,618 | 33,096 | -4,478 | 2,947 | 17,439 | -14,491 |
| 5 | 8,787 | 7,367 | 1,420 | 1,674 | 2,650 | -976 |
| 6 | 4,073 | 3,869 | 203 | 2,647 | 2,299 | 348 |
| 7 | 1,338 | 1,610 | -272 | 901 | 985 | -84 |
| 8 | 2,656 | 2,172 | 484 | 1,109 | 1,407 | -298 |
| 9 | 18,298 | 13,217 | 5,081 | 7,338 | 7,593 | -255 |
| 10 | 501 | 429 | 72 | -64 | -8 | -56 |
| 11 | 27,080 | 24,033 | 3,047 | 15,264 | 22,012 | -6,748 |
| 12 | 1,954 | 2,012 | -58 | -1,441 | -200 | -1,241 |
| 13 | 1,828 | 2,005 | -177 | 567 | 644 | -77 |
| 14 | 17,958 | 14,225 | 3,734 | 14,972 | 6,280 | 8,692 |
| 15 | 1,640 | 587 | 1,054 | 4,373 | 1,713 | 2,660 |
| 16 | 8,194 | 5,702 | 2,491 | 8,958 | 7,142 | 1,815 |
| 17 | -34 | 5,098 | -5,132 | -12,142 | -119,550 | 107,408 |
| 18 | 17,631 | 33,067 | -15,436 | 8,411 | 21,035 | -12,623 |
| 19 | 18,922 | 14,388 | 4,535 | 2,343 | 2,420 | -76 |
| 20 | 3,356 | 2,732 | 624 | 710 | 621 | 88 |
| 21 | 15,100 | 14,754 | 345 | 2,751 | 28,910 | -26,159 |
| 22 | 4,289 | 1,671 | 2,618 | 3,125 | -193 | 3,319 |
| 23 | 3,362 | 2,578 | 784 | 1,012 | 2,321 | -1,309 |
| 24 | 16,602 | 15,401 | 1,201 | -361 | 10,649 | -11,010 |
| 25 | 17,527 | 17,293 | 234 | 20,272 | 24,364 | -4,091 |
| 26 | 1,565 | 308 | 1,257 | 144 | 87 | 57 |
| 27 | 34,100 | 44,871 | -10,770 | 8,426 | 21,016 | -12,590 |
| 28 | 42,214 | 43,338 | -1,123 | 46,947 | 48,667 | -1,721 |
| 29 | 28,832 | 19,346 | 9,486 | 7,989 | 10,630 | -2,641 |
| 30 | 15,973 | 17,483 | -1,510 | 4,259 | 4,636 | -377 |
| 31 | 2,903 | 2,352 | 551 | 383 | 944 | -560 |
| 32 | 6,759 | 5,541 | 1,218 | 13,681 | 12,274 | 1,408 |
| 33 | 9,394 | -4,591 | 13,985 | 7,757 | -2,121 | 9,878 |
| 34 | 13,805 | 12,580 | 1,224 | 3,263 | 15,395 | -12,132 |
| 35 | 18,643 | 16,966 | 1,678 | -10,964 | -9,529 | -1,435 |
| 36 | 2,314 | 1,588 | 726 | 5,267 | 1,668 | 3,600 |
| 37 | 34,864 | 24,383 | 10,481 | 2,115 | 11,659 | -9,544 |
| 38 | 3,919 | 2,446 | 1,473 | -21 | 1,892 | -1,913 |
| 39 | 2,176 | 2,530 | -354 | 442 | 1,657 | -1,216 |
| 40 | 2,251 | 2,230 | 21 | 16,658 | 16,047 | 611 |
| 41 | 14,185 | 14,144 | 41 | 18,443 | 16,353 | 2,090 |
| 42 | 8,780 | 10,304 | -1,524 | 4,998 | 5,603 | -605 |
| 43 | 3,374 | 3,829 | -456 | -225 | 4,219 | -4,444 |
| 44 | 64,934 | 31,258 | 33,676 | -46,217 | 639 | -46,856 |
| 45 | 934 | 1,291 | -357 | -14 | 264 | -278 |
| 46 | 2,982 | 3,772 | -790 | 6,008 | 5,856 | 152 |
| 47 | 15,645 | 13,912 | 1,733 | 150 | 2,816 | -2,666 |
| 48 | 2,177 | 820 | 1,357 | 490 | 235 | 255 |
| 49 | 1,638 | 1,320 | 317 | -334 | 1,318 | -1,652 |
| 50 | 8,748 | 9,113 | -366 | 13,268 | 12,588 | 681 |
| 51 | 8,338 | 7,648 | 690 | -2,326 | 7,841 | -10,167 |
| 52 | 7,712 | 9,055 | -1,343 | 5,612 | 6,906 | -1,293 |
| 53 | 3,845 | 1,296 | 2,549 | 4,053 | 718 | 3,334 |
| 54 | 1,087 | 572 | 515 | -71 | 62 | -132 |
| 55 | 1,130 | 738 | 392 | -509 | -72 | -437 |
| 56 | 7,718 | 4,992 | 2,726 | -1,786 | -4,284 | 2,497 |
| 57 | 1,660 | 1,948 | -288 | -818 | -77 | -741 |
| 58 | 11,161 | 10,431 | 729 | 16,669 | 60,432 | -43,763 |
| 59 | -3,909 | 17,724 | -21,633 | -11,474 | 18,469 | -29,944 |
| 60 | 11,616 | 5,951 | 5,665 | -1,577 | -141 | -1,436 |
| 61 | 4,127 | 3,634 | 493 | 3,581 | 3,349 | 233 |
| 62 | 4,935 | 4,293 | 642 | 2,397 | 2,780 | -383 |
| 63 | 2,525 | 2,430 | 95 | 2,560 | 1,363 | 1,197 |
| 64 | 3,376 | 2,624 | 752 | 1,165 | 1,499 | -334 |
| 65 | 1,821 | 976 | 845 | -95 | 370 | -465 |
| 66 | 24,024 | 16,533 | 7,491 | 9,619 | 8,874 | 744 |
| 67 | 9,208 | 10,166 | -958 | -4,216 | 3,146 | -7,362 |
| 68 | 7,596 | 9,016 | -1,420 | 9,143 | 8,760 | 384 |
| 69 | 4,339 | 3,103 | 1,237 | -1,075 | 2,533 | -3,608 |
| 70 | 8,374 | 7,204 | 1,170 | 21,715 | 3,990 | 17,725 |
| 71 | 4,753 | 1,784 | 2,969 | -125 | 224 | -350 |
| 72 | 13,283 | 12,582 | 702 | 6,241 | 5,306 | 935 |
| 73 | 11,265 | 13,770 | -2,506 | 37,668 | 35,442 | 2,226 |
| 74 | 2,676 | 1,525 | 1,152 | 239 | 178 | 61 |
| 75 | 6,661 | 4,153 | 2,508 | 937 | 1,308 | -371 |
| 76 | 2,507 | 1,506 | 1,001 | 1,694 | 1,872 | -178 |
| 77 | 1,654 | 1,346 | 308 | 1,095 | 565 | 531 |
| 78 | 11,614 | 14,318 | -2,704 | 7,742 | 9,246 | -1,504 |
| 79 | 1,134 | 1,023 | 111 | 922 | 596 | 326 |
| 80 | 1,800 | 1,511 | 289 | -175 | 200 | -374 |
| 81 | 2,054 | 1,327 | 726 | 2,709 | 1,850 | 859 |
| 82 | 12,292 | 14,575 | -2,283 | -2,941 | 758 | -3,699 |
| 83 | 7,372 | 6,891 | 481 | -2,503 | 1,545 | -4,048 |
| 84 | 42,385 | 27,691 | 14,694 | -3,025 | -6,283 | 3,258 |
| 85 | 9,384 | 6,868 | 2,516 | -7,871 | -1,370 | -6,500 |
| 86 | 3,057 | 3,307 | -250 | 891 | 2,682 | -1,791 |
| 87 | 7,082 | 6,285 | 797 | 685 | 1,321 | -636 |
| 88 | 4,552 | 5,198 | -646 | 1,682 | 2,704 | -1,022 |
| 89 | 3,747 | 4,779 | -1,032 | 2,818 | 2,318 | 500 |
| 90 | 1,282 | 373 | 910 | 2,258 | 593 | 1,664 |
| 91 | 1,558 | 1,044 | 514 | 444 | 409 | 35 |
| 92 | 4,588 | 2,896 | 1,692 | -3,281 | 798 | -4,079 |
| 93 | 20,142 | 12,894 | 7,248 | 15,989 | 12,725 | 3,264 |
| 94 | 2,266 | 1,703 | 562 | 4,142 | 1,367 | 2,774 |
| 95 | 60,387 | 44,778 | 15,609 | -22,206 | -15,193 | -7,013 |
| 96 | 1,459 | 603 | 856 | 197 | 271 | -75 |
| 97 | 57,668 | 43,400 | 14,268 | -5,109 | -10,954 | 5,844 |
| 98 | 37,564 | 38,003 | -439 | 37,674 | 31,964 | 5,710 |
| 99 | 4,146 | 5,893 | -1,747 | 2,220 | 4,349 | -2,129 |
| 100 | 57,376 | 53,576 | 3,800 | 2,201 | 18,382 | -16,182 |
| 101 | 5,645 | 18,645 | -13,001 | 6,272 | 9,226 | -2,955 |
| 102 | 3,007 | 2,975 | 32 | -3,637 | 151 | -3,788 |
| 103 | 879 | 670 | 210 | -462 | -335 | -127 |
| 104 | 22,792 | 22,624 | 168 | 29,495 | 39,855 | -10,360 |
| 105 | 4,052 | 1,782 | 2,270 | 882 | 2,013 | -1,132 |
| 106 | 17,338 | 3,949 | 13,389 | 55,708 | 39,449 | 16,260 |
| 107 | 38,591 | 37,551 | 1,040 | 51,143 | 46,061 | 5,081 |
| 108 | 6,483 | 4,086 | 2,397 | 66,239 | 46,638 | 19,602 |
| 109 | 7,098 | 7,547 | -449 | 2,125 | 3,982 | -1,856 |
| 110 | 1,267 | 116 | 1,151 | -44 | -60 | 16 |
| 111 | 9,691 | 5,690 | 4,001 | 324 | -2,449 | 2,773 |
| 112 | 1,010 | 468 | 542 | -178 | -322 | 144 |
| 113 | 11,966 | 14,318 | -2,352 | 10,945 | 14,974 | -4,029 |
| 114 | 11,700 | 14,044 | -2,344 | 5,155 | 8,687 | -3,532 |
| 115 | 26,888 | 19,280 | 7,608 | -3,494 | 795 | -4,290 |
| 116 | 5,486 | 3,926 | 1,560 | 1,985 | 1,666 | 319 |
| 117 | 12,061 | 13,728 | -1,667 | -401 | 1,041 | -1,442 |
| 118 | 3,625 | 4,299 | -674 | 4,530 | 6,719 | -2,189 |
| 119 | 14,570 | 21,022 | -6,452 | -18,449 | -14,846 | -3,603 |
| 120 | 5,451 | 2,525 | 2,926 | 1,699 | 1,664 | 35 |
| 121 | 7,501 | 4,544 | 2,957 | -4,194 | 957 | -5,151 |
| 122 | 5,393 | 2,697 | 2,696 | -3,015 | -1,992 | -1,023 |
| 123 | 15,832 | 17,688 | -1,856 | -116,907 | -24,207 | -92,700 |
| 124 | 4,761 | 4,863 | -102 | -504 | -262 | -242 |
| 125 | 791 | 283 | 508 | -9 | 27 | -37 |
| 126 | 1,681 | 824 | 857 | 661 | 209 | 452 |
| 127 | 1,602 | 1,632 | -30 | 115 | 233 | -118 |
| 128 | 2,419 | 1,728 | 691 | 1,179 | 888 | 291 |

## Diagnostics

Unweighted policy-point averages describe where sampled policies lie; they are diagnostics only and do not replace external Pareto evidence.

| environment | scheduler | policy-point avg memorized | policy-point avg time | policy-point avg efficiency | policy-point avg reviews |
| --- | --- | --- | --- | --- | --- |
| fsrs6 | FSRS6 Cost ADR schedHV stdpre 128 | 6,170.4 | 55.54 | 28.17 | 203.42 |
| fsrs6 | ADR Portfolio pop16 128 | 6,414.2 | 56.40 | 26.22 | 208.98 |
| lstm | FSRS6 Cost ADR schedHV stdpre 128 | 6,138.1 | 61.63 | 28.26 | 242.79 |
| lstm | ADR Portfolio pop16 128 | 6,391.5 | 60.36 | 26.78 | 241.42 |

Train-overfit final HV gain by user:

| run | user | final training HV gain |
| --- | --- | --- |
| FSRS6 Cost ADR schedHV stdpre 128 | 1 | 12,745 |
| FSRS6 Cost ADR schedHV stdpre 128 | 2 | 21,261 |
| FSRS6 Cost ADR schedHV stdpre 128 | 3 | 3,663 |
| FSRS6 Cost ADR schedHV stdpre 128 | 4 | 28,322 |
| FSRS6 Cost ADR schedHV stdpre 128 | 5 | 8,709 |
| FSRS6 Cost ADR schedHV stdpre 128 | 6 | 4,150 |
| FSRS6 Cost ADR schedHV stdpre 128 | 7 | 1,300 |
| FSRS6 Cost ADR schedHV stdpre 128 | 8 | 2,609 |
| FSRS6 Cost ADR schedHV stdpre 128 | 9 | 18,081 |
| FSRS6 Cost ADR schedHV stdpre 128 | 10 | 515 |
| FSRS6 Cost ADR schedHV stdpre 128 | 11 | 27,861 |
| FSRS6 Cost ADR schedHV stdpre 128 | 12 | 1,940 |
| FSRS6 Cost ADR schedHV stdpre 128 | 13 | 1,782 |
| FSRS6 Cost ADR schedHV stdpre 128 | 14 | 17,952 |
| FSRS6 Cost ADR schedHV stdpre 128 | 15 | 1,789 |
| FSRS6 Cost ADR schedHV stdpre 128 | 16 | 8,273 |
| FSRS6 Cost ADR schedHV stdpre 128 | 17 | -792 |
| FSRS6 Cost ADR schedHV stdpre 128 | 18 | 17,301 |
| FSRS6 Cost ADR schedHV stdpre 128 | 19 | 18,728 |
| FSRS6 Cost ADR schedHV stdpre 128 | 20 | 3,426 |
| FSRS6 Cost ADR schedHV stdpre 128 | 21 | 14,854 |
| FSRS6 Cost ADR schedHV stdpre 128 | 22 | 4,181 |
| FSRS6 Cost ADR schedHV stdpre 128 | 23 | 3,469 |
| FSRS6 Cost ADR schedHV stdpre 128 | 24 | 17,388 |
| FSRS6 Cost ADR schedHV stdpre 128 | 25 | 17,062 |
| FSRS6 Cost ADR schedHV stdpre 128 | 26 | 1,537 |
| FSRS6 Cost ADR schedHV stdpre 128 | 27 | 33,796 |
| FSRS6 Cost ADR schedHV stdpre 128 | 28 | 41,573 |
| FSRS6 Cost ADR schedHV stdpre 128 | 29 | 29,131 |
| FSRS6 Cost ADR schedHV stdpre 128 | 30 | 16,779 |
| FSRS6 Cost ADR schedHV stdpre 128 | 31 | 2,811 |
| FSRS6 Cost ADR schedHV stdpre 128 | 32 | 6,524 |
| FSRS6 Cost ADR schedHV stdpre 128 | 33 | 9,930 |
| FSRS6 Cost ADR schedHV stdpre 128 | 34 | 14,490 |
| FSRS6 Cost ADR schedHV stdpre 128 | 35 | 18,308 |
| FSRS6 Cost ADR schedHV stdpre 128 | 36 | 2,113 |
| FSRS6 Cost ADR schedHV stdpre 128 | 37 | 35,068 |
| FSRS6 Cost ADR schedHV stdpre 128 | 38 | 3,772 |
| FSRS6 Cost ADR schedHV stdpre 128 | 39 | 2,154 |
| FSRS6 Cost ADR schedHV stdpre 128 | 40 | 2,038 |
| FSRS6 Cost ADR schedHV stdpre 128 | 41 | 13,843 |
| FSRS6 Cost ADR schedHV stdpre 128 | 42 | 9,138 |
| FSRS6 Cost ADR schedHV stdpre 128 | 43 | 3,395 |
| FSRS6 Cost ADR schedHV stdpre 128 | 44 | 65,343 |
| FSRS6 Cost ADR schedHV stdpre 128 | 45 | 882 |
| FSRS6 Cost ADR schedHV stdpre 128 | 46 | 3,074 |
| FSRS6 Cost ADR schedHV stdpre 128 | 47 | 15,533 |
| FSRS6 Cost ADR schedHV stdpre 128 | 48 | 2,144 |
| FSRS6 Cost ADR schedHV stdpre 128 | 49 | 1,727 |
| FSRS6 Cost ADR schedHV stdpre 128 | 50 | 8,943 |
| FSRS6 Cost ADR schedHV stdpre 128 | 51 | 8,645 |
| FSRS6 Cost ADR schedHV stdpre 128 | 52 | 7,627 |
| FSRS6 Cost ADR schedHV stdpre 128 | 53 | 3,777 |
| FSRS6 Cost ADR schedHV stdpre 128 | 54 | 1,096 |
| FSRS6 Cost ADR schedHV stdpre 128 | 55 | 1,227 |
| FSRS6 Cost ADR schedHV stdpre 128 | 56 | 7,813 |
| FSRS6 Cost ADR schedHV stdpre 128 | 57 | 1,647 |
| FSRS6 Cost ADR schedHV stdpre 128 | 58 | 11,112 |
| FSRS6 Cost ADR schedHV stdpre 128 | 59 | -3,689 |
| FSRS6 Cost ADR schedHV stdpre 128 | 60 | 11,463 |
| FSRS6 Cost ADR schedHV stdpre 128 | 61 | 4,133 |
| FSRS6 Cost ADR schedHV stdpre 128 | 62 | 5,006 |
| FSRS6 Cost ADR schedHV stdpre 128 | 63 | 2,539 |
| FSRS6 Cost ADR schedHV stdpre 128 | 64 | 3,479 |
| FSRS6 Cost ADR schedHV stdpre 128 | 65 | 1,739 |
| FSRS6 Cost ADR schedHV stdpre 128 | 66 | 22,777 |
| FSRS6 Cost ADR schedHV stdpre 128 | 67 | 9,368 |
| FSRS6 Cost ADR schedHV stdpre 128 | 68 | 7,697 |
| FSRS6 Cost ADR schedHV stdpre 128 | 69 | 4,035 |
| FSRS6 Cost ADR schedHV stdpre 128 | 70 | 9,056 |
| FSRS6 Cost ADR schedHV stdpre 128 | 71 | 4,660 |
| FSRS6 Cost ADR schedHV stdpre 128 | 72 | 13,974 |
| FSRS6 Cost ADR schedHV stdpre 128 | 73 | 11,378 |
| FSRS6 Cost ADR schedHV stdpre 128 | 74 | 2,809 |
| FSRS6 Cost ADR schedHV stdpre 128 | 75 | 6,303 |
| FSRS6 Cost ADR schedHV stdpre 128 | 76 | 2,519 |
| FSRS6 Cost ADR schedHV stdpre 128 | 77 | 1,760 |
| FSRS6 Cost ADR schedHV stdpre 128 | 78 | 11,193 |
| FSRS6 Cost ADR schedHV stdpre 128 | 79 | 1,334 |
| FSRS6 Cost ADR schedHV stdpre 128 | 80 | 1,787 |
| FSRS6 Cost ADR schedHV stdpre 128 | 81 | 2,028 |
| FSRS6 Cost ADR schedHV stdpre 128 | 82 | 13,004 |
| FSRS6 Cost ADR schedHV stdpre 128 | 83 | 7,655 |
| FSRS6 Cost ADR schedHV stdpre 128 | 84 | 42,298 |
| FSRS6 Cost ADR schedHV stdpre 128 | 85 | 9,670 |
| FSRS6 Cost ADR schedHV stdpre 128 | 86 | 3,239 |
| FSRS6 Cost ADR schedHV stdpre 128 | 87 | 7,101 |
| FSRS6 Cost ADR schedHV stdpre 128 | 88 | 4,496 |
| FSRS6 Cost ADR schedHV stdpre 128 | 89 | 3,913 |
| FSRS6 Cost ADR schedHV stdpre 128 | 90 | 1,366 |
| FSRS6 Cost ADR schedHV stdpre 128 | 91 | 1,462 |
| FSRS6 Cost ADR schedHV stdpre 128 | 92 | 4,537 |
| FSRS6 Cost ADR schedHV stdpre 128 | 93 | 18,952 |
| FSRS6 Cost ADR schedHV stdpre 128 | 94 | 2,230 |
| FSRS6 Cost ADR schedHV stdpre 128 | 95 | 61,859 |
| FSRS6 Cost ADR schedHV stdpre 128 | 96 | 1,426 |
| FSRS6 Cost ADR schedHV stdpre 128 | 97 | 58,063 |
| FSRS6 Cost ADR schedHV stdpre 128 | 98 | 38,092 |
| FSRS6 Cost ADR schedHV stdpre 128 | 99 | 4,852 |
| FSRS6 Cost ADR schedHV stdpre 128 | 100 | 57,464 |
| FSRS6 Cost ADR schedHV stdpre 128 | 101 | 6,003 |
| FSRS6 Cost ADR schedHV stdpre 128 | 102 | 3,004 |
| FSRS6 Cost ADR schedHV stdpre 128 | 103 | 960 |
| FSRS6 Cost ADR schedHV stdpre 128 | 104 | 24,913 |
| FSRS6 Cost ADR schedHV stdpre 128 | 105 | 3,878 |
| FSRS6 Cost ADR schedHV stdpre 128 | 106 | 17,257 |
| FSRS6 Cost ADR schedHV stdpre 128 | 107 | 38,896 |
| FSRS6 Cost ADR schedHV stdpre 128 | 108 | 5,976 |
| FSRS6 Cost ADR schedHV stdpre 128 | 109 | 6,827 |
| FSRS6 Cost ADR schedHV stdpre 128 | 110 | 1,142 |
| FSRS6 Cost ADR schedHV stdpre 128 | 111 | 9,058 |
| FSRS6 Cost ADR schedHV stdpre 128 | 112 | 868 |
| FSRS6 Cost ADR schedHV stdpre 128 | 113 | 12,849 |
| FSRS6 Cost ADR schedHV stdpre 128 | 114 | 11,718 |
| FSRS6 Cost ADR schedHV stdpre 128 | 115 | 27,190 |
| FSRS6 Cost ADR schedHV stdpre 128 | 116 | 5,635 |
| FSRS6 Cost ADR schedHV stdpre 128 | 117 | 12,197 |
| FSRS6 Cost ADR schedHV stdpre 128 | 118 | 3,661 |
| FSRS6 Cost ADR schedHV stdpre 128 | 119 | 14,563 |
| FSRS6 Cost ADR schedHV stdpre 128 | 120 | 5,873 |
| FSRS6 Cost ADR schedHV stdpre 128 | 121 | 7,913 |
| FSRS6 Cost ADR schedHV stdpre 128 | 122 | 5,494 |
| FSRS6 Cost ADR schedHV stdpre 128 | 123 | 16,076 |
| FSRS6 Cost ADR schedHV stdpre 128 | 124 | 4,901 |
| FSRS6 Cost ADR schedHV stdpre 128 | 125 | 786 |
| FSRS6 Cost ADR schedHV stdpre 128 | 126 | 1,730 |
| FSRS6 Cost ADR schedHV stdpre 128 | 127 | 1,365 |
| FSRS6 Cost ADR schedHV stdpre 128 | 128 | 2,385 |
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
| FSRS6 Cost ADR schedHV stdpre 128 | 128 | 1,375,699 |
| ADR Portfolio pop16 128 | 128 | 1,539,934 |

## Artifact Paths

- Report summary: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/report/report_summary.json`
- Run-local report: `artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off/report/report.md`
- Published report: `docs/rl_scheduler/experiments/2026-05-27-fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.md`
