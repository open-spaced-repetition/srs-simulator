# FSRS6 ADR pop16 gen10 experiment report

Date: 2026-05-12

## Question

Evaluate whether the pop16/off16 FSRS6 ADR portfolio experiment can reduce the
SMS-EMOA generation count from 20 to 10 while preserving the LSTM-environment
Pareto improvement.

## Runs

| run | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- |
| `fsrs6_adr_portfolio_users_1_8_v3_baseline16x5` | `fsrs6_adr_portfolio_users_1_8_v3.toml` snapshot | `population=64`, `offspring=64`, `generations=20` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1.toml` | `population=16`, `offspring=16`, `generations=10` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `population=16`, `offspring=16`, `generations=20` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.toml` | `population=16`, `offspring=16`, `generations=30` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The gen10 run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1.toml \
  --run-id fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1 \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Config commit: `86687380cf2ffc3e789bd6389a8892ebff15ebaf`
- Worktree recorded by artifacts: `dirty=false`
- CUDA device: NVIDIA GeForce RTX 4090 D
- Training peak allocated CUDA memory: 176,737,792 bytes
- Training peak reserved CUDA memory: 201,326,592 bytes
- Sweep peak allocated CUDA memory: 1,332,619,776 bytes
- Sweep peak reserved CUDA memory: 1,363,148,800 bytes
- Observed Windows shared GPU memory stayed around 81-165 MiB during checks and
  did not indicate VRAM spill.

## External Pareto Results

Scheduler-only hypervolume values below are the sum of per-user `HV delta` from
`analyze-pareto`, comparing FSRS6 ADR against the FSRS6 baseline under the same
low-budget baseline DR manifest. Average memorized/time/efficiency are now
diagnostic-only in `analysis.md`.

| environment | run | HV delta sum | HV delta / baseline HV | HV delta median | HV delta max | scheduler frontier points |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | v3 64/64 gen20 | 109,224 | 3.926% | 7,392 | 43,351 | 128 |
| FSRS6 | pop16 gen10 | 74,882 | 2.691% | 5,583 | 25,235 | 127 |
| FSRS6 | pop16 gen20 | 96,880 | 3.482% | 6,327 | 37,079 | 128 |
| FSRS6 | pop16 gen30 | 104,360 | 3.751% | 6,523 | 40,771 | 128 |
| LSTM | v3 64/64 gen20 | 47,097 | 1.658% | 3,592 | 13,120 | 125 |
| LSTM | pop16 gen10 | 42,718 | 1.503% | 2,981 | 14,606 | 125 |
| LSTM | pop16 gen20 | 55,849 | 1.966% | 4,110 | 19,119 | 125 |
| LSTM | pop16 gen30 | 52,781 | 1.858% | 3,924 | 15,470 | 125 |

Target LSTM same-budget memory lift AUC, target scheduler rows only. It uses linear
interpolation over the common covered time-budget interval. Positive values mean
the scheduler remembers more cards at the same budget. Relative same-budget memory lift AUC is the
simple average of each user's same-budget memory lift AUC divided by that user's covered
baseline memory AUC.

| run | AUC users | budget coverage | span coverage | same-budget memory lift AUC | relative same-budget memory lift AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 64/64 gen20 | 8/8 | 83/111 | 83.110% | +60.8 | +0.933% |
| pop16 gen10 | 8/8 | 82/111 | 82.927% | +35.4 | +0.533% |
| pop16 gen20 | 8/8 | 79/111 | 82.466% | +62.2 | +0.953% |
| pop16 gen30 | 8/8 | 80/111 | 82.451% | +58.2 | +0.886% |

Target LSTM same-target time saved AUC, target scheduler rows only. Positive
values mean the scheduler reaches the same memorized-card targets faster. It
uses linear interpolation over the common covered memory-target interval.

| run | AUC users | target coverage | span coverage | same-target time saved AUC | relative same-target time saved AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 64/64 gen20 | 8/8 | 89/111 | 88.298% | +1.32 | +5.589% |
| pop16 gen10 | 8/8 | 83/111 | 83.264% | +0.91 | +3.425% |
| pop16 gen20 | 8/8 | 81/111 | 83.775% | +1.82 | +5.906% |
| pop16 gen30 | 8/8 | 86/111 | 86.052% | +1.64 | +5.674% |

Delta from pop16 gen10 to pop16 gen20:

| environment | HV delta sum change | HV delta percent-point change | same-budget memory lift AUC change | same-target time saved AUC change |
| --- | ---: | ---: | ---: | ---: |
| FSRS6 | +21,998 | +0.791 pp | +24.1 | +1.47 |
| LSTM | +13,131 | +0.462 pp | +26.8 | +0.91 |

Interpretation:

- Gen10 is materially undertrained versus gen20 on the target LSTM metric.
- The earlier average-efficiency comparison was a policy-point diagnostic; the
  more relevant LSTM same-budget memory lift AUC also favors gen20 by +26.8 memorized cards.
- Gen10's corrected same-target time saved AUC is now favorable versus the FSRS6
  baseline, but gen20 still reaches the same targets 0.91 minutes faster.
- Gen10 also falls below the comparable 64/64 v3 run on LSTM HV and AUC, while
  gen20 remains the strongest target-LSTM run in this comparison.

## Training HV by Generation

Training HV is the SMS-EMOA objective evaluated in the FSRS6 training
environment. It is useful for judging optimizer convergence, but it is not a
direct substitute for LSTM external validation.

| run | final training HV gain | train elapsed seconds | peak allocated CUDA memory |
| --- | ---: | ---: | ---: |
| v3 64/64 gen20 | 144,704 | 392.9 | 714,942,976 |
| pop16 gen10 | 87,165 | 178.9 | 176,737,792 |
| pop16 gen20 | 107,746 | 328.8 | 176,990,720 |
| pop16 gen30 | 115,389 | 481.0 | 177,866,752 |

pop16 gen10 training HV contribution by generation window:

| generations | HV gain | share of final training gain | avg gain per generation |
| --- | ---: | ---: | ---: |
| 0-4 | 66,432 | 76.2% | 13,286 |
| 5-9 | 20,733 | 23.8% | 4,147 |

Last five pop16 gen10 generations:

| generation | cumulative training HV gain | generation gain | gain / final gain | users with gain |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 71,185 | 4,753 | 5.45% | 8 |
| 6 | 74,668 | 3,483 | 4.00% | 8 |
| 7 | 80,437 | 5,769 | 6.62% | 8 |
| 8 | 83,210 | 2,773 | 3.18% | 8 |
| 9 | 87,165 | 3,956 | 4.54% | 8 |

The last generation still added 4.54% of the final gen10 training HV gain, and
every user improved in each of the last five generations. That is a clear sign
that 10 generations stops well before the pop16 optimizer has stabilized.

## Conclusion

Do not promote 10 generations for the LSTM-focused pop16/off16 experiment.

The 10-generation run is faster, but it cuts too much search:

- LSTM HV delta sum: 42,718 for gen10 vs 55,849 for gen20
- LSTM regression from gen20: -13,131 HV, or -23.51%
- LSTM same-budget memory lift AUC: +35.4 for gen10 vs +62.2 for gen20
- LSTM same-target time saved AUC: +0.91 for gen10 vs +1.82 for gen20
- LSTM result also falls below the comparable 64/64 v3 run by -4,379 HV
- Training HV is still rising quickly at generation 9

The current best setting remains pop16/off16 with 20 generations. The 30-
generation run showed that more FSRS6 training HV does not necessarily improve
LSTM external HV, while this 10-generation run shows that reducing to 10
generations underfits. The useful search interval is therefore around 20
generations for the current objective and data.

## Artifact Paths

- Gen10 run root:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1`
- Gen10 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Pop16 gen20 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Pop16 gen30 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Comparable v3 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8_baseline16x5/fsrs6_adr_portfolio_users_1_8_v3_baseline16x5/analyze-pareto/analyze_pareto_outputs/analysis.md`
