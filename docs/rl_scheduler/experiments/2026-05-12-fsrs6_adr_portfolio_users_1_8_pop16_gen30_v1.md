# FSRS6 ADR pop16 gen30 experiment report

Date: 2026-05-12

## Question

Evaluate whether the pop16/off16 FSRS6 ADR portfolio experiment should use more
than 20 SMS-EMOA generations, with LSTM-environment Pareto improvement as the
primary decision signal.

## Runs

| run | config | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- |
| `fsrs6_adr_portfolio_users_1_8_v3_baseline16x5` | `fsrs6_adr_portfolio_users_1_8_v3.toml` snapshot | `population=64`, `offspring=64`, `generations=20` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `population=16`, `offspring=16`, `generations=20` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.toml` | `population=16`, `offspring=16`, `generations=30` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The gen30 run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.toml \
  --run-id fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1 \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Config commit: `f98fed75e202657d3e3591341879195f19b42884`
- Worktree recorded by artifacts: `dirty=false`
- CUDA device: NVIDIA GeForce RTX 4090 D
- Training peak allocated CUDA memory: 177,866,752 bytes
- Training peak reserved CUDA memory: 203,423,744 bytes
- Sweep peak allocated CUDA memory: 1,309,124,608 bytes
- Sweep peak reserved CUDA memory: 1,344,274,432 bytes
- Observed Windows shared GPU memory stayed around 80-170 MiB during training
  checks and did not indicate VRAM spill.

## External Pareto Results

Scheduler-only hypervolume values below are the sum of per-user `HV delta` from
`analyze-pareto`, comparing FSRS6 ADR against the FSRS6 baseline under the same
low-budget baseline DR manifest. Average memorized/time/efficiency are now
diagnostic-only in `analysis.md`.

| environment | run | HV delta sum | HV delta / baseline HV | HV delta median | HV delta max | scheduler frontier points |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | v3 64/64 gen20 | 109,224 | 3.926% | 7,392 | 43,351 | 128 |
| FSRS6 | pop16 gen20 | 96,880 | 3.482% | 6,327 | 37,079 | 128 |
| FSRS6 | pop16 gen30 | 104,360 | 3.751% | 6,523 | 40,771 | 128 |
| LSTM | v3 64/64 gen20 | 47,097 | 1.658% | 3,592 | 13,120 | 125 |
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
| pop16 gen20 | 8/8 | 79/111 | 82.466% | +62.2 | +0.953% |
| pop16 gen30 | 8/8 | 80/111 | 82.451% | +58.2 | +0.886% |

Target LSTM same-target time saved AUC, target scheduler rows only. Positive
values mean the scheduler reaches the same memorized-card targets faster. It
uses linear interpolation over the common covered memory-target interval.

| run | AUC users | target coverage | span coverage | same-target time saved AUC | relative same-target time saved AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 64/64 gen20 | 8/8 | 89/111 | 88.298% | +1.32 | +5.589% |
| pop16 gen20 | 8/8 | 81/111 | 83.775% | +1.82 | +5.906% |
| pop16 gen30 | 8/8 | 86/111 | 86.052% | +1.64 | +5.674% |

Delta from pop16 gen20 to pop16 gen30:

| environment | HV delta sum change | HV delta percent change | same-budget memory lift AUC change | same-target time saved AUC change | scheduler frontier point change |
| --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | +7,480 | +7.72% | +3.2 | +0.32 | 0 |
| LSTM | -3,068 | -5.49% | -3.9 | -0.18 | 0 |

Interpretation:

- Increasing pop16 from 20 to 30 generations improved the FSRS6 external Pareto
  HV and improved the FSRS6 interpolated AUCs.
- The same change hurt the LSTM external Pareto HV, which is the metric we care
  about most for this decision. Under interpolation, LSTM same-budget memory lift AUC
  worsens by 3.9 memorized cards, and same-target time saved AUC worsens by 0.18
  minutes.
- The LSTM result for pop16 gen30 remains above the 64/64 v3 run, but it loses
  much of the HV advantage that pop16 gen20 had and no longer improves the
  corrected LSTM AUCs.

## Training HV by Generation

Training HV is the SMS-EMOA objective evaluated in the FSRS6 training
environment. It is useful for judging optimizer convergence, but it is not a
direct substitute for LSTM external validation.

| run | final training HV gain | train elapsed seconds | peak allocated CUDA memory |
| --- | ---: | ---: | ---: |
| v3 64/64 gen20 | 144,704 | 392.9 | 714,942,976 |
| pop16 gen20 | 107,746 | 328.8 | 176,990,720 |
| pop16 gen30 | 115,389 | 481.0 | 177,866,752 |

pop16 gen30 training HV contribution by generation window:

| generations | HV gain | share of final training gain | avg gain per generation |
| --- | ---: | ---: | ---: |
| 0-4 | 66,432 | 57.6% | 13,286 |
| 5-9 | 20,733 | 18.0% | 4,147 |
| 10-14 | 13,191 | 11.4% | 2,638 |
| 15-19 | 7,389 | 6.4% | 1,478 |
| 20-24 | 5,324 | 4.6% | 1,065 |
| 25-29 | 2,319 | 2.0% | 464 |
| 20-29 | 7,643 | 6.6% | 764 |

Last five pop16 gen30 generations:

| generation | cumulative training HV gain | generation gain | gain / previous cumulative | gain / final gain | users with gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 113,167 | 97 | 0.085% | 0.084% | 7 |
| 26 | 113,296 | 129 | 0.114% | 0.112% | 7 |
| 27 | 114,140 | 845 | 0.745% | 0.732% | 5 |
| 28 | 114,960 | 820 | 0.719% | 0.711% | 7 |
| 29 | 115,389 | 429 | 0.373% | 0.372% | 5 |

The 30-generation run is closer to convergence than the 20-generation run, but
it is still not perfectly flat: generations 25-29 add about 2.0% of final
training HV gain. However, that extra FSRS6 training HV did not translate into
better LSTM Pareto HV.

## Conclusion

Do not promote 30 generations as the formal pop16 default if LSTM improvement is
the primary objective. The 30-generation run is useful evidence that pop16 can
keep improving the FSRS6 training objective after 20 generations, but the LSTM
external validation regressed relative to pop16 gen20:

- pop16 gen20 LSTM HV delta sum: 55,849
- pop16 gen30 LSTM HV delta sum: 52,781
- regression: -3,068 HV, or -5.49%
- LSTM same-budget memory lift AUC also falls from +62.2 to +58.2.
- LSTM same-target time saved AUC weakens from +1.82 to +1.64 minutes.

The current best setting for the LSTM-focused comparison remains pop16/off16
with 20 generations. A 25-generation run could still be tested as a compromise,
but 30 generations did not improve the target LSTM result in this run.

## Artifact Paths

- Gen30 run root:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1`
- Gen30 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Pop16 gen20 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- Comparable v3 analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8_baseline16x5/fsrs6_adr_portfolio_users_1_8_v3_baseline16x5/analyze-pareto/analyze_pareto_outputs/analysis.md`
