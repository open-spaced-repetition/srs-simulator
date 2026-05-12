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

Hypervolume values below are the sum of per-user `HV delta` from
`analyze-pareto`, comparing FSRS6 ADR against the FSRS6 baseline under the same
low-budget baseline DR manifest.

| environment | run | HV delta sum | HV delta / baseline HV | ADR avg memorized | ADR avg time | ADR avg efficiency | ADR frontier points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | v3 64/64 gen20 | 114,353 | 4.110% | 6,497.3 | 47.97 | 26.84 | 128 |
| FSRS6 | pop16 gen20 | 104,884 | 3.770% | 6,553.0 | 50.42 | 25.42 | 128 |
| FSRS6 | pop16 gen30 | 111,226 | 3.998% | 6,547.7 | 47.96 | 25.63 | 128 |
| LSTM | v3 64/64 gen20 | 68,782 | 2.421% | 6,364.3 | 58.93 | 24.05 | 117 |
| LSTM | pop16 gen20 | 73,610 | 2.591% | 6,431.5 | 60.86 | 22.88 | 122 |
| LSTM | pop16 gen30 | 70,314 | 2.475% | 6,410.0 | 58.68 | 22.84 | 119 |

Delta from pop16 gen20 to pop16 gen30:

| environment | HV delta sum change | HV delta percent change | avg memorized change | avg time change | frontier point change |
| --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | +6,343 | +6.05% | -5.3 | -2.46 | 0 |
| LSTM | -3,295 | -4.48% | -21.5 | -2.18 | -3 |

Interpretation:

- Increasing pop16 from 20 to 30 generations improved the FSRS6 external Pareto
  HV, and reduced average time versus pop16 gen20.
- The same change hurt the LSTM external Pareto HV, which is the metric we care
  about most for this decision.
- The LSTM result for pop16 gen30 remains above the 64/64 v3 run, but it loses
  much of the advantage that pop16 gen20 had.

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

- pop16 gen20 LSTM HV delta sum: 73,610
- pop16 gen30 LSTM HV delta sum: 70,314
- regression: -3,295 HV, or -4.48%

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
