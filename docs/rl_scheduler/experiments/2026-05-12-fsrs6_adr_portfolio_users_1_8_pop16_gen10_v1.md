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

Hypervolume values below are the sum of per-user `HV delta` from
`analyze-pareto`, comparing FSRS6 ADR against the FSRS6 baseline under the same
low-budget baseline DR manifest.

| environment | run | HV delta sum | HV delta / baseline HV | ADR avg memorized | ADR avg time | ADR avg efficiency | ADR frontier points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | v3 64/64 gen20 | 114,353 | 4.110% | 6,497.3 | 47.97 | 26.84 | 128 |
| FSRS6 | pop16 gen10 | 85,125 | 3.059% | 6,493.1 | 49.76 | 26.58 | 127 |
| FSRS6 | pop16 gen20 | 104,884 | 3.770% | 6,553.0 | 50.42 | 25.42 | 128 |
| FSRS6 | pop16 gen30 | 111,226 | 3.998% | 6,547.7 | 47.96 | 25.63 | 128 |
| LSTM | v3 64/64 gen20 | 68,782 | 2.421% | 6,364.3 | 58.93 | 24.05 | 117 |
| LSTM | pop16 gen10 | 66,041 | 2.324% | 6,384.1 | 59.46 | 24.23 | 121 |
| LSTM | pop16 gen20 | 73,610 | 2.591% | 6,431.5 | 60.86 | 22.88 | 122 |
| LSTM | pop16 gen30 | 70,314 | 2.475% | 6,410.0 | 58.68 | 22.84 | 119 |

Delta from pop16 gen10 to pop16 gen20:

| environment | HV delta sum change | HV delta percent-point change | avg memorized change | avg time change | avg efficiency change |
| --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | +19,759 | +0.710 pp | +59.9 | +0.66 | -1.16 |
| LSTM | +7,569 | +0.266 pp | +47.4 | +1.40 | -1.35 |

Interpretation:

- Gen10 is materially undertrained versus gen20 on the target LSTM metric.
- Gen10 has higher LSTM average efficiency than gen20 because it learns a more
  conservative frontier, but it gives up substantial LSTM HV and memorized
  cards.
- Gen10 also falls below the comparable 64/64 v3 run on LSTM HV, while gen20
  and gen30 both remain above it.

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

- LSTM HV delta sum: 66,041 for gen10 vs 73,610 for gen20
- LSTM regression from gen20: -7,569 HV, or -10.28%
- LSTM result also falls below the comparable 64/64 v3 run by -2,741 HV
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
