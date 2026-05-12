# FSRS6 AP pop16 portfolio experiment report

Date: 2026-05-12

## Question

Train an FSRS6 AP portfolio with the same budget as
`fsrs6_adr_portfolio_users_1_8_pop16_v1`, then compare AP against ADR and the
shared FSRS6 baseline.

The matched portfolio budget is `population=16`, `offspring=16`,
`generations=20`, and `portfolio_size=16`.

## Runs

| run | config | scheduler | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_ap_portfolio_users_1_8_pop16_v1` | `fsrs6_ap_portfolio_users_1_8_pop16_v1.toml` | `fsrs6_ap` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `fsrs6_adr` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The AP run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_ap_portfolio_users_1_8_pop16_v1.toml \
  --run-id fsrs6_ap_portfolio_users_1_8_pop16_v1 \
  --skip-manifest \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Config commit: `2e9ce23eefd0e2e378e49267b530eed501791dda`
- Worktree recorded by artifacts: `dirty=false`
- CUDA device: NVIDIA GeForce RTX 4090 D
- AP training peak allocated CUDA memory: 178,057,728 bytes
- AP training peak reserved CUDA memory: 201,326,592 bytes
- AP sweep peak allocated CUDA memory: 1,381,069,312 bytes
- Observed Windows shared GPU memory samples were around 282-398 MiB and did not
  indicate VRAM spill.

## External Pareto Results

Hypervolume values are sums of per-user `HV delta` from `analyze-pareto`,
comparing the trained scheduler against the same FSRS6 baseline manifest.

| environment | run | HV delta sum | HV delta / baseline HV | avg memorized | avg time | avg efficiency | target frontier points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | AP pop16 | 92,877 | 3.338% | 6,556.5 | 52.74 | 24.97 | 128 |
| FSRS6 | ADR pop16 | 104,884 | 3.770% | 6,553.0 | 50.42 | 25.42 | 128 |
| LSTM | AP pop16 | 70,190 | 2.470% | 6,454.4 | 63.76 | 23.12 | 119 |
| LSTM | ADR pop16 | 73,610 | 2.591% | 6,431.5 | 60.86 | 22.88 | 122 |

AP minus ADR:

| environment | HV delta change | relative change | percent-point change | memorized change | time change | efficiency change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | -12,007 | -11.45% | -0.432 pp | +3.5 | +2.32 | -0.45 |
| LSTM | -3,419 | -4.65% | -0.120 pp | +22.9 | +2.90 | +0.24 |

Interpretation:

- AP pop16 does not beat ADR pop16 on hypervolume in either evaluation
  environment.
- The LSTM gap is small: AP is behind ADR by 3.4k HV, or 4.65% relative to ADR.
- AP has slightly higher average memorized counts, especially on LSTM, but spends
  more time. Hypervolume prefers the ADR frontier shape under this budget.
- FSRS6 is the clearer loss for AP: -12.0k HV and -0.432 percentage points.

## AP Policy Parameter Distribution

The AP run wrote 128 portfolio child policies: 8 users x 16 policies. Each policy
contains a desired retention plus 21 FSRS6 weight deltas from the user's fitted
baseline weights.

Selected AP desired-retention distribution:

| statistic | value |
| --- | ---: |
| count | 128 |
| mean | 0.8216 |
| std | 0.1378 |
| min | 0.5021 |
| p25 | 0.7604 |
| median | 0.8846 |
| p75 | 0.9153 |
| max | 0.9800 |

Selected desired-retention range by user:

| user | min DR | median DR | max DR | mean DR |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5448 | 0.8987 | 0.9420 | 0.8142 |
| 2 | 0.5294 | 0.8888 | 0.9800 | 0.8506 |
| 3 | 0.5478 | 0.8602 | 0.9510 | 0.7931 |
| 4 | 0.5302 | 0.8713 | 0.9541 | 0.8373 |
| 5 | 0.5912 | 0.8589 | 0.9353 | 0.8052 |
| 6 | 0.5806 | 0.9046 | 0.9767 | 0.8783 |
| 7 | 0.5021 | 0.6395 | 0.9414 | 0.6830 |
| 8 | 0.8839 | 0.9118 | 0.9555 | 0.9113 |

Largest AP weight movements by mean absolute delta:

| parameter | mean abs delta | delta std | p95 abs delta | search mean | search std | min weight | max weight | bound hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w3` | 6.872 | 8.902 | 16.808 | -0.078 | 0.788 | 0.001 | 80.147 | 19 |
| `w2` | 3.310 | 4.365 | 9.205 | -0.221 | 0.725 | 0.001 | 18.415 | 45 |
| `w1` | 2.595 | 3.580 | 8.393 | -0.048 | 1.037 | 0.001 | 18.080 | 45 |
| `w0` | 0.888 | 1.484 | 3.105 | -0.323 | 1.026 | 0.001 | 9.724 | 66 |
| `w16` | 0.338 | 0.425 | 0.873 | -0.234 | 0.886 | 1.000 | 4.966 | 9 |
| `w14` | 0.231 | 0.265 | 0.570 | +0.373 | 0.928 | 1.168 | 2.470 | 0 |
| `w6` | 0.208 | 0.247 | 0.578 | -0.323 | 0.825 | 2.232 | 3.881 | 0 |
| `w4` | 0.194 | 0.234 | 0.445 | +0.077 | 0.821 | 5.616 | 7.444 | 0 |
| `w8` | 0.164 | 0.198 | 0.390 | -0.231 | 1.018 | 1.168 | 2.594 | 0 |
| `w9` | 0.139 | 0.118 | 0.265 | +1.193 | 1.317 | 0.000 | 0.758 | 1 |

Parameters with any bound hits:

| parameter | bound hits | min weight | max weight | bounds |
| --- | ---: | ---: | ---: | --- |
| `w0` | 66 | 0.001 | 9.724 | `(0.001, 100.0)` |
| `w1` | 45 | 0.001 | 18.080 | `(0.001, 100.0)` |
| `w2` | 45 | 0.001 | 18.415 | `(0.001, 100.0)` |
| `w3` | 19 | 0.001 | 80.147 | `(0.001, 100.0)` |
| `w7` | 24 | 0.001 | 0.200 | `(0.001, 0.75)` |
| `w9` | 1 | 0.000 | 0.758 | `(0.0, 0.8)` |
| `w12` | 34 | 0.001 | 0.250 | `(0.001, 0.25)` |
| `w15` | 8 | 0.000 | 0.694 | `(0.0, 1.0)` |
| `w16` | 9 | 1.000 | 4.966 | `(1.0, 6.0)` |
| `w18` | 18 | 0.000 | 0.418 | `(0.0, 2.0)` |
| `w19` | 18 | 0.010 | 0.296 | `(0.01, 0.8)` |
| `w20` | 40 | 0.100 | 0.751 | `(0.1, 0.8)` |

The AP optimizer mostly moves the first four stability-related FSRS6 weights,
with frequent clipping to the lower bound for `w0`-`w3`. That suggests the
bounded AP search space is active, not just making tiny local adjustments.

## Training HV And Runtime

Training HV is measured in the FSRS6 training environment for both runs, so AP
and ADR are directly comparable as optimizer traces.

| run | final training HV gain sum | train elapsed seconds | train peak allocated CUDA memory | sweep elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| AP pop16 | 94,922.7 | 309.2 | 178,057,728 | 44.4 |
| ADR pop16 | 107,745.9 | 328.8 | 176,990,720 | 45.1 |

Training HV gain by generation window:

| run | generations 0-4 | generations 5-9 | generations 10-14 | generations 15-19 |
| --- | ---: | ---: | ---: | ---: |
| AP pop16 | 62,831.1 (66.2%) | 14,046.7 (14.8%) | 11,337.1 (11.9%) | 6,707.9 (7.1%) |
| ADR pop16 | 66,431.9 (61.7%) | 20,733.3 (19.2%) | 13,191.3 (12.2%) | 7,389.3 (6.9%) |

AP has a similar convergence shape to ADR: the final five generations still add
about 7% of final training HV. This run does not support reducing below 20
generations for AP.

## Conclusion

Under the matched pop16/off16/gen20 budget, AP is competitive but not better
than ADR.

- FSRS6 HV: AP is -12,007 behind ADR.
- LSTM HV: AP is -3,419 behind ADR.
- AP has slightly better LSTM average memorized and efficiency, but its frontier
  hypervolume remains lower.
- AP parameters show substantial movement and frequent bounds, so the result is
  not just a no-op version of FSRS6.

For the current objective, keep ADR pop16 as the stronger formal baseline. AP
may still be worth revisiting with a larger or differently constrained search
space, because the LSTM gap is small and its memorized/efficiency averages are
not worse.

## Artifact Paths

- AP pop16 run root:
  `artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/fsrs6_ap_portfolio_users_1_8_pop16_v1`
- AP pop16 analysis:
  `artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/fsrs6_ap_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- ADR pop16 comparison analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
