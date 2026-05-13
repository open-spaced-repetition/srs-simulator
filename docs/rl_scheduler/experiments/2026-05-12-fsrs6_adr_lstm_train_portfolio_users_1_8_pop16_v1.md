# FSRS6 ADR LSTM-trained pop16 policy comparison

Date: 2026-05-12

## Question

Train FSRS6 ADR portfolio policies in the LSTM environment, using the same
budget as `fsrs6_adr_portfolio_users_1_8_pop16_v1`, then compare the learned
policy parameter distribution against the FSRS6-environment-trained policies.

The main target is external LSTM-environment Pareto improvement.

## Runs

| run | config | training env | portfolio budget | baseline DR manifest |
| --- | --- | --- | --- | --- |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `fsrs6` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |
| `fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1` | `fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml` | `lstm` | `population=16`, `offspring=16`, `generations=20`, `portfolio=16` | `fsrs6_users_1_8_16dr_pop16_gen5.json` |

The LSTM-trained run was executed with:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml \
  --run-id fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1 \
  --skip-manifest \
  --skip-baseline-sweep
```

All formal stages passed: `dry-run`, `preflight`, `stage-baseline`,
`train-overfit`, `sweep`, `build-pareto`, and `analyze-pareto`.

Provenance:

- Config commit: `72e17d8e482a4965efe27cd5b6fd7a98ca27949f`
- Worktree recorded by artifacts: `dirty=false`
- CUDA device: NVIDIA GeForce RTX 4090 D
- LSTM-trained training peak allocated CUDA memory: 1,611,250,688 bytes
- LSTM-trained training peak reserved CUDA memory: 1,660,944,384 bytes
- LSTM-trained sweep peak allocated CUDA memory: 1,611,250,688 bytes
- Observed Windows shared GPU memory samples stayed around 64-165 MiB and did
  not indicate VRAM spill.

## External Pareto Results

Scheduler-only hypervolume values are sums of per-user `HV delta` from
`analyze-pareto`, comparing FSRS6 ADR against the same low-budget FSRS6
baseline manifest. Policy point averages are now treated as diagnostics in the
generated analysis, not as primary Pareto quality metrics.

| environment | run | HV delta sum | HV delta / baseline HV | HV delta min | p25 | median | p75 | max | scheduler frontier points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | FSRS-trained pop16 gen20 | 96,880 | 3.482% | 1,238 | 1,918 | 6,327 | 9,894 | 37,079 | 128 |
| FSRS6 | LSTM-trained pop16 gen20 | 54,706 | 1.966% | 1,205 | 1,320 | 3,497 | 4,638 | 23,991 | 124 |
| LSTM | FSRS-trained pop16 gen20 | 55,849 | 1.966% | 733 | 1,556 | 4,110 | 7,161 | 19,119 | 125 |
| LSTM | LSTM-trained pop16 gen20 | 91,189 | 3.209% | 1,089 | 1,547 | 3,728 | 13,221 | 38,095 | 128 |

Budget-memory gain AUC integrates memorized-card gain over all FSRS6-baseline
frontier time budgets per user, using linear interpolation between each
scheduler's Pareto frontier points. Positive values mean the scheduler remembers
more cards at the same budget.

| environment | run | AUC users | budget coverage | span coverage | memory gain AUC | relative gain AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | FSRS-trained | 8/8 | 101/115 | 98.365% | +65.4 | +0.954% |
| FSRS6 | LSTM-trained | 8/8 | 98/115 | 97.778% | +39.0 | +0.570% |
| LSTM | FSRS-trained | 8/8 | 92/111 | 98.428% | +28.1 | +0.414% |
| LSTM | LSTM-trained | 8/8 | 93/111 | 97.837% | +43.2 | +0.637% |

Memory-target regret AUC integrates time regret over all FSRS6-baseline frontier
memory targets per user, using linear interpolation between each scheduler's
Pareto frontier points. Negative values mean the scheduler reaches the same
memorized-card targets faster.

| environment | run | AUC users | target coverage | span coverage | time regret AUC | relative regret AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | FSRS-trained | 8/8 | 106/115 | 98.094% | -2.92 | -10.591% |
| FSRS6 | LSTM-trained | 8/8 | 105/115 | 97.773% | -1.04 | -3.744% |
| LSTM | FSRS-trained | 8/8 | 102/111 | 97.854% | -1.02 | -2.599% |
| LSTM | LSTM-trained | 8/8 | 103/111 | 97.573% | -4.42 | -11.278% |

LSTM-trained minus FSRS-trained:

| environment | HV delta change | relative change | percent-point change | budget-gain AUC change | target-regret AUC change |
| --- | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | -42,174 | -43.53% | -1.516 pp | -26.4 | +1.88 |
| LSTM | +35,340 | +63.28% | +1.244 pp | +15.1 | -3.40 |

Interpretation:

- Training directly in LSTM substantially improves the target LSTM external HV:
  +35.3k HV, or +63.3% relative to the FSRS-trained pop16 run.
- The gain trades off FSRS6 external performance: FSRS6 HV drops by 42.2k.
- On the LSTM external envelope, LSTM training moves interpolated budget-gain
  AUC from +28.1 to +43.2 memorized cards and target-regret AUC from -1.02
  minutes to -4.42 minutes.

## Policy Parameter Distribution

Both runs wrote 128 portfolio child policies: 8 users x 16 policies. All policies
use `fsrs6_adr_log_poly_v1` with six coefficients:

`c0 + c_s * S_norm + c_d * D_norm + c_s*d * S_norm * D_norm + c_s2 * S_norm^2 + c_d2 * D_norm^2`

Global coefficient distribution:

| coef | FSRS mean | LSTM mean | mean diff | FSRS median | LSTM median | median diff | FSRS std | LSTM std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c0` | 1.120 | 0.924 | -0.197 | 1.182 | 1.099 | -0.082 | 0.792 | 1.210 |
| `c_s` | 0.771 | 0.372 | -0.399 | 0.472 | 0.391 | -0.081 | 1.448 | 1.153 |
| `c_d` | -0.453 | -0.282 | +0.171 | -0.388 | -0.293 | +0.095 | 1.126 | 0.990 |
| `c_s*d` | 0.275 | 0.175 | -0.100 | 0.265 | 0.178 | -0.087 | 0.841 | 0.937 |
| `c_s2` | 0.691 | 0.771 | +0.080 | 0.683 | 0.843 | +0.160 | 1.020 | 1.261 |
| `c_d2` | -0.665 | -0.216 | +0.449 | -0.546 | -0.122 | +0.424 | 1.284 | 0.790 |

LSTM-trained coefficient quantiles:

| coef | min | p25 | median | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c0` | -3.227 | 0.217 | 1.099 | 1.717 | 3.290 |
| `c_s` | -3.124 | -0.212 | 0.391 | 1.140 | 2.381 |
| `c_d` | -3.362 | -0.858 | -0.293 | 0.182 | 3.168 |
| `c_s*d` | -1.915 | -0.492 | 0.178 | 0.815 | 2.342 |
| `c_s2` | -3.065 | -0.224 | 0.843 | 1.597 | 3.570 |
| `c_d2` | -2.293 | -0.732 | -0.122 | 0.410 | 1.331 |

Per-user mean coefficient vector distance, LSTM-trained minus FSRS-trained:

| user | mean-vector L2 | d_c0 | d_c_s | d_c_d | d_c_s*d | d_c_s2 | d_c_d2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.169 | +1.296 | -1.205 | -0.277 | -0.826 | -0.882 | +0.182 |
| 2 | 2.317 | -0.493 | -1.291 | +0.678 | +0.327 | +1.581 | +0.628 |
| 3 | 1.237 | -1.006 | +0.552 | +0.288 | -0.088 | +0.330 | +0.122 |
| 4 | 1.712 | +0.607 | -1.110 | +0.618 | -0.559 | -0.754 | -0.262 |
| 5 | 1.989 | -0.852 | +0.448 | +0.602 | -0.779 | +0.792 | +1.198 |
| 6 | 2.301 | -0.486 | -0.735 | -0.544 | +1.634 | +0.780 | +0.972 |
| 7 | 2.030 | -0.246 | -0.901 | +1.351 | -0.257 | -0.169 | +1.153 |
| 8 | 2.090 | -0.397 | +1.049 | -1.348 | -0.251 | -1.034 | -0.402 |

The clearest global differences are lower `c_s`, less negative `c_d2`, and a
slightly lower intercept for LSTM-trained policies. User-level shifts are not a
uniform translation; every user has a visibly different coefficient mean vector.

## Policy Output Surface Summary

To make the coefficients interpretable, each policy was evaluated on a fixed
grid:

- Stability `S`: `0.1`, `1`, `7`, `30`, `180`, `1000`, `9125`
- Difficulty `D`: `1`, `3`, `5`, `7`, `10`

Across all 4,480 policy/grid evaluations:

| run | mean DR | std | min | p25 | median | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS-trained | 0.8460 | 0.1287 | 0.5001 | 0.7841 | 0.8904 | 0.9462 | 0.9798 |
| LSTM-trained | 0.8360 | 0.1308 | 0.5015 | 0.7689 | 0.8824 | 0.9372 | 0.9795 |

Average output DR by stability, averaged across difficulties:

| S | FSRS-trained avg DR | LSTM-trained avg DR | diff |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.8103 | 0.8127 | +0.0024 |
| 1 | 0.8279 | 0.8213 | -0.0066 |
| 7 | 0.8415 | 0.8300 | -0.0115 |
| 30 | 0.8501 | 0.8366 | -0.0135 |
| 180 | 0.8584 | 0.8441 | -0.0143 |
| 1000 | 0.8643 | 0.8505 | -0.0138 |
| 9125 | 0.8696 | 0.8571 | -0.0124 |

Average output DR by difficulty, averaged across stabilities:

| D | FSRS-trained avg DR | LSTM-trained avg DR | diff |
| ---: | ---: | ---: | ---: |
| 1 | 0.8739 | 0.8507 | -0.0232 |
| 3 | 0.8661 | 0.8460 | -0.0201 |
| 5 | 0.8534 | 0.8394 | -0.0140 |
| 7 | 0.8357 | 0.8307 | -0.0050 |
| 10 | 0.8009 | 0.8135 | +0.0125 |

The LSTM-trained policies are slightly lower on average, especially for easy to
medium difficulty, but are higher at the hardest difficulty level. That matches
the coefficient shift: difficulty curvature becomes much less negative.

## Training HV And Runtime

Training HV is measured in each run's own training environment, so FSRS-trained
and LSTM-trained HV values are not directly the same objective. They are useful
for optimizer convergence and budget parity checks.

| run | final training HV gain sum | train elapsed seconds | train peak allocated CUDA memory | sweep elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| FSRS-trained pop16 gen20 | 107,745.9 | 328.8 | 176,990,720 | 45.1 |
| LSTM-trained pop16 gen20 | 110,627.3 | 535.8 | 1,611,250,688 | 44.3 |

Training HV gain by generation window:

| run | generations 0-4 | generations 5-9 | generations 10-14 | generations 15-19 |
| --- | ---: | ---: | ---: | ---: |
| FSRS-trained | 66,431.9 (61.7%) | 20,733.3 (19.2%) | 13,191.3 (12.2%) | 7,389.3 (6.9%) |
| LSTM-trained | 72,608.7 (65.6%) | 17,442.1 (15.8%) | 13,848.9 (12.5%) | 6,727.5 (6.1%) |

The LSTM-trained run has a similar convergence shape to the FSRS-trained run:
most HV arrives in the first 10 generations, but the final 5 generations still
contribute about 6% of final training HV. This does not suggest cutting below
20 generations for this comparison.

## Conclusion

For the stated priority, LSTM-trained FSRS6 ADR is better than the FSRS-trained
pop16 baseline.

- Target LSTM HV improves from 55,849 to 91,189, a +35,340 gain.
- Relative LSTM HV improvement increases from 1.966% to 3.209%.
- FSRS6 HV regresses from 96,880 to 54,706, so the learned policy is more
  environment-specific.
- Policy parameters differ materially, especially lower stability slope
  (`c_s`) and much less negative difficulty curvature (`c_d2`).

Use this LSTM-trained variant when LSTM external validation is the primary
objective. Keep the FSRS-trained variant when cross-environment FSRS6 robustness
matters.

## Artifact Paths

- LSTM-trained run root:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1`
- LSTM-trained analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
- FSRS-trained comparison analysis:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1/analyze-pareto/analyze_pareto_outputs/analysis.md`
