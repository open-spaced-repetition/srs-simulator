# Single-Card Tradeoff Experiments

This experiment family simulates an iid single-card lifecycle with no daily study-budget constraints. Card-level metrics are linearly scaled to a 10,000-card deck so scheduler frontiers can be compared by expected memorized cards and study minutes per day.

## Quickstart

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_default --particles 10000 --deck-scale 10000
```

When CUDA is available, `tradeoff.py` uses `cuda` by default; pass `--torch-device cpu` to force CPU.

Pass `--env fsrs6 --user-id <id>` to load per-user FSRS-6 weights from `../srs-benchmark`; add `--button-usage ../Anki-button-usage/button_usage.jsonl` to use that user's first/review rating probabilities and learning/review costs in the single-card oracle, PPO, and distillation rollouts. `--env fsrs6_default` keeps the built-in FSRS-6 parameters and default costs.

For supported FSRS-6 sweeps, desired-retention targets are batched in one vectorized run by default. The default targets are `0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`; override with `--target-retentions`, or pass `--target-retentions ""` to use the range flags. Fixed-interval sweeps are batched the same way. Plain `--sched fixed` runs intervals `8,16,32,64,128,256,512` by default; override with `--fixed-intervals`. Mixed scheduler families are run as one batch per family. Pass `--target-batch-size 1` to run targets/intervals sequentially.

By default, the script writes a pairwise memory-target regret AUC CSV next to the main CSV. `time_regret_auc` is the average extra deck-scaled minutes/day needed by the scheduler versus the baseline over their common covered memory-target interval, and `relative_regret_auc_percent` divides that by the baseline time AUC.

## UVFA PPO

UVFA PPO single-card experiment, goal-conditioned over FSRS-6 target-retention actions:

```bash
uv run experiments/single_card_tradeoff/uvfa_ppo.py --days 1825 --eval-particles 10000 --deck-scale 10000
```

The PPO objective is `card_expected_retrievability - goal_cost_weight * card_minutes_per_day`, with default training goal weights `16,32,64,128,256,512,1024`; the standard tradeoff sweep evaluates scalarized learned policies and oracles at `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024` by default. It normalizes advantages per goal, uses rich state features and a residual policy/value network with hidden size 64 and depth 3 by default, and uses a finite-horizon FSRS grid oracle as the default warmup/regularization guide. Pass `--guide-policy static` for the older static-FSRS target prior, or `--guide-policy none` for plain PPO. The script writes a comparable CSV/plot under `artifacts/single_card_tradeoff/`, includes fixed-interval and static-FSRS reference curves, and reports whether the learned UVFA policy beats the selected baseline. The default pass/fail baseline is the best fixed interval; use `--baseline fsrs` or `--baseline overall` for stricter static-FSRS comparisons.

After training a policy, include it in the standard single-card Pareto sweep with `--sched uvfa_ppo`:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_default,fixed,uvfa_ppo --uvfa-ppo-policy artifacts/single_card_tradeoff/uvfa_ppo_policy.pt
```

Search UVFA PPO model-scale hyperparameters:

```bash
uv run experiments/single_card_tradeoff/uvfa_ppo_hparam_search.py --days 1825 --eval-particles 3000 --save-models
```

## Recurrent Interval PPO

Recurrent UVFA PPO over continuous log-interval actions:

```bash
uv run experiments/single_card_tradeoff/uvfa_ppo_rnn_interval.py --days 1825 --eval-particles 10000 --deck-scale 10000
```

This variant uses a GRU belief-state encoder over the event observation sequence, concatenates the hidden state with the sampled cost-weight preference, and trains Gaussian PPO in log days. The environment exponentiates the sampled action, rounds it to physical review days, and clamps the interval to `--max-interval-days` (default `days * 4`). Its belief observation does not expose the simulator's internal stability or difficulty state. By default, training uses a finite-horizon FSRS grid oracle as a continuous log-interval warmup and PPO regularization guide; pass `--guide-policy static` or `--guide-policy none` to ablate it.

After training a recurrent interval policy, include it in the standard single-card Pareto sweep with `--sched uvfa_ppo_rnn_interval`:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_default,fixed,uvfa_ppo_rnn_interval --uvfa-ppo-rnn-interval-policy artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_policy.pt
```

## Discrete Oracle Distillation

Train a pure FSRS-6 oracle distillation baseline, then compare it directly with the DP oracle and UVFA PPO:

```bash
uv run experiments/single_card_tradeoff/oracle_distill.py --days 1825 --eval-particles 10000 --deck-scale 10000
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle,fsrs6_oracle_distill,uvfa_ppo --oracle-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_distill_policy.pt --uvfa-ppo-policy artifacts/single_card_tradeoff/uvfa_ppo_policy.pt
```

`oracle_distill.py` defaults to the `oracle_rho4` observation (`log remaining/stability ratio`, difficulty, goal cost weight, and stability), the `residual:16:2` architecture, and CUDA when available. Its default teacher weights are `0,1,2,4,8,16,32,64,128,256,512,1024`, so the zero-cost edge is trained directly while the standard tradeoff evaluation still probes intermediate weights. This model has 1,536 parameters, about 30% of the previous `residual:32:2` default. In the default FSRS-6 10k-particle, 3-seed comparison it slightly improved pairwise time-regret AUC against the previous default (`-0.0587` deck-minutes/day with 100% overlap) while preserving the broader frontier coverage that short small-model training missed. Pass `--obs-mode oracle` to train the older 4-feature oracle observation, or `--obs-mode rich` to train on the larger rollout observation instead. The script also accepts the same `--env fsrs6 --user-id <id>` and `--button-usage` options as the single-card tradeoff runner.

To rerun the discrete oracle distillation model-size search:

```bash
uv run experiments/single_card_tradeoff/oracle_distill_hparam_search.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --eval-particles 3000 --save-models
```

## Grid Oracle

Estimate a finite-horizon FSRS-6 grid oracle frontier:

```bash
uv run experiments/single_card_tradeoff/oracle_frontier.py --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Use the same finite-horizon oracle policy table as a scheduler in the standard single-card sweep:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle --oracle-cost-weights 16,32,64,128,256,512,1024
```

The oracle script uses expected Bellman backups over a `(log stability, difficulty)` grid and discrete desired-retention actions. In `tradeoff.py`, `fsrs6_oracle` solves all requested scalarization weights in one batched DP pass and evaluates them in one batched Monte Carlo rollout. It writes a single-card CSV/plot under `artifacts/single_card_tradeoff/` and, by default, includes static-FSRS reference rows evaluated with Monte Carlo particles.

The same desired-retention action space also has an infinite-horizon average-reward oracle. It removes the remaining-horizon state and solves a stationary SMDP policy over `(log stability, difficulty)`:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_infinite --oracle-cost-weights 16,32,64,128,256,512,1024
```

Train the stationary oracle distillation policy, then compare it in the same finite lifecycle tradeoff evaluation:

```bash
uv run experiments/single_card_tradeoff/oracle_infinite_distill.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle,fsrs6_oracle_infinite,fsrs6_oracle_infinite_distill --oracle-infinite-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt
```

The infinite oracle optimizes long-run average `retrievability - goal_cost_weight * minutes` and reports its stationary gain, policy-iteration count, and residual to stdout. The tradeoff CSV still reports the existing finite lifecycle metrics so it remains comparable with the finite-horizon oracle and distilled policies.

For a stationary policy optimized against the finite 1825-day new-card lifecycle, use `fsrs6_oracle_stationary_finite`. It keeps the same `(stability, difficulty, goal cost weight)` policy input as the infinite oracle, initializes from the finite-horizon oracle's visited-state actions, then runs occupancy-weighted policy iteration on the finite lifecycle objective:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_stationary_finite --oracle-cost-weights 16,32,64,128,256,512,1024
```

Train and compare the corresponding distilled stationary finite-lifecycle policy:

```bash
uv run experiments/single_card_tradeoff/oracle_stationary_finite_distill.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_stationary_finite,fsrs6_oracle_stationary_finite_distill --oracle-stationary-finite-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
```

The stationary finite oracle reports finite-lifecycle objective, policy-iteration count, and residual to stdout. The distilled checkpoint uses `policy_type=fsrs6_oracle_stationary_finite_distill` and defaults to the `oracle_stationary` observation (`stability`, `difficulty`, and goal cost weight only).

Visualize the solved oracle policy's output distribution over states visited by the policy rollout:

```bash
uv run experiments/single_card_tradeoff/oracle_policy_outputs.py --source rollout --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Pass `--source table` to count every nonterminal `(remaining, stability, difficulty)` grid cell equally instead of weighting by rollout visits. The script writes `artifacts/single_card_tradeoff/fsrs6_oracle_policy_outputs.csv` and `.png`.

Visualize the stationary finite-lifecycle oracle directly over its stationary `(stability, difficulty, goal cost weight)` policy table:

```bash
uv run experiments/single_card_tradeoff/oracle_stationary_finite_policy_viz.py --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 0,16,64,256,1024
```

This writes `action_summary.csv`, `binned_actions.csv`, `grid_actions.csv`, `findings.md`, `action_distribution.png`, and `policy_heatmaps.png` under `artifacts/single_card_tradeoff/stationary_finite_policy_viz/`. Use `--selected-weights` to choose which weights appear in the `(s,d)` heatmap panel.

## Interval Oracle And Distillation

The single-card sweep supports an integer-interval FSRS-6 oracle that enumerates every feasible next interval `1..remaining+1`, where `remaining+1` means no further review before the horizon:

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_interval --oracle-cost-weights 16,32,64 --oracle-interval-chunk-size 64
```

Train the 4D log-interval distillation policy, then include it in the same sweep:

```bash
uv run experiments/single_card_tradeoff/oracle_interval_distill.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_interval_distill --oracle-interval-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt
```

The interval distillation model keeps the same 4D scalar-output network. Its default `residual:64:3` sweet-spot architecture has 25,857 parameters, about 45% of the earlier `residual:96:3` model, while preserving the time-regret advantage in the default FSRS-6 comparison. The more aggressive `residual:32:2` candidate has 4,609 parameters, about 8% of the earlier model, but gives up a small amount of pairwise time-regret AUC against `fsrs6_oracle_distill`. The default training loss weights underpredicted intervals more heavily for high cost weights, mixes in student-rollout states after warmup, and snaps predicted intervals near the remaining horizon to the terminal no-more-review action. These are fixed training/inference rules and do not add learned parameters.

To rerun the model-size search:

```bash
uv run experiments/single_card_tradeoff/oracle_interval_distill_hparam_search.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --eval-particles 3000 --save-models
```

## Continuous Desired-Retention Distillation

The integer-interval oracle can also be distilled into a continuous desired-retention policy. This keeps the compact `oracle_rho4` residual `16x2` default and executes by converting the predicted retention into the nearest integer review interval inside the day-level simulator. Its default training is interval-aware: the model still outputs retention, but the loss penalizes the log interval implied by that retention, weights high-cost underprediction heavily, and uses student-rollout states after warmup to preserve frontier coverage.

```bash
uv run experiments/single_card_tradeoff/oracle_retention_distill.py --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_oracle_distill,fsrs6_oracle_retention_distill --oracle-retention-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt
```

## Findings

`single_card_tradeoff` is best understood as a frontier experiment, not a full deck scheduler benchmark. It removes daily budget constraints and isolates the memory-time tradeoff for one iid card lifecycle. The metrics can be deck-scaled, but they should not be read as a complete workload simulation.

The most useful summary metric is `time_regret_auc`, but it is only meaningful together with `span_coverage_percent`. A negative `time_regret_auc` means a scheduler uses fewer deck-scaled minutes/day than the baseline at the same memory target over their common memory interval. Low coverage means the comparison only covers a narrow part of the frontier.

Stationary finite-lifecycle oracle comparison:

The stationary finite oracle keeps the policy input stationary, `(stability, difficulty, goal cost weight)`, but optimizes that stationary policy against the same finite 1825-day new-card lifecycle used by the standard evaluation. This is a constrained-policy oracle, not the unrestricted finite-horizon `fsrs6_oracle`, so it should be judged by how much unrestricted finite-horizon value it preserves after removing the remaining-time input.

The direct comparison below uses `fsrs6_default`, 1825 days, 10,000 particles, `deck_scale=10000`, default 64x32 oracle grids, and the standard evaluation cost weights `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`. `fsrs6_oracle` was rerun as `artifacts/single_card_tradeoff/fsrs6_oracle_exact_compare/results.csv`; the stationary finite exact rows are from `artifacts/single_card_tradeoff/stationary_finite_compare/results.csv`; the current compact distill rows are from `artifacts/single_card_tradeoff/default_oracle_distill_vs_stationary_finite_distill/results.csv`.

| scheduler | policy input | vs `fsrs6_default` `time_regret_auc` | coverage |
| --- | --- | ---: | ---: |
| `fsrs6_oracle` | `(remaining, stability, difficulty, goal weight)` | -3.5985 | 79.0% |
| `fsrs6_oracle_distill` | `oracle_rho4`, `residual:16:2` | -3.6443 | 78.8% |
| `fsrs6_oracle_stationary_finite` | `(stability, difficulty, goal weight)` | -3.3900 | 78.1% |
| `fsrs6_oracle_stationary_finite_distill` | `(stability, difficulty, goal weight)` | -3.5814 | 72.2% |

The stationary finite exact oracle is close to the unrestricted finite-horizon oracle while using a much smaller policy table. Its main weakness in this run is the mid-cost region: at representative weights its scalar objective is about `0.007-0.008` below `fsrs6_oracle` at `w=64` and `w=256`, while the zero-cost and high-cost endpoints are effectively tied within rollout/grid noise.

| scheduler | exact policy table entries | distill checkpoint parameters | table-to-distill compression |
| --- | ---: | ---: | ---: |
| `fsrs6_oracle` -> `fsrs6_oracle_distill` | 63,539,200 | 1,536 | 41,367x |
| `fsrs6_oracle_stationary_finite` -> `fsrs6_oracle_stationary_finite_distill` | 34,816 | 1,520 | 22.9x |

The unrestricted finite oracle has the larger relative compression ratio because its table includes the 1825-step remaining-time dimension. The stationary finite oracle has already removed that dimension before distillation, so there is less redundancy left for the neural policy to compress. After retraining the default `fsrs6_oracle_distill_policy.pt` as the compact `oracle_rho4`, `residual:16:2` model, the two deployed neural policies are almost the same size: `1,536` parameters for `fsrs6_oracle_distill` versus `1,520` for `fsrs6_oracle_stationary_finite_distill`. The stationary finite distill is only 16 parameters smaller because it drops one input feature, not because the learned network is structurally different.

Representative scalarization points:

| weight | scheduler | card R | deck minutes/day | scalar objective |
| ---: | --- | ---: | ---: | ---: |
| 0 | `fsrs6_oracle` | 0.9884 | 59.35 | 0.9884 |
| 0 | `fsrs6_oracle_distill` | 0.9885 | 62.41 | 0.9885 |
| 0 | `fsrs6_oracle_stationary_finite` | 0.9883 | 58.40 | 0.9883 |
| 0 | `fsrs6_oracle_stationary_finite_distill` | 0.9883 | 56.87 | 0.9883 |
| 16 | `fsrs6_oracle` | 0.9752 | 25.27 | 0.9347 |
| 16 | `fsrs6_oracle_distill` | 0.9765 | 25.63 | 0.9355 |
| 16 | `fsrs6_oracle_stationary_finite` | 0.9751 | 25.44 | 0.9344 |
| 16 | `fsrs6_oracle_stationary_finite_distill` | 0.9773 | 26.34 | 0.9351 |
| 64 | `fsrs6_oracle` | 0.9491 | 16.78 | 0.8417 |
| 64 | `fsrs6_oracle_distill` | 0.9557 | 17.94 | 0.8409 |
| 64 | `fsrs6_oracle_stationary_finite` | 0.9487 | 17.83 | 0.8346 |
| 64 | `fsrs6_oracle_stationary_finite_distill` | 0.9585 | 18.88 | 0.8376 |
| 256 | `fsrs6_oracle` | 0.8377 | 8.64 | 0.6165 |
| 256 | `fsrs6_oracle_distill` | 0.8116 | 7.77 | 0.6126 |
| 256 | `fsrs6_oracle_stationary_finite` | 0.8210 | 8.31 | 0.6082 |
| 256 | `fsrs6_oracle_stationary_finite_distill` | 0.8555 | 9.95 | 0.6008 |
| 1024 | `fsrs6_oracle` | 0.5262 | 2.72 | 0.2480 |
| 1024 | `fsrs6_oracle_distill` | 0.5271 | 2.70 | 0.2504 |
| 1024 | `fsrs6_oracle_stationary_finite` | 0.5314 | 2.71 | 0.2538 |
| 1024 | `fsrs6_oracle_stationary_finite_distill` | 0.5656 | 3.13 | 0.2456 |

The current default `fsrs6_oracle_distill` run used 4,194,304 teacher transitions. Its final cross-entropy was `0.56995`, train teacher-action agreement was `75.11%`, and eval agreement was `74.38%`.

The stationary finite distillation run used 4,194,304 teacher transitions. Its final cross-entropy was `0.58973`, train teacher-action agreement was `73.55%`, eval agreement was `72.97%`, and all teacher policies converged with policy-iteration counts `[3,3,3,3,4,2,6,3,5,3,8,1]`.

Stationary finite action distribution:

The exact stationary finite oracle policy was visualized over the equal-weighted `(stability, difficulty)` policy table with 1825 days, a 64x32 grid, and representative weights `0,16,64,256,1024`. The artifacts are in `artifacts/single_card_tradeoff/stationary_finite_policy_viz/`: `findings.md`, `action_summary.csv`, `binned_actions.csv`, `grid_actions.csv`, `action_distribution.png`, and `policy_heatmaps.png`.

| weight | modal retention | modal share | mean action retention | normalized entropy | objective |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.98 | 55.4% | 0.9168 | 0.572 | 0.987743 |
| 16 | 0.93 | 21.5% | 0.8794 | 0.737 | 0.937493 |
| 64 | 0.93 | 23.1% | 0.8378 | 0.757 | 0.842184 |
| 256 | 0.10 | 23.5% | 0.6654 | 0.714 | 0.615004 |
| 1024 | 0.10 | 76.8% | 0.2652 | 0.321 | 0.250814 |

The dominant pattern is cost sensitivity: as `w` rises, the table-average desired-retention action falls from `0.9168` to `0.2652`, and the policy eventually collapses toward the cheapest action. The transition is not smooth, though. At `w=256`, the policy is still mixed and bimodal: `0.10` is the modal action, but high-retention actions such as `0.90` and `0.85` together still occupy a large share of the table. By `w=1024`, `0.10` covers `76.8%` of grid cells and entropy drops sharply.

| weight | low stability mean | high stability mean | low difficulty mean | high difficulty mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7438 | 0.8700 | 0.9182 | 0.9258 |
| 16 | 0.7436 | 0.8150 | 0.9000 | 0.8627 |
| 64 | 0.7428 | 0.7880 | 0.8670 | 0.7966 |
| 256 | 0.7361 | 0.2037 | 0.7599 | 0.4509 |
| 1024 | 0.3250 | 0.2037 | 0.5358 | 0.1130 |

The axis slices explain where the high-cost collapse happens. Low-stability states keep relatively high desired-retention actions through `w=256`, while high-stability states move to much lower actions because they can tolerate long intervals. Difficulty matters most once cost is high: at `w=1024`, the high-difficulty bin averages only `0.1130`, which means the oracle largely stops spending reviews on the hardest states, while the low-difficulty bin still averages `0.5358`.

Current compact default baseline:

| scheduler | model | parameters | vs `fsrs6_default` `time_regret_auc` | coverage |
| --- | --- | ---: | ---: | ---: |
| `fsrs6_oracle_distill` | `oracle_rho4`, `residual:16:2` | 1,536 | -3.6443 | 78.8% |
| `fsrs6_oracle_stationary_finite_distill` | `oracle_stationary`, `residual:16:2` | 1,520 | -3.5814 | 72.2% |
| `fsrs6_oracle_retention_distill` | `oracle_rho4`, `residual:16:2` | 1,536 | -3.4359 | 80.4% |

The retrained compact `fsrs6_oracle_distill` remains the strongest compact default baseline in this group. Directly against `fsrs6_oracle_stationary_finite_distill`, it has `-0.1719` deck-minutes/day time-regret AUC over 100% of the stationary-finite distill span, while the stationary-finite distill covers only `91.6%` of the oracle-distill span when the baseline is reversed. The repaired continuous desired-retention distill now covers the frontier about as broadly as the discrete distill. Directly against `fsrs6_oracle_distill`, it still has a small positive regret (`+0.1837` deck-minutes/day over 100% overlap in the default comparison), so the discrete distill remains the strongest compact default baseline.

Stationary finite distill compression sweep:

The compression sweep artifacts are under `artifacts/single_card_tradeoff/stationary_finite_compression/`. The first group trained smaller networks directly against the stationary finite oracle labels. The second group distilled the current `1,520`-parameter stationary finite checkpoint into smaller students, either with teacher-forced rollouts or with student-rollout states after warmup.

| candidate | parameters | table compression | vs `fsrs6_oracle_distill` relative regret | vs `fsrs6_oracle_distill` coverage | vs `fsrs6_default` relative regret | vs `fsrs6_default` coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current `residual:16:2` | 1,520 | 22.9x | +1.92% | 91.6% | -28.16% | 72.2% |
| exact-label `residual:8:1` | 352 | 98.9x | +5.34% | 30.5% | -17.95% | 24.0% |
| exact-label `residual:6:2` | 340 | 102.4x | +6.79% | 33.9% | -17.13% | 26.7% |
| exact-label `mlp:8` | 248 | 140.4x | +2.78% | 11.8% | -21.90% | 9.3% |
| teacher-forced student `residual:8:2` | 512 | 68.0x | +4.83% | 25.1% | -18.17% | 19.8% |
| student-rollout `residual:8:2` | 512 | 68.0x | +4.81% | 88.0% | -25.82% | 69.3% |
| student-rollout `residual:14:2` | 1,220 | 28.5x | +2.98% | 80.0% | -26.45% | 63.0% |

The useful compressed candidate is `student-rollout residual:8:2` (`sroll_residual_h8_d2.pt`). It cuts the stationary finite distill from `1,520` to `512` parameters and raises table compression from `22.9x` to `68.0x`, while preserving most of the default-baseline span (`69.3%` versus `72.2%`). Its cost is higher regret: relative to `fsrs6_oracle_distill`, it is `+4.81%` instead of `+1.92%`; relative to `fsrs6_default`, it is still strongly better at `-25.82%`. The 100x+ compression attempts are not reliable frontier policies despite sometimes attractive AUC values, because their span coverage collapses to roughly `9-34%`.

Before the interval-aware repair, `fsrs6_oracle_retention_distill` looked deceptively strong against `fsrs6_default` (`time_regret_auc=-5.2367`) but had only `15.1%` coverage. Its points were concentrated in the high-memory region, so the AUC was computed over too narrow a memory span. The repaired version expands coverage to `80.4%` and moves the high-cost end from `card_mem≈0.900` to `card_mem≈0.518` at `w=1024`.

PPO oracle-guide ablation results are in `artifacts/single_card_tradeoff/ppo_ablation/ppo_oracle_warmup_ablation_summary.csv`. These runs use the same 10k-particle FSRS-6 default evaluation grid as the PPO comparison above.

| scheduler | ablation | warmup labels | PPO transitions | vs `fsrs6_default` `time_regret_auc` | coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `uvfa_ppo` | default oracle guide | 131,072 | 2,359,296 | -3.6769 | 79.0% |
| `uvfa_ppo` | no guide | 0 | 2,359,296 | -2.9455 | 80.2% |
| `uvfa_ppo` | oracle warmup only | 131,072 | 0 | -2.4558 | 27.3% |
| `uvfa_ppo` | static guide | 131,072 | 2,359,296 | -2.4375 | 73.6% |
| `uvfa_ppo_rnn_interval` | default oracle guide | 1,572,864 | 2,359,296 | -3.6857 | 78.0% |
| `uvfa_ppo_rnn_interval` | no guide | 0 | 2,359,296 | -0.1692 | 85.8% |
| `uvfa_ppo_rnn_interval` | oracle warmup only | 1,572,864 | 0 | -3.6098 | 78.1% |
| `uvfa_ppo_rnn_interval` | static guide | 1,572,864 | 2,359,296 | +6.8610 | 30.0% |

The UVFA PPO result is not purely inherited from oracle warmup: removing the guide still keeps a useful frontier (`time_regret_auc=-2.9455`), while oracle warmup alone covers only the high-memory portion of the frontier. The RNN interval result is much more oracle-imitation dominated: its oracle-warmup-only checkpoint is within `0.0759` deck-minutes/day of the default oracle-guided checkpoint against `fsrs6_default`, and the no-guide run is nearly flat against the baseline. The static-guide RNN run is unstable at the low-cost edge, spending `1263.13` deck-minutes/day at `w=0`, so its positive AUC should be read as a failed ablation rather than a useful frontier.

## Lessons

- Coverage is a first-class metric. Always report `time_regret_auc`, `span_coverage_percent`, and enough endpoint rows to show whether the frontier covers both high-memory and low-cost regimes.
- Desired retention is not a naturally well-conditioned coordinate for an integer-interval teacher. Small retention errors can become large interval errors, especially for high stability, high cost weights, and horizon-terminal actions.
- For continuous desired-retention distillation, train on the implied interval, not just the retention value. The repaired default keeps retention as the action output but supervises `log(interval(retention))`.
- Underpredicting intervals is worse than overpredicting intervals at high cost weights. The effective training recipe weights high-cost underprediction and terminal-action underprediction more heavily.
- Student rollout matters. Teacher-forced states alone did not fix coverage; mixing student-rollout states after warmup exposed the model to the states it actually creates.
- For stationary finite distill compression, 100x+ table compression is too aggressive under the current recipe because coverage collapses. The best observed smaller point is the 512-parameter student-rollout `residual:8:2` model: about one third of the current checkpoint size, with a manageable regret increase and much better span coverage than teacher-forced compression.
- PPO guide ablations need a warmup-only control. A guided PPO checkpoint can look like a reinforcement-learning win even when most of the deployed behavior came from supervised oracle labels, especially for the RNN interval policy.
- For the RNN interval policy, the current default is best interpreted as an oracle-imitation policy with PPO fine-tuning. The dense AUC barely changes from oracle warmup only to full training, while no-guide PPO does not recover the same frontier.
- Removing remaining time at the oracle layer is a powerful structural compression. `fsrs6_oracle_stationary_finite` gives up a small amount of mid-cost scalar objective versus unrestricted `fsrs6_oracle`, but its exact policy table is 1825x smaller before any neural distillation.
- The exact stationary finite policy is not just a lower-retention version of the finite oracle. Its action table is cost-sensitive and state-sensitive: at `w=256` it is still mixed and bimodal, while at `w=1024` most high-difficulty states collapse to the cheapest retention action. That explains why very small stationary finite distills lose span coverage quickly despite the smaller teacher table.
- `oracle_rho4` is a strong compact observation. It gives a 1,536-parameter model enough horizon and stability information to cover the frontier when the loss geometry is right.
- Train cost weights should be sparse but cover scale: use `0 + 2^0..2^10`. Evaluation should remain denser with `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`.
- Integer interval actions are the most direct continuous-action target for the finite-horizon oracle. Continuous desired retention can work, but it needs interval-aware loss shaping and coverage validation.
