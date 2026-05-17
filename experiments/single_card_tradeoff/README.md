# Single-Card Tradeoff Experiments

This experiment family simulates an iid single-card lifecycle with no daily study-budget constraints. Card-level metrics are linearly scaled to a 10,000-card deck so scheduler frontiers can be compared by expected memorized cards and study minutes per day.

## Quickstart

```bash
uv run experiments/single_card_tradeoff/tradeoff.py --env fsrs6_default --sched fsrs6_default --particles 10000 --deck-scale 10000
```

When CUDA is available, `tradeoff.py` uses `cuda` by default; pass `--torch-device cpu` to force CPU.

Pass `--env fsrs6 --user-id <id>` to load per-user FSRS-6 weights from `../srs-benchmark`; add `--button-usage ../Anki-button-usage/button_usage.jsonl` to use that user's first/review rating probabilities and learning/review costs in the single-card oracle, PPO, and distillation rollouts. `--env fsrs6_default` keeps the built-in FSRS-6 parameters and default costs.

For supported FSRS-6 sweeps, desired-retention targets are batched in one vectorized run by default. The default targets are `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`; override with `--target-retentions`, or pass `--target-retentions ""` to use the range flags. Single-card target and oracle-action retentions must be at least `0.5`. Fixed-interval sweeps are batched the same way. Plain `--sched fixed` runs intervals `8,16,32,64,128,256,512` by default; override with `--fixed-intervals`. Mixed scheduler families are run as one batch per family. Pass `--target-batch-size 1` to run targets/intervals sequentially.

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

The stationary finite oracle reports finite-lifecycle objective, policy-iteration count, and residual to stdout. The distilled checkpoint uses `policy_type=fsrs6_oracle_stationary_finite_distill` and defaults to the compressed `oracle_stationary` residual `8x2` policy: 476 parameters, 128 distillation epochs, teacher cost weights `0,16,64,256,1024`, and `uniform_table` exact-policy supervision. Its observation remains `stability`, `difficulty`, and goal cost weight only.

For per-user benchmark distillation, `oracle_stationary_finite_distill_multiuser.py --per-user-models` trains one independent stationary finite distill policy per FSRS-6 user in a single Python process. The exact teacher uses `FSRS6BatchedStationaryFiniteOracle`, whose policy shape is `[user, cost_weight, stability, difficulty]`; `--oracle-teacher-user-batch-size 0` is the default and solves all requested users in one batched DP call. The student training then stacks `U` ordinary `PolicyValueNet` states with `torch.func.stack_module_state` and uses `vmap` over the user dimension, so each user has a separate 476-parameter model while the ensemble trains and evaluates as one batch. The per-user default supervision is `--per-user-supervision uniform_table`: every train step samples the exact stationary policy table with equal mass on each sparse teacher cost weight, rather than weighting labels by rollout event counts. Use `--per-user-supervision rollout` only to reproduce the older teacher-forcing baseline. The default path without `--per-user-models` remains the older shared-student diagnostic baseline.

```bash
uv run experiments/single_card_tradeoff/oracle_stationary_finite_distill_multiuser.py --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 --button-usage ../Anki-button-usage/button_usage.jsonl --per-user-models --out-dir artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu --no-progress
```

On the first eight benchmark users, the uniform-table per-user run wrote `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/` with 8 checkpoints, 476 parameters per checkpoint, 3,808 total trainable ensemble parameters, `teacher_s=61.06`, `train_s=38.19`, `agreement_s=0.01`, `eval_s=311.22`, mean final CE `0.69721`, mean train table agreement `72.41%`, and mean full-table agreement `72.39%` on the CUDA test environment. Mean `fsrs6` baseline span coverage was `97.98%`, and mean relative regret AUC was `-12.84%` over the eight users. The older rollout teacher-forcing per-user artifact `stationary_finite_distill_first8_users_per_user_batched/` had mean span coverage `95.82%`, mean relative regret AUC `-4.32%`, and a user-2 high-cost interpolation failure (`+18.20%` relative regret AUC). Uniform exact-table supervision fixes that failure: user 2 improves to `96.30%` coverage and `-15.09%` relative regret AUC.

The shared-student diagnostic artifact is `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_batched/`. It used one 476-parameter model for all eight users and reported mean span coverage `96.71%` and mean relative regret AUC `-6.38%`. The user-2 failure in the rollout per-user path was caused by event-count sampling bias: `w=0` produces many more review events than high cost weights, so the cross-entropy loss was dominated by low-cost actions even though reset-time cost weights were uniform. The current per-user default removes that bias by supervising the exact `[user, cost_weight, stability, difficulty]` table with equal samples per sparse teacher cost weight.

Visualize the solved oracle policy's output distribution over states visited by the policy rollout:

```bash
uv run experiments/single_card_tradeoff/oracle_policy_outputs.py --source rollout --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Pass `--source table` to count every nonterminal `(remaining, stability, difficulty)` grid cell equally instead of weighting by rollout visits. The script writes `artifacts/single_card_tradeoff/fsrs6_oracle_policy_outputs.csv` and `.png`.

Visualize the stationary finite-lifecycle oracle directly over its stationary `(stability, difficulty, goal cost weight)` policy table:

```bash
uv run experiments/single_card_tradeoff/oracle_stationary_finite_policy_viz.py --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 0,16,64,256,1024 --action-retentions 0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98 --distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
```

This writes `action_summary.csv`, `binned_actions.csv`, `grid_actions.csv`, `findings.md`, `action_distribution.png`, and `policy_heatmaps.png` under `artifacts/single_card_tradeoff/stationary_finite_policy_viz/`. When `--distill-policy` is provided it also writes `distill_action_summary.csv`, `distill_binned_actions.csv`, `distill_grid_actions.csv`, `distill_exact_comparison.csv`, `distill_action_distribution.png`, `distill_policy_heatmaps.png`, and `distill_exact_difference_heatmaps.png`. Use `--selected-weights` to choose which weights appear in the `(s,d)` heatmap panel.

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

No-sub-0.5 action-space rerun:

The current default target/action retention grid is `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`. Checkpoints whose action grid includes values below `0.5` now fail validation in the tradeoff runner, and the report below only uses artifacts that were verified to use the new action grid.

The current compact comparison uses `fsrs6_default`, 1825 days, 10,000 particles, `deck_scale=10000`, and the standard scalarization weights for each scheduler implementation. The combined artifacts are in `artifacts/single_card_tradeoff/no_sub05_distill_compare/`. The default desired-retention baseline rows come from `artifacts/single_card_tradeoff/no_sub05_default_vs_stationary_finite_distill/results.csv`; the learned-policy evaluations are `artifacts/single_card_tradeoff/fsrs6_oracle_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_results.csv`, `artifacts/single_card_tradeoff/uvfa_ppo_results.csv`, and `artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_results.csv`.

| scheduler | policy input | parameters | vs `fsrs6_default` `time_regret_auc` | relative regret | coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_distill` | `oracle_rho4`, `residual:16:2` | 1,468 | -3.6263 | -24.18% | 86.3% |
| `fsrs6_oracle_stationary_finite_distill` | `oracle_stationary`, `residual:16:2` | 1,452 | -3.4070 | -23.70% | 94.7% |
| `fsrs6_oracle_retention_distill` | `oracle_rho4`, `residual:16:2` | 1,468 | -3.3248 | -21.79% | 83.5% |
| `fsrs6_oracle_infinite_distill` | `oracle_stationary`, `residual:16:2` | 1,452 | -4.8242 | -14.66% | 9.0% |
| `uvfa_ppo` | `rich`, `residual:64:3` | 27,148 | -3.4886 | -24.89% | 99.2% |
| `uvfa_ppo_rnn_interval` | `belief`, `GRU:128` | 87,559 | -3.0423 | -21.67% | 69.2% |

Directly against `fsrs6_oracle_distill`, the current `fsrs6_oracle_stationary_finite_distill` has `+0.0944` deck-minutes/day time-regret AUC, `+0.83%` relative regret, and `99.997%` coverage. The continuous desired-retention distill has `+0.3542` deck-minutes/day, `+3.07%` relative regret, and `96.7%` coverage. The default `uvfa_ppo` checkpoint has `-0.0639` deck-minutes/day, `-0.56%` relative regret, and `99.9%` coverage against `fsrs6_oracle_distill`; it is slightly ahead on the shared span, but it uses an order of magnitude more parameters. The recurrent interval PPO has `+0.1289` deck-minutes/day, `+1.19%` relative regret, and `80.2%` coverage against `fsrs6_oracle_distill`. Reversing the stationary-finite baseline gives `fsrs6_oracle_distill` `-0.0944` deck-minutes/day over `91.1%` of the stationary-finite-distill span. The stationary finite distill is slightly worse than the unrestricted finite-horizon distill on shared span, but it covers more of the `fsrs6_default` retention span after clipping actions below `0.5`.

The average-reward infinite distill should not be judged by its negative default-baseline AUC alone. Its frontier covers only `9.0%` of the default retention span after the action floor is applied, so the AUC is computed over a narrow high-memory region. Directly against `fsrs6_oracle_distill`, it has `+15.59%` relative regret over only `10.4%` of the oracle-distill span.

| scheduler | exact policy table entries | distill checkpoint parameters | table-to-distill compression |
| --- | ---: | ---: | ---: |
| `fsrs6_oracle` -> `fsrs6_oracle_distill` | 41,113,600 | 1,468 | 28,006x |
| `fsrs6_oracle_stationary_finite` -> `fsrs6_oracle_stationary_finite_distill` | 22,528 | 1,452 | 15.5x |

The unrestricted finite oracle still has the much larger relative compression ratio because its table includes the 1825-step remaining-time dimension. The stationary finite oracle has already removed that dimension before neural distillation, so there is less table redundancy left to compress. After removing four low-retention actions, both neural checkpoints also shrink: `fsrs6_oracle_distill` is 1,468 parameters and `fsrs6_oracle_stationary_finite_distill` is 1,452 parameters.

Representative scalarization points:

| weight | scheduler | card R | deck minutes/day | scalar objective |
| ---: | --- | ---: | ---: | ---: |
| 0 | `fsrs6_oracle_distill` | 0.9884 | 62.23 | 0.9884 |
| 0 | `fsrs6_oracle_stationary_finite_distill` | 0.9884 | 59.52 | 0.9884 |
| 0 | `fsrs6_oracle_retention_distill` | 0.9948 | 370.60 | 0.9948 |
| 0 | `fsrs6_oracle_infinite_distill` | 0.9859 | 51.37 | 0.9859 |
| 0 | `uvfa_ppo` | 0.9882 | 54.67 | 0.9882 |
| 16 | `fsrs6_oracle_distill` | 0.9771 | 26.20 | 0.9351 |
| 16 | `fsrs6_oracle_stationary_finite_distill` | 0.9769 | 25.89 | 0.9355 |
| 16 | `fsrs6_oracle_retention_distill` | 0.9753 | 24.29 | 0.9365 |
| 16 | `fsrs6_oracle_infinite_distill` | 0.9715 | 25.81 | 0.9302 |
| 16 | `uvfa_ppo` | 0.9780 | 26.81 | 0.9351 |
| 16 | `uvfa_ppo_rnn_interval` | 0.9721 | 22.98 | 0.9353 |
| 64 | `fsrs6_oracle_distill` | 0.9547 | 17.59 | 0.8421 |
| 64 | `fsrs6_oracle_stationary_finite_distill` | 0.9558 | 18.12 | 0.8398 |
| 64 | `fsrs6_oracle_retention_distill` | 0.9522 | 17.24 | 0.8418 |
| 64 | `fsrs6_oracle_infinite_distill` | 0.9642 | 21.79 | 0.8248 |
| 64 | `uvfa_ppo` | 0.9536 | 17.65 | 0.8406 |
| 64 | `uvfa_ppo_rnn_interval` | 0.9497 | 16.81 | 0.8421 |
| 256 | `fsrs6_oracle_distill` | 0.8430 | 9.24 | 0.6065 |
| 256 | `fsrs6_oracle_stationary_finite_distill` | 0.8387 | 9.18 | 0.6036 |
| 256 | `fsrs6_oracle_retention_distill` | 0.8393 | 9.77 | 0.5893 |
| 256 | `fsrs6_oracle_infinite_distill` | 0.9591 | 20.54 | 0.4332 |
| 256 | `uvfa_ppo` | 0.8419 | 9.22 | 0.6058 |
| 256 | `uvfa_ppo_rnn_interval` | 0.8648 | 10.13 | 0.6054 |
| 1024 | `fsrs6_oracle_distill` | 0.6861 | 5.92 | 0.0797 |
| 1024 | `fsrs6_oracle_stationary_finite_distill` | 0.6567 | 5.70 | 0.0727 |
| 1024 | `fsrs6_oracle_retention_distill` | 0.6961 | 7.38 | -0.0598 |
| 1024 | `fsrs6_oracle_infinite_distill` | 0.9543 | 19.29 | -1.0209 |
| 1024 | `uvfa_ppo` | 0.6407 | 5.33 | 0.0945 |
| 1024 | `uvfa_ppo_rnn_interval` | 0.7297 | 7.15 | -0.0020 |

The current default `fsrs6_oracle_distill` run used 4,194,304 teacher transitions. Its final cross-entropy was `0.60431`, train teacher-action agreement was `74.26%`, and eval agreement was `73.21%`.

The stationary finite distillation run used 4,194,304 teacher transitions. Its final cross-entropy was `0.60990`, train teacher-action agreement was `73.73%`, eval agreement was `72.88%`, and all teacher policies converged with policy-iteration counts `[3,3,3,3,4,2,6,3,5,5,10,8]`.

The infinite average-reward distillation run used 4,194,304 teacher transitions. Its final cross-entropy was `0.84524`, train teacher-action agreement was `63.52%`, eval agreement was `62.45%`, and all teacher policies converged with policy-iteration counts `[7,9,9,8,8,8,9,10,9,9,9,10]`.

The continuous desired-retention distillation run used 4,194,304 teacher transitions. Its final loss was `0.46096`, final interval loss was `0.43893`, eval log-interval MAE was `1.13859`, and rounded interval agreement was `10.12%`. The low rounded agreement reflects the continuous-retention objective; its frontier coverage is the more useful deployment check.

The default `uvfa_ppo` run used 2,359,296 training transitions, 36 PPO updates, the oracle guide policy, and the `rich` residual `64x3` network. Training took `208.69s`; the resulting checkpoint was verified to use the clipped action grid and passed the fixed-interval baseline check.

The default `uvfa_ppo_rnn_interval` run used 2,359,296 training transitions, 36 PPO updates, the oracle guide policy, and the `belief` GRU-128 policy. Training took `240.77s`; the resulting 87,559-parameter checkpoint was verified to use the clipped action grid and passed the fixed-interval baseline check.

PPO guide ablations after clipping actions:

The no-sub-0.5 ablation artifacts are under `artifacts/single_card_tradeoff/ppo_ablation/`. All six checkpoints were verified to use the clipped 11-action grid with minimum action `0.5`, and the tradeoff CSVs below use the clipped 11-target `fsrs6_default` baseline.

| candidate | guide setup | parameters | train transitions | updates | train time | fixed-baseline check | vs `fsrs6_default` relative regret | coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `uvfa_ppo_no_guide` | no guide | 27,148 | 2,359,296 | 36 | 28.54s | pass | -22.74% | 85.9% |
| `uvfa_ppo_static_guide` | static guide | 27,148 | 2,359,296 | 36 | 31.93s | fail | -8.52% | 54.0% |
| `uvfa_ppo_oracle_warmup_only` | oracle warmup, no PPO updates | 27,148 | 0 | 0 | 188.67s | fail | -16.40% | 64.2% |
| `uvfa_ppo_rnn_interval_no_guide` | no guide, no warmup prior | 87,559 | 2,359,296 | 36 | 40.58s | fail | +0.93% | 87.8% |
| `uvfa_ppo_rnn_interval_static_guide` | static guide | 87,559 | 2,359,296 | 36 | 48.80s | fail | -2.22% | 42.7% |
| `uvfa_ppo_rnn_interval_oracle_warmup_only` | oracle warmup, no PPO updates | 87,559 | 0 | 0 | 196.54s | pass | -21.31% | 72.6% |

The ablation takeaway did not change after removing actions below `0.5`: the default oracle-guided PPO runs remain the strongest PPO checkpoints in the current report. Discrete PPO without a guide still beats `fsrs6_default`, but it trails the default oracle-guided checkpoint on both relative regret and coverage. The RNN interval warmup-only checkpoint is useful as a prior-quality check, but its coverage is lower than the default trained RNN checkpoint.

Stationary finite action distribution with sub-0.5 actions removed:

The exact stationary finite oracle policy was visualized over the equal-weighted `(stability, difficulty)` policy table with 1825 days, a 64x32 grid, representative weights `0,16,64,256,1024`, and only desired-retention actions `>=0.5`. The artifacts are in `artifacts/single_card_tradeoff/stationary_finite_policy_viz/`: `findings.md`, `action_summary.csv`, `binned_actions.csv`, `grid_actions.csv`, `action_distribution.png`, and `policy_heatmaps.png`. All policies converged, with policy-iteration counts `[3,2,3,5,8]`.

| weight | modal retention | modal share | mean action retention | normalized entropy | objective |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.98 | 55.4% | 0.9238 | 0.646 | 0.987743 |
| 16 | 0.93 | 21.5% | 0.8929 | 0.833 | 0.937493 |
| 64 | 0.93 | 23.1% | 0.8638 | 0.853 | 0.842184 |
| 256 | 0.50 | 24.9% | 0.7608 | 0.774 | 0.612241 |
| 1024 | 0.50 | 52.6% | 0.6541 | 0.651 | 0.121657 |

The dominant pattern is still cost sensitivity, but the action floor changes the high-cost behavior. As `w` rises, the table-average desired-retention action falls from `0.9238` to `0.6541`, and the policy eventually concentrates on the cheapest available action, `0.50`. The transition remains mixed: at `w=256`, `0.50` is only barely modal at `24.9%`, while `0.90` covers `24.0%` and `0.85` covers `18.8%`. By `w=1024`, `0.50` covers `52.6%` of grid cells, but the high-cost scalar objective drops to `0.121657` because the oracle can no longer choose very low review-cost actions below `0.5`.

| weight | low stability mean | high stability mean | low difficulty mean | high difficulty mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7438 | 0.9200 | 0.9245 | 0.9289 |
| 16 | 0.7436 | 0.8900 | 0.9063 | 0.8784 |
| 64 | 0.7428 | 0.8727 | 0.8841 | 0.8278 |
| 256 | 0.7422 | 0.5537 | 0.8196 | 0.6341 |
| 1024 | 0.7420 | 0.5537 | 0.6639 | 0.6039 |

The axis slices explain where the clipped high-cost shift happens. Low-stability states stay near `0.742` even at high `w`, while high-stability states move down to about `0.554` because they can tolerate long intervals. Difficulty still matters once cost is high, but the no-sub-0.5 action constraint prevents the previous near-abandonment of hard states: at `w=1024`, the high-difficulty bin averages `0.6039` instead of collapsing below `0.5`.

Stationary finite sparse-weight and model-size compression:

The newer stationary finite compression artifacts are under `artifacts/single_card_tradeoff/stationary_finite_cost_weight_ablation/` and `artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/`. These runs keep the clipped 11-action grid, evaluate against `fsrs6_default` with 10,000 particles, and use the standard 17 evaluation scalarization weights. The cost-weight ablation changes only the teacher training weights; all three models below use the same 1,452-parameter `oracle_stationary` residual `16x2` architecture.

| train weights | parameters | eval seeds | `time_regret_auc` | relative regret | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,2,4,8,16,32,64,128,256,512,1024` | 1,452 | 3 | -3.4043 +/- 0.0451 | -23.69% +/- 0.27% | 94.31% +/- 0.31% |
| `0,4,16,64,256,1024` | 1,452 | 3 | -3.3352 +/- 0.0395 | -23.53% +/- 0.22% | 96.88% +/- 0.17% |
| `0,16,64,256,1024` | 1,452 | 3 | -3.3904 +/- 0.0572 | -23.67% +/- 0.35% | 95.01% +/- 0.26% |

The five-weight schedule `0,16,64,256,1024` is a good sparse teacher-weight candidate for stationary finite distillation. It does not reduce checkpoint parameters by itself, but it preserves default-baseline relative regret and span coverage while cutting the number of teacher policies that must be solved.

Using that five-weight teacher schedule, the direct model-size ablation gives:

| candidate | parameters | reduction vs 1,452 | epochs | eval seeds | teacher agreement | relative regret | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual:16:2` | 1,452 | 0.0% | 64 | 3 | 73.76% | -23.67% +/- 0.35% | 95.01% +/- 0.26% |
| `residual:12:2` | 900 | 38.0% | 64 | 1 | 73.12% | -23.58% | 97.71% |
| `residual:10:2` | 672 | 53.7% | 64 | 1 | 71.45% | -22.68% | 85.66% |
| `residual:8:2` | 476 | 67.2% | 64 | 1 | 71.64% | -22.29% | 86.66% |
| `residual:8:2` | 476 | 67.2% | 128 | 3 | 73.01% | -23.50% +/- 0.25% | 95.27% +/- 0.27% |
| `residual:8:1` | 316 | 78.2% | 64 | 1 | 69.64% | -21.54% | 73.56% |
| `residual:6:1` | 216 | 85.1% | 64 | 1 | 69.45% | -19.68% | 73.68% |

The important correction to the older compression section is that 476 parameters are viable when the small model is trained directly from the stationary finite teacher and given enough epochs. A 64-epoch `residual:8:2` run underfits and loses coverage, but the same 476-parameter architecture at 128 epochs recovers coverage to `95.27%` and keeps relative regret within `0.17` percentage points of the 1,452-parameter sparse baseline. The `residual:8:1` and `residual:6:1` runs are below the current capacity floor: they retain a favorable AUC only over a much narrower span.

The default compressed stationary finite candidate is therefore `oracle_stationary`, `residual:8:2`, 476 parameters, 128 distillation epochs, and train weights `0,16,64,256,1024`. Relative to the 22,528-entry stationary finite policy table, this is a `47.3x` table-to-checkpoint compression; relative to the previous 1,452-parameter checkpoint, it removes `67.2%` of learned parameters.

The older `stationary_finite_no_sub05_compression/` teacher-forced and student-rollout students should be treated as historical compression recipes, not as the current best small-model result. Older exact infinite-oracle, historical PPO-compare, oracle-distill hparam ablation, retention-distill coverage-fix, smoke, legacy stationary-finite compression, and historical PPO view artifacts were not reused as current no-sub-0.5 findings unless their checkpoints were verified to contain only actions `>=0.5` and their tradeoff CSVs were verified against the clipped 11-target baseline. The current report relies on the root default distill/PPO checkpoints, `no_sub05_*` comparison directories, `stationary_finite_cost_weight_ablation/`, `stationary_finite_model_size_ablation/`, `stationary_finite_policy_viz/`, and the six `ppo_ablation` checkpoints listed above. Older directories such as `ablation/`, `ppo_compare/`, `retention_distill_compare/`, `retention_distill_coverage_fix/`, `smoke/`, `stationary_finite_compression/`, `stationary_finite_no_sub05_compression/`, `stationary_finite_no_sub05_distill/`, and legacy `_view` files should be treated as historical unless rerun under the clipped target/action space and the current training recipe.

## Lessons

- Coverage is a first-class metric. Always report `time_regret_auc`, `span_coverage_percent`, and enough endpoint rows to show whether the frontier covers both high-memory and low-cost regimes.
- Desired retention is not a naturally well-conditioned coordinate for an integer-interval teacher. Small retention errors can become large interval errors, especially for high stability, high cost weights, and horizon-terminal actions.
- For continuous desired-retention distillation, train on the implied interval, not just the retention value. The repaired default keeps retention as the action output but supervises `log(interval(retention))`.
- Underpredicting intervals is worse than overpredicting intervals at high cost weights. The effective training recipe weights high-cost underprediction and terminal-action underprediction more heavily.
- Compression claims must be rerun after changing the action grid and after changing the training recipe. The old low-action-space compression sweep overstated the usefulness of small stationary finite students for the clipped action space, while the newer direct sparse-teacher run shows that a 476-parameter `residual:8:2` model can recover coverage if trained longer.
- Student rollout is not automatically better. In the older no-sub-0.5 compression check it improved span coverage versus teacher-forcing, but the 476-parameter student-rollout model lost the default-baseline time-regret advantage. The current best small stationary finite per-user result is direct exact-table distillation with sparse cost weights, equal samples per cost weight, and 128 epochs.
- Removing remaining time at the oracle layer is a powerful structural compression. With 64x32 grids and 11 clipped actions, the unrestricted finite oracle table has 41,113,600 entries, while the stationary finite table has 22,528 entries before any neural distillation.
- The exact stationary finite policy is not just a lower-retention version of the finite oracle. Its action table is cost-sensitive and state-sensitive: with sub-0.5 actions removed, `w=256` remains mixed and bimodal, while `w=1024` concentrates on the new cheapest action `0.50` and loses high-cost scalar objective. That explains why very small stationary finite distills lose span coverage quickly despite the smaller teacher table.
- `oracle_rho4` is a strong compact observation. With the clipped 11-action output head, it gives a 1,468-parameter model enough horizon and stability information to cover the frontier when the loss geometry is right.
- Train cost weights should be sparse but cover scale. For stationary finite distillation, `0,16,64,256,1024` preserved relative regret and coverage in the current 3-seed evaluation; the broader `0 + 2^0..2^10` schedule remains a conservative default for unrestricted finite-oracle distillation. Evaluation should remain denser with `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`.
- Do not train stationary finite distills from raw teacher-forcing event counts without rebalancing cost weights. Low-cost policies produce many more review events, so event-level cross-entropy can drown out high-cost policy boundaries. Uniform exact-table supervision over `(cost_weight, stability, difficulty)` fixed the first-8 user-2 high-cost interpolation failure without adding teacher cost weights.
- Integer interval actions are the most direct continuous-action target for the finite-horizon oracle. Continuous desired retention can work, but it needs interval-aware loss shaping and coverage validation.
