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

Visualize the solved oracle policy's output distribution over states visited by the policy rollout:

```bash
uv run experiments/single_card_tradeoff/oracle_policy_outputs.py --source rollout --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Pass `--source table` to count every nonterminal `(remaining, stability, difficulty)` grid cell equally instead of weighting by rollout visits. The script writes `artifacts/single_card_tradeoff/fsrs6_oracle_policy_outputs.csv` and `.png`.

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

Current compact default baseline:

| scheduler | model | parameters | vs `fsrs6_default` `time_regret_auc` | coverage |
| --- | --- | ---: | ---: | ---: |
| `fsrs6_oracle_distill` | `oracle_rho4`, `residual:16:2` | 1,536 | -3.6443 | 78.8% |
| `fsrs6_oracle_retention_distill` | `oracle_rho4`, `residual:16:2` | 1,536 | -3.4359 | 80.4% |

The repaired continuous desired-retention distill now covers the frontier about as broadly as the discrete distill. Directly against `fsrs6_oracle_distill`, it still has a small positive regret (`+0.1837` deck-minutes/day over 100% overlap in the default comparison), so the discrete distill remains the strongest compact default baseline.

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
- PPO guide ablations need a warmup-only control. A guided PPO checkpoint can look like a reinforcement-learning win even when most of the deployed behavior came from supervised oracle labels, especially for the RNN interval policy.
- For the RNN interval policy, the current default is best interpreted as an oracle-imitation policy with PPO fine-tuning. The dense AUC barely changes from oracle warmup only to full training, while no-guide PPO does not recover the same frontier.
- `oracle_rho4` is a strong compact observation. It gives a 1,536-parameter model enough horizon and stability information to cover the frontier when the loss geometry is right.
- Train cost weights should be sparse but cover scale: use `0 + 2^0..2^10`. Evaluation should remain denser with `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`.
- Integer interval actions are the most direct continuous-action target for the finite-horizon oracle. Continuous desired retention can work, but it needs interval-aware loss shaping and coverage validation.
