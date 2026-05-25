# Single-Card Tradeoff Experiments

This experiment family simulates an iid single-card lifecycle with no daily study-budget constraints. Card-level metrics are linearly scaled to a 10,000-card deck so scheduler frontiers can be compared by expected memorized cards and study minutes per day.

## Quickstart

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_default --particles 10000 --deck-scale 10000
```

When CUDA is available, `tradeoff.py` uses `cuda` by default; pass `--torch-device cpu` to force CPU.

Pass `--env fsrs6 --user-id <id>` to load per-user FSRS-6 weights from `../srs-benchmark`; use `--user-ids 1,2,3` to evaluate multiple users in one vectorized batch. Add `--button-usage ../Anki-button-usage/button_usage.jsonl` to use each user's first/review rating probabilities and learning/review costs in the single-card oracle, PPO, and distillation rollouts. Review-button `long_term_transition` Markov behavior is opt-in with `--review-markov-transition`; the default uses marginal review probabilities only. `--env fsrs6_default` keeps the built-in FSRS-6 parameters while still applying per-user button usage when provided.

For supported FSRS-6 sweeps, desired-retention targets are batched in one vectorized run by default. Multi-user vectorized scheduler runs batch rows over `(user_id, scheduler point)`, where a point is a desired-retention target, fixed interval, or ADR policy. The default targets are `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`; override with `--target-retentions`, or pass `--target-retentions ""` to use the range flags. Single-card target and oracle-action retentions must be at least `0.5`. Fixed-interval sweeps are batched the same way. Plain `--sched fixed` runs intervals `8,16,32,64,128,256,512` by default; override with `--fixed-intervals`. Mixed scheduler families are run as one batch per family. Pass `--target-batch-size 1` to run rows sequentially, or `--target-batch-size 0` to run all rows for a scheduler family in one chunk.

By default, the script writes a pairwise same-target time saved AUC CSV next to the main CSV. `same_target_time_saved_auc` is the average deck-scaled minutes/day saved by the scheduler versus the baseline over their common covered memory-target interval, and `relative_same_target_time_saved_auc_percent` divides that by the baseline time AUC. Positive values mean the scheduler reaches the same memory target faster.

Infer the rollout-level cost-weight ranges implied by the static FSRS6 desired-retention frontier:

```bash
uv run python -m experiments.single_card_tradeoff.cli.fsrs6_implied_cost_weight \
  --results artifacts/single_card_tradeoff/oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/combined_results.csv \
  --out-dir artifacts/single_card_tradeoff/fsrs6_implied_cost_weight/first8
```

This first-phase inverse analysis compares only `fsrs6` desired-retention points
against other `fsrs6` points for the same user and rollout settings. Supported
points get a `lambda_min/lambda_max` interval for the scalar objective
`card_expected_retrievability - lambda * card_minutes_per_day`; unsupported
points get a best-fit lambda and regret.

## FSRS6 ADR Policies

`tradeoff.py` can evaluate trained ADR schedulers in the same iid single-card lifecycle as the static FSRS baselines and distilled oracle schedulers. Pass one policy JSON with `--fsrs6-adr-policy`, or expand a full trained portfolio with `--fsrs6-adr-policy-root`, `--fsrs6-adr-train-run-root`, or `--fsrs6-adr-policy-manifest`. If no ADR source is passed and the local first-eight portfolio artifact exists, the runner uses `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1`.

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff \
  --env fsrs6 --user-id 1 \
  --sched fsrs6,fsrs6_adr,fsrs6_oracle_stationary_finite_distill \
  --fsrs6-adr-train-run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1 \
  --oracle-stationary-finite-distill-policy artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/user_1_policy.pt
```

ADR rows include `fsrs6_adr_policy`, `fsrs6_adr_baseline_desired_retention`, `fsrs6_adr_lambda_value`, and `fsrs6_adr_policy_index` columns so a portfolio frontier can be compared directly with static baseline DR rows and the 476-parameter stationary finite distill frontier.

The first-eight ADR versus 476-parameter distill comparison is configured in TOML and writes combined cross-user summary tables and plots:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/adr_vs_476_tradeoff_first8_users.toml \
  --stage evaluate
```

The formal runner reads semantic config sections and `[[tasks]]` entries, then
writes stage records under each workflow root. Supported stages are `dry-run`,
`preflight`, `train`, `evaluate`, `analyze`, `benchmark`, `visualize`, `report`,
and `all`. Legacy `run_tradeoff_config.py` remains the implementation behind the
`tradeoff_config` task kind; checked-in configs no longer use `[[commands]]` or
`expected_outputs` as the primary contract. For configs with multiple users, that
task still calls `tradeoff.py --user-ids ...` once by default, then splits the
combined CSVs back into per-user compatibility files.

The train-weight control workflow is now generated from the compact semantic
sections in its TOML:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/distill_train_weights_1_4_control_markov_off.toml \
  --stage all
```

## Native FSRS6 ADR Training

Train one FSRS6 ADR policy per `(user, cost weight)` directly on the
single-card objective:

```bash
uv run python -m experiments.single_card_tradeoff.cli.fsrs6_adr_train_multiuser \
  --env fsrs6_default \
  --user-ids 1,2,3,4,5,6,7,8 \
  --cost-weights 16,32,64,128,256,512,1024 \
  --torch-device cuda \
  --out-dir artifacts/single_card_tradeoff/fsrs6_adr_single_card_direct_multiuser \
  --no-progress
```

The trainer writes per-job `train-overfit/train_outputs/user_<id>/lambda_<w>/`
artifacts, including `policy.json`, `metadata.json`, and `metrics.json`, plus
root `summary.csv`, `train_history.csv`, and `policy_manifest.toml` files. Point
`tradeoff.py` at the run root with `--fsrs6-adr-train-run-root ...` or at the
manifest with `--fsrs6-adr-policy-manifest ...`.

## UVFA PPO

UVFA PPO single-card experiment, goal-conditioned over FSRS-6 target-retention actions:

```bash
uv run python -m experiments.single_card_tradeoff.cli.uvfa_ppo --days 1825 --eval-particles 10000 --deck-scale 10000
```

The PPO objective is `card_expected_retrievability - goal_cost_weight * card_minutes_per_day`, with default training goal weights `16,32,64,128,256,512,1024`; the standard tradeoff sweep evaluates scalarized learned policies and oracles at `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024` by default. It normalizes advantages per goal, uses rich state features and a residual policy/value network with hidden size 64 and depth 3 by default, and uses a finite-horizon FSRS grid oracle as the default warmup/regularization guide. Pass `--guide-policy static` for the older static-FSRS target prior, or `--guide-policy none` for plain PPO. The script writes a comparable CSV/plot under `artifacts/single_card_tradeoff/`, includes fixed-interval and static-FSRS reference curves, and reports whether the learned UVFA policy beats the selected baseline. The default pass/fail baseline is the best fixed interval; use `--baseline fsrs` or `--baseline overall` for stricter static-FSRS comparisons.

After training a policy, include it in the standard single-card Pareto sweep with `--sched uvfa_ppo`:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_default,fixed,uvfa_ppo --uvfa-ppo-policy artifacts/single_card_tradeoff/uvfa_ppo_policy.pt
```

Search UVFA PPO model-scale hyperparameters:

```bash
uv run python -m experiments.single_card_tradeoff.cli.uvfa_ppo_hparam_search --days 1825 --eval-particles 3000 --save-models
```

## Recurrent Interval PPO

Recurrent UVFA PPO over continuous log-interval actions:

```bash
uv run python -m experiments.single_card_tradeoff.cli.uvfa_ppo_rnn_interval --days 1825 --eval-particles 10000 --deck-scale 10000
```

This variant uses a GRU belief-state encoder over the event observation sequence, concatenates the hidden state with the sampled cost-weight preference, and trains Gaussian PPO in log days. The environment exponentiates the sampled action, rounds it to physical review days, and clamps the interval to `--max-interval-days` (default `days * 4`). Its belief observation does not expose the simulator's internal stability or difficulty state. By default, training uses a finite-horizon FSRS grid oracle as a continuous log-interval warmup and PPO regularization guide; pass `--guide-policy static` or `--guide-policy none` to ablate it.

After training a recurrent interval policy, include it in the standard single-card Pareto sweep with `--sched uvfa_ppo_rnn_interval`:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_default,fixed,uvfa_ppo_rnn_interval --uvfa-ppo-rnn-interval-policy artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_policy.pt
```

## Discrete Oracle Distillation

Train a pure FSRS-6 oracle distillation baseline, then compare it directly with the DP oracle and UVFA PPO:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_distill --days 1825 --eval-particles 10000 --deck-scale 10000
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle,fsrs6_oracle_distill,uvfa_ppo --oracle-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_distill_policy.pt --uvfa-ppo-policy artifacts/single_card_tradeoff/uvfa_ppo_policy.pt
```

`oracle_distill.py` defaults to the `oracle_rho4` observation (`log remaining/stability ratio`, difficulty, goal cost weight, and stability), the `residual:16:2` architecture, and CUDA when available. Its default teacher weights are `0,1,2,4,8,16,32,64,128,256,512,1024`, so the zero-cost edge is trained directly while the standard tradeoff evaluation still probes intermediate weights. This model has 1,536 parameters, about 30% of the previous `residual:32:2` default. In the default FSRS-6 10k-particle, 3-seed comparison it slightly improved pairwise same-target time saved AUC against the previous default (`+0.0587` deck-minutes/day with 100% overlap) while preserving the broader frontier coverage that short small-model training missed. Pass `--obs-mode oracle` to train the older 4-feature oracle observation, or `--obs-mode rich` to train on the larger rollout observation instead. The script also accepts the same `--env fsrs6 --user-id <id>` and `--button-usage` options as the single-card tradeoff runner.

To rerun the discrete oracle distillation model-size search:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_distill_hparam_search --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --eval-particles 3000 --save-models
```

## Grid Oracle

Estimate a finite-horizon FSRS-6 grid oracle frontier:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_frontier --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Use the same finite-horizon oracle policy table as a scheduler in the standard single-card sweep:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle --oracle-cost-weights 16,32,64,128,256,512,1024
```

The oracle script uses expected Bellman backups over a `(log stability, difficulty)` grid and discrete desired-retention actions. In `tradeoff.py`, `fsrs6_oracle` solves all requested scalarization weights in one batched DP pass and evaluates them in one batched Monte Carlo rollout. It writes a single-card CSV/plot under `artifacts/single_card_tradeoff/` and, by default, includes static-FSRS reference rows evaluated with Monte Carlo particles.

The same desired-retention action space also has an infinite-horizon average-reward oracle. It removes the remaining-horizon state and solves a stationary SMDP policy over `(log stability, difficulty)`:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_infinite --oracle-cost-weights 16,32,64,128,256,512,1024
```

Train the stationary oracle distillation policy, then compare it in the same finite lifecycle tradeoff evaluation:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_infinite_distill --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle,fsrs6_oracle_infinite,fsrs6_oracle_infinite_distill --oracle-infinite-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_policy.pt
```

The infinite oracle optimizes long-run average `retrievability - goal_cost_weight * minutes` and reports its stationary gain, policy-iteration count, and residual to stdout. The tradeoff CSV still reports the existing finite lifecycle metrics so it remains comparable with the finite-horizon oracle and distilled policies.

For a stationary policy optimized against the finite 1825-day new-card lifecycle, use `fsrs6_oracle_stationary_finite`. It keeps the same `(stability, difficulty, goal cost weight)` policy input as the infinite oracle, initializes from the finite-horizon oracle's visited-state actions, then runs occupancy-weighted policy iteration on the finite lifecycle objective:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_stationary_finite --oracle-cost-weights 16,32,64,128,256,512,1024
```

Train and compare the corresponding distilled stationary finite-lifecycle policy:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_stationary_finite,fsrs6_oracle_stationary_finite_distill --oracle-stationary-finite-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
```

The stationary finite oracle reports finite-lifecycle objective, policy-iteration count, and residual to stdout. The distilled checkpoint uses `policy_type=fsrs6_oracle_stationary_finite_distill` and defaults to the compressed `oracle_stationary` residual `8x2` policy: 476 parameters, 128 distillation epochs, teacher cost weights `0,4,16,64,256,1024`, and `uniform_table` exact-policy supervision. Its observation remains `stability`, `difficulty`, and goal cost weight only.

For per-user benchmark distillation, `oracle_stationary_finite_distill_multiuser.py --per-user-models` trains one independent stationary finite distill policy per FSRS-6 user in a single Python process. The exact teacher uses `FSRS6BatchedStationaryFiniteOracle`, whose policy shape is `[user, cost_weight, stability, difficulty]`; `--oracle-teacher-user-batch-size 0` is the default and solves all requested users in one batched DP call. The student training then stacks `U` ordinary `PolicyValueNet` states with `torch.func.stack_module_state` and uses `vmap` over the user dimension, so each user has a separate 476-parameter model while the ensemble trains and evaluates as one batch. The per-user default supervision is `--per-user-supervision uniform_table`: every train step samples the exact stationary policy table with equal mass on each teacher cost weight, rather than weighting labels by rollout event counts. Use `--per-user-supervision teacher_occupancy` to sample states by the exact stationary teacher occupancy within each teacher cost weight, or `--per-user-supervision rollout` only to reproduce the older teacher-forcing baseline. The default path without `--per-user-models` remains the older shared-student diagnostic baseline.

To run the first-eight discrete stationary finite distill occupancy ablation with the same budget on both arms:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/stationary_finite_distill_teacher_occupancy_ablation.toml \
  --stage train
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/stationary_finite_distill_teacher_occupancy_ablation.toml \
  --stage evaluate
```

The continuous desired-retention stationary finite distill has the same multi-user shape via `oracle_continuous_stationary_finite_distill_multiuser.py`. Its teacher uses `FSRS6BatchedContinuousStationaryFiniteOracle`, whose policy table stores continuous retentions as `[user, cost_weight, stability, difficulty]`; the exact DP still enumerates attainable rounded intervals internally, and the student loss supervises implied log-interval behavior plus retention logits. It writes per-user checkpoints with `policy_type=fsrs6_oracle_continuous_stationary_finite_distill` for the tradeoff runner:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_continuous_stationary_finite_distill_multiuser --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 --button-usage ../Anki-button-usage/button_usage.jsonl --cost-weights 0,4,16,64,256,1024 --eval-cost-weights 0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --epochs 128 --steps-per-epoch 64 --table-samples-per-weight 256 --eval-particles 10000 --torch-device cuda --out-dir artifacts/single_card_tradeoff/continuous_stationary_finite_distill_first8_eval_weights_add_025_05_markov_off --no-progress
```

Use `--loss-weighting baseline|underpred|qgap|underpred_qgap` to run the continuous distill loss ablation with the same teacher and training budget. `baseline` is the old unweighted implied-interval plus retention-logit loss. `underpred` adds high-cost and terminal underprediction weights, `qgap` computes exact stationary teacher interval-margin weights after the DP solve, and `underpred_qgap` combines both. The multi-user continuous default is `underpred` with underprediction weights `8/32`, a `16`-wide residual student, and teacher-occupancy table sampling.

Use `--table-sampling uniform_table|teacher_occupancy|mixed` to choose where continuous table supervision samples states. `teacher_occupancy` is the default and samples states by exact stationary teacher occupancy. `uniform_table` preserves the original full-table distribution, and `mixed` combines occupancy samples with uniform table samples; tune the uniform share with `--mixed-table-uniform-fraction` (default `0.5`).

All oracle DP entrypoints now cache per `(user, weight)` under `artifacts/single_card_tradeoff/dp_cache` by default. Pass `--no-dp-cache` to disable it or `--refresh-dp-cache` to force recomputation.

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 --button-usage ../Anki-button-usage/button_usage.jsonl --per-user-models --out-dir artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu --no-progress
```

On the first eight benchmark users, the uniform-table per-user run wrote `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu/` with 8 checkpoints, 476 parameters per checkpoint, 3,808 total trainable ensemble parameters, `teacher_s=58.82`, `train_s=34.89`, `agreement_s=0.01`, `eval_s=66.94`, mean final CE `0.69721`, mean train table agreement `72.41%`, and mean full-table agreement `72.39%` on the CUDA test environment. Mean `fsrs6` baseline span coverage was `97.55%`, and mean relative time saved AUC was `12.36%` over the eight users. The older rollout teacher-forcing per-user artifact `stationary_finite_distill_first8_users_per_user_batched/` had mean span coverage `95.82%`, mean relative time saved AUC `4.32%`, and a user-2 high-cost interpolation failure (`-18.20%` relative time saved AUC). Uniform exact-table supervision fixes that failure: user 2 improves to `96.18%` coverage and `15.30%` relative time saved AUC.

To reproduce the exact teacher versus per-user distill comparison consumed by the report, run the same multi-user CLI in eval-only mode. This solves the exact stationary finite table at each evaluation cost weight, loads the saved per-user checkpoints, and writes `results.csv`, `regret_auc.csv`, `summary.csv`, and `mean_summary.csv`:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 --button-usage ../Anki-button-usage/button_usage.jsonl --eval-exact-vs-distill --distill-dir artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_per_user_uniform_table_supervision_fsrs6_baseline_gpu --out-dir artifacts/single_card_tradeoff/stationary_finite_exact_vs_distill_first8_users --torch-device cuda --no-progress
```

The multi-user evaluator now batches retention and cost-weight groups by default (`--eval-group-batch-size 0`); set `--eval-group-batch-size 1` to reproduce the older per-group rollout shape. On CUDA with the existing first-eight checkpoints and 1,000 particles per user/group, batching reduced static-retention baseline evaluation from `87.21s` to `24.70s` (`3.53x`) and per-user distill cost-weight evaluation from `99.20s` to `21.17s` (`4.69x`). This affects only the multi-user single-card evaluation path; the event and vectorized simulator engines are unchanged.

When `--torch-device` resolves to CUDA, `tradeoff.py`, the multi-user distill script, and the low-parameter direct-search script start the same GPU memory monitor used by the rl_scheduler experiment runner. Each run writes `gpu_monitor/gpu_memory.jsonl`, `gpu_monitor/summary.json`, and `performance_summary.json` under the configured output directory; pass `--no-gpu-monitor-enabled` only for CPU or diagnostic runs where those artifacts are not needed.

For a direct low-parameter policy-search baseline, `low_param_direct_policy_search_multiuser.py` optimizes one independent 7-parameter monotone stationary policy per user with cross-entropy-method search. The policy outputs a continuous desired retention in `[0.5,0.98]` from `(stability, difficulty, cost weight)` and is evaluated against `fsrs6` and the per-user stationary finite distill baseline:

```bash
uv run python -m experiments.single_card_tradeoff.cli.low_param_direct_policy_search_multiuser --env fsrs6 --user-ids 1,2,3,4,5,6,7,8 --button-usage ../Anki-button-usage/button_usage.jsonl --generations 64 --population-size 32 --elite-count 8 --train-particles 64 --eval-particles 10000 --torch-device cuda --out-dir artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users --no-progress
```

The first-eight 7-parameter run wrote `artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users/` with 56 total trainable parameters, `train_s=210.81`, and `eval_s=94.11`. Against `fsrs6`, it reached mean span coverage `72.37%`, mean same-target time saved AUC `2.7373`, and mean relative time saved AUC `7.50%`. Directly against `fsrs6_oracle_stationary_finite_distill_per_user`, it reached mean span coverage `68.48%`, mean same-target time saved AUC `-1.1129`, and mean relative time saved AUC `-4.43%`. A dense-training-weight rerun at `low_param_direct_policy_search_first8_users_dense_weights/` kept the same 7 parameters but reduced coverage to `66.45%` vs `fsrs6` and `62.74%` vs distill, so the current 7-parameter monotone family is much smaller but too restrictive to match the 476-parameter distill frontier.

The shared-student diagnostic artifact is `artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_batched/`. It used one 476-parameter model for all eight users and reported mean span coverage `96.71%` and mean relative time saved AUC `6.38%`. The user-2 failure in the rollout per-user path was caused by event-count sampling bias: `w=0` produces many more review events than high cost weights, so the cross-entropy loss was dominated by low-cost actions even though reset-time cost weights were uniform. The current per-user default removes that bias by supervising the exact `[user, cost_weight, stability, difficulty]` table with equal samples per sparse teacher cost weight.

Visualize the solved oracle policy's output distribution over states visited by the policy rollout:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_policy_outputs --source rollout --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 16,32,64,128,256,512,1024
```

Pass `--source table` to count every nonterminal `(remaining, stability, difficulty)` grid cell equally instead of weighting by rollout visits. The script writes `artifacts/single_card_tradeoff/fsrs6_oracle_policy_outputs.csv` and `.png`.

Visualize the stationary finite-lifecycle oracle directly over its stationary `(stability, difficulty, goal cost weight)` policy table:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_stationary_finite_policy_viz --days 1825 --s-grid-size 64 --d-grid-size 32 --cost-weights 0,4,16,64,256,1024 --action-retentions 0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98 --distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_policy.pt
```

This writes `action_summary.csv`, `binned_actions.csv`, `grid_actions.csv`, `findings.md`, `action_distribution.png`, and `policy_heatmaps.png` under `artifacts/single_card_tradeoff/stationary_finite_policy_viz/`. When `--distill-policy` is provided it also writes `distill_action_summary.csv`, `distill_binned_actions.csv`, `distill_grid_actions.csv`, `distill_exact_comparison.csv`, `distill_action_distribution.png`, `distill_policy_heatmaps.png`, and `distill_exact_difference_heatmaps.png`. Use `--selected-weights` to choose which weights appear in the `(s,d)` heatmap panel.

To reproduce the first-eight per-user exact, 476-parameter, and residual:4:1 policy-grid comparison:

```bash
uv run python -m experiments.single_card_tradeoff.cli.first8_r4d1_vs_476_policy_viz --no-progress
```

This writes exact, 476-parameter, residual:4:1, and difference heatmaps under `artifacts/single_card_tradeoff/first8_r4d1_vs_476_policy_viz/`, plus pairwise CSV summaries for `476_vs_exact`, `r4d1_vs_exact`, and `r4d1_vs_476`.

## Interval Oracle And Distillation

The single-card sweep supports an integer-interval FSRS-6 oracle that enumerates every feasible next interval `1..remaining+1`, where `remaining+1` means no further review before the horizon:

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_interval --oracle-cost-weights 16,32,64 --oracle-interval-chunk-size 64
```

Train the 4D log-interval distillation policy, then include it in the same sweep:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_interval_distill --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_interval_distill --oracle-interval-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt
```

The interval distillation model keeps the same 4D scalar-output network. Its default `residual:64:3` sweet-spot architecture has 25,857 parameters, about 45% of the earlier `residual:96:3` model, while preserving the time-saved advantage in the default FSRS-6 comparison. The more aggressive `residual:32:2` candidate has 4,609 parameters, about 8% of the earlier model, but gives up a small amount of pairwise same-target time saved AUC against `fsrs6_oracle_distill`. The default training loss weights underpredicted intervals more heavily for high cost weights, mixes in student-rollout states after warmup, and snaps predicted intervals near the remaining horizon to the terminal no-more-review action. These are fixed training/inference rules and do not add learned parameters.

To rerun the model-size search:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_interval_distill_hparam_search --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --eval-particles 3000 --save-models
```

## Continuous Desired-Retention Distillation

The integer-interval oracle can also be distilled into a continuous desired-retention policy. This keeps the compact `oracle_rho4` residual `16x2` default and executes by converting the predicted retention into the nearest integer review interval inside the day-level simulator. Its default training is interval-aware: the model still outputs retention, but the loss penalizes the log interval implied by that retention, weights high-cost underprediction heavily, and uses student-rollout states after warmup to preserve frontier coverage.

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_retention_distill --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_distill,fsrs6_oracle_retention_distill --oracle-retention-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt
```

The continuous desired-retention oracle solves the same single-card lifecycle
while constraining actions to a retention interval, defaulting to `[0.5,0.98]`.
The DP enumerates only rounded day intervals attainable from that retention
range, stores the canonical continuous retention in the policy table, and
executes by interpolating retention before the simulator rounds it back to days.

```bash
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_continuous_retention --oracle-cost-weights 16,64 --oracle-continuous-retention-min 0.5 --oracle-continuous-retention-max 0.98
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_continuous_stationary_finite --oracle-cost-weights 16,64 --oracle-continuous-interval-chunk-size 64
```

Analyze how a hidden uniformly distributed lifecycle end date changes the
continuous-retention policy surface relative to the fixed-horizon oracle:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/continuous_uniform_h_policy_analysis_first8.toml \
  --stage analyze
```

Solve the best stationary approximation under the same hidden Uniform-H
terminal model and compare it with `FSRS6ContinuousStationaryFiniteOracle`:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/continuous_uniform_h_stationary_analysis_first8.toml \
  --stage analyze
```

Train the compact stationary finite continuous-retention student, then evaluate
it in the standard sweep:

```bash
uv run python -m experiments.single_card_tradeoff.cli.oracle_continuous_stationary_finite_distill --days 1825 --oracle-s-grid-size 64 --oracle-d-grid-size 32 --oracle-interval-chunk-size 64 --model-out artifacts/single_card_tradeoff/fsrs6_oracle_continuous_stationary_finite_distill_policy.pt
uv run python -m experiments.single_card_tradeoff.cli.tradeoff --env fsrs6_default --sched fsrs6_oracle_continuous_stationary_finite_distill --oracle-continuous-stationary-finite-distill-policy artifacts/single_card_tradeoff/fsrs6_oracle_continuous_stationary_finite_distill_policy.pt
```

## Findings

`single_card_tradeoff` is best understood as a frontier experiment, not a full deck scheduler benchmark. It removes daily budget constraints and isolates the memory-time tradeoff for one iid card lifecycle. The metrics can be deck-scaled, but they should not be read as a complete workload simulation.

The formal machine-generated reports for the current artifacts are published under
[`docs/single_card_tradeoff/experiments`](../../docs/single_card_tradeoff/experiments),
with one independent report per experiment and a short index at
[`2026-05-17-index.md`](../../docs/single_card_tradeoff/experiments/2026-05-17-index.md).
Regenerate them with:

```bash
uv run python -m experiments.single_card_tradeoff.cli.run_experiment \
  --config experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml \
  --stage report
```

The suite config is intentionally thin and includes one smaller profile per
experiment from `experiments/single_card_tradeoff/configs/reports/`; the
first-eight stationary finite profile remains at
`experiments/single_card_tradeoff/configs/stationary_finite_first8_report.toml`.

The most useful summary metric is `same_target_time_saved_auc`, but it is only meaningful together with `span_coverage_percent`. A positive `same_target_time_saved_auc` means a scheduler uses fewer deck-scaled minutes/day than the baseline at the same memory target over their common memory interval. Low coverage means the comparison only covers a narrow part of the frontier.

No-sub-0.5 action-space rerun:

The current default target/action retention grid is `0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.93,0.96,0.98`. Checkpoints whose action grid includes values below `0.5` now fail validation in the tradeoff runner, and the report below only uses artifacts that were verified to use the new action grid.

The current compact comparison uses `fsrs6_default`, 1825 days, 10,000 particles, `deck_scale=10000`, and the standard scalarization weights for each scheduler implementation. The combined artifacts are in `artifacts/single_card_tradeoff/no_sub05_distill_compare/`. The default desired-retention baseline rows come from `artifacts/single_card_tradeoff/no_sub05_default_vs_stationary_finite_distill/results.csv`; the learned-policy evaluations are `artifacts/single_card_tradeoff/fsrs6_oracle_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_stationary_finite_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_infinite_distill_results.csv`, `artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_results.csv`, `artifacts/single_card_tradeoff/uvfa_ppo_results.csv`, and `artifacts/single_card_tradeoff/uvfa_ppo_rnn_interval_results.csv`.

| scheduler | policy input | parameters | vs `fsrs6_default` `same_target_time_saved_auc` | relative time saved | coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `fsrs6_oracle_distill` | `oracle_rho4`, `residual:16:2` | 1,468 | 3.6263 | 24.18% | 86.3% |
| `fsrs6_oracle_stationary_finite_distill` | `oracle_stationary`, `residual:16:2` | 1,452 | 3.4070 | 23.70% | 94.7% |
| `fsrs6_oracle_retention_distill` | `oracle_rho4`, `residual:16:2` | 1,468 | 3.3248 | 21.79% | 83.5% |
| `fsrs6_oracle_infinite_distill` | `oracle_stationary`, `residual:16:2` | 1,452 | 4.8242 | 14.66% | 9.0% |
| `uvfa_ppo` | `rich`, `residual:64:3` | 27,148 | 3.4886 | 24.89% | 99.2% |
| `uvfa_ppo_rnn_interval` | `belief`, `GRU:128` | 87,559 | 3.0423 | 21.67% | 69.2% |

Directly against `fsrs6_oracle_distill`, the current `fsrs6_oracle_stationary_finite_distill` has `-0.0944` deck-minutes/day same-target time saved AUC, `-0.83%` relative time saved, and `99.997%` coverage. The continuous desired-retention distill has `-0.3542` deck-minutes/day, `-3.07%` relative time saved, and `96.7%` coverage. The default `uvfa_ppo` checkpoint has `+0.0639` deck-minutes/day, `+0.56%` relative time saved, and `99.9%` coverage against `fsrs6_oracle_distill`; it is slightly ahead on the shared span, but it uses an order of magnitude more parameters. The recurrent interval PPO has `-0.1289` deck-minutes/day, `-1.19%` relative time saved, and `80.2%` coverage against `fsrs6_oracle_distill`. Reversing the stationary-finite baseline gives `fsrs6_oracle_distill` `+0.0944` deck-minutes/day over `91.1%` of the stationary-finite-distill span. The stationary finite distill is slightly worse than the unrestricted finite-horizon distill on shared span, but it covers more of the `fsrs6_default` retention span after clipping actions below `0.5`.

The average-reward infinite distill should not be judged by its positive default-baseline AUC alone. Its frontier covers only `9.0%` of the default retention span after the action floor is applied, so the AUC is computed over a narrow high-memory region. Directly against `fsrs6_oracle_distill`, it has `-15.59%` relative time saved over only `10.4%` of the oracle-distill span.

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

| candidate | guide setup | parameters | train transitions | updates | train time | fixed-baseline check | vs `fsrs6_default` relative time saved | coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `uvfa_ppo_no_guide` | no guide | 27,148 | 2,359,296 | 36 | 28.54s | pass | -22.74% | 85.9% |
| `uvfa_ppo_static_guide` | static guide | 27,148 | 2,359,296 | 36 | 31.93s | fail | -8.52% | 54.0% |
| `uvfa_ppo_oracle_warmup_only` | oracle warmup, no PPO updates | 27,148 | 0 | 0 | 188.67s | fail | -16.40% | 64.2% |
| `uvfa_ppo_rnn_interval_no_guide` | no guide, no warmup prior | 87,559 | 2,359,296 | 36 | 40.58s | fail | +0.93% | 87.8% |
| `uvfa_ppo_rnn_interval_static_guide` | static guide | 87,559 | 2,359,296 | 36 | 48.80s | fail | -2.22% | 42.7% |
| `uvfa_ppo_rnn_interval_oracle_warmup_only` | oracle warmup, no PPO updates | 87,559 | 0 | 0 | 196.54s | pass | -21.31% | 72.6% |

The ablation takeaway did not change after removing actions below `0.5`: the default oracle-guided PPO runs remain the strongest PPO checkpoints in the current report. Discrete PPO without a guide still beats `fsrs6_default`, but it trails the default oracle-guided checkpoint on both relative time saved and coverage. The RNN interval warmup-only checkpoint is useful as a prior-quality check, but its coverage is lower than the default trained RNN checkpoint.

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

| train weights | parameters | eval seeds | `same_target_time_saved_auc` | relative time saved | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0,1,2,4,8,16,32,64,128,256,512,1024` | 1,452 | 3 | 3.4043 +/- 0.0451 | 23.69% +/- 0.27% | 94.31% +/- 0.31% |
| `0,4,16,64,256,1024` | 1,452 | 3 | 3.3352 +/- 0.0395 | 23.53% +/- 0.22% | 96.88% +/- 0.17% |
| `0,16,64,256,1024` | 1,452 | 3 | 3.3904 +/- 0.0572 | 23.67% +/- 0.35% | 95.01% +/- 0.26% |

The historical five-weight schedule `0,16,64,256,1024` is a good sparse teacher-weight candidate for stationary finite distillation. The current default uses `0,4,16,64,256,1024`: the add-4-only row passed all gates, kept the user-2 repair, and had the best mean relative AUC among the passing rows.

Rerun the model-size ablation with all non-network variables aligned to the
current default stationary finite distill recipe:

```bash
uv run python -m experiments.single_card_tradeoff.cli.stationary_finite_model_size_ablation --torch-device cuda --no-progress
```

The rerun uses the six-weight teacher schedule `0,4,16,64,256,1024`, the
clipped 11-action grid, `uniform_table` supervision, 128 epochs, 64 steps per
epoch, 10,000 eval particles, and eval seeds `42,43,44` for every network
size:

| candidate | parameters | reduction vs 1,452 | epochs | eval seeds | teacher agreement | relative time saved | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual:16:2` | 1,452 | 0.0% | 128 | 3 | 77.45% | 23.00% +/- 0.23% | 96.57% +/- 0.17% |
| `residual:12:2` | 900 | 38.0% | 128 | 3 | 77.00% | 23.30% +/- 0.33% | 97.60% +/- 0.26% |
| `residual:10:2` | 672 | 53.7% | 128 | 3 | 75.67% | 23.20% +/- 0.33% | 98.11% +/- 0.25% |
| `residual:8:2` | 476 | 67.2% | 128 | 3 | 74.98% | 22.15% +/- 0.28% | 98.41% +/- 0.18% |
| `residual:8:1` | 316 | 78.2% | 128 | 3 | 74.52% | 22.88% +/- 0.45% | 98.53% +/- 0.23% |
| `residual:6:1` | 216 | 85.1% | 128 | 3 | 73.71% | 22.10% +/- 0.33% | 98.37% +/- 0.17% |

Rerun the quick sub-216 and structured sweep with the same recipe:

```bash
uv run python -m experiments.single_card_tradeoff.cli.stationary_finite_model_size_ablation --torch-device cuda --candidate-set sub216 --summary-prefix sub216 --no-progress
```

This adds smaller residual, MLP, linear, and quadratic policies under
`sub216_summary.csv` without overwriting the aligned residual table:

| candidate | family | parameters | epochs | teacher agreement | relative time saved | coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `residual:5:1` | residual | 172 | 128 | 71.41% | 20.00% +/- 0.34% | 95.30% +/- 0.29% |
| `residual:4:1` | residual | 132 | 128 | 69.36% | 19.10% +/- 0.32% | 85.80% +/- 0.36% |
| `residual:3:1` | residual | 96 | 128 | 51.90% | -74.46% +/- 10.13% | 81.88% +/- 0.30% |
| `mlp:8` | MLP | 212 | 128 | 70.13% | 17.47% +/- 0.11% | 98.56% +/- 0.18% |
| `mlp:6` | MLP | 150 | 128 | 66.39% | 13.71% +/- 0.22% | 96.66% +/- 0.25% |
| `mlp:4` | MLP | 96 | 128 | 61.17% | 6.60% +/- 0.83% | 89.78% +/- 0.33% |
| `linear` | structured | 44 | 128 | 43.15% | -86.56% +/- 2.73% | 98.79% +/- 0.07% |
| `quadratic` | structured | 110 | 128 | 50.71% | 10.05% +/- 0.63% | 99.36% +/- 0.17% |

The sub-216 sweep did not find a replacement for the 216-parameter
`residual:6:1` baseline. `residual:5:1` is the best time-saved sub-216 row, but it
loses about 2.1 relative-time-saved points and 3.1 coverage points versus
`residual:6:1`. `mlp:8`, `linear`, and `quadratic` preserve broad span coverage,
but their time saved AUC is much weaker, so high coverage alone is not enough to
justify further compression. The structured policies look like useful lower
bounds, not practical replacements.

The longer-epoch follow-up is recorded in
`docs/single_card_tradeoff/experiments/2026-05-18-stationary_finite_epoch_extension.md`.

The important correction to the older compression section is that the earlier
64-epoch `residual:8:1` and `residual:6:1` rows were undertrained. When epochs
and eval seeds are aligned to the current default recipe, the 316-parameter
`residual:8:1` student recovers frontier coverage and remains competitive with
larger students. The 216-parameter `residual:6:1` student also keeps coverage,
but its relative time saved is weaker in this rerun. Because this ablation still
uses one training seed per architecture, the practical default remains
`oracle_stationary`, `residual:8:2`, 476 parameters, 128 distillation epochs,
and train weights `0,4,16,64,256,1024`. The 316-parameter student remains the
lowest-risk 128-epoch compression candidate.

Visualize the aligned model-size policies side-by-side against the exact
stationary finite oracle:

```bash
uv run python -m experiments.single_card_tradeoff.cli.stationary_finite_arch_policy_viz --torch-device cuda --no-progress
```

This writes combined policy heatmaps, exact-difference heatmaps, mean-action
curves, and `arch_policy_summary.csv` under
`artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/policy_viz/`.
For the sub-216 sweep:

```bash
uv run python -m experiments.single_card_tradeoff.cli.stationary_finite_arch_policy_viz --torch-device cuda --candidate-set sub216 --out-dir artifacts/single_card_tradeoff/stationary_finite_model_size_ablation/policy_viz_sub216 --no-progress
```

That run wrote the same plot set plus `findings.md`, where `residual:5:1` had
the highest sub-216 exact-table match at 71.41%.

The older `stationary_finite_no_sub05_compression/` teacher-forced and student-rollout students should be treated as historical compression recipes, not as the current best small-model result. Older exact infinite-oracle, historical PPO-compare, oracle-distill hparam ablation, retention-distill coverage-fix, smoke, legacy stationary-finite compression, and historical PPO view artifacts were not reused as current no-sub-0.5 findings unless their checkpoints were verified to contain only actions `>=0.5` and their tradeoff CSVs were verified against the clipped 11-target baseline. The current report relies on the root default distill/PPO checkpoints, `no_sub05_*` comparison directories, `stationary_finite_cost_weight_ablation/`, `stationary_finite_model_size_ablation/`, `stationary_finite_policy_viz/`, and the six `ppo_ablation` checkpoints listed above. Older directories such as `ablation/`, `ppo_compare/`, `retention_distill_compare/`, `retention_distill_coverage_fix/`, `smoke/`, `stationary_finite_compression/`, `stationary_finite_no_sub05_compression/`, `stationary_finite_no_sub05_distill/`, and legacy `_view` files should be treated as historical unless rerun under the clipped target/action space and the current training recipe.

## Lessons

- Coverage is a first-class metric. Always report `same_target_time_saved_auc`, `span_coverage_percent`, and enough endpoint rows to show whether the frontier covers both high-memory and low-cost regimes.
- Desired retention is not a naturally well-conditioned coordinate for an integer-interval teacher. Small retention errors can become large interval errors, especially for high stability, high cost weights, and horizon-terminal actions.
- For continuous desired-retention distillation, train on the implied interval, not just the retention value. The repaired default keeps retention as the action output but supervises `log(interval(retention))`.
- Underpredicting intervals is worse than overpredicting intervals at high cost weights. The effective training recipe weights high-cost underprediction and terminal-action underprediction more heavily.
- Compression claims must be rerun after changing the action grid and after changing the training recipe. The old low-action-space compression sweep overstated the usefulness of small stationary finite students for the clipped action space, while the aligned sparse-teacher rerun shows that 316-parameter `residual:8:1` and 216-parameter `residual:6:1` models recover coverage when trained for 128 epochs.
- Student rollout is not automatically better. In the older no-sub-0.5 compression check it improved span coverage versus teacher-forcing, but the 476-parameter student-rollout model lost the default-baseline time-saved advantage. The current best small stationary finite per-user result is direct exact-table distillation with sparse cost weights, equal samples per cost weight, and 128 epochs.
- Removing remaining time at the oracle layer is a powerful structural compression. With 64x32 grids and 11 clipped actions, the unrestricted finite oracle table has 41,113,600 entries, while the stationary finite table has 22,528 entries before any neural distillation.
- The exact stationary finite policy is not just a lower-retention version of the finite oracle. Its action table is cost-sensitive and state-sensitive: with sub-0.5 actions removed, `w=256` remains mixed and bimodal, while `w=1024` concentrates on the new cheapest action `0.50` and loses high-cost scalar objective. That explains why very small stationary finite distills need the aligned exact-table recipe and enough epochs despite the smaller teacher table.
- `oracle_rho4` is a strong compact observation. With the clipped 11-action output head, it gives a 1,468-parameter model enough horizon and stability information to cover the frontier when the loss geometry is right.
- Train cost weights should be sparse but cover scale and the low-weight bend. For stationary finite distillation, the default is now `0,4,16,64,256,1024`; `add_4_only` is the chosen row because it passes all gates and has the best mean relative AUC among the passing candidates. Evaluation should remain denser with `0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024`.
- Do not train stationary finite distills from raw teacher-forcing event counts without rebalancing cost weights. Low-cost policies produce many more review events, so event-level cross-entropy can drown out high-cost policy boundaries. Uniform exact-table supervision over `(cost_weight, stability, difficulty)` fixed the earlier first-8 user-2 high-cost interpolation failure; the later low-weight frontier bend required adding direct low-weight labels at `1` and `4`.
- Integer interval actions are the most direct continuous-action target for the finite-horizon oracle. Continuous desired retention can work, but it needs interval-aware loss shaping and coverage validation.
