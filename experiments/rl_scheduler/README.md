# RL Scheduler Training Experiments

This directory contains the training entry points, TOML profiles, and run
inspection tools for scheduler-learning experiments. The `rl_scheduler` family
name is broad by design: it covers PPO/DQN-style reinforcement learning, FQI,
CEM, CMA-ES, portfolio search, and other policy-search methods as long as the
output is a scheduler artifact that can enter the same external evaluation
pipeline.

The current implemented research lines are `fsrs6_adr` and `fsrs6_ap`
(Adaptive Parameters): black-box optimizers learn scheduler-side FSRS-6
retention policies, or directly search the 21 FSRS-6 scheduler weights, then
use stability `S` and difficulty `D` to compute the next interval.
A core rule is
that training and evaluation must not read the environment's hidden memory
state. A learned scheduler must maintain its own scheduler state. For these
FSRS-6 policy-search schedulers, that means the scheduler implements its own
FSRS-6 state update to obtain `S` and `D`.

## Directory Layout

- `run_experiment.py`: TOML-driven stage runner.
- `train_cmaes_fsrs6_adr.py`: CMA-ES FSRS-6 trainer for ordinary `fsrs6_adr`
  policies over `S,D`.
- `train_cmaes_fsrs6_ap.py`: CMA-ES FSRS-6 trainer that searches adaptive
  scheduler parameters as bounded deltas from each user's fitted FSRS-6 weights.
- `train_fsrs6_ap_portfolio.py`: SMS-EMOA trainer that exports a portfolio of
  ordinary `fsrs6_ap` child policies, optimizing hypervolume against the FSRS-6
  DR-grid baseline.
- `policy_search_common.py`: shared policy-search settings, metric, artifact,
  and evaluation helpers used by the active trainers.
- `train-overfit` can run these trainers through `[training.batch]` so users are
  batched in one process rather than launched as parallel training subprocesses.
- `plot_fsrs6_adr_policy_surfaces.py`: Plotly HTML visualizer for learned
  `f(S, D) -> desired_retention` surfaces across DR values.
- `inspect_run.py`: reads machine-readable evidence under a run root.
- `validate_artifact.py`: validates scheduler artifact metadata and referenced
  files.
- `configs/`: reproducible experiment profiles. Formal runs should start from a
  checked-in TOML profile here.

The restart roadmap and SOP live under `docs/rl_scheduler/reboot/`. This README
focuses on how to train, evaluate, and decide whether a scheduler-learning
experiment should continue.

## Core Terms

- **TOML profile**: the reproducible experiment configuration. User splits,
  seed, days, deck size, limits, engine, environment, short-term mode, fuzz,
  GPU guard, training settings, sweep, Pareto build, and Pareto analysis
  settings belong in TOML.
- **run id**: the stable identifier for one run. Stage outputs are written to
  `<output_root>/<run_id>/<stage>/`. Use a fixed run id when continuing or
  reproducing a run.
- **preflight**: the machine and configuration check before formal execution.
  It parses the profile, writes config snapshots, records git/uv/python/torch
  and CUDA evidence, checks output and baseline roots, and runs CUDA smoke/guard
  checks when the profile requires GPU.
- **stage-baseline**: exact metadata-based staging of FSRS-6 baseline JSONL
  logs from `baseline.log_root` into the current run. Formal workflows should
  not silently rerun baselines as a fallback.
- **train-overfit**: train a separate policy on the training user and compare it
  against the training-user baseline. If a policy family cannot beat baseline
  even when overfitting is allowed, stop that family before generalization
  checks. Native scheduler evaluation profiles, such as FSRS-3-vs-FSRS-6
  sweeps, may omit this stage because they do not produce scheduler artifacts.
- **scheduler artifact**: the trained policy package. It contains at least
  `metadata.json` and a policy/checkpoint file. Metadata records scheduler name,
  training users, seed, lambda, baseline DR, config snapshot, and policy path.
- **sweep**: external simulation of artifacts and baselines. The current
  batched sweep can batch `(user, scheduler, scheduler parameter)` lanes in one
  simulator call, such as several FSRS-6 desired-retention values plus several
  FSRS6 ADR policies. In formal experiment runs, sweep logs are written
  under `<output_root>/<run_id>/sweep/sweep_outputs/`; the standalone
  `run_sweep_users_batched.py --config` entrypoint still honors the TOML
  `[sweep].log_dir` shared-log setting.
- **build-pareto/analyze-pareto**: the external efficiency frontier and Markdown
  comparison report built from sweep logs. Formal `build-pareto` scans the
  run root so staged baselines and run-local sweep outputs are compared without
  stale shared retention-sweep logs. Internal
  reward, loss, acceptance rate, and promotion flags are diagnostics only; they
  do not replace Pareto evidence. Pareto charts should be generated per user;
  do not generate a user-aggregated Pareto plot.

## Standard Stage Flow

Prefer explicit stages over jumping straight to `all`:

Native scheduler evaluation profiles that omit `train-overfit` should skip that
command and run the stages listed in their TOML profile.

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage dry-run \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage preflight \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage stage-baseline \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage train-overfit \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage sweep \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage build-pareto \
  --run-id <run-id>

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/<profile>.toml \
  --stage analyze-pareto \
  --run-id <run-id>
```

Inspect a run:

```bash
uv run python experiments/rl_scheduler/inspect_run.py \
  --run-root <output_root>/<run-id>
```

Validate one artifact:

```bash
uv run python experiments/rl_scheduler/validate_artifact.py \
  --metadata <metadata.json> \
  --require-files
```

Visualize learned FSRS6 ADR policy surfaces:

```bash
uv run python experiments/rl_scheduler/plot_fsrs6_adr_policy_surfaces.py \
  --train-run-root <output_root>/<run-id> \
  --users 1,2,3 \
  --lambda-values 0.5
```

The visualizer writes one interactive Plotly HTML per user/lambda under
`experiments/rl_scheduler/plots/fsrs6_adr_policy_surfaces/`. Each figure uses
stability `S` and difficulty `D` as the horizontal axes, policy output retention
as the vertical axis. Ordinary ADR artifacts use one translucent surface per
baseline DR. ADR portfolio child artifacts have no baseline DR, so the
visualizer orders and colors their surfaces by `memorized_average / deck`.
Lambda-less ADR portfolio runs are grouped as `lambda_none`; omit
`--lambda-values` when plotting only new portfolio child artifacts.

## Current Main Experiments

Representative profiles:

- `configs/fsrs6_adr_cmaes_users_1_8.toml`: the ordinary
  `fsrs6_adr` scheduler trained with CMA-ES using the 6-parameter
  `fsrs6_adr_log_poly_v1` policy per user and baseline desired retention.
- `configs/fsrs6_adr_linear_cmaes_users_1_8.toml`: the ordinary
  `fsrs6_adr` scheduler trained with CMA-ES using one simplified 3-parameter
  `fsrs6_adr_log_linear_v1` policy per user and baseline desired retention.
- `configs/fsrs6_adr_linear_portfolio_users_1_8.toml`: the ordinary
  `fsrs6_adr` scheduler trained as 23 simplified 3-parameter
  `fsrs6_adr_log_linear_v1` portfolio children per user with SMS-EMOA
  hypervolume optimization.
- `configs/fsrs6_ap_cmaes_users_1_8.toml`: the adaptive-parameter family that
  trains 21 bounded FSRS-6 scheduler weights as deltas from each user's fitted
  baseline, batches both users and DR values, and still emits one artifact per
  `(user, DR, lambda)` policy.
- `configs/fsrs6_ap_portfolio_users_1_8_v2.toml`: the adaptive-parameter portfolio
  family that jointly mutates runtime desired retention and the 21 AP weight
  deltas, then exports 23 no-DR `fsrs6_ap` child artifacts per user.

Abandoned directions:

- qNEHVI/MOBO for ADR portfolio search was prototyped and abandoned because the
  BoTorch acquisition step was too slow at safe settings and exceeded the
  available 24 GB RTX 4090 D VRAM at useful batch settings.
- Evidence from the first 8 users: `proposal_batch_size = 1` was memory-safe but
  spent about 168 seconds per proposal iteration, while simulator evaluation
  took about 16.5 seconds. `proposal_batch_size = 16`, `mc_samples = 128`,
  `num_restarts = 8`, and `raw_samples = 128` failed before the first useful
  iteration with roughly 49.8 GB PyTorch allocated memory / 52.8 GB reserved and
  an additional 11 GB allocation attempt. Reducing the optimizer batch limit to
  2 still reached about 24.06 GB before the process was killed.
- Decision: do not keep the BoTorch dependency, qNEHVI trainer, qNEHVI configs,
  experiment runner/schema integration, or memory sweep script. Prefer the
  existing SMS-EMOA portfolio path or lower-overhead optimizers for this
  experiment family.

Training target:

- Baseline scheduler: FSRS-6.
- Candidate schedulers: FSRS6 ADR and FSRS6 AP (Adaptive Parameters).
- ADR action: emit desired retention from scheduler-side FSRS-6 `S,D`.
- AP action: search the full 21 FSRS-6 scheduler weights as bounded
  standardized deltas from each user's fitted FSRS-6 weights, then evaluate the
  resulting ordinary FSRS-6 scheduler.
- Main-profile DR grid: `0.52..0.96` in steps of `0.02`.
- ADR overfit gate: each artifact is trained for one baseline DR, and both
  relative memorized-average gain and relative memorized-per-minute gain must be
  greater than `0.0` against the same-user same-DR baseline. Batched
  train-overfit runs accept the batch when at least 80% of artifact points pass
  that gate.
- AP overfit gate: each `(user, DR, lambda)` artifact is checked against the
  same-user same-DR FSRS-6 baseline with the same `0.0` floor on both
  memorized-average and memorized-per-minute gains.
- Constraint handling: ADR candidates below the `0.0` relative-gain floor
  are ranked below every candidate satisfying both gate constraints.

For ordinary `fsrs6_adr`, the default policy uses 6 log-polynomial features over
normalized `S,D`; set `training.policy_search.feature_version = "fsrs6_adr_log_linear_v1"`
to train the simplified 3-parameter linear variant.
CMA-ES profiles keep the same `[training.policy_search]` policy/evaluation settings and put
optimizer-specific settings such as population size, generations, `sigma0`,
initial mean, and coefficient bounds in `[training.optimizer]`. AP profiles add
`[training.ap].dr_batch_size` and `weight_delta_scale`, plus
`training.batch_baseline_desired_retention_values = true` when DR values should
be batched inside each training job.

Batching model:

- `training.batch_baseline_desired_retention_values = true` batches the DR grid
  inside one training command.
- `training.policy_search.dr_batch_size` controls how many DR values enter one GPU chunk.
- CMA-ES effective lanes are approximately `dr_batch_size * population_size`.
- AP uses the same idea, but batches `dr_batch_size * population_size` lanes
  per job and writes one policy artifact per user/DR/lambda.
- Portfolio SMS-EMOA selection for ADR and AP uses a lightweight
  bounded process pool for batches with at least 8 users, capped at 32 workers
  by default, to avoid the old unbounded spawned-worker memory growth. Worker
  payloads contain only primitive metric tuples and do not import trainer/Torch
  modules. Set `FSRS6_PORTFOLIO_SELECTION_PROCESS_POOL=0` to force local
  selection globally, or `=1` to force the pool for smaller batches; cap workers
  with `FSRS6_PORTFOLIO_SELECTION_WORKERS`. ADR-only runs can use
  `FSRS6_ADR_PORTFOLIO_SELECTION_PROCESS_POOL` and
  `FSRS6_ADR_PORTFOLIO_SELECTION_WORKERS`.
- `[sweep]` contains the batch sweep envs, scheduler list, retention grid, log
  root, and batch sizing. The formal runner uses that table directly, so no
  separate retention_sweep TOML is needed.

Sampling benchmark:

- `benchmark_sampling.py` measures ADR portfolio candidate sampling cost by
  reusing the same `_evaluate_adr_coefficients` / `simulate_multiuser` path as
  `train_fsrs6_adr_portfolio.py`. It times bundle setup separately from
  candidate evaluation, writes per-repeat JSONL plus CSV, and records PyTorch
  CUDA memory together with `nvidia-smi` dedicated memory/utilization samples.
- Formal benchmark methodology, matrix definitions, output fields, and result
  tables live in [`sampling_benchmark.md`](sampling_benchmark.md).
- Example:

```bash
uv run python experiments/rl_scheduler/benchmark_sampling.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml \
  --environment fsrs6 \
  --candidate-mode fixed-dr \
  --fixed-desired-retention 0.98 \
  --users 1,2,3,4,5,6,7,8 \
  --lane-shapes 8x4 \
  --repeats 1 \
  --warmup 0
```

- Results are written under
  `artifacts/rl_scheduler/sampling_benchmark/<run-id>/summary.json`,
  `samples.jsonl`, and `samples.csv`. On Linux `nvidia-smi` does not expose the
  Windows shared-GPU-memory counter; use dedicated-memory and utilization
  samples as local evidence, and still check shared memory externally when
  diagnosing spill behavior.
- Use `benchmark_sampling_matrix.py` for the full fixed-DR FSRS6/LSTM matrix and
  confirmation pass. `--lstm-max-batch` accepts a positive integer or `off` for
  LSTM cells.

## Reproducibility Requirements

Formal experiments must satisfy these rules:

- Add or copy a `configs/*.toml` profile before running a new experiment. Do not
  rely on shell history for parameters.
- Record `seed`, `users`, `simulation`, `gpu_guard`, `performance`, `training`,
  `sweep`, `build_pareto`, and `analyze_pareto` settings in TOML.
- Treat `[sweep].log_dir` and `[build_pareto].log_dir` as standalone
  retention-sweep defaults. The formal runner uses the rest of those tables but
  isolates outputs under the current run root.
- Preserve config snapshots, resolved configs, command records, manifests, gate
  summaries, and performance summaries for each run.
- Use a stable `--run-id` so later `sweep`, `build-pareto`, and
  `analyze-pareto` stages align with the same outputs.
- Do not edit artifact metadata by hand to make a run pass. Metadata mismatch is
  a run failure.

## GPU And Throughput

Prefer batch-level parallelism before same-GPU multiprocessing:

- In training, increase optimizer population size, `dr_batch_size`, or candidate
  lanes until GPU utilization and throughput approach the platform limit.
- In sweep, batch `(user, scheduler, scheduler parameter)` lanes together.
- Tune `--max-lanes-per-batch` by trial runs for each sweep profile and GPU.
  LSTM environments usually need smaller lane chunks because model state and
  recurrent ops use more memory; FSRS environments can usually use larger lane
  chunks before hitting OOM.
- Use same-GPU process fanout only after single-process batching has plateaued
  and memory is still clearly underused.
- Performance-related changes must report before/after results for the affected
  path.

Logging rules:

- Retention sweeps write JSONL summaries by default.
- Do not write daily CSV sidecars or batch GPU CSV logs unless diagnosing
  simulation behavior or using a CSV-only plotting helper.
- If CSV diagnostics are needed, set `performance.diagnostic_csv_logs = true`
  explicitly and record the reason.

## Promotion Order

Advance a new policy family in this order:

1. Training-user overfit: prove the direction can beat baseline even before
   generalization pressure.
2. Same-user external sweep: confirm the trained artifact still improves under
   the independent sweep path.
3. Build Pareto: build per-user baseline + candidate Pareto JSON and PNG from
   sweep logs. Do not use a user-aggregated Pareto plot as evidence.
4. Analyze Pareto: write the configured scheduler comparison report from those
   per-user Pareto JSON files.
5. Validation and reserved test: unlock only after the earlier stages pass.

Stop conditions:

- `train-overfit` cannot beat baseline on the training user.
- Sweep log metadata does not match TOML/artifact metadata.
- Pareto advantage disappears, or it depends on missing baselines or mismatched
  user sets.
- A formal GPU path silently falls back to CPU.
- CSV diagnostics are produced without an explicit reason and output root.

## Trainer Integration Contract

New PPO/FQI/CEM/CMA-ES/portfolio variants should emit the same scheduler artifact shape:

- `metadata.json`
- a policy file, such as `policy.json`, or a checkpoint
- a training progress log, such as `training_progress.jsonl`
- optional `metrics.json`

Connect trainers through `training.command_template` in TOML. The template
should support at least:

- `{config_path}`
- `{user_id}`
- `{output_dir}`
- `{command_record_path}`

CMA-ES and other scalar-score trainers should also support `{lambda_value}`.
ADR/AP portfolio trainers do not use `training.lambda_grid` and should not
include `{lambda_value}` or `--lambda` in their command templates.

If the policy depends on scheduler state, implement that state update in the
scheduler. Do not read hidden memory state from the environment.

## Checklist

- Before running: `dry-run` and `preflight` pass.
- Baseline: `stage-baseline` matches exact engine, environment, scheduler, user,
  and DR metadata.
- Training: every required user, lambda, and baseline DR combination has an
  artifact or an explicit failure. Portfolio trainers are per-user and
  lambda-less.
- Sweep: `batch_lanes` matches the expected `(user, scheduler, parameter)` count.
- Build Pareto: `build_pareto_summary.json` contains both `result_paths` and `plot_paths`;
  generated Pareto plots are per-user, not user-aggregated.
- Analyze Pareto: `analyze_pareto_summary.json` points to the Markdown report.
- Disk: CSV count should be zero unless the run is explicitly diagnostic.
