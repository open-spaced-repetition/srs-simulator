# RL Scheduler Training Experiments

This directory contains the training entry points, TOML profiles, and run
inspection tools for scheduler-learning experiments. The `rl_scheduler` family
name is broad by design: it covers PPO/DQN-style reinforcement learning, FQI,
CEM, CMA-ES, portfolio search, and other policy-search methods as long as the
output is a scheduler artifact that can enter the same external evaluation
pipeline.

The current implemented research lines are `fsrs6_adr`, `fsrs6_cost_adr`,
`fsrs6_ap` (Adaptive Parameters), and `anki_sm2_ap`: black-box optimizers learn
scheduler-side FSRS-6 retention policies, learn a cost-conditioned interval
policy, directly search the 21 FSRS-6 scheduler weights, or search the 7 Anki
SM2 runtime parameters.
A core rule is
that training and evaluation must not read the environment's hidden memory
state. A learned scheduler must maintain its own scheduler state. For these
FSRS-6 policy-search schedulers, that means the scheduler implements its own
FSRS-6 state update to obtain `S` and `D`.

## Directory Layout

- `run_experiment.py`: TOML-driven stage runner.
- `train_cmaes_fsrs6_adr.py`: CMA-ES FSRS-6 trainer for ordinary `fsrs6_adr`
  policies over `S,D`.
- `train_cmaes_fsrs6_cost_adr.py`: CMA-ES trainer for one 24-parameter
  `fsrs6_cost_adr` policy per user. The policy emits desired retention values
  by default, or can be configured as an interval-head ablation, from
  scheduler-side `S,D` and a requested scalar cost weight. It can
  optionally initialize CMA-ES from an existing per-user Cost-ADR policy or a
  built-in action-head-specific mean, and can run CMA-ES in a diagonal z-space
  scaled by the first-eight sample std.
- `train_cmaes_fsrs6_ap.py`: CMA-ES FSRS-6 trainer that searches adaptive
  scheduler parameters as bounded deltas from each user's fitted FSRS-6 weights.
- `train_fsrs6_ap_portfolio.py`: SMS-EMOA trainer that exports a portfolio of
  ordinary `fsrs6_ap` child policies, optimizing hypervolume against the FSRS-6
  DR-grid baseline.
- `train_anki_sm2_ap_portfolio.py`: SMS-EMOA trainer that exports a portfolio of
  no-DR `anki_sm2_ap` child policies over the 7 Anki SM2 parameters.
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
  review Markov transition mode, GPU guard, training settings, sweep, Pareto
  build, and Pareto analysis settings belong in TOML.
- **run id**: the stable identifier for one run. Stage outputs are written to
  `<output_root>/<run_id>/<stage>/`. Use a fixed run id when continuing or
  reproducing a run.
- **preflight**: the machine and configuration check before formal execution.
  It parses the profile, writes config snapshots, records git/uv/python/torch
  and CUDA evidence, checks output and baseline roots, and runs CUDA smoke/guard
  checks when the profile requires GPU.
- **stage-baseline**: exact metadata-based staging of FSRS-6 baseline JSONL
  logs from `baseline.log_root` into the current run. When
  `[baseline_dr_selection].manifest` is configured, staging uses the selected
  per-user DR values from that manifest instead of a global DR grid. Formal
  workflows should not silently rerun baselines as a fallback. The configured
  `simulation.review_markov_transition` value is part of the required log
  metadata match.
- **train-overfit**: train a separate policy on the training user and compare it
  against the training-user baseline. If a policy family cannot beat baseline
  even when overfitting is allowed, stop that family before generalization
  checks. Native scheduler evaluation profiles, such as FSRS-3-vs-FSRS-6
  sweeps, may omit this stage because they do not produce scheduler artifacts.
- **scheduler artifact**: the trained policy package. It contains at least
  `metadata.json` and a policy/checkpoint file. Metadata records scheduler name,
  training users, seed, lambda, baseline DR, review Markov transition mode,
  config snapshot, and policy path.
- **sweep**: external simulation of artifacts and baselines. The current
  batched sweep can batch `(user, scheduler, scheduler parameter)` lanes in one
  simulator call, such as several FSRS-6 desired-retention values plus several
  FSRS6 ADR policies. In formal experiment runs, sweep logs are written
  under `<output_root>/<run_id>/sweep/sweep_outputs/`; the standalone
  `run_sweep_users_batched.py --config` entrypoint still honors the TOML
  `[sweep].log_dir` shared-log setting.
- **build-pareto/analyze-pareto**: the external Pareto frontier plus
  `analysis.md` and `analysis_summary.json` built from sweep logs. Formal
  `build-pareto` scans the
  run root so staged baselines and run-local sweep outputs are compared without
  stale shared retention-sweep logs. `analyze-pareto` uses scheduler-only HV
  delta, HV delta / baseline HV, per-user HV delta five-number summaries, and
  coverage-aware same-budget memory lift AUC plus same-target time saved AUC over the
  shared covered interval of linearly interpolated scheduler Pareto frontiers as
  primary Pareto evidence.
  Unweighted policy-point averages of memorized cards, time, and efficiency are
  diagnostics only. Internal reward, loss, acceptance rate, and promotion flags
  are diagnostics only; they do not replace Pareto evidence. Pareto charts
  should be generated per user; do not generate a user-aggregated Pareto plot.
- **report**: configured `[report]` profiles run
  `generate_experiment_report.py` after formal stages. The generator reads
  current and comparison `analysis_summary.json` files, writes
  `<run-root>/report/report_summary.json`, renders `<run-root>/report/report.md`,
  and copies the same Markdown to the configured docs path. Formal reports must
  cite `report_summary.json` and must not hand-copy metrics from `analysis.md`.
- **GPU monitor**: CUDA `train-overfit` and `sweep` stages automatically write
  `<stage>/gpu_monitor/gpu_memory.jsonl` and `<stage>/gpu_monitor/summary.json`
  unless `performance.gpu_monitor_enabled = false`. Baseline DR selection and
  baseline sweep subprocesses launched by `run_portfolio_workflow.py` use the
  same monitor. Use these artifacts, not manual screenshots, to judge shared
  GPU memory spill.

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

Compare per-user Pareto fronts from two or more formal run roots:

```bash
uv run python experiments/rl_scheduler/plot_pareto_run_comparison.py \
  --series FSRS-trained=artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_pop16_v1 \
  --series LSTM-trained=artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1 \
  --env lstm \
  --scheduler fsrs6_adr \
  --out-dir experiments/rl_scheduler/plots/adr_lstm_pareto_comparison \
  --title-prefix "LSTM environment FSRS6 ADR Pareto comparison"
```

The Pareto comparison plotter reads each run's
`build-pareto/build_pareto_outputs/simulation_results_retention_sweep_user_*.json`
files and writes one PNG per user. It defaults to the same axis layout as
`build-pareto`: memorized cards on the x-axis and study minutes per day on the
y-axis. The first series' `fsrs6` baseline is included by default; pass
`--no-baseline` to plot only the compared scheduler series. Each `--series`
argument can point either at a formal run root or directly at a
`build_pareto_outputs` directory.

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
- `configs/fsrs6_adr_portfolio_users_1_128_pop16_v1.toml`: the matched-budget
  ordinary ADR portfolio scaled to the first 128 users. It keeps the first-eight
  pop16/off16/gen20 budget, the 16 selected baseline DRs per user, and the
  FSRS6/LSTM external evaluation grid, and is the comparison root for the
  first-128 Cost-ADR run.
- `configs/fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml`: the
  cost-conditioned ADR scheduler trained as one 24-parameter
  `fsrs6_cost_adr` interval policy per user with CMA-ES pop16/gen20. CMA-ES
  starts from all-zero coefficients, evaluates exactly the 16 goal cost weights
  `0,1,2,4,8,16,32,48,64,96,128,192,256,384,512,1024`, and maximizes pure
  hypervolume delta over the FSRS-6 baseline frontier. Coverage is a diagnostic,
  not a training penalty or promotion metric.
- `configs/fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1.toml`: a
  diagnostic Cost-ADR rerun for users 1-2 that keeps the same pop16/gen20 and
  16 cost weights, but scores candidates with a coverage-aware hypervolume
  objective. The objective subtracts a baseline-HV-scaled penalty when the
  candidate-only frontier covers less than 90% of the FSRS-6 baseline time span
  or memory span.
- `configs/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1.toml`: the formal
  first-eight-user Cost-ADR coverage-aware rerun. It keeps the same users,
  pop16/gen20 CMA-ES budget, 16 cost weights, FSRS6/LSTM evaluation envs, and
  multi-user in-process training batch settings as the pure-HV Cost-ADR profile,
  but enables the 90% time-span and memory-span coverage penalty during
  training.
- `configs/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml`: a
  quality-aware Cost-ADR iteration that keeps the coverage-aware objective but
  adds a small soft penalty for baseline-dominated candidate points. It is a
  diagnostic training profile, not the promoted comparison row.
- `configs/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml`: a
  Cost-ADR selector evaluation. It reuses prior per-user artifacts and selects
  between coverage-aware and quality-v2 policies by each user's training
  `best_hypervolume_delta`, then runs the standard FSRS6/LSTM sweep and Pareto
  analysis.
- `configs/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml`: the
  fair Cost-ADR improvement profile. It initializes every user's CMA-ES from the
  built-in mean of the first-eight single-card distill 24-parameter policies,
  uses fixed `[-64, 64]` coefficient bounds and `sigma0=6.0`, and
  trains/evaluates exactly the matched 16 cost weights. Training uses the
  original Cost-ADR union-contribution HV objective; promotion still depends on
  external scheduler-only Pareto metrics.
- `configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml`: the
  scheduler-HV Cost-ADR repair profile. It keeps the fair matched 16 cost
  weights and fixed `[-64, 64]` coefficient bounds, initializes every user from
  the built-in first-eight mean, and runs CMA-ES in a diagonal z-space scaled by
  the first-eight sample std with floor `0.5`. Training and the overfit gate use
  scheduler-only HV versus the FSRS-6 baseline frontier; coverage remains a
  diagnostic. Interpret this profile against ADR only on the matched 16-point
  sweep, not on the dense-weight diagnostic sweep.
- `configs/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml`:
  the matched Cost-ADR default-action profile for deciding whether the scheduler
  should emit desired retention instead of interval. It keeps the same
  first-eight users, pop16/gen20 budget, 16 cost weights, scheduler-HV
  objective, and diagonal search scale as the interval-head repair profile, but
  uses `action_head = "desired_retention"` and a baseline cost-decay
  initializer instead of the interval distill mean.
- `configs/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml`:
  the Cost-ADR default-action profile initialized from the built-in first-eight
  interval-implied R coefficient mean. It uses wider `[0.30, 0.995]` R bounds,
  the matched 16 cost weights, scheduler-HV training objective, and no
  coefficient preconditioning. This is the default retention-head setup; the
  older std-preconditioned retention-head configs are retained as historical
  diagnostics.
- `configs/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml`:
  the same default retention-head Cost-ADR setup, but trained in the LSTM
  environment. It keeps pop16/gen20, `sigma0 = 1.0`, no coefficient
  preconditioning, and the matched 16 cost weights, while capping in-process
  LSTM training batches at 1024 lanes.
- `configs/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml`: the
  first-128-user scale-up of the scheduler-HV std-preconditioned Cost-ADR
  profile. It keeps pop16/gen20 CMA-ES, the matched 16 cost weights, and
  compares against `fsrs6_adr_portfolio_users_1_128_pop16_v1.toml`.
- `configs/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml`: a
  dense-grid diagnostic profile. It initializes each user's 24-parameter
  interval policy from the single-card continuous-distill Cost-ADR artifacts,
  expands CMA-ES bounds around the imported coefficients, trains against a
  19-point cost-weight grid, and evaluates against a 38-point sweep grid. Do not
  use this run as the official matched-16 comparison against ADR.
- `configs/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml`: the ADR
  portfolio scheduler trained with the same pop16/off16/gen20 budget while the
  ADR scheduler state uses default FSRS-6 weights instead of per-user fitted
  FSRS-6 weights.
- `configs/fsrs6_ap_cmaes_users_1_8.toml`: the adaptive-parameter family that
  trains 21 bounded FSRS-6 scheduler weights as deltas from each user's fitted
  baseline, batches both users and DR values, and still emits one artifact per
  `(user, DR, lambda)` policy.
- `configs/fsrs6_ap_portfolio_users_1_8_v2.toml`: the adaptive-parameter portfolio
  family that jointly mutates runtime desired retention and the 21 AP weight
  deltas, then exports 23 no-DR `fsrs6_ap` child artifacts per user.
- `configs/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml`: the Anki SM2
  adaptive-parameter portfolio with the matched pop16/off16/gen20 budget and
  7-parameter bounds based on Anki 24.11, with the interval upper bounds capped
  at 100 days for search.

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
- Cost-conditioned ADR action: emit an interval from scheduler-side FSRS-6
  `S,D` and a goal cost weight. These artifacts use action space
  `sd_cost_interval_function`, have no lambda, and have no baseline DR because
  one policy supplies the whole cost-weight curve.
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
Set `training.policy_search.scheduler_weight_source = "fsrs6_default"` to train
and export `fsrs6_default_adr`, which keeps the configured training environment
unchanged but evaluates ADR scheduler-side state with default FSRS-6 weights.
CMA-ES profiles keep the same `[training.policy_search]` policy/evaluation settings and put
optimizer-specific settings such as population size, generations, `sigma0`,
initial mean, and coefficient bounds in `[training.optimizer]`. FSRS AP profiles add
`[training.ap].dr_batch_size` and `weight_delta_scale`; Anki SM2 AP portfolio
profiles add `parameter_delta_scale`. Both use
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

Portfolio baseline selection:

- Portfolio ADR/AP profiles use `[baseline_dr_selection]` manifests with 16
  per-user FSRS-6 DR values selected under the `fsrs6` environment by
  `select_fsrs6_baseline_drs.py`. The selector writes a JSONL progress log next
  to the manifest by default, with one `generation_evaluated` record per user
  and CMA-ES generation containing the current hypervolume statistics. It
  evaluates multiple users in the same GPU batch up to `--max-lanes-per-batch`
  lanes, defaulting to 8192. Portfolio profiles use the low-budget selector
  setting (`population_size = 16`, `generations = 5`) so the per-user baseline
  DR search budget stays close to the portfolio training lane budget.
- Use `run_portfolio_workflow.py` as the single entry point for manifest
  generation, manifest-driven FSRS6 baseline sweep across `fsrs6,lstm`, and the
  formal experiment stages. When `[report].enabled = true`, the workflow also
  runs the report generator after successful formal stages. Existing manifests
  are reused by default; pass `--force-manifest` to regenerate them, or
  `--skip-report` to suppress the report step. For example:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml \
  --run-id fsrs6_adr_portfolio_users_1_8_v3
```

Matched-budget cost-conditioned ADR run:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1_markov_off
```

Mean-initialized fair Cost-ADR run:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off
```

Matched-budget ordinary ADR run for users 1-128:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_pop16_v1.toml \
  --run-id fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off
```

Scheduler-HV std-preconditioned Cost-ADR run:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off
```

Cost-ADR default-action profile for users 1-8:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off
```

Cost-ADR default-action profile with first-eight interval-implied R mean:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1_markov_off
```

LSTM-trained Cost-ADR default-action profile with no coefficient
preconditioning:

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
```

Scheduler-HV std-preconditioned Cost-ADR run for users 1-128:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off
```

Distill-initialized dense-weight Cost-ADR run:

```bash
uv run python experiments/rl_scheduler/run_portfolio_workflow.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml \
  --run-id fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1_markov_off
```

Coverage-aware Cost-ADR diagnostic for users 1-2:

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1_markov_off
```

Coverage-aware Cost-ADR formal rerun for users 1-8:

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off
```

Quality-aware Cost-ADR v2 diagnostic for users 1-8:

```bash
uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1_markov_off
```

Hybrid Cost-ADR selector evaluation:

```bash
uv run python experiments/rl_scheduler/build_fsrs6_cost_adr_hybrid.py \
  --source coverage=artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off \
  --source quality_v2=artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1_markov_off \
  --baseline-run-root artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1_markov_off \
  --output-run-root artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off \
  --users 1-8

uv run python experiments/rl_scheduler/run_experiment.py \
  --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml \
  --stage all \
  --run-id fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1_markov_off
```

Interpret the hybrid row as a selector over already trained Cost-ADR artifacts,
not as a single CMA-ES run. The selector uses training HV only; external
FSRS6/LSTM sweep metrics are held out for evaluation.

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
- Record `simulation.review_markov_transition` explicitly for formal reruns.
  The default is `false`, which keeps `button_usage` marginal probabilities and
  costs but ignores `long_term_transition`.
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
  DR, and review Markov metadata; portfolio profiles match the
  manifest-selected DRs per user.
- Training: every required user, lambda, and baseline DR combination has an
  artifact or an explicit failure. Portfolio trainers are per-user and
  lambda-less.
- Sweep: `batch_lanes` matches the expected `(user, scheduler, parameter)` count.
- Build Pareto: `build_pareto_summary.json` contains both `result_paths` and `plot_paths`;
  generated Pareto plots are per-user, not user-aggregated.
- Analyze Pareto: `analysis.md` and `analysis_summary.json` are present under
  `analyze_pareto_outputs/`, and `analyze_pareto_summary.json` records both.
- Report: formal experiment reports are generated by
  `generate_experiment_report.py`; the Markdown starts by citing
  `<run-root>/report/report_summary.json`.
- GPU monitor: CUDA stages have `<stage>/gpu_monitor/summary.json`, and
  `performance_summary.json` records the monitor summary path and shared-memory
  spill fields.
- Disk: CSV count should be zero unless the run is explicitly diagnostic.
