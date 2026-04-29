# RL Scheduler Experiment SOP

## 1. Branch And Environment

1. Start from clean `master`/`main`.
2. Create a feature branch for one infrastructure slice or one experiment
   family.
3. Confirm status:

   ```bash
   git status --short --branch
   uv run pyright
   uv run ruff format
   ```

4. Record Python, Torch, CUDA, device, `uv.lock` hash, and dirty status in the
   first run record.

Rules:

- Do not push remotes from automation.
- Do not mix unrelated infrastructure and experiment-result changes.
- Use Conventional Commits if committing.

## 2. Infrastructure Change SOP

Before implementation:

1. Write a short coding plan with scope, non-goals, files expected to change,
   schemas affected, tests, and performance impact.
2. Identify whether README examples or dependency docs need updates.
3. For performance-related changes, run a baseline first and report affected
   engine/path.

Implementation rules:

- Prefer typed schema models over free-form dicts for config, artifact metadata,
  manifests, and gate summaries.
- Store reproducible experiment parameters in checked-in TOML. The TOML must
  include user splits, seeds, lambda grid, baseline source, engine, short-term
  mode, fuzz, limits, GPU guard, and gate thresholds.
- Keep scheduler math separate from artifact parsing and experiment orchestration.
- New fast paths must have equivalence or semantic correctness tests.
- New defaults must preserve existing behavior unless explicitly versioned.

Verification:

```bash
uv run ruff format
uv run pyright
```

Run targeted tests for the changed slice. For new experiment infrastructure,
include both positive and negative schema tests.

## 3. Formal Experiment SOP

Formal experiment execution must be reproducible from checked-in config and
archived artifacts.

Required stages:

1. `dry-run`
2. `preflight`
3. `stage-baseline`
4. `train-overfit`
5. `sweep`
6. `pareto`
7. `select`
8. `aggregate`
9. `reserved-test` when promoted beyond validation

Every non-dry stage must write:

- config snapshot
- resolved config
- command record
- exit code
- stdout/stderr paths
- git commit and dirty status
- environment summary
- produced artifact paths

The `all` runner executes configured stages in order and stops on the first
non-zero stage result. It writes `all_summary.json` under
`<output_root>/<run_id>/all/`.

Inspect a run root with:

```bash
uv run python experiments/rl_scheduler/inspect_run.py --run-root <output_root>/<run_id>
```

Validate scheduler artifact metadata with:

```bash
uv run python experiments/rl_scheduler/validate_artifact.py --metadata <artifact_metadata.json> --require-files
```

## 4. Stage Definitions

`dry-run`:

- Parse config.
- Validate schema.
- Resolve paths, users, updates, seeds, lambda grid, and commands.
- Print or write preview output only.
- Do not allocate CUDA.
- Do not create formal experiment evidence.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage dry-run --run-id <id>
  ```

`preflight`:

- Validate config and output roots.
- Validate exact baseline source availability.
- Validate artifact/profile compatibility if reusing artifacts.
- Run a guarded CUDA probe or representative memory smoke.
- Revalidate existing preflight summaries before skipping.
- Concrete operations: parse TOML, materialize resolved config, verify baseline
  metadata keys, check output-root writeability, collect git/uv/Python/Torch/CUDA
  versions, run the GPU guard command, and write a preflight summary.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage preflight --run-id <id>
  ```

- Current evidence files: `config_snapshot.toml`, `resolved_config.json`,
  `command_record.json`, `run_record.json`, `gpu_summary.json`,
  `gate_summary.json`, `preflight_summary.json`, and `manifest.json`.

`stage-baseline`:

- Copy or hardlink exact FSRS6 baseline logs into the run-specific root.
- Validate metadata for user, DR, seed, engine, short-term, fuzz, limits,
  priority, scheduler priority, days, deck, and button usage.
- Baseline TOML must declare `baseline.desired_retention_values`; staging
  requires every configured user to have exact JSONL logs for those DR points.
- Do not rerun FSRS6 automatically when exact logs are missing.
- Current implementation stages JSONL logs only. CSV sidecars remain diagnostic
  artifacts and are not copied into formal baseline staging.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage stage-baseline --run-id <id>
  ```

`train-overfit`:

- Train one policy per training user or per small training-user group.
- Allow overfit for feasibility diagnosis.
- Save checkpoints and full artifact metadata.
- Configure `training.command_template` in TOML. The runner formats and executes
  this command once per `users.train` and `training.lambda_grid` pair.
- Supported placeholders include `{user_id}`, `{lambda_value}`, `{lambda_token}`,
  `{run_id}`, `{seed}`, `{family}`, `{engine}`, `{scheduler}`, `{repo_root}`,
  `{stage_root}`, `{output_dir}`, `{config_path}`, `{config_snapshot_path}`,
  `{command_record_path}`, `{stdout_path}`, and `{stderr_path}`.
- The command must write scheduler policy artifact metadata matching
  `training.artifact_metadata_glob` under `{output_dir}`. The default glob is
  `metadata.json`.
- The runner validates artifact files, family, seed, engine, training user, and
  lambda value before passing the stage.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage train-overfit --run-id <id>
  ```

- Current evidence files: `config_snapshot.toml`, `resolved_config.json`,
  `command_record.json`, per-training-command records plus stdout/stderr,
  `gate_summary.json`, `training_summary.json`, `run_record.json`, and
  `manifest.json`.

`sweep`:

- Run the candidate through the real external batched sweep path.
- Use the exact same simulation protocol as the baseline.
- Validate log completeness and artifact metadata before continuing.
- Do not write retention_sweep daily CSV sidecars or batched GPU CSV logs unless
  `--diagnostic-csv-logs` is explicitly enabled for simulation-environment
  diagnosis.
- Configure `sweep.command_template` in TOML. The runner reads the current run's
  `train-overfit/training_summary.json`, validates every scheduler artifact, and
  executes the command once per artifact.
- Supported placeholders include `{artifact_id}`, `{artifact_metadata_path}`,
  `{policy_path}`, `{scheduler_name}`, `{user_id}`, `{lambda_value}`,
  `{lambda_token}`, `{run_id}`, `{seed}`, `{family}`, `{engine}`, `{repo_root}`,
  `{stage_root}`, `{output_dir}`, `{config_path}`, `{config_snapshot_path}`,
  `{command_record_path}`, `{stdout_path}`, and `{stderr_path}`.
- The command must write JSONL logs matching `sweep.log_glob` under
  `{output_dir}`. The default glob is `*.jsonl`.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage sweep --run-id <id>
  ```

- Current evidence files: `config_snapshot.toml`, `resolved_config.json`,
  `command_record.json`, per-sweep-command records plus stdout/stderr,
  `gate_summary.json`, `sweep_summary.json`, `run_record.json`, and
  `manifest.json`.

`pareto`:

- Build combined FSRS6 + candidate Pareto JSON and PNG.
- Validate FSRS DR grid, candidate lambda grid, artifact path, scheduler fields,
  engine, short-term, fuzz, and user id.
- Configure `pareto.command_template` in TOML. The runner requires passed
  `stage-baseline` and `sweep` summaries before executing the command.
- Supported placeholders include `{run_id}`, `{seed}`, `{family}`, `{engine}`,
  `{repo_root}`, `{stage_root}`, `{output_dir}`, `{baseline_stage_root}`,
  `{baseline_logs_dir}`, `{sweep_stage_root}`, `{sweep_outputs_dir}`,
  `{config_path}`, `{config_snapshot_path}`, `{command_record_path}`,
  `{stdout_path}`, and `{stderr_path}`.
- The command must write Pareto JSON matching `pareto.result_glob` and plot
  files matching `pareto.plot_glob` under `{output_dir}`. Defaults are `*.json`
  and `*.png`.
- Current command:

  ```bash
  uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage pareto --run-id <id>
  ```

- Current evidence files: `config_snapshot.toml`, `resolved_config.json`,
  `command_record.json`, Pareto command record plus stdout/stderr,
  `gate_summary.json`, `pareto_summary.json`, `run_record.json`, and
  `manifest.json`.

`select`:

- Select checkpoints using external Pareto metrics only.
- Do not select by training loss, reward, or internal feasibility.
- Manifest must include selected artifact, sweep logs, Pareto outputs, GPU logs,
  and selector rationale.

`aggregate`:

- Aggregate strict dominance, high-memory wins, DR95 wins, user coverage,
  near-overlap, and time-worse feasible rate.
- In formal mode, gate failure exits non-zero.

`reserved-test`:

- Run only after frozen validation selection.
- Do not tune on reserved-test results.
- Report pass/fail without changing thresholds.

## 5. Failure Handling

Failure classes:

- `invalid-config`: schema or config resolution failed.
- `invalid-baseline`: exact FSRS6 baseline missing or mismatched.
- `invalid-artifact`: artifact schema, profile, or metadata mismatch.
- `gpu-guard-failed`: memory cap, shared growth, or GPU probe failed.
- `incomplete-output`: logs, Pareto JSON, PNG, manifest, or summary missing.
- `gate-failed`: output is complete but research gate failed.
- `runner-failed`: command exited non-zero unexpectedly.

Handling:

- Stop immediately for all required-stage failures.
- Archive complete non-promotable outputs when possible.
- Do not silently fall back to CPU for formal GPU experiments.
- Do not weaken gates after seeing results.

## 6. Report SOP

Report order:

1. Decision
2. Protocol and config
3. Artifacts and logs
4. Dominance-first metrics
5. High-memory / DR95 / lambda bucket / user coverage
6. Near-overlap and time-worse diagnostics
7. GPU and performance evidence
8. Failure interpretation or next step

Required report fields:

- artifact path
- config path and snapshot path
- commit and dirty status
- train/validation/test users
- baseline source
- lambda grid
- seed strategy
- simulator signature
- Pareto JSON/PNG paths
- gate summary path
- whether the run advanced the frontier
