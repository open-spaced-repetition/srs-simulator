---
name: rl-experiment-infrastructure
description: Use when building, reviewing, or running RL scheduler experiment infrastructure in srs-simulator; covers typed configs, artifact schemas, train-user overfit gates, baseline reuse, GPU guard evidence, Pareto validation, and formal report requirements.
---

# RL Experiment Infrastructure

Use this skill for RL scheduler experiment infrastructure work in
`srs-simulator`.

## Core Workflow

1. Read `AGENTS.md` and project docs relevant to the experiment.
2. Start from a clean `main`/`master` base unless the user explicitly says
   otherwise.
3. For infrastructure changes, write or update a scoped coding plan before
   implementation when the change affects schemas, runners, artifacts, or gates.
4. Prefer typed contracts for config, metadata, manifests, command records, GPU
   summaries, and gate summaries.
5. Keep scheduler math separate from orchestration, artifact parsing, and report
   generation.
6. Run targeted tests plus:

   ```bash
   uv run ruff format
   uv run pyright
   ```

## Formal Experiment Rules

- Use checked-in structured config for formal runs.
- Snapshot config and write resolved config for every non-dry run.
- Record git commit, dirty status, `uv.lock` hash, Python/Torch/CUDA summary,
  command records, and artifact paths.
- Store reproducibility parameters in TOML: users, seeds, lambda grid, baseline,
  simulator protocol, GPU guard, paths, and gate thresholds.
- Reuse exact FSRS6 baseline logs; validate metadata before use.
- Do not rerun baseline automatically inside a formal runner.
- Keep retention_sweep CSV output off by default; use `--diagnostic-csv-logs`
  only when diagnosing simulation behavior or generating CSV-specific plots.
- External Pareto dominance is the promotion authority.
- Internal reward, loss, action coverage, and feasibility are diagnostic only.
- Train-user overfit feasibility must pass before broad validation.
- Reserved test is never used for tuning.

## Stage Semantics

- `dry-run`: validate config and print resolved commands; no formal evidence.
- `preflight`: validate config, paths, baseline source, artifact compatibility,
  and GPU guard evidence.
- `stage-baseline`: copy or hardlink exact baseline logs after metadata checks.
- `train-overfit`: train per-user or small-group policies for feasibility.
- `sweep`: run the real external batched sweep path.
- `pareto`: build and validate combined baseline + candidate Pareto JSON/PNG.
- `select`: choose checkpoints from external Pareto metrics only.
- `aggregate`: compute gates and fail non-zero in formal mode when gates fail.
- `reserved-test`: run only with frozen config and selected artifact.

Current runner entry point:

```bash
uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage dry-run --run-id <id>
uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage preflight --run-id <id>
uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage stage-baseline --run-id <id>
uv run python experiments/rl_scheduler/run_experiment.py --config <profile.toml> --stage all --run-id <id>
```

`dry-run`, `preflight`, `stage-baseline`, `train-overfit`, `sweep`, `pareto`,
and `select` are implemented. `train-overfit` requires
`training.command_template`, executes it once per training user and lambda value,
and validates scheduler policy metadata under `{output_dir}`. `sweep` requires
`sweep.command_template`, executes it once per trained artifact, and validates
JSONL logs under `{output_dir}`. `pareto` requires `pareto.command_template` and
validates Pareto JSON/PNG artifacts under `{output_dir}`. `select` requires
`select.command_template` and validates selection JSON pointing to a scheduler
artifact. Treat later stages as unavailable until they return machine-readable
evidence and enforce their gates. `all` is fail-fast and writes
`all_summary.json`.

## Required Metrics

Always report:

- strict dominant points
- high-memory strict wins
- DR95 strict wins
- user coverage
- lambda bucket coverage
- near-overlap rate
- feasible-time-worse rate
- GPU peak dedicated memory and shared-memory growth

## Failure Handling

Classify failures as:

- `invalid-config`
- `invalid-baseline`
- `invalid-artifact`
- `gpu-guard-failed`
- `incomplete-output`
- `gate-failed`
- `runner-failed`

Stop formal workflows immediately on required-stage failure. Archive complete
non-promotable outputs when possible.

## Build vs Buy Guidance

Prefer mature libraries for infrastructure:

- `pydantic`: config, metadata, manifest, gate schemas.
- `polars`: JSONL/table aggregation.
- `nvidia-ml-py`: GPU metrics.
- `rich`: progress UI.

Keep domain rules custom:

- SRS simulator semantics.
- FSRS6 baseline exactness.
- dominance-first gates.
- train-user overfit feasibility.
- high-memory/DR95/near-overlap/time-worse definitions.
