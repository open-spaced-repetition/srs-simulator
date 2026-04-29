# RL Scheduler Reboot Roadmap

## Goals

Rebuild the RL scheduler experiment stack from a clean base with stronger
contracts and less bespoke infrastructure. The new system must make a formal
experiment reproducible from checked-in config and artifacts, fail fast on gate
failure, and keep research decisions separated from orchestration glue.

## Non-Goals

- Do not port all previous RL/FQI/CEM experiments at once.
- Do not introduce a heavyweight workflow platform before typed contracts are
  stable.
- Do not use validation or reserved-test users to tune a scheduler family.
- Do not treat internal training reward, loss, or feasibility counts as
  promotion evidence.

## Phase 0: Clean Base And Inventory

Deliverables:

- Start from clean `master`/`main`.
- Record baseline commit, `uv.lock` hash, Python/Torch/CUDA versions, and dirty
  worktree status in a bootstrap note.
- Decide the minimal dependency set for the infrastructure slice.
- Identify which previous files are references only, not code to copy forward.

Gate:

- `uv run ruff format`
- `uv run pyright`
- A short inventory document explaining what will be reimplemented, imported,
  or dropped.

Stop conditions:

- Dirty worktree with unrelated changes.
- Missing baseline data source.
- Ambiguous base branch or dependency lock state.

## Phase 1: Typed Contracts

Deliverables:

- `ExperimentConfig` schema for TOML profiles.
- `RunRecord`, `CommandRecord`, `ArtifactManifest`, `GateSummary`, and
  `GpuGuardSummary` schemas.
- Versioned scheduler artifact metadata schemas for any new RL/FQI family.
- Unified scheduler capability registry for event/vectorized/batched support.
- First slice may use stdlib typed dataclasses to avoid dependency churn; migrate
  to `pydantic` when validators or JSON schema emission outgrow the scaffold.

Recommended tools:

- `pydantic` for config, artifact, manifest, and gate schemas.
- Keep domain-specific validators for baseline exactness, Pareto provenance,
  and promotion gates.

Gate:

- Schema unit tests cover valid examples and malformed inputs.
- JSON schema or equivalent docs are emitted for formal artifacts.
- No formal runner accepts untyped `dict[str, Any]` for experiment identity.

Stop conditions:

- Artifact metadata can be generated from multiple independent code paths.
- Runner can skip a stage based only on file existence.
- CLI overrides are not recorded in the resolved config.

## Phase 2: Minimal Formal Runner

Deliverables:

- A single TOML-driven runner with stages:
  `dry-run`, `preflight`, `stage-baseline`, `train-overfit`, `sweep`,
  `pareto`, `select`, `aggregate`, `reserved-test`.
- Initial implementation supports `dry-run`, `preflight`, and
  `stage-baseline`; all later stages must return explicit non-zero
  unsupported-stage results until their contracts are implemented.
- Stage state is machine-readable and includes command, exit code, stdout/stderr
  paths, config snapshot, git commit, dirty status, and environment summary.
- `all` fails fast when any required stage, artifact audit, or aggregate gate
  fails.

Gate:

- Dry-run prints resolved commands and target paths without writing formal
  outputs.
- Preflight validates GPU guard evidence, baseline source, config schema, and
  output roots.
- Stage skip requires semantic validation, not non-empty files.
- Retention sweep stages must not write daily or batch CSV simulation logs by
  default. Enable CSV output only through an explicit diagnostic flag, and record
  the reason/path once the formal runner exists.

Stop conditions:

- Aggregate reports `passed=false` but exits zero in formal mode.
- Preflight is skipped because an old log exists but is not revalidated.
- Any stage relies on shell history for reproducibility.

## Phase 3: Train-User Overfit Feasibility

Deliverables:

- A low-cost workflow that trains a separate policy for one or a small set of
  training users.
- Same-user external sweep and Pareto build against exact FSRS6 baseline.
- Aggregate gate focused on strict dominance, high-memory wins, DR95 wins,
  near-overlap, and time-worse feasible rate.

Research rule:

- This stage is a necessary-condition test, not promotion. If a family cannot
  beat baseline while allowed to overfit training users, do not run broader
  validation for that family.

Gate:

- Passes on train users with predeclared thresholds.
- Produces config snapshot, run record, GPU summary, Pareto JSON/PNG, manifest,
  and gate summary.

Stop conditions:

- Wins are concentrated in low-memory or low-value regions.
- Feasible points mostly use more time than baseline.
- The strategy only matches FSRS6 frontier through near-overlap.

## Phase 4: Fresh Validation And Sensitivity

Deliverables:

- Independent validation users not used in train-user overfit.
- At least two or three seeds or a documented bootstrap sensitivity check.
- Fresh validation gate separate from any pooled diagnostics.

Gate:

- External Pareto dominance advances across users and buckets.
- Gate remains stable under seed or bucket sensitivity checks.
- No threshold is weakened after seeing validation results.

Stop conditions:

- Train-user pass does not reproduce on independent users.
- Seed sensitivity flips the conclusion.
- High-memory/DR95 wins disappear.

## Phase 5: Reserved Test And Archival

Deliverables:

- Reserved test run using frozen config and selected artifact only.
- Report archive with all machine-readable summaries and human interpretation.
- Decision: promote, reject, or revise family.

Gate:

- Reserved test was not used for tuning.
- Report lists exact artifact, config, commit, baseline source, user split,
  lambda grid, GPU evidence, Pareto outputs, and final decision.

Stop conditions:

- Any reserved-test command differs from frozen validation protocol without a
  written amendment.
- Final decision cannot be reproduced from archived artifacts.

## Phase 6: Scale And Library Adoption

Adopt heavier tooling only after the typed contracts are stable.

Candidates:

- `polars` for JSONL/Pareto aggregation.
- `nvidia-ml-py` for GPU metrics.
- `rich` for progress UI.
- `MLflow` or another tracker for run/artifact comparison.
- `Hydra/OmegaConf` only when profile composition outgrows plain TOML.
- `Prefect` or `Snakemake` only when local stage execution cannot handle retry,
  caching, or distributed execution needs.

Gate:

- Library adoption reduces custom code and preserves existing domain gates.
- New dependency is documented in `README.md` when user-facing.
