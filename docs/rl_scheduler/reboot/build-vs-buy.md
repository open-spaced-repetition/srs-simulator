# Build vs Buy Audit

This document identifies custom infrastructure that should be replaced or
reduced during the reboot, and domain logic that should remain custom.

## Replace Or Reduce

### Artifact Metadata And Config Validation

Current pattern:

- Free-form `dict[str, Any]` metadata.
- Repeated `metadata.get(...)` checks.
- Handwritten cross-field `ValueError` branches.

Replacement:

- `pydantic` or `msgspec` for artifact schemas, experiment config, manifests,
  and gate summaries.
- For the first dependency-light slice, stdlib dataclasses are acceptable as a
  typed scaffold; replace them once schema export, richer errors, or versioned
  unions become real needs.
- Use `Enum` or `Literal` for modes and family identifiers.
- Use discriminated unions for versioned FQI/RLPPO metadata.

Keep custom:

- Cross-section consistency rules.
- Promotion gate semantics.
- Tensor shape and feature remapping checks.

### TOML Runner And Stage State

Current pattern:

- Manual TOML parsing.
- Manual template substitution.
- Manual stage ordering and skip logic.
- Manual resolved summary writing.

Replacement:

- `TOML + pydantic` for the first reboot.
- Consider `Hydra/OmegaConf` only when profile composition becomes complex.
- Consider `Prefect` or `Snakemake` only when retry, caching, or distributed
  execution exceeds a local runner.

Keep custom:

- Experiment stage semantics.
- Baseline exactness.
- Gate failure classes.

### GPU Monitoring

Current pattern:

- Wrapper scripts and platform-specific parsing.
- String/log based success inference.

Replacement:

- `nvidia-ml-py` for NVML metrics.
- Keep subprocess execution simple.
- Use `tenacity` only for explicitly safe retries.

Keep custom:

- Dedicated memory cap.
- Shared-memory growth tolerance.
- Formal-run no-silent-fallback policy.

### Progress And Fanout

Current pattern:

- Custom JSON progress events.
- Custom tqdm worker bars.
- Custom thread/process fanout.

Replacement:

- `rich.progress` for multi-task progress UI.
- Keep `concurrent.futures` for simple local execution.
- Consider `Ray` or `Dask` only when distributed scheduling is needed.

Keep custom:

- Machine-readable command records.
- Stage-level state transitions.

### Tabular Aggregation And Pareto Computation

Current pattern:

- Manual JSONL scans.
- Manual grouping and summaries.
- Custom O(n^2) frontier logic.

Replacement:

- `polars` for scan/filter/group/aggregate.
- `paretoset` or a small vectorized NumPy implementation for non-dominated
  filtering.

Keep custom:

- Strict dominance epsilon.
- Time-smaller / memory-larger convention.
- High-memory, DR95, near-overlap, and time-worse definitions.

### Experiment Tracking

Current pattern:

- Paths, JSON summaries, reports, and plots manually linked.

Replacement:

- Keep local JSON manifests first.
- Consider `MLflow` once schemas stabilize for run comparison, metrics, and
  artifact indexing.

Keep custom:

- Formal report requirements.
- Reserved-test governance.

## Do Not Replace

- Simulator event/vectorized/batched engine semantics.
- SRS memory/scheduler/behavior/cost models.
- Exact FSRS6 baseline reuse rules.
- Train-user overfit feasibility gate.
- Dominance-first promotion policy.
- Scheduler family identity checks.
- Reserved-test isolation.

## Adoption Order

1. `pydantic` for contracts.
2. `polars` for analysis.
3. `nvidia-ml-py` for GPU evidence.
4. `rich` for progress UI.
5. `MLflow` after manifest schemas stabilize.
6. `Hydra/OmegaConf` if TOML profile composition becomes painful.
7. Workflow engine only when local runner limits become real.
