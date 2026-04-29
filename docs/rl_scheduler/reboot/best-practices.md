# RL Scheduler Reboot Best Practices

## Architecture

- Keep simulator core, scheduler runtime, artifact schema, and experiment
  orchestration separate.
- Treat the event engine as the semantic reference.
- Make vectorized and batched engines expose typed ops and explicit capability
  flags.
- Do not let `simulator/` import from `experiments/`.
- Keep artifact loaders responsible for schema normalization only; schedulers
  should consume normalized config.
- Split large experimental schedulers into policy, features, artifact schema,
  batch ops, and scoring modules.

## Configuration

- Formal experiments must be driven by checked-in TOML or another structured
  config.
- CLI overrides must be few, explicit, and recorded.
- Each formal run snapshots the config and writes a resolved config summary.
- Lambda grids, seeds, user splits, engine, short-term, fuzz, days, deck, limits,
  priority, scheduler priority, GPU guard, paths, selector gates, and aggregate
  gates must live in config.

## Schema And Provenance

- Use typed models for:
  - experiment config
  - artifact metadata
  - manifests
  - command records
  - GPU summaries
  - gate summaries
- Reject metadata/payload mismatches loudly.
- Skip stages only after semantic validation of prior outputs.
- Include data/code digests where stale-cache risk exists.

## Research Methodology

- External Pareto dominance is the promotion authority.
- Internal reward, loss, action coverage, feasibility count, or
  `promotion_eligible` can diagnose but cannot promote.
- Always inspect high-memory, DR95, low-lambda, user coverage, near-overlap, and
  time-worse behavior.
- Do train-user overfit feasibility before broad validation.
- If a family fails under train-user overfit, stop that family unless the
  failure points to a specific implementation bug.
- Use independent validation and reserved test splits.
- Do not tune thresholds on validation/test outcomes.

## Baselines

- Reuse exact FSRS6 baseline logs when available.
- Validate metadata, not only filenames.
- Use run-specific log roots to avoid contaminating historical logs.
- Do not rerun baseline automatically as a fallback in formal workflows.

## GPU And Performance

- Increase GPU utilization by batching within a job before adding competing
  processes on one GPU.
- Prefer batching along `(user, lambda)`, scheduler parameter, candidate, or
  checkpoint axes.
- Record workload shape, batch size, elapsed time, simulator calls/sec, peak
  dedicated memory, shared memory growth, and fallback status.
- Guarded CUDA smoke must precede larger formal runs.
- Existing preflight logs must be revalidated before reuse.
- Performance changes require before/after results by affected engine/path.
- Retention sweeps should skip daily CSV sidecars and batched GPU CSV logs by
  default. Enable CSV only for diagnostics or CSV-specific plots, then record
  the reason and output root.

## Reporting

- Put the decision first.
- Report dominance-first metrics before internal diagnostics.
- Separate `gate-failed` from `runner-failed`.
- A failed experiment still needs an archived report if outputs are complete.
- Reports must be reproducible from the config snapshot, manifests, and
  artifacts without shell history.

## Dependency Discipline

- Add libraries to remove infrastructure code, not to obscure domain logic.
- Start with small, high-leverage dependencies:
  - schema/config: `pydantic`
  - tabular aggregation: `polars`
  - GPU metrics: `nvidia-ml-py`
  - progress UI: `rich`
- Defer workflow engines until local typed runners are not enough.
- Update README dependency notes when user-visible dependencies change.

## Testing

- Cover true CLI/runner paths, not only helper functions.
- For schema, test valid fixtures, missing fields, wrong versions, wrong
  environment, wrong user split, duplicate lambda values, and metadata drift.
- For batching, test one-candidate equivalence or explicit semantic invariants.
- For gates, test both pass and fail outputs and exit codes.
- For artifact loaders, test old artifact compatibility separately from strict
  formal-run requirements.
