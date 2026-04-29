# GPU Utilization Plan

This plan turns GPU usage from an incidental runtime detail into an explicit
experiment-infrastructure objective. The goal is higher throughput without
changing simulator semantics or weakening formal experiment gates.

## Current State

- The batched engine parallelizes users inside one simulation call.
- Retention sweeps currently loop over environment, scheduler, and scheduler
  parameter values outside the batched engine.
- Multi-GPU execution assigns user batches to devices, but each batch still runs
  one scheduler configuration at a time.
- Per-user daily CSV sidecars and batch GPU CSV logs are diagnostic-only and
  should remain off by default.

Implication: current throughput mostly scales along the user axis. GPU
utilization can still be low when the user batch is small, when a single
configuration has insufficient work, or when Python orchestration dominates
many small scheduler-parameter runs.

## Principles

- Batch inside one process before adding competing processes on the same GPU.
- Preserve reference semantics. Every new batched path needs equivalence checks
  against the existing single-configuration path.
- Prefer homogeneous batching first: same environment, same scheduler family,
  multiple users, and multiple parameter values.
- Keep heterogeneous scheduler families separate unless a shared protocol
  removes real duplication without hiding scheduler-specific state.
- Treat MPS and multi-process fanout as secondary tools for filling idle GPU
  time after in-process batching has been exhausted.
- Record workload shape and GPU evidence for every formal run.

## Target Batching Axes

Priority order:

1. `user`: already supported by `simulate_multiuser`.
2. `scheduler_param`: desired retention, lambda, fixed interval, or checkpoint
   index when the scheduler ops are homogeneous.
3. `candidate`: multiple trained artifacts with the same scheduler op contract.
4. `seed`: only for sensitivity sweeps where stochastic independence is
   preserved and logged.
5. `scheduler`: only after families share a stable batched abstraction.

Initial engineering target:

```text
current: for param in params: simulate_multiuser(users, scheduler, param)
target:  simulate_multiuser_grid(users, scheduler, params)
```

The first implementation should flatten `(user, scheduler_param)` into a
super-batch when that is mathematically equivalent. Output logs must still be
written per `(user, scheduler_param)` point.

## Profiling SOP

Run profiling before and after any performance-related change.

1. Select a small reproducible profile and a representative formal profile.
2. Record command, config snapshot, git commit, dirty status, `uv.lock` hash,
   Python/Torch/CUDA versions, device name, and driver-visible memory.
3. Sweep batch size upward until throughput plateaus or memory becomes the
   limiting factor.
4. For each point, record:
   - workload shape: users, days, deck, scheduler count, parameter count
   - batch size and effective super-batch size
   - elapsed seconds
   - simulator calls/sec or user-days/sec
   - peak dedicated memory
   - shared-memory growth
   - GPU fallback status
   - output log count
5. Keep CSV simulation logs disabled unless diagnosing simulator behavior.
6. Report event, vectorized, and batched baselines only for paths affected by
   the change.

Recommended first commands:

```bash
uv run python experiments/retention_sweep/run_sweep_users_batched.py --start-user 1 --end-user 100 --env lstm --sched fsrs6 --batch-size 25 --no-progress
uv run python experiments/retention_sweep/run_sweep_users_batched.py --start-user 1 --end-user 100 --env lstm --sched fsrs6 --batch-size 100 --no-progress
```

Formal benchmarking profiles should use checked-in TOML once the runner exposes
GPU performance summaries.

## Implementation Tasks

### Task A: Measurement Contract

- Extend `GpuGuardSummary` or add a performance summary artifact with workload
  shape, effective batch shape, elapsed time, throughput, peak memory, and
  fallback status.
- Prefer `nvidia-ml-py` for NVML metrics once the stdlib scaffold becomes too
  weak.
- Add tests that the summary is written and referenced by manifests.

### Task B: Batch-Size Tuning Harness

- Add a command or documented profile that sweeps batch sizes for a fixed
  simulator protocol.
- Fail clearly on OOM and record the largest passing batch size.
- Keep logs compact; write JSON summaries, not per-day CSV, by default.

### Task C: `(user, scheduler_param)` Super-Batching

- Start with FSRS6 desired-retention grids because the scheduler ops are
  homogeneous and already parameterized by a scalar desired retention.
- Shape tensors so each `(user, desired_retention)` pair is one effective user
  lane, while preserving per-user weights and per-parameter metadata.
- Validate one-parameter equivalence against the existing batched path.
- Validate multi-parameter output count and metadata exactness.
- Add memory-pressure tests with small synthetic workloads.

### Task D: Candidate/Checkpoint Batching

- Batch multiple artifacts only when they share scheduler op shape and feature
  version.
- Keep artifact identity explicit in output metadata.
- Do not pool candidates in Pareto or aggregate metrics unless the selector
  consumes artifact identity.

### Task E: Multi-GPU Scheduling

- Keep round-robin user-batch assignment as the baseline.
- Add device-level summaries: batches assigned, elapsed time, peak memory, and
  failures.
- Consider work stealing only after single-device batches have stable size and
  duration estimates.

### Task F: Formal Gates

- Preflight must reject formal GPU runs that silently fall back to CPU.
- Formal performance reports must include before/after throughput for affected
  paths.
- A utilization optimization cannot change promotion metrics unless the report
  includes equivalence evidence.

## Acceptance Criteria

- A maintainer can reproduce the GPU benchmark from TOML, manifests, and command
  records.
- The current single-configuration batched path and the new super-batched path
  agree on totals within the existing deterministic tolerance.
- Retention sweeps still avoid CSV sidecars by default.
- GPU evidence is machine-readable and attached to each formal stage manifest.
- If GPU utilization remains low, the report identifies the bottleneck:
  insufficient work, Python orchestration, data loading, kernel launch overhead,
  memory pressure, or CPU fallback.
