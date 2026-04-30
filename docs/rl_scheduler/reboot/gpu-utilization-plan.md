# GPU Utilization Plan

This plan turns GPU usage from an incidental runtime detail into an explicit
experiment-infrastructure objective. The goal is to reduce experiment
turnaround time by batching enough independent work into each GPU call while
preserving simulator semantics, reproducibility, and formal gates.

The current priority is the RL scheduler line, especially `sa_fsrs6` training
and validation. The same principles also apply to retention sweeps and future
scheduler families.

## Terms

- **Preflight**: a stage that validates config, code provenance, CUDA
  availability, dependency versions, baseline-log roots, output roots, and GPU
  guard settings before a formal run writes expensive artifacts.
- **GPU guard**: a machine-checkable check that a formal GPU run used the
  expected CUDA device and did not silently fall back to CPU.
- **Effective lane**: one independent simulated trajectory inside the batched
  engine. A lane can represent a user, a user plus retention value, a user plus
  candidate policy, or a repeated seed.
- **Super-batch**: a flattened batch over multiple axes, such as
  `(user, desired_retention)` or `(user, lambda, chain)`, executed by one
  simulator call.
- **Common random numbers**: using aligned random streams across candidate
  lanes so that policy differences are less confounded by simulation noise.
  When this is not implemented, the run must record that candidates used
  independent lane draws.
- **Diagnostic CSV logs**: per-day CSV sidecars and batch GPU CSV logs used for
  simulator debugging. They are off by default and must stay off for normal
  retention sweeps to avoid excessive disk usage.

## Current State

- `simulate_multiuser` parallelizes along the user dimension.
- Batched retention sweeps still loop over environment, scheduler, and desired
  retention outside the engine.
- `run_sweep_users_batched.py` supports one CUDA device through `--torch-device`
  and round-robin multi-device process assignment through `--cuda-devices`.
- `sa_fsrs6` training can evaluate multiple chains by duplicating a train user
  into many lanes, but trainer progress is written only at the end.
- `SAFSRS6BatchSchedulerOps` already supports per-lane coefficients, so the
  scheduler side can evaluate multiple policy candidates without reading
  environment memory state. The scheduler must keep its own FSRS-6 `S` and `D`
  state.
- Per-user daily CSV sidecars and batch GPU CSV logs are diagnostic-only and
  remain disabled unless `--diagnostic-csv-logs` is explicitly set.

Observed failure mode from the first GPU trial:

- A full-size `sa_fsrs6` overfit run with 32 chains, 1825 days, deck 10000, and
  LSTM environment produced sustained CUDA activity but no intermediate
  training artifact for more than 12 minutes.
- GPU utilization fluctuated instead of saturating, while CPU stayed mostly
  idle. This means the path is on CUDA, but the run shape is too coarse for
  fast research iteration and lacks progress instrumentation.

Implication: do not start with full formal training profiles. First add
measurement and small overfit probes, then increase effective lanes until
throughput plateaus or memory becomes the limiting factor.

## Utilization Strategy

The order of operations is:

1. **Measure first**: every performance run writes JSON summaries with workload
   shape, elapsed time, throughput, memory, and fallback status.
2. **Batch inside one process**: prefer a larger effective lane count in a
   single simulator call before launching competing processes on one GPU.
3. **Reuse loaded state**: load LSTM/FSRS weights once per unique user and avoid
   reloading them for each retention value, lambda, or candidate.
4. **Flatten homogeneous axes**: combine axes only when env ops, scheduler ops,
   and tensor shapes are compatible.
5. **Scale by throughput plateau, not utilization alone**: GPU utilization is
   noisy for day-by-day simulation loops. The main success metric is
   user-days/sec or candidate-days/sec at fixed semantics.
6. **Use multi-process fanout only to fill idle gaps**: after in-process lanes
   no longer improve throughput, run multiple independent jobs per GPU with a
   cap and explicit device assignment.
7. **Keep formal semantics stable**: batching changes may not alter promotion
   metrics unless equivalence or stochastic-invariance evidence is recorded.

## Parallel Axes

Use these axes in priority order.

1. `user`: already supported; safest for sweeps and validation.
2. `scheduler_param`: desired retention, fixed interval, or similar scalar
   parameters for homogeneous schedulers.
3. `candidate`: SA chains, policy artifacts, checkpoints, or proposal batches.
4. `lambda`: reward tradeoff values for training and selection.
5. `seed`: only for sensitivity estimates or variance reduction.
6. `scheduler`: only after families share a stable batched protocol.

The target shapes are:

```text
retention sweep today:
  for dr in desired_retentions:
      simulate_multiuser(users, scheduler, dr)

retention sweep target:
  simulate_multiuser_grid(lanes = users x desired_retentions)

SA training today:
  simulate_multiuser(lanes = chains for one user and one lambda)

SA training target:
  simulate_multiuser(lanes = users x lambdas x chains x repeats)
```

Output identity must remain explicit. Every result row or log must map back to:

```text
user_id, environment, scheduler, desired_retention, policy_id,
lambda_value, chain_id, seed, config_path, run_id
```

## Measurement Contract

Add a machine-readable performance summary for every formal GPU stage and every
benchmark probe.

Required fields:

- `git_commit`, dirty status, `uv.lock` hash, Python version, Torch version,
  CUDA version, GPU name, and device index.
- Stage name, config path, run id, command, and output root.
- Workload shape: users, days, deck, env, schedulers, desired-retention count,
  lambda count, chains, repeats, seeds, and effective lane count.
- Execution shape: process count, devices, batch size, super-batch axes,
  chunking strategy, dtype, and whether `torch.compile` is enabled.
- Runtime metrics: elapsed seconds, simulator calls/sec, user-days/sec,
  candidate-days/sec, and output log count.
- GPU metrics: peak allocated memory, peak reserved memory, NVML peak used
  memory, median utilization, p90 utilization, median power, p90 power, and CPU
  fallback status.
- Disk metrics: JSONL count and total bytes; CSV log count must be zero unless
  diagnostics are enabled.
- Failure class: `oom`, `cpu-fallback`, `timeout`, `no-progress`,
  `runner-failed`, `gate-failed`, or `metric-regression`.

Use JSON as the default summary format. CSV is acceptable only for optional
benchmark tables, not simulator daily logs.

Recommended implementation:

- Keep the existing preflight `gpu_summary.json`.
- Add `performance_summary.json` to train, sweep, and benchmark stages.
- Use `nvidia-ml-py` for NVML sampling once stdlib/Torch memory counters are
  insufficient.
- Sample NVML every 1-5 seconds in a parent-side monitor process.
- Include summary paths in each stage manifest.

## Batch-Size Tuning

Before running full experiments, find a good lane count for each profile.

Tuning process:

1. Start with a short reproducible probe: fewer days and a small user set.
2. Sweep effective lane counts upward: `1, 2, 4, 8, 16, 32, 64, ...`.
3. Stop when throughput plateaus, p90 memory exceeds the configured memory
   budget, or an OOM occurs.
4. Record the largest passing lane count and the fastest lane count.
5. Promote the fastest stable value into checked-in TOML.

Suggested defaults:

```text
memory_budget_fraction = 0.80
utilization_probe_seconds = 60
oom_backoff_factor = 0.50
throughput_plateau_tolerance = 0.05
minimum_formal_effective_lanes = 64 for SA probes when memory allows
```

For a single 24 GB GPU, avoid treating `chains = 32` as a saturation target.
It is a starting point. Increase lanes until user-days/sec or
candidate-days/sec stops improving.

## SA Training Plan

### Goals

- Quickly answer whether a scheduler family can beat FSRS-6 on one train user
  under overfit conditions.
- Keep enough GPU work in each evaluation to amortize Python orchestration and
  kernel launch overhead.
- Write progress artifacts often enough that long runs can be diagnosed and
  resumed.

### Required Trainer Changes

1. Add progress checkpoints:
   - after config load
   - after baseline evaluation
   - after initial candidate evaluation
   - after each annealing iteration
   - after writing policy and metrics
2. Write `training_progress.jsonl` with iteration, elapsed seconds, current best
   score, relative memorized gain, relative efficiency gain, accepted count,
   effective lanes, GPU memory, and seed.
3. Add a formal timeout with a clear `timeout` failure class.
4. Add `candidate_batch_size` or `chains_per_eval` so a large SA population can
   be evaluated in chunks without changing run identity.
5. Add `repeats_per_candidate` when stochastic noise is high.
6. Record whether candidate lanes used common random numbers or independent
   lane draws.
7. Avoid duplicating static LSTM/FSRS weight tensors per chain when possible.
   Keep unique user weights once and index them by `lane_user_idx`; duplicate
   only per-lane dynamic card state.

### Probe Profiles

Use three training profiles rather than one full profile.

```text
smoke:
  purpose: code path and CUDA guard
  days: 30-90
  deck: 1000-3000
  chains: 8-16
  iterations: 1-2

overfit-probe:
  purpose: determine whether the idea can beat baseline on one train user
  days: 365-730
  deck: 5000-10000
  chains: autotuned, usually 64+
  iterations: 4-16

formal-overfit:
  purpose: full train-user gate before validation
  days: target experiment horizon
  deck: target experiment deck
  chains: autotuned
  iterations: configured by research budget
```

The stop rule is strict: if the family cannot beat FSRS-6 under train-user
overfit after reasonable probe iterations and implementation checks, stop broad
validation for that family.

### Candidate Batching

For `sa_fsrs6`, one effective lane should represent:

```text
(train_user_id, lambda_value, chain_id, repeat_id)
```

The scheduler lane receives its own coefficient vector. The environment lane
receives the same user model but independent dynamic memory state. The scheduler
must maintain internal FSRS-6 `S` and `D`; it must not read the environment
memory state.

Acceptance tests:

- A one-candidate, one-user lane agrees with the existing trainer path within
  deterministic tolerance.
- Multiple candidate lanes produce the same number of metric records as the
  lane metadata table.
- Changing lane order does not change artifact identity.
- If common random numbers are implemented, two identical policies receive
  identical metrics for the same seed and lane metadata.

## Retention Sweep Plan

### Desired-Retention Super-Batching

Start with FSRS-6 because desired retention is a homogeneous scalar parameter.
Flatten `(user, desired_retention)` into the lane dimension:

```text
lane_user_ids = repeat(users, each = len(drs))
lane_desired_retention = tile(drs, len(users))
```

Then run one `simulate_multiuser` call for the super-batch and split outputs
back to per-user JSONL logs.

Implementation notes:

- Load each unique user's weights once.
- Expand or index weights to lanes only when scheduler/env ops require a
  per-lane tensor.
- Keep output filenames and metadata exact, including desired retention.
- Keep daily CSV sidecars disabled unless `--diagnostic-csv-logs` is enabled.
- Preserve `LogFilenameFilter` compatibility for downstream staging and Pareto
  scans.

### Scheduler Families

Initial candidates for parameter super-batching:

- `fsrs6`, `fsrs6_default`: desired retention.
- `fsrs3`, `fsrs3_default`: desired retention.
- `fixed`: fixed interval.
- `sa_fsrs6`: candidate policy artifacts or checkpoints if feature versions
  match.

Do not super-batch heterogeneous schedulers until a typed protocol can express
their state shapes and parameter metadata without special cases in the engine.

## Multi-GPU And Process Scheduling

Single GPU:

- Prefer one process with a large effective lane count.
- If the GPU still shows low throughput after lane autotuning, allow a small
  number of concurrent independent processes on the same device.
- Cap same-GPU fanout in TOML and record it. Start with 2, then measure.

Multiple GPUs:

- Keep the existing `--cuda-devices 0,1,...` round-robin assignment as the
  baseline.
- Record per-device work assignment, elapsed time, peak memory, utilization,
  and failures.
- Add duration estimates before work stealing. Work stealing is useful only
  after batch sizes and per-batch duration are predictable.

Avoid:

- Launching many tiny Python processes that each load weights and run a small
  GPU workload.
- Mixing formal training, sweep, and plotting processes on the same GPU without
  a run-level scheduler.
- Relying on MPS before in-process batching has been exhausted.

## Planned TOML Fields

Add these fields to future experiment profiles once the runner supports them:

```toml
[performance]
device = "cuda"
memory_budget_fraction = 0.80
nvml_sample_interval_seconds = 2.0
timeout_seconds = 3600
progress_interval_seconds = 30
write_performance_summary = true
diagnostic_csv_logs = false

[performance.autotune]
enabled = true
axis = "effective_lanes"
candidate_lanes = [8, 16, 32, 64, 128, 256]
plateau_tolerance = 0.05

[training.sa]
chains = 128
chains_per_eval = 128
repeats_per_candidate = 1
candidate_axis = "chain"
common_random_numbers = false

[sweep]
superbatch_axes = ["user", "desired_retention"]
max_effective_lanes = 256
```

Until these fields are implemented, encode the chosen values in the checked-in
profile name and command template, and record them in the run report.

## Profiling SOP

Run profiling before and after any performance-related change.

1. Select a small reproducible probe and one representative formal profile.
2. Run `preflight` and verify CUDA availability, code commit, dirty status,
   Torch/CUDA versions, and output root.
3. Sweep effective lane count upward.
4. Record JSON performance summaries for every point.
5. Compare throughput, memory, output count, and gate metrics.
6. Keep simulator CSV logs disabled unless diagnosing simulator behavior.
7. Report event, vectorized, and batched baselines only for paths affected by
   the change.

Useful commands:

```bash
uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/sa_fsrs6_gpu_trial.toml --stage preflight --run-id gpu-probe

uv run python experiments/retention_sweep/run_sweep_users_batched.py \
  --start-user 1 --end-user 100 \
  --env lstm --sched fsrs6 \
  --batch-size 25 \
  --start-retention 0.9 --end-retention 0.9 --step 0.01 \
  --torch-device cuda --no-progress

uv run python experiments/retention_sweep/run_sweep_users_batched.py \
  --start-user 1 --end-user 100 \
  --env lstm --sched fsrs6 \
  --batch-size 100 \
  --start-retention 0.9 --end-retention 0.9 --step 0.01 \
  --torch-device cuda --no-progress

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
```

Do not add `--diagnostic-csv-logs` to retention sweeps unless the run is
specifically diagnosing simulator behavior.

## Implementation Roadmap

### Phase 0: Instrumentation

- Add trainer progress JSONL.
- Add `performance_summary.json` and manifest references.
- Add timeout classification.
- Add NVML sampling behind an optional dependency or clear import error.
- Add tests for summary creation and CSV-disabled defaults.

Exit criteria:

- A long trainer run can be interrupted or timeout-classified with enough
  progress data to know the last completed phase.
- Formal GPU stages fail loudly on CPU fallback.

### Phase 1: Autotuned Probes

- Add a batch-size/effective-lane tuning command.
- Check in smoke and overfit-probe TOML profiles.
- Record fastest stable lane count per GPU/profile.
- Use JSON summaries only.

Exit criteria:

- A maintainer can reproduce the selected lane count from TOML and artifacts.
- Probe turnaround is short enough for repeated research iteration.

### Phase 2: SA Candidate Super-Batching

- Represent `(user, lambda, chain, repeat)` as lane metadata.
- Evaluate multiple SA proposals per simulator call.
- Chunk large populations with `chains_per_eval`.
- Keep scheduler-side FSRS-6 `S`/`D` independent of environment state.
- Add equivalence and lane-metadata tests.

Exit criteria:

- Train-user overfit runs write progress every iteration.
- Increasing chains improves candidate-days/sec until a measured plateau.

### Phase 3: Retention Parameter Super-Batching

- Implement `(user, desired_retention)` lanes for FSRS6-style schedulers.
- Split outputs back to exact per-user/per-retention JSONL files.
- Preserve downstream Pareto and baseline-staging compatibility.
- Add one-parameter equivalence tests against the existing batched path.

Exit criteria:

- A desired-retention grid uses one simulator call per homogeneous scheduler
  group instead of one call per retention value.
- Output count and metadata match the old path.

### Phase 4: Multi-GPU And Fanout

- Add per-device performance summaries.
- Balance work by estimated lane-days, not only user count.
- Add optional same-GPU process fanout with an explicit cap.
- Evaluate MPS only after in-process batching and capped fanout have been
  measured.

Exit criteria:

- Multi-GPU runs report per-device throughput and imbalance.
- Same-GPU fanout is used only when it improves throughput without memory
  pressure or reproducibility risk.

## Acceptance Criteria

- GPU benchmark and experiment runs are reproducible from TOML, manifests,
  command records, and JSON summaries.
- Formal GPU runs cannot silently fall back to CPU.
- Trainer progress is visible before final metrics are written.
- Retention sweeps still avoid CSV sidecars by default.
- Super-batched paths preserve output identity and pass equivalence or
  stochastic-invariance tests.
- Performance reports identify the bottleneck when utilization remains low:
  insufficient lanes, Python orchestration, data loading, kernel launch
  overhead, memory pressure, CPU fallback, or disk I/O.
- A train-user overfit failure blocks broad validation unless the report points
  to a specific implementation bug.

## Initial SA FSRS-6 Lane Probe

Probe context:

- Date: 2026-04-30
- GPU: NVIDIA GeForce RTX 4090 D, 24 GB
- Config: `experiments/rl_scheduler/configs/sa_fsrs6_overfit_probe.toml`
- Workload: LSTM environment, user 1, 365 days, deck 5000, short-term steps
- Command family: `experiments/rl_scheduler/tune_sa_fsrs6_lanes.py`

Results:

| Lanes | Elapsed sec | Candidate-days/sec | Peak reserved bytes |
| ---: | ---: | ---: | ---: |
| 1 | 21.47 | 17.0 | 27,262,976 |
| 8 | 42.38 | 68.9 | 75,497,472 |
| 16 | 49.84 | 117.2 | 274,726,912 |
| 32 | 59.02 | 197.9 | 1,161,822,208 |
| 64 | 60.62 | 385.4 | 4,261,412,864 |
| 128 | 64.06 | 729.3 | 8,176,795,648 |
| 256 | 65.12 | 1,434.9 | 8,589,934,592 |
| 512 | 58.10 | 3,216.8 | 2,518,679,552 |
| 1024 | 67.74 | 5,517.9 | 3,185,573,888 |
| 2048 | 87.52 | 8,541.6 | 5,809,111,040 |
| 4096 | 103.68 | 14,420.2 | 22,978,494,464 |

Decision:

- Use `chains = 2048` for the first overfit-probe profile. It has strong
  throughput while staying comfortably below the 80% memory-budget target by
  PyTorch reserved memory.
- Treat `chains = 4096` as a diagnostic upper bound for this workload. It was
  fastest, but PyTorch reserved memory exceeded the configured 80% budget on a
  24 GB GPU.
- Do not extrapolate these values to the 1825-day, deck-10000 formal profile;
  rerun lane tuning for that workload before increasing its chains.
