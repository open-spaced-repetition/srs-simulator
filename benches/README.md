# Benches

Lightweight performance baselines for the simulator. These benches are intended
to be run locally before and after performance-sensitive changes.

Prereqs
- The `srs-benchmark` repo should be available (either next to this repo or
  passed via `--srs-benchmark-root`).
- Use `uv` to run the scripts.

Run the default suite:

```bash
uv run python benches/run_bench.py --srs-benchmark-root ../srs-benchmark
```

Run a single scenario:

```bash
uv run python benches/run_bench.py --scenario event_lstm_lstm --srs-benchmark-root ../srs-benchmark
```

Run batched retention sweep baselines:

```bash
uv run python benches/run_batched_bench.py --srs-benchmark-root ../srs-benchmark
```

Notes
- The runner disables plots and progress bars to reduce noise.
- Report results by engine (event vs vectorized) when evaluating perf changes.
