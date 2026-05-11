# ADR Sampling Benchmark

This document is the formal home for ADR sampling benchmark methodology and
results. The README keeps only a short entry point.

## Methodology

The benchmark measures the FSRS6 ADR candidate evaluation path used by
`train_fsrs6_adr_portfolio.py`. Each cell shells out to
`benchmark_sampling.py`, which builds one batched simulation bundle, times
candidate evaluation separately from setup, and writes per-repeat JSONL, CSV,
and a `summary.json`.

All formal matrix cells use:

- Scheduler/search path: FSRS6 ADR candidate evaluation.
- Candidate mode: `fixed-dr`.
- Fixed desired retention: `0.98`.
- Users: `1..128`.
- Warmup repeats: `0`.
- Matrix repeats: `1`.
- Confirmation repeats: `3`.

Fixed DR keeps all lanes on the same scheduling policy so the benchmark isolates
environment cost, lane shape, total lane count, and LSTM batch cap effects. It
does not measure policy mutation or portfolio quality.

`SRS_LSTM_MAX_BATCH` controls how many LSTM lane/card entries enter one recurrent
forward chunk. Integer values cap the chunk size. `off` disables chunking. The
matrix runner sets this environment variable independently for each LSTM child
process and records `effective_lstm_max_batch` in every summary. FSRS6 cells
record `null`.

Linux `nvidia-smi` reports dedicated FB memory and utilization counters, but it
does not expose Windows-style shared GPU memory. For GPU OOM and spill analysis,
check host shared GPU memory externally; shared GPU memory above 1 GiB likely
indicates VRAM spill and severe simulator slowdown.

Run metadata for the measurements below:

- Date: 2026-05-11.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB.
- Driver: 591.86.
- PyTorch: 2.9.1+cu126.
- PyTorch CUDA: 12.6.
- Config:
  `experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml`.
- Fixed desired retention: `0.98`.

## Matrix

Common lane shapes, run for both `fsrs6` and `lstm`:

| total lanes | shapes |
| ---: | --- |
| 256 | `8x32`, `16x16`, `32x8`, `64x4`, `128x2` |
| 512 | `8x64`, `16x32`, `32x16`, `64x8`, `128x4` |
| 1024 | `8x128`, `16x64`, `32x32`, `64x16`, `128x8` |

Additional FSRS6 throughput shapes:

| total lanes | shapes |
| ---: | --- |
| 2048 | `8x256`, `16x128`, `32x64`, `64x32`, `128x16` |
| 4096 | `8x512`, `16x256`, `32x128`, `64x64`, `128x32` |
| 8192 | `8x1024`, `16x512`, `32x256`, `64x128`, `128x64` |

LSTM batch caps:

- `1024`
- `2048`
- `4096`
- `8192`
- `20000`
- `off`

Matrix size:

- LSTM: 90 cells.
- FSRS6 common: 15 cells.
- FSRS6 extended: 15 cells.
- Total matrix pass: 120 cells.

The confirmation pass selects, per environment and total-lane level, the fastest
configuration, the best reviews/s configuration, and the lowest-memory
configuration among cells within 90 percent of the best reviews/s rate.

## Commands

Full matrix:

```bash
uv run python experiments/rl_scheduler/benchmark_sampling_matrix.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml \
  --run-id sampling_fixed_dr_098_YYYYMMDD
```

Single-cell smoke command:

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

LSTM smoke command with an explicit batch cap:

```bash
uv run python experiments/rl_scheduler/benchmark_sampling.py \
  --config experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml \
  --environment lstm \
  --candidate-mode fixed-dr \
  --fixed-desired-retention 0.98 \
  --lstm-max-batch 1024 \
  --users 1,2,3,4,5,6,7,8 \
  --lane-shapes 8x4 \
  --repeats 1 \
  --warmup 0
```

## Output

The matrix runner writes:

- `artifacts/rl_scheduler/sampling_benchmark_matrix/<run-id>/matrix_summary.json`
- `artifacts/rl_scheduler/sampling_benchmark_matrix/<run-id>/matrix_summary.csv`
- Per-cell `summary.json`, `samples.jsonl`, and `samples.csv` under `cells/`
  and `confirmation/`.
- Per-cell stdout/stderr logs under `_subprocess_logs/`.

Each successful row includes:

- `mean_seconds`
- `mean_lanes_per_second`
- `mean_reviews_per_second`
- `mean_reviews_per_lane`
- `mean_metric_total_reviews`
- `mean_metric_total_lapses`
- `mean_metric_total_cost`
- `max_torch_peak_reserved_memory_bytes`
- `max_nvidia_smi_peak_memory_used_mib`
- `max_nvidia_smi_peak_utilization_gpu_percent`
- `max_nvidia_smi_peak_utilization_memory_percent`
- `effective_lstm_max_batch`

Rows with OOM, timeout, subprocess error, or missing/invalid summaries are kept
with `status=failed` and a `failure_kind`.

## Results

The full 120-cell matrix was started as
`sampling_fixed_dr_098_20260511_0846`, but the first LSTM matrix cell
(`8x32`, 256 lanes, `SRS_LSTM_MAX_BATCH=1024`) exceeded 17 minutes and was
still running. That full LSTM matrix was stopped as infeasible for this fixed
DR 0.98 workload. The completed results below cover:

- Full FSRS6 common + extended matrix:
  `sampling_fsrs6_matrix_fixed_dr_098_20260511`.
- LSTM batch-cap diagnostic on 32 lanes.
- LSTM cap 8192 versus 20000 at 256, 512, and 1024 lanes.
- LSTM 1024-lane higher-cap saturation sweep.
- LSTM 8-user lane scaling at cap 8192 for 256, 512, and 1024 lanes.
- Focused confirmation runs for selected FSRS6 and LSTM configurations.

### FSRS6 Best By Lane Count

| lanes | best shape by reviews/s | seconds | lanes/s | reviews/s | torch GiB | dedicated MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | `64x4` | 14.53 | 17.62 | 23.20M | 0.43 | 1194 |
| 512 | `32x16` | 17.25 | 29.68 | 42.18M | 0.87 | 1629 |
| 1024 | `32x32` | 23.76 | 43.10 | 61.27M | 1.78 | 2560 |
| 2048 | `8x256` | 44.47 | 46.05 | 67.04M | 3.50 | 4335 |
| 4096 | `8x512` | 82.42 | 49.70 | 72.33M | 7.37 | 8298 |
| 8192 | `8x1024` | 161.48 | 50.73 | 73.85M | 15.34 | 16505 |

### FSRS6 Common Matrix

| lanes | shape | seconds | lanes/s | reviews/s | torch GiB | dedicated MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | `8x32` | 17.96 | 14.25 | 20.74M | 0.45 | 1199 |
| 256 | `16x16` | 18.43 | 13.89 | 18.41M | 0.43 | 1184 |
| 256 | `32x8` | 20.44 | 12.53 | 17.80M | 0.45 | 1200 |
| 256 | `64x4` | 14.53 | 17.62 | 23.20M | 0.43 | 1194 |
| 256 | `128x2` | 16.50 | 15.51 | 19.27M | 0.43 | 1178 |
| 512 | `8x64` | 18.22 | 28.10 | 40.89M | 0.85 | 1607 |
| 512 | `16x32` | 17.41 | 29.41 | 38.98M | 0.83 | 1587 |
| 512 | `32x16` | 17.25 | 29.68 | 42.18M | 0.87 | 1629 |
| 512 | `64x8` | 16.87 | 30.35 | 39.98M | 0.85 | 1607 |
| 512 | `128x4` | 16.47 | 31.08 | 38.59M | 0.83 | 1587 |
| 1024 | `8x128` | 24.80 | 41.30 | 60.14M | 1.65 | 2427 |
| 1024 | `16x64` | 26.17 | 39.13 | 51.88M | 1.63 | 2409 |
| 1024 | `32x32` | 23.76 | 43.10 | 61.27M | 1.78 | 2560 |
| 1024 | `64x16` | 24.09 | 42.51 | 55.98M | 1.66 | 2439 |
| 1024 | `128x8` | 22.97 | 44.58 | 55.38M | 1.65 | 2428 |

### FSRS6 Extended Matrix

| lanes | shape | seconds | lanes/s | reviews/s | torch GiB | dedicated MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | `8x256` | 44.47 | 46.05 | 67.04M | 3.50 | 4335 |
| 2048 | `16x128` | 42.08 | 48.67 | 64.52M | 3.30 | 4125 |
| 2048 | `32x64` | 45.04 | 45.47 | 64.63M | 3.48 | 4317 |
| 2048 | `64x32` | 43.39 | 47.20 | 62.16M | 3.46 | 4297 |
| 2048 | `128x16` | 42.09 | 48.66 | 60.45M | 3.44 | 4274 |
| 4096 | `8x512` | 82.42 | 49.70 | 72.33M | 7.37 | 8298 |
| 4096 | `16x256` | 79.43 | 51.56 | 68.36M | 6.88 | 7795 |
| 4096 | `32x128` | 83.37 | 49.13 | 69.84M | 7.10 | 8080 |
| 4096 | `64x64` | 79.63 | 51.44 | 67.74M | 6.76 | 7708 |
| 4096 | `128x32` | 78.77 | 52.00 | 64.59M | 6.78 | 7736 |
| 8192 | `8x1024` | 161.48 | 50.73 | 73.85M | 15.34 | 16505 |
| 8192 | `16x512` | 154.38 | 53.06 | 70.35M | 14.64 | 15793 |
| 8192 | `32x256` | 164.64 | 49.76 | 70.74M | 14.29 | 15427 |
| 8192 | `64x128` | 157.85 | 51.90 | 68.35M | 13.86 | 14981 |
| 8192 | `128x64` | 155.72 | 52.61 | 65.35M | 13.71 | 14826 |

### LSTM Batch Cap Diagnostic

These rows use the same 32-lane `8x4` LSTM workload.

| cap | seconds | lanes/s | reviews/s | torch GiB | dedicated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1024` | 149.96 | 0.213 | 0.43M | 0.20 | 1622 |
| `2048` | 91.88 | 0.348 | 0.71M | 0.21 | 940 |
| `4096` | 57.65 | 0.555 | 1.12M | 0.37 | 1101 |
| `8192` | 33.98 | 0.942 | 1.91M | 1.28 | 2065 |
| `20000` | 30.43 | 1.052 | 2.13M | 7.64 | 9253 |
| `off` | 215.85 | 0.148 | 0.30M | 40.79 | 24089 |

### LSTM Cap 8192 Versus 20000

These rows use the same 8-user shape family. The cap 8192 rows are from
`sampling_lstm_confirmation_cap8192_20260511` and
`sampling_lstm_8user_scaling_cap8192_20260511`; the cap 20000 rows are from
`sampling_lstm_8x32_cap20000_20260511`,
`sampling_lstm_8x64_cap20000_20260511`, and
`sampling_lstm_8x128_cap20000_20260511`.

| lanes | shape | cap 8192 sec | cap 20000 sec | speedup | cap 8192 reviews/s | cap 20000 reviews/s | cap 8192 GiB/MiB | cap 20000 GiB/MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 256 | `8x32` | 160.37 | 100.96 | 1.59x | 3.24M | 5.14M | 1.54 / 2392 | 3.26 / 4322 |
| 512 | `8x64` | 306.31 | 191.39 | 1.60x | 3.39M | 5.42M | 4.03 / 5008 | 3.29 / 4358 |
| 1024 | `8x128` | 591.42 | 347.08 | 1.70x | 3.51M | 5.98M | 11.83 / 12990 | 8.98 / 10190 |

For larger lane counts, cap 20000 is consistently faster than cap 8192 and does
not show a dedicated-memory spill signal on this run. Linux `nvidia-smi` does
not expose shared GPU memory, so this conclusion is based on dedicated memory
and PyTorch reserved memory staying below physical VRAM. The `off` run remains
unsafe because it exceeded physical VRAM in PyTorch reserved memory and reached
24089 MiB dedicated memory even at 32 lanes.

### LSTM 1024-Lane Cap Sweep

These rows use `8x128`, fixed DR 0.98, 1024 total lanes. The `32768` and
`98304` confirmation rows use three repeats; other completed rows use one
repeat. Two larger probes were stopped before completion after early dedicated
memory reached the physical VRAM boundary.

| cap | repeats | seconds | lanes/s | reviews/s | gpu util % | mem util % | torch GiB | dedicated MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20000` | 1 | 347.08 | 2.95 | 5.98M | 87 | 68 | 8.98 | 10190 |
| `24000` | 1 | 318.86 | 3.21 | 6.51M | 89 | 72 | 10.89 | 12062 |
| `32768` | 1 | 274.84 | 3.73 | 7.55M | 92 | 80 | 6.13 | 7176 |
| `32768` confirm | 3 | 296.28 | 3.46 | 7.01M | 93 | 79 | 6.13 | 7180 |
| `36864` | 1 | 297.55 | 3.44 | 6.97M | 94 | 81 | 6.61 | 7641 |
| `49152` | 1 | 281.35 | 3.64 | 7.37M | 95 | 86 | 7.84 | 8901 |
| `65536` | 1 | 279.74 | 3.66 | 7.42M | 96 | 89 | 10.14 | 11271 |
| `98304` | 1 | 271.24 | 3.78 | 7.65M | 97 | 92 | 17.42 | 18755 |
| `98304` confirm | 3 | 273.14 | 3.75 | 7.60M | 97 | 93 | 17.42 | 18910 |
| `106496` | 1 | 271.84 | 3.77 | 7.63M | 97 | 94 | 19.94 | 21499 |

Stopped probes:

| cap | status |
| --- | --- |
| `114688` | stopped at 36s after dedicated memory reached 23988 MiB |
| `131072` | stopped at 36s after dedicated memory reached 23998 MiB |
| `2048 lanes / 8x256 / cap 12000` | stopped after dedicated memory reached about 24066 MiB |
| `2048 lanes / 8x256 / cap 20000` | stopped after external shared-GPU-memory monitoring showed spill |

The throughput curve flattens after `98304`: `106496` is effectively tied on
time but consumes another 2.5 GiB of PyTorch reserved memory, and the next two
larger caps immediately approach full physical VRAM. For this workload and GPU,
`98304` is the highest confirmed throughput point that still leaves several GiB
of dedicated-memory headroom. It should still be treated as a throughput
configuration, not a routine default, because Linux `nvidia-smi` does not expose
shared GPU memory and the cap operates close to the memory boundary.

### Environment Comparison

These rows compare the same 8-user shapes. LSTM uses
`SRS_LSTM_MAX_BATCH=20000`.

| lanes | shape | FSRS6 sec | FSRS6 reviews/s | LSTM sec | LSTM reviews/s | LSTM/FSRS sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | `8x32` | 17.96 | 20.74M | 100.96 | 5.14M | 5.6x |
| 512 | `8x64` | 18.22 | 40.89M | 191.39 | 5.42M | 10.5x |
| 1024 | `8x128` | 24.80 | 60.14M | 347.08 | 5.98M | 14.0x |

### Confirmation

| environment | shape | cap | repeats | mean sec | lanes/s | reviews/s | torch GiB | dedicated MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FSRS6 | `16x128` | n/a | 3 | 43.08 | 47.55 | 63.04M | 3.30 | 4182 |
| FSRS6 | `128x32` | n/a | 3 | 78.41 | 52.24 | 64.89M | 6.78 | 7765 |
| FSRS6 | `16x512` | n/a | 3 | 153.21 | 53.47 | 70.89M | 14.64 | 15824 |
| LSTM | `8x4` | `20000` | 3 | 30.59 | 1.05 | 2.13M | 7.64 | 8647 |
| LSTM | `8x32` | `20000` | 1 | 100.96 | 2.54 | 5.14M | 3.26 | 4322 |
| LSTM | `8x64` | `20000` | 1 | 191.39 | 2.68 | 5.42M | 3.29 | 4358 |
| LSTM | `8x128` | `20000` | 1 | 347.08 | 2.95 | 5.98M | 8.98 | 10190 |
| LSTM | `8x128` | `32768` | 3 | 296.28 | 3.46 | 7.01M | 6.13 | 7180 |
| LSTM | `8x128` | `98304` | 3 | 273.14 | 3.75 | 7.60M | 17.42 | 18910 |
| LSTM | `8x4` | `8192` | 3 | 37.03 | 0.86 | 1.75M | 1.28 | 2134 |
| LSTM | `8x32` | `8192` | 3 | 160.37 | 1.60 | 3.24M | 1.54 | 2392 |

## Analysis

FSRS6: the best practical default is 2048 lanes around `16x128` for routine
experiments. It confirmed at 43.08s and 63.04M reviews/s while using only
3.30 GiB PyTorch reserved memory and 4182 MiB dedicated memory. 4096 lanes
(`128x32` in confirmation) improves lanes/s to 52.24 but only modestly improves
review throughput for a much longer generation and about double the memory.
8192 lanes is the throughput ceiling region, not the default: it reaches
70.89M reviews/s but takes 153.21s and saturates memory bandwidth.

LSTM: the better conservative cap on this 24 GiB GPU is
`SRS_LSTM_MAX_BATCH=20000`, not 8192. The initial 32-lane-only comparison
understated this because both caps were invoked relatively few times. At 256,
512, and 1024 lanes, cap 20000 is 1.59x, 1.60x, and 1.70x faster than cap 8192.
It also stayed below physical VRAM in both PyTorch reserved memory and
dedicated-memory samples. The code default is now `65536`, which is a stronger
throughput compromise than 20000 while staying well below the 98304 memory
profile in this sweep. For a dedicated throughput run at 1024 lanes, cap `98304`
is better: the three-repeat confirmation finished in 273.14s at 7.60M reviews/s,
1.27x the cap 20000 review throughput and 1.08x the cap 32768 confirmation
throughput. Keep cap 8192 as a conservative fallback when the GPU is shared,
when other processes are resident, or when external shared-GPU-memory monitoring
shows spill. `off` is not viable: it reserved 40.79 GiB in PyTorch, hit
24089 MiB dedicated memory, and was slower than every capped run.

User-parallel shape matters for FSRS6, but less than total lanes once the GPU is
saturated. At 512-1024 lanes, shapes differ by several seconds and by roughly
10-20% review throughput. At 4096-8192 lanes, the system is mostly
memory-bandwidth-bound, and shape choices trade workload mix and setup overhead
rather than changing the ceiling.

LSTM under fixed DR 0.98 is primarily chunk-overhead/compute-bound up to the
`98304` region. Increasing the cap from 8192 to 20000 reduces chunking overhead
substantially, and raising 1024-lane throughput caps further continues to help
until the curve flattens around `98304`. Above that point, memory becomes the
limiting risk: `106496` is not faster, `114688` and `131072` immediately approach
full dedicated VRAM, and 2048-lane probes spill or approach the physical limit.

Safe lane caps from these runs:

- FSRS6 routine default: 2048 lanes.
- FSRS6 throughput experiment: 4096 lanes.
- FSRS6 upper bound on this 24 GiB GPU: 8192 lanes, only when long generations
  and memory-bandwidth saturation are acceptable.
- LSTM conservative run: 256 lanes at cap 20000.
- LSTM default throughput run: 1024 lanes at cap 65536, with shared-memory
  monitoring enabled.
- LSTM upper bound for this fixed DR 0.98 workload: 1024 lanes at cap 98304,
  only when 4-5 minute cells and high dedicated memory are acceptable.
- Use cap 8192 as a conservative memory fallback; avoid cap 1024 for
  performance sweeps and avoid `off` for memory safety.
