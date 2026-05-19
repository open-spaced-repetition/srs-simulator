# Oracle Stationary Finite CPU/GPU Benchmark

Machine summary: `artifacts/single_card_tradeoff/reports/current/report_summary.json`
Config: `experiments/single_card_tradeoff/configs/single_card_tradeoff_report_suite.toml`

## Question

How do CPU and CUDA runtimes compare for the default `oracle_stationary_finite_distill` workloads?

## Evidence

The benchmark runs `oracle_stationary_finite_distill.py` with the formal default single-card workload: `fsrs6_default`, 1825 days, default stationary finite teacher weights, 128 epochs, 64 steps per epoch, and 10,000 evaluation particles.

The multi-user benchmark runs `oracle_stationary_finite_distill_multiuser.py --per-user-models` for the first eight benchmark users with the same default stationary finite distill recipe and button-usage costs.

Each workload/device pair is run once, so the report gives observed timings without a variance estimate. CUDA memory and spill fields come from the benchmark GPU monitor artifacts.

Source artifacts:
- `oracle_stationary_finite_cpu_gpu_summary`: `artifacts/single_card_tradeoff/oracle_stationary_finite_cpu_gpu_benchmark/summary.csv`
- `oracle_stationary_finite_cpu_gpu_runs`: `artifacts/single_card_tradeoff/oracle_stationary_finite_cpu_gpu_benchmark/runs.csv`
- `oracle_stationary_finite_cpu_gpu_metadata`: `artifacts/single_card_tradeoff/oracle_stationary_finite_cpu_gpu_benchmark/metadata.json`
- `oracle_stationary_finite_multiuser_cpu_gpu_summary`: `artifacts/single_card_tradeoff/oracle_stationary_finite_multiuser_cpu_gpu_benchmark/summary.csv`
- `oracle_stationary_finite_multiuser_cpu_gpu_runs`: `artifacts/single_card_tradeoff/oracle_stationary_finite_multiuser_cpu_gpu_benchmark/runs.csv`
- `oracle_stationary_finite_multiuser_cpu_gpu_metadata`: `artifacts/single_card_tradeoff/oracle_stationary_finite_multiuser_cpu_gpu_benchmark/metadata.json`

Notes:
- Configured repeats per device: 1; CUDA available: True.
- Benchmark settings: epochs=128, steps_per_epoch=64, eval_particles=10000.
- Multi-user benchmark settings: users=[1, 2, 3, 4, 5, 6, 7, 8], repeats=1, eval_particles=10000.

## Results

### Single-user device timing

| device | wall_s | speedup_vs_cpu | train_s | eval_s | repeats |
| --- | --- | --- | --- | --- | --- |
| cpu | 48.35 | 1.00x | 42.32 | 4.68 | 1 |
| cuda | 100.97 | 0.48x | 78.37 | 21.21 | 1 |

### Single-user quality guard

| device | params | final CE | train agreement | eval agreement |
| --- | --- | --- | --- | --- |
| cpu | 476 | 0.60906 | 75.17% | 75.34% |
| cuda | 476 | 0.60612 | 75.36% | 74.98% |

### Single-user CUDA memory

| device | samples | dedicated MiB | shared peak MiB | spill |
| --- | --- | --- | --- | --- |
| cuda | 28 | 2585.0 | 172.7 | no |

### Multi-user device timing

| device | wall_s | speedup_vs_cpu | teacher_s | train_s | eval_s |
| --- | --- | --- | --- | --- | --- |
| cpu | 456.63 | 1.00x | 175.88 | 34.10 | 244.23 |
| cuda | 175.68 | 2.60x | 60.64 | 42.27 | 70.82 |

### Multi-user quality guard

| device | users | params/user | ensemble params | mean CE | eval agreement |
| --- | --- | --- | --- | --- | --- |
| cpu | 8 | 476 | 3,808 | 0.70141 | 72.25% |
| cuda | 8 | 476 | 3,808 | 0.69721 | 72.39% |

### Multi-user CUDA memory

| device | samples | dedicated MiB | shared peak MiB | spill |
| --- | --- | --- | --- | --- |
| cuda | 49 | 3255.0 | 211.9 | no |

## Reproduction Profile

The TOML profile records the command and expected outputs used to reproduce this report input. CUDA reruns write `performance_summary.json` and `gpu_monitor/` memory samples under their configured output directories.

| command | expected outputs present |
| --- | --- |
| benchmark_oracle_stationary_finite_cpu_gpu | 10/10 |
| benchmark_oracle_stationary_finite_multiuser_cpu_gpu | 12/12 |

## Conclusion

Single-user: CUDA is slower than CPU for the default stationary finite distill workload in the single observed run: CPU is 2.09x faster. Multi-user: CUDA completes the default stationary finite distill workload 2.60x faster than CPU in the single observed run. Treat both as point estimates until the benchmark is rerun with multiple repeats.

## Artifacts

- Published report: `docs/single_card_tradeoff/experiments/2026-05-18-oracle_stationary_finite_cpu_gpu_benchmark.md`
- Artifact report: `artifacts/single_card_tradeoff/reports/current/oracle_stationary_finite_cpu_gpu_benchmark.md`
- Report root: `artifacts/single_card_tradeoff/reports/current`
