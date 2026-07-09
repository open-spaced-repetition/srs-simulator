# RL Scheduler Seed Refactor Plan

Date: 2026-07-08

## Objective

Remove the seed coupling between RL scheduler training and formal evaluation.
The old setup reused top-level `seed = 42` for training simulations, staged
baselines, and final sweeps. That can bias policy search toward seed-specific
simulator quirks.

The refactor separates:

- training seed: top-level `seed`, still used for optimizer provenance and
  training artifact metadata.
- evaluation seed: `[sweep].seed`, used by staged baselines, formal sweeps,
  Pareto building, reserved tests, and evaluation command templates.
- training simulation seed: `training.policy_search.simulation_seed_strategy`,
  used only inside candidate simulation during training.

Formal checked-in RL profiles use `seed = 42` and `[sweep].seed = 43`.

## Tracking Status

Last updated: 2026-07-09 after all C6 configs were attempted: forty runner
passes, one documented superseded quality-v1 training failure, and zero missing
runs.

Update this table whenever a checkpoint changes state. Keep evidence to
machine-readable artifacts, committed config/docs, and exact commands.

| checkpoint | status | evidence |
| --- | --- | --- |
| C1. Schema And Runner Plumbing | Done | `[sweep].seed`, `ExperimentConfig.evaluation_seed`, evaluation-seed validation, and command placeholders implemented. |
| C2. Training Seed Rotation | Done | Shared helper added; ADR, Cost-ADR, AP, portfolio, and in-process batch training record `simulation_seed`. |
| C3. Formal Config Migration | Done | All 41 checked-in RL scheduler configs load; formal sweep configs use `[sweep].seed = 43`; learned profiles use generation seed rotation. |
| C4. Code Formatting And Static Checks | Done | `ruff format`, focused `unittest`, `pyright`, config-load check, and `git diff --check` passed. |
| C5. Baseline Pre-Generation | Done | Generated 4,720 seed-43 FSRS6 baseline JSONL logs across manifest and ordinary-grid baselines; metadata validation and representative `stage-baseline` gates passed. |
| C6. Learned Profile Reruns | Done With Superseded Failure | Forty C6 runner passes and zero missing configs; `fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1` remains a documented quality-v1 objective/search failure superseded by `quality_v2` and the dependent hybrid profile. Completed runs validated training seed rotation where applicable, seed-43 sweep logs where applicable, Pareto and analysis outputs where applicable, artifact file validation where applicable, and GPU monitor spill fields. |

Latest verification commands:

```bash
uv run ruff format simulator/experiment_infra/schemas.py simulator/experiment_infra/runner.py simulator/batched_sweep/config.py experiments/retention_sweep/build_pareto_users.py experiments/rl_scheduler/policy_search_common.py experiments/rl_scheduler/portfolio_training_common.py experiments/rl_scheduler/run_portfolio_workflow.py experiments/rl_scheduler/train_cmaes_fsrs6_adr.py experiments/rl_scheduler/train_cmaes_fsrs6_cost_adr.py experiments/rl_scheduler/train_cmaes_fsrs6_ap.py simulator/experiment_infra/training_batch.py tests/test_batched_sweep_config.py tests/test_experiment_infra_schemas.py tests/test_experiment_infra_runner.py tests/test_rl_scheduler_portfolio_workflow.py tests/test_rl_scheduler_unified_workflow_config.py
uv run python -m unittest tests.test_batched_sweep_config tests.test_experiment_infra_schemas tests.test_experiment_infra_runner tests.test_rl_scheduler_unified_workflow_config tests.test_rl_scheduler_portfolio_workflow
uv run python -m unittest tests.test_train_fsrs6_adr tests.test_fsrs6_cost_adr_train tests.test_train_fsrs6_ap_portfolio tests.test_fsrs6_ap_policy tests.test_experiment_infra_artifacts
uv run pyright
uv run python -c 'from pathlib import Path; from simulator.experiment_infra import ExperimentConfig; paths=sorted(Path("experiments/rl_scheduler/configs").glob("*.toml")); [ExperimentConfig.from_toml(path) for path in paths]; print(f"loaded {len(paths)} configs")'
git diff --check
```

## C5 Completion Log

C5 was completed after GPU access became available.

GPU availability check:

```bash
uv run python -c 'import torch; print("cuda_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("cuda_version", torch.version.cuda)'
```

Result:

- `cuda_available True`
- `device_count 1`
- CUDA build version: `12.6`
- Device: NVIDIA GeForce RTX 4090 D

Executed baseline generation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_portfolio_workflow.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml --skip-manifest --skip-formal-stages --skip-report
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_portfolio_workflow.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_v3.toml --skip-manifest --skip-formal-stages --skip-report
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/retention_sweep/run_sweep_users_batched.py --config experiments/rl_scheduler/configs/fsrs6_adr_cmaes_users_1_8.toml --env fsrs6,lstm --sched fsrs6 --run-id fsrs6_baseline_users_1_8_grid_seed43 --no-progress
```

Generated seed-43 JSONL logs:

| run id | logs | users | envs | validation |
| --- | ---: | ---: | --- | --- |
| `fsrs6_baseline_users_1_8_16dr_pop16_gen5` | 256 | 8 | `fsrs6,lstm` | metadata errors: 0 |
| `fsrs6_baseline_users_1_128_16dr_pop16_gen5` | 4,096 | 128 | `fsrs6,lstm` | metadata errors: 0 |
| `fsrs6_baseline_users_1_8_grid_seed43` | 368 | 8 | `fsrs6,lstm` | metadata errors: 0 |

Total `seed=43` JSONL logs under `logs/retention_sweep`: 4,720.

Metadata validation checked the first meta record in every generated JSONL and
confirmed:

- `seed = 43`
- `scheduler = fsrs6`
- `engine = batched`
- expected `run_id`
- expected user sets and `fsrs6,lstm` environments

GPU monitor evidence:

- users 1-8 manifest:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/fsrs6_adr_portfolio_users_1_8_v3/workflow/sweep-fsrs6-baseline/gpu_monitor/summary.json`
  - `nvidia_smi_peak_memory_used_mib = 4197.0`
  - `nvidia_smi_peak_utilization_percent = 58.0`
  - `shared_memory_spill_detected = false`
- users 1-128 manifest:
  `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128/fsrs6_adr_portfolio_users_1_128_v3/workflow/sweep-fsrs6-baseline/gpu_monitor/summary.json`
  - `nvidia_smi_peak_memory_used_mib = 8729.0`
  - `nvidia_smi_peak_utilization_percent = 91.0`
  - `shared_memory_spill_detected = false`

Representative `stage-baseline` validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml --stage stage-baseline --run-id c5_seed43_baseline_check_users_1_8_manifest
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_v3.toml --stage stage-baseline --run-id c5_seed43_baseline_check_users_1_128_manifest
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_cmaes_users_1_8.toml --stage stage-baseline --run-id c5_seed43_baseline_check_users_1_8_grid
```

Representative `stage-baseline` artifacts:

- `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c5_seed43_baseline_check_users_1_8_manifest/stage-baseline/baseline_summary.json`
  - `passed = true`, `failures = []`, `staged_logs = 256`
- `artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128/c5_seed43_baseline_check_users_1_128_manifest/stage-baseline/baseline_summary.json`
  - `passed = true`, `failures = []`, `staged_logs = 4096`
- `artifacts/rl_scheduler/fsrs6_adr_cmaes_users_1_8/c5_seed43_baseline_check_users_1_8_grid/stage-baseline/baseline_summary.json`
  - `passed = true`, `failures = []`, `staged_logs = 368`

All three corresponding `gate_summary.json` files have `passed = true` and
`failures = []`.

## C5 Executed Commands

Manifest-driven users 1-8 baseline:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_portfolio_workflow.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml --skip-manifest --skip-formal-stages --skip-report
```

Manifest-driven users 1-128 baseline:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_portfolio_workflow.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_v3.toml --skip-manifest --skip-formal-stages --skip-report
```

Ordinary users 1-8 FSRS6 DR-grid baseline for non-manifest profiles:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/retention_sweep/run_sweep_users_batched.py --config experiments/rl_scheduler/configs/fsrs6_adr_cmaes_users_1_8.toml --env fsrs6,lstm --sched fsrs6 --run-id fsrs6_baseline_users_1_8_grid_seed43 --no-progress
```

Dry-run lane counts already checked:

- users 1-8 manifest: 256 lanes, LSTM max lanes 1024.
- users 1-128 manifest: 4096 lanes, LSTM split into 2 batches at max lanes
  1024.
- users 1-8 ordinary grid: 368 lanes.

## Seed Policy

- `stage-baseline` must only stage pre-generated baseline logs. It must not
  silently run new baselines when seed-43 logs are missing.
- Training artifacts keep metadata `seed = 42`.
- Formal evaluation logs must have metadata `seed = 43`.
- Learning profiles use generation-level training simulation seed rotation:
  `simulation_seed = seed + generation * simulation_seed_stride + offset`.
- `simulation_seed_stride = 1009`.
- Candidates in the same generation share the same simulation seed. This keeps
  candidate comparisons fair within a generation and avoids increasing training
  volume.
- SMS-EMOA portfolio training evaluates the initial population with helper
  generation `0`, then evaluates recorded offspring generations `0..N-1` with
  helper generations `1..N`. In progress files, offspring records therefore
  show `simulation_seed = seed + (recorded_generation + 1) * stride`.
- AP training keeps its existing DR-chunk offset on top of the generation seed.

## Checkpoints

### C1. Schema And Runner Plumbing

Status: Done.

Target:

- Add `[sweep].seed` to the experiment schema.
- Add `ExperimentConfig.evaluation_seed`.
- Use `evaluation_seed` for baseline staging, sweep execution, sweep log
  validation, Pareto command defaults, and reserved-test validation.
- Keep artifact metadata validation on top-level `seed`.
- Add command-template placeholders:
  `training_seed`, `evaluation_seed`; keep `{seed}` as training seed for train
  commands and evaluation seed for evaluation commands.

Checks:

```bash
uv run python -m unittest tests.test_experiment_infra_schemas tests.test_experiment_infra_runner
uv run pyright
```

### C2. Training Seed Rotation

Status: Done.

Target:

- Add shared training simulation seed helpers in `policy_search_common.py`.
- Apply the helpers to standalone CMA-ES ADR, Cost-ADR, AP, portfolio training,
  and in-process batch training.
- Record `simulation_seed` in training progress/history.

Checks:

```bash
uv run python -m unittest tests.test_experiment_infra_schemas
uv run python -m unittest tests.test_experiment_infra_runner
```

### C3. Formal Config Migration

Status: Done.

Target:

- Add `[sweep].seed = 43` to every formal config with a sweep stage.
- Add generation seed rotation settings to every checked-in learned profile
  with `train-overfit`.
- Do not add training seed rotation to native FSRS3 evaluation or hybrid-only
  eval configs.

Checks:

```bash
uv run python -m unittest tests.test_rl_scheduler_unified_workflow_config
uv run python -m unittest tests.test_rl_scheduler_portfolio_workflow
```

### C4. Code Formatting And Static Checks

Status: Done.

Target:

- Keep imports, formatting, and types clean.
- Do not run long experiments before these checks pass.

Checks:

```bash
uv run ruff format
uv run python -m unittest tests.test_experiment_infra_schemas tests.test_experiment_infra_runner tests.test_rl_scheduler_unified_workflow_config tests.test_rl_scheduler_portfolio_workflow
uv run pyright
```

### C5. Baseline Pre-Generation

Status: Done.

Target:

- Generate seed-43 FSRS6 baseline logs before formal learned reruns.
- Use the configured baseline DR manifests for portfolio profiles.
- Confirm `stage-baseline` passes only by copying exact seed-43 logs.

Evidence:

- Baseline JSONL metadata shows `seed = 43`, `scheduler = fsrs6`,
  `engine = batched`, and expected run IDs for all 4,720 generated logs.
- `stage-baseline/baseline_summary.json` passes and lists all required users and
  DRs for the manifest users 1-8, manifest users 1-128, and ordinary users 1-8
  grid checks.
- All representative `gate_summary.json` files have `passed = true` and
  `failures = []`.

### C6. Learned Profile Reruns

Status: Done with one documented superseded quality-v1 failure.

Target:

- Rerun training for learned profiles using generation seed rotation.
- Rerun sweep/build-pareto/analyze-pareto with evaluation seed 43.
- Regenerate reports only from current `analysis_summary.json`.

Evidence:

- Training artifact metadata remains `seed = 42`.
- Training progress/history includes per-generation `simulation_seed`.
- Sweep logs and Pareto inputs show `seed = 43`.
- Reports cite `report_summary.json`.

Current attempted C6 runs:

| config | run id | status | evidence |
| --- | --- | --- | --- |
| `anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml` | `c6_seed43_anki_sm2_ap_portfolio_users_1_8_pop16_20_v1` | Done | `all_summary.json` passed; 128 Anki SM2 AP portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` and scheduler `anki_sm2_ap` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1` | Done | `all_summary.json` passed; 2 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 32 sweep JSONL logs validated with `seed = 43`; build-pareto produced 2 JSON and 2 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1` | Done | Hybrid root assembled from the C6 coverage and quality-v2 source runs; selected coverage for users 1,7,8 and quality-v2 for users 2-6; all selected artifacts validate with required files and metadata `seed = 42`; `all_summary.json` passed; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; sweep GPU monitor reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1` | Done | `all_summary.json` passed; 2 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 32 sweep JSONL logs validated with `seed = 43`; build-pareto produced 2 JSON and 2 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1` | Superseded Failure | `stage-baseline` passed and staged 256 seed-43 baseline logs; `train-overfit` failed with `runner-failed`; 6 of 8 commands succeeded and artifacts validated; failed users were 5 and 8, each with `best_hypervolume_delta = 0.0`; training progress still recorded `simulation_seed = 42 + generation * 1009` through generation 19 for all users; no formal sweep ran; train GPU monitor reported `shared_memory_spill_detected = false`. This quality-v1 recipe is an already documented objective/search failure and is superseded by the passing `quality_v2` and hybrid profiles. |
| `fsrs3_scheduler_users_1_8.toml` | `c6_seed43_fsrs3_scheduler_users_1_8` | Done | Evaluation-only run; `all_summary.json` passed; `stage-baseline` staged 256 seed-43 FSRS6 baseline logs; 256 sweep JSONL logs validated with `seed = 43`, scheduler `fsrs3`, engine `batched`, and environments `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; sweep GPU monitor reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_cmaes_users_1_8.toml` | `c6_seed43_fsrs6_adr_cmaes_users_1_8` | Done | `all_summary.json` passed; 184 artifacts validated with metadata `seed = 42`; 184 training progress files recorded generations `0-19` with `simulation_seed = 42 + generation * 1009`; 368 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_linear_cmaes_users_1_8.toml` | `c6_seed43_fsrs6_adr_linear_cmaes_users_1_8` | Done | `all_summary.json` passed; 184 artifacts validated with metadata `seed = 42`; 184 training progress files recorded generations `0-19` with `simulation_seed = 42 + generation * 1009`; 368 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_linear_portfolio_users_1_8.toml` | `c6_seed43_fsrs6_adr_linear_portfolio_users_1_8` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1` | Done | `all_summary.json` passed; 128 LSTM-trained portfolio child artifacts validated with metadata `seed = 42` and environment `lstm`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_128_pop16_v1.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_128_pop16_v1` | Done | `all_summary.json` passed; 2048 portfolio child artifacts validated with metadata `seed = 42`; 128 training progress files recorded initial population `simulation_seed = 42` plus 2560 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 4096 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 128 JSON and 128 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_128_v3.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_128_v3` | Done | `all_summary.json` passed; 2048 portfolio child artifacts validated with metadata `seed = 42`; 128 training progress files recorded initial population `simulation_seed = 42` plus 2560 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 4096 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 128 JSON and 128 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 80 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 240 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_v1` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_portfolio_users_1_8_v3.toml` | `c6_seed43_fsrs6_adr_portfolio_users_1_8_v3` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_adr_time_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_adr_time_portfolio_users_1_8_pop16_v1` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_adr_time` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_ap_cmaes_users_1_8.toml` | `c6_seed43_fsrs6_ap_cmaes_users_1_8` | Done | `all_summary.json` passed; 184 artifacts validated with metadata `seed = 42`; 8 training progress files recorded generations `0-19` with `simulation_seed = 42 + generation * 1009`; 368 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_ap_portfolio_users_1_128_v1.toml` | `c6_seed43_fsrs6_ap_portfolio_users_1_128_v1` | Done | `all_summary.json` passed; 2048 AP portfolio child artifacts validated with metadata `seed = 42`; 128 training progress files recorded initial population `simulation_seed = 42` plus 2560 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 4096 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_ap` across `fsrs6,lstm`; build-pareto produced 128 JSON and 128 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_ap_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_ap_portfolio_users_1_8_pop16_v1` | Done | `all_summary.json` passed; 128 AP portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_ap_portfolio_users_1_8_v2.toml` | `c6_seed43_fsrs6_ap_portfolio_users_1_8_v2` | Done | `all_summary.json` passed; 128 AP portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml` | `c6_seed43_fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1` | Done | `all_summary.json` passed; 128 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_default_adr` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1` | Done | Rerun after adding the missing wrapper default `per_user_models = false`; `all_summary.json` passed; `stage-baseline` staged 256 seed-43 baseline logs; 128 oracle distill portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_oracle_stationary_finite_distill` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.toml` | `c6_seed43_fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1` | Done | `all_summary.json` passed; `stage-baseline` staged 256 seed-43 baseline logs; 128 oracle distill r4d1/e512 portfolio child artifacts validated with metadata `seed = 42`; 8 training progress files recorded initial population `simulation_seed = 42` plus 160 SMS-EMOA generation records with `simulation_seed = 42 + (recorded_generation + 1) * 1009`; 256 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_oracle_stationary_finite_distill` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 608 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1` | Done | `all_summary.json` passed; 128 artifacts validated with metadata `seed = 42`; 128 training progress files recorded 2560 CMA-ES generation records with `simulation_seed = 42 + generation * 1009`; 4096 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_cost_adr` across `fsrs6,lstm`; build-pareto produced 128 JSON and 128 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1` | Done | Train-only run; `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; no sweep stage was configured; train GPU monitor reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1` | Done | `all_summary.json` passed; 128 artifacts validated with metadata `seed = 42`; 128 training progress files recorded 2560 CMA-ES generation records with `simulation_seed = 42 + generation * 1009`; 4096 sweep JSONL logs validated with `seed = 43` and scheduler `fsrs6_cost_adr` across `fsrs6,lstm`; build-pareto produced 128 JSON and 128 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42` and training environment `lstm`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 256 sweep JSONL logs validated with `seed = 43` across `fsrs6,lstm`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 128 fsrs6-only sweep JSONL logs validated with `seed = 43`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 128 fsrs6-only sweep JSONL logs validated with `seed = 43`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false`. |
| `fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1` | Done | Clean rerun after CUDA allocator bootstrap validation passed all runner gates; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 128 fsrs6-only sweep JSONL logs validated with `seed = 43`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false` with train summed shared-memory peak `267333632` bytes and sweep summed shared-memory peak `246632448` bytes. The earlier mixed rerun root was archived as `_mixed_archive_20260709T1034` and is not counted. |
| `fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1.toml` | `c6_seed43_fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1` | Done | `all_summary.json` passed; 8 artifacts validated with metadata `seed = 42`; 20 training generations per user recorded `simulation_seed = 42 + generation * 1009`; 128 fsrs6-only sweep JSONL logs validated with `seed = 43`; build-pareto produced 8 JSON and 8 PNG outputs; analyze-pareto produced `analysis_summary.json`; train and sweep GPU monitors reported `shared_memory_spill_detected = false` with train summed shared-memory peak `248270848` bytes and sweep summed shared-memory peak `248135680` bytes. |

First completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_2/c6_seed43_fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1
```

First completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_2/c6_seed43_fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_2/c6_seed43_fsrs6_cost_adr_coverage_users_1_2_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

First completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 32`.
- `train-overfit`: `passed = true`, `artifacts_validated = 2`,
  `commands_succeeded = 2`.
- `sweep`: `passed = true`, `logs_validated = 32`, `batch_lanes = 32`.
- `build-pareto`: `passed = true`, `result_json_files = 2`,
  `plot_files = 2`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `4131.0` MiB, peak utilization `84.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4131.0` MiB, peak utilization `42.0%`,
  `shared_memory_spill_detected = false`.

Second completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1
```

Second completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Second completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- `sweep`: `passed = true`, `logs_validated = 256`, `batch_lanes = 256`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `7360.0` MiB, peak utilization `95.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `7356.0` MiB, peak utilization `58.0%`,
  `shared_memory_spill_detected = false`.

Third completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/c6_seed43_fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1
```

Third completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/c6_seed43_fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Third completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- `sweep`: `passed = true`, `logs_validated = 256`, `batch_lanes = 256`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `7346.0` MiB, peak utilization `95.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `7356.0` MiB, peak utilization `65.0%`,
  `shared_memory_spill_detected = false`.

Hybrid completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/c6_seed43_fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1
```

Hybrid build and validation commands:

```bash
uv run python experiments/rl_scheduler/build_fsrs6_cost_adr_hybrid.py --source coverage=artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1 --source quality_v2=artifacts/rl_scheduler/fsrs6_cost_adr_quality_v2_users_1_8/c6_seed43_fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1 --baseline-run-root artifacts/rl_scheduler/fsrs6_cost_adr_coverage_users_1_8/c6_seed43_fsrs6_cost_adr_coverage_users_1_8_pop16_gen20_v1 --output-run-root artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/c6_seed43_fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1 --users 1-8
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_quality_hybrid_users_1_8/c6_seed43_fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
uv run python -m unittest tests.test_fsrs6_cost_adr_hybrid
```

Hybrid completed run metrics:

- selected sources: `coverage = 3` users (`1,7,8`), `quality_v2 = 5`
  users (`2-6`).
- `preflight`: `passed = true`, `gpu_guard_passed = 1`.
- `train-overfit`: preassembled hybrid root, `passed = true`,
  `artifacts_validated = 8`; selected artifacts pass `--require-files`.
- `sweep`: `passed = true`, `logs_validated = 256`, `batch_lanes = 256`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- sweep GPU monitor: peak `4568.0` MiB, peak utilization `67.0%`,
  `shared_memory_spill_detected = false`.

Hybrid infrastructure fix:

- `experiments/rl_scheduler/build_fsrs6_cost_adr_hybrid.py` now copies selected
  source training command records referenced by artifact metadata into the
  hybrid train root.
- `tests/test_fsrs6_cost_adr_hybrid.py` covers the copied command records.

Fifth completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_quality_users_1_2/c6_seed43_fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1
```

Fifth completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_quality_users_1_2/c6_seed43_fsrs6_cost_adr_quality_users_1_2_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Fifth completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 32`.
- `train-overfit`: `passed = true`, `artifacts_validated = 2`,
  `commands_succeeded = 2`.
- `sweep`: `passed = true`, `logs_validated = 32`, `batch_lanes = 32`.
- `build-pareto`: `passed = true`, `result_json_files = 2`,
  `plot_files = 2`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `4141.0` MiB, peak utilization `82.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4127.0` MiB, peak utilization `41.0%`,
  `shared_memory_spill_detected = false`.

Failed C6 run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_quality_users_1_8/c6_seed43_fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1
```

Failed C6 run command:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1
```

Failed C6 run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = false`, `failures = ["runner-failed"]`,
  `commands_attempted = 8`, `commands_succeeded = 6`,
  `artifacts_validated = 6`.
- batch overfit gate: `passed_points = 6`, `required_points = 7`,
  `pass_fraction = 0.75`, `required_pass_fraction = 0.8`.
- failed training notes: `user = 5, baseline_dr = 0.9` and `user = 8,
  baseline_dr = 0.9`.
- no candidate sweep logs were produced because the run stopped before
  `sweep`.
- train GPU monitor: peak `7708.0` MiB, peak utilization `96.0%`,
  `shared_memory_spill_detected = false`.

FSRS3 evaluation-only completed run artifact root:

```text
artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/c6_seed43_fsrs3_scheduler_users_1_8
```

FSRS3 evaluation-only completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs3_scheduler_users_1_8.toml --stage all --run-id c6_seed43_fsrs3_scheduler_users_1_8
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs3_scheduler_users_1_8/c6_seed43_fsrs3_scheduler_users_1_8
```

FSRS3 evaluation-only completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs3`, `engine = batched`,
  `run_id = c6_seed43_fsrs3_scheduler_users_1_8`, users `1-8`,
  environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- sweep GPU monitor: peak `3853.0` MiB, peak utilization `51.0%`,
  `shared_memory_spill_detected = false`.

CMA-ES FSRS6 ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_cmaes_users_1_8/c6_seed43_fsrs6_adr_cmaes_users_1_8
```

CMA-ES FSRS6 ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_cmaes_users_1_8.toml --stage all --run-id c6_seed43_fsrs6_adr_cmaes_users_1_8
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_cmaes_users_1_8/c6_seed43_fsrs6_adr_cmaes_users_1_8
find artifacts/rl_scheduler/fsrs6_adr_cmaes_users_1_8/c6_seed43_fsrs6_adr_cmaes_users_1_8/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

CMA-ES FSRS6 ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 368`.
- `train-overfit`: `passed = true`, `artifacts_validated = 184`,
  `commands_succeeded = 184`.
- training progress: 184 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009` across 3,680 generation records.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `environment = fsrs6`, 23 baseline desired-retention values, `lambda = 0.5`.
- `sweep`: `passed = true`, `batch_lanes = 368`, `log_paths = 368`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`,
  23 desired-retention values.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `10664.0` MiB, peak utilization `95.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `10625.0` MiB, peak utilization `57.0%`,
  `shared_memory_spill_detected = false`.

Linear CMA-ES FSRS6 ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_linear_cmaes_users_1_8/c6_seed43_fsrs6_adr_linear_cmaes_users_1_8
```

Linear CMA-ES FSRS6 ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_linear_cmaes_users_1_8.toml --stage all --run-id c6_seed43_fsrs6_adr_linear_cmaes_users_1_8
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_linear_cmaes_users_1_8/c6_seed43_fsrs6_adr_linear_cmaes_users_1_8
find artifacts/rl_scheduler/fsrs6_adr_linear_cmaes_users_1_8/c6_seed43_fsrs6_adr_linear_cmaes_users_1_8/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Linear CMA-ES FSRS6 ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 368`.
- `train-overfit`: `passed = true`, `artifacts_validated = 184`,
  `commands_succeeded = 184`.
- training progress: 184 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009` across 3,680 generation records.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `environment = fsrs6`, 23 baseline desired-retention values, `lambda = 0.5`.
- `sweep`: `passed = true`, `batch_lanes = 368`, `log_paths = 368`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`,
  23 desired-retention values.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `16167.0` MiB, peak utilization `99.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `10917.0` MiB, peak utilization `67.0%`,
  `shared_memory_spill_detected = false`.

Linear FSRS6 ADR portfolio completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_linear_portfolio_users_1_8/c6_seed43_fsrs6_adr_linear_portfolio_users_1_8
```

Linear FSRS6 ADR portfolio completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml --stage all --run-id c6_seed43_fsrs6_adr_linear_portfolio_users_1_8
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_linear_portfolio_users_1_8/c6_seed43_fsrs6_adr_linear_portfolio_users_1_8
find artifacts/rl_scheduler/fsrs6_adr_linear_portfolio_users_1_8/c6_seed43_fsrs6_adr_linear_portfolio_users_1_8/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Linear FSRS6 ADR portfolio completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `feature_version = fsrs6_adr_log_linear_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `3395.0` MiB, peak utilization `57.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4079.0` MiB, peak utilization `64.0%`,
  `shared_memory_spill_detected = false`.

LSTM-train FSRS6 ADR portfolio completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1
```

LSTM-train FSRS6 ADR portfolio completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1
find artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_lstm_train_portfolio_users_1_8_pop16_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

LSTM-train FSRS6 ADR portfolio completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `environment = lstm`, `feature_version = fsrs6_adr_log_poly_v1`, 16
  portfolio child policies per user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `4287.0` MiB, peak utilization `73.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4143.0` MiB, peak utilization `66.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 ADR portfolio pop16 gen10 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1
```

FSRS6 ADR portfolio pop16 gen10 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1
find artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen10_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 ADR portfolio pop16 gen10 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 80 SMS-EMOA generation records cover recorded
  generations `0-9` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `feature_version = fsrs6_adr_log_poly_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2867.0` MiB, peak utilization `57.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4060.0` MiB, peak utilization `52.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 ADR portfolio pop16 gen20 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_v1
```

FSRS6 ADR portfolio pop16 gen20 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_v1
find artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 ADR portfolio pop16 gen20 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `feature_version = fsrs6_adr_log_poly_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2930.0` MiB, peak utilization `59.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4048.0` MiB, peak utilization `62.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 ADR portfolio pop16 gen30 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1
```

FSRS6 ADR portfolio pop16 gen30 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1
find artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_pop16_gen30_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 ADR portfolio pop16 gen30 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 240 SMS-EMOA generation records cover recorded
  generations `0-29` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `feature_version = fsrs6_adr_log_poly_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2888.0` MiB, peak utilization `62.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4049.0` MiB, peak utilization `62.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 ADR portfolio v3 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_v3
```

FSRS6 ADR portfolio v3 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_8_v3
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_v3
find artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_8/c6_seed43_fsrs6_adr_portfolio_users_1_8_v3/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 ADR portfolio v3 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr`,
  `feature_version = fsrs6_adr_log_poly_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `3404.0` MiB, peak utilization `68.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4085.0` MiB, peak utilization `59.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 ADR time portfolio pop16 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/c6_seed43_fsrs6_adr_time_portfolio_users_1_8_pop16_v1
```

FSRS6 ADR time portfolio pop16 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_time_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_time_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/c6_seed43_fsrs6_adr_time_portfolio_users_1_8_pop16_v1
find artifacts/rl_scheduler/fsrs6_adr_time_portfolio_users_1_8/c6_seed43_fsrs6_adr_time_portfolio_users_1_8_pop16_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 ADR time portfolio pop16 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_adr_time`,
  `feature_version = fsrs6_adr_log_poly_time_v1`, 16 portfolio child policies
  per user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_adr_time`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2872.0` MiB, peak utilization `62.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4063.0` MiB, peak utilization `64.0%`,
  `shared_memory_spill_detected = false`.

AP CMA-ES FSRS6 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_ap_cmaes_users_1_8/c6_seed43_fsrs6_ap_cmaes_users_1_8
```

AP CMA-ES FSRS6 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_ap_cmaes_users_1_8.toml --stage all --run-id c6_seed43_fsrs6_ap_cmaes_users_1_8
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_ap_cmaes_users_1_8/c6_seed43_fsrs6_ap_cmaes_users_1_8
find artifacts/rl_scheduler/fsrs6_ap_cmaes_users_1_8/c6_seed43_fsrs6_ap_cmaes_users_1_8/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

AP CMA-ES FSRS6 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 368`.
- `train-overfit`: `passed = true`, `artifacts_validated = 184`,
  `commands_succeeded = 184`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009` across 160 generation records.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_ap`,
  `environment = fsrs6`, 23 baseline desired-retention values.
- `sweep`: `passed = true`, `batch_lanes = 368`, `log_paths = 368`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_ap`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`,
  23 desired-retention values.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `10626.0` MiB, peak utilization `99.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `10630.0` MiB, peak utilization `60.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 AP portfolio pop16 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_pop16_v1
```

FSRS6 AP portfolio pop16 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_ap_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_ap_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_pop16_v1
find artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_pop16_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 AP portfolio pop16 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; 160 SMS-EMOA generation records cover
  recorded generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_ap`,
  `action_space = fsrs6_ap_weight_delta_portfolio_child`, 16 portfolio child
  policies per user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_ap`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2873.0` MiB, peak utilization `62.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4065.0` MiB, peak utilization `63.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 AP portfolio v2 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_v2
```

FSRS6 AP portfolio v2 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_ap_portfolio_users_1_8_v2.toml --stage all --run-id c6_seed43_fsrs6_ap_portfolio_users_1_8_v2
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_v2
find artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_8/c6_seed43_fsrs6_ap_portfolio_users_1_8_v2/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 AP portfolio v2 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; 160 SMS-EMOA generation records cover
  recorded generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_ap`,
  `action_space = fsrs6_ap_weight_delta_portfolio_child`, 16 portfolio child
  policies per user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_ap`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `3397.0` MiB, peak utilization `69.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `4123.0` MiB, peak utilization `66.0%`,
  `shared_memory_spill_detected = false`.

FSRS6 default ADR portfolio completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/c6_seed43_fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1
```

FSRS6 default ADR portfolio completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml --stage all --run-id c6_seed43_fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/c6_seed43_fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1
find artifacts/rl_scheduler/fsrs6_default_adr_portfolio_users_1_8/c6_seed43_fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

FSRS6 default ADR portfolio completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; initial population records use
  `simulation_seed = 42`; 160 SMS-EMOA generation records cover recorded
  generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = fsrs6_default_adr`,
  `feature_version = fsrs6_adr_log_poly_v1`, 16 portfolio child policies per
  user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_default_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2866.0` MiB, peak utilization `56.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `3824.0` MiB, peak utilization `56.0%`,
  `shared_memory_spill_detected = false`.

Anki SM2 AP portfolio completed run artifact root:

```text
artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/c6_seed43_anki_sm2_ap_portfolio_users_1_8_pop16_20_v1
```

Anki SM2 AP portfolio completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml --stage all --run-id c6_seed43_anki_sm2_ap_portfolio_users_1_8_pop16_20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/c6_seed43_anki_sm2_ap_portfolio_users_1_8_pop16_20_v1
find artifacts/rl_scheduler/anki_sm2_ap_portfolio_users_1_8/c6_seed43_anki_sm2_ap_portfolio_users_1_8_pop16_20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Anki SM2 AP portfolio completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files; 160 SMS-EMOA generation records cover
  recorded generations `0-19` with
  `simulation_seed = 42 + (recorded_generation + 1) * 1009`.
- artifact metadata: `seed = 42`, `scheduler_name = anki_sm2_ap`,
  `action_space = anki_sm2_ap_params_portfolio_child`, 16 portfolio child
  policies per user.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = anki_sm2_ap`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `2870.0` MiB, peak utilization `56.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `3944.0` MiB, peak utilization `58.0%`,
  `shared_memory_spill_detected = false`.

CMA-ES Cost-ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_cmaes_users_1_8/c6_seed43_fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1
```

CMA-ES Cost-ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_cmaes_users_1_8/c6_seed43_fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_cmaes_users_1_8/c6_seed43_fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

CMA-ES Cost-ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `7368.0` MiB, peak utilization `96.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `7383.0` MiB, peak utilization `59.0%`,
  `shared_memory_spill_detected = false`.

Meaninit16w Cost-ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/c6_seed43_fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1
```

Meaninit16w Cost-ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/c6_seed43_fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_meaninit16w_users_1_8/c6_seed43_fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Meaninit16w Cost-ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `6533.0` MiB, peak utilization `93.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `6543.0` MiB, peak utilization `74.0%`,
  `shared_memory_spill_detected = false`.

Distill24 DenseW Cost-ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/c6_seed43_fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1
```

Distill24 DenseW Cost-ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/c6_seed43_fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_distill24_densew_users_1_8/c6_seed43_fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

Distill24 DenseW Cost-ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 608`, `log_paths = 608`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `6464.0` MiB, peak utilization `91.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `6474.0` MiB, peak utilization `76.0%`,
  `shared_memory_spill_detected = false`.

SchedHV stdpre Cost-ADR completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1
```

SchedHV stdpre Cost-ADR completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

SchedHV stdpre Cost-ADR completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- artifact metadata: `seed = 42`, `environment = fsrs6`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `10935.0` MiB, peak utilization `89.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `10929.0` MiB, peak utilization `65.0%`,
  `shared_memory_spill_detected = false`.

RetHead 365d train-only completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_horizon_days_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1
```

RetHead 365d train-only completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_horizon_days_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_horizon_days_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead 365d train-only completed run metrics:

- stages configured: `dry-run`, `preflight`, `train-overfit`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- no `sweep` stage was configured or produced.
- train GPU monitor: peak `5666.0` MiB, peak utilization `78.0%`,
  `shared_memory_spill_detected = false`.

RetHead schedHV stdpre completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1
```

RetHead schedHV stdpre completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead schedHV stdpre completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- artifact metadata: `seed = 42`, `environment = fsrs6`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `10992.0` MiB, peak utilization `86.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `10813.0` MiB, peak utilization `64.0%`,
  `shared_memory_spill_detected = false`.

RetHead intervalinit wide completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1
```

RetHead intervalinit wide completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead intervalinit wide completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `5828.0` MiB, peak utilization `88.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `5838.0` MiB, peak utilization `68.0%`,
  `shared_memory_spill_detected = false`.

RetHead intervalinit wide nopre completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
```

RetHead intervalinit wide nopre completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead intervalinit wide nopre completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `5971.0` MiB, peak utilization `89.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `5965.0` MiB, peak utilization `63.0%`,
  `shared_memory_spill_detected = false`.

RetHead LSTM-train intervalinit wide nopre completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
```

RetHead LSTM-train intervalinit wide nopre completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_lstmtrain_intervalinit_wide_nopre_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead LSTM-train intervalinit wide nopre completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- artifact metadata: `seed = 42`, `environment = lstm`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `8949.0` MiB, peak utilization `96.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `8696.0` MiB, peak utilization `68.0%`,
  `shared_memory_spill_detected = false`.

RetHead ablate drop sqrt-z completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1
```

RetHead ablate drop sqrt-z completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead ablate drop sqrt-z completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 128`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 128`, `log_paths = 128`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environment `fsrs6`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `6018.0` MiB, peak utilization `90.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `6018.0` MiB, peak utilization `37.0%`,
  `shared_memory_spill_detected = false`.

RetHead ablate drop sqrt-z xd2 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1
```

RetHead ablate drop sqrt-z xd2 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead ablate drop sqrt-z xd2 completed run metrics:

- `stage-baseline`: `passed = true`, `staged_logs = 128`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 128`, `log_paths = 128`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environment `fsrs6`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: peak `6982.0` MiB, peak utilization `90.0%`,
  `shared_memory_spill_detected = false`.
- sweep GPU monitor: peak `6397.0` MiB, peak utilization `58.0%`,
  `shared_memory_spill_detected = false`.

RetHead ablate drop xd2 clean rerun artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1
```

RetHead ablate drop xd2 clean rerun validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead ablate drop xd2 clean rerun metrics:

- runner result: `all_summary.json` passed.
- `stage-baseline`: `passed = true`, `staged_logs = 128`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 128`, `log_paths = 128`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environment `fsrs6`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_single_adapter_bytes = 239411200`,
  `shared_memory_peak_summed_bytes = 267333632`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_single_adapter_bytes = 218710016`,
  `shared_memory_peak_summed_bytes = 246632448`.
- note: an earlier same-run-id rerun passed but appended to old progress files;
  that mixed root was archived as
  `c6_seed43_fsrs6_cost_adr_rethead_ablate_drop_xd2_users_1_8_pop16_gen20_v1_mixed_archive_20260709T1034`
  and is not counted.

RetHead ablate z2-only completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1
```

RetHead ablate z2-only completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1
find artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/c6_seed43_fsrs6_cost_adr_rethead_ablate_z2_only_users_1_8_pop16_gen20_v1/train-overfit/train_outputs -name metadata.json -print -exec uv run python experiments/rl_scheduler/validate_artifact.py --metadata {} --require-files \;
```

RetHead ablate z2-only completed run metrics:

- runner result: `all_summary.json` passed.
- `stage-baseline`: `passed = true`, `staged_logs = 128`.
- `train-overfit`: `passed = true`, `artifacts_validated = 8`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, generations `0-19`,
  `simulation_seed = 42 + generation * 1009`.
- `sweep`: `passed = true`, `batch_lanes = 128`, `log_paths = 128`.
- sweep log metadata: `seed = 43`, `scheduler = fsrs6_cost_adr`,
  `engine = batched`, users `1-8`, environment `fsrs6`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `result_files = 1`.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_single_adapter_bytes = 220348416`,
  `shared_memory_peak_summed_bytes = 248270848`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_single_adapter_bytes = 220213248`,
  `shared_memory_peak_summed_bytes = 248135680`.

Oracle stationary finite distill completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/c6_seed43_fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1
```

Oracle stationary finite distill completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8/c6_seed43_fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1
uv run python -m unittest tests.test_rl_scheduler_portfolio_workflow
```

Oracle stationary finite distill completed run metrics:

- rerun followed a wrapper fix that added the missing
  `per_user_models = false` parser default to the in-process distill args.
- independent artifact verifier reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`, elapsed `969.3s`.
- training progress: 8 progress files, initial population
  `simulation_seed = 42`, 20 SMS-EMOA generation records per user, final
  recorded generation `19`, final `simulation_seed = 20222`, matching
  `42 + (recorded_generation + 1) * 1009`.
- artifact metadata: 128 policy metadata files with `seed = 42`,
  scheduler `fsrs6_oracle_stationary_finite_distill`, and action space
  `fsrs6_oracle_stationary_finite_distill_goal_weight_portfolio_child`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`,
  `logs_validated = 256`.
- sweep log metadata: `seed = 43`, scheduler
  `fsrs6_oracle_stationary_finite_distill`, engine `batched`, users `1-8`,
  environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: peak `2828.0` MiB, peak utilization `59.0%`,
  `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 275681280`.
- sweep GPU monitor: peak `3919.0` MiB, peak utilization `81.0%`,
  `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 250437632`.

Oracle stationary finite distill r4d1/e512 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/c6_seed43_fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1
```

Oracle stationary finite distill r4d1/e512 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8/c6_seed43_fsrs6_oracle_stationary_finite_distill_r4d1_e512_portfolio_users_1_8_pop16_v1
```

Oracle stationary finite distill r4d1/e512 completed run metrics:

- independent artifact verifier reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 256`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 8`.
- training progress: 8 progress files, initial population
  `simulation_seed = 42`, 20 SMS-EMOA generation records per user, final
  recorded generation `19`, final `simulation_seed = 20222`, matching
  `42 + (recorded_generation + 1) * 1009`.
- artifact metadata: 128 policy metadata files with `seed = 42`,
  scheduler `fsrs6_oracle_stationary_finite_distill`, and action space
  `fsrs6_oracle_stationary_finite_distill_goal_weight_portfolio_child`.
- `sweep`: `passed = true`, `batch_lanes = 256`, `log_paths = 256`,
  `logs_validated = 256`.
- sweep log metadata: `seed = 43`, scheduler
  `fsrs6_oracle_stationary_finite_distill`, engine `batched`, users `1-8`,
  environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 8`,
  `plot_files = 8`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: peak `2823.0` MiB, peak utilization `59.0%`,
  `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 210108416`.
- sweep GPU monitor: peak `3994.0` MiB, peak utilization `51.0%`,
  `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 207548416`.

ADR 128-user pop16 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/c6_seed43_fsrs6_adr_portfolio_users_1_128_pop16_v1
```

ADR 128-user pop16 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_pop16_v1.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_128_pop16_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128_pop16/c6_seed43_fsrs6_adr_portfolio_users_1_128_pop16_v1
```

ADR 128-user pop16 completed run metrics:

- independent validator reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 4096`.
- `train-overfit`: `passed = true`, `artifacts_validated = 2048`,
  `commands_succeeded = 128`, elapsed `829.1s`.
- training progress: 128 progress files, initial population
  `simulation_seed = 42`, 20 SMS-EMOA generation records per user, final
  recorded generation `19`, final `simulation_seed = 20222`, matching
  `42 + (recorded_generation + 1) * 1009`.
- artifact metadata: 2048 policy metadata files with `seed = 42`.
- `sweep`: `passed = true`, `batch_lanes = 4096`,
  `logs_validated = 4096`.
- sweep log metadata: `seed = 43`, scheduler `fsrs6_adr`,
  engine `batched`, users `1-128`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 128`,
  `plot_files = 128`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 254779392`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 223666176`.

ADR 128-user v3 completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128/c6_seed43_fsrs6_adr_portfolio_users_1_128_v3
```

ADR 128-user v3 completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_128_v3.toml --stage all --run-id c6_seed43_fsrs6_adr_portfolio_users_1_128_v3
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_adr_portfolio_users_1_128/c6_seed43_fsrs6_adr_portfolio_users_1_128_v3
```

ADR 128-user v3 completed run metrics:

- independent validator reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 4096`.
- `train-overfit`: `passed = true`, `artifacts_validated = 2048`,
  `commands_succeeded = 128`.
- training progress: 128 progress files, initial population
  `simulation_seed = 42`, 20 SMS-EMOA generation records per user, final
  recorded generation `19`, final `simulation_seed = 20222`, matching
  `42 + (recorded_generation + 1) * 1009`.
- artifact metadata: 2048 policy metadata files with `seed = 42`,
  scheduler `fsrs6_adr`, and action space
  `sd_retention_function_portfolio_child`.
- `sweep`: `passed = true`, `batch_lanes = 4096`,
  `logs_validated = 4096`.
- sweep log metadata: `seed = 43`, scheduler `fsrs6_adr`,
  engine `batched`, users `1-128`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 128`,
  `plot_files = 128`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 242896896`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 210735104`.

AP 128-user completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_128/c6_seed43_fsrs6_ap_portfolio_users_1_128_v1
```

AP 128-user completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_ap_portfolio_users_1_128_v1.toml --stage all --run-id c6_seed43_fsrs6_ap_portfolio_users_1_128_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_ap_portfolio_users_1_128/c6_seed43_fsrs6_ap_portfolio_users_1_128_v1
```

AP 128-user completed run metrics:

- independent validator reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 4096`.
- `train-overfit`: `passed = true`, `artifacts_validated = 2048`,
  `commands_succeeded = 128`.
- training progress: 128 progress files, initial population
  `simulation_seed = 42`, 20 SMS-EMOA generation records per user, final
  recorded generation `19`, final `simulation_seed = 20222`, matching
  `42 + (recorded_generation + 1) * 1009`.
- artifact metadata: 2048 policy metadata files with `seed = 42`,
  scheduler `fsrs6_ap`, and action space
  `fsrs6_ap_weight_delta_portfolio_child`.
- `sweep`: `passed = true`, `batch_lanes = 4096`,
  `logs_validated = 4096`.
- sweep log metadata: `seed = 43`, scheduler `fsrs6_ap`,
  engine `batched`, users `1-128`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 128`,
  `plot_files = 128`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 210915328`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 210751488`.

Cost-ADR schedHV stdpre 128-user completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1
```

Cost-ADR schedHV stdpre 128-user completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_schedhv_stdpre_users_1_128/c6_seed43_fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1
```

Cost-ADR schedHV stdpre 128-user completed run metrics:

- independent validator reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 4096`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 128`, elapsed `10057.8s`.
- training progress: 128 progress files, 20 CMA-ES generation records per
  user, final generation `19`, final `simulation_seed = 19213`, matching
  `42 + generation * 1009`.
- artifact metadata: 128 policy metadata files with `seed = 42`,
  scheduler `fsrs6_cost_adr`, and action space `sd_cost_interval_function`.
- `sweep`: `passed = true`, `batch_lanes = 4096`,
  `logs_validated = 4096`.
- sweep log metadata: `seed = 43`, scheduler `fsrs6_cost_adr`,
  engine `batched`, users `1-128`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 128`,
  `plot_files = 128`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 266801152`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 210161664`.

Cost-ADR RetHead interval-init wide nopre 128-user completed run artifact root:

```text
artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1
```

Cost-ADR RetHead interval-init wide nopre 128-user completed run validation commands:

```bash
MPLCONFIGDIR=/home/jarrett/.config/matplotlib uv run python experiments/rl_scheduler/run_experiment.py --config experiments/rl_scheduler/configs/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1.toml --stage all --run-id c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1
uv run python experiments/rl_scheduler/inspect_run.py --run-root artifacts/rl_scheduler/fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128/c6_seed43_fsrs6_cost_adr_rethead_intervalinit_wide_nopre_users_1_128_pop16_gen20_v1
```

Cost-ADR RetHead interval-init wide nopre 128-user completed run metrics:

- independent validator reported `errors = 0`.
- `all_summary.json`: `passed = true`, `exit_code = 0`, `stopped_at = null`.
- `stage-baseline`: `passed = true`, `staged_logs = 4096`.
- `train-overfit`: `passed = true`, `artifacts_validated = 128`,
  `commands_succeeded = 128`.
- training progress: 128 progress files, 20 CMA-ES generation records per
  user, final generation `19`, final `simulation_seed = 19213`, matching
  `42 + generation * 1009`.
- artifact metadata: 128 policy metadata files with `seed = 42`,
  scheduler `fsrs6_cost_adr`, and action space `sd_cost_retention_function`.
- `sweep`: `passed = true`, `batch_lanes = 4096`,
  `logs_validated = 4096`.
- sweep log metadata: `seed = 43`, scheduler `fsrs6_cost_adr`,
  engine `batched`, users `1-128`, environments `fsrs6,lstm`.
- `build-pareto`: `passed = true`, `result_json_files = 128`,
  `plot_files = 128`.
- `analyze-pareto`: `passed = true`, `analysis_summary.json` produced.
- train GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 267591680`.
- sweep GPU monitor: `shared_memory_spill_detected = false`,
  `shared_memory_peak_summed_bytes = 267915264`.

## Rerun Matrix

Audit state:

- C5 seed-43 baseline logs are present under `logs/retention_sweep` for the
  manifest users 1-8 baseline, manifest users 1-128 baseline, and ordinary
  users 1-8 grid baseline.
- C6 runner scan after the RetHead structure-ablation reruns:
  `runner_passed = 40`, `runner_failed = 1`, `missing = 0`, `total = 41`.
- Effective C6 status: all configs attempted; 40 runner-passing runs completed
  with seed-43 evaluation where configured; the only runner failure is
  `fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1`, a documented quality-v1
  objective/search failure superseded by `quality_v2` and the dependent hybrid
  selector.
- C6 closure checks passed: targeted `uv run ruff format` on the touched
  training entrypoints left files unchanged; focused `unittest` passed 68
  tests; `uv run pyright` reported 0 errors; `git diff --check` passed; no
  experiment process remained after the final scan.

Full retrain plus seed-43 evaluation:

- `anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml`
- `fsrs6_adr_*`
- `fsrs6_ap_*`
- `fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml`
- `fsrs6_cost_adr_coverage_*`
- `fsrs6_cost_adr_quality_*`
- `fsrs6_cost_adr_quality_v2_users_1_8_pop16_gen20_v1.toml`
- `fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml`
- `fsrs6_cost_adr_schedhv_stdpre_*`
- `fsrs6_cost_adr_rethead_*` profiles with sweep stages
- `fsrs6_cost_adr_distill24_densew_users_1_8_pop16_gen20_v1.toml`
- `fsrs6_default_adr_portfolio_users_1_8_pop16_20_v1.toml`
- `fsrs6_oracle_stationary_finite_distill_*`

Train-only rerun:

- `fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1.toml`

Seed-43 evaluation only:

- `fsrs3_scheduler_users_1_8.toml`
- `fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1.toml`

Hybrid dependency:

- Rebuild `fsrs6_cost_adr_quality_hybrid_users_1_8_pop16_gen20_v1` after its
  source policies, especially coverage and quality-v2, are retrained under this
  seed policy.

Known superseded failure:

- `fsrs6_cost_adr_quality_users_1_8_pop16_gen20_v1.toml` failed its
  `train-overfit` gate before sweep: 6 of 8 commands succeeded, and failed
  users were 5 and 8. Both failed users completed all generations with
  `simulation_seed = 42 + generation * 1009` but had
  `best_hypervolume_delta = 0.0`. Training GPU monitor reported no
  shared-memory spill. The existing quality-v2 recipe restored training pass
  for all users, and the hybrid selector depends on the C6 coverage and
  quality-v2 source runs, so no further quality-v1 rerun is planned.

C6 residual queue:

- No missing C6 configs remain.
- No GPU spill stop-condition failures remain after the clean RetHead
  structure-ablation reruns.
- Keep the quality-v1 failure as a superseded failed datapoint unless future
  work intentionally revives the quality-v1 objective.

Unaffected formal profiles:

- None identified in the current checked-in formal config set.

## Stop Conditions

Stop a rerun family if:

- seed-43 baselines are missing or fail metadata validation;
- training progress lacks `simulation_seed`;
- artifact metadata seed is not 42;
- formal evaluation logs are not seed 43;
- GPU monitor artifacts show VRAM spill/shared-memory spill beyond the accepted
  lane caps.
