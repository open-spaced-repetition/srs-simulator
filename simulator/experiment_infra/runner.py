from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simulator.experiment_infra.artifacts import (
    SchedulerArtifactMetadata,
    validate_scheduler_artifact,
)
from simulator.experiment_infra.schemas import (
    ArtifactManifest,
    CommandRecord,
    ExperimentConfig,
    FailureClass,
    GateSummary,
    GpuGuardSummary,
    PerformanceSummary,
    RunRecord,
    StageName,
)
from simulator.experiment_infra.baseline_dr_selection import (
    BaselineDRManifest,
    load_baseline_dr_manifest,
)
from simulator.experiment_infra.gpu_monitor import GpuMonitor, GpuMonitorSummary
from simulator.retention_sweep.log_filter import LogFilenameFilter
from simulator.batched_engine.mixed_scheduler import (
    MixedBatchSchedulerOps as _MixedBatchSchedulerOps,
    MixedSchedulerGroup as _MixedSchedulerGroup,
)
from simulator.scheduler_catalog import (
    PolicySource,
    action_space_allows_lambda_none,
    run_id_scoped_sweep_schedulers,
    schedulers_for_policy_source,
)


SUPPORTED_RUNNER_STAGES = {
    StageName.DRY_RUN,
    StageName.PREFLIGHT,
    StageName.STAGE_BASELINE,
    StageName.TRAIN_OVERFIT,
    StageName.SWEEP,
    StageName.BUILD_PARETO,
    StageName.ANALYZE_PARETO,
    StageName.PARETO,
    StageName.SELECT,
    StageName.AGGREGATE,
    StageName.RESERVED_TEST,
}

COMMAND_TIMEOUT_EXIT_CODE = 124
TRAIN_OVERFIT_GATE_PASS_FRACTION = 0.8
RUN_ID_SCOPED_SWEEP_SCHEDULERS = run_id_scoped_sweep_schedulers()
FSRS6_ADR_POLICY_SOURCE_SCHEDULERS = schedulers_for_policy_source(
    PolicySource.FSRS6_ADR
)
FSRS6_COST_ADR_POLICY_SOURCE_SCHEDULERS = schedulers_for_policy_source(
    PolicySource.FSRS6_COST_ADR
)


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    exit_code: int
    stage: StageName
    run_id: str
    stage_root: Path | None
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainCommandJob:
    user_id: int
    baseline_desired_retention: float
    baseline_desired_retention_token: str
    lambda_value: float | None
    lambda_token: str | None
    output_dir: Path
    command_record_path: Path
    stdout_path: Path
    stderr_path: Path
    command: list[str]


@dataclass(frozen=True, slots=True)
class SweepBatchLane:
    source: str
    metadata_path: Path | None
    metadata: SchedulerArtifactMetadata | None
    user_id: int
    scheduler_name: str
    scheduler_spec: str
    desired_retention: float | None
    fixed_interval: float | None
    fsrs6_adr_policy_path: Path | None
    lambda_value: float | None
    lambda_token: str | None
    baseline_desired_retention: float | None
    baseline_desired_retention_token: str | None
    output_dir: Path


@dataclass(frozen=True, slots=True)
class AllExecutionResult:
    exit_code: int
    run_id: str
    all_root: Path
    summary: dict[str, Any]


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def default_run_id(config: ExperimentConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.name}_{stamp}"


def stage_command(
    *,
    config_path: Path,
    stage: StageName,
    run_id: str | None = None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "experiments/rl_scheduler/run_experiment.py",
        "--config",
        str(config_path),
        "--stage",
        stage.value,
    ]
    if run_id is not None:
        command.extend(["--run-id", run_id])
    return command


def build_dry_run_preview(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
) -> dict[str, Any]:
    return {
        "type": "dry-run",
        "run_id": run_id,
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "resolved_config": config.to_dict(),
        "supported_stages": sorted(stage.value for stage in SUPPORTED_RUNNER_STAGES),
        "planned_stages": [
            {
                "stage": stage.value,
                "supported": stage in SUPPORTED_RUNNER_STAGES,
                "command": stage_command(
                    config_path=config_path,
                    stage=stage,
                    run_id=run_id,
                ),
                "writes_formal_outputs": stage != StageName.DRY_RUN
                and stage in SUPPORTED_RUNNER_STAGES,
            }
            for stage in config.stages
        ],
    }


def run_stage(
    *,
    config_path: Path,
    stage: StageName,
    repo_root: Path,
    run_id: str | None = None,
    command: list[str] | None = None,
) -> StageExecutionResult:
    config = ExperimentConfig.from_toml(config_path)
    actual_run_id = run_id or default_run_id(config)
    if stage == StageName.DRY_RUN:
        return StageExecutionResult(
            exit_code=0,
            stage=stage,
            run_id=actual_run_id,
            stage_root=None,
            summary=build_dry_run_preview(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=actual_run_id,
            ),
        )
    if stage == StageName.PREFLIGHT:
        return run_preflight(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.STAGE_BASELINE:
        return run_stage_baseline(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.TRAIN_OVERFIT:
        return run_train_overfit(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.SWEEP:
        return run_sweep(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.BUILD_PARETO:
        return run_build_pareto(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.ANALYZE_PARETO:
        return run_analyze_pareto(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.PARETO:
        return run_pareto(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.SELECT:
        return run_select(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.AGGREGATE:
        return run_aggregate(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    if stage == StageName.RESERVED_TEST:
        return run_reserved_test(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=command
            or stage_command(
                config_path=config_path, stage=stage, run_id=actual_run_id
            ),
        )
    return StageExecutionResult(
        exit_code=2,
        stage=stage,
        run_id=actual_run_id,
        stage_root=None,
        summary={
            "type": "unsupported-stage",
            "stage": stage.value,
            "supported_stages": sorted(
                stage.value for stage in SUPPORTED_RUNNER_STAGES
            ),
        },
    )


def run_all(
    *,
    config_path: Path,
    repo_root: Path,
    run_id: str | None = None,
) -> AllExecutionResult:
    config = ExperimentConfig.from_toml(config_path)
    actual_run_id = run_id or default_run_id(config)
    output_root = _resolve_repo_path(repo_root, config.output_root)
    all_root = output_root / actual_run_id / "all"
    all_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    exit_code = 0
    stopped_at: str | None = None
    for stage in config.stages:
        result = run_stage(
            config_path=config_path,
            stage=stage,
            repo_root=repo_root,
            run_id=actual_run_id,
            command=stage_command(
                config_path=config_path,
                stage=stage,
                run_id=actual_run_id,
            ),
        )
        stage_summary = {
            "stage": stage.value,
            "exit_code": result.exit_code,
            "stage_root": str(result.stage_root) if result.stage_root else None,
            "summary_type": result.summary.get("type"),
        }
        results.append(stage_summary)
        if result.exit_code != 0:
            exit_code = result.exit_code
            stopped_at = stage.value
            break

    summary = {
        "type": "all",
        "run_id": actual_run_id,
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "stopped_at": stopped_at,
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "all_root": str(all_root),
        "stage_results": results,
    }
    _write_json(all_root / "all_summary.json", summary)
    return AllExecutionResult(
        exit_code=exit_code,
        run_id=actual_run_id,
        all_root=all_root,
        summary=summary,
    )


def run_train_overfit(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    perf_started = time.monotonic()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.TRAIN_OVERFIT.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    training_summary_path = stage_root / "training_summary.json"
    performance_summary_path = stage_root / "performance_summary.json"
    manifest_path = stage_root / "manifest.json"
    commands_root = stage_root / "commands"
    outputs_root = stage_root / "train_outputs"
    gpu_monitor = _maybe_start_stage_gpu_monitor(config=config, stage_root=stage_root)

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    progress_paths: list[Path] = []
    artifact_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    commands_attempted = 0
    commands_succeeded = 0
    runtime_batch_metrics: dict[str, Any] = {}
    baseline_dr_values = _training_baseline_desired_retention_values(config)

    if not config.train_command_template and not config.training_batch.enabled:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("training.command_template is required for train-overfit.")
    elif (
        config.training_batch.enabled
        and config.training_batch.trainer == "auto"
        and not config.train_command_template
    ):
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append(
            "training.command_template is required for training.batch.trainer = 'auto'."
        )
    elif not baseline_dr_values:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append(
            "training.policy_search.baseline_desired_retention_values must contain numbers "
            "without duplicates."
        )
    else:
        jobs, job_notes = _build_train_command_jobs(
            config=config,
            config_path=config_path,
            repo_root=repo_root,
            run_id=run_id,
            stage_root=stage_root,
            outputs_root=outputs_root,
            commands_root=commands_root,
            baseline_dr_values=baseline_dr_values,
        )
        if job_notes:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.extend(job_notes)
        else:
            job_results: list[dict[str, Any]] = []
            batch_runs_attempted = 0
            batch_runs_succeeded = 0
            max_effective_lanes = 0
            resolved_batch_trainer: str | None = None
            if config.training_batch.enabled:
                (
                    job_results,
                    batch_runs_attempted,
                    batch_runs_succeeded,
                    max_effective_lanes,
                    resolved_batch_trainer,
                    batch_notes,
                ) = _run_train_in_process_batches(
                    jobs=jobs,
                    config=config,
                    config_path=config_path,
                    repo_root=repo_root,
                    commands_root=commands_root,
                )
                if batch_notes:
                    notes.extend(batch_notes)
                    if not job_results:
                        failures.append(FailureClass.INVALID_CONFIG)
            else:
                max_parallel = min(
                    config.train_max_parallel_commands, max(len(jobs), 1)
                )
                if max_parallel <= 1:
                    for job in jobs:
                        result = _run_train_command_job(
                            job=job,
                            config=config,
                            repo_root=repo_root,
                        )
                        job_results.append(result)
                        if result["failure"] is not None:
                            break
                else:
                    ordered_results: list[dict[str, Any] | None] = [None] * len(jobs)
                    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                        future_to_index = {
                            executor.submit(
                                _run_train_command_job,
                                job=job,
                                config=config,
                                repo_root=repo_root,
                            ): index
                            for index, job in enumerate(jobs)
                        }
                        for future in as_completed(future_to_index):
                            ordered_results[future_to_index[future]] = future.result()
                    job_results = [result for result in ordered_results if result]

            commands_attempted = len(job_results)
            for result in job_results:
                command_records.append(result["command_record_path"])
                if result["stdout_path"] is not None:
                    stdout_paths.append(result["stdout_path"])
                if result["stderr_path"] is not None:
                    stderr_paths.append(result["stderr_path"])
                if result["progress_path"] is not None:
                    progress_paths.append(result["progress_path"])
                command_results.append(result["command_result"])
                failure = result["failure"]
                if failure is not None:
                    failures.append(failure)
                    note = result["note"]
                    if note is not None:
                        notes.append(note)
                    continue
                commands_succeeded += 1
                artifact_paths.extend(result["artifact_paths"])

            if config.training_batch.enabled:
                runtime_batch_metrics = {
                    "batch_runs_attempted": batch_runs_attempted,
                    "batch_runs_succeeded": batch_runs_succeeded,
                    "max_effective_lanes_per_simulation": max_effective_lanes,
                    "resolved_batch_trainer": resolved_batch_trainer,
                }
            else:
                runtime_batch_metrics = {}
    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gpu_monitor_summary = _stop_stage_gpu_monitor(gpu_monitor)
    gate_summary = GateSummary(
        gate_name=StageName.TRAIN_OVERFIT.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "training_users": float(len(config.users.train)),
            "lambda_values": float(_training_reported_lambda_count(config)),
            "baseline_desired_retention_values": float(len(baseline_dr_values)),
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "artifacts_validated": float(len(artifact_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())
    performance_summary = _build_stage_performance_summary(
        config=config,
        stage=StageName.TRAIN_OVERFIT,
        passed=passed,
        elapsed_seconds=time.monotonic() - perf_started,
        stage_root=stage_root,
        failures=unique_failures,
        notes=notes,
        runtime_metrics={
            "commands_attempted": commands_attempted,
            "commands_succeeded": commands_succeeded,
            "artifacts_validated": len(artifact_paths),
            **runtime_batch_metrics,
        },
        execution_shape={
            "process_count": 1,
            "subprocess_count": 0
            if config.training_batch.enabled
            else commands_attempted,
            "max_parallel_commands": config.train_max_parallel_commands,
            "training_batch": config.training_batch.to_dict(),
            "timeout_seconds": config.performance.timeout_seconds,
            **runtime_batch_metrics,
        },
        gpu_monitor_summary=gpu_monitor_summary,
    )
    if config.performance.write_performance_summary:
        _write_json(performance_summary_path, performance_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "train-overfit",
        "run_id": run_id,
        "stage": StageName.TRAIN_OVERFIT.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "commands_root": str(commands_root),
        "outputs_root": str(outputs_root),
        "command_template": list(config.train_command_template),
        "training_batch": config.training_batch.to_dict(),
        "artifact_metadata_glob": config.train_artifact_glob,
        "baseline_desired_retention_values": list(baseline_dr_values),
        "command_results": command_results,
        "artifact_paths": [str(path) for path in artifact_paths],
        "training_progress_paths": [str(path) for path in progress_paths],
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "performance_summary_path": str(performance_summary_path)
        if config.performance.write_performance_summary
        else None,
        "gpu_monitor_summary_path": str(gpu_monitor_summary.summary_path)
        if gpu_monitor_summary is not None
        else None,
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(training_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.TRAIN_OVERFIT,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            training_summary_path,
            *(
                (performance_summary_path,)
                if config.performance.write_performance_summary
                else ()
            ),
            *(
                (gpu_monitor_summary.jsonl_path, gpu_monitor_summary.summary_path)
                if gpu_monitor_summary is not None
                else ()
            ),
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *progress_paths,
            *artifact_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "training_summary": training_summary_path,
    }
    if config.performance.write_performance_summary:
        manifest_artifacts["performance_summary"] = performance_summary_path
    if gpu_monitor_summary is not None:
        manifest_artifacts["gpu_monitor_jsonl"] = gpu_monitor_summary.jsonl_path
        manifest_artifacts["gpu_monitor_summary"] = gpu_monitor_summary.summary_path
    manifest_artifacts.update(
        {
            f"training_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {f"training_stdout_{index}": path for index, path in enumerate(stdout_paths)}
    )
    manifest_artifacts.update(
        {f"training_stderr_{index}": path for index, path in enumerate(stderr_paths)}
    )
    manifest_artifacts.update(
        {
            f"training_progress_{index}": path
            for index, path in enumerate(progress_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"scheduler_artifact_metadata_{index}": path
            for index, path in enumerate(artifact_paths)
        }
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.TRAIN_OVERFIT,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_sweep(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    perf_started = time.monotonic()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.SWEEP.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    sweep_summary_path = stage_root / "sweep_summary.json"
    performance_summary_path = stage_root / "performance_summary.json"
    manifest_path = stage_root / "manifest.json"
    commands_root = stage_root / "commands"
    outputs_root = stage_root / "sweep_outputs"
    batched_sweep_record_path = commands_root / "batched_sweep_record.json"
    train_summary_path = (
        output_root / run_id / StageName.TRAIN_OVERFIT.value / "training_summary.json"
    )
    gpu_monitor = _maybe_start_stage_gpu_monitor(config=config, stage_root=stage_root)

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    log_paths: list[Path] = []
    artifact_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    commands_attempted = 0
    commands_succeeded = 0
    batch_runs_attempted = 0
    batch_runs_succeeded = 0
    batch_lane_count = 0
    batched_retention_sweep = bool(
        config.sweep_batched.envs and config.sweep_batched.schedulers
    )

    if (
        not batched_retention_sweep
        and not config.sweep_batch_scheduler_artifacts
        and not config.sweep_command_template
    ):
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("sweep.command_template is required for sweep.")

    train_artifact_paths: list[Path] = []
    requires_train_artifacts = _sweep_requires_train_artifacts(
        config=config,
        batched_retention_sweep=batched_retention_sweep,
    )
    if not failures and requires_train_artifacts:
        train_artifact_paths, artifact_notes = _read_train_artifact_paths(
            train_summary_path
        )
        if artifact_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(artifact_notes)

    if not failures and batched_retention_sweep:
        batch_runs_attempted = 1
        try:
            batched_result = _run_configured_batched_retention_sweep(
                config=config,
                repo_root=repo_root,
                run_id=run_id,
                output_root=output_root,
                log_dir=outputs_root,
                record_path=batched_sweep_record_path,
            )
        except Exception as exc:
            failures.append(FailureClass.RUNNER_FAILED)
            notes.append(f"Batched retention sweep failed: {exc}")
        else:
            batch_runs_succeeded = 1
            batch_lane_count = batched_result["batch_lane_count"]
            log_paths.extend(batched_result["log_paths"])
            command_results.extend(batched_result["lane_results"])

    if (
        not failures
        and config.sweep_batch_scheduler_artifacts
        and not batched_retention_sweep
    ):
        artifact_lanes: list[SweepBatchLane] = []
        for metadata_path in train_artifact_paths:
            artifact_paths.append(metadata_path)
            try:
                metadata = validate_scheduler_artifact(
                    metadata_path, require_files=True
                )
            except ValueError as exc:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    f"Invalid scheduler artifact metadata {metadata_path}: {exc}"
                )
                break
            artifact_note = _validate_sweep_artifact_metadata(
                metadata_path=metadata_path,
                metadata=metadata,
                config=config,
            )
            if artifact_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(artifact_note)
                break

            user_id = metadata.training_user_ids[0]
            artifact_lanes.append(
                _build_sweep_artifact_lane(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    user_id=user_id,
                    outputs_root=outputs_root,
                )
            )

        if not failures:
            jobs = [
                *_build_sweep_baseline_lanes(
                    config=config,
                    repo_root=repo_root,
                    outputs_root=outputs_root,
                ),
                *artifact_lanes,
            ]
            batch_lane_count = len(jobs)
            batch_runs_attempted = 1
            try:
                _run_batched_sweep_jobs(
                    config=config,
                    repo_root=repo_root,
                    run_id=run_id,
                    jobs=jobs,
                    record_path=batched_sweep_record_path,
                )
            except Exception as exc:
                failures.append(FailureClass.RUNNER_FAILED)
                notes.append(f"Batched sweep failed: {exc}")
            else:
                batch_runs_succeeded = 1
                for job in jobs:
                    matched_logs, log_note = _collect_sweep_logs(
                        output_dir=job.output_dir,
                        log_glob=config.sweep_log_glob,
                    )
                    if log_note is not None:
                        failures.append(FailureClass.INVALID_CONFIG)
                        notes.append(log_note)
                        break
                    if not matched_logs:
                        failures.append(FailureClass.INCOMPLETE_OUTPUT)
                        notes.append(
                            f"No sweep JSONL logs matched {config.sweep_log_glob!r} "
                            f"in {job.output_dir}."
                        )
                        break
                    log_note = _validate_sweep_logs(
                        log_paths=matched_logs,
                        config=config,
                        job=job,
                    )
                    if log_note is not None:
                        failures.append(FailureClass.INVALID_ARTIFACT)
                        notes.append(log_note)
                        break
                    log_paths.extend(matched_logs)
                    command_results.append(_sweep_batch_lane_result(job))

    if (
        not failures
        and not config.sweep_batch_scheduler_artifacts
        and not batched_retention_sweep
    ):
        for metadata_path in train_artifact_paths:
            artifact_paths.append(metadata_path)
            try:
                metadata = validate_scheduler_artifact(
                    metadata_path, require_files=True
                )
            except ValueError as exc:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    f"Invalid scheduler artifact metadata {metadata_path}: {exc}"
                )
                break
            artifact_note = _validate_sweep_artifact_metadata(
                metadata_path=metadata_path,
                metadata=metadata,
                config=config,
            )
            if artifact_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(artifact_note)
                break

            user_id = metadata.training_user_ids[0]
            allows_lambda_none = action_space_allows_lambda_none(metadata.action_space)
            if metadata.lambda_value is None and not allows_lambda_none:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    f"Invalid scheduler artifact metadata {metadata_path}: "
                    "lambda_value is required for sweep."
                )
                break
            lambda_value = metadata.lambda_value
            lambda_token = (
                _format_lambda_token(lambda_value) if lambda_value is not None else None
            )
            baseline_dr = metadata.baseline_desired_retention
            if baseline_dr is not None:
                assert lambda_token is not None
                baseline_dr_token = _format_retention_token(baseline_dr)
                output_dir = (
                    outputs_root
                    / f"user_{user_id}"
                    / f"dr_{baseline_dr_token}"
                    / f"lambda_{lambda_token}"
                )
                command_stem = (
                    f"user_{user_id}_dr_{baseline_dr_token}_lambda_{lambda_token}"
                )
            else:
                baseline_dr_token = None
                if allows_lambda_none:
                    policy_token = metadata.policy_path.parent.name
                    output_dir = outputs_root / f"user_{user_id}" / policy_token
                    command_stem = f"user_{user_id}_{policy_token}"
                else:
                    assert lambda_token is not None
                    output_dir = (
                        outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
                    )
                    command_stem = f"user_{user_id}_lambda_{lambda_token}"
            command_record = commands_root / f"{command_stem}_command.json"
            stdout_path = commands_root / f"{command_stem}_stdout.txt"
            stderr_path = commands_root / f"{command_stem}_stderr.txt"
            try:
                sweep_command = _format_sweep_command(
                    config=config,
                    config_path=config_path,
                    repo_root=repo_root,
                    run_id=run_id,
                    stage_root=stage_root,
                    output_dir=output_dir,
                    metadata_path=metadata_path,
                    metadata=metadata,
                    command_record_path=command_record,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            except ValueError as exc:
                failures.append(FailureClass.INVALID_CONFIG)
                notes.append(
                    "Invalid sweep.command_template for "
                    f"artifact={metadata_path}: {exc}"
                )
                break

            output_dir.mkdir(parents=True, exist_ok=True)
            commands_attempted += 1
            sweep_command_record = _run_recorded_command(
                command=sweep_command,
                cwd=repo_root,
                command_record_path=command_record,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(sweep_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(command_record)
            stdout_paths.append(stdout_path)
            stderr_paths.append(stderr_path)
            command_results.append(
                {
                    "artifact_metadata_path": str(metadata_path),
                    "artifact_id": metadata.artifact_id,
                    "user_id": user_id,
                    "baseline_desired_retention": baseline_dr,
                    "baseline_desired_retention_token": baseline_dr_token,
                    "lambda_value": lambda_value,
                    "lambda_token": lambda_token,
                    "output_dir": str(output_dir),
                    "command_record_path": str(command_record),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    f"Sweep command timed out for artifact={metadata_path}."
                    if timed_out
                    else f"Sweep command failed for artifact={metadata_path}."
                )
                break

            commands_succeeded += 1
            matched_logs, log_note = _collect_sweep_logs(
                output_dir=output_dir,
                log_glob=config.sweep_log_glob,
            )
            if log_note is not None:
                failures.append(FailureClass.INVALID_CONFIG)
                notes.append(log_note)
                break
            if not matched_logs:
                failures.append(FailureClass.INCOMPLETE_OUTPUT)
                notes.append(
                    f"No sweep JSONL logs matched {config.sweep_log_glob!r} "
                    f"in {output_dir}."
                )
                break
            log_note = _validate_sweep_logs(
                log_paths=matched_logs,
                config=config,
                job=_build_sweep_artifact_lane(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    user_id=user_id,
                    outputs_root=outputs_root,
                ),
            )
            if log_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(log_note)
                break
            log_paths.extend(matched_logs)

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gpu_monitor_summary = _stop_stage_gpu_monitor(gpu_monitor)
    gate_summary = GateSummary(
        gate_name=StageName.SWEEP.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "input_artifacts": float(len(train_artifact_paths)),
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "batch_runs_attempted": float(batch_runs_attempted),
            "batch_runs_succeeded": float(batch_runs_succeeded),
            "batch_lanes": float(batch_lane_count),
            "logs_validated": float(len(log_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())
    performance_summary = _build_stage_performance_summary(
        config=config,
        stage=StageName.SWEEP,
        passed=passed,
        elapsed_seconds=time.monotonic() - perf_started,
        stage_root=stage_root,
        failures=unique_failures,
        notes=notes,
        runtime_metrics={
            "input_artifacts": len(train_artifact_paths),
            "commands_attempted": commands_attempted,
            "commands_succeeded": commands_succeeded,
            "batch_runs_attempted": batch_runs_attempted,
            "batch_runs_succeeded": batch_runs_succeeded,
            "batch_lanes": batch_lane_count,
            "logs_validated": len(log_paths),
        },
        execution_shape={
            "process_count": 1,
            "subprocess_count": commands_attempted,
            "batch_scheduler_artifacts": config.sweep_batch_scheduler_artifacts,
            "batch_lane_count": batch_lane_count
            if config.sweep_batch_scheduler_artifacts
            else 0,
            "timeout_seconds": config.performance.timeout_seconds,
        },
        gpu_monitor_summary=gpu_monitor_summary,
    )
    if config.performance.write_performance_summary:
        _write_json(performance_summary_path, performance_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "sweep",
        "run_id": run_id,
        "stage": StageName.SWEEP.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "commands_root": str(commands_root),
        "outputs_root": str(outputs_root),
        "train_summary_path": str(train_summary_path),
        "command_template": list(config.sweep_command_template),
        "log_glob": config.sweep_log_glob,
        "batch_scheduler_artifacts": config.sweep_batch_scheduler_artifacts,
        "batched_retention_sweep": batched_retention_sweep,
        "batched_sweep_config": config.sweep_batched.to_dict(),
        "batch_runs_attempted": batch_runs_attempted,
        "batch_runs_succeeded": batch_runs_succeeded,
        "batch_lanes": batch_lane_count,
        "input_artifact_paths": [str(path) for path in artifact_paths],
        "log_paths": [str(path) for path in log_paths],
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "performance_summary_path": str(performance_summary_path)
        if config.performance.write_performance_summary
        else None,
        "gpu_monitor_summary_path": str(gpu_monitor_summary.summary_path)
        if gpu_monitor_summary is not None
        else None,
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(sweep_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.SWEEP,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            sweep_summary_path,
            *(
                (performance_summary_path,)
                if config.performance.write_performance_summary
                else ()
            ),
            *(
                (gpu_monitor_summary.jsonl_path, gpu_monitor_summary.summary_path)
                if gpu_monitor_summary is not None
                else ()
            ),
            manifest_path,
            *artifact_paths,
            *command_records,
            *(
                (batched_sweep_record_path,)
                if config.sweep_batch_scheduler_artifacts
                and batched_sweep_record_path.exists()
                else ()
            ),
            *stdout_paths,
            *stderr_paths,
            *log_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "sweep_summary": sweep_summary_path,
    }
    if config.performance.write_performance_summary:
        manifest_artifacts["performance_summary"] = performance_summary_path
    if gpu_monitor_summary is not None:
        manifest_artifacts["gpu_monitor_jsonl"] = gpu_monitor_summary.jsonl_path
        manifest_artifacts["gpu_monitor_summary"] = gpu_monitor_summary.summary_path
    manifest_artifacts.update(
        {
            f"scheduler_artifact_metadata_{index}": path
            for index, path in enumerate(artifact_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"sweep_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    if config.sweep_batch_scheduler_artifacts and batched_sweep_record_path.exists():
        manifest_artifacts["batched_sweep_record"] = batched_sweep_record_path
    manifest_artifacts.update(
        {f"sweep_stdout_{index}": path for index, path in enumerate(stdout_paths)}
    )
    manifest_artifacts.update(
        {f"sweep_stderr_{index}": path for index, path in enumerate(stderr_paths)}
    )
    manifest_artifacts.update(
        {f"sweep_log_{index}": path for index, path in enumerate(log_paths)}
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.SWEEP,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_build_pareto(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.BUILD_PARETO.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    build_command_record_path = stage_root / "commands" / "build_pareto_command.json"
    build_stdout_path = stage_root / "commands" / "build_pareto_stdout.txt"
    build_stderr_path = stage_root / "commands" / "build_pareto_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    build_summary_path = stage_root / "build_pareto_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "build_pareto_outputs"
    run_root = output_root / run_id
    baseline_stage_root = run_root / StageName.STAGE_BASELINE.value
    sweep_stage_root = run_root / StageName.SWEEP.value

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    command_results: list[dict[str, Any]] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    result_paths: list[Path] = []
    plot_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0

    for summary_path, stage_name in (
        (baseline_stage_root / "baseline_summary.json", StageName.STAGE_BASELINE),
        (sweep_stage_root / "sweep_summary.json", StageName.SWEEP),
    ):
        summary_notes = _read_passed_stage_summary(summary_path, stage_name)
        if summary_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(summary_notes)

    if not failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            build_command = _format_build_pareto_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                run_root=run_root,
                baseline_stage_root=baseline_stage_root,
                sweep_stage_root=sweep_stage_root,
                command_record_path=build_command_record_path,
                stdout_path=build_stdout_path,
                stderr_path=build_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid build_pareto.command_template: {exc}")
        else:
            commands_attempted = 1
            build_command_record = _run_recorded_command(
                command=build_command,
                cwd=repo_root,
                command_record_path=build_command_record_path,
                stdout_path=build_stdout_path,
                stderr_path=build_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(build_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(build_command_record_path)
            stdout_paths.append(build_stdout_path)
            stderr_paths.append(build_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "command_record_path": str(build_command_record_path),
                    "stdout_path": str(build_stdout_path),
                    "stderr_path": str(build_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Build-Pareto command timed out."
                    if timed_out
                    else "Build-Pareto command failed."
                )
            else:
                commands_succeeded = 1

    if not failures:
        result_paths, result_note = _collect_paths(
            root=output_dir,
            path_glob=config.build_pareto.result_glob,
            field_name="build_pareto.result_glob",
        )
        if result_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(result_note)
        elif not result_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                "No Build-Pareto result JSON matched "
                f"{config.build_pareto.result_glob!r} in {output_dir}."
            )
        else:
            json_note = _validate_json_files(result_paths)
            if json_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(json_note)

    if not failures and not config.build_pareto.no_plot:
        plot_paths, plot_note = _collect_paths(
            root=output_dir,
            path_glob=config.build_pareto.plot_glob,
            field_name="build_pareto.plot_glob",
        )
        if plot_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(plot_note)
        elif not plot_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                f"No Build-Pareto plot matched {config.build_pareto.plot_glob!r} "
                f"in {output_dir}."
            )

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.BUILD_PARETO.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "result_json_files": float(len(result_paths)),
            "plot_files": float(len(plot_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    stage_command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, stage_command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "build-pareto",
        "run_id": run_id,
        "stage": StageName.BUILD_PARETO.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "run_root": str(run_root),
        "baseline_stage_root": str(baseline_stage_root),
        "sweep_stage_root": str(sweep_stage_root),
        "build_pareto_config": config.build_pareto.to_dict(),
        "result_paths": [str(path) for path in result_paths],
        "plot_paths": [str(path) for path in plot_paths],
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(build_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.BUILD_PARETO,
        command=stage_command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            build_summary_path,
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *result_paths,
            *plot_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "build_pareto_summary": build_summary_path,
    }
    manifest_artifacts.update(
        {
            f"build_pareto_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {
            f"build_pareto_stdout_{index}": path
            for index, path in enumerate(stdout_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"build_pareto_stderr_{index}": path
            for index, path in enumerate(stderr_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"build_pareto_result_{index}": path
            for index, path in enumerate(result_paths)
        }
    )
    manifest_artifacts.update(
        {f"build_pareto_plot_{index}": path for index, path in enumerate(plot_paths)}
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.BUILD_PARETO,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_analyze_pareto(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.ANALYZE_PARETO.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    analyze_command_record_path = (
        stage_root / "commands" / "analyze_pareto_command.json"
    )
    analyze_stdout_path = stage_root / "commands" / "analyze_pareto_stdout.txt"
    analyze_stderr_path = stage_root / "commands" / "analyze_pareto_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    analyze_summary_path = stage_root / "analyze_pareto_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "analyze_pareto_outputs"
    analysis_data_summary_path = output_dir / "analysis_summary.json"
    run_root = output_root / run_id
    build_stage_root = run_root / StageName.BUILD_PARETO.value
    build_summary_path = build_stage_root / "build_pareto_summary.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    command_results: list[dict[str, Any]] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    result_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0

    summary_notes = _read_passed_stage_summary(
        build_summary_path, StageName.BUILD_PARETO
    )
    if summary_notes:
        failures.append(FailureClass.INCOMPLETE_OUTPUT)
        notes.extend(summary_notes)

    if not failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            analyze_command = _format_analyze_pareto_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                run_root=run_root,
                build_stage_root=build_stage_root,
                command_record_path=analyze_command_record_path,
                stdout_path=analyze_stdout_path,
                stderr_path=analyze_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid analyze_pareto.command_template: {exc}")
        else:
            commands_attempted = 1
            analyze_command_record = _run_recorded_command(
                command=analyze_command,
                cwd=repo_root,
                command_record_path=analyze_command_record_path,
                stdout_path=analyze_stdout_path,
                stderr_path=analyze_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(analyze_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(analyze_command_record_path)
            stdout_paths.append(analyze_stdout_path)
            stderr_paths.append(analyze_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "analysis_summary_path": str(analysis_data_summary_path),
                    "command_record_path": str(analyze_command_record_path),
                    "stdout_path": str(analyze_stdout_path),
                    "stderr_path": str(analyze_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Analyze-Pareto command timed out."
                    if timed_out
                    else "Analyze-Pareto command failed."
                )
            else:
                commands_succeeded = 1

    if not failures:
        result_paths, result_note = _collect_paths(
            root=output_dir,
            path_glob=config.analyze_pareto.result_glob,
            field_name="analyze_pareto.result_glob",
        )
        if result_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(result_note)
        elif not result_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                "No Analyze-Pareto result matched "
                f"{config.analyze_pareto.result_glob!r} in {output_dir}."
            )
        else:
            empty_paths = [path for path in result_paths if path.stat().st_size == 0]
            if empty_paths:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(f"Analyze-Pareto output is empty: {empty_paths[0]}")
            elif not analysis_data_summary_path.exists():
                failures.append(FailureClass.INCOMPLETE_OUTPUT)
                notes.append(
                    "Analyze-Pareto machine summary is missing: "
                    f"{analysis_data_summary_path}"
                )
            elif analysis_data_summary_path.stat().st_size == 0:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    "Analyze-Pareto machine summary is empty: "
                    f"{analysis_data_summary_path}"
                )

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.ANALYZE_PARETO.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "result_files": float(len(result_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    stage_command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, stage_command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "analyze-pareto",
        "run_id": run_id,
        "stage": StageName.ANALYZE_PARETO.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "run_root": str(run_root),
        "build_stage_root": str(build_stage_root),
        "analyze_pareto_config": config.analyze_pareto.to_dict(),
        "result_paths": [str(path) for path in result_paths],
        "analysis_summary_path": str(analysis_data_summary_path)
        if analysis_data_summary_path.exists()
        else None,
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(analyze_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.ANALYZE_PARETO,
        command=stage_command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            analyze_summary_path,
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *result_paths,
            *(
                (analysis_data_summary_path,)
                if analysis_data_summary_path.exists()
                else ()
            ),
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "analyze_pareto_summary": analyze_summary_path,
    }
    if analysis_data_summary_path.exists():
        manifest_artifacts["analysis_summary"] = analysis_data_summary_path
    manifest_artifacts.update(
        {
            f"analyze_pareto_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {
            f"analyze_pareto_stdout_{index}": path
            for index, path in enumerate(stdout_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"analyze_pareto_stderr_{index}": path
            for index, path in enumerate(stderr_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"analyze_pareto_result_{index}": path
            for index, path in enumerate(result_paths)
        }
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.ANALYZE_PARETO,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_pareto(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.PARETO.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    pareto_command_record_path = stage_root / "commands" / "pareto_command.json"
    pareto_stdout_path = stage_root / "commands" / "pareto_stdout.txt"
    pareto_stderr_path = stage_root / "commands" / "pareto_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    pareto_summary_path = stage_root / "pareto_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "pareto_outputs"
    baseline_stage_root = output_root / run_id / StageName.STAGE_BASELINE.value
    sweep_stage_root = output_root / run_id / StageName.SWEEP.value
    baseline_summary_path = baseline_stage_root / "baseline_summary.json"
    sweep_summary_path = sweep_stage_root / "sweep_summary.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    result_paths: list[Path] = []
    plot_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []

    if not config.pareto_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("pareto.command_template is required for pareto.")
    for summary_path, stage_name in (
        (baseline_summary_path, StageName.STAGE_BASELINE),
        (sweep_summary_path, StageName.SWEEP),
    ):
        if not failures:
            summary_notes = _read_passed_stage_summary(summary_path, stage_name)
            if summary_notes:
                failures.append(FailureClass.INCOMPLETE_OUTPUT)
                notes.extend(summary_notes)

    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0
    if not failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            pareto_command = _format_pareto_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                baseline_stage_root=baseline_stage_root,
                sweep_stage_root=sweep_stage_root,
                command_record_path=pareto_command_record_path,
                stdout_path=pareto_stdout_path,
                stderr_path=pareto_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid pareto.command_template: {exc}")
        else:
            commands_attempted = 1
            pareto_command_record = _run_recorded_command(
                command=pareto_command,
                cwd=repo_root,
                command_record_path=pareto_command_record_path,
                stdout_path=pareto_stdout_path,
                stderr_path=pareto_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(pareto_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(pareto_command_record_path)
            stdout_paths.append(pareto_stdout_path)
            stderr_paths.append(pareto_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "command_record_path": str(pareto_command_record_path),
                    "stdout_path": str(pareto_stdout_path),
                    "stderr_path": str(pareto_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Pareto command timed out."
                    if timed_out
                    else "Pareto command failed."
                )
            else:
                commands_succeeded = 1

    if not failures:
        result_paths, result_note = _collect_paths(
            root=output_dir,
            path_glob=config.pareto_result_glob,
            field_name="pareto.result_glob",
        )
        if result_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(result_note)
        elif not result_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                f"No Pareto result JSON matched {config.pareto_result_glob!r} "
                f"in {output_dir}."
            )
        else:
            json_note = _validate_json_files(result_paths)
            if json_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(json_note)

    if not failures:
        plot_paths, plot_note = _collect_paths(
            root=output_dir,
            path_glob=config.pareto_plot_glob,
            field_name="pareto.plot_glob",
        )
        if plot_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(plot_note)
        elif not plot_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                f"No Pareto plot matched {config.pareto_plot_glob!r} in {output_dir}."
            )

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.PARETO.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "result_json_files": float(len(result_paths)),
            "plot_files": float(len(plot_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "pareto",
        "run_id": run_id,
        "stage": StageName.PARETO.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "baseline_stage_root": str(baseline_stage_root),
        "sweep_stage_root": str(sweep_stage_root),
        "command_template": list(config.pareto_command_template),
        "result_glob": config.pareto_result_glob,
        "plot_glob": config.pareto_plot_glob,
        "result_paths": [str(path) for path in result_paths],
        "plot_paths": [str(path) for path in plot_paths],
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(pareto_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.PARETO,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            pareto_summary_path,
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *result_paths,
            *plot_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "pareto_summary": pareto_summary_path,
    }
    manifest_artifacts.update(
        {
            f"pareto_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {f"pareto_stdout_{index}": path for index, path in enumerate(stdout_paths)}
    )
    manifest_artifacts.update(
        {f"pareto_stderr_{index}": path for index, path in enumerate(stderr_paths)}
    )
    manifest_artifacts.update(
        {f"pareto_result_{index}": path for index, path in enumerate(result_paths)}
    )
    manifest_artifacts.update(
        {f"pareto_plot_{index}": path for index, path in enumerate(plot_paths)}
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.PARETO,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_select(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.SELECT.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    select_command_record_path = stage_root / "commands" / "select_command.json"
    select_stdout_path = stage_root / "commands" / "select_stdout.txt"
    select_stderr_path = stage_root / "commands" / "select_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    select_summary_path = stage_root / "select_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "select_outputs"
    pareto_stage_root = output_root / run_id / StageName.PARETO.value
    pareto_summary_path = pareto_stage_root / "pareto_summary.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    selection_paths: list[Path] = []
    selected_artifact_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0

    if not config.select_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("select.command_template is required for select.")
    if not failures:
        summary_notes = _read_passed_stage_summary(
            pareto_summary_path, StageName.PARETO
        )
        if summary_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(summary_notes)

    if not failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            select_command = _format_select_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                pareto_stage_root=pareto_stage_root,
                command_record_path=select_command_record_path,
                stdout_path=select_stdout_path,
                stderr_path=select_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid select.command_template: {exc}")
        else:
            commands_attempted = 1
            select_command_record = _run_recorded_command(
                command=select_command,
                cwd=repo_root,
                command_record_path=select_command_record_path,
                stdout_path=select_stdout_path,
                stderr_path=select_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(select_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(select_command_record_path)
            stdout_paths.append(select_stdout_path)
            stderr_paths.append(select_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "command_record_path": str(select_command_record_path),
                    "stdout_path": str(select_stdout_path),
                    "stderr_path": str(select_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Select command timed out."
                    if timed_out
                    else "Select command failed."
                )
            else:
                commands_succeeded = 1

    if not failures:
        selection_paths, selection_note = _collect_paths(
            root=output_dir,
            path_glob=config.select_result_glob,
            field_name="select.result_glob",
        )
        if selection_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(selection_note)
        elif not selection_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                f"No selection JSON matched {config.select_result_glob!r} "
                f"in {output_dir}."
            )
        else:
            selection_note = _validate_selection_files(selection_paths, config=config)
            if selection_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(selection_note)
            else:
                selected_artifact_paths = [
                    _selection_artifact_path(path) for path in selection_paths
                ]

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.SELECT.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "selection_files": float(len(selection_paths)),
            "selected_artifacts": float(len(selected_artifact_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "select",
        "run_id": run_id,
        "stage": StageName.SELECT.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "pareto_stage_root": str(pareto_stage_root),
        "command_template": list(config.select_command_template),
        "result_glob": config.select_result_glob,
        "selection_paths": [str(path) for path in selection_paths],
        "selected_artifact_paths": [str(path) for path in selected_artifact_paths],
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(select_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.SELECT,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            select_summary_path,
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *selection_paths,
            *selected_artifact_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "select_summary": select_summary_path,
    }
    manifest_artifacts.update(
        {
            f"select_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {f"select_stdout_{index}": path for index, path in enumerate(stdout_paths)}
    )
    manifest_artifacts.update(
        {f"select_stderr_{index}": path for index, path in enumerate(stderr_paths)}
    )
    manifest_artifacts.update(
        {f"selection_{index}": path for index, path in enumerate(selection_paths)}
    )
    manifest_artifacts.update(
        {
            f"selected_artifact_metadata_{index}": path
            for index, path in enumerate(selected_artifact_paths)
        }
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.SELECT,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_aggregate(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.AGGREGATE.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    aggregate_command_record_path = stage_root / "commands" / "aggregate_command.json"
    aggregate_stdout_path = stage_root / "commands" / "aggregate_stdout.txt"
    aggregate_stderr_path = stage_root / "commands" / "aggregate_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    aggregate_summary_path = stage_root / "aggregate_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "aggregate_outputs"
    select_stage_root = output_root / run_id / StageName.SELECT.value
    select_summary_path = select_stage_root / "select_summary.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    aggregate_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0

    if not config.aggregate_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("aggregate.command_template is required for aggregate.")
    if not failures:
        summary_notes = _read_passed_stage_summary(
            select_summary_path, StageName.SELECT
        )
        if summary_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(summary_notes)

    if not failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            aggregate_command = _format_aggregate_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                select_stage_root=select_stage_root,
                command_record_path=aggregate_command_record_path,
                stdout_path=aggregate_stdout_path,
                stderr_path=aggregate_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid aggregate.command_template: {exc}")
        else:
            commands_attempted = 1
            aggregate_command_record = _run_recorded_command(
                command=aggregate_command,
                cwd=repo_root,
                command_record_path=aggregate_command_record_path,
                stdout_path=aggregate_stdout_path,
                stderr_path=aggregate_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(aggregate_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(aggregate_command_record_path)
            stdout_paths.append(aggregate_stdout_path)
            stderr_paths.append(aggregate_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "command_record_path": str(aggregate_command_record_path),
                    "stdout_path": str(aggregate_stdout_path),
                    "stderr_path": str(aggregate_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Aggregate command timed out."
                    if timed_out
                    else "Aggregate command failed."
                )
            else:
                commands_succeeded = 1

    aggregate_gate_passed = False
    if not failures:
        aggregate_paths, aggregate_note = _collect_paths(
            root=output_dir,
            path_glob=config.aggregate_result_glob,
            field_name="aggregate.result_glob",
        )
        if aggregate_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(aggregate_note)
        elif not aggregate_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                f"No aggregate JSON matched {config.aggregate_result_glob!r} "
                f"in {output_dir}."
            )
        else:
            aggregate_note, aggregate_gate_passed = _validate_aggregate_files(
                aggregate_paths
            )
            if aggregate_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(aggregate_note)
            elif not aggregate_gate_passed:
                failures.append(FailureClass.GATE_FAILED)
                notes.append("Aggregate result reported passed=false.")

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.AGGREGATE.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "aggregate_files": float(len(aggregate_paths)),
            "aggregate_gate_passed": 1.0 if aggregate_gate_passed else 0.0,
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "aggregate",
        "run_id": run_id,
        "stage": StageName.AGGREGATE.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "select_stage_root": str(select_stage_root),
        "command_template": list(config.aggregate_command_template),
        "result_glob": config.aggregate_result_glob,
        "aggregate_paths": [str(path) for path in aggregate_paths],
        "aggregate_gate_passed": aggregate_gate_passed,
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(aggregate_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.AGGREGATE,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            aggregate_summary_path,
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *aggregate_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "aggregate_summary": aggregate_summary_path,
    }
    manifest_artifacts.update(
        {
            f"aggregate_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {f"aggregate_stdout_{index}": path for index, path in enumerate(stdout_paths)}
    )
    manifest_artifacts.update(
        {f"aggregate_stderr_{index}": path for index, path in enumerate(stderr_paths)}
    )
    manifest_artifacts.update(
        {
            f"aggregate_result_{index}": path
            for index, path in enumerate(aggregate_paths)
        }
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.AGGREGATE,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_reserved_test(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.RESERVED_TEST.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    reserved_command_record_path = (
        stage_root / "commands" / "reserved_test_command.json"
    )
    reserved_stdout_path = stage_root / "commands" / "reserved_test_stdout.txt"
    reserved_stderr_path = stage_root / "commands" / "reserved_test_stderr.txt"
    run_record_path = stage_root / "run_record.json"
    reserved_summary_path = stage_root / "reserved_test_summary.json"
    manifest_path = stage_root / "manifest.json"
    output_dir = stage_root / "reserved_test_outputs"
    select_stage_root = output_root / run_id / StageName.SELECT.value
    aggregate_stage_root = output_root / run_id / StageName.AGGREGATE.value
    select_summary_path = select_stage_root / "select_summary.json"
    aggregate_summary_path = aggregate_stage_root / "aggregate_summary.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []
    log_paths: list[Path] = []
    selected_artifact_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    command_records: list[Path] = []
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []
    commands_attempted = 0
    commands_succeeded = 0

    if not config.users.reserved_test:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("users.reserved_test is required for reserved-test.")
    if not config.reserved_test_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("reserved_test.command_template is required for reserved-test.")
    for summary_path, stage_name in (
        (select_summary_path, StageName.SELECT),
        (aggregate_summary_path, StageName.AGGREGATE),
    ):
        if not failures:
            summary_notes = _read_passed_stage_summary(summary_path, stage_name)
            if summary_notes:
                failures.append(FailureClass.INCOMPLETE_OUTPUT)
                notes.extend(summary_notes)

    metadata_path: Path | None = None
    metadata: SchedulerArtifactMetadata | None = None
    if not failures:
        selected_artifact_paths, selected_notes = _read_selected_artifact_paths(
            select_summary_path
        )
        if selected_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(selected_notes)
        elif len(selected_artifact_paths) != 1:
            failures.append(FailureClass.INVALID_ARTIFACT)
            notes.append(
                "reserved-test requires exactly one selected artifact, got "
                f"{len(selected_artifact_paths)}."
            )
        else:
            metadata_path = selected_artifact_paths[0]
            try:
                metadata = validate_scheduler_artifact(
                    metadata_path, require_files=True
                )
            except ValueError as exc:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    f"Invalid selected scheduler artifact {metadata_path}: {exc}"
                )
            else:
                artifact_note = _validate_sweep_artifact_metadata(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    config=config,
                )
                if artifact_note is not None:
                    failures.append(FailureClass.INVALID_ARTIFACT)
                    notes.append(artifact_note)

    if not failures and metadata_path is not None and metadata is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            reserved_command = _format_reserved_test_command(
                config=config,
                config_path=config_path,
                repo_root=repo_root,
                run_id=run_id,
                stage_root=stage_root,
                output_dir=output_dir,
                metadata_path=metadata_path,
                metadata=metadata,
                command_record_path=reserved_command_record_path,
                stdout_path=reserved_stdout_path,
                stderr_path=reserved_stderr_path,
            )
        except ValueError as exc:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(f"Invalid reserved_test.command_template: {exc}")
        else:
            commands_attempted = 1
            reserved_command_record = _run_recorded_command(
                command=reserved_command,
                cwd=repo_root,
                command_record_path=reserved_command_record_path,
                stdout_path=reserved_stdout_path,
                stderr_path=reserved_stderr_path,
                timeout_seconds=config.performance.timeout_seconds,
            )
            exit_code = _record_exit_code(reserved_command_record)
            timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
            command_records.append(reserved_command_record_path)
            stdout_paths.append(reserved_stdout_path)
            stderr_paths.append(reserved_stderr_path)
            command_results.append(
                {
                    "output_dir": str(output_dir),
                    "command_record_path": str(reserved_command_record_path),
                    "stdout_path": str(reserved_stdout_path),
                    "stderr_path": str(reserved_stderr_path),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                }
            )
            if exit_code != 0:
                failures.append(
                    FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
                )
                notes.append(
                    "Reserved-test command timed out."
                    if timed_out
                    else "Reserved-test command failed."
                )
            else:
                commands_succeeded = 1

    if not failures and metadata is not None and metadata_path is not None:
        log_paths, log_note = _collect_paths(
            root=output_dir,
            path_glob=config.reserved_test_log_glob,
            field_name="reserved_test.log_glob",
        )
        if log_note is not None:
            failures.append(FailureClass.INVALID_CONFIG)
            notes.append(log_note)
        elif not log_paths:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.append(
                "No reserved-test JSONL logs matched "
                f"{config.reserved_test_log_glob!r} in {output_dir}."
            )
        else:
            log_note = _validate_reserved_test_logs(
                log_paths=log_paths,
                config=config,
                metadata=metadata,
                metadata_path=metadata_path,
            )
            if log_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(log_note)

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.RESERVED_TEST.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "reserved_users": float(len(config.users.reserved_test)),
            "logs_validated": float(len(log_paths)),
            "selected_artifacts": float(len(selected_artifact_paths)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "reserved-test",
        "run_id": run_id,
        "stage": StageName.RESERVED_TEST.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "select_stage_root": str(select_stage_root),
        "aggregate_stage_root": str(aggregate_stage_root),
        "command_template": list(config.reserved_test_command_template),
        "log_glob": config.reserved_test_log_glob,
        "selected_artifact_paths": [str(path) for path in selected_artifact_paths],
        "log_paths": [str(path) for path in log_paths],
        "command_results": command_results,
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(reserved_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.RESERVED_TEST,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            reserved_summary_path,
            manifest_path,
            *selected_artifact_paths,
            *command_records,
            *stdout_paths,
            *stderr_paths,
            *log_paths,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest_artifacts: dict[str, Path] = {
        "config_snapshot": config_snapshot_path,
        "resolved_config": resolved_config_path,
        "gate_summary": gate_summary_path,
        "command_record": command_record_path,
        "run_record": run_record_path,
        "reserved_test_summary": reserved_summary_path,
    }
    manifest_artifacts.update(
        {
            f"selected_artifact_metadata_{index}": path
            for index, path in enumerate(selected_artifact_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"reserved_test_command_record_{index}": path
            for index, path in enumerate(command_records)
        }
    )
    manifest_artifacts.update(
        {
            f"reserved_test_stdout_{index}": path
            for index, path in enumerate(stdout_paths)
        }
    )
    manifest_artifacts.update(
        {
            f"reserved_test_stderr_{index}": path
            for index, path in enumerate(stderr_paths)
        }
    )
    manifest_artifacts.update(
        {f"reserved_test_log_{index}": path for index, path in enumerate(log_paths)}
    )
    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts=manifest_artifacts,
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path, *command_records),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.RESERVED_TEST,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_stage_baseline(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.STAGE_BASELINE.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    baseline_summary_path = stage_root / "baseline_summary.json"
    manifest_path = stage_root / "manifest.json"
    staged_root = stage_root / "baseline_logs"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    baseline_root = _resolve_repo_path(repo_root, config.baseline.log_root)
    failures: list[FailureClass] = []
    notes: list[str] = []
    staged_logs: list[Path] = []
    baseline_envs = config.baseline.environments or (config.simulation.environment,)
    required_users = (
        set(config.users.train)
        | set(config.users.validation)
        | set(config.users.reserved_test)
    )
    logs_by_env_user: dict[tuple[str, int], list[Path]] = {}
    retentions_by_env_user: dict[tuple[str, int], set[float]] = {}
    selected_retentions_by_user: dict[int, tuple[float, ...]] = {}

    try:
        selected_retentions_by_user = _baseline_dr_values_by_user(
            config=config,
            repo_root=repo_root,
            user_ids=sorted(required_users),
            fallback=config.baseline.desired_retention_values,
        )
    except ValueError as exc:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append(f"Invalid baseline DR selection manifest: {exc}")

    if not baseline_root.exists():
        failures.append(FailureClass.INVALID_BASELINE)
        notes.append(f"Baseline log root does not exist: {baseline_root}")
    elif not failures:
        all_selected_retentions = tuple(
            sorted(
                {
                    retention
                    for retentions in selected_retentions_by_user.values()
                    for retention in retentions
                }
            )
        )
        filename_filter = _baseline_filename_filter(
            config,
            environments=baseline_envs,
            desired_retention_values=all_selected_retentions,
        )
        candidate_paths = _baseline_candidate_paths(
            baseline_root=baseline_root,
            required_users=required_users,
            filename_filter=filename_filter,
        )
        for path in candidate_paths:
            meta = _read_log_meta(path)
            if meta is None:
                continue
            user_id = meta.get("user_id")
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                notes.append(f"Skipping {path}: missing integer user_id.")
                continue
            expected_retentions = selected_retentions_by_user.get(user_id, ())
            environment = meta.get("environment")
            if environment not in baseline_envs:
                continue
            if (
                config.baseline_dr_selection.manifest is not None
                and expected_retentions
                and _matched_retention_value(
                    meta.get("desired_retention"),
                    expected_retentions,
                )
                is None
            ):
                continue
            metadata_errors = _baseline_metadata_errors(
                config=config,
                meta=meta,
                expected_environment=str(environment),
                expected_desired_retention_values=expected_retentions,
            )
            if metadata_errors:
                notes.extend(f"{path}: {error}" for error in metadata_errors)
                continue
            env_user_key = (str(environment), user_id)
            logs_by_env_user.setdefault(env_user_key, []).append(path)
            retention_value = _matched_retention_value(
                meta.get("desired_retention"),
                expected_retentions,
            )
            if retention_value is not None:
                retentions_by_env_user.setdefault(env_user_key, set()).add(
                    retention_value
                )

        missing_env_users = [
            f"env={environment},user={user_id}"
            for environment in baseline_envs
            for user_id in sorted(required_users)
            if (environment, user_id) not in logs_by_env_user
        ]
        if missing_env_users:
            failures.append(FailureClass.INVALID_BASELINE)
            notes.append(
                "Missing exact baseline logs for environment/user pairs: "
                + ", ".join(missing_env_users)
            )
        if not logs_by_env_user:
            failures.append(FailureClass.INVALID_BASELINE)
            notes.append("No exact baseline logs matched the config.")
        if any(selected_retentions_by_user.values()):
            missing_pairs: list[str] = []
            for environment in baseline_envs:
                for user_id in sorted(required_users):
                    required_retentions = set(
                        selected_retentions_by_user.get(user_id, ())
                    )
                    missing_retentions = sorted(
                        required_retentions
                        - retentions_by_env_user.get((environment, user_id), set())
                    )
                    for retention in missing_retentions:
                        missing_pairs.append(
                            f"env={environment},user={user_id},ret={retention:.12g}"
                        )
            if missing_pairs:
                failures.append(FailureClass.INVALID_BASELINE)
                notes.append(
                    "Missing exact baseline retention points: "
                    + ", ".join(missing_pairs)
                )

        if not failures:
            for environment, user_id in sorted(logs_by_env_user):
                for source in logs_by_env_user[(environment, user_id)]:
                    source_meta = _read_log_meta(source) or {}
                    retention_value = _matched_retention_value(
                        source_meta.get("desired_retention"),
                        selected_retentions_by_user.get(user_id, ()),
                    )
                    dest_parent = staged_root / f"env_{environment}" / f"user_{user_id}"
                    if retention_value is not None:
                        dest_parent = dest_parent / (
                            "dr_" + _format_retention_token(retention_value)
                        )
                    dest = dest_parent / source.name
                    _stage_baseline_file(
                        source=source,
                        dest=dest,
                        mode=config.baseline.stage_mode,
                    )
                    staged_logs.append(dest)

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.STAGE_BASELINE.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "matched_environment_user_pairs": float(len(logs_by_env_user)),
            "matched_user_retention_pairs": float(
                sum(len(values) for values in retentions_by_env_user.values())
            ),
            "staged_logs": float(len(staged_logs)),
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    summary = {
        "type": "stage-baseline",
        "run_id": run_id,
        "stage": StageName.STAGE_BASELINE.value,
        "passed": passed,
        "failures": [failure.value for failure in unique_failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "baseline_root": str(baseline_root),
        "baseline_environments": list(baseline_envs),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "stage_mode": config.baseline.stage_mode,
        "matched_users": sorted({user_id for _, user_id in logs_by_env_user}),
        "matched_environment_user_pairs": [
            {"environment": environment, "user_id": user_id}
            for environment, user_id in sorted(logs_by_env_user)
        ],
        "matched_retentions_by_user": {
            str(user_id): sorted(
                {
                    retention
                    for (
                        environment,
                        env_user_id,
                    ), retentions in retentions_by_env_user.items()
                    if env_user_id == user_id
                    for retention in retentions
                }
            )
            for user_id in sorted({user_id for _, user_id in logs_by_env_user})
        },
        "matched_retentions_by_environment_user": {
            f"{environment}:user_{user_id}": sorted(values)
            for (environment, user_id), values in sorted(retentions_by_env_user.items())
        },
        "selected_retentions_by_user": {
            str(user_id): list(values)
            for user_id, values in sorted(selected_retentions_by_user.items())
        },
        "staged_logs": [str(path) for path in staged_logs],
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(baseline_summary_path, summary)

    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.STAGE_BASELINE,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gate_summary_path,
            baseline_summary_path,
            manifest_path,
            *staged_logs,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts={
            "config_snapshot": config_snapshot_path,
            "resolved_config": resolved_config_path,
            "gate_summary": gate_summary_path,
            "command_record": command_record_path,
            "run_record": run_record_path,
            "baseline_summary": baseline_summary_path,
            **{f"baseline_log_{index}": path for index, path in enumerate(staged_logs)},
        },
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path,),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.STAGE_BASELINE,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def run_preflight(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    command: list[str],
) -> StageExecutionResult:
    started_at = utc_timestamp()
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.PREFLIGHT.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gpu_summary_path = stage_root / "gpu_summary.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    preflight_summary_path = stage_root / "preflight_summary.json"
    manifest_path = stage_root / "manifest.json"

    shutil.copyfile(config_path, config_snapshot_path)
    _write_json(resolved_config_path, config.to_dict())

    failures: list[FailureClass] = []
    notes: list[str] = []

    baseline_root = _resolve_repo_path(repo_root, config.baseline.log_root)
    if not baseline_root.exists():
        failures.append(FailureClass.INVALID_BASELINE)
        notes.append(f"Baseline log root does not exist: {baseline_root}")

    gpu_summary = _build_gpu_summary(config)
    if not gpu_summary.passed:
        failures.append(FailureClass.GPU_GUARD_FAILED)
    _write_json(gpu_summary_path, gpu_summary.to_dict())

    passed = not failures
    gate_summary = GateSummary(
        gate_name=StageName.PREFLIGHT.value,
        passed=passed,
        failures=tuple(failures),
        metrics={
            "baseline_root_exists": 1.0 if baseline_root.exists() else 0.0,
            "gpu_guard_passed": 1.0 if gpu_summary.passed else 0.0,
        },
        thresholds={},
    )
    _write_json(gate_summary_path, gate_summary.to_dict())

    finished_at = utc_timestamp()
    exit_code = 0 if passed else 1
    command_record = CommandRecord(
        command=tuple(command),
        cwd=repo_root,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=None,
        stderr_path=None,
    )
    _write_json(command_record_path, command_record.to_dict())

    provenance = collect_environment_summary(repo_root)
    run_record = RunRecord(
        run_id=run_id,
        stage=StageName.PREFLIGHT,
        command=command_record,
        config_path=config_path,
        config_snapshot_path=config_snapshot_path,
        resolved_config_path=resolved_config_path,
        git_commit=provenance["git_commit"],
        dirty=bool(provenance["dirty"]),
        uv_lock_hash=provenance["uv_lock_hash"],
        python_version=provenance["python_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        artifact_paths=(
            gpu_summary_path,
            gate_summary_path,
            preflight_summary_path,
            manifest_path,
        ),
    )
    _write_json(run_record_path, run_record.to_dict())

    summary = {
        "type": "preflight",
        "run_id": run_id,
        "stage": StageName.PREFLIGHT.value,
        "passed": passed,
        "failures": [failure.value for failure in failures],
        "notes": notes,
        "repo_root": str(repo_root),
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gpu_summary_path": str(gpu_summary_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
        "manifest_path": str(manifest_path),
        "environment": provenance,
    }
    _write_json(preflight_summary_path, summary)

    manifest = ArtifactManifest(
        run_id=run_id,
        artifacts={
            "config_snapshot": config_snapshot_path,
            "resolved_config": resolved_config_path,
            "gpu_summary": gpu_summary_path,
            "gate_summary": gate_summary_path,
            "command_record": command_record_path,
            "run_record": run_record_path,
            "preflight_summary": preflight_summary_path,
        },
        config_snapshot_path=config_snapshot_path,
        gate_summary_path=gate_summary_path,
        command_record_paths=(command_record_path,),
    )
    _write_json(manifest_path, manifest.to_dict())

    return StageExecutionResult(
        exit_code=exit_code,
        stage=StageName.PREFLIGHT,
        run_id=run_id,
        stage_root=stage_root,
        summary=summary,
    )


def collect_environment_summary(repo_root: Path) -> dict[str, Any]:
    torch_summary = _torch_summary()
    return {
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD") or "unknown",
        "dirty": bool(_git_output(repo_root, "status", "--porcelain")),
        "uv_lock_hash": _file_sha256(repo_root / "uv.lock") or "missing",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        **torch_summary,
    }


def _build_gpu_summary(config: ExperimentConfig) -> GpuGuardSummary:
    torch_summary = _torch_summary()
    cuda_available = bool(torch_summary["cuda_available"])
    requested_device = config.gpu_guard.device or (
        "cuda" if config.gpu_guard.required else "cpu"
    )
    notes: list[str] = []
    passed = True
    fallback_used = False

    if requested_device.startswith("cuda") and not cuda_available:
        passed = False
        notes.append("CUDA device requested but torch.cuda.is_available() is false.")
    elif config.gpu_guard.required and not cuda_available:
        passed = False
        notes.append("GPU guard requires CUDA but CUDA is unavailable.")
    elif requested_device == "cpu" and not config.gpu_guard.required:
        fallback_used = not cuda_available

    if passed and config.gpu_guard.smoke and requested_device.startswith("cuda"):
        smoke_ok, smoke_note = _run_cuda_smoke(requested_device)
        passed = smoke_ok
        notes.append(smoke_note)

    return GpuGuardSummary(
        passed=passed,
        device=requested_device,
        workload_shape={
            "days": config.simulation.days,
            "deck": config.simulation.deck,
            "train_users": len(config.users.train),
            "lambda_values": _training_reported_lambda_count(config),
        },
        fallback_used=fallback_used,
        notes=tuple(notes),
    )


def _build_stage_performance_summary(
    *,
    config: ExperimentConfig,
    stage: StageName,
    passed: bool,
    elapsed_seconds: float,
    stage_root: Path,
    failures: tuple[FailureClass, ...],
    notes: list[str],
    runtime_metrics: dict[str, Any],
    execution_shape: dict[str, Any],
    gpu_monitor_summary: GpuMonitorSummary | None = None,
) -> PerformanceSummary:
    device = _resolve_performance_device(config)
    runtime = {
        "elapsed_seconds": elapsed_seconds,
        **runtime_metrics,
    }
    candidate_days = _candidate_days(config=config, stage=stage)
    if candidate_days is not None:
        runtime["candidate_days"] = candidate_days
        runtime["candidate_days_per_second"] = candidate_days / max(
            elapsed_seconds, 1e-9
        )
    user_days = _user_days(config=config, stage=stage)
    if user_days is not None:
        runtime["user_days"] = user_days
        runtime["user_days_per_second"] = user_days / max(elapsed_seconds, 1e-9)

    return PerformanceSummary(
        stage=stage,
        passed=passed,
        device=device,
        workload_shape=_performance_workload_shape(config, stage),
        execution_shape={
            "device": device,
            "write_performance_summary": config.performance.write_performance_summary,
            "diagnostic_csv_logs": config.performance.diagnostic_csv_logs,
            **execution_shape,
        },
        runtime_metrics=runtime,
        gpu_metrics={
            **_gpu_performance_metrics(device),
            **_gpu_monitor_performance_metrics(gpu_monitor_summary),
        },
        disk_metrics=_disk_metrics(stage_root),
        failure_class=failures[0] if failures else None,
        notes=tuple(notes),
    )


def _maybe_start_stage_gpu_monitor(
    *,
    config: ExperimentConfig,
    stage_root: Path,
) -> GpuMonitor | None:
    device = _resolve_performance_device(config)
    enabled = config.performance.gpu_monitor_enabled
    if enabled is None:
        enabled = device.startswith("cuda")
    if not enabled:
        return None
    monitor = GpuMonitor(
        output_dir=stage_root / "gpu_monitor",
        interval_seconds=config.performance.gpu_monitor_interval_seconds,
    )
    monitor.start()
    return monitor


def _stop_stage_gpu_monitor(monitor: GpuMonitor | None) -> GpuMonitorSummary | None:
    if monitor is None:
        return None
    return monitor.stop()


def _gpu_monitor_performance_metrics(
    summary: GpuMonitorSummary | None,
) -> dict[str, Any]:
    if summary is None:
        return {
            "gpu_monitor_enabled": False,
            "gpu_monitor_summary_path": None,
            "gpu_monitor_jsonl_path": None,
            "gpu_monitor_shared_memory_peak_single_adapter_bytes": None,
            "gpu_monitor_shared_memory_peak_summed_bytes": None,
            "gpu_monitor_shared_memory_spill_detected": None,
        }
    return {
        "gpu_monitor_enabled": summary.enabled,
        "gpu_monitor_summary_path": str(summary.summary_path),
        "gpu_monitor_jsonl_path": str(summary.jsonl_path),
        "gpu_monitor_sample_count": summary.sample_count,
        "gpu_monitor_shared_memory_peak_single_adapter_bytes": (
            summary.shared_memory_peak_single_adapter_bytes
        ),
        "gpu_monitor_shared_memory_peak_summed_bytes": (
            summary.shared_memory_peak_summed_bytes
        ),
        "gpu_monitor_shared_memory_spill_threshold_bytes": (
            summary.shared_memory_spill_threshold_bytes
        ),
        "gpu_monitor_shared_memory_spill_detected": (
            summary.shared_memory_spill_detected
        ),
        "gpu_monitor_nvidia_smi_peak_memory_used_mib": (
            summary.nvidia_smi_peak_memory_used_mib
        ),
        "gpu_monitor_nvidia_smi_peak_utilization_percent": (
            summary.nvidia_smi_peak_utilization_percent
        ),
    }


def _resolve_performance_device(config: ExperimentConfig) -> str:
    if config.performance.device:
        return config.performance.device
    if config.gpu_guard.device:
        return config.gpu_guard.device
    torch_device = config.training_policy_search.get("torch_device")
    if isinstance(torch_device, str) and torch_device.strip():
        return torch_device.strip()
    return "cuda" if config.gpu_guard.required else "cpu"


def _performance_workload_shape(
    config: ExperimentConfig, stage: StageName
) -> dict[str, Any]:
    shape: dict[str, Any] = {
        "days": config.simulation.days,
        "deck": config.simulation.deck,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": config.simulation.review_markov_transition,
        "train_users": len(config.users.train),
        "validation_users": len(config.users.validation),
        "reserved_test_users": len(config.users.reserved_test),
        "lambda_values": _training_reported_lambda_count(config),
        "training_baseline_desired_retention_values": len(
            _training_baseline_desired_retention_values(config)
        ),
        "baseline_retention_values": len(config.baseline.desired_retention_values),
    }
    candidate_lanes = _training_candidate_lanes(config)
    if candidate_lanes is not None:
        shape["candidate_lanes"] = candidate_lanes
    if stage == StageName.TRAIN_OVERFIT and candidate_lanes is not None:
        shape["effective_lanes"] = (
            len(config.users.train)
            * _training_effective_lambda_count(config)
            * len(_training_baseline_desired_retention_values(config))
            * candidate_lanes
        )
    elif stage == StageName.SWEEP:
        shape["effective_lanes"] = (
            len(config.users.train)
            * _training_effective_lambda_count(config)
            * len(_training_baseline_desired_retention_values(config))
        )
    return shape


def _training_candidate_lanes(config: ExperimentConfig) -> int | None:
    values = []
    for raw in (config.training_optimizer, config.training_portfolio):
        for key in ("population_size", "offspring_size"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                continue
            values.append(value)
    if not values:
        return None
    return max(values)


def _candidate_days(*, config: ExperimentConfig, stage: StageName) -> int | None:
    if stage != StageName.TRAIN_OVERFIT:
        return None
    candidate_lanes = _training_candidate_lanes(config)
    if candidate_lanes is None:
        return None
    return (
        config.simulation.days
        * len(config.users.train)
        * _training_effective_lambda_count(config)
        * len(_training_baseline_desired_retention_values(config))
        * candidate_lanes
    )


def _user_days(*, config: ExperimentConfig, stage: StageName) -> int | None:
    if stage == StageName.TRAIN_OVERFIT:
        return (
            config.simulation.days
            * len(config.users.train)
            * _training_effective_lambda_count(config)
            * len(_training_baseline_desired_retention_values(config))
        )
    if stage == StageName.SWEEP:
        return (
            config.simulation.days
            * len(config.users.train)
            * _training_effective_lambda_count(config)
            * len(_training_baseline_desired_retention_values(config))
        )
    return None


def _gpu_performance_metrics(device: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "torch_cuda_available": False,
        "cuda_device_count": 0,
        "fallback_used": device.startswith("cuda"),
        "peak_allocated_memory_bytes": None,
        "peak_reserved_memory_bytes": None,
        "current_allocated_memory_bytes": None,
        "current_reserved_memory_bytes": None,
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        metrics["torch_cuda_available"] = cuda_available
        metrics["cuda_device_count"] = (
            torch.cuda.device_count() if cuda_available else 0
        )
        metrics["fallback_used"] = device.startswith("cuda") and not cuda_available
        if device.startswith("cuda") and cuda_available:
            torch_device = torch.device(device)
            if torch_device.index is not None:
                torch.cuda.set_device(torch_device)
            current_device = torch.cuda.current_device()
            metrics["device_index"] = current_device
            metrics["device_name"] = torch.cuda.get_device_name(current_device)
            metrics["peak_allocated_memory_bytes"] = torch.cuda.max_memory_allocated(
                current_device
            )
            metrics["peak_reserved_memory_bytes"] = torch.cuda.max_memory_reserved(
                current_device
            )
            metrics["current_allocated_memory_bytes"] = torch.cuda.memory_allocated(
                current_device
            )
            metrics["current_reserved_memory_bytes"] = torch.cuda.memory_reserved(
                current_device
            )
    except Exception as exc:  # pragma: no cover - hardware-dependent.
        metrics["notes"] = [f"torch CUDA metrics unavailable: {exc}"]
    return metrics


def _disk_metrics(stage_root: Path) -> dict[str, Any]:
    files = [path for path in stage_root.rglob("*") if path.is_file()]
    jsonl_files = [path for path in files if path.suffix == ".jsonl"]
    csv_files = [path for path in files if path.suffix == ".csv"]
    json_files = [path for path in files if path.suffix == ".json"]
    return {
        "file_count": len(files),
        "total_bytes": sum(_file_size(path) for path in files),
        "json_count": len(json_files),
        "json_bytes": sum(_file_size(path) for path in json_files),
        "jsonl_count": len(jsonl_files),
        "jsonl_bytes": sum(_file_size(path) for path in jsonl_files),
        "csv_count": len(csv_files),
        "csv_bytes": sum(_file_size(path) for path in csv_files),
    }


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _run_cuda_smoke(device: str) -> tuple[bool, str]:
    try:
        import torch

        tensor = torch.empty((1,), device=device)
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        return True, f"CUDA smoke succeeded on {device}."
    except Exception as exc:  # pragma: no cover - hardware-dependent.
        return False, f"CUDA smoke failed on {device}: {exc}"


def _torch_summary() -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        return {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "torch_version": f"unavailable: {exc}",
            "cuda_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
        }


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_log_meta(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
    except OSError:
        return None
    if not first_line:
        return None
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if record.get("type") != "meta":
        return None
    data = record.get("data")
    return data if isinstance(data, dict) else None


def _baseline_metadata_errors(
    *,
    config: ExperimentConfig,
    meta: dict[str, Any],
    expected_environment: str | None = None,
    expected_desired_retention_values: tuple[float, ...] | None = None,
) -> list[str]:
    errors = _simulation_metadata_errors(
        config=config,
        meta=meta,
        expected_engine=config.baseline.expected_engine,
        expected_scheduler=config.baseline.scheduler,
        expected_environment=expected_environment,
    )
    expected_retentions = (
        expected_desired_retention_values
        if expected_desired_retention_values is not None
        else config.baseline.desired_retention_values
    )
    if expected_retentions:
        actual_retention = meta.get("desired_retention")
        if _matched_retention_value(actual_retention, expected_retentions) is None:
            errors.append(
                "metadata desired_retention expected one of "
                f"{list(expected_retentions)!r}, "
                f"got {actual_retention!r}"
            )
    return errors


def _baseline_filename_filter(
    config: ExperimentConfig,
    *,
    environments: Sequence[str] | None = None,
    desired_retention_values: tuple[float, ...] | None = None,
) -> LogFilenameFilter:
    short_term = "on" if config.simulation.short_term_source else "off"
    short_term_source = config.simulation.short_term_source or "any"
    retention_values = (
        desired_retention_values
        if desired_retention_values is not None
        else config.baseline.desired_retention_values
    )
    retention_values_by_scheduler = None
    if len(retention_values) == 1:
        retention_values_by_scheduler = {
            config.baseline.scheduler: round(retention_values[0], 2)
        }
    start_retention, end_retention = _filename_retention_bounds(retention_values)
    return LogFilenameFilter(
        envs=list(environments or (config.simulation.environment,)),
        scheds=[config.baseline.scheduler],
        engine=config.baseline.expected_engine,
        short_term=short_term,
        short_term_source=short_term_source,
        start_retention=start_retention,
        end_retention=end_retention,
        priority=config.simulation.priority,
        retention_values_by_scheduler=retention_values_by_scheduler,
    )


def _filename_retention_bounds(
    retention_values: Sequence[float],
) -> tuple[float | None, float | None]:
    if not retention_values:
        return None, None
    # Log filenames store ret= rounded to two decimals. Keep filename filtering
    # broad enough for continuous DRs; exact matching still happens via metadata.
    tolerance = 0.005000001
    return min(retention_values) - tolerance, max(retention_values) + tolerance


def _baseline_candidate_paths(
    *,
    baseline_root: Path,
    required_users: set[int],
    filename_filter: LogFilenameFilter,
) -> list[Path]:
    user_dir_paths: list[Path] = []
    for user_id in sorted(required_users):
        user_dir = baseline_root / f"user_{user_id}"
        if user_dir.is_dir():
            user_dir_paths.extend(sorted(user_dir.rglob("*.jsonl")))
    if user_dir_paths:
        filtered = [
            path for path in user_dir_paths if filename_filter.matches(path.name)
        ]
        return filtered or user_dir_paths

    all_paths = sorted(baseline_root.rglob("*.jsonl"))
    filtered = [path for path in all_paths if filename_filter.matches(path.name)]
    return filtered or all_paths


def _simulation_metadata_errors(
    *,
    config: ExperimentConfig,
    meta: dict[str, Any],
    expected_engine: str,
    expected_scheduler: str | None = None,
    expected_user_id: int | None = None,
    expected_environment: str | None = None,
) -> list[str]:
    expected: dict[str, Any] = {
        "engine": expected_engine,
        "days": config.simulation.days,
        "deck_size": config.simulation.deck,
        "learn_limit": config.simulation.learn_limit,
        "review_limit": config.simulation.review_limit,
        "cost_limit_minutes": config.simulation.cost_limit_minutes,
        "priority": config.simulation.priority,
        "environment": expected_environment or config.simulation.environment,
        "scheduler_priority": config.simulation.scheduler_priority,
        "seed": config.seed,
        "fuzz": config.simulation.fuzz,
        "short_term": bool(config.simulation.short_term_source),
        "short_term_source": config.simulation.short_term_source,
        "review_markov_transition": config.simulation.review_markov_transition,
    }
    if expected_scheduler is not None:
        expected["scheduler"] = expected_scheduler
    if expected_user_id is not None:
        expected["user_id"] = expected_user_id
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual_value = meta.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (float, int)) or not math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-9
            ):
                errors.append(
                    f"metadata {key} expected {expected_value!r}, got {actual_value!r}"
                )
        elif actual_value != expected_value:
            errors.append(
                f"metadata {key} expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def _matched_retention_value(
    value: Any, expected_values: tuple[float, ...]
) -> float | None:
    if not expected_values:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    actual = float(value)
    for expected in expected_values:
        if math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            return expected
    return None


def _stage_baseline_file(*, source: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "hardlink":
        dest.hardlink_to(source)
    else:
        shutil.copy2(source, dest)


def _run_recorded_command(
    *,
    command: list[str],
    cwd: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None = None,
) -> CommandRecord:
    command_record_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_timestamp()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        exit_code = COMMAND_TIMEOUT_EXIT_CODE
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_note = f"Command timed out after {timeout_seconds} seconds."
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(
            f"{stderr}\n{timeout_note}\n" if stderr else f"{timeout_note}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        exit_code = 127
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
    finished_at = utc_timestamp()
    command_record = CommandRecord(
        command=tuple(command),
        cwd=cwd,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    _write_json(command_record_path, command_record.to_dict())
    return command_record


def _record_exit_code(command_record: CommandRecord) -> int:
    return 1 if command_record.exit_code is None else command_record.exit_code


def _format_train_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    user_id: int,
    lambda_value: float | None,
    baseline_desired_retention: float,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    lambda_token = (
        _format_lambda_token(lambda_value) if lambda_value is not None else ""
    )
    baseline_dr_token = _format_retention_token(baseline_desired_retention)
    values: dict[str, Any] = {
        "user_id": user_id,
        "lambda_value": lambda_value if lambda_value is not None else "",
        "lambda_token": lambda_token,
        "baseline_desired_retention": baseline_desired_retention,
        "baseline_desired_retention_token": baseline_dr_token,
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "scheduler": config.baseline.scheduler,
        "repo_root": str(repo_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.train_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _build_train_command_jobs(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    outputs_root: Path,
    commands_root: Path,
    baseline_dr_values: tuple[float, ...],
) -> tuple[list[TrainCommandJob], list[str]]:
    jobs: list[TrainCommandJob] = []
    notes: list[str] = []
    portfolio_trainer = _training_uses_portfolio_trainer(config)
    batch_baseline_dr = (
        _training_batches_baseline_dr_grid(config) and not portfolio_trainer
    )
    include_baseline_dr_in_path = (
        _training_uses_baseline_dr_grid(config)
        and not batch_baseline_dr
        and not portfolio_trainer
    )
    job_baseline_dr_values = (
        (_training_primary_baseline_desired_retention(config, baseline_dr_values),)
        if batch_baseline_dr or portfolio_trainer
        else baseline_dr_values
    )
    for user_id in config.users.train:
        for baseline_dr in job_baseline_dr_values:
            baseline_dr_token = _format_retention_token(baseline_dr)
            for lambda_value in _training_lambda_values_for_jobs(config):
                lambda_token = (
                    _format_lambda_token(lambda_value)
                    if lambda_value is not None
                    else None
                )
                if portfolio_trainer:
                    output_dir = outputs_root / f"user_{user_id}"
                    command_stem = f"user_{user_id}"
                elif include_baseline_dr_in_path:
                    assert lambda_token is not None
                    output_dir = (
                        outputs_root
                        / f"user_{user_id}"
                        / f"dr_{baseline_dr_token}"
                        / f"lambda_{lambda_token}"
                    )
                    command_stem = (
                        f"user_{user_id}_dr_{baseline_dr_token}_lambda_{lambda_token}"
                    )
                else:
                    assert lambda_token is not None
                    output_dir = (
                        outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
                    )
                    command_stem = f"user_{user_id}_lambda_{lambda_token}"
                command_record = commands_root / f"{command_stem}_command.json"
                stdout_path = commands_root / f"{command_stem}_stdout.txt"
                stderr_path = commands_root / f"{command_stem}_stderr.txt"
                train_command: list[str] = []
                if config.train_command_template:
                    try:
                        train_command = _format_train_command(
                            config=config,
                            config_path=config_path,
                            repo_root=repo_root,
                            run_id=run_id,
                            stage_root=stage_root,
                            output_dir=output_dir,
                            user_id=user_id,
                            lambda_value=lambda_value,
                            baseline_desired_retention=baseline_dr,
                            command_record_path=command_record,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                    except (KeyError, ValueError) as exc:
                        notes.append(
                            "Invalid training.command_template for "
                            f"user={user_id}, baseline_dr={baseline_dr}, "
                            f"lambda={lambda_value}: {exc}"
                        )
                        return jobs, notes
                jobs.append(
                    TrainCommandJob(
                        user_id=user_id,
                        baseline_desired_retention=baseline_dr,
                        baseline_desired_retention_token=baseline_dr_token,
                        lambda_value=lambda_value,
                        lambda_token=lambda_token,
                        output_dir=output_dir,
                        command_record_path=command_record,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        command=train_command,
                    )
                )
    return jobs, notes


def _run_train_command_job(
    *,
    job: TrainCommandJob,
    config: ExperimentConfig,
    repo_root: Path,
) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    train_command_record = _run_recorded_command(
        command=job.command,
        cwd=repo_root,
        command_record_path=job.command_record_path,
        stdout_path=job.stdout_path,
        stderr_path=job.stderr_path,
        timeout_seconds=config.performance.timeout_seconds,
    )
    exit_code = _record_exit_code(train_command_record)
    timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
    return _finalize_train_job_result(
        job=job,
        config=config,
        exit_code=exit_code,
        timed_out=timed_out,
        command_record_path=job.command_record_path,
        stdout_path=job.stdout_path,
        stderr_path=job.stderr_path,
        command_result_extra={},
        failure_verb="Training command",
    )


def _train_failure_note(
    *,
    failure_verb: str,
    timed_out: bool,
    job: TrainCommandJob,
) -> str:
    status = "timed out" if timed_out else "failed"
    parts = [
        f"{failure_verb} {status} for user={job.user_id}",
        f"baseline_dr={job.baseline_desired_retention}",
    ]
    if job.lambda_value is not None:
        parts.append(f"lambda={job.lambda_value}")
    return ", ".join(parts) + "."


def _finalize_train_job_result(
    *,
    job: TrainCommandJob,
    config: ExperimentConfig,
    exit_code: int,
    timed_out: bool,
    command_record_path: Path,
    stdout_path: Path | None,
    stderr_path: Path | None,
    command_result_extra: dict[str, Any],
    failure_verb: str,
) -> dict[str, Any]:
    progress_path = job.output_dir / "training_progress.jsonl"
    progress_path_exists = progress_path.exists()
    command_result = {
        "user_id": job.user_id,
        "baseline_desired_retention": job.baseline_desired_retention,
        "baseline_desired_retention_token": job.baseline_desired_retention_token,
        "output_dir": str(job.output_dir),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path) if stdout_path is not None else None,
        "stderr_path": str(stderr_path) if stderr_path is not None else None,
        "training_progress_path": str(progress_path) if progress_path_exists else None,
        "exit_code": exit_code,
        "timed_out": timed_out,
        **command_result_extra,
    }
    if job.lambda_value is not None:
        command_result["lambda_value"] = job.lambda_value
        command_result["lambda_token"] = job.lambda_token
    result: dict[str, Any] = {
        "job": job,
        "command_record_path": command_record_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "progress_path": progress_path if progress_path_exists else None,
        "command_result": command_result,
        "artifact_paths": [],
        "failure": None,
        "note": None,
        "succeeded": False,
    }
    if exit_code != 0:
        result["failure"] = (
            FailureClass.TIMEOUT if timed_out else FailureClass.RUNNER_FAILED
        )
        if timed_out:
            result["note"] = _train_failure_note(
                failure_verb=failure_verb,
                timed_out=True,
                job=job,
            )
        else:
            result["note"] = _train_failure_note(
                failure_verb=failure_verb,
                timed_out=False,
                job=job,
            )
        return result

    try:
        matched_artifacts = sorted(
            path
            for path in job.output_dir.glob(config.train_artifact_glob)
            if path.is_file()
        )
    except ValueError as exc:
        result["failure"] = FailureClass.INVALID_CONFIG
        result["note"] = (
            "Invalid training.artifact_metadata_glob "
            f"{config.train_artifact_glob!r}: {exc}"
        )
        return result
    if not matched_artifacts:
        result["failure"] = FailureClass.INVALID_ARTIFACT
        result["note"] = (
            "No scheduler artifact metadata matched "
            f"{config.train_artifact_glob!r} in {job.output_dir}."
        )
        return result

    invalid_artifact_note = _validate_train_artifacts(
        artifact_paths=matched_artifacts,
        config=config,
        user_id=job.user_id,
        lambda_value=job.lambda_value,
        allowed_baseline_desired_retentions=_training_baseline_desired_retention_values(
            config
        )
        if _training_batches_baseline_dr_grid(config)
        else None,
        baseline_desired_retention=job.baseline_desired_retention
        if _training_metadata_requires_baseline_dr(config)
        and not _training_batches_baseline_dr_grid(config)
        else None,
    )
    if invalid_artifact_note is not None:
        result["failure"] = FailureClass.INVALID_ARTIFACT
        result["note"] = invalid_artifact_note
        return result
    result["artifact_paths"] = matched_artifacts
    result["succeeded"] = True
    return result


def _run_train_in_process_batches(
    *,
    jobs: list[TrainCommandJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    commands_root: Path,
) -> tuple[list[dict[str, Any]], int, int, int, str | None, list[str]]:
    from simulator.experiment_infra.training_batch import (
        InProcessTrainJob,
        estimate_lanes_per_job,
        resolve_in_process_trainer,
        run_in_process_train_batch,
    )

    notes: list[str] = []
    try:
        trainer = resolve_in_process_trainer(
            configured_trainer=config.training_batch.trainer,
            command_template=config.train_command_template,
        )
    except ValueError as exc:
        return [], 0, 0, 0, None, [str(exc)]

    try:
        lanes_per_job = estimate_lanes_per_job(trainer=trainer, config=config)
    except ValueError as exc:
        return [], 0, 0, 0, trainer, [str(exc)]
    batches = _build_train_user_batches(
        jobs=jobs,
        batch_size=config.training_batch.batch_size,
        max_lanes_per_batch=config.training_batch.max_lanes_per_batch,
        lanes_per_job=lanes_per_job,
    )
    commands_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    batch_runs_attempted = 0
    batch_runs_succeeded = 0
    max_effective_lanes = 0
    elapsed_started = time.monotonic()
    for batch_index, batch_jobs in enumerate(batches):
        batch_runs_attempted += 1
        batch_record_path = commands_root / f"in_process_batch_{batch_index}.json"
        started_at = utc_timestamp()
        outcomes = []
        error_note: str | None = None
        try:
            outcomes = run_in_process_train_batch(
                trainer=trainer,
                jobs=[
                    InProcessTrainJob(
                        user_id=job.user_id,
                        baseline_desired_retention=job.baseline_desired_retention,
                        baseline_desired_retention_token=(
                            job.baseline_desired_retention_token
                        ),
                        lambda_value=job.lambda_value,
                        lambda_token=job.lambda_token,
                        output_dir=job.output_dir,
                        command_record_path=job.command_record_path,
                        stdout_path=job.stdout_path,
                        stderr_path=job.stderr_path,
                    )
                    for job in batch_jobs
                ],
                config=config,
                config_path=config_path,
                repo_root=repo_root,
            )
        except Exception as exc:  # noqa: BLE001 - preserve runner summary on trainer errors.
            error_note = f"In-process training batch {batch_index} failed: {exc}"
            notes.append(error_note)

        finished_at = utc_timestamp()
        outcome_gate = _train_outcome_gate_summary(outcomes)
        batch_gate_passed = bool(outcome_gate["passed"])
        batch_exit_code = 1 if error_note else 0
        if outcomes and batch_gate_passed:
            batch_runs_succeeded += 1
        elif outcomes:
            batch_exit_code = 1
        max_effective_lanes = max(
            max_effective_lanes,
            len(batch_jobs) * lanes_per_job,
        )
        _write_json(
            batch_record_path,
            {
                "type": "in-process-training-batch",
                "execution_mode": "in_process_batch",
                "trainer": trainer,
                "batch_index": batch_index,
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_code": batch_exit_code,
                "error": error_note,
                "user_ids": sorted({job.user_id for job in batch_jobs}),
                "job_count": len(batch_jobs),
                "lanes_per_job_estimate": lanes_per_job,
                "effective_lanes_estimate": len(batch_jobs) * lanes_per_job,
                "overfit_gate": outcome_gate,
                "outcomes": [
                    _in_process_outcome_record(outcome) for outcome in outcomes
                ],
            },
        )
        outcome_status_by_key = {
            (
                outcome.job.user_id,
                outcome.job.lambda_value,
                outcome.job.baseline_desired_retention,
            ): outcome
            for outcome in outcomes
        }
        for job in batch_jobs:
            outcome = outcome_status_by_key.get(
                (job.user_id, job.lambda_value, job.baseline_desired_retention)
            )
            _write_json(
                job.command_record_path,
                _in_process_job_record(
                    job=job,
                    outcome=outcome,
                    batch_gate_passed=batch_gate_passed,
                    outcome_gate=outcome_gate,
                    trainer=trainer,
                    batch_index=batch_index,
                    batch_record_path=batch_record_path,
                    started_at=started_at,
                    finished_at=finished_at,
                ),
            )

        outcome_by_key = {
            (
                outcome.job.user_id,
                outcome.job.lambda_value,
                outcome.job.baseline_desired_retention,
            ): outcome
            for outcome in outcomes
        }
        for job in batch_jobs:
            outcome = outcome_by_key.get(
                (job.user_id, job.lambda_value, job.baseline_desired_retention)
            )
            if outcome is None:
                result = _finalize_train_job_result(
                    job=job,
                    config=config,
                    exit_code=1,
                    timed_out=False,
                    command_record_path=batch_record_path,
                    stdout_path=None,
                    stderr_path=None,
                    command_result_extra={
                        "execution_mode": "in_process_batch",
                        "batch_index": batch_index,
                        "batch_record_path": str(batch_record_path),
                        "trainer": trainer,
                    },
                    failure_verb="In-process training",
                )
            else:
                result = _finalize_train_job_result(
                    job=job,
                    config=config,
                    exit_code=0 if outcome.passed or batch_gate_passed else 1,
                    timed_out=False,
                    command_record_path=batch_record_path,
                    stdout_path=None,
                    stderr_path=None,
                    command_result_extra={
                        "execution_mode": "in_process_batch",
                        "batch_index": batch_index,
                        "batch_record_path": str(batch_record_path),
                        "trainer": trainer,
                        "overfit_gate_passed": outcome.passed,
                        "batch_overfit_gate": outcome_gate,
                        "artifact_paths_reported": [
                            str(path) for path in outcome.artifact_paths
                        ],
                    },
                    failure_verb="In-process training",
                )
            results.append(result)

        if any(result["failure"] is not None for result in results[-len(batch_jobs) :]):
            break
        timeout = config.performance.timeout_seconds
        if timeout is not None and time.monotonic() - elapsed_started > timeout:
            notes.append(
                "In-process training exceeded performance.timeout_seconds after "
                f"batch {batch_index}; stopping before the next batch."
            )
            break
    return (
        results,
        batch_runs_attempted,
        batch_runs_succeeded,
        max_effective_lanes,
        trainer,
        notes,
    )


def _train_outcome_gate_summary(
    outcomes: Sequence[Any],
) -> dict[str, float | int | bool]:
    total_count = len(outcomes)
    passed_count = sum(1 for outcome in outcomes if outcome.passed)
    required_count = (
        max(
            1,
            math.ceil(total_count * TRAIN_OVERFIT_GATE_PASS_FRACTION - 1e-12),
        )
        if total_count
        else 0
    )
    pass_fraction = passed_count / total_count if total_count else 0.0
    return {
        "pass_fraction_required": TRAIN_OVERFIT_GATE_PASS_FRACTION,
        "points": total_count,
        "passed_points": passed_count,
        "required_passed_points": required_count,
        "pass_fraction": pass_fraction,
        "passed": total_count > 0 and passed_count >= required_count,
    }


def _in_process_outcome_record(outcome: Any) -> dict[str, Any]:
    record = {
        "user_id": outcome.job.user_id,
        "baseline_desired_retention": outcome.job.baseline_desired_retention,
        "passed": outcome.passed,
        "artifact_paths": [str(path) for path in outcome.artifact_paths],
        "progress_path": str(outcome.progress_path)
        if outcome.progress_path is not None
        else None,
        "error": outcome.error,
    }
    if outcome.job.lambda_value is not None:
        record["lambda_value"] = outcome.job.lambda_value
    return record


def _in_process_job_record(
    *,
    job: TrainCommandJob,
    outcome: Any | None,
    batch_gate_passed: bool,
    outcome_gate: dict[str, Any],
    trainer: str,
    batch_index: int,
    batch_record_path: Path,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    record = {
        "type": "in-process-training-job",
        "execution_mode": "in_process_batch",
        "trainer": trainer,
        "batch_index": batch_index,
        "batch_record_path": str(batch_record_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": (
            0 if outcome is not None and (outcome.passed or batch_gate_passed) else 1
        ),
        "user_id": job.user_id,
        "baseline_desired_retention": job.baseline_desired_retention,
        "output_dir": str(job.output_dir),
        "overfit_gate_passed": outcome.passed if outcome is not None else False,
        "batch_overfit_gate": outcome_gate,
    }
    if job.lambda_value is not None:
        record["lambda_value"] = job.lambda_value
    return record


def _build_train_user_batches(
    *,
    jobs: list[TrainCommandJob],
    batch_size: int | None,
    max_lanes_per_batch: int | None,
    lanes_per_job: int,
) -> list[list[TrainCommandJob]]:
    jobs_by_user: dict[int, list[TrainCommandJob]] = {}
    for job in jobs:
        jobs_by_user.setdefault(job.user_id, []).append(job)
    user_ids = list(jobs_by_user)
    if batch_size is not None:
        return [
            [
                job
                for user_id in user_ids[index : index + batch_size]
                for job in jobs_by_user[user_id]
            ]
            for index in range(0, len(user_ids), batch_size)
        ]
    if max_lanes_per_batch is None:
        return [[job for user_id in user_ids for job in jobs_by_user[user_id]]]

    batches: list[list[TrainCommandJob]] = []
    current: list[TrainCommandJob] = []
    current_lanes = 0
    for user_id in user_ids:
        user_jobs = jobs_by_user[user_id]
        user_lanes = len(user_jobs) * lanes_per_job
        if current and current_lanes + user_lanes > max_lanes_per_batch:
            batches.append(current)
            current = []
            current_lanes = 0
        current.extend(user_jobs)
        current_lanes += user_lanes
    if current:
        batches.append(current)
    return batches


def _build_sweep_artifact_lane(
    *,
    metadata_path: Path,
    metadata: SchedulerArtifactMetadata,
    user_id: int,
    outputs_root: Path,
) -> SweepBatchLane:
    lambda_value = metadata.lambda_value
    allows_lambda_none = action_space_allows_lambda_none(metadata.action_space)
    if lambda_value is None and not allows_lambda_none:
        raise ValueError(
            f"Invalid scheduler artifact metadata {metadata_path}: "
            "lambda_value is required for sweep."
        )
    lambda_token = (
        _format_lambda_token(lambda_value) if lambda_value is not None else None
    )
    baseline_dr = metadata.baseline_desired_retention
    if baseline_dr is not None:
        assert lambda_token is not None
        baseline_dr_token = _format_retention_token(baseline_dr)
        output_dir = (
            outputs_root
            / f"user_{user_id}"
            / f"sched_{metadata.scheduler_name}"
            / f"dr_{baseline_dr_token}"
            / f"lambda_{lambda_token}"
        )
    else:
        baseline_dr_token = None
        output_dir = (
            outputs_root / f"user_{user_id}" / f"sched_{metadata.scheduler_name}"
        )
        if allows_lambda_none:
            output_dir = output_dir / metadata.policy_path.parent.name
        else:
            assert lambda_token is not None
            output_dir = output_dir / f"lambda_{lambda_token}"
    return SweepBatchLane(
        source="artifact",
        metadata_path=metadata_path,
        metadata=metadata,
        user_id=user_id,
        scheduler_name=metadata.scheduler_name,
        scheduler_spec=metadata.scheduler_name,
        desired_retention=None,
        fixed_interval=None,
        fsrs6_adr_policy_path=metadata.policy_path,
        lambda_value=lambda_value,
        lambda_token=lambda_token,
        baseline_desired_retention=baseline_dr,
        baseline_desired_retention_token=baseline_dr_token,
        output_dir=output_dir,
    )


def _build_sweep_baseline_lanes(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    outputs_root: Path,
) -> list[SweepBatchLane]:
    if config.baseline.scheduler != "fsrs6":
        return []
    fallback = (
        config.baseline.desired_retention_values
        or _training_baseline_desired_retention_values(config)
    )
    dr_values_by_user = _baseline_dr_values_by_user(
        config=config,
        repo_root=repo_root,
        user_ids=config.users.train,
        fallback=fallback,
    )
    lanes: list[SweepBatchLane] = []
    for user_id in config.users.train:
        for desired_retention in dr_values_by_user[user_id]:
            dr_token = _format_retention_token(desired_retention)
            lanes.append(
                SweepBatchLane(
                    source="baseline",
                    metadata_path=None,
                    metadata=None,
                    user_id=user_id,
                    scheduler_name=config.baseline.scheduler,
                    scheduler_spec=config.baseline.scheduler,
                    desired_retention=desired_retention,
                    fixed_interval=None,
                    fsrs6_adr_policy_path=None,
                    lambda_value=None,
                    lambda_token=None,
                    baseline_desired_retention=desired_retention,
                    baseline_desired_retention_token=dr_token,
                    output_dir=(
                        outputs_root
                        / f"user_{user_id}"
                        / f"sched_{config.baseline.scheduler}"
                        / f"dr_{dr_token}"
                    ),
                )
            )
    return lanes


def _sweep_batch_lane_result(job: SweepBatchLane) -> dict[str, Any]:
    return {
        "source": job.source,
        "artifact_metadata_path": str(job.metadata_path)
        if job.metadata_path is not None
        else None,
        "artifact_id": job.metadata.artifact_id if job.metadata is not None else None,
        "user_id": job.user_id,
        "scheduler": job.scheduler_name,
        "scheduler_spec": job.scheduler_spec,
        "desired_retention": job.desired_retention,
        "fixed_interval": job.fixed_interval,
        "fsrs6_adr_policy": str(job.fsrs6_adr_policy_path)
        if job.fsrs6_adr_policy_path is not None
        else None,
        "baseline_desired_retention": job.baseline_desired_retention,
        "baseline_desired_retention_token": job.baseline_desired_retention_token,
        "lambda_value": job.lambda_value,
        "lambda_token": job.lambda_token,
        "output_dir": str(job.output_dir),
        "command_record_path": None,
        "stdout_path": None,
        "stderr_path": None,
        "exit_code": 0,
        "timed_out": False,
        "execution_mode": "batched-in-process",
    }


def _same_fsrs6_adr_policy_shape(lhs: Any, rhs: Any) -> bool:
    lhs_bounds = lhs.bounds
    rhs_bounds = rhs.bounds
    return (
        lhs.feature_version == rhs.feature_version
        and math.isclose(
            lhs.retention_min,
            rhs.retention_min,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            lhs.retention_max,
            rhs.retention_max,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(lhs_bounds.s_min, rhs_bounds.s_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs_bounds.s_max, rhs_bounds.s_max, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs_bounds.d_min, rhs_bounds.d_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs_bounds.d_max, rhs_bounds.d_max, rel_tol=0.0, abs_tol=1e-9)
    )


def _run_batched_sweep_jobs(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    run_id: str,
    jobs: Sequence[SweepBatchLane],
    record_path: Path,
) -> None:
    if not jobs:
        raise ValueError("No sweep lanes were provided for batched sweep.")
    if config.simulation.engine != "batched":
        raise ValueError("Batched sweep requires simulation.engine = 'batched'.")
    if config.simulation.environment not in {"fsrs6", "fsrs6_default"}:
        raise ValueError(
            "Batched sweep currently supports fsrs6 or fsrs6_default environments."
        )
    unsupported = sorted(
        {
            job.scheduler_name
            for job in jobs
            if job.scheduler_name not in {"fsrs6", *FSRS6_ADR_POLICY_SOURCE_SCHEDULERS}
        }
    )
    if unsupported:
        raise ValueError(
            "Batched sweep currently supports fsrs6 and FSRS6 ADR-family lanes, "
            f"got {unsupported}."
        )
    for job in jobs:
        if job.scheduler_name == "fsrs6" and job.desired_retention is None:
            raise ValueError("Batched fsrs6 sweep lanes require desired_retention.")
        if (
            job.scheduler_name in FSRS6_ADR_POLICY_SOURCE_SCHEDULERS
            and job.fsrs6_adr_policy_path is None
        ):
            raise ValueError(
                f"Batched {job.scheduler_name} sweep lanes require a policy path."
            )

    import argparse

    import torch

    import simulate as simulate_cli
    from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
    from simulator.batched_sweep.weights import (
        build_default_fsrs6_weights,
        load_fsrs6_weights,
    )
    from simulator.benchmark_loader import (
        parse_result_overrides,
        resolve_benchmark_root,
    )
    from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
    from simulator.defaults import (
        DEFAULT_COST_LIMIT_MINUTES,
        DEFAULT_LEARN_LIMIT,
        DEFAULT_REVIEW_LIMIT,
        DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    )
    from simulator.math.fsrs import Bounds
    from simulator.models.fsrs import FSRS6BatchEnvOps
    from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
    from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
    from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps
    from simulator.short_term_config import resolve_short_term_config
    from simulator.batched_engine.multiuser_engine import simulate_multiuser

    device_name = _resolve_performance_device(config)
    torch_device = config.training_policy_search.get("torch_device")
    if isinstance(torch_device, str) and torch_device.strip():
        device_name = torch_device.strip()
    device = torch.device(device_name)

    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=config.training_policy_search.get("learning_steps"),
        relearning_steps=config.training_policy_search.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    learning_steps_arg = (
        ",".join(str(step) for step in learning_steps)
        if short_term_source == "steps"
        else None
    )
    relearning_steps_arg = (
        ",".join(str(step) for step in relearning_steps)
        if short_term_source == "steps"
        else None
    )

    lane_user_ids = [job.user_id for job in jobs]
    lane_index_by_scheduler: dict[str, list[int]] = {}
    for lane_index, job in enumerate(jobs):
        lane_index_by_scheduler.setdefault(job.scheduler_name, []).append(lane_index)
    scheduler_names = set(lane_index_by_scheduler)
    fit_fsrs_adr_schedulers = FSRS6_ADR_POLICY_SOURCE_SCHEDULERS - {"fsrs6_default_adr"}
    benchmark_root = resolve_benchmark_root(repo_root, None).resolve()
    needs_fit_fsrs_weights = (
        config.simulation.environment == "fsrs6"
        or "fsrs6" in scheduler_names
        or bool(fit_fsrs_adr_schedulers & scheduler_names)
    )
    fsrs_weights: torch.Tensor | None = None
    if needs_fit_fsrs_weights:
        fsrs_weights, active_user_ids = load_fsrs6_weights(
            repo_root=repo_root,
            user_ids=lane_user_ids,
            benchmark_root=benchmark_root,
            benchmark_partition=None,
            overrides=parse_result_overrides(None),
            short_term=bool(short_term_source),
            device=device,
        )
        if fsrs_weights is None or active_user_ids != lane_user_ids:
            raise ValueError(
                "Batched sweep could not load FSRS-6 weights for all policy lanes."
            )

    if config.simulation.environment == "fsrs6":
        if fsrs_weights is None:
            raise ValueError("Batched fsrs6 environment requires FSRS-6 weights.")
        env_weights = fsrs_weights.to(device)
    else:
        env_weights = build_default_fsrs6_weights(
            user_ids=lane_user_ids,
            device=device,
        )
    env_ops = FSRS6BatchEnvOps(
        weights=env_weights,
        bounds=Bounds(),
        device=device,
        dtype=torch.float32,
    )

    scheduler_groups: list[_MixedSchedulerGroup] = []

    if fsrs6_indices := lane_index_by_scheduler.get("fsrs6"):
        if fsrs_weights is None:
            raise ValueError("Batched fsrs6 scheduler requires FSRS-6 weights.")
        fsrs_weights = fsrs_weights.to(device)
        group_indices = torch.tensor(
            fsrs6_indices,
            device=device,
            dtype=torch.int64,
        )
        fsrs6_desired_retentions: list[float] = []
        for index in fsrs6_indices:
            desired = jobs[index].desired_retention
            if desired is None:
                raise ValueError("Batched fsrs6 sweep lanes require desired_retention.")
            fsrs6_desired_retentions.append(desired)
        desired_retention = torch.tensor(
            fsrs6_desired_retentions,
            device=device,
            dtype=torch.float32,
        )
        scheduler_groups.append(
            _MixedSchedulerGroup(
                lane_indices=group_indices,
                ops=FSRS6BatchSchedulerOps(
                    weights=fsrs_weights.index_select(0, group_indices),
                    desired_retention=desired_retention,
                    bounds=Bounds(),
                    priority_mode=config.simulation.scheduler_priority,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

    for adr_scheduler_name in sorted(FSRS6_ADR_POLICY_SOURCE_SCHEDULERS):
        if not (fsrs6_adr_indices := lane_index_by_scheduler.get(adr_scheduler_name)):
            continue
        group_indices = torch.tensor(
            fsrs6_adr_indices,
            device=device,
            dtype=torch.int64,
        )
        adr_policy_paths: list[Path] = []
        for index in fsrs6_adr_indices:
            policy_path = jobs[index].fsrs6_adr_policy_path
            if policy_path is None:
                raise ValueError(
                    f"Batched {adr_scheduler_name} sweep lanes require a policy path."
                )
            adr_policy_paths.append(policy_path)
        policies = [FSRS6ADRPolicy.from_json(path) for path in adr_policy_paths]
        template = policies[0]
        for policy in policies[1:]:
            if not _same_fsrs6_adr_policy_shape(policy, template):
                raise ValueError(
                    f"Batched {adr_scheduler_name} sweep requires identical policy "
                    "feature version, retention bounds, and FSRS bounds."
                )
        coefficients = torch.tensor(
            [policy.coefficients for policy in policies],
            device=device,
            dtype=torch.float32,
        )
        if adr_scheduler_name == "fsrs6_default_adr":
            scheduler_weights = build_default_fsrs6_weights(
                user_ids=lane_user_ids,
                device=device,
            )
        else:
            if fsrs_weights is None:
                raise ValueError(
                    f"Batched {adr_scheduler_name} scheduler requires FSRS-6 weights."
                )
            scheduler_weights = fsrs_weights.to(device)
        scheduler_groups.append(
            _MixedSchedulerGroup(
                lane_indices=group_indices,
                ops=FSRS6ADRBatchSchedulerOps(
                    weights=scheduler_weights.index_select(0, group_indices),
                    policy=template,
                    coefficients=coefficients,
                    bounds=Bounds(),
                    priority_mode=config.simulation.scheduler_priority,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

    sched_ops = _MixedBatchSchedulerOps(
        groups=scheduler_groups,
        lane_count=len(jobs),
        device=device,
        dtype=torch.float32,
    )

    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
        review_markov_success_weights,
    ) = load_usage(
        lane_user_ids,
        DEFAULT_BUTTON_USAGE_PATH,
        review_markov_transition=config.simulation.review_markov_transition,
    )
    review_markov_success_weights = (
        review_markov_success_weights.to(device)
        if review_markov_success_weights is not None
        else None
    )
    learn_limit = (
        config.simulation.learn_limit
        if config.simulation.learn_limit is not None
        else DEFAULT_LEARN_LIMIT
    )
    review_limit = (
        config.simulation.review_limit
        if config.simulation.review_limit is not None
        else DEFAULT_REVIEW_LIMIT
    )
    cost_limit_minutes = (
        config.simulation.cost_limit_minutes
        if config.simulation.cost_limit_minutes is not None
        else DEFAULT_COST_LIMIT_MINUTES
    )
    behavior, cost_model = build_behavior_cost(
        len(lane_user_ids),
        deck_size=config.simulation.deck,
        learn_limit=learn_limit,
        review_limit=review_limit,
        cost_limit_minutes=cost_limit_minutes,
        learn_costs=learn_costs.to(device),
        review_costs=review_costs.to(device),
        first_rating_prob=first_rating_prob.to(device),
        review_rating_prob=review_rating_prob.to(device),
        learning_rating_prob=learning_rating_prob.to(device),
        relearning_rating_prob=relearning_rating_prob.to(device),
        state_rating_costs=state_rating_costs.to(device),
        review_markov_success_weights=review_markov_success_weights,
        short_term=bool(short_term_source),
    )

    short_term_threshold = _training_policy_search_float(
        config,
        "short_term_threshold",
        0.5,
    )
    short_term_loops_limit = _training_policy_search_int(
        config,
        "short_term_loops_limit",
        DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    )
    stats_list = simulate_multiuser(
        days=config.simulation.days,
        deck_size=config.simulation.deck,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        seed=config.seed,
        device=device,
        dtype=torch.float32,
        fuzz=config.simulation.fuzz,
        priority_mode=config.simulation.priority,
        progress=False,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        short_term_threshold=short_term_threshold,
        short_term_loops_limit=short_term_loops_limit,
    )

    for job, stats in zip(jobs, stats_list, strict=True):
        user_log_dir = job.output_dir / f"user_{job.user_id}"
        log_args = argparse.Namespace(
            engine="batched",
            days=config.simulation.days,
            deck=config.simulation.deck,
            learn_limit=learn_limit,
            review_limit=review_limit,
            cost_limit_minutes=cost_limit_minutes,
            priority=config.simulation.priority,
            environment=config.simulation.environment,
            scheduler=job.scheduler_name,
            scheduler_spec=job.scheduler_spec,
            run_id=run_id,
            user_id=job.user_id,
            button_usage=str(DEFAULT_BUTTON_USAGE_PATH),
            review_markov_transition=config.simulation.review_markov_transition,
            desired_retention=job.desired_retention,
            scheduler_priority=config.simulation.scheduler_priority,
            sspmmc_policy=None,
            fsrs6_adr_policy=job.fsrs6_adr_policy_path,
            fixed_interval=job.fixed_interval,
            seed=config.seed,
            fuzz=config.simulation.fuzz,
            short_term_source=short_term_source,
            learning_steps=learning_steps_arg,
            relearning_steps=relearning_steps_arg,
            short_term_threshold=short_term_threshold,
            short_term_loops_limit=short_term_loops_limit,
            log_dir=user_log_dir,
            log_reviews=False,
            write_daily_csv=config.performance.diagnostic_csv_logs,
        )
        simulate_cli._write_log(log_args, stats)

    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        record_path,
        {
            "type": "batched-sweep-record",
            "batch_lane_count": len(jobs),
            "scheduler_names": sorted({job.scheduler_name for job in jobs}),
            "environment": config.simulation.environment,
            "engine": config.simulation.engine,
            "review_markov_transition": (config.simulation.review_markov_transition),
            "device": str(device),
            "seed": config.seed,
            "artifact_metadata_paths": [
                str(job.metadata_path) for job in jobs if job.metadata_path is not None
            ],
            "output_dirs": [str(job.output_dir) for job in jobs],
            "lanes": [_sweep_batch_lane_result(job) for job in jobs],
        },
    )


def _run_configured_batched_retention_sweep(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    run_id: str,
    output_root: Path,
    log_dir: Path,
    record_path: Path,
) -> dict[str, Any]:
    import argparse

    from simulator.batched_sweep.execution import run_batches
    from simulator.batched_sweep.plan import build_batched_sweep_plan
    from simulator.batched_sweep.runner import _build_sweep_lanes
    from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
    from simulator.defaults import (
        DEFAULT_COST_LIMIT_MINUTES,
        DEFAULT_LEARN_LIMIT,
        DEFAULT_REVIEW_LIMIT,
        DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    )
    from simulator.scheduler_spec import parse_scheduler_spec

    sweep_config = config.sweep_batched
    run_root = output_root / run_id
    scheduler_names = {parse_scheduler_spec(raw)[0] for raw in sweep_config.schedulers}
    args = argparse.Namespace(
        config=None,
        user_ids=list(config.users.train),
        start_user=min(config.users.train),
        end_user=max(config.users.train),
        batch_size=sweep_config.batch_size,
        max_lanes_per_batch=sweep_config.max_lanes_per_batch,
        env_batch_overrides=sweep_config.env_overrides,
        torch_device=sweep_config.torch_device,
        cuda_devices=sweep_config.cuda_devices,
        srs_benchmark_root=None,
        benchmark_result=None,
        benchmark_partition=sweep_config.benchmark_partition,
        log_dir=log_dir,
        log_layout=sweep_config.log_layout,
        run_id=run_id,
        start_retention=sweep_config.start_retention,
        end_retention=sweep_config.end_retention,
        step=sweep_config.step,
        days=config.simulation.days,
        deck=config.simulation.deck,
        learn_limit=config.simulation.learn_limit
        if config.simulation.learn_limit is not None
        else DEFAULT_LEARN_LIMIT,
        review_limit=config.simulation.review_limit
        if config.simulation.review_limit is not None
        else DEFAULT_REVIEW_LIMIT,
        cost_limit_minutes=config.simulation.cost_limit_minutes
        if config.simulation.cost_limit_minutes is not None
        else DEFAULT_COST_LIMIT_MINUTES,
        seed=config.seed,
        priority=config.simulation.priority,
        scheduler_priority=config.simulation.scheduler_priority,
        button_usage=DEFAULT_BUTTON_USAGE_PATH,
        review_markov_transition=config.simulation.review_markov_transition,
        no_log=sweep_config.no_log,
        no_progress=sweep_config.no_progress,
        diagnostic_csv_logs=config.performance.diagnostic_csv_logs,
        fuzz=config.simulation.fuzz,
        short_term_source=config.simulation.short_term_source,
        learning_steps=config.training_policy_search.get("learning_steps"),
        relearning_steps=config.training_policy_search.get("relearning_steps"),
        short_term_threshold=_training_policy_search_float(
            config,
            "short_term_threshold",
            0.5,
        ),
        short_term_loops_limit=_training_policy_search_int(
            config,
            "short_term_loops_limit",
            DEFAULT_SHORT_TERM_LOOPS_LIMIT,
        ),
        fsrs6_adr_policy=None,
        fsrs6_adr_policy_root=None,
        fsrs6_adr_train_run_root=run_root
        if scheduler_names & FSRS6_ADR_POLICY_SOURCE_SCHEDULERS
        else None,
        fsrs6_adr_policy_manifest=None,
        fsrs6_adr_lambda_values=_sweep_policy_lambda_values(config)
        if scheduler_names & FSRS6_ADR_POLICY_SOURCE_SCHEDULERS
        else None,
        fsrs6_cost_adr_policy=None,
        fsrs6_cost_adr_policy_root=None,
        fsrs6_cost_adr_train_run_root=run_root
        if scheduler_names & FSRS6_COST_ADR_POLICY_SOURCE_SCHEDULERS
        else None,
        fsrs6_cost_adr_policy_manifest=None,
        fsrs6_cost_adr_cost_weights=sweep_config.fsrs6_cost_adr_cost_weights,
        fsrs6_oracle_stationary_finite_distill_policy=None,
        fsrs6_oracle_stationary_finite_distill_policy_root=None,
        fsrs6_oracle_stationary_finite_distill_train_run_root=run_root
        if "fsrs6_oracle_stationary_finite_distill" in scheduler_names
        else None,
        fsrs6_oracle_stationary_finite_distill_policy_manifest=None,
        fsrs6_ap_policy=None,
        fsrs6_ap_policy_root=None,
        fsrs6_ap_train_run_root=run_root if "fsrs6_ap" in scheduler_names else None,
        fsrs6_ap_policy_manifest=None,
        fsrs6_ap_lambda_values=_sweep_policy_lambda_values(config)
        if "fsrs6_ap" in scheduler_names
        else None,
        anki_sm2_ap_policy=None,
        anki_sm2_ap_policy_root=None,
        anki_sm2_ap_train_run_root=run_root
        if "anki_sm2_ap" in scheduler_names
        else None,
        anki_sm2_ap_policy_manifest=None,
        fsrs3_dr_manifest=_resolve_repo_path(repo_root, sweep_config.fsrs3_dr_manifest)
        if sweep_config.fsrs3_dr_manifest is not None
        else None,
        fsrs6_dr_manifest=_resolve_baseline_dr_manifest_path(
            config=config,
            repo_root=repo_root,
        ),
    )
    plan = build_batched_sweep_plan(
        repo_root=repo_root,
        args=args,
        envs=list(sweep_config.envs),
        schedulers=list(sweep_config.schedulers),
    )
    if plan.total_lanes < 1:
        raise ValueError("Configured batched sweep did not produce any lanes.")

    overall = None
    if not sweep_config.no_progress:
        from tqdm import tqdm

        overall = tqdm(
            total=plan.total_user_days,
            desc="Overall",
            unit="user-day",
            leave=True,
        )
    try:
        run_batches(
            args=args,
            ctx=plan.ctx,
            batches=plan.batches,
            batches_by_env=plan.batches_by_env,
            devices=plan.devices,
            device=plan.device,
            overall=overall,
        )
    finally:
        if overall is not None:
            overall.close()

    lanes = [
        lane
        for environment in plan.ctx.envs
        for batch in plan.batches_by_env.get(environment, plan.batches)
        for lane in _build_sweep_lanes(
            batch=batch,
            ctx=plan.ctx,
            environment=environment,
        )
    ]
    log_paths: list[Path] = []
    lane_results: list[dict[str, Any]] = []
    for lane in lanes:
        matched_logs, log_note = _collect_sweep_logs(
            output_dir=lane.final_log_dir,
            log_glob=config.sweep_log_glob,
        )
        if log_note is not None:
            raise ValueError(log_note)
        matched_logs = _filter_batched_retention_lane_logs(
            log_paths=matched_logs,
            config=config,
            run_id=run_id,
            lane=lane,
        )
        if not matched_logs:
            raise ValueError(
                f"No sweep JSONL logs matched {config.sweep_log_glob!r} "
                f"in {lane.final_log_dir}."
            )
        validation_note = _validate_batched_retention_lane_logs(
            log_paths=matched_logs,
            config=config,
            lane=lane,
        )
        if validation_note is not None:
            raise ValueError(validation_note)
        log_paths.extend(matched_logs)
        lane_results.append(
            {
                "source": "configured-batched-retention",
                "environment": lane.environment,
                "user_id": lane.user_id,
                "scheduler": lane.scheduler_name,
                "scheduler_spec": lane.scheduler_spec,
                "desired_retention": lane.desired_retention,
                "fixed_interval": lane.fixed_interval,
                "fsrs6_adr_policy": str(lane.fsrs6_adr_policy)
                if lane.fsrs6_adr_policy is not None
                else None,
                "fsrs6_adr_baseline_desired_retention": (
                    lane.fsrs6_adr_baseline_desired_retention
                ),
                "fsrs6_adr_lambda_value": lane.fsrs6_adr_lambda_value,
                "fsrs6_cost_adr_policy": str(lane.fsrs6_cost_adr_policy)
                if lane.fsrs6_cost_adr_policy is not None
                else None,
                "fsrs6_cost_adr_goal_cost_weight": (
                    lane.fsrs6_cost_adr_goal_cost_weight
                ),
                "fsrs6_oracle_stationary_finite_distill_policy": str(
                    lane.fsrs6_oracle_stationary_finite_distill_policy
                )
                if lane.fsrs6_oracle_stationary_finite_distill_policy is not None
                else None,
                "fsrs6_oracle_stationary_finite_distill_goal_cost_weight": (
                    lane.fsrs6_oracle_stationary_finite_distill_goal_cost_weight
                ),
                "fsrs6_ap_policy": str(lane.fsrs6_ap_policy)
                if lane.fsrs6_ap_policy is not None
                else None,
                "fsrs6_ap_baseline_desired_retention": (
                    lane.fsrs6_ap_baseline_desired_retention
                ),
                "fsrs6_ap_lambda_value": lane.fsrs6_ap_lambda_value,
                "anki_sm2_ap_policy": str(lane.anki_sm2_ap_policy)
                if lane.anki_sm2_ap_policy is not None
                else None,
                "output_dir": str(lane.final_log_dir),
                "log_paths": [str(path) for path in matched_logs],
                "exit_code": 0,
                "timed_out": False,
                "execution_mode": "batched-retention-config",
            }
        )

    record_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        record_path,
        {
            "type": "batched-retention-sweep-record",
            "run_id": run_id,
            "batch_lane_count": len(lanes),
            "total_lanes": plan.total_lanes,
            "total_user_days": plan.total_user_days,
            "batches": plan.batches,
            "batches_by_env": plan.batches_by_env,
            "envs": list(plan.ctx.envs),
            "schedulers": list(plan.ctx.schedulers),
            "log_root": str(plan.ctx.log_root),
            "log_layout": plan.ctx.log_layout,
            "device": str(plan.device) if plan.device is not None else None,
            "devices": plan.devices,
            "seed": config.seed,
            "lanes": lane_results,
        },
    )
    return {
        "batch_lane_count": len(lanes),
        "log_paths": log_paths,
        "lane_results": lane_results,
    }


def _batched_retention_lane_filename_filter(
    *,
    config: ExperimentConfig,
    run_id: str | None,
    lane: Any,
) -> LogFilenameFilter:
    short_term = "on" if config.simulation.short_term_source else "off"
    short_term_source = config.simulation.short_term_source or "any"
    retention_values_by_scheduler = None
    start_retention = None
    end_retention = None
    if lane.desired_retention is not None:
        start_retention, end_retention = _filename_retention_bounds(
            (float(lane.desired_retention),)
        )
        retention_values_by_scheduler = {
            lane.scheduler_name: round(float(lane.desired_retention), 2)
        }
    return LogFilenameFilter(
        envs=[lane.environment],
        scheds=[lane.scheduler_name],
        engine="batched",
        short_term=short_term,
        short_term_source=short_term_source,
        seed=config.seed,
        run_id=run_id
        if lane.scheduler_name in RUN_ID_SCOPED_SWEEP_SCHEDULERS
        else None,
        start_retention=start_retention,
        end_retention=end_retention,
        priority=config.simulation.priority,
        retention_values_by_scheduler=retention_values_by_scheduler,
    )


def _filter_batched_retention_lane_logs(
    *,
    log_paths: Sequence[Path],
    config: ExperimentConfig,
    run_id: str | None,
    lane: Any,
) -> list[Path]:
    filename_filter = _batched_retention_lane_filename_filter(
        config=config,
        run_id=run_id,
        lane=lane,
    )
    return [path for path in log_paths if filename_filter.matches(path.name)]


def _validate_batched_retention_lane_logs(
    *,
    log_paths: Sequence[Path],
    config: ExperimentConfig,
    lane: Any,
) -> str | None:
    for path in log_paths:
        records = _read_log_meta_and_totals(path)
        if records is None:
            return f"Sweep log is missing meta or totals record: {path}"
        meta, _ = records
        errors = _simulation_metadata_errors(
            config=config,
            meta=meta,
            expected_engine="batched",
            expected_scheduler=lane.scheduler_name,
            expected_user_id=lane.user_id,
            expected_environment=lane.environment,
        )
        if lane.desired_retention is not None:
            actual_retention = meta.get("desired_retention")
            if not isinstance(actual_retention, (float, int)) or not math.isclose(
                float(actual_retention),
                lane.desired_retention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "metadata desired_retention expected "
                    f"{lane.desired_retention!r}, got {actual_retention!r}"
                )
        if lane.fsrs6_adr_baseline_desired_retention is not None:
            actual_baseline_dr = meta.get("fsrs6_adr_baseline_desired_retention")
            if not isinstance(actual_baseline_dr, (float, int)) or not math.isclose(
                float(actual_baseline_dr),
                lane.fsrs6_adr_baseline_desired_retention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "metadata fsrs6_adr_baseline_desired_retention expected "
                    f"{lane.fsrs6_adr_baseline_desired_retention!r}, "
                    f"got {actual_baseline_dr!r}"
                )
        if lane.fsrs6_ap_baseline_desired_retention is not None:
            actual_baseline_dr = meta.get("fsrs6_ap_baseline_desired_retention")
            if not isinstance(actual_baseline_dr, (float, int)) or not math.isclose(
                float(actual_baseline_dr),
                lane.fsrs6_ap_baseline_desired_retention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "metadata fsrs6_ap_baseline_desired_retention expected "
                    f"{lane.fsrs6_ap_baseline_desired_retention!r}, "
                    f"got {actual_baseline_dr!r}"
                )
        if lane.fsrs6_cost_adr_policy is not None:
            actual_policy = meta.get("fsrs6_cost_adr_policy")
            if actual_policy != str(lane.fsrs6_cost_adr_policy):
                errors.append(
                    "metadata fsrs6_cost_adr_policy expected "
                    f"{lane.fsrs6_cost_adr_policy!s}, got {actual_policy!r}"
                )
        if lane.fsrs6_cost_adr_goal_cost_weight is not None:
            actual_weight = meta.get("fsrs6_cost_adr_goal_cost_weight")
            if not isinstance(actual_weight, (float, int)) or not math.isclose(
                float(actual_weight),
                lane.fsrs6_cost_adr_goal_cost_weight,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "metadata fsrs6_cost_adr_goal_cost_weight expected "
                    f"{lane.fsrs6_cost_adr_goal_cost_weight!r}, "
                    f"got {actual_weight!r}"
                )
        if lane.anki_sm2_ap_policy is not None:
            actual_policy = meta.get("anki_sm2_ap_policy")
            if actual_policy != str(lane.anki_sm2_ap_policy):
                errors.append(
                    "metadata anki_sm2_ap_policy expected "
                    f"{lane.anki_sm2_ap_policy!s}, got {actual_policy!r}"
                )
        if errors:
            return (
                f"Sweep log metadata mismatch for {path} "
                f"(configured batched lane): " + "; ".join(errors)
            )
    return None


def _training_policy_search_float(
    config: ExperimentConfig,
    key: str,
    default: float,
) -> float:
    value = config.training_policy_search.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"training.policy_search.{key} must be a number.")
    return float(value)


def _training_policy_search_int(
    config: ExperimentConfig,
    key: str,
    default: int,
) -> int:
    value = config.training_policy_search.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"training.policy_search.{key} must be an integer.")
    return value


def _format_sweep_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    metadata_path: Path,
    metadata: SchedulerArtifactMetadata,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    user_id = metadata.training_user_ids[0]
    lambda_value = metadata.lambda_value
    lambda_token = (
        _format_lambda_token(lambda_value) if lambda_value is not None else ""
    )
    baseline_dr = metadata.baseline_desired_retention
    values: dict[str, Any] = {
        "artifact_id": metadata.artifact_id,
        "artifact_metadata_path": str(metadata_path),
        "policy_path": str(metadata.policy_path),
        "scheduler_name": metadata.scheduler_name,
        "user_id": user_id,
        "lambda_value": lambda_value if lambda_value is not None else "",
        "lambda_token": lambda_token,
        "baseline_desired_retention": baseline_dr if baseline_dr is not None else "",
        "baseline_desired_retention_token": _format_retention_token(baseline_dr)
        if baseline_dr is not None
        else "",
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.sweep_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _format_pareto_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    baseline_stage_root: Path,
    sweep_stage_root: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    values: dict[str, Any] = {
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "baseline_stage_root": str(baseline_stage_root),
        "baseline_logs_dir": str(baseline_stage_root / "baseline_logs"),
        "sweep_stage_root": str(sweep_stage_root),
        "sweep_outputs_dir": str(sweep_stage_root / "sweep_outputs"),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.pareto_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _format_build_pareto_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    run_root: Path,
    baseline_stage_root: Path,
    sweep_stage_root: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    # Formal experiment runs must not scan the standalone retention-sweep log
    # root from TOML.  The run root contains both stage-baseline/baseline_logs
    # and sweep/sweep_outputs, so build-pareto can compare baseline and learned
    # scheduler outputs without being polluted by stale shared logs.
    log_dir = run_root
    values: dict[str, Any] = {
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "baseline_stage_root": str(baseline_stage_root),
        "baseline_logs_dir": str(baseline_stage_root / "baseline_logs"),
        "sweep_stage_root": str(sweep_stage_root),
        "sweep_outputs_dir": str(sweep_stage_root / "sweep_outputs"),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if config.build_pareto.command_template:
        try:
            return [
                item.format(**values) for item in config.build_pareto.command_template
            ]
        except KeyError as exc:
            raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
        except IndexError as exc:
            raise ValueError("positional format fields are not supported") from exc
    return [
        "uv",
        "run",
        "python",
        "experiments/retention_sweep/build_pareto_users.py",
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
        "--log-dir",
        str(log_dir),
        "--output-dir",
        str(output_dir),
    ]


def _format_analyze_pareto_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    run_root: Path,
    build_stage_root: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    log_dir = _resolve_repo_path(
        repo_root,
        config.analyze_pareto.log_dir or (build_stage_root / "build_pareto_outputs"),
    )
    output_path = output_dir / "analysis.md"
    summary_path = output_dir / "analysis_summary.json"
    values: dict[str, Any] = {
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "log_dir": str(log_dir),
        "build_stage_root": str(build_stage_root),
        "build_pareto_outputs_dir": str(build_stage_root / "build_pareto_outputs"),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    if config.analyze_pareto.command_template:
        try:
            return [
                item.format(**values) for item in config.analyze_pareto.command_template
            ]
        except KeyError as exc:
            raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
        except IndexError as exc:
            raise ValueError("positional format fields are not supported") from exc
    return [
        "uv",
        "run",
        "python",
        "experiments/retention_sweep/analyze_scheduler_comparison.py",
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
        "--log-dir",
        str(log_dir),
        "--output-path",
        str(output_path),
        "--summary-path",
        str(summary_path),
    ]


def _format_select_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    pareto_stage_root: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    values: dict[str, Any] = {
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "run_root": str(stage_root.parent),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "train_stage_root": str(stage_root.parent / StageName.TRAIN_OVERFIT.value),
        "train_summary_path": str(
            stage_root.parent / StageName.TRAIN_OVERFIT.value / "training_summary.json"
        ),
        "sweep_stage_root": str(stage_root.parent / StageName.SWEEP.value),
        "pareto_stage_root": str(pareto_stage_root),
        "pareto_outputs_dir": str(pareto_stage_root / "pareto_outputs"),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.select_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _format_aggregate_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    select_stage_root: Path,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    values: dict[str, Any] = {
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "run_root": str(stage_root.parent),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "train_stage_root": str(stage_root.parent / StageName.TRAIN_OVERFIT.value),
        "sweep_stage_root": str(stage_root.parent / StageName.SWEEP.value),
        "pareto_stage_root": str(stage_root.parent / StageName.PARETO.value),
        "select_stage_root": str(select_stage_root),
        "select_outputs_dir": str(select_stage_root / "select_outputs"),
        "select_summary_path": str(select_stage_root / "select_summary.json"),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.aggregate_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _format_reserved_test_command(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str,
    stage_root: Path,
    output_dir: Path,
    metadata_path: Path,
    metadata: SchedulerArtifactMetadata,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    reserved_user_ids = ",".join(str(user_id) for user_id in config.users.reserved_test)
    values: dict[str, Any] = {
        "artifact_id": metadata.artifact_id,
        "artifact_metadata_path": str(metadata_path),
        "policy_path": str(metadata.policy_path),
        "scheduler_name": metadata.scheduler_name,
        "reserved_user_ids": reserved_user_ids,
        "reserved_user_start": min(config.users.reserved_test),
        "reserved_user_end": max(config.users.reserved_test),
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
        "review_markov_transition": str(
            config.simulation.review_markov_transition
        ).lower(),
        "repo_root": str(repo_root),
        "run_root": str(stage_root.parent),
        "stage_root": str(stage_root),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "config_snapshot_path": str(stage_root / "config_snapshot.toml"),
        "command_record_path": str(command_record_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        return [item.format(**values) for item in config.reserved_test_command_template]
    except KeyError as exc:
        raise ValueError(f"unknown placeholder {{{exc.args[0]}}}") from exc
    except IndexError as exc:
        raise ValueError("positional format fields are not supported") from exc


def _format_lambda_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def _format_retention_token(value: float) -> str:
    return _format_lambda_token(value)


def _resolve_baseline_dr_manifest_path(
    *, config: ExperimentConfig, repo_root: Path
) -> Path | None:
    manifest = config.baseline_dr_selection.manifest
    if manifest is None:
        return None
    return _resolve_repo_path(repo_root, manifest)


def _load_baseline_dr_manifest(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    user_ids: Sequence[int] | None = None,
) -> BaselineDRManifest | None:
    manifest_path = _resolve_baseline_dr_manifest_path(
        config=config,
        repo_root=repo_root,
    )
    if manifest_path is None:
        return None
    return load_baseline_dr_manifest(
        manifest_path,
        target_count=config.baseline_dr_selection.target_count,
        user_ids=user_ids,
        tolerance=config.baseline_dr_selection.tolerance,
    )


def _baseline_dr_values_for_user(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    user_id: int,
    fallback: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    manifest = _load_baseline_dr_manifest(
        config=config,
        repo_root=repo_root,
        user_ids=(user_id,),
    )
    if manifest is not None:
        return manifest.values_for_user(user_id)
    return (
        fallback
        if fallback is not None
        else _training_baseline_desired_retention_values(config)
    )


def _baseline_dr_values_by_user(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    user_ids: Sequence[int],
    fallback: tuple[float, ...] | None = None,
) -> dict[int, tuple[float, ...]]:
    manifest = _load_baseline_dr_manifest(
        config=config,
        repo_root=repo_root,
        user_ids=user_ids,
    )
    if manifest is not None:
        return {
            int(user_id): manifest.values_for_user(int(user_id)) for user_id in user_ids
        }
    values = (
        fallback
        if fallback is not None
        else _training_baseline_desired_retention_values(config)
    )
    return {int(user_id): values for user_id in user_ids}


def _training_baseline_desired_retention_values(
    config: ExperimentConfig,
) -> tuple[float, ...]:
    raw_values = config.training_policy_search.get("baseline_desired_retention_values")
    if raw_values is None:
        raw_single = config.training_policy_search.get(
            "baseline_desired_retention", 0.90
        )
        if isinstance(raw_single, bool) or not isinstance(raw_single, (float, int)):
            return (0.90,)
        return (float(raw_single),)
    if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
        return ()
    values: list[float] = []
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (float, int)):
            return ()
        values.append(float(raw_value))
    if len(set(values)) != len(values):
        return ()
    return tuple(values)


def _training_uses_baseline_dr_grid(config: ExperimentConfig) -> bool:
    return "baseline_desired_retention_values" in config.training_policy_search


def _training_batches_baseline_dr_grid(config: ExperimentConfig) -> bool:
    return (
        config.train_batch_baseline_desired_retention_values
        and _training_uses_baseline_dr_grid(config)
    )


def _training_uses_portfolio_trainer(config: ExperimentConfig) -> bool:
    if config.training_portfolio and not config.training_optimizer:
        return True
    if config.training_batch.trainer in {
        "fsrs6_cost_adr_cmaes",
        "fsrs6_adr_portfolio",
        "fsrs6_oracle_stationary_finite_distill_portfolio",
        "fsrs6_ap_portfolio",
        "anki_sm2_ap_portfolio",
    }:
        return True
    script_names = {Path(item).name for item in config.train_command_template}
    return bool(
        {
            "train_cmaes_fsrs6_cost_adr.py",
            "train_fsrs6_adr_portfolio.py",
            "train_fsrs6_oracle_stationary_finite_distill_portfolio.py",
            "train_fsrs6_ap_portfolio.py",
            "train_anki_sm2_ap_portfolio.py",
        }
        & script_names
    )


def _training_lambda_values_for_jobs(
    config: ExperimentConfig,
) -> tuple[float | None, ...]:
    if _training_uses_portfolio_trainer(config):
        return (None,)
    return tuple(config.lambda_grid)


def _training_effective_lambda_count(config: ExperimentConfig) -> int:
    return len(_training_lambda_values_for_jobs(config))


def _training_reported_lambda_count(config: ExperimentConfig) -> int:
    return 0 if _training_uses_portfolio_trainer(config) else len(config.lambda_grid)


def _sweep_policy_lambda_values(config: ExperimentConfig) -> tuple[float, ...] | None:
    if _training_uses_portfolio_trainer(config):
        return None
    return config.lambda_grid


def _training_primary_baseline_desired_retention(
    config: ExperimentConfig,
    baseline_dr_values: tuple[float, ...],
) -> float:
    raw_single = config.training_policy_search.get("baseline_desired_retention")
    if isinstance(raw_single, bool) or not isinstance(raw_single, (float, int)):
        return baseline_dr_values[0]
    return float(raw_single)


def _training_metadata_requires_baseline_dr(config: ExperimentConfig) -> bool:
    return (
        "baseline_desired_retention" in config.training_policy_search
        or "baseline_desired_retention_values" in config.training_policy_search
    )


def _validate_train_artifacts(
    *,
    artifact_paths: list[Path],
    config: ExperimentConfig,
    user_id: int,
    lambda_value: float | None,
    baseline_desired_retention: float | None = None,
    allowed_baseline_desired_retentions: tuple[float, ...] | None = None,
) -> str | None:
    observed_baseline_desired_retentions: list[float] = []
    for path in artifact_paths:
        try:
            metadata = validate_scheduler_artifact(path, require_files=True)
        except ValueError as exc:
            return f"Invalid scheduler artifact metadata {path}: {exc}"
        if metadata.family != config.family:
            return (
                f"Invalid scheduler artifact metadata {path}: family expected "
                f"{config.family!r}, got {metadata.family!r}."
            )
        if metadata.seed != config.seed:
            return (
                f"Invalid scheduler artifact metadata {path}: seed expected "
                f"{config.seed}, got {metadata.seed}."
            )
        if metadata.engine.value != config.simulation.engine:
            return (
                f"Invalid scheduler artifact metadata {path}: engine expected "
                f"{config.simulation.engine!r}, got {metadata.engine.value!r}."
            )
        if metadata.environment != config.simulation.environment:
            return (
                f"Invalid scheduler artifact metadata {path}: environment expected "
                f"{config.simulation.environment!r}, got {metadata.environment!r}."
            )
        if metadata.review_markov_transition != (
            config.simulation.review_markov_transition
        ):
            return (
                f"Invalid scheduler artifact metadata {path}: "
                "review_markov_transition expected "
                f"{config.simulation.review_markov_transition!r}, got "
                f"{metadata.review_markov_transition!r}."
            )
        if metadata.training_user_ids != (user_id,):
            return (
                f"Invalid scheduler artifact metadata {path}: training_user_ids "
                f"expected [{user_id}], got {list(metadata.training_user_ids)}."
            )
        allows_lambda_none = action_space_allows_lambda_none(metadata.action_space)
        if not allows_lambda_none:
            if lambda_value is None:
                return (
                    f"Invalid scheduler artifact metadata {path}: lambda_value is "
                    "required for non-portfolio trainers."
                )
            if metadata.lambda_value is None or not math.isclose(
                metadata.lambda_value,
                lambda_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return (
                    f"Invalid scheduler artifact metadata {path}: "
                    f"lambda_value expected {lambda_value}, "
                    f"got {metadata.lambda_value}."
                )
        elif (
            lambda_value is not None
            and metadata.lambda_value is not None
            and not math.isclose(
                metadata.lambda_value,
                lambda_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return (
                f"Invalid scheduler artifact metadata {path}: lambda_value expected "
                f"{lambda_value}, got {metadata.lambda_value}."
            )
        if baseline_desired_retention is not None and not allows_lambda_none:
            if metadata.baseline_desired_retention is None or not math.isclose(
                metadata.baseline_desired_retention,
                baseline_desired_retention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return (
                    f"Invalid scheduler artifact metadata {path}: "
                    "baseline_desired_retention expected "
                    f"{baseline_desired_retention}, "
                    f"got {metadata.baseline_desired_retention}."
                )
        if allowed_baseline_desired_retentions is not None and not allows_lambda_none:
            if metadata.baseline_desired_retention is None:
                return (
                    f"Invalid scheduler artifact metadata {path}: "
                    "baseline_desired_retention is required for batched DR grid."
                )
            if not any(
                math.isclose(
                    metadata.baseline_desired_retention,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for expected in allowed_baseline_desired_retentions
            ):
                return (
                    f"Invalid scheduler artifact metadata {path}: "
                    "baseline_desired_retention expected one of "
                    f"{list(allowed_baseline_desired_retentions)}, "
                    f"got {metadata.baseline_desired_retention}."
                )
            observed_baseline_desired_retentions.append(
                metadata.baseline_desired_retention
            )
    if allowed_baseline_desired_retentions is not None:
        for expected in allowed_baseline_desired_retentions:
            matches = [
                observed
                for observed in observed_baseline_desired_retentions
                if math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
            ]
            if observed_baseline_desired_retentions and len(matches) != 1:
                return (
                    "Invalid scheduler artifact metadata set: "
                    f"baseline_desired_retention {expected} expected exactly once, "
                    f"got {len(matches)} matches."
                )
    return None


def _read_train_artifact_paths(summary_path: Path) -> tuple[list[Path], list[str]]:
    if not summary_path.exists():
        return [], [f"Missing train-overfit summary: {summary_path}"]
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Cannot read train-overfit summary {summary_path}: {exc}"]
    if not isinstance(payload, dict):
        return [], [f"Train-overfit summary must be an object: {summary_path}"]
    if payload.get("passed") is not True:
        return [], [f"Train-overfit summary did not pass: {summary_path}"]
    raw_paths = payload.get("artifact_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        return [], [f"Train-overfit summary has no artifact_paths: {summary_path}"]
    paths: list[Path] = []
    notes: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str) or not raw_path.strip():
            notes.append(
                f"Train-overfit artifact_paths[{index}] must be a non-empty string."
            )
            continue
        paths.append(Path(raw_path))
    return paths, notes


def _sweep_requires_train_artifacts(
    *,
    config: ExperimentConfig,
    batched_retention_sweep: bool,
) -> bool:
    if config.sweep_batch_scheduler_artifacts or not batched_retention_sweep:
        return True
    return _configured_batched_sweep_uses_trained_schedulers(config)


def _configured_batched_sweep_uses_trained_schedulers(
    config: ExperimentConfig,
) -> bool:
    from simulator.scheduler_spec import parse_scheduler_spec

    return any(
        parse_scheduler_spec(raw)[0] in RUN_ID_SCOPED_SWEEP_SCHEDULERS
        for raw in config.sweep_batched.schedulers
    )


def _validate_sweep_artifact_metadata(
    *,
    metadata_path: Path,
    metadata: SchedulerArtifactMetadata,
    config: ExperimentConfig,
) -> str | None:
    if metadata.family != config.family:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: family expected "
            f"{config.family!r}, got {metadata.family!r}."
        )
    if metadata.seed != config.seed:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: seed expected "
            f"{config.seed}, got {metadata.seed}."
        )
    if metadata.engine.value != config.simulation.engine:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: engine expected "
            f"{config.simulation.engine!r}, got {metadata.engine.value!r}."
        )
    if metadata.environment != config.simulation.environment:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: environment "
            f"expected {config.simulation.environment!r}, got {metadata.environment!r}."
        )
    if metadata.review_markov_transition != config.simulation.review_markov_transition:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: "
            "review_markov_transition expected "
            f"{config.simulation.review_markov_transition!r}, got "
            f"{metadata.review_markov_transition!r}."
        )
    if len(metadata.training_user_ids) != 1:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: sweep requires "
            "exactly one training_user_id."
        )
    allows_lambda_none = action_space_allows_lambda_none(metadata.action_space)
    if metadata.lambda_value is None and not allows_lambda_none:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: lambda_value is "
            "required for sweep."
        )
    baseline_dr_values = _training_baseline_desired_retention_values(config)
    if _training_metadata_requires_baseline_dr(config) and not allows_lambda_none:
        actual_dr = metadata.baseline_desired_retention
        if actual_dr is None or not any(
            math.isclose(actual_dr, expected, rel_tol=0.0, abs_tol=1e-9)
            for expected in baseline_dr_values
        ):
            return (
                f"Invalid scheduler artifact metadata {metadata_path}: "
                "baseline_desired_retention expected one of "
                f"{list(baseline_dr_values)!r}, got {actual_dr!r}."
            )
    return None


def _collect_sweep_logs(
    *,
    output_dir: Path,
    log_glob: str,
) -> tuple[list[Path], str | None]:
    return _collect_paths(
        root=output_dir,
        path_glob=log_glob,
        field_name="sweep.log_glob",
    )


def _validate_sweep_logs(
    *,
    log_paths: list[Path],
    config: ExperimentConfig,
    job: SweepBatchLane,
) -> str | None:
    for path in log_paths:
        records = _read_log_meta_and_totals(path)
        if records is None:
            return f"Sweep log is missing meta or totals record: {path}"
        meta, _ = records
        errors = _simulation_metadata_errors(
            config=config,
            meta=meta,
            expected_engine=config.simulation.engine,
            expected_scheduler=job.scheduler_name,
            expected_user_id=job.user_id,
        )
        if job.desired_retention is not None:
            actual_retention = meta.get("desired_retention")
            if not isinstance(actual_retention, (float, int)) or not math.isclose(
                float(actual_retention),
                job.desired_retention,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "metadata desired_retention expected "
                    f"{job.desired_retention!r}, got {actual_retention!r}"
                )
        if errors:
            source = (
                f"artifact {job.metadata_path}"
                if job.metadata_path is not None
                else f"{job.source} lane"
            )
            return f"Sweep log metadata mismatch for {path} ({source}): " + "; ".join(
                errors
            )
    return None


def _read_log_meta_and_totals(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    meta: dict[str, Any] | None = None
    totals: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if not isinstance(record, dict):
                    continue
                data = record.get("data")
                if record.get("type") == "meta" and isinstance(data, dict):
                    meta = data
                elif record.get("type") == "totals" and isinstance(data, dict):
                    totals = data
                if meta is not None and totals is not None:
                    return meta, totals
    except OSError:
        return None
    return None


def _read_passed_stage_summary(summary_path: Path, stage_name: StageName) -> list[str]:
    if not summary_path.exists():
        return [f"Missing {stage_name.value} summary: {summary_path}"]
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read {stage_name.value} summary {summary_path}: {exc}"]
    if not isinstance(payload, dict):
        return [f"{stage_name.value} summary must be an object: {summary_path}"]
    if payload.get("passed") is not True:
        return [f"{stage_name.value} summary did not pass: {summary_path}"]
    return []


def _read_selected_artifact_paths(summary_path: Path) -> tuple[list[Path], list[str]]:
    if not summary_path.exists():
        return [], [f"Missing select summary: {summary_path}"]
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Cannot read select summary {summary_path}: {exc}"]
    if not isinstance(payload, dict):
        return [], [f"Select summary must be an object: {summary_path}"]
    raw_paths = payload.get("selected_artifact_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        return [], [f"Select summary has no selected_artifact_paths: {summary_path}"]
    paths: list[Path] = []
    notes: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str) or not raw_path.strip():
            notes.append(
                f"Select summary selected_artifact_paths[{index}] must be a "
                "non-empty string."
            )
            continue
        paths.append(Path(raw_path))
    return paths, notes


def _collect_paths(
    *,
    root: Path,
    path_glob: str,
    field_name: str,
) -> tuple[list[Path], str | None]:
    try:
        paths = sorted(path for path in root.glob(path_glob) if path.is_file())
    except ValueError as exc:
        return [], f"Invalid {field_name} {path_glob!r}: {exc}"
    return paths, None


def _validate_json_files(paths: list[Path]) -> str | None:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return f"Invalid JSON artifact {path}: {exc}"
    return None


def _validate_reserved_test_logs(
    *,
    log_paths: list[Path],
    config: ExperimentConfig,
    metadata: SchedulerArtifactMetadata,
    metadata_path: Path,
) -> str | None:
    expected_users = set(config.users.reserved_test)
    seen_users: set[int] = set()
    for path in log_paths:
        records = _read_log_meta_and_totals(path)
        if records is None:
            return f"Reserved-test log is missing meta or totals record: {path}"
        meta, _ = records
        user_id = meta.get("user_id")
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            return f"Reserved-test log has invalid user_id: {path}"
        if user_id not in expected_users:
            return (
                f"Reserved-test log {path} has unexpected user_id {user_id}; "
                f"expected one of {sorted(expected_users)}."
            )
        seen_users.add(user_id)
        errors = _simulation_metadata_errors(
            config=config,
            meta=meta,
            expected_engine=config.simulation.engine,
            expected_scheduler=metadata.scheduler_name,
            expected_user_id=user_id,
        )
        if errors:
            return (
                f"Reserved-test log metadata mismatch for {path} "
                f"(artifact {metadata_path}): " + "; ".join(errors)
            )
    missing_users = sorted(expected_users - seen_users)
    if missing_users:
        return "Missing reserved-test logs for users: " + ", ".join(
            str(user_id) for user_id in missing_users
        )
    return None


def _validate_selection_files(
    paths: list[Path],
    *,
    config: ExperimentConfig,
) -> str | None:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return f"Invalid selection JSON {path}: {exc}"
        if not isinstance(payload, dict):
            return f"Selection JSON must be an object: {path}"
        raw_artifact_path = payload.get("selected_artifact_metadata_path")
        if not isinstance(raw_artifact_path, str) or not raw_artifact_path.strip():
            return (
                f"Selection JSON must contain selected_artifact_metadata_path: {path}"
            )
        reason = payload.get("selection_reason")
        if not isinstance(reason, str) or not reason.strip():
            return f"Selection JSON must contain selection_reason: {path}"
        artifact_path = _resolve_selection_artifact_path(
            selection_path=path,
            raw_artifact_path=raw_artifact_path,
        )
        try:
            metadata = validate_scheduler_artifact(artifact_path, require_files=True)
        except ValueError as exc:
            return f"Invalid selected scheduler artifact {artifact_path}: {exc}"
        artifact_note = _validate_sweep_artifact_metadata(
            metadata_path=artifact_path,
            metadata=metadata,
            config=config,
        )
        if artifact_note is not None:
            return artifact_note
    return None


def _selection_artifact_path(selection_path: Path) -> Path:
    with selection_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _resolve_selection_artifact_path(
        selection_path=selection_path,
        raw_artifact_path=str(payload["selected_artifact_metadata_path"]),
    )


def _resolve_selection_artifact_path(
    *,
    selection_path: Path,
    raw_artifact_path: str,
) -> Path:
    path = Path(raw_artifact_path)
    if path.is_absolute():
        return path
    return (selection_path.parent / path).resolve()


def _validate_aggregate_files(paths: list[Path]) -> tuple[str | None, bool]:
    gate_passed = True
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return f"Invalid aggregate JSON {path}: {exc}", False
        if not isinstance(payload, dict):
            return f"Aggregate JSON must be an object: {path}", False
        passed = payload.get("passed")
        if not isinstance(passed, bool):
            return f"Aggregate JSON must contain boolean passed: {path}", False
        gate_passed = gate_passed and passed
    return None, gate_passed


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
