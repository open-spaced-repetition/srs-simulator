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
from simulator.retention_sweep.log_filter import LogFilenameFilter


SUPPORTED_RUNNER_STAGES = {
    StageName.DRY_RUN,
    StageName.PREFLIGHT,
    StageName.STAGE_BASELINE,
    StageName.TRAIN_OVERFIT,
    StageName.SWEEP,
    StageName.PARETO,
    StageName.SELECT,
    StageName.AGGREGATE,
    StageName.RESERVED_TEST,
}

COMMAND_TIMEOUT_EXIT_CODE = 124


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    exit_code: int
    stage: StageName
    run_id: str
    stage_root: Path | None
    summary: dict[str, Any]


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
    baseline_dr_values = _training_baseline_desired_retention_values(config)
    include_baseline_dr_in_path = _training_uses_baseline_dr_grid(config)

    if not config.train_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("training.command_template is required for train-overfit.")
    elif not baseline_dr_values:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append(
            "training.sa.baseline_desired_retention_values must contain numbers "
            "without duplicates."
        )
    else:
        for user_id in config.users.train:
            for baseline_dr in baseline_dr_values:
                baseline_dr_token = _format_retention_token(baseline_dr)
                for lambda_value in config.lambda_grid:
                    lambda_token = _format_lambda_token(lambda_value)
                    if include_baseline_dr_in_path:
                        output_dir = (
                            outputs_root
                            / f"user_{user_id}"
                            / f"dr_{baseline_dr_token}"
                            / f"lambda_{lambda_token}"
                        )
                        command_stem = (
                            f"user_{user_id}_dr_{baseline_dr_token}_"
                            f"lambda_{lambda_token}"
                        )
                    else:
                        output_dir = (
                            outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
                        )
                        command_stem = f"user_{user_id}_lambda_{lambda_token}"
                    command_record = commands_root / f"{command_stem}_command.json"
                    stdout_path = commands_root / f"{command_stem}_stdout.txt"
                    stderr_path = commands_root / f"{command_stem}_stderr.txt"
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
                        failures.append(FailureClass.INVALID_CONFIG)
                        notes.append(
                            "Invalid training.command_template for "
                            f"user={user_id}, baseline_dr={baseline_dr}, "
                            f"lambda={lambda_value}: {exc}"
                        )
                        break

                    output_dir.mkdir(parents=True, exist_ok=True)
                    commands_attempted += 1
                    train_command_record = _run_recorded_command(
                        command=train_command,
                        cwd=repo_root,
                        command_record_path=command_record,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout_seconds=config.performance.timeout_seconds,
                    )
                    exit_code = _record_exit_code(train_command_record)
                    timed_out = exit_code == COMMAND_TIMEOUT_EXIT_CODE
                    progress_path = output_dir / "training_progress.jsonl"
                    progress_path_exists = progress_path.exists()
                    if progress_path_exists:
                        progress_paths.append(progress_path)
                    command_records.append(command_record)
                    stdout_paths.append(stdout_path)
                    stderr_paths.append(stderr_path)
                    command_results.append(
                        {
                            "user_id": user_id,
                            "baseline_desired_retention": baseline_dr,
                            "baseline_desired_retention_token": baseline_dr_token,
                            "lambda_value": lambda_value,
                            "lambda_token": lambda_token,
                            "output_dir": str(output_dir),
                            "command_record_path": str(command_record),
                            "stdout_path": str(stdout_path),
                            "stderr_path": str(stderr_path),
                            "training_progress_path": str(progress_path)
                            if progress_path_exists
                            else None,
                            "exit_code": exit_code,
                            "timed_out": timed_out,
                        }
                    )
                    if exit_code != 0:
                        failures.append(
                            FailureClass.TIMEOUT
                            if timed_out
                            else FailureClass.RUNNER_FAILED
                        )
                        if timed_out:
                            notes.append(
                                "Training command timed out for "
                                f"user={user_id}, baseline_dr={baseline_dr}, "
                                f"lambda={lambda_value}."
                            )
                        else:
                            notes.append(
                                "Training command failed for "
                                f"user={user_id}, baseline_dr={baseline_dr}, "
                                f"lambda={lambda_value}."
                            )
                        break

                    commands_succeeded += 1
                    try:
                        matched_artifacts = sorted(
                            path
                            for path in output_dir.glob(config.train_artifact_glob)
                            if path.is_file()
                        )
                    except ValueError as exc:
                        failures.append(FailureClass.INVALID_CONFIG)
                        notes.append(
                            "Invalid training.artifact_metadata_glob "
                            f"{config.train_artifact_glob!r}: {exc}"
                        )
                        break
                    if not matched_artifacts:
                        failures.append(FailureClass.INVALID_ARTIFACT)
                        notes.append(
                            "No scheduler artifact metadata matched "
                            f"{config.train_artifact_glob!r} in {output_dir}."
                        )
                        break

                    invalid_artifact_note = _validate_train_artifacts(
                        artifact_paths=matched_artifacts,
                        config=config,
                        user_id=user_id,
                        lambda_value=lambda_value,
                        baseline_desired_retention=baseline_dr
                        if _training_metadata_requires_baseline_dr(config)
                        else None,
                    )
                    if invalid_artifact_note is not None:
                        failures.append(FailureClass.INVALID_ARTIFACT)
                        notes.append(invalid_artifact_note)
                        break
                    artifact_paths.extend(matched_artifacts)
                if failures:
                    break
            if failures:
                break

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.TRAIN_OVERFIT.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "training_users": float(len(config.users.train)),
            "lambda_values": float(len(config.lambda_grid)),
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
        },
        execution_shape={
            "process_count": 1,
            "subprocess_count": commands_attempted,
            "timeout_seconds": config.performance.timeout_seconds,
        },
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
    train_summary_path = (
        output_root / run_id / StageName.TRAIN_OVERFIT.value / "training_summary.json"
    )

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

    if not config.sweep_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("sweep.command_template is required for sweep.")

    train_artifact_paths: list[Path] = []
    if not failures:
        train_artifact_paths, artifact_notes = _read_train_artifact_paths(
            train_summary_path
        )
        if artifact_notes:
            failures.append(FailureClass.INCOMPLETE_OUTPUT)
            notes.extend(artifact_notes)

    if not failures:
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
            if metadata.lambda_value is None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(
                    f"Invalid scheduler artifact metadata {metadata_path}: "
                    "lambda_value is required for sweep."
                )
                break
            lambda_value = metadata.lambda_value
            lambda_token = _format_lambda_token(lambda_value)
            baseline_dr = metadata.baseline_desired_retention
            if baseline_dr is not None:
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
                output_dir = outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
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
                metadata=metadata,
                metadata_path=metadata_path,
            )
            if log_note is not None:
                failures.append(FailureClass.INVALID_ARTIFACT)
                notes.append(log_note)
                break
            log_paths.extend(matched_logs)

    unique_failures = tuple(dict.fromkeys(failures))
    passed = not unique_failures
    gate_summary = GateSummary(
        gate_name=StageName.SWEEP.value,
        passed=passed,
        failures=unique_failures,
        metrics={
            "input_artifacts": float(len(train_artifact_paths)),
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
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
            "logs_validated": len(log_paths),
        },
        execution_shape={
            "process_count": 1,
            "subprocess_count": commands_attempted,
            "timeout_seconds": config.performance.timeout_seconds,
        },
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
            manifest_path,
            *artifact_paths,
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
        "sweep_summary": sweep_summary_path,
    }
    if config.performance.write_performance_summary:
        manifest_artifacts["performance_summary"] = performance_summary_path
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
    logs_by_user: dict[int, list[Path]] = {}
    retentions_by_user: dict[int, set[float]] = {}

    if not baseline_root.exists():
        failures.append(FailureClass.INVALID_BASELINE)
        notes.append(f"Baseline log root does not exist: {baseline_root}")
    else:
        required_users = (
            set(config.users.train)
            | set(config.users.validation)
            | set(config.users.reserved_test)
        )
        filename_filter = _baseline_filename_filter(config)
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
            metadata_errors = _baseline_metadata_errors(config=config, meta=meta)
            if metadata_errors:
                notes.extend(f"{path}: {error}" for error in metadata_errors)
                continue
            logs_by_user.setdefault(user_id, []).append(path)
            retention_value = _matched_retention_value(
                meta.get("desired_retention"),
                config.baseline.desired_retention_values,
            )
            if retention_value is not None:
                retentions_by_user.setdefault(user_id, set()).add(retention_value)

        missing_users = sorted(required_users - set(logs_by_user))
        if missing_users:
            failures.append(FailureClass.INVALID_BASELINE)
            notes.append(
                "Missing exact baseline logs for users: "
                + ", ".join(str(user_id) for user_id in missing_users)
            )
        if not logs_by_user:
            failures.append(FailureClass.INVALID_BASELINE)
            notes.append("No exact baseline logs matched the config.")
        if config.baseline.desired_retention_values:
            missing_pairs: list[str] = []
            required_retentions = set(config.baseline.desired_retention_values)
            for user_id in sorted(required_users):
                missing_retentions = sorted(
                    required_retentions - retentions_by_user.get(user_id, set())
                )
                for retention in missing_retentions:
                    missing_pairs.append(f"user={user_id},ret={retention:.2f}")
            if missing_pairs:
                failures.append(FailureClass.INVALID_BASELINE)
                notes.append(
                    "Missing exact baseline retention points: "
                    + ", ".join(missing_pairs)
                )

        if not failures:
            for user_id in sorted(logs_by_user):
                for source in logs_by_user[user_id]:
                    dest = staged_root / f"user_{user_id}" / source.name
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
            "matched_users": float(len(logs_by_user)),
            "matched_user_retention_pairs": float(
                sum(len(values) for values in retentions_by_user.values())
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
        "output_root": str(output_root),
        "stage_root": str(stage_root),
        "stage_mode": config.baseline.stage_mode,
        "matched_users": sorted(logs_by_user),
        "matched_retentions_by_user": {
            str(user_id): sorted(values)
            for user_id, values in sorted(retentions_by_user.items())
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
            "lambda_values": len(config.lambda_grid),
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
        gpu_metrics=_gpu_performance_metrics(device),
        disk_metrics=_disk_metrics(stage_root),
        failure_class=failures[0] if failures else None,
        notes=tuple(notes),
    )


def _resolve_performance_device(config: ExperimentConfig) -> str:
    if config.performance.device:
        return config.performance.device
    if config.gpu_guard.device:
        return config.gpu_guard.device
    torch_device = config.training_sa.get("torch_device")
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
        "train_users": len(config.users.train),
        "validation_users": len(config.users.validation),
        "reserved_test_users": len(config.users.reserved_test),
        "lambda_values": len(config.lambda_grid),
        "training_baseline_desired_retention_values": len(
            _training_baseline_desired_retention_values(config)
        ),
        "baseline_retention_values": len(config.baseline.desired_retention_values),
    }
    chains = _training_chains(config)
    if chains is not None:
        shape["chains"] = chains
    if stage == StageName.TRAIN_OVERFIT and chains is not None:
        shape["effective_lanes"] = (
            len(config.users.train)
            * len(config.lambda_grid)
            * len(_training_baseline_desired_retention_values(config))
            * chains
        )
    elif stage == StageName.SWEEP:
        shape["effective_lanes"] = (
            len(config.users.train)
            * max(len(config.lambda_grid), 1)
            * len(_training_baseline_desired_retention_values(config))
        )
    return shape


def _training_chains(config: ExperimentConfig) -> int | None:
    value = config.training_sa.get("chains")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _candidate_days(*, config: ExperimentConfig, stage: StageName) -> int | None:
    if stage != StageName.TRAIN_OVERFIT:
        return None
    chains = _training_chains(config)
    if chains is None:
        return None
    return (
        config.simulation.days
        * len(config.users.train)
        * len(config.lambda_grid)
        * len(_training_baseline_desired_retention_values(config))
        * chains
    )


def _user_days(*, config: ExperimentConfig, stage: StageName) -> int | None:
    if stage == StageName.TRAIN_OVERFIT:
        return (
            config.simulation.days
            * len(config.users.train)
            * len(config.lambda_grid)
            * len(_training_baseline_desired_retention_values(config))
        )
    if stage == StageName.SWEEP:
        return (
            config.simulation.days
            * len(config.users.train)
            * max(len(config.lambda_grid), 1)
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
    *, config: ExperimentConfig, meta: dict[str, Any]
) -> list[str]:
    errors = _simulation_metadata_errors(
        config=config,
        meta=meta,
        expected_engine=config.baseline.expected_engine,
        expected_scheduler=config.baseline.scheduler,
    )
    if config.baseline.desired_retention_values:
        actual_retention = meta.get("desired_retention")
        if (
            _matched_retention_value(
                actual_retention, config.baseline.desired_retention_values
            )
            is None
        ):
            errors.append(
                "metadata desired_retention expected one of "
                f"{list(config.baseline.desired_retention_values)!r}, "
                f"got {actual_retention!r}"
            )
    return errors


def _baseline_filename_filter(config: ExperimentConfig) -> LogFilenameFilter:
    short_term = "on" if config.simulation.short_term_source else "off"
    short_term_source = config.simulation.short_term_source or "any"
    retention_values_by_scheduler = None
    if len(config.baseline.desired_retention_values) == 1:
        retention_values_by_scheduler = {
            config.baseline.scheduler: round(
                config.baseline.desired_retention_values[0], 2
            )
        }
    return LogFilenameFilter(
        envs=[config.simulation.environment],
        scheds=[config.baseline.scheduler],
        engine=config.baseline.expected_engine,
        short_term=short_term,
        short_term_source=short_term_source,
        start_retention=min(config.baseline.desired_retention_values)
        if config.baseline.desired_retention_values
        else None,
        end_retention=max(config.baseline.desired_retention_values)
        if config.baseline.desired_retention_values
        else None,
        priority=config.simulation.priority,
        retention_values_by_scheduler=retention_values_by_scheduler,
    )


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
            user_dir_paths.extend(sorted(user_dir.glob("*.jsonl")))
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
) -> list[str]:
    expected: dict[str, Any] = {
        "engine": expected_engine,
        "days": config.simulation.days,
        "deck_size": config.simulation.deck,
        "learn_limit": config.simulation.learn_limit,
        "review_limit": config.simulation.review_limit,
        "cost_limit_minutes": config.simulation.cost_limit_minutes,
        "priority": config.simulation.priority,
        "environment": config.simulation.environment,
        "scheduler_priority": config.simulation.scheduler_priority,
        "seed": config.seed,
        "fuzz": config.simulation.fuzz,
        "short_term": bool(config.simulation.short_term_source),
        "short_term_source": config.simulation.short_term_source,
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
    lambda_value: float,
    baseline_desired_retention: float,
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    lambda_token = _format_lambda_token(lambda_value)
    baseline_dr_token = _format_retention_token(baseline_desired_retention)
    values: dict[str, Any] = {
        "user_id": user_id,
        "lambda_value": lambda_value,
        "lambda_token": lambda_token,
        "baseline_desired_retention": baseline_desired_retention,
        "baseline_desired_retention_token": baseline_dr_token,
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
        "environment": config.simulation.environment,
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
    assert metadata.lambda_value is not None
    lambda_value = metadata.lambda_value
    lambda_token = _format_lambda_token(lambda_value)
    baseline_dr = metadata.baseline_desired_retention
    values: dict[str, Any] = {
        "artifact_id": metadata.artifact_id,
        "artifact_metadata_path": str(metadata_path),
        "policy_path": str(metadata.policy_path),
        "scheduler_name": metadata.scheduler_name,
        "user_id": user_id,
        "lambda_value": lambda_value,
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


def _training_baseline_desired_retention_values(
    config: ExperimentConfig,
) -> tuple[float, ...]:
    raw_values = config.training_sa.get("baseline_desired_retention_values")
    if raw_values is None:
        raw_single = config.training_sa.get("baseline_desired_retention", 0.90)
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
    return "baseline_desired_retention_values" in config.training_sa


def _training_metadata_requires_baseline_dr(config: ExperimentConfig) -> bool:
    return (
        "baseline_desired_retention" in config.training_sa
        or "baseline_desired_retention_values" in config.training_sa
    )


def _validate_train_artifacts(
    *,
    artifact_paths: list[Path],
    config: ExperimentConfig,
    user_id: int,
    lambda_value: float,
    baseline_desired_retention: float | None = None,
) -> str | None:
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
        if metadata.training_user_ids != (user_id,):
            return (
                f"Invalid scheduler artifact metadata {path}: training_user_ids "
                f"expected [{user_id}], got {list(metadata.training_user_ids)}."
            )
        if metadata.lambda_value is None or not math.isclose(
            metadata.lambda_value,
            lambda_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return (
                f"Invalid scheduler artifact metadata {path}: lambda_value expected "
                f"{lambda_value}, got {metadata.lambda_value}."
            )
        if baseline_desired_retention is not None:
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
    if len(metadata.training_user_ids) != 1:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: sweep requires "
            "exactly one training_user_id."
        )
    if metadata.lambda_value is None:
        return (
            f"Invalid scheduler artifact metadata {metadata_path}: lambda_value is "
            "required for sweep."
        )
    baseline_dr_values = _training_baseline_desired_retention_values(config)
    if _training_metadata_requires_baseline_dr(config):
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
    metadata: SchedulerArtifactMetadata,
    metadata_path: Path,
) -> str | None:
    user_id = metadata.training_user_ids[0]
    for path in log_paths:
        records = _read_log_meta_and_totals(path)
        if records is None:
            return f"Sweep log is missing meta or totals record: {path}"
        meta, _ = records
        errors = _simulation_metadata_errors(
            config=config,
            meta=meta,
            expected_engine=config.simulation.engine,
            expected_scheduler=metadata.scheduler_name,
            expected_user_id=user_id,
        )
        if errors:
            return (
                f"Sweep log metadata mismatch for {path} "
                f"(artifact {metadata_path}): " + "; ".join(errors)
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
