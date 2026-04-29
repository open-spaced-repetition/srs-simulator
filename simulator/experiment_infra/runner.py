from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
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
    RunRecord,
    StageName,
)


SUPPORTED_RUNNER_STAGES = {
    StageName.DRY_RUN,
    StageName.PREFLIGHT,
    StageName.STAGE_BASELINE,
    StageName.TRAIN_OVERFIT,
    StageName.SWEEP,
    StageName.PARETO,
    StageName.SELECT,
}


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
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.TRAIN_OVERFIT.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    training_summary_path = stage_root / "training_summary.json"
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
    artifact_paths: list[Path] = []
    command_results: list[dict[str, Any]] = []
    commands_attempted = 0
    commands_succeeded = 0

    if not config.train_command_template:
        failures.append(FailureClass.INVALID_CONFIG)
        notes.append("training.command_template is required for train-overfit.")
    else:
        for user_id in config.users.train:
            for lambda_value in config.lambda_grid:
                lambda_token = _format_lambda_token(lambda_value)
                output_dir = outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
                command_record = (
                    commands_root / f"user_{user_id}_lambda_{lambda_token}_command.json"
                )
                stdout_path = (
                    commands_root / f"user_{user_id}_lambda_{lambda_token}_stdout.txt"
                )
                stderr_path = (
                    commands_root / f"user_{user_id}_lambda_{lambda_token}_stderr.txt"
                )
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
                        command_record_path=command_record,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                except (KeyError, ValueError) as exc:
                    failures.append(FailureClass.INVALID_CONFIG)
                    notes.append(
                        "Invalid training.command_template for "
                        f"user={user_id}, lambda={lambda_value}: {exc}"
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
                )
                exit_code = _record_exit_code(train_command_record)
                command_records.append(command_record)
                stdout_paths.append(stdout_path)
                stderr_paths.append(stderr_path)
                command_results.append(
                    {
                        "user_id": user_id,
                        "lambda_value": lambda_value,
                        "lambda_token": lambda_token,
                        "output_dir": str(output_dir),
                        "command_record_path": str(command_record),
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "exit_code": exit_code,
                    }
                )
                if exit_code != 0:
                    failures.append(FailureClass.RUNNER_FAILED)
                    notes.append(
                        "Training command failed for "
                        f"user={user_id}, lambda={lambda_value}."
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
                )
                if invalid_artifact_note is not None:
                    failures.append(FailureClass.INVALID_ARTIFACT)
                    notes.append(invalid_artifact_note)
                    break
                artifact_paths.extend(matched_artifacts)
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
            "commands_attempted": float(commands_attempted),
            "commands_succeeded": float(commands_succeeded),
            "artifacts_validated": float(len(artifact_paths)),
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
        "command_results": command_results,
        "artifact_paths": [str(path) for path in artifact_paths],
        "config_snapshot_path": str(config_snapshot_path),
        "resolved_config_path": str(resolved_config_path),
        "gate_summary_path": str(gate_summary_path),
        "command_record_path": str(command_record_path),
        "run_record_path": str(run_record_path),
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
            manifest_path,
            *command_records,
            *stdout_paths,
            *stderr_paths,
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
    output_root = _resolve_repo_path(repo_root, config.output_root)
    stage_root = output_root / run_id / StageName.SWEEP.value
    stage_root.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = stage_root / "config_snapshot.toml"
    resolved_config_path = stage_root / "resolved_config.json"
    gate_summary_path = stage_root / "gate_summary.json"
    command_record_path = stage_root / "command_record.json"
    run_record_path = stage_root / "run_record.json"
    sweep_summary_path = stage_root / "sweep_summary.json"
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
            output_dir = outputs_root / f"user_{user_id}" / f"lambda_{lambda_token}"
            command_record = (
                commands_root / f"user_{user_id}_lambda_{lambda_token}_command.json"
            )
            stdout_path = (
                commands_root / f"user_{user_id}_lambda_{lambda_token}_stdout.txt"
            )
            stderr_path = (
                commands_root / f"user_{user_id}_lambda_{lambda_token}_stderr.txt"
            )
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
            )
            exit_code = _record_exit_code(sweep_command_record)
            command_records.append(command_record)
            stdout_paths.append(stdout_path)
            stderr_paths.append(stderr_path)
            command_results.append(
                {
                    "artifact_metadata_path": str(metadata_path),
                    "artifact_id": metadata.artifact_id,
                    "user_id": user_id,
                    "lambda_value": lambda_value,
                    "lambda_token": lambda_token,
                    "output_dir": str(output_dir),
                    "command_record_path": str(command_record),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "exit_code": exit_code,
                }
            )
            if exit_code != 0:
                failures.append(FailureClass.RUNNER_FAILED)
                notes.append(f"Sweep command failed for artifact={metadata_path}.")
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
            )
            exit_code = _record_exit_code(pareto_command_record)
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
                }
            )
            if exit_code != 0:
                failures.append(FailureClass.RUNNER_FAILED)
                notes.append("Pareto command failed.")
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
            )
            exit_code = _record_exit_code(select_command_record)
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
                }
            )
            if exit_code != 0:
                failures.append(FailureClass.RUNNER_FAILED)
                notes.append("Select command failed.")
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
        for path in sorted(baseline_root.rglob("*.jsonl")):
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

        required_users = (
            set(config.users.train)
            | set(config.users.validation)
            | set(config.users.reserved_test)
        )
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
        )
        exit_code = completed.returncode
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
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
    command_record_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[str]:
    lambda_token = _format_lambda_token(lambda_value)
    values: dict[str, Any] = {
        "user_id": user_id,
        "lambda_value": lambda_value,
        "lambda_token": lambda_token,
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
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
    values: dict[str, Any] = {
        "artifact_id": metadata.artifact_id,
        "artifact_metadata_path": str(metadata_path),
        "policy_path": str(metadata.policy_path),
        "scheduler_name": metadata.scheduler_name,
        "user_id": user_id,
        "lambda_value": lambda_value,
        "lambda_token": lambda_token,
        "run_id": run_id,
        "seed": config.seed,
        "family": config.family,
        "engine": config.simulation.engine,
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


def _format_lambda_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def _validate_train_artifacts(
    *,
    artifact_paths: list[Path],
    config: ExperimentConfig,
    user_id: int,
    lambda_value: float,
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
