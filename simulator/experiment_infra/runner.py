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
    errors: list[str] = []
    expected: dict[str, Any] = {
        "scheduler": config.baseline.scheduler,
        "engine": config.baseline.expected_engine,
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


def _stage_baseline_file(*, source: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "hardlink":
        dest.hardlink_to(source)
    else:
        shutil.copy2(source, dest)


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
