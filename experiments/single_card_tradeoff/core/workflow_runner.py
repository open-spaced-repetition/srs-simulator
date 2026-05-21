from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.single_card_tradeoff.core.reporting import REPO_ROOT, resolve_repo_path
from experiments.single_card_tradeoff.core.workflow_config import (
    SingleCardWorkflowStage,
    WorkflowConfig,
    enabled_tasks_for_stage,
    load_workflow_config,
    workflow_stage_sequence,
)
from experiments.single_card_tradeoff.core.workflow_tasks import (
    expected_artifacts,
    task_command,
    task_record,
)


@dataclass(frozen=True)
class StageResult:
    stage: SingleCardWorkflowStage
    stage_root: Path
    passed: bool
    summary_path: Path | None
    task_count: int


def run_workflow(
    *,
    config_path: Path,
    stage: str,
    run_id: str | None = None,
    dry_run: bool = False,
) -> list[StageResult]:
    config = load_workflow_config(config_path)
    results: list[StageResult] = []
    for stage_name in workflow_stage_sequence(stage):
        results.append(
            run_workflow_stage(
                config,
                stage=stage_name,
                run_id=run_id,
                dry_run=dry_run,
            )
        )
        if not results[-1].passed:
            break
    return results


def run_workflow_stage(
    config: WorkflowConfig,
    *,
    stage: SingleCardWorkflowStage,
    run_id: str | None = None,
    dry_run: bool = False,
) -> StageResult:
    tasks = enabled_tasks_for_stage(config, stage)
    if stage == SingleCardWorkflowStage.DRY_RUN or dry_run:
        summary = _stage_summary(config, stage=stage, tasks=tasks, results=[])
        print(json.dumps(summary, indent=2, sort_keys=True))
        return StageResult(
            stage=stage,
            stage_root=_stage_root(config, stage=stage, run_id=run_id),
            passed=True,
            summary_path=None,
            task_count=len(tasks),
        )

    stage_root = _stage_root(config, stage=stage, run_id=run_id)
    stage_root.mkdir(parents=True, exist_ok=True)
    _write_json(stage_root / "resolved_tasks.json", _resolved_tasks(config))

    if stage == SingleCardWorkflowStage.PREFLIGHT:
        records = [_preflight_task_record(task) for task in tasks]
        summary = _stage_summary(config, stage=stage, tasks=tasks, results=records)
        summary["passed"] = True
        summary_path = stage_root / "preflight_summary.json"
        _write_json(summary_path, summary)
        return StageResult(
            stage=stage,
            stage_root=stage_root,
            passed=True,
            summary_path=summary_path,
            task_count=len(tasks),
        )

    command_root = stage_root / "commands"
    command_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    passed = True
    for task in tasks:
        record = task_record(task)
        record_path = command_root / f"{_safe_name(task.name)}.json"
        _write_json(record_path, record)
        started = _now()
        completed = None
        returncode = 0
        try:
            subprocess.run(task_command(task), cwd=REPO_ROOT, check=True)
            completed = _now()
        except subprocess.CalledProcessError as exc:
            completed = _now()
            returncode = int(exc.returncode)
            passed = False
        artifacts = expected_artifacts(task)
        results.append(
            {
                "name": task.name,
                "stage": task.stage.value,
                "kind": task.kind,
                "command_record": _display_repo_path(record_path),
                "started_at": started,
                "completed_at": completed,
                "returncode": returncode,
                "passed": returncode == 0,
                "expected_artifacts": [_display_repo_path(path) for path in artifacts],
                "available_expected_artifacts": [
                    _display_repo_path(path) for path in artifacts if path.exists()
                ],
                "missing_expected_artifacts": [
                    _display_repo_path(path) for path in artifacts if not path.exists()
                ],
            }
        )
        if not passed:
            break

    summary = _stage_summary(config, stage=stage, tasks=tasks, results=results)
    summary["passed"] = passed
    summary_path = stage_root / f"{stage.value}_summary.json"
    _write_json(summary_path, summary)
    return StageResult(
        stage=stage,
        stage_root=stage_root,
        passed=passed,
        summary_path=summary_path,
        task_count=len(tasks),
    )


def workflow_run_root(config: WorkflowConfig) -> Path:
    workflow = config.raw.get("workflow")
    if isinstance(workflow, dict) and isinstance(workflow.get("run_root"), str):
        return resolve_repo_path(Path(workflow["run_root"]))
    outputs = config.raw.get("outputs")
    if isinstance(outputs, dict):
        for key in ("comparison_root", "root"):
            value = outputs.get(key)
            if isinstance(value, str):
                return resolve_repo_path(Path(value)) / "workflow"
    report = config.raw.get("report")
    if isinstance(report, dict) and isinstance(report.get("root"), str):
        return resolve_repo_path(Path(report["root"])) / "workflow"
    return REPO_ROOT / "artifacts" / "single_card_tradeoff" / "workflows" / config.name


def _stage_root(
    config: WorkflowConfig,
    *,
    stage: SingleCardWorkflowStage,
    run_id: str | None,
) -> Path:
    return workflow_run_root(config) / (run_id or config.name) / stage.value


def _resolved_tasks(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "type": "single-card-tradeoff-workflow",
        "schema_version": 1,
        "generated_at": _now(),
        "config_path": _display_repo_path(config.config_path),
        "name": config.name,
        "tasks": [task_record(task) for task in config.tasks if task.enabled],
    }


def _preflight_task_record(task: Any) -> dict[str, Any]:
    record = task_record(task)
    artifacts = expected_artifacts(task)
    record["available_expected_artifacts"] = [
        _display_repo_path(path) for path in artifacts if path.exists()
    ]
    record["missing_expected_artifacts"] = [
        _display_repo_path(path) for path in artifacts if not path.exists()
    ]
    return record


def _stage_summary(
    config: WorkflowConfig,
    *,
    stage: SingleCardWorkflowStage,
    tasks: Sequence[Any],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "single-card-tradeoff-stage-summary",
        "schema_version": 1,
        "generated_at": _now(),
        "config_path": _display_repo_path(config.config_path),
        "name": config.name,
        "stage": stage.value,
        "task_count": len(tasks),
        "tasks": [task_record(task) for task in tasks],
        "results": list(results),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _display_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
