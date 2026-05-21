from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
import tomllib

from experiments.single_card_tradeoff.core.reporting import REPO_ROOT


class SingleCardWorkflowStage(StrEnum):
    DRY_RUN = "dry-run"
    PREFLIGHT = "preflight"
    TRAIN = "train"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    BENCHMARK = "benchmark"
    VISUALIZE = "visualize"
    REPORT = "report"


EXECUTION_STAGES: tuple[SingleCardWorkflowStage, ...] = (
    SingleCardWorkflowStage.TRAIN,
    SingleCardWorkflowStage.EVALUATE,
    SingleCardWorkflowStage.ANALYZE,
    SingleCardWorkflowStage.BENCHMARK,
    SingleCardWorkflowStage.VISUALIZE,
    SingleCardWorkflowStage.REPORT,
)


@dataclass(frozen=True)
class WorkflowTask:
    name: str
    stage: SingleCardWorkflowStage
    kind: str
    description: str | None
    options: Mapping[str, Any]
    config_path: Path
    enabled: bool = True
    source: str = "explicit"


@dataclass(frozen=True)
class WorkflowConfig:
    config_path: Path
    name: str
    family: str
    seed: int
    raw: Mapping[str, Any]
    tasks: tuple[WorkflowTask, ...]


def load_workflow_config(path: Path) -> WorkflowConfig:
    config_path = path.expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config must be a TOML table: {config_path}")
    return workflow_config_from_mapping(raw, config_path=config_path)


def workflow_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
    synthesize: bool = True,
) -> WorkflowConfig:
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"schema_version must be 1 in {config_path}.")
    name = _str(raw.get("name"), "name")
    family = _str(raw.get("family", "single_card_tradeoff"), "family")
    if family != "single_card_tradeoff":
        raise ValueError(f"family must be 'single_card_tradeoff' in {config_path}.")
    seed = _int(raw.get("seed", 42), "seed")

    explicit_tasks = tuple(_parse_explicit_tasks(raw, config_path=config_path))
    if raw.get("commands") is not None:
        raise ValueError(
            f"{config_path} still uses legacy [[commands]]. "
            "Use [[tasks]] or a semantic workflow section instead."
        )
    generated_tasks = (
        tuple(_synthesize_tasks(raw, config_path=config_path))
        if synthesize and not explicit_tasks
        else ()
    )
    return WorkflowConfig(
        config_path=config_path,
        name=name,
        family=family,
        seed=seed,
        raw=raw,
        tasks=explicit_tasks + generated_tasks,
    )


def _parse_explicit_tasks(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
) -> list[WorkflowTask]:
    task_rows = raw.get("tasks", [])
    if task_rows is None:
        return []
    if not isinstance(task_rows, list):
        raise ValueError("tasks must be an array of TOML tables.")
    tasks: list[WorkflowTask] = []
    for index, row in enumerate(task_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"tasks[{index}] must be a TOML table.")
        name = _str(row.get("name"), f"tasks[{index}].name")
        kind = _str(row.get("kind"), f"tasks[{index}].kind")
        stage = _stage(row.get("stage"), f"tasks[{index}].stage")
        description = _optional_str(
            row.get("description"), f"tasks[{index}].description"
        )
        enabled = _bool(row.get("enabled", True), f"tasks[{index}].enabled")
        options = row.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError(f"tasks[{index}].options must be a TOML table.")
        tasks.append(
            WorkflowTask(
                name=name,
                stage=stage,
                kind=kind,
                description=description,
                options=dict(options),
                config_path=config_path,
                enabled=enabled,
            )
        )
    return tasks


def _synthesize_tasks(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
) -> list[WorkflowTask]:
    tasks: list[WorkflowTask] = []
    if _is_tradeoff_run_config(raw):
        tasks.append(_tradeoff_run_config_task(raw, config_path=config_path))
    if _is_train_weight_control_config(raw):
        tasks.extend(_train_weight_control_tasks(raw, config_path=config_path))
    return tasks


def _is_tradeoff_run_config(raw: Mapping[str, Any]) -> bool:
    experiment = raw.get("experiment")
    outputs = raw.get("outputs")
    return (
        isinstance(experiment, Mapping)
        and isinstance(outputs, Mapping)
        and experiment.get("schedulers") is not None
        and outputs.get("root") is not None
    )


def _tradeoff_run_config_task(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
) -> WorkflowTask:
    name = _str(raw.get("name"), "name")
    return WorkflowTask(
        name=f"evaluate_{name}",
        stage=SingleCardWorkflowStage.EVALUATE,
        kind="tradeoff_config",
        description=(
            "Evaluate the TOML-configured single-card tradeoff and aggregate "
            "per-user compatibility outputs."
        ),
        options={"config": _display_repo_path(config_path)},
        config_path=config_path,
        source="generated",
    )


def _is_train_weight_control_config(raw: Mapping[str, Any]) -> bool:
    return (
        isinstance(raw.get("experiment"), Mapping)
        and isinstance(raw.get("training"), Mapping)
        and isinstance(raw.get("outputs"), Mapping)
        and isinstance(raw.get("treatments"), list)
    )


def _train_weight_control_tasks(
    raw: Mapping[str, Any],
    *,
    config_path: Path,
) -> list[WorkflowTask]:
    experiment = _table(raw, "experiment")
    training = _table(raw, "training")
    outputs = _table(raw, "outputs")
    fsrs6_adr = _table(raw, "fsrs6_adr", required=False)
    treatments = raw.get("treatments")
    if not isinstance(treatments, list) or not treatments:
        raise ValueError("treatments must be a non-empty array of TOML tables.")

    tasks: list[WorkflowTask] = []
    common_train_options: dict[str, Any] = {
        "env": experiment.get("env", "fsrs6"),
        "user_ids": experiment.get("user_ids"),
        "button_usage": experiment.get("button_usage"),
        "per_user_models": training.get("per_user_models", True),
        "per_user_supervision": training.get("per_user_supervision"),
        "eval_cost_weights": experiment.get("formal_eval_cost_weights"),
        "action_retentions": experiment.get("action_retentions"),
        "network": training.get("network"),
        "hidden_size": training.get("hidden_size"),
        "network_depth": training.get("network_depth"),
        "oracle_s_grid_size": training.get("oracle_s_grid_size"),
        "oracle_d_grid_size": training.get("oracle_d_grid_size"),
        "epochs": training.get("epochs"),
        "steps_per_epoch": training.get("steps_per_epoch"),
        "table_samples_per_weight": training.get("table_samples_per_weight"),
        "eval_particles": training.get("eval_particles"),
        "seed": raw.get("seed", 42),
        "torch_device": experiment.get("torch_device"),
        "no_progress": True,
    }
    for index, treatment in enumerate(treatments):
        if not isinstance(treatment, Mapping):
            raise ValueError(f"treatments[{index}] must be a TOML table.")
        label = _str(treatment.get("label"), f"treatments[{index}].label")
        options = {
            key: value
            for key, value in common_train_options.items()
            if value is not None
        }
        options.update(
            {
                "cost_weights": treatment.get("training_cost_weights"),
                "out_dir": treatment.get("out_dir"),
            }
        )
        tasks.append(
            WorkflowTask(
                name=f"train_{label}",
                stage=SingleCardWorkflowStage.TRAIN,
                kind="stationary_finite_distill_multiuser",
                description=f"Train stationary finite distill treatment {label}.",
                options=options,
                config_path=config_path,
                source="generated",
            )
        )

    treatment_specs = [
        f"{_str(item.get('label'), 'treatments[].label')}="
        f"{_str(item.get('out_dir'), 'treatments[].out_dir')}"
        for item in treatments
        if isinstance(item, Mapping)
    ]
    exact_common = {
        "env": experiment.get("env", "fsrs6"),
        "button_usage": experiment.get("button_usage"),
        "action_retentions": experiment.get("action_retentions"),
        "oracle_s_grid_size": training.get("oracle_s_grid_size"),
        "oracle_d_grid_size": training.get("oracle_d_grid_size"),
        "distill_policy": treatment_specs,
        "skip_default_policies": True,
        "fsrs6_adr_train_run_root": fsrs6_adr.get("train_run_root"),
        "torch_device": experiment.get("torch_device"),
        "no_progress": True,
    }
    formal_options = {
        **{key: value for key, value in exact_common.items() if value is not None},
        "user_ids": experiment.get("user_ids"),
        "cost_weights": experiment.get("formal_eval_cost_weights"),
        "out_dir": outputs.get("formal_exact_value_root"),
    }
    tasks.append(
        WorkflowTask(
            name="evaluate_exact_value_formal",
            stage=SingleCardWorkflowStage.EVALUATE,
            kind="stationary_finite_exact_value_eval",
            description="Evaluate treatments on the formal exact-value grid.",
            options=formal_options,
            config_path=config_path,
            source="generated",
        )
    )
    dense_options = {
        **{key: value for key, value in exact_common.items() if value is not None},
        "user_ids": [2],
        "cost_weights": experiment.get("dense_loww_eval_cost_weights"),
        "out_dir": outputs.get("dense_loww_exact_value_root"),
    }
    tasks.append(
        WorkflowTask(
            name="evaluate_exact_value_user2_dense_loww",
            stage=SingleCardWorkflowStage.EVALUATE,
            kind="stationary_finite_exact_value_eval",
            description="Evaluate user 2 on the dense low-cost exact-value grid.",
            options=dense_options,
            config_path=config_path,
            source="generated",
        )
    )
    candidate = _workflow_table(raw).get("candidate_label", "add_4_only")
    analysis_options = {
        "treatment": treatment_specs,
        "formal_exact_dir": outputs.get("formal_exact_value_root"),
        "dense_loww_exact_dir": outputs.get("dense_loww_exact_value_root"),
        "out_dir": outputs.get("comparison_root"),
        "candidate_label": candidate,
    }
    tasks.append(
        WorkflowTask(
            name="analyze_train_weight_control",
            stage=SingleCardWorkflowStage.ANALYZE,
            kind="train_weight_control_analysis",
            description="Aggregate train-weight treatment gates and diagnostics.",
            options=analysis_options,
            config_path=config_path,
            source="generated",
        )
    )
    return tasks


def _workflow_table(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = raw.get("workflow")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("workflow must be a TOML table.")
    return value


def _table(
    raw: Mapping[str, Any],
    key: str,
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing [{key}] table.")
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[{key}] must be a TOML table.")
    return value


def _stage(value: Any, label: str) -> SingleCardWorkflowStage:
    raw = _str(value, label)
    try:
        return SingleCardWorkflowStage(raw)
    except ValueError as exc:
        allowed = ", ".join(stage.value for stage in SingleCardWorkflowStage)
        raise ValueError(f"{label} must be one of {allowed}.") from exc


def _str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _str(value, label)


def _int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    return int(value)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return bool(value)


def _display_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def workflow_stage_sequence(stage: str) -> tuple[SingleCardWorkflowStage, ...]:
    if stage == "all":
        return EXECUTION_STAGES
    return (SingleCardWorkflowStage(stage),)


def enabled_tasks_for_stage(
    config: WorkflowConfig,
    stage: SingleCardWorkflowStage,
) -> tuple[WorkflowTask, ...]:
    if stage in (SingleCardWorkflowStage.DRY_RUN, SingleCardWorkflowStage.PREFLIGHT):
        return tuple(task for task in config.tasks if task.enabled)
    return tuple(task for task in config.tasks if task.enabled and task.stage == stage)


def coerce_scalar_sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array.")
    return tuple(value)
