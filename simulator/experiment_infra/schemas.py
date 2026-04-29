from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class StageName(StrEnum):
    DRY_RUN = "dry-run"
    PREFLIGHT = "preflight"
    STAGE_BASELINE = "stage-baseline"
    TRAIN_OVERFIT = "train-overfit"
    SWEEP = "sweep"
    PARETO = "pareto"
    SELECT = "select"
    AGGREGATE = "aggregate"
    RESERVED_TEST = "reserved-test"


class FailureClass(StrEnum):
    INVALID_CONFIG = "invalid-config"
    INVALID_BASELINE = "invalid-baseline"
    INVALID_ARTIFACT = "invalid-artifact"
    GPU_GUARD_FAILED = "gpu-guard-failed"
    INCOMPLETE_OUTPUT = "incomplete-output"
    GATE_FAILED = "gate-failed"
    RUNNER_FAILED = "runner-failed"


@dataclass(frozen=True, slots=True)
class GpuGuardConfig:
    required: bool = False
    device: str | None = None
    smoke: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> GpuGuardConfig:
        raw = raw or {}
        device = raw.get("device")
        if device is not None:
            device = _require_str(device, "gpu_guard.device")
        return cls(
            required=_require_bool(raw.get("required", False), "gpu_guard.required"),
            device=device,
            smoke=_require_bool(raw.get("smoke", False), "gpu_guard.smoke"),
        )

    def __post_init__(self) -> None:
        if self.device is not None and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("gpu_guard.device must be cpu, cuda, or cuda:<index>.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "device": self.device,
            "smoke": self.smoke,
        }


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table/object.")
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _optional_int(
    value: Any, field_name: str, *, minimum: int | None = None
) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name, minimum=minimum)


def _optional_float(
    value: Any, field_name: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return result


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return value


def _positive_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    values = tuple(
        _require_int(item, f"{field_name}[{index}]", minimum=1)
        for index, item in enumerate(_require_sequence(value, field_name))
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate user ids.")
    return values


def _float_tuple(
    value: Any, field_name: str, *, allow_empty: bool = False
) -> tuple[float, ...]:
    values = tuple(
        _optional_float(item, f"{field_name}[{index}]")
        for index, item in enumerate(_require_sequence(value, field_name))
    )
    if any(item is None for item in values):
        raise ValueError(f"{field_name} must contain only numbers.")
    result = tuple(float(item) for item in values if item is not None)
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicate values.")
    return result


def _str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    return tuple(
        _require_str(item, f"{field_name}[{index}]")
        for index, item in enumerate(_require_sequence(value, field_name))
    )


def _stage_tuple(value: Any, field_name: str) -> tuple[StageName, ...]:
    stages: list[StageName] = []
    for index, item in enumerate(_require_sequence(value, field_name)):
        raw = _require_str(item, f"{field_name}[{index}]")
        try:
            stages.append(StageName(raw))
        except ValueError as exc:
            allowed = ", ".join(stage.value for stage in StageName)
            raise ValueError(
                f"{field_name}[{index}] must be one of: {allowed}."
            ) from exc
    if not stages:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(stages)) != len(stages):
        raise ValueError(f"{field_name} must not contain duplicate stages.")
    return tuple(stages)


def _mapping_of_paths(value: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(path) for key, path in value.items()}


@dataclass(frozen=True, slots=True)
class UserSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...] = ()
    reserved_test: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> UserSplit:
        train = _positive_int_tuple(raw.get("train", []), "users.train")
        validation = _positive_int_tuple(raw.get("validation", []), "users.validation")
        reserved_test = _positive_int_tuple(
            raw.get("reserved_test", []), "users.reserved_test"
        )
        return cls(train=train, validation=validation, reserved_test=reserved_test)

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("users.train must contain at least one user.")
        groups = {
            "train": set(self.train),
            "validation": set(self.validation),
            "reserved_test": set(self.reserved_test),
        }
        overlaps = (
            groups["train"] & groups["validation"]
            or groups["train"] & groups["reserved_test"]
            or groups["validation"] & groups["reserved_test"]
        )
        if overlaps:
            raise ValueError("User split groups must be disjoint.")

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "reserved_test": list(self.reserved_test),
        }


@dataclass(frozen=True, slots=True)
class BaselineSource:
    scheduler: str
    log_root: Path
    expected_engine: str = "batched"
    stage_mode: str = "copy"
    desired_retention_values: tuple[float, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BaselineSource:
        return cls(
            scheduler=_require_str(raw.get("scheduler"), "baseline.scheduler"),
            log_root=Path(_require_str(raw.get("log_root"), "baseline.log_root")),
            expected_engine=_require_str(
                raw.get("expected_engine", "batched"), "baseline.expected_engine"
            ),
            stage_mode=_require_str(
                raw.get("stage_mode", "copy"), "baseline.stage_mode"
            ),
            desired_retention_values=_float_tuple(
                raw.get("desired_retention_values", []),
                "baseline.desired_retention_values",
                allow_empty=True,
            ),
        )

    def __post_init__(self) -> None:
        if self.scheduler != "fsrs6":
            raise ValueError("baseline.scheduler must be fsrs6 for formal gates.")
        if self.expected_engine not in {"event", "vectorized", "batched"}:
            raise ValueError("baseline.expected_engine is invalid.")
        if self.stage_mode not in {"copy", "hardlink"}:
            raise ValueError("baseline.stage_mode must be copy or hardlink.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler,
            "log_root": str(self.log_root),
            "expected_engine": self.expected_engine,
            "stage_mode": self.stage_mode,
            "desired_retention_values": list(self.desired_retention_values),
        }


@dataclass(frozen=True, slots=True)
class SimulationScope:
    engine: str
    days: int
    deck: int
    learn_limit: int | None = None
    review_limit: int | None = None
    cost_limit_minutes: float | None = None
    priority: str = "review-first"
    scheduler_priority: str = "low_retrievability"
    short_term_source: str | None = None
    fuzz: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SimulationScope:
        short_term_source = raw.get("short_term_source")
        if short_term_source is not None:
            short_term_source = _require_str(
                short_term_source, "simulation.short_term_source"
            )
        return cls(
            engine=_require_str(raw.get("engine"), "simulation.engine"),
            days=_require_int(raw.get("days"), "simulation.days", minimum=1),
            deck=_require_int(raw.get("deck"), "simulation.deck", minimum=1),
            learn_limit=_optional_int(
                raw.get("learn_limit"), "simulation.learn_limit", minimum=0
            ),
            review_limit=_optional_int(
                raw.get("review_limit"), "simulation.review_limit", minimum=0
            ),
            cost_limit_minutes=_optional_float(
                raw.get("cost_limit_minutes"),
                "simulation.cost_limit_minutes",
                minimum=0.0,
            ),
            priority=_require_str(
                raw.get("priority", "review-first"), "simulation.priority"
            ),
            scheduler_priority=_require_str(
                raw.get("scheduler_priority", "low_retrievability"),
                "simulation.scheduler_priority",
            ),
            short_term_source=short_term_source,
            fuzz=_require_bool(raw.get("fuzz", False), "simulation.fuzz"),
        )

    def __post_init__(self) -> None:
        if self.engine not in {"event", "vectorized", "batched"}:
            raise ValueError("simulation.engine is invalid.")
        if self.priority not in {"review-first", "new-first"}:
            raise ValueError("simulation.priority is invalid.")
        if self.short_term_source not in {None, "steps", "sched"}:
            raise ValueError("simulation.short_term_source is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "days": self.days,
            "deck": self.deck,
            "learn_limit": self.learn_limit,
            "review_limit": self.review_limit,
            "cost_limit_minutes": self.cost_limit_minutes,
            "priority": self.priority,
            "scheduler_priority": self.scheduler_priority,
            "short_term_source": self.short_term_source,
            "fuzz": self.fuzz,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    family: str
    seed: int
    output_root: Path
    stages: tuple[StageName, ...]
    users: UserSplit
    baseline: BaselineSource
    simulation: SimulationScope
    gpu_guard: GpuGuardConfig
    lambda_grid: tuple[float, ...]
    train_command_template: tuple[str, ...] = ()
    train_artifact_glob: str = "metadata.json"
    sweep_command_template: tuple[str, ...] = ()
    sweep_log_glob: str = "*.jsonl"
    pareto_command_template: tuple[str, ...] = ()
    pareto_result_glob: str = "*.json"
    pareto_plot_glob: str = "*.png"
    select_command_template: tuple[str, ...] = ()
    select_result_glob: str = "selection.json"
    config_path: Path | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentConfig:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_mapping(raw, config_path=path)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, config_path: Path | None = None
    ) -> ExperimentConfig:
        schema_version = _require_int(raw.get("schema_version"), "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION}, got {schema_version}."
            )
        training = _require_mapping(raw.get("training"), "training")
        sweep = _require_mapping(raw.get("sweep", {}), "sweep")
        pareto = _require_mapping(raw.get("pareto", {}), "pareto")
        select = _require_mapping(raw.get("select", {}), "select")
        return cls(
            name=_require_str(raw.get("name"), "name"),
            family=_require_str(raw.get("family"), "family"),
            seed=_require_int(raw.get("seed"), "seed", minimum=0),
            output_root=Path(_require_str(raw.get("output_root"), "output_root")),
            stages=_stage_tuple(raw.get("stages"), "stages"),
            users=UserSplit.from_mapping(_require_mapping(raw.get("users"), "users")),
            baseline=BaselineSource.from_mapping(
                _require_mapping(raw.get("baseline"), "baseline")
            ),
            simulation=SimulationScope.from_mapping(
                _require_mapping(raw.get("simulation"), "simulation")
            ),
            gpu_guard=GpuGuardConfig.from_mapping(raw.get("gpu_guard")),
            lambda_grid=_float_tuple(
                training.get("lambda_grid"), "training.lambda_grid"
            ),
            train_command_template=_str_tuple(
                training.get("command_template", []), "training.command_template"
            ),
            train_artifact_glob=_require_str(
                training.get("artifact_metadata_glob", "metadata.json"),
                "training.artifact_metadata_glob",
            ),
            sweep_command_template=_str_tuple(
                sweep.get("command_template", []), "sweep.command_template"
            ),
            sweep_log_glob=_require_str(
                sweep.get("log_glob", "*.jsonl"), "sweep.log_glob"
            ),
            pareto_command_template=_str_tuple(
                pareto.get("command_template", []), "pareto.command_template"
            ),
            pareto_result_glob=_require_str(
                pareto.get("result_glob", "*.json"), "pareto.result_glob"
            ),
            pareto_plot_glob=_require_str(
                pareto.get("plot_glob", "*.png"), "pareto.plot_glob"
            ),
            select_command_template=_str_tuple(
                select.get("command_template", []), "select.command_template"
            ),
            select_result_glob=_require_str(
                select.get("result_glob", "selection.json"), "select.result_glob"
            ),
            config_path=config_path,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "family": self.family,
            "seed": self.seed,
            "output_root": str(self.output_root),
            "stages": [stage.value for stage in self.stages],
            "users": self.users.to_dict(),
            "baseline": self.baseline.to_dict(),
            "simulation": self.simulation.to_dict(),
            "gpu_guard": self.gpu_guard.to_dict(),
            "training": {
                "lambda_grid": list(self.lambda_grid),
                "command_template": list(self.train_command_template),
                "artifact_metadata_glob": self.train_artifact_glob,
            },
            "sweep": {
                "command_template": list(self.sweep_command_template),
                "log_glob": self.sweep_log_glob,
            },
            "pareto": {
                "command_template": list(self.pareto_command_template),
                "result_glob": self.pareto_result_glob,
                "plot_glob": self.pareto_plot_glob,
            },
            "select": {
                "command_template": list(self.select_command_template),
                "result_glob": self.select_result_glob,
            },
            "config_path": str(self.config_path) if self.config_path else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class CommandRecord:
    command: tuple[str, ...]
    cwd: Path
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must not be empty.")
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise ValueError("exit_code must be an integer or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "cwd": str(self.cwd),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
        }


@dataclass(frozen=True, slots=True)
class GpuGuardSummary:
    passed: bool
    device: str
    workload_shape: Mapping[str, int] = field(default_factory=dict)
    batch_size: int | None = None
    elapsed_seconds: float | None = None
    simulator_calls_per_second: float | None = None
    peak_dedicated_memory_bytes: int | None = None
    shared_memory_growth_bytes: int | None = None
    fallback_used: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.peak_dedicated_memory_bytes is not None:
            _require_int(
                self.peak_dedicated_memory_bytes,
                "peak_dedicated_memory_bytes",
                minimum=0,
            )
        if self.shared_memory_growth_bytes is not None:
            _require_int(
                self.shared_memory_growth_bytes,
                "shared_memory_growth_bytes",
                minimum=0,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "device": self.device,
            "workload_shape": dict(self.workload_shape),
            "batch_size": self.batch_size,
            "elapsed_seconds": self.elapsed_seconds,
            "simulator_calls_per_second": self.simulator_calls_per_second,
            "peak_dedicated_memory_bytes": self.peak_dedicated_memory_bytes,
            "shared_memory_growth_bytes": self.shared_memory_growth_bytes,
            "fallback_used": self.fallback_used,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class GateSummary:
    gate_name: str
    passed: bool
    failures: tuple[FailureClass, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed and self.failures:
            raise ValueError("passed gate summaries must not list failures.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "failures": [failure.value for failure in self.failures],
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    run_id: str
    artifacts: Mapping[str, Path]
    config_snapshot_path: Path
    gate_summary_path: Path | None = None
    command_record_paths: tuple[Path, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "artifacts": _mapping_of_paths(self.artifacts),
            "config_snapshot_path": str(self.config_snapshot_path),
            "gate_summary_path": str(self.gate_summary_path)
            if self.gate_summary_path
            else None,
            "command_record_paths": [str(path) for path in self.command_record_paths],
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    stage: StageName
    command: CommandRecord
    config_path: Path
    config_snapshot_path: Path
    resolved_config_path: Path
    git_commit: str
    dirty: bool
    uv_lock_hash: str
    python_version: str
    torch_version: str
    cuda_version: str | None = None
    artifact_paths: tuple[Path, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage.value,
            "command": self.command.to_dict(),
            "config_path": str(self.config_path),
            "config_snapshot_path": str(self.config_snapshot_path),
            "resolved_config_path": str(self.resolved_config_path),
            "git_commit": self.git_commit,
            "dirty": self.dirty,
            "uv_lock_hash": self.uv_lock_hash,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "artifact_paths": [str(path) for path in self.artifact_paths],
        }
