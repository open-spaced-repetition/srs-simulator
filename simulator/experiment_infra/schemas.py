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
    BUILD_PARETO = "build-pareto"
    ANALYZE_PARETO = "analyze-pareto"
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
    TIMEOUT = "timeout"


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


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    device: str | None = None
    memory_budget_fraction: float | None = None
    nvml_sample_interval_seconds: float = 2.0
    timeout_seconds: float | None = None
    progress_interval_seconds: float = 30.0
    write_performance_summary: bool = True
    diagnostic_csv_logs: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> PerformanceConfig:
        raw = raw or {}
        device = raw.get("device")
        if device is not None:
            device = _require_str(device, "performance.device")
        nvml_sample_interval_seconds = _optional_float(
            raw.get("nvml_sample_interval_seconds", 2.0),
            "performance.nvml_sample_interval_seconds",
            minimum=0.0,
        )
        progress_interval_seconds = _optional_float(
            raw.get("progress_interval_seconds", 30.0),
            "performance.progress_interval_seconds",
            minimum=0.0,
        )
        return cls(
            device=device,
            memory_budget_fraction=_optional_float(
                raw.get("memory_budget_fraction"),
                "performance.memory_budget_fraction",
                minimum=0.0,
            ),
            nvml_sample_interval_seconds=float(nvml_sample_interval_seconds or 0.0),
            timeout_seconds=_optional_float(
                raw.get("timeout_seconds"),
                "performance.timeout_seconds",
                minimum=0.0,
            ),
            progress_interval_seconds=float(progress_interval_seconds or 0.0),
            write_performance_summary=_require_bool(
                raw.get("write_performance_summary", True),
                "performance.write_performance_summary",
            ),
            diagnostic_csv_logs=_require_bool(
                raw.get("diagnostic_csv_logs", False),
                "performance.diagnostic_csv_logs",
            ),
        )

    def __post_init__(self) -> None:
        if self.device is not None and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("performance.device must be cpu, cuda, or cuda:<index>.")
        if self.memory_budget_fraction is not None and not (
            0.0 < self.memory_budget_fraction <= 1.0
        ):
            raise ValueError("performance.memory_budget_fraction must be in (0, 1].")
        if (
            self.nvml_sample_interval_seconds is not None
            and self.nvml_sample_interval_seconds <= 0.0
        ):
            raise ValueError("performance.nvml_sample_interval_seconds must be > 0.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0.0:
            raise ValueError("performance.timeout_seconds must be > 0.")
        if (
            self.progress_interval_seconds is not None
            and self.progress_interval_seconds <= 0.0
        ):
            raise ValueError("performance.progress_interval_seconds must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "memory_budget_fraction": self.memory_budget_fraction,
            "nvml_sample_interval_seconds": self.nvml_sample_interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "progress_interval_seconds": self.progress_interval_seconds,
            "write_performance_summary": self.write_performance_summary,
            "diagnostic_csv_logs": self.diagnostic_csv_logs,
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


def _sweep_environment_batch_configs(
    value: Any,
    field_name: str,
) -> dict[str, SweepEnvironmentBatchConfig]:
    if value is None:
        return {}
    raw = _require_mapping(value, field_name)
    configs: dict[str, SweepEnvironmentBatchConfig] = {}
    for environment, item in raw.items():
        if not isinstance(environment, str) or not environment.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        path = f"{field_name}.{environment}"
        configs[environment] = SweepEnvironmentBatchConfig.from_mapping(
            _require_mapping(item, path),
            path=path,
        )
    return configs


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
    environments: tuple[str, ...] = ()

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
            environments=_str_tuple(
                raw.get("environments", []), "baseline.environments"
            ),
        )

    def __post_init__(self) -> None:
        if self.scheduler != "fsrs6":
            raise ValueError("baseline.scheduler must be fsrs6 for formal gates.")
        if self.expected_engine not in {"event", "batched"}:
            raise ValueError("baseline.expected_engine is invalid.")
        if self.stage_mode not in {"copy", "hardlink"}:
            raise ValueError("baseline.stage_mode must be copy or hardlink.")
        for environment in self.environments:
            if environment not in {
                "lstm",
                "fsrs6",
                "fsrs6_default",
                "fsrs3",
                "fsrs3_default",
            }:
                raise ValueError(
                    "baseline.environments contains an invalid environment."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler,
            "log_root": str(self.log_root),
            "expected_engine": self.expected_engine,
            "stage_mode": self.stage_mode,
            "desired_retention_values": list(self.desired_retention_values),
            "environments": list(self.environments),
        }


@dataclass(frozen=True, slots=True)
class SimulationScope:
    engine: str
    days: int
    deck: int
    environment: str = "lstm"
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
            environment=_require_str(
                raw.get("environment", "lstm"), "simulation.environment"
            ),
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
        if self.engine not in {"event", "batched"}:
            raise ValueError("simulation.engine is invalid.")
        if self.environment not in {
            "lstm",
            "fsrs6",
            "fsrs6_default",
            "fsrs3",
            "fsrs3_default",
        }:
            raise ValueError("simulation.environment is invalid.")
        if self.priority not in {"review-first", "new-first"}:
            raise ValueError("simulation.priority is invalid.")
        if self.short_term_source not in {None, "steps", "sched"}:
            raise ValueError("simulation.short_term_source is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "environment": self.environment,
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
class SweepEnvironmentBatchConfig:
    batch_size: int | None = None
    max_lanes_per_batch: int | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        path: str,
    ) -> SweepEnvironmentBatchConfig:
        return cls(
            batch_size=_optional_int(
                raw.get("batch_size"), f"{path}.batch_size", minimum=1
            ),
            max_lanes_per_batch=_optional_int(
                raw.get("max_lanes_per_batch"),
                f"{path}.max_lanes_per_batch",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_lanes_per_batch": self.max_lanes_per_batch,
        }


@dataclass(frozen=True, slots=True)
class BatchedSweepStageConfig:
    envs: tuple[str, ...] = ()
    schedulers: tuple[str, ...] = ()
    log_dir: Path | None = None
    log_layout: str = "user"
    batch_size: int | None = None
    max_lanes_per_batch: int | None = None
    env_overrides: Mapping[str, SweepEnvironmentBatchConfig] = field(
        default_factory=dict
    )
    torch_device: str | None = None
    cuda_devices: str | None = None
    benchmark_partition: str = "0"
    start_retention: float = 0.50
    end_retention: float = 0.98
    step: float = 0.02
    no_progress: bool = True
    no_log: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BatchedSweepStageConfig:
        log_dir = raw.get("log_dir")
        torch_device = raw.get("torch_device")
        cuda_devices = raw.get("cuda_devices")
        env_overrides = _sweep_environment_batch_configs(
            raw.get("env_overrides"),
            "sweep.env_overrides",
        )
        return cls(
            envs=_str_tuple(raw.get("envs", []), "sweep.envs"),
            schedulers=_str_tuple(raw.get("schedulers", []), "sweep.schedulers"),
            log_dir=Path(_require_str(log_dir, "sweep.log_dir"))
            if log_dir is not None
            else None,
            log_layout=_require_str(raw.get("log_layout", "user"), "sweep.log_layout"),
            batch_size=_optional_int(
                raw.get("batch_size"), "sweep.batch_size", minimum=1
            ),
            max_lanes_per_batch=_optional_int(
                raw.get("max_lanes_per_batch"), "sweep.max_lanes_per_batch", minimum=1
            ),
            env_overrides=env_overrides,
            torch_device=_require_str(torch_device, "sweep.torch_device")
            if torch_device is not None
            else None,
            cuda_devices=_require_str(cuda_devices, "sweep.cuda_devices")
            if cuda_devices is not None
            else None,
            benchmark_partition=_require_str(
                raw.get("benchmark_partition", "0"), "sweep.benchmark_partition"
            ),
            start_retention=float(
                _optional_float(
                    raw.get("start_retention", 0.50), "sweep.start_retention"
                )
                or 0.50
            ),
            end_retention=float(
                _optional_float(raw.get("end_retention", 0.98), "sweep.end_retention")
                or 0.98
            ),
            step=float(_optional_float(raw.get("step", 0.02), "sweep.step") or 0.02),
            no_progress=_require_bool(
                raw.get("no_progress", True), "sweep.no_progress"
            ),
            no_log=_require_bool(raw.get("no_log", False), "sweep.no_log"),
        )

    def __post_init__(self) -> None:
        for environment in self.envs:
            if environment not in {"lstm", "fsrs6", "fsrs6_default"}:
                raise ValueError("sweep.envs contains an invalid batched environment.")
        for environment in self.env_overrides:
            if environment not in {"lstm", "fsrs6", "fsrs6_default"}:
                raise ValueError(
                    "sweep.env_overrides contains an invalid batched environment."
                )
            if self.envs and environment not in self.envs:
                raise ValueError(
                    "sweep.env_overrides contains an environment not listed in sweep.envs."
                )
        if self.log_layout not in {"user", "sweep"}:
            raise ValueError("sweep.log_layout must be user or sweep.")
        if self.start_retention <= 0.0 or self.end_retention >= 1.0:
            raise ValueError("sweep retention bounds must satisfy 0 < value < 1.")
        if self.end_retention < self.start_retention:
            raise ValueError("sweep.end_retention must be >= sweep.start_retention.")
        if self.step <= 0.0:
            raise ValueError("sweep.step must be > 0.")
        if self.torch_device and self.cuda_devices:
            raise ValueError(
                "sweep.torch_device cannot be combined with sweep.cuda_devices."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "envs": list(self.envs),
            "schedulers": list(self.schedulers),
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "log_layout": self.log_layout,
            "batch_size": self.batch_size,
            "max_lanes_per_batch": self.max_lanes_per_batch,
            "env_overrides": {
                environment: config.to_dict()
                for environment, config in self.env_overrides.items()
            },
            "torch_device": self.torch_device,
            "cuda_devices": self.cuda_devices,
            "benchmark_partition": self.benchmark_partition,
            "start_retention": self.start_retention,
            "end_retention": self.end_retention,
            "step": self.step,
            "no_progress": self.no_progress,
            "no_log": self.no_log,
        }


@dataclass(frozen=True, slots=True)
class TrainingBatchConfig:
    enabled: bool = False
    trainer: str = "auto"
    batch_size: int | None = None
    max_lanes_per_batch: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> TrainingBatchConfig:
        raw = raw or {}
        return cls(
            enabled=_require_bool(raw.get("enabled", False), "training.batch.enabled"),
            trainer=_require_str(raw.get("trainer", "auto"), "training.batch.trainer"),
            batch_size=_optional_int(
                raw.get("batch_size"), "training.batch.batch_size", minimum=1
            ),
            max_lanes_per_batch=_optional_int(
                raw.get("max_lanes_per_batch"),
                "training.batch.max_lanes_per_batch",
                minimum=1,
            ),
        )

    def __post_init__(self) -> None:
        if self.trainer not in {
            "auto",
            "fsrs6_adr_portfolio",
            "fsrs6_adr_cmaes",
            "fsrs6_adp_cmaes",
            "fsrs6_adp_portfolio",
        }:
            raise ValueError(
                "training.batch.trainer must be auto, fsrs6_adr_cmaes, "
                "fsrs6_adr_portfolio, fsrs6_adp_cmaes, "
                "or fsrs6_adp_portfolio."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trainer": self.trainer,
            "batch_size": self.batch_size,
            "max_lanes_per_batch": self.max_lanes_per_batch,
        }


@dataclass(frozen=True, slots=True)
class BuildParetoConfig:
    command_template: tuple[str, ...] = ()
    envs: tuple[str, ...] = ()
    schedulers: tuple[str, ...] = ()
    log_dir: Path | None = None
    result_glob: str = "simulation_results_retention_sweep_user_*.json"
    plot_glob: str = "*.png"
    start_retention: float = 0.50
    end_retention: float = 0.98
    short_term: str = "off"
    short_term_source: str = "any"
    engine: str = "batched"
    max_parallel: int = 1
    compare_short_term: bool = False
    compare_engine: bool = False
    no_plot: bool = False
    hide_labels: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BuildParetoConfig:
        log_dir = raw.get("log_dir")
        return cls(
            command_template=_str_tuple(
                raw.get("command_template", []), "build_pareto.command_template"
            ),
            envs=_str_tuple(raw.get("envs", []), "build_pareto.envs"),
            schedulers=_str_tuple(raw.get("schedulers", []), "build_pareto.schedulers"),
            log_dir=Path(_require_str(log_dir, "build_pareto.log_dir"))
            if log_dir is not None
            else None,
            result_glob=_require_str(
                raw.get(
                    "result_glob", "simulation_results_retention_sweep_user_*.json"
                ),
                "build_pareto.result_glob",
            ),
            plot_glob=_require_str(
                raw.get("plot_glob", "*.png"), "build_pareto.plot_glob"
            ),
            start_retention=float(
                _optional_float(
                    raw.get("start_retention", 0.50), "build_pareto.start_retention"
                )
                or 0.50
            ),
            end_retention=float(
                _optional_float(
                    raw.get("end_retention", 0.98), "build_pareto.end_retention"
                )
                or 0.98
            ),
            short_term=_require_str(
                raw.get("short_term", "off"), "build_pareto.short_term"
            ),
            short_term_source=_require_str(
                raw.get("short_term_source", "any"), "build_pareto.short_term_source"
            ),
            engine=_require_str(raw.get("engine", "batched"), "build_pareto.engine"),
            max_parallel=_require_int(
                raw.get("max_parallel", 1), "build_pareto.max_parallel", minimum=1
            ),
            compare_short_term=_require_bool(
                raw.get("compare_short_term", False), "build_pareto.compare_short_term"
            ),
            compare_engine=_require_bool(
                raw.get("compare_engine", False), "build_pareto.compare_engine"
            ),
            no_plot=_require_bool(raw.get("no_plot", False), "build_pareto.no_plot"),
            hide_labels=_require_bool(
                raw.get("hide_labels", True), "build_pareto.hide_labels"
            ),
        )

    def __post_init__(self) -> None:
        if self.short_term not in {"on", "off", "any"}:
            raise ValueError("build_pareto.short_term must be on, off, or any.")
        if self.short_term_source not in {"steps", "sched", "any"}:
            raise ValueError(
                "build_pareto.short_term_source must be steps, sched, or any."
            )
        if self.engine not in {"event", "batched", "any"}:
            raise ValueError("build_pareto.engine is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_template": list(self.command_template),
            "envs": list(self.envs),
            "schedulers": list(self.schedulers),
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "result_glob": self.result_glob,
            "plot_glob": self.plot_glob,
            "start_retention": self.start_retention,
            "end_retention": self.end_retention,
            "short_term": self.short_term,
            "short_term_source": self.short_term_source,
            "engine": self.engine,
            "max_parallel": self.max_parallel,
            "compare_short_term": self.compare_short_term,
            "compare_engine": self.compare_engine,
            "no_plot": self.no_plot,
            "hide_labels": self.hide_labels,
        }


@dataclass(frozen=True, slots=True)
class AnalyzeParetoConfig:
    command_template: tuple[str, ...] = ()
    envs: tuple[str, ...] = ()
    schedulers: tuple[str, ...] = ()
    comparisons: tuple[str, ...] = ()
    log_dir: Path | None = None
    result_glob: str = "analysis.md"
    start_retention: float = 0.50
    end_retention: float = 0.98
    short_term: str = "off"
    engine: str = "batched"
    fuzz: str = "off"
    metric: str = "avg_accum_memorized_per_hour"
    no_dedupe: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AnalyzeParetoConfig:
        log_dir = raw.get("log_dir")
        return cls(
            command_template=_str_tuple(
                raw.get("command_template", []), "analyze_pareto.command_template"
            ),
            envs=_str_tuple(raw.get("envs", []), "analyze_pareto.envs"),
            schedulers=_str_tuple(
                raw.get("schedulers", []), "analyze_pareto.schedulers"
            ),
            comparisons=_str_tuple(
                raw.get("comparisons", []), "analyze_pareto.comparisons"
            ),
            log_dir=Path(_require_str(log_dir, "analyze_pareto.log_dir"))
            if log_dir is not None
            else None,
            result_glob=_require_str(
                raw.get("result_glob", "analysis.md"), "analyze_pareto.result_glob"
            ),
            start_retention=float(
                _optional_float(
                    raw.get("start_retention", 0.50), "analyze_pareto.start_retention"
                )
                or 0.50
            ),
            end_retention=float(
                _optional_float(
                    raw.get("end_retention", 0.98), "analyze_pareto.end_retention"
                )
                or 0.98
            ),
            short_term=_require_str(
                raw.get("short_term", "off"), "analyze_pareto.short_term"
            ),
            engine=_require_str(raw.get("engine", "batched"), "analyze_pareto.engine"),
            fuzz=_require_str(raw.get("fuzz", "off"), "analyze_pareto.fuzz"),
            metric=_require_str(
                raw.get("metric", "avg_accum_memorized_per_hour"),
                "analyze_pareto.metric",
            ),
            no_dedupe=_require_bool(
                raw.get("no_dedupe", False), "analyze_pareto.no_dedupe"
            ),
        )

    def __post_init__(self) -> None:
        if self.short_term not in {"on", "off", "any"}:
            raise ValueError("analyze_pareto.short_term must be on, off, or any.")
        if self.engine not in {"event", "batched", "any"}:
            raise ValueError("analyze_pareto.engine is invalid.")
        if self.fuzz not in {"on", "off", "any"}:
            raise ValueError("analyze_pareto.fuzz must be on, off, or any.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_template": list(self.command_template),
            "envs": list(self.envs),
            "schedulers": list(self.schedulers),
            "comparisons": list(self.comparisons),
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "result_glob": self.result_glob,
            "start_retention": self.start_retention,
            "end_retention": self.end_retention,
            "short_term": self.short_term,
            "engine": self.engine,
            "fuzz": self.fuzz,
            "metric": self.metric,
            "no_dedupe": self.no_dedupe,
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
    performance: PerformanceConfig
    lambda_grid: tuple[float, ...]
    training_policy_search: Mapping[str, Any] = field(default_factory=dict)
    training_portfolio: Mapping[str, Any] = field(default_factory=dict)
    training_adp: Mapping[str, Any] = field(default_factory=dict)
    training_optimizer: Mapping[str, Any] = field(default_factory=dict)
    training_batch: TrainingBatchConfig = field(default_factory=TrainingBatchConfig)
    train_command_template: tuple[str, ...] = ()
    train_artifact_glob: str = "metadata.json"
    train_max_parallel_commands: int = 1
    train_batch_baseline_desired_retention_values: bool = False
    sweep_command_template: tuple[str, ...] = ()
    sweep_log_glob: str = "*.jsonl"
    sweep_batch_scheduler_artifacts: bool = False
    sweep_batched: BatchedSweepStageConfig = field(
        default_factory=BatchedSweepStageConfig
    )
    build_pareto: BuildParetoConfig = field(default_factory=BuildParetoConfig)
    analyze_pareto: AnalyzeParetoConfig = field(default_factory=AnalyzeParetoConfig)
    pareto_command_template: tuple[str, ...] = ()
    pareto_result_glob: str = "*.json"
    pareto_plot_glob: str = "*.png"
    select_command_template: tuple[str, ...] = ()
    select_result_glob: str = "selection.json"
    aggregate_command_template: tuple[str, ...] = ()
    aggregate_result_glob: str = "aggregate.json"
    reserved_test_command_template: tuple[str, ...] = ()
    reserved_test_log_glob: str = "*.jsonl"
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
        build_pareto = _require_mapping(raw.get("build_pareto", {}), "build_pareto")
        analyze_pareto = _require_mapping(
            raw.get("analyze_pareto", {}), "analyze_pareto"
        )
        pareto = _require_mapping(raw.get("pareto", {}), "pareto")
        select = _require_mapping(raw.get("select", {}), "select")
        aggregate = _require_mapping(raw.get("aggregate", {}), "aggregate")
        reserved_test = _require_mapping(raw.get("reserved_test", {}), "reserved_test")
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
            performance=PerformanceConfig.from_mapping(raw.get("performance")),
            lambda_grid=_float_tuple(
                training.get("lambda_grid"), "training.lambda_grid"
            ),
            training_policy_search=dict(
                _require_mapping(
                    training.get("policy_search", {}),
                    "training.policy_search",
                )
            ),
            training_portfolio=dict(
                _require_mapping(training.get("portfolio", {}), "training.portfolio")
            ),
            training_adp=dict(
                _require_mapping(training.get("adp", {}), "training.adp")
            ),
            training_optimizer=dict(
                _require_mapping(training.get("optimizer", {}), "training.optimizer")
            ),
            training_batch=TrainingBatchConfig.from_mapping(
                _require_mapping(training.get("batch", {}), "training.batch")
            ),
            train_command_template=_str_tuple(
                training.get("command_template", []), "training.command_template"
            ),
            train_artifact_glob=_require_str(
                training.get("artifact_metadata_glob", "metadata.json"),
                "training.artifact_metadata_glob",
            ),
            train_max_parallel_commands=_require_int(
                training.get("max_parallel_commands", 1),
                "training.max_parallel_commands",
                minimum=1,
            ),
            train_batch_baseline_desired_retention_values=_require_bool(
                training.get("batch_baseline_desired_retention_values", False),
                "training.batch_baseline_desired_retention_values",
            ),
            sweep_command_template=_str_tuple(
                sweep.get("command_template", []), "sweep.command_template"
            ),
            sweep_log_glob=_require_str(
                sweep.get("log_glob", "*.jsonl"), "sweep.log_glob"
            ),
            sweep_batch_scheduler_artifacts=_require_bool(
                sweep.get("batch_scheduler_artifacts", False),
                "sweep.batch_scheduler_artifacts",
            ),
            sweep_batched=BatchedSweepStageConfig.from_mapping(sweep),
            build_pareto=BuildParetoConfig.from_mapping(build_pareto),
            analyze_pareto=AnalyzeParetoConfig.from_mapping(analyze_pareto),
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
            aggregate_command_template=_str_tuple(
                aggregate.get("command_template", []),
                "aggregate.command_template",
            ),
            aggregate_result_glob=_require_str(
                aggregate.get("result_glob", "aggregate.json"),
                "aggregate.result_glob",
            ),
            reserved_test_command_template=_str_tuple(
                reserved_test.get("command_template", []),
                "reserved_test.command_template",
            ),
            reserved_test_log_glob=_require_str(
                reserved_test.get("log_glob", "*.jsonl"),
                "reserved_test.log_glob",
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
            "performance": self.performance.to_dict(),
            "training": {
                "lambda_grid": list(self.lambda_grid),
                "policy_search": dict(self.training_policy_search),
                "portfolio": dict(self.training_portfolio),
                "adp": dict(self.training_adp),
                "optimizer": dict(self.training_optimizer),
                "batch": self.training_batch.to_dict(),
                "command_template": list(self.train_command_template),
                "artifact_metadata_glob": self.train_artifact_glob,
                "max_parallel_commands": self.train_max_parallel_commands,
                "batch_baseline_desired_retention_values": (
                    self.train_batch_baseline_desired_retention_values
                ),
            },
            "sweep": {
                "command_template": list(self.sweep_command_template),
                "log_glob": self.sweep_log_glob,
                "batch_scheduler_artifacts": self.sweep_batch_scheduler_artifacts,
                **self.sweep_batched.to_dict(),
            },
            "build_pareto": self.build_pareto.to_dict(),
            "analyze_pareto": {
                **self.analyze_pareto.to_dict(),
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
            "aggregate": {
                "command_template": list(self.aggregate_command_template),
                "result_glob": self.aggregate_result_glob,
            },
            "reserved_test": {
                "command_template": list(self.reserved_test_command_template),
                "log_glob": self.reserved_test_log_glob,
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
class PerformanceSummary:
    stage: StageName
    passed: bool
    device: str
    workload_shape: Mapping[str, Any] = field(default_factory=dict)
    execution_shape: Mapping[str, Any] = field(default_factory=dict)
    runtime_metrics: Mapping[str, Any] = field(default_factory=dict)
    gpu_metrics: Mapping[str, Any] = field(default_factory=dict)
    disk_metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_class: FailureClass | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "passed": self.passed,
            "device": self.device,
            "workload_shape": dict(self.workload_shape),
            "execution_shape": dict(self.execution_shape),
            "runtime_metrics": dict(self.runtime_metrics),
            "gpu_metrics": dict(self.gpu_metrics),
            "disk_metrics": dict(self.disk_metrics),
            "failure_class": self.failure_class.value if self.failure_class else None,
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
