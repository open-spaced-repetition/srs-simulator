from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from simulator.experiment_infra.capabilities import EngineName, supports_scheduler
from simulator.experiment_infra.schemas import SCHEMA_VERSION


class ArtifactKind(StrEnum):
    SCHEDULER_POLICY = "scheduler-policy"


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


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number or null.")
    return float(value)


def _require_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return value


def _int_tuple(
    value: Any, field_name: str, *, require_nonempty: bool
) -> tuple[int, ...]:
    values = tuple(
        _require_int(item, f"{field_name}[{index}]", minimum=1)
        for index, item in enumerate(_require_sequence(value, field_name))
    )
    if require_nonempty and not values:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate user ids.")
    return values


def _str_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    values = tuple(
        _require_str(item, f"{field_name}[{index}]")
        for index, item in enumerate(_require_sequence(value, field_name))
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate values.")
    return values


@dataclass(frozen=True, slots=True)
class SchedulerArtifactMetadata:
    artifact_id: str
    family: str
    scheduler_name: str
    environment: str
    engine: EngineName
    training_user_ids: tuple[int, ...]
    validation_user_ids: tuple[int, ...]
    seed: int
    policy_path: Path
    feature_version: str
    action_space: str
    created_at: str
    code_commit: str
    lambda_value: float | None = None
    baseline_desired_retention: float | None = None
    config_snapshot_path: Path | None = None
    training_command_path: Path | None = None
    metrics_path: Path | None = None
    capabilities: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    artifact_kind: ArtifactKind = ArtifactKind.SCHEDULER_POLICY

    @classmethod
    def from_json(cls, path: Path) -> SchedulerArtifactMetadata:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("Artifact metadata JSON must be an object.")
        return cls.from_mapping(raw, metadata_path=path)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        metadata_path: Path | None = None,
    ) -> SchedulerArtifactMetadata:
        schema_version = _require_int(raw.get("schema_version"), "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION}, got {schema_version}."
            )
        try:
            artifact_kind = ArtifactKind(
                _require_str(raw.get("artifact_kind"), "artifact_kind")
            )
        except ValueError as exc:
            raise ValueError("artifact_kind must be scheduler-policy.") from exc
        base_path = metadata_path.parent if metadata_path is not None else None
        return cls(
            artifact_id=_require_str(raw.get("artifact_id"), "artifact_id"),
            family=_require_str(raw.get("family"), "family"),
            scheduler_name=_require_str(raw.get("scheduler_name"), "scheduler_name"),
            environment=_require_str(raw.get("environment"), "environment"),
            engine=EngineName(_require_str(raw.get("engine"), "engine")),
            training_user_ids=_int_tuple(
                raw.get("training_user_ids", []),
                "training_user_ids",
                require_nonempty=True,
            ),
            validation_user_ids=_int_tuple(
                raw.get("validation_user_ids", []),
                "validation_user_ids",
                require_nonempty=False,
            ),
            seed=_require_int(raw.get("seed"), "seed", minimum=0),
            policy_path=_resolve_metadata_path(
                _require_str(raw.get("policy_path"), "policy_path"),
                base_path=base_path,
            ),
            feature_version=_require_str(raw.get("feature_version"), "feature_version"),
            action_space=_require_str(raw.get("action_space"), "action_space"),
            created_at=_require_str(raw.get("created_at"), "created_at"),
            code_commit=_require_str(raw.get("code_commit"), "code_commit"),
            lambda_value=_optional_float(raw.get("lambda_value"), "lambda_value"),
            baseline_desired_retention=_optional_float(
                raw.get("baseline_desired_retention"),
                "baseline_desired_retention",
            ),
            config_snapshot_path=_optional_path(
                raw.get("config_snapshot_path"),
                field_name="config_snapshot_path",
                base_path=base_path,
            ),
            training_command_path=_optional_path(
                raw.get("training_command_path"),
                field_name="training_command_path",
                base_path=base_path,
            ),
            metrics_path=_optional_path(
                raw.get("metrics_path"),
                field_name="metrics_path",
                base_path=base_path,
            ),
            capabilities=_str_tuple(raw.get("capabilities", []), "capabilities"),
            schema_version=schema_version,
            artifact_kind=artifact_kind,
        )

    def __post_init__(self) -> None:
        if set(self.training_user_ids) & set(self.validation_user_ids):
            raise ValueError(
                "training_user_ids and validation_user_ids must be disjoint."
            )
        if not supports_scheduler(
            scheduler=self.scheduler_name,
            engine=self.engine.value,
            environment=self.environment,
        ):
            raise ValueError(
                f"{self.scheduler_name} does not support "
                f"engine={self.engine.value}, environment={self.environment}."
            )

    def validate_files(self) -> None:
        required_paths = {"policy_path": self.policy_path}
        for name, path in required_paths.items():
            if not path.exists():
                raise ValueError(f"{name} does not exist: {path}")
        for name, path in (
            ("config_snapshot_path", self.config_snapshot_path),
            ("training_command_path", self.training_command_path),
            ("metrics_path", self.metrics_path),
        ):
            if path is not None and not path.exists():
                raise ValueError(f"{name} does not exist: {path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind.value,
            "artifact_id": self.artifact_id,
            "family": self.family,
            "scheduler_name": self.scheduler_name,
            "environment": self.environment,
            "engine": self.engine.value,
            "training_user_ids": list(self.training_user_ids),
            "validation_user_ids": list(self.validation_user_ids),
            "seed": self.seed,
            "policy_path": str(self.policy_path),
            "feature_version": self.feature_version,
            "action_space": self.action_space,
            "created_at": self.created_at,
            "code_commit": self.code_commit,
            "lambda_value": self.lambda_value,
            "baseline_desired_retention": self.baseline_desired_retention,
            "config_snapshot_path": str(self.config_snapshot_path)
            if self.config_snapshot_path
            else None,
            "training_command_path": str(self.training_command_path)
            if self.training_command_path
            else None,
            "metrics_path": str(self.metrics_path) if self.metrics_path else None,
            "capabilities": list(self.capabilities),
        }


def validate_scheduler_artifact(
    metadata_path: Path, *, require_files: bool = False
) -> SchedulerArtifactMetadata:
    metadata = SchedulerArtifactMetadata.from_json(metadata_path)
    if require_files:
        metadata.validate_files()
    return metadata


def _resolve_metadata_path(value: str, *, base_path: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or base_path is None:
        return path
    return (base_path / path).resolve()


def _optional_path(
    value: Any,
    *,
    field_name: str,
    base_path: Path | None,
) -> Path | None:
    if value is None:
        return None
    return _resolve_metadata_path(_require_str(value, field_name), base_path=base_path)
