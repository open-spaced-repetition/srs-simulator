from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DR_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BaselineDRManifestEntry:
    user_id: int
    desired_retention_values: tuple[float, ...]
    objective: Mapping[str, Any]
    reference_point: Mapping[str, Any]
    selected_metrics: tuple[Mapping[str, Any], ...]
    optimizer: Mapping[str, Any]
    config_snapshot: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BaselineDRManifest:
    path: Path
    entries_by_user: Mapping[int, BaselineDRManifestEntry]
    target_count: int | None = None
    selection_environment: str | None = None
    reference: str | None = None
    optimizer: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def values_for_user(self, user_id: int) -> tuple[float, ...]:
        entry = self.entries_by_user.get(int(user_id))
        if entry is None:
            raise KeyError(f"Baseline DR manifest {self.path} has no user {user_id}.")
        return entry.desired_retention_values

    def contains_value(
        self,
        user_id: int,
        desired_retention: float,
        *,
        tolerance: float = DEFAULT_DR_TOLERANCE,
    ) -> bool:
        try:
            values = self.values_for_user(user_id)
        except KeyError:
            return False
        return any(
            math.isclose(
                float(desired_retention),
                expected,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for expected in values
        )


def load_baseline_dr_manifest(
    path: Path,
    *,
    target_count: int | None = None,
    user_ids: Sequence[int] | None = None,
    retention_min: float = 0.0,
    retention_max: float = 1.0,
    tolerance: float = DEFAULT_DR_TOLERANCE,
) -> BaselineDRManifest:
    manifest_path = path.expanduser()
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read baseline DR manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Baseline DR manifest must be a JSON object.")

    raw_entries = payload.get("users", payload.get("entries"))
    if isinstance(raw_entries, Mapping):
        raw_entries = [
            {"user_id": user_id, **entry}
            for user_id, entry in raw_entries.items()
            if isinstance(entry, Mapping)
        ]
    if isinstance(raw_entries, str) or not isinstance(raw_entries, Sequence):
        raise ValueError("Baseline DR manifest must contain a users array.")

    raw_target_count = payload.get("target_count")
    manifest_target_count = _optional_int(raw_target_count, "target_count")
    if (
        target_count is not None
        and manifest_target_count is not None
        and manifest_target_count != target_count
    ):
        raise ValueError(
            "Baseline DR manifest target_count expected "
            f"{target_count}, got {manifest_target_count}."
        )
    expected_target_count = (
        target_count if target_count is not None else manifest_target_count
    )

    entries: dict[int, BaselineDRManifestEntry] = {}
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Baseline DR manifest users[{index}] must be an object.")
        entry = _parse_manifest_entry(
            raw_entry,
            index=index,
            target_count=expected_target_count,
            retention_min=retention_min,
            retention_max=retention_max,
            tolerance=tolerance,
        )
        if entry.user_id in entries:
            raise ValueError(
                f"Baseline DR manifest contains duplicate user_id {entry.user_id}."
            )
        entries[entry.user_id] = entry

    if user_ids is not None:
        missing = [int(user_id) for user_id in user_ids if int(user_id) not in entries]
        if missing:
            raise ValueError(
                "Baseline DR manifest is missing users: "
                + ", ".join(str(user_id) for user_id in missing)
            )

    raw_selection_environment = payload.get("selection_environment")
    selection_environment = (
        str(raw_selection_environment)
        if raw_selection_environment is not None
        else None
    )
    raw_reference = payload.get("reference")
    reference = str(raw_reference) if raw_reference is not None else None
    raw_optimizer = payload.get("optimizer")
    optimizer = raw_optimizer if isinstance(raw_optimizer, Mapping) else None
    metadata = payload.get("metadata")
    return BaselineDRManifest(
        path=manifest_path,
        entries_by_user=entries,
        target_count=manifest_target_count,
        selection_environment=selection_environment,
        reference=reference,
        optimizer=optimizer,
        metadata=metadata if isinstance(metadata, Mapping) else None,
    )


def _parse_manifest_entry(
    raw: Mapping[str, Any],
    *,
    index: int,
    target_count: int | None,
    retention_min: float,
    retention_max: float,
    tolerance: float,
) -> BaselineDRManifestEntry:
    user_id = _int(raw.get("user_id"), f"users[{index}].user_id")
    raw_values = raw.get(
        "desired_retention_values",
        raw.get("desired_retention_values_sorted"),
    )
    values = _float_tuple(raw_values, f"users[{index}].desired_retention_values")
    expected_count = target_count
    if expected_count is None:
        expected_count = _optional_int(
            raw.get("target_count"), f"users[{index}].target_count"
        )
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"users[{index}].desired_retention_values must contain "
            f"{expected_count} values, got {len(values)}."
        )
    _validate_retention_values(
        values,
        field_name=f"users[{index}].desired_retention_values",
        retention_min=retention_min,
        retention_max=retention_max,
        tolerance=tolerance,
    )
    return BaselineDRManifestEntry(
        user_id=user_id,
        desired_retention_values=values,
        objective=_mapping_or_empty(raw.get("objective")),
        reference_point=_mapping_or_empty(raw.get("reference_point")),
        selected_metrics=tuple(
            item
            for item in _sequence_or_empty(raw.get("selected_metrics"))
            if isinstance(item, Mapping)
        ),
        optimizer=_mapping_or_empty(raw.get("optimizer")),
        config_snapshot=_mapping_or_empty(raw.get("config_snapshot")),
    )


def _validate_retention_values(
    values: tuple[float, ...],
    *,
    field_name: str,
    retention_min: float,
    retention_max: float,
    tolerance: float,
) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    if any(value <= retention_min or value >= retention_max for value in values):
        raise ValueError(
            f"{field_name} values must satisfy {retention_min} < value < {retention_max}."
        )
    for previous, current in zip(values, values[1:], strict=False):
        if current < previous:
            raise ValueError(f"{field_name} must be sorted in ascending order.")
        if math.isclose(current, previous, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{field_name} must not contain duplicate values.")


def _int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().isdigit():
            return int(value)
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _int(value, field_name)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    values: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (float, int)):
            raise ValueError(f"{field_name}[{index}] must be a number.")
        values.append(float(item))
    return tuple(values)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return value


__all__ = [
    "BaselineDRManifest",
    "BaselineDRManifestEntry",
    "DEFAULT_DR_TOLERANCE",
    "load_baseline_dr_manifest",
]
