from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.fsrs6_oracle_stationary_finite_distill_policy import (
    FSRS6OracleStationaryFiniteDistillPolicy,
)


SCHEDULER_NAME = "fsrs6_oracle_stationary_finite_distill"


@dataclass(frozen=True, slots=True)
class FSRS6OracleStationaryFiniteDistillPolicySpec:
    user_id: int
    goal_cost_weight: float
    policy_index: int | None
    path: Path


def resolve_fsrs6_oracle_stationary_finite_distill_policy_specs(
    *,
    user_ids: Sequence[int],
    policy_root: Path | None = None,
    train_run_root: Path | None = None,
    policy_manifest: Path | None = None,
) -> tuple[FSRS6OracleStationaryFiniteDistillPolicySpec, ...]:
    sources = [
        policy_root is not None,
        train_run_root is not None,
        policy_manifest is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Configure exactly one FSRS6 oracle stationary finite distill policy "
            "source: policy_root, train_run_root, or policy_manifest."
        )
    if train_run_root is not None:
        policy_root = train_run_root / "train-overfit" / "train_outputs"
    if policy_manifest is not None:
        return _load_policy_manifest(policy_manifest=policy_manifest, user_ids=user_ids)
    if policy_root is None:
        raise AssertionError("policy_root must be set after source validation.")
    return _discover_policy_root(policy_root=policy_root, user_ids=user_ids)


def _discover_policy_root(
    *,
    policy_root: Path,
    user_ids: Sequence[int],
) -> tuple[FSRS6OracleStationaryFiniteDistillPolicySpec, ...]:
    root = policy_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"FSRS6 oracle stationary finite distill policy root does not exist: {root}"
        )
    user_set = set(user_ids)
    specs: list[FSRS6OracleStationaryFiniteDistillPolicySpec] = []
    for policy_path in sorted(root.rglob("policy.json")):
        path_user_id = _extract_path_int(policy_path, "user_")
        if path_user_id is not None and path_user_id not in user_set:
            continue
        spec = _spec_from_policy_path(policy_path)
        if spec.user_id not in user_set:
            continue
        specs.append(spec)
    if not specs:
        raise FileNotFoundError(
            "No FSRS6 oracle stationary finite distill policies under "
            f"{root} matched users={list(user_ids)}."
        )
    _reject_duplicate_policy_specs(specs)
    return tuple(specs)


def _load_policy_manifest(
    *,
    policy_manifest: Path,
    user_ids: Sequence[int],
) -> tuple[FSRS6OracleStationaryFiniteDistillPolicySpec, ...]:
    manifest_path = policy_manifest.expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            "FSRS6 oracle stationary finite distill policy manifest does not exist: "
            f"{manifest_path}"
        )
    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)
    entries = _manifest_entries(raw)
    user_set = set(user_ids)
    specs: list[FSRS6OracleStationaryFiniteDistillPolicySpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"policies[{index}] must be a TOML table.")
        user_id = _require_int(entry.get("user_id"), f"policies[{index}].user_id")
        if user_id not in user_set:
            raise ValueError(
                f"Policy manifest entry {index} uses user_id={user_id}, "
                f"which is outside configured users {list(user_ids)}."
            )
        path = _require_path(
            entry.get("path"),
            f"policies[{index}].path",
            base_path=manifest_path.parent,
        )
        specs.append(
            _validate_policy_spec(
                path=path,
                user_id=user_id,
                policy_index=_optional_int(
                    entry.get("policy_index"), f"policies[{index}].policy_index"
                ),
                source=f"manifest entry {index}",
            )
        )
    if not specs:
        raise ValueError(f"Policy manifest {manifest_path} did not produce any lanes.")
    _reject_duplicate_policy_specs(specs)
    return tuple(specs)


def _manifest_entries(raw: Mapping[str, Any]) -> Sequence[Any]:
    entries = raw.get("policies")
    if isinstance(entries, Mapping):
        entries = entries.get("entries")
    if entries is None:
        entries = raw.get("policy")
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise ValueError("Policy manifest must contain [[policies]] entries.")
    return entries


def _spec_from_policy_path(
    policy_path: Path,
) -> FSRS6OracleStationaryFiniteDistillPolicySpec:
    policy = FSRS6OracleStationaryFiniteDistillPolicy.from_json(policy_path)
    metadata = _load_sibling_metadata(policy_path)
    path_user_id = _extract_path_int(policy_path, "user_")
    path_policy_index = _extract_path_int(policy_path, "policy_")
    metadata_user_id = _metadata_user_id(metadata, policy_path)
    metadata_policy_index = _metadata_int(metadata, "portfolio_index", policy_path)

    user_id = policy.user_id
    if user_id is None:
        user_id = metadata_user_id if metadata_user_id is not None else path_user_id
    if user_id is None:
        raise ValueError(
            f"Could not infer user_id for FSRS6 oracle stationary finite distill "
            f"policy {policy_path}. Use a user_<id> path component, policy JSON "
            "user_id, or metadata.json."
        )
    for source, candidate in (
        ("metadata", metadata_user_id),
        ("path", path_user_id),
    ):
        if candidate is not None and candidate != user_id:
            raise ValueError(
                f"Policy {policy_path} user_id mismatch: policy has {user_id}, "
                f"{source} has {candidate}."
            )
    return FSRS6OracleStationaryFiniteDistillPolicySpec(
        user_id=user_id,
        goal_cost_weight=policy.goal_cost_weight,
        policy_index=metadata_policy_index
        if metadata_policy_index is not None
        else policy.portfolio_index
        if policy.portfolio_index is not None
        else path_policy_index,
        path=policy_path.resolve(),
    )


def _validate_policy_spec(
    *,
    path: Path,
    user_id: int,
    policy_index: int | None,
    source: str,
) -> FSRS6OracleStationaryFiniteDistillPolicySpec:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FSRS6 oracle stationary finite distill policy for {source}: {path}"
        )
    spec = _spec_from_policy_path(path)
    if spec.user_id != user_id:
        raise ValueError(
            f"Policy {path} has user_id={spec.user_id}, expected {user_id} from "
            f"{source}."
        )
    return FSRS6OracleStationaryFiniteDistillPolicySpec(
        user_id=user_id,
        goal_cost_weight=spec.goal_cost_weight,
        policy_index=policy_index if policy_index is not None else spec.policy_index,
        path=path.resolve(),
    )


def _reject_duplicate_policy_specs(
    specs: Sequence[FSRS6OracleStationaryFiniteDistillPolicySpec],
) -> None:
    by_key: dict[tuple[int, int | None], Path] = {}
    for spec in specs:
        key = (spec.user_id, spec.policy_index)
        previous = by_key.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate FSRS6 oracle stationary finite distill policy for "
                f"user={spec.user_id}, policy_index={spec.policy_index}: "
                f"{previous} and {spec.path}"
            )
        by_key[key] = spec.path


def _load_sibling_metadata(path: Path) -> Mapping[str, Any] | None:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Artifact metadata must be a JSON object: {metadata_path}")
    scheduler_name = raw.get("scheduler_name")
    if scheduler_name is not None and scheduler_name != SCHEDULER_NAME:
        raise ValueError(
            f"Artifact metadata {metadata_path} scheduler_name={scheduler_name!r}; "
            f"expected {SCHEDULER_NAME!r}."
        )
    policy_path_raw = raw.get("policy_path")
    if isinstance(policy_path_raw, str) and policy_path_raw.strip():
        metadata_policy_path = Path(policy_path_raw)
        if not metadata_policy_path.is_absolute():
            metadata_policy_path = metadata_path.parent / metadata_policy_path
        if metadata_policy_path.resolve() != path.resolve():
            raise ValueError(
                f"Artifact metadata {metadata_path} points to "
                f"{metadata_policy_path}, not {path}."
            )
    return raw


def _metadata_user_id(metadata: Mapping[str, Any] | None, path: Path) -> int | None:
    if metadata is None or "training_user_ids" not in metadata:
        return None
    raw = metadata["training_user_ids"]
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ValueError(f"metadata training_user_ids must be an array: {path}")
    if len(raw) != 1:
        raise ValueError(
            f"metadata training_user_ids must contain exactly one user for {path}."
        )
    return _require_int(raw[0], "metadata.training_user_ids[0]")


def _metadata_int(
    metadata: Mapping[str, Any] | None,
    key: str,
    path: Path,
) -> int | None:
    if metadata is None or key not in metadata:
        return None
    return _optional_int(metadata[key], f"metadata.{key} for {path}")


def _extract_path_int(path: Path, prefix: str) -> int | None:
    for part in reversed(path.parts):
        if part.startswith(prefix):
            token = part[len(prefix) :]
            try:
                return int(token)
            except ValueError:
                continue
    return None


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, field_name)


def _require_path(value: Any, field_name: str, *, base_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


__all__ = [
    "FSRS6OracleStationaryFiniteDistillPolicySpec",
    "resolve_fsrs6_oracle_stationary_finite_distill_policy_specs",
]
