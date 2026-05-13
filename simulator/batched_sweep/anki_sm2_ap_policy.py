from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.anki_sm2_ap_policy import AnkiSM2APPolicy


@dataclass(frozen=True, slots=True)
class AnkiSM2APPolicySpec:
    user_id: int
    policy_index: int | None
    path: Path


def resolve_anki_sm2_ap_policy_specs(
    *,
    user_ids: Sequence[int],
    policy_root: Path | None = None,
    train_run_root: Path | None = None,
    policy_manifest: Path | None = None,
) -> tuple[AnkiSM2APPolicySpec, ...]:
    sources = [
        policy_root is not None,
        train_run_root is not None,
        policy_manifest is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Configure exactly one Anki SM2 AP policy source: "
            "policy_root, train_run_root, or policy_manifest."
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
) -> tuple[AnkiSM2APPolicySpec, ...]:
    root = policy_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Anki SM2 AP policy root does not exist: {root}")
    user_set = set(user_ids)
    specs: list[AnkiSM2APPolicySpec] = []
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
            f"No Anki SM2 AP policies under {root} matched users={list(user_ids)}."
        )
    _reject_duplicate_policy_specs(specs)
    _require_all_users(specs=specs, user_ids=user_ids)
    return tuple(specs)


def _load_policy_manifest(
    *,
    policy_manifest: Path,
    user_ids: Sequence[int],
) -> tuple[AnkiSM2APPolicySpec, ...]:
    manifest_path = policy_manifest.expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Anki SM2 AP policy manifest does not exist: {manifest_path}"
        )
    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)
    entries = _manifest_entries(raw)
    user_set = set(user_ids)
    specs: list[AnkiSM2APPolicySpec] = []
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
        policy_index = _optional_int(
            entry.get("policy_index"),
            f"policies[{index}].policy_index",
        )
        specs.append(
            _validate_policy_spec(
                path=path,
                user_id=user_id,
                policy_index=policy_index,
                source=f"manifest entry {index}",
            )
        )
    if not specs:
        raise ValueError(f"Policy manifest {manifest_path} did not produce any lanes.")
    _reject_duplicate_policy_specs(specs)
    _require_all_users(specs=specs, user_ids=user_ids)
    return tuple(specs)


def _spec_from_policy_path(policy_path: Path) -> AnkiSM2APPolicySpec:
    AnkiSM2APPolicy.from_json(policy_path)
    metadata = _load_sibling_metadata(policy_path)
    path_user_id = _extract_path_int(policy_path, "user_")
    path_policy_index = _extract_path_int(policy_path, "policy_")
    metadata_user_id = _metadata_user_id(metadata, policy_path)
    metadata_policy_index = _metadata_int(metadata, "portfolio_index", policy_path)
    user_id = metadata_user_id if metadata_user_id is not None else path_user_id
    if user_id is None:
        raise ValueError(
            f"Could not infer user_id for Anki SM2 AP policy {policy_path}. "
            "Use a user_<id> path component or metadata.json."
        )
    if (
        metadata_user_id is not None
        and path_user_id is not None
        and metadata_user_id != path_user_id
    ):
        raise ValueError(
            f"Policy {policy_path} user_id mismatch: path has {path_user_id}, "
            f"metadata has {metadata_user_id}."
        )
    if metadata is not None:
        _validate_metadata(metadata=metadata, path=policy_path)
    return AnkiSM2APPolicySpec(
        user_id=user_id,
        policy_index=metadata_policy_index
        if metadata_policy_index is not None
        else path_policy_index,
        path=policy_path.resolve(),
    )


def _validate_policy_spec(
    *,
    path: Path,
    user_id: int,
    policy_index: int | None,
    source: str,
) -> AnkiSM2APPolicySpec:
    if not path.exists():
        raise FileNotFoundError(f"Missing Anki SM2 AP policy for {source}: {path}")
    AnkiSM2APPolicy.from_json(path)
    metadata = _load_sibling_metadata(path)
    if metadata is not None:
        _validate_metadata(metadata=metadata, path=path)
        metadata_user_id = _metadata_user_id(metadata, path)
        if metadata_user_id is not None and metadata_user_id != user_id:
            raise ValueError(
                f"Policy {path} metadata user_id={metadata_user_id}, "
                f"expected {user_id} from {source}."
            )
        metadata_policy_index = _metadata_int(metadata, "portfolio_index", path)
        if policy_index is None:
            policy_index = metadata_policy_index
    return AnkiSM2APPolicySpec(
        user_id=user_id,
        policy_index=policy_index
        if policy_index is not None
        else _extract_path_int(path, "policy_"),
        path=path.resolve(),
    )


def _validate_metadata(*, metadata: Mapping[str, Any], path: Path) -> None:
    scheduler_name = metadata.get("scheduler_name")
    if scheduler_name is not None and scheduler_name != "anki_sm2_ap":
        raise ValueError(
            f"Metadata for {path} has scheduler_name={scheduler_name!r}; "
            "expected 'anki_sm2_ap'."
        )
    action_space = metadata.get("action_space")
    if (
        action_space is not None
        and action_space != "anki_sm2_ap_params_portfolio_child"
    ):
        raise ValueError(
            f"Metadata for {path} has action_space={action_space!r}; "
            "expected 'anki_sm2_ap_params_portfolio_child'."
        )
    policy_path_raw = metadata.get("policy_path")
    metadata_path = path.parent / "metadata.json"
    if isinstance(policy_path_raw, str) and policy_path_raw.strip():
        metadata_policy_path = Path(policy_path_raw)
        if not metadata_policy_path.is_absolute():
            metadata_policy_path = metadata_path.parent / metadata_policy_path
        if metadata_policy_path.resolve() != path.resolve():
            raise ValueError(
                f"Metadata {metadata_path} points to policy_path="
                f"{metadata_policy_path}, not {path}."
            )


def _load_sibling_metadata(path: Path) -> Mapping[str, Any] | None:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Metadata {metadata_path} must be a JSON object.")
    return raw


def _manifest_entries(raw: Mapping[str, Any]) -> Sequence[Any]:
    entries = raw.get("policies")
    if entries is None:
        entries = raw.get("policy")
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise ValueError("Policy manifest must contain [[policies]] entries.")
    return entries


def _require_all_users(
    *,
    specs: Sequence[AnkiSM2APPolicySpec],
    user_ids: Sequence[int],
) -> None:
    users_with_specs = {spec.user_id for spec in specs}
    missing = [user_id for user_id in user_ids if user_id not in users_with_specs]
    if missing:
        raise FileNotFoundError(
            "Missing Anki SM2 AP policies for users: "
            + ", ".join(str(user_id) for user_id in missing)
        )


def _reject_duplicate_policy_specs(specs: Sequence[AnkiSM2APPolicySpec]) -> None:
    seen: dict[tuple[int, int | str], Path] = {}
    for spec in specs:
        key: tuple[int, int | str] = (
            spec.user_id,
            spec.policy_index if spec.policy_index is not None else str(spec.path),
        )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate Anki SM2 AP policy for "
                f"user_id={spec.user_id}, policy_index={spec.policy_index}: "
                f"{previous} and {spec.path}"
            )
        seen[key] = spec.path


def _metadata_user_id(metadata: Mapping[str, Any] | None, path: Path) -> int | None:
    if metadata is None:
        return None
    training_user_ids = metadata.get("training_user_ids")
    if training_user_ids is not None:
        if (
            isinstance(training_user_ids, Sequence)
            and not isinstance(training_user_ids, str)
            and len(training_user_ids) == 1
        ):
            return _require_int(training_user_ids[0], f"{path}.training_user_ids[0]")
        raise ValueError(f"Metadata {path} training_user_ids must contain one user id.")
    return _metadata_int(metadata, "user_id", path)


def _metadata_int(
    metadata: Mapping[str, Any] | None,
    key: str,
    path: Path,
) -> int | None:
    if metadata is None or key not in metadata or metadata.get(key) is None:
        return None
    return _require_int(metadata.get(key), f"{path}.{key}")


def _extract_path_int(path: Path, prefix: str) -> int | None:
    for part in reversed(path.parts):
        if part.startswith(prefix):
            value = part[len(prefix) :]
            if value.isdigit():
                return int(value)
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
        raise ValueError(f"{field_name} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = base_path / path
    return path


__all__ = ["AnkiSM2APPolicySpec", "resolve_anki_sm2_ap_policy_specs"]
