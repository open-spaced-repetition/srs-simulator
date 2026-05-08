from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.fsrs6_adp_policy import FSRS6ADPPolicy


@dataclass(frozen=True, slots=True)
class FSRS6ADPPolicySpec:
    user_id: int
    baseline_desired_retention: float
    lambda_value: float | None
    path: Path


def resolve_fsrs6_adp_policy_specs(
    *,
    user_ids: Sequence[int],
    dr_values: Sequence[float],
    policy_root: Path | None = None,
    train_run_root: Path | None = None,
    policy_manifest: Path | None = None,
    lambda_values: Sequence[float] | None = None,
) -> tuple[FSRS6ADPPolicySpec, ...]:
    sources = [
        policy_root is not None,
        train_run_root is not None,
        policy_manifest is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Configure exactly one FSRS6 ADP policy source: "
            "policy_root, train_run_root, or policy_manifest."
        )
    if train_run_root is not None:
        policy_root = train_run_root / "train-overfit" / "train_outputs"
    if policy_manifest is not None:
        return _load_policy_manifest(
            policy_manifest=policy_manifest,
            user_ids=user_ids,
            dr_values=dr_values,
            lambda_values=lambda_values,
        )
    if policy_root is None:
        raise AssertionError("policy_root must be set after source validation.")
    return _discover_policy_root(
        policy_root=policy_root,
        user_ids=user_ids,
        dr_values=dr_values,
        lambda_values=lambda_values,
    )


def _discover_policy_root(
    *,
    policy_root: Path,
    user_ids: Sequence[int],
    dr_values: Sequence[float],
    lambda_values: Sequence[float] | None,
) -> tuple[FSRS6ADPPolicySpec, ...]:
    root = policy_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"FSRS6 ADP policy root does not exist: {root}")
    user_set = set(user_ids)
    lambda_filter = _normalized_lambda_filter(lambda_values)
    specs: list[FSRS6ADPPolicySpec] = []
    for policy_path in sorted(root.rglob("policy.json")):
        path_user_id = _extract_path_int(policy_path, "user_")
        if path_user_id is not None and path_user_id not in user_set:
            continue
        spec = _spec_from_policy_path(policy_path, dr_values=dr_values)
        if spec.user_id not in user_set:
            continue
        if not _matches_grid(spec.baseline_desired_retention, dr_values):
            continue
        if lambda_filter is not None and not _matches_lambda(
            spec.lambda_value,
            lambda_filter,
        ):
            continue
        specs.append(spec)
    if not specs:
        raise FileNotFoundError(
            f"No FSRS6 ADP policies under {root} matched users={list(user_ids)} "
            f"and DR grid={list(dr_values)}."
        )
    _reject_duplicate_policy_specs(specs)
    _require_complete_policy_root(
        specs=specs,
        user_ids=user_ids,
        dr_values=dr_values,
        lambda_values=lambda_values,
    )
    return tuple(specs)


def _load_policy_manifest(
    *,
    policy_manifest: Path,
    user_ids: Sequence[int],
    dr_values: Sequence[float],
    lambda_values: Sequence[float] | None,
) -> tuple[FSRS6ADPPolicySpec, ...]:
    manifest_path = policy_manifest.expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"FSRS6 ADP policy manifest does not exist: {manifest_path}"
        )
    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)
    entries = _manifest_entries(raw)
    user_set = set(user_ids)
    lambda_filter = _normalized_lambda_filter(lambda_values)
    specs: list[FSRS6ADPPolicySpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"policies[{index}] must be a TOML table.")
        user_id = _require_int(entry.get("user_id"), f"policies[{index}].user_id")
        baseline_dr = _require_float(
            entry.get("baseline_desired_retention"),
            f"policies[{index}].baseline_desired_retention",
        )
        lambda_value = _optional_float(
            entry.get("lambda_value"),
            f"policies[{index}].lambda_value",
        )
        path = _require_path(
            entry.get("path"),
            f"policies[{index}].path",
            base_path=manifest_path.parent,
        )
        if user_id not in user_set:
            raise ValueError(
                f"Policy manifest entry {index} uses user_id={user_id}, "
                f"which is outside configured users {list(user_ids)}."
            )
        if not _matches_grid(baseline_dr, dr_values):
            raise ValueError(
                f"Policy manifest entry {index} uses "
                f"baseline_desired_retention={baseline_dr}, which is outside "
                f"configured retention grid {list(dr_values)}."
            )
        if lambda_filter is not None and not _matches_lambda(
            lambda_value,
            lambda_filter,
        ):
            continue
        specs.append(
            _validate_policy_spec(
                path=path,
                user_id=user_id,
                baseline_desired_retention=baseline_dr,
                lambda_value=lambda_value,
                source=f"manifest entry {index}",
            )
        )
    if not specs:
        raise ValueError(f"Policy manifest {manifest_path} did not produce any lanes.")
    _reject_duplicate_policy_specs(specs)
    return tuple(specs)


def _spec_from_policy_path(
    policy_path: Path,
    *,
    dr_values: Sequence[float],
) -> FSRS6ADPPolicySpec:
    policy = FSRS6ADPPolicy.from_json(policy_path)
    metadata = _load_sibling_metadata(policy_path)
    path_user_id = _extract_path_int(policy_path, "user_")
    path_lambda = _extract_path_float(policy_path, "lambda_")
    metadata_user_id = _metadata_user_id(metadata, policy_path)
    metadata_baseline_dr = _metadata_float(
        metadata,
        "baseline_desired_retention",
        policy_path,
    )
    metadata_lambda = _metadata_float(metadata, "lambda_value", policy_path)
    user_id = metadata_user_id if metadata_user_id is not None else path_user_id
    if user_id is None:
        raise ValueError(
            f"Could not infer user_id for FSRS6 ADP policy {policy_path}. "
            "Use a user_<id> path component or metadata.json."
        )
    baseline_dr = (
        metadata_baseline_dr
        if metadata_baseline_dr is not None
        else policy.baseline_desired_retention
    )
    if not math.isclose(
        baseline_dr,
        policy.baseline_desired_retention,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Policy {policy_path} has baseline_desired_retention="
            f"{policy.baseline_desired_retention}, metadata has {baseline_dr}."
        )
    lambda_value = metadata_lambda if metadata_lambda is not None else path_lambda
    return FSRS6ADPPolicySpec(
        user_id=user_id,
        baseline_desired_retention=_matching_grid_value(baseline_dr, dr_values),
        lambda_value=lambda_value,
        path=policy_path.resolve(),
    )


def _validate_policy_spec(
    *,
    path: Path,
    user_id: int,
    baseline_desired_retention: float,
    lambda_value: float | None,
    source: str,
) -> FSRS6ADPPolicySpec:
    if not path.exists():
        raise FileNotFoundError(f"Missing FSRS6 ADP policy for {source}: {path}")
    policy = FSRS6ADPPolicy.from_json(path)
    if not math.isclose(
        policy.baseline_desired_retention,
        baseline_desired_retention,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Policy {path} has baseline_desired_retention="
            f"{policy.baseline_desired_retention}, expected "
            f"{baseline_desired_retention} from {source}."
        )
    metadata = _load_sibling_metadata(path)
    metadata_user_id = _metadata_user_id(metadata, path)
    if metadata_user_id is not None and metadata_user_id != user_id:
        raise ValueError(
            f"Policy {path} metadata user_id={metadata_user_id}, "
            f"expected {user_id} from {source}."
        )
    metadata_lambda = _metadata_float(metadata, "lambda_value", path)
    if (
        lambda_value is not None
        and metadata_lambda is not None
        and not math.isclose(lambda_value, metadata_lambda, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            f"Policy {path} metadata lambda_value={metadata_lambda}, "
            f"expected {lambda_value} from {source}."
        )
    return FSRS6ADPPolicySpec(
        user_id=user_id,
        baseline_desired_retention=baseline_desired_retention,
        lambda_value=lambda_value if lambda_value is not None else metadata_lambda,
        path=path.resolve(),
    )


def _manifest_entries(raw: Mapping[str, Any]) -> Sequence[Any]:
    entries = raw.get("policies")
    if isinstance(entries, Mapping):
        entries = entries.get("entries")
    if entries is None:
        entries = raw.get("policy")
    if isinstance(entries, str) or not isinstance(entries, Sequence):
        raise ValueError("Policy manifest must contain [[policies]] entries.")
    return entries


def _require_complete_policy_root(
    *,
    specs: Sequence[FSRS6ADPPolicySpec],
    user_ids: Sequence[int],
    dr_values: Sequence[float],
    lambda_values: Sequence[float] | None,
) -> None:
    if lambda_values is not None:
        expected_lambdas = tuple(float(value) for value in lambda_values)
    else:
        expected_lambdas = tuple(
            sorted(
                {spec.lambda_value for spec in specs if spec.lambda_value is not None}
            )
        )
        if not expected_lambdas and any(spec.lambda_value is None for spec in specs):
            expected_lambdas = (None,)
    observed = {
        _policy_key(
            spec.user_id,
            spec.baseline_desired_retention,
            spec.lambda_value,
        )
        for spec in specs
    }
    missing: list[tuple[int, float, float | None]] = []
    for user_id in user_ids:
        for baseline_dr in dr_values:
            for lambda_value in expected_lambdas:
                key = _policy_key(user_id, baseline_dr, lambda_value)
                if key not in observed:
                    missing.append((int(user_id), float(baseline_dr), lambda_value))
    if missing:
        preview = "\n".join(
            _format_missing_policy(user_id, baseline_dr, lambda_value)
            for user_id, baseline_dr, lambda_value in missing[:10]
        )
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing FSRS6 ADP policies:\n{preview}{suffix}")


def _reject_duplicate_policy_specs(specs: Sequence[FSRS6ADPPolicySpec]) -> None:
    by_key: dict[tuple[int, int, int | None], Path] = {}
    for spec in specs:
        key = _policy_key(
            spec.user_id,
            spec.baseline_desired_retention,
            spec.lambda_value,
        )
        previous = by_key.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate FSRS6 ADP policy for "
                f"user={spec.user_id}, baseline_desired_retention="
                f"{spec.baseline_desired_retention}, lambda={spec.lambda_value}: "
                f"{previous} and {spec.path}"
            )
        by_key[key] = spec.path


def _policy_key(
    user_id: int,
    baseline_desired_retention: float,
    lambda_value: float | None,
) -> tuple[int, int, int | None]:
    lambda_key = (
        None if lambda_value is None else round(float(lambda_value) * 1_000_000)
    )
    return (
        int(user_id),
        round(float(baseline_desired_retention) * 1_000_000),
        lambda_key,
    )


def _format_missing_policy(
    user_id: int,
    baseline_dr: float,
    lambda_value: float | None,
) -> str:
    label = f"user={user_id}, baseline_desired_retention={baseline_dr:.2f}"
    if lambda_value is not None:
        label += f", lambda={lambda_value:g}"
    return label


def _matches_grid(value: float, dr_values: Sequence[float]) -> bool:
    return any(math.isclose(value, dr, rel_tol=0.0, abs_tol=1e-9) for dr in dr_values)


def _matching_grid_value(value: float, dr_values: Sequence[float]) -> float:
    for dr in dr_values:
        if math.isclose(value, dr, rel_tol=0.0, abs_tol=1e-9):
            return float(dr)
    return float(value)


def _normalized_lambda_filter(
    lambda_values: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if lambda_values is None:
        return None
    return tuple(float(value) for value in lambda_values)


def _matches_lambda(value: float | None, lambda_values: Sequence[float]) -> bool:
    if value is None:
        return False
    return any(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
        for expected in lambda_values
    )


def _load_sibling_metadata(policy_path: Path) -> dict[str, Any]:
    metadata_path = policy_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _metadata_user_id(metadata: Mapping[str, Any], policy_path: Path) -> int | None:
    raw = metadata.get("training_user_ids")
    if raw is None:
        return None
    if isinstance(raw, str) or not isinstance(raw, Sequence) or len(raw) != 1:
        raise ValueError(
            f"metadata.training_user_ids in {policy_path} must contain one user id."
        )
    return _require_int(raw[0], "metadata.training_user_ids[0]")


def _metadata_float(
    metadata: Mapping[str, Any],
    field_name: str,
    policy_path: Path,
) -> float | None:
    if field_name not in metadata or metadata[field_name] is None:
        return None
    try:
        return _require_float(metadata[field_name], f"metadata.{field_name}")
    except ValueError as exc:
        raise ValueError(f"Invalid metadata for {policy_path}: {exc}") from exc


def _extract_path_int(path: Path, prefix: str) -> int | None:
    for part in reversed(path.parts):
        if part.startswith(prefix):
            suffix = part[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
    return None


def _extract_path_float(path: Path, prefix: str) -> float | None:
    for part in reversed(path.parts):
        if part.startswith(prefix):
            suffix = part[len(prefix) :]
            try:
                return float(suffix.replace("neg_", "-").replace("p", "."))
            except ValueError:
                return None
    return None


def _require_path(value: Any, field_name: str, *, base_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path.")
    path = Path(value)
    if not path.is_absolute():
        path = base_path / path
    return path


def _require_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, field_name)


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


__all__ = [
    "FSRS6ADPPolicySpec",
    "resolve_fsrs6_adp_policy_specs",
]
