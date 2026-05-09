from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from simulator.fsrs6_adr_policy import FSRS6ADRPolicy


@dataclass(frozen=True, slots=True)
class FSRS6ADRPolicySpec:
    user_id: int
    baseline_desired_retention: float | None
    lambda_value: float | None
    policy_index: int | None
    path: Path


def format_float_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def parse_float_token(value: str) -> float:
    token = value.strip()
    if not token:
        raise ValueError("empty float token")
    token = token.replace("neg_", "-").replace("p", ".")
    return float(token)


def resolve_fsrs6_adr_policy_specs(
    *,
    user_ids: Sequence[int],
    dr_values: Sequence[float],
    policy_root: Path | None = None,
    train_run_root: Path | None = None,
    policy_manifest: Path | None = None,
    lambda_values: Sequence[float] | None = None,
) -> tuple[FSRS6ADRPolicySpec, ...]:
    sources = [
        policy_root is not None,
        train_run_root is not None,
        policy_manifest is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "Configure exactly one FSRS6 ADR policy source: "
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
) -> tuple[FSRS6ADRPolicySpec, ...]:
    root = policy_root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"FSRS6 ADR policy root does not exist: {root}")
    user_set = set(user_ids)
    lambda_filter = _normalized_lambda_filter(lambda_values)
    specs: list[FSRS6ADRPolicySpec] = []
    for policy_path in sorted(root.rglob("policy.json")):
        path_user_id = _extract_path_int(policy_path, "user_")
        if path_user_id is not None and path_user_id not in user_set:
            continue
        spec = _spec_from_policy_path(policy_path, dr_values=dr_values)
        if spec.user_id not in user_set:
            continue
        if spec.baseline_desired_retention is not None and not _matches_grid(
            spec.baseline_desired_retention, dr_values
        ):
            continue
        if lambda_filter is not None and not _matches_lambda(
            spec.lambda_value, lambda_filter
        ):
            continue
        specs.append(spec)

    if not specs:
        raise FileNotFoundError(
            "No FSRS6 ADR policies under "
            f"{root} matched users={list(user_ids)} and DR grid={list(dr_values)}."
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
) -> tuple[FSRS6ADRPolicySpec, ...]:
    manifest_path = policy_manifest.expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"FSRS6 ADR policy manifest does not exist: {manifest_path}"
        )
    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)
    entries = _manifest_entries(raw)
    user_set = set(user_ids)
    lambda_filter = _normalized_lambda_filter(lambda_values)
    specs: list[FSRS6ADRPolicySpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"policies[{index}] must be a TOML table.")
        user_id = _require_int(entry.get("user_id"), f"policies[{index}].user_id")
        baseline_dr = _optional_float(
            entry.get("baseline_desired_retention"),
            f"policies[{index}].baseline_desired_retention",
        )
        lambda_value = _optional_float(
            entry.get("lambda_value"), f"policies[{index}].lambda_value"
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
        if baseline_dr is not None and not _matches_grid(baseline_dr, dr_values):
            raise ValueError(
                f"Policy manifest entry {index} uses "
                f"baseline_desired_retention={baseline_dr}, which is outside "
                f"configured retention grid {list(dr_values)}."
            )
        if lambda_filter is not None and not _matches_lambda(
            lambda_value, lambda_filter
        ):
            continue
        specs.append(
            _validate_policy_spec(
                path=path,
                user_id=user_id,
                baseline_desired_retention=baseline_dr,
                lambda_value=lambda_value,
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
    *,
    dr_values: Sequence[float],
) -> FSRS6ADRPolicySpec:
    policy = FSRS6ADRPolicy.from_json(policy_path)
    metadata = _load_sibling_metadata(policy_path)
    path_user_id = _extract_path_int(policy_path, "user_")
    path_lambda = _extract_path_float(policy_path, "lambda_")
    path_policy_index = _extract_path_int(policy_path, "policy_")
    metadata_user_id = _metadata_user_id(metadata, policy_path)
    metadata_baseline_dr = _metadata_float(
        metadata,
        "baseline_desired_retention",
        policy_path,
    )
    metadata_lambda = _metadata_float(metadata, "lambda_value", policy_path)
    metadata_policy_index = _metadata_int(metadata, "portfolio_index", policy_path)

    user_id = metadata_user_id if metadata_user_id is not None else path_user_id
    if user_id is None:
        raise ValueError(
            f"Could not infer user_id for FSRS6 ADR policy {policy_path}. "
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

    metadata_has_null_baseline = (
        metadata is not None
        and "baseline_desired_retention" in metadata
        and metadata.get("baseline_desired_retention") is None
    )
    if metadata_has_null_baseline:
        baseline_dr = None
    else:
        baseline_dr = (
            metadata_baseline_dr
            if metadata_baseline_dr is not None
            else policy.baseline_desired_retention
        )
    if baseline_dr is None:
        if policy.baseline_desired_retention is not None:
            raise ValueError(
                f"Policy {policy_path} has baseline_desired_retention="
                f"{policy.baseline_desired_retention}, metadata has null."
            )
        matched_dr = None
    elif policy.baseline_desired_retention is None or not math.isclose(
        baseline_dr,
        policy.baseline_desired_retention,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Policy {policy_path} has baseline_desired_retention="
            f"{policy.baseline_desired_retention}, metadata has {baseline_dr}."
        )
    else:
        matched_dr = _matching_grid_value(baseline_dr, dr_values)
    lambda_value = metadata_lambda if metadata_lambda is not None else path_lambda
    if (
        metadata_lambda is not None
        and path_lambda is not None
        and not math.isclose(metadata_lambda, path_lambda, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            f"Policy {policy_path} lambda mismatch: path has {path_lambda}, "
            f"metadata has {metadata_lambda}."
        )
    return FSRS6ADRPolicySpec(
        user_id=user_id,
        baseline_desired_retention=matched_dr,
        lambda_value=lambda_value,
        policy_index=metadata_policy_index
        if metadata_policy_index is not None
        else path_policy_index,
        path=policy_path.resolve(),
    )


def _validate_policy_spec(
    *,
    path: Path,
    user_id: int,
    baseline_desired_retention: float | None,
    lambda_value: float | None,
    policy_index: int | None,
    source: str,
) -> FSRS6ADRPolicySpec:
    if not path.exists():
        raise FileNotFoundError(f"Missing FSRS6 ADR policy for {source}: {path}")
    policy = FSRS6ADRPolicy.from_json(path)
    if baseline_desired_retention is None:
        if policy.baseline_desired_retention is not None:
            raise ValueError(
                f"Policy {path} has baseline_desired_retention="
                f"{policy.baseline_desired_retention}, expected null from {source}."
            )
    elif policy.baseline_desired_retention is None or not math.isclose(
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
    metadata_baseline_dr = _metadata_float(metadata, "baseline_desired_retention", path)
    if baseline_desired_retention is None:
        if metadata_baseline_dr is not None:
            raise ValueError(
                f"Policy {path} metadata baseline_desired_retention="
                f"{metadata_baseline_dr}, expected null."
            )
    elif metadata_baseline_dr is not None and not math.isclose(
        metadata_baseline_dr,
        baseline_desired_retention,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Policy {path} metadata baseline_desired_retention="
            f"{metadata_baseline_dr}, expected {baseline_desired_retention}."
        )
    metadata_lambda = _metadata_float(metadata, "lambda_value", path)
    effective_lambda = lambda_value if lambda_value is not None else metadata_lambda
    if (
        lambda_value is not None
        and metadata_lambda is not None
        and not math.isclose(lambda_value, metadata_lambda, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            f"Policy {path} metadata lambda_value={metadata_lambda}, "
            f"expected {lambda_value} from {source}."
        )
    return FSRS6ADRPolicySpec(
        user_id=user_id,
        baseline_desired_retention=baseline_desired_retention,
        lambda_value=effective_lambda,
        policy_index=policy_index
        if policy_index is not None
        else _metadata_int(metadata, "portfolio_index", path)
        if metadata is not None
        else _extract_path_int(path, "policy_"),
        path=path.resolve(),
    )


def _require_complete_policy_root(
    *,
    specs: Sequence[FSRS6ADRPolicySpec],
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
    if any(spec.baseline_desired_retention is None for spec in specs):
        return
    observed = {
        _policy_key(
            spec.user_id,
            spec.baseline_desired_retention,
            spec.lambda_value,
            spec.policy_index,
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
        raise FileNotFoundError(f"Missing FSRS6 ADR policies:\n{preview}{suffix}")


def _reject_duplicate_policy_specs(specs: Sequence[FSRS6ADRPolicySpec]) -> None:
    by_key: dict[tuple[int, int | None, int | None, int | None], Path] = {}
    for spec in specs:
        key = _policy_key(
            spec.user_id,
            spec.baseline_desired_retention,
            spec.lambda_value,
            spec.policy_index,
        )
        previous = by_key.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate FSRS6 ADR policy for "
                f"user={spec.user_id}, baseline_desired_retention="
                f"{spec.baseline_desired_retention}, lambda={spec.lambda_value}: "
                f"{previous} and {spec.path}"
            )
        by_key[key] = spec.path


def _policy_key(
    user_id: int,
    baseline_desired_retention: float | None,
    lambda_value: float | None,
    policy_index: int | None = None,
) -> tuple[int, int | None, int | None, int | None]:
    lambda_key = (
        None if lambda_value is None else round(float(lambda_value) * 1_000_000)
    )
    baseline_key = (
        None
        if baseline_desired_retention is None
        else round(float(baseline_desired_retention) * 1_000_000)
    )
    return (
        int(user_id),
        baseline_key,
        lambda_key,
        policy_index if baseline_key is None else None,
    )


def _format_missing_policy(
    user_id: int, baseline_dr: float, lambda_value: float | None
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


def _matches_lambda(
    value: float | None,
    lambda_values: Sequence[float],
) -> bool:
    if value is None:
        return False
    return any(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
        for expected in lambda_values
    )


def _load_sibling_metadata(path: Path) -> Mapping[str, Any] | None:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Artifact metadata must be a JSON object: {metadata_path}")
    scheduler_name = raw.get("scheduler_name")
    if scheduler_name is not None and scheduler_name != "fsrs6_adr":
        raise ValueError(
            f"Artifact metadata {metadata_path} scheduler_name={scheduler_name!r}; "
            "expected 'fsrs6_adr'."
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


def _metadata_float(
    metadata: Mapping[str, Any] | None,
    key: str,
    path: Path,
) -> float | None:
    if metadata is None or key not in metadata:
        return None
    return _optional_float(metadata[key], f"metadata.{key} for {path}")


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


def _extract_path_float(path: Path, prefix: str) -> float | None:
    for part in reversed(path.parts):
        if part.startswith(prefix):
            token = part[len(prefix) :]
            try:
                return parse_float_token(token)
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


def _require_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, field_name)


def _require_path(value: Any, field_name: str, *, base_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


__all__ = [
    "FSRS6ADRPolicySpec",
    "format_float_token",
    "parse_float_token",
    "resolve_fsrs6_adr_policy_specs",
]
