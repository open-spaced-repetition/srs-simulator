from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from simulator.math.fsrs import Bounds


POLICY_KIND = "fsrs6-cost-conditioned-adr"
FEATURE_VERSION_INTERVAL_MONO = "fsrs6_cost_adr_interval_mono_v1"
FEATURE_VERSION_RETENTION_MONO = "fsrs6_cost_adr_retention_mono_v1"
FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z = (
    "fsrs6_cost_adr_retention_mono_drop_sqrt_z_v1"
)
FEATURE_VERSION_RETENTION_MONO_DROP_XD2 = "fsrs6_cost_adr_retention_mono_drop_xd2_v1"
FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2 = (
    "fsrs6_cost_adr_retention_mono_drop_sqrt_z_xd2_v1"
)
FEATURE_VERSION_RETENTION_MONO_Z2_ONLY = "fsrs6_cost_adr_retention_mono_z2_only_v1"
ACTION_HEAD_INTERVAL = "interval"
ACTION_HEAD_RETENTION = "desired_retention"
DEFAULT_ACTION_HEAD = ACTION_HEAD_RETENTION
DEFAULT_FEATURE_VERSION = FEATURE_VERSION_RETENTION_MONO
STATE_FEATURE_COUNT_COMPACT = 6
STATE_FEATURE_COUNT_HINGE = 8
COEFFICIENT_GROUP_COUNT = 4
COST_BASIS_SQRT_Z = "sqrt_z"
COST_BASIS_Z = "z"
COST_BASIS_Z2 = "z2"
FULL_COST_BASIS = (COST_BASIS_SQRT_Z, COST_BASIS_Z, COST_BASIS_Z2)
COMPACT_STATE_FEATURE_INDICES = (0, 1, 2, 3, 4, 5)
DROP_XD2_STATE_FEATURE_INDICES = (0, 1, 2, 3, 4)
HINGE_STATE_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7)
FEATURE_COUNTS = {
    FEATURE_VERSION_INTERVAL_MONO: {
        STATE_FEATURE_COUNT_COMPACT * COEFFICIENT_GROUP_COUNT,
        STATE_FEATURE_COUNT_HINGE * COEFFICIENT_GROUP_COUNT,
    },
    FEATURE_VERSION_RETENTION_MONO: {
        STATE_FEATURE_COUNT_COMPACT * COEFFICIENT_GROUP_COUNT,
        STATE_FEATURE_COUNT_HINGE * COEFFICIENT_GROUP_COUNT,
    },
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z: {
        STATE_FEATURE_COUNT_COMPACT * 3,
    },
    FEATURE_VERSION_RETENTION_MONO_DROP_XD2: {
        len(DROP_XD2_STATE_FEATURE_INDICES) * COEFFICIENT_GROUP_COUNT,
    },
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2: {
        len(DROP_XD2_STATE_FEATURE_INDICES) * 3,
    },
    FEATURE_VERSION_RETENTION_MONO_Z2_ONLY: {
        STATE_FEATURE_COUNT_COMPACT * 2,
    },
}
FEATURE_VERSION_ACTION_HEADS = {
    FEATURE_VERSION_INTERVAL_MONO: ACTION_HEAD_INTERVAL,
    FEATURE_VERSION_RETENTION_MONO: ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z: ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_DROP_XD2: ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2: ACTION_HEAD_RETENTION,
    FEATURE_VERSION_RETENTION_MONO_Z2_ONLY: ACTION_HEAD_RETENTION,
}


ActionHead = Literal["interval", "desired_retention"]


@dataclass(frozen=True, slots=True)
class FSRS6CostConditionedADRPolicy:
    coefficients: tuple[float, ...]
    action_head: ActionHead = DEFAULT_ACTION_HEAD
    feature_version: str = DEFAULT_FEATURE_VERSION
    cost_weight_min: float = 0.0
    cost_weight_max: float = 1024.0
    retention_min: float = 0.5
    retention_max: float = 0.98
    max_interval_days: float | None = None
    bounds: Bounds = Bounds()
    title: str = "FSRS6 cost-conditioned ADR"
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> FSRS6CostConditionedADRPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Cost-conditioned ADR policy {policy_path} must be JSON.")
        policy_kind = raw.get("policy_kind", raw.get("policy_type"))
        if policy_kind not in {POLICY_KIND, "fsrs6_cost_conditioned_adr"}:
            raise ValueError(
                f"Unsupported cost-conditioned ADR policy_kind {policy_kind!r}."
            )
        bounds_raw = raw.get("bounds", {})
        if bounds_raw is None:
            bounds_raw = {}
        if not isinstance(bounds_raw, dict):
            raise ValueError("bounds must be an object when provided.")
        feature_version = _require_str(
            raw.get("feature_version", DEFAULT_FEATURE_VERSION),
            "feature_version",
        )
        action_head = (
            _action_head(raw["action_head"])
            if "action_head" in raw
            else _action_head_for_feature_version(feature_version)
        )
        return cls(
            coefficients=_float_tuple(raw.get("coefficients"), "coefficients"),
            action_head=action_head,
            feature_version=feature_version,
            cost_weight_min=_float(raw.get("cost_weight_min", 0.0), "cost_weight_min"),
            cost_weight_max=_float(
                raw.get("cost_weight_max", 1024.0), "cost_weight_max"
            ),
            retention_min=_float(raw.get("retention_min", 0.5), "retention_min"),
            retention_max=_float(raw.get("retention_max", 0.98), "retention_max"),
            max_interval_days=_optional_float(
                raw.get("max_interval_days"), "max_interval_days"
            ),
            bounds=Bounds(
                s_min=_float(bounds_raw.get("s_min", Bounds().s_min), "bounds.s_min"),
                s_max=_float(bounds_raw.get("s_max", Bounds().s_max), "bounds.s_max"),
                d_min=_float(bounds_raw.get("d_min", Bounds().d_min), "bounds.d_min"),
                d_max=_float(bounds_raw.get("d_max", Bounds().d_max), "bounds.d_max"),
            ),
            title=_require_str(raw.get("title", "FSRS6 cost-conditioned ADR"), "title"),
            metadata=raw,
        )

    @classmethod
    def baseline_retention(
        cls,
        *,
        desired_retention: float = 0.9,
        state_feature_count: int = STATE_FEATURE_COUNT_HINGE,
        cost_weight_max: float = 1024.0,
        retention_min: float = 0.5,
        retention_max: float = 0.98,
        bounds: Bounds = Bounds(),
    ) -> FSRS6CostConditionedADRPolicy:
        if state_feature_count not in {
            STATE_FEATURE_COUNT_COMPACT,
            STATE_FEATURE_COUNT_HINGE,
        }:
            raise ValueError("state_feature_count must be 6 or 8.")
        ratio = (desired_retention - retention_min) / (retention_max - retention_min)
        ratio = min(1.0 - 1e-9, max(1e-9, ratio))
        coefficients = [0.0 for _ in range(state_feature_count * 4)]
        coefficients[0] = math.log(ratio / (1.0 - ratio))
        for group_idx in range(1, COEFFICIENT_GROUP_COUNT):
            coefficients[group_idx * state_feature_count] = -40.0
        return cls(
            coefficients=tuple(coefficients),
            action_head=ACTION_HEAD_RETENTION,
            feature_version=FEATURE_VERSION_RETENTION_MONO,
            cost_weight_max=cost_weight_max,
            retention_min=retention_min,
            retention_max=retention_max,
            bounds=bounds,
            title="FSRS6 cost-conditioned ADR baseline",
        )

    def __post_init__(self) -> None:
        _validate_feature_version(self.feature_version, len(self.coefficients))
        if self.action_head not in {ACTION_HEAD_INTERVAL, ACTION_HEAD_RETENTION}:
            raise ValueError(f"Unsupported action_head {self.action_head!r}.")
        expected_action_head = action_head_for_feature_version(self.feature_version)
        if self.action_head != expected_action_head:
            raise ValueError(
                f"action_head={self.action_head!r} expects feature_version "
                f"compatible with {self.action_head!r}; "
                f"{self.feature_version!r} uses {expected_action_head!r}."
            )
        if self.cost_weight_min < 0.0 or not math.isfinite(self.cost_weight_min):
            raise ValueError("cost_weight_min must be finite and >= 0.")
        if self.cost_weight_max <= self.cost_weight_min or not math.isfinite(
            self.cost_weight_max
        ):
            raise ValueError("cost_weight_max must be finite and > cost_weight_min.")
        if not (0.0 < self.retention_min < self.retention_max < 1.0):
            raise ValueError("retention_min/max must satisfy 0 < min < max < 1.")
        if self.max_interval_days is not None and self.max_interval_days < 1.0:
            raise ValueError("max_interval_days must be >= 1 when provided.")
        if self.bounds.s_min <= 0 or self.bounds.s_max <= self.bounds.s_min:
            raise ValueError("bounds must satisfy 0 < s_min < s_max.")
        if self.bounds.d_max <= self.bounds.d_min:
            raise ValueError("bounds must satisfy d_min < d_max.")

    @property
    def state_feature_count(self) -> int:
        return len(self.state_feature_indices)

    @property
    def state_feature_indices(self) -> tuple[int, ...]:
        return state_feature_indices_for_feature_version(
            self.feature_version,
            coefficient_count=len(self.coefficients),
        )

    @property
    def cost_basis(self) -> tuple[str, ...]:
        return cost_basis_for_feature_version(
            self.feature_version,
            coefficient_count=len(self.coefficients),
        )

    @property
    def coefficient_group_count(self) -> int:
        return 1 + len(self.cost_basis)

    @property
    def parameter_count(self) -> int:
        return len(self.coefficients)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": POLICY_KIND,
            "feature_version": self.feature_version,
            "title": self.title,
            "action_head": self.action_head,
            "coefficients": list(self.coefficients),
            "cost_weight_min": self.cost_weight_min,
            "cost_weight_max": self.cost_weight_max,
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "max_interval_days": self.max_interval_days,
            "state_feature_indices": list(self.state_feature_indices),
            "cost_basis": list(self.cost_basis),
            "bounds": {
                "s_min": self.bounds.s_min,
                "s_max": self.bounds.s_max,
                "d_min": self.bounds.d_min,
                "d_max": self.bounds.d_max,
            },
        }

    def write_json(self, path: str | Path) -> None:
        policy_path = Path(path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate_action(
        self,
        stability: float,
        difficulty: float,
        *,
        cost_weight: float,
    ) -> float:
        value = _monotone_value(
            self.coefficients,
            stability=stability,
            difficulty=difficulty,
            cost_weight=cost_weight,
            bounds=self.bounds,
            cost_weight_min=self.cost_weight_min,
            cost_weight_max=self.cost_weight_max,
            state_feature_count=self.state_feature_count,
            feature_version=self.feature_version,
            cost_sign=-1.0 if self.action_head == ACTION_HEAD_RETENTION else 1.0,
        )
        if self.action_head == ACTION_HEAD_INTERVAL:
            interval = math.exp(value)
            if self.max_interval_days is not None:
                interval = min(float(self.max_interval_days), interval)
            return max(1.0, interval)
        return self.retention_min + (
            self.retention_max - self.retention_min
        ) * _sigmoid(value)

    def evaluate_interval(
        self,
        stability: float,
        difficulty: float,
        *,
        cost_weight: float,
    ) -> float:
        if self.action_head != ACTION_HEAD_INTERVAL:
            raise ValueError("evaluate_interval requires action_head='interval'.")
        return self.evaluate_action(
            stability,
            difficulty,
            cost_weight=cost_weight,
        )

    def evaluate_retention(
        self,
        stability: float,
        difficulty: float,
        *,
        cost_weight: float,
    ) -> float:
        if self.action_head != ACTION_HEAD_RETENTION:
            raise ValueError(
                "evaluate_retention requires action_head='desired_retention'."
            )
        return self.evaluate_action(
            stability,
            difficulty,
            cost_weight=cost_weight,
        )


def state_features(
    stability: float,
    difficulty: float,
    bounds: Bounds = Bounds(),
    *,
    state_feature_count: int = STATE_FEATURE_COUNT_HINGE,
) -> tuple[float, ...]:
    x_s, x_d = normalized_inputs(stability, difficulty, bounds)
    compact = (1.0, x_s, x_d, x_s * x_d, x_s * x_s, x_d * x_d)
    if state_feature_count == STATE_FEATURE_COUNT_COMPACT:
        return compact
    if state_feature_count == STATE_FEATURE_COUNT_HINGE:
        return (*compact, max(0.0, x_s - 0.5), max(0.0, x_d - 0.5))
    raise ValueError("state_feature_count must be 6 or 8.")


def normalized_inputs(
    stability: float,
    difficulty: float,
    bounds: Bounds = Bounds(),
) -> tuple[float, float]:
    s = min(bounds.s_max, max(bounds.s_min, float(stability)))
    d = min(bounds.d_max, max(bounds.d_min, float(difficulty)))
    log_s_min = math.log(bounds.s_min)
    log_s_span = math.log(bounds.s_max) - log_s_min
    s_norm = (math.log(s) - log_s_min) / log_s_span
    d_norm = (d - bounds.d_min) / (bounds.d_max - bounds.d_min)
    return min(1.0, max(0.0, s_norm)), min(1.0, max(0.0, d_norm))


def normalized_cost_weight(
    cost_weight: float,
    *,
    cost_weight_min: float = 0.0,
    cost_weight_max: float = 1024.0,
) -> float:
    weight = min(cost_weight_max, max(cost_weight_min, float(cost_weight)))
    lo = math.log1p(cost_weight_min)
    hi = math.log1p(cost_weight_max)
    if hi <= lo:
        raise ValueError("cost_weight_max must be > cost_weight_min.")
    return min(1.0, max(0.0, (math.log1p(weight) - lo) / (hi - lo)))


def _monotone_value(
    coefficients: Sequence[float],
    *,
    stability: float,
    difficulty: float,
    cost_weight: float,
    bounds: Bounds,
    cost_weight_min: float,
    cost_weight_max: float,
    state_feature_count: int,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    cost_sign: float = 1.0,
) -> float:
    feature_indices = state_feature_indices_for_feature_version(
        feature_version,
        coefficient_count=len(coefficients),
    )
    if len(feature_indices) != state_feature_count:
        raise ValueError("state_feature_count does not match feature_version.")
    full_feature_count = (
        STATE_FEATURE_COUNT_HINGE
        if max(feature_indices) >= STATE_FEATURE_COUNT_COMPACT
        else STATE_FEATURE_COUNT_COMPACT
    )
    all_features = state_features(
        stability,
        difficulty,
        bounds,
        state_feature_count=full_feature_count,
    )
    phi = tuple(all_features[index] for index in feature_indices)
    z = normalized_cost_weight(
        cost_weight,
        cost_weight_min=cost_weight_min,
        cost_weight_max=cost_weight_max,
    )
    cost_basis = cost_basis_for_feature_version(
        feature_version,
        coefficient_count=len(coefficients),
    )
    group_count = 1 + len(cost_basis)
    if len(coefficients) != state_feature_count * group_count:
        raise ValueError("coefficient count does not match policy structure.")
    groups = [
        coefficients[idx : idx + state_feature_count]
        for idx in range(0, len(coefficients), state_feature_count)
    ]
    base = _dot(groups[0], phi)
    basis_values = {
        COST_BASIS_SQRT_Z: math.sqrt(z),
        COST_BASIS_Z: z,
        COST_BASIS_Z2: z * z,
    }
    cost_effect = 0.0
    for group, basis in zip(groups[1:], cost_basis, strict=True):
        cost_effect += _softplus(_dot(group, phi)) * basis_values[basis]
    return base + cost_sign * cost_effect


def action_head_for_feature_version(feature_version: str) -> ActionHead:
    try:
        return cast(ActionHead, FEATURE_VERSION_ACTION_HEADS[feature_version])
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_VERSION_ACTION_HEADS))
        raise ValueError(
            f"Unsupported cost ADR feature_version {feature_version!r}; "
            f"expected one of: {supported}."
        ) from exc


def state_feature_indices_for_feature_version(
    feature_version: str,
    *,
    coefficient_count: int,
) -> tuple[int, ...]:
    _validate_feature_version(feature_version, coefficient_count)
    if feature_version in {
        FEATURE_VERSION_INTERVAL_MONO,
        FEATURE_VERSION_RETENTION_MONO,
    }:
        group_count = COEFFICIENT_GROUP_COUNT
        state_feature_count = coefficient_count // group_count
        if state_feature_count == STATE_FEATURE_COUNT_COMPACT:
            return COMPACT_STATE_FEATURE_INDICES
        if state_feature_count == STATE_FEATURE_COUNT_HINGE:
            return HINGE_STATE_FEATURE_INDICES
    if feature_version == FEATURE_VERSION_RETENTION_MONO_DROP_XD2:
        return DROP_XD2_STATE_FEATURE_INDICES
    if feature_version == FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2:
        return DROP_XD2_STATE_FEATURE_INDICES
    return COMPACT_STATE_FEATURE_INDICES


def cost_basis_for_feature_version(
    feature_version: str,
    *,
    coefficient_count: int,
) -> tuple[str, ...]:
    _validate_feature_version(feature_version, coefficient_count)
    if feature_version == FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z:
        return (COST_BASIS_Z, COST_BASIS_Z2)
    if feature_version == FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2:
        return (COST_BASIS_Z, COST_BASIS_Z2)
    if feature_version == FEATURE_VERSION_RETENTION_MONO_Z2_ONLY:
        return (COST_BASIS_Z2,)
    return FULL_COST_BASIS


def parameter_count_for_feature_version(
    feature_version: str,
    *,
    default_state_feature_count: int = STATE_FEATURE_COUNT_COMPACT,
) -> int:
    if feature_version in {
        FEATURE_VERSION_INTERVAL_MONO,
        FEATURE_VERSION_RETENTION_MONO,
    }:
        if default_state_feature_count not in {
            STATE_FEATURE_COUNT_COMPACT,
            STATE_FEATURE_COUNT_HINGE,
        }:
            raise ValueError("default_state_feature_count must be 6 or 8.")
        return default_state_feature_count * COEFFICIENT_GROUP_COUNT
    expected = FEATURE_COUNTS.get(feature_version)
    if expected is None:
        supported = ", ".join(sorted(FEATURE_COUNTS))
        raise ValueError(
            f"Unsupported cost ADR feature_version {feature_version!r}; "
            f"expected one of: {supported}."
        )
    if len(expected) != 1:
        raise ValueError(
            f"feature_version {feature_version!r} needs an explicit coefficient count."
        )
    return next(iter(expected))


def project_coefficients_to_feature_version(
    coefficients: Sequence[float],
    *,
    source_feature_version: str,
    target_feature_version: str,
) -> tuple[float, ...]:
    source_count = len(coefficients)
    source_features = state_feature_indices_for_feature_version(
        source_feature_version,
        coefficient_count=source_count,
    )
    source_basis = cost_basis_for_feature_version(
        source_feature_version,
        coefficient_count=source_count,
    )
    target_count = parameter_count_for_feature_version(target_feature_version)
    target_features = state_feature_indices_for_feature_version(
        target_feature_version,
        coefficient_count=target_count,
    )
    target_basis = cost_basis_for_feature_version(
        target_feature_version,
        coefficient_count=target_count,
    )
    source_group_names = ("base", *source_basis)
    target_group_names = ("base", *target_basis)
    source_group_count = len(source_group_names)
    source_feature_count = len(source_features)
    if source_count != source_group_count * source_feature_count:
        raise ValueError("source coefficients do not match source feature_version.")
    result: list[float] = []
    for group_name in target_group_names:
        if group_name not in source_group_names:
            raise ValueError(
                f"Cannot project missing coefficient group {group_name!r}."
            )
        source_group_index = source_group_names.index(group_name)
        for feature_index in target_features:
            if feature_index not in source_features:
                raise ValueError(
                    f"Cannot project missing state feature index {feature_index}."
                )
            source_feature_index = source_features.index(feature_index)
            result.append(
                float(
                    coefficients[
                        source_group_index * source_feature_count + source_feature_index
                    ]
                )
            )
    return tuple(result)


def _validate_feature_version(feature_version: str, coefficient_count: int) -> None:
    if feature_version not in FEATURE_COUNTS:
        supported = ", ".join(sorted(FEATURE_COUNTS))
        raise ValueError(
            f"Unsupported feature_version {feature_version!r}; expected {supported}."
        )
    if coefficient_count not in FEATURE_COUNTS[feature_version]:
        expected = ", ".join(
            str(value) for value in sorted(FEATURE_COUNTS[feature_version])
        )
        raise ValueError(
            f"{feature_version!r} expects one of {expected} coefficients; "
            f"got {coefficient_count}."
        )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _action_head(value: Any) -> ActionHead:
    if value == ACTION_HEAD_INTERVAL:
        return ACTION_HEAD_INTERVAL
    if value in {ACTION_HEAD_RETENTION, "retention"}:
        return ACTION_HEAD_RETENTION
    raise ValueError(
        f"action_head must be {ACTION_HEAD_INTERVAL!r} or {ACTION_HEAD_RETENTION!r}."
    )


def _action_head_for_feature_version(feature_version: str) -> ActionHead:
    return action_head_for_feature_version(feature_version)


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _float(value, field_name)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    result = tuple(
        _float(item, f"{field_name}[{idx}]") for idx, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    return result


__all__ = [
    "ACTION_HEAD_INTERVAL",
    "ACTION_HEAD_RETENTION",
    "COST_BASIS_SQRT_Z",
    "COST_BASIS_Z",
    "COST_BASIS_Z2",
    "FEATURE_VERSION_INTERVAL_MONO",
    "FEATURE_VERSION_RETENTION_MONO",
    "FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z",
    "FEATURE_VERSION_RETENTION_MONO_DROP_XD2",
    "FEATURE_VERSION_RETENTION_MONO_DROP_SQRT_Z_XD2",
    "FEATURE_VERSION_RETENTION_MONO_Z2_ONLY",
    "FSRS6CostConditionedADRPolicy",
    "POLICY_KIND",
    "STATE_FEATURE_COUNT_COMPACT",
    "STATE_FEATURE_COUNT_HINGE",
    "action_head_for_feature_version",
    "cost_basis_for_feature_version",
    "normalized_cost_weight",
    "normalized_inputs",
    "parameter_count_for_feature_version",
    "project_coefficients_to_feature_version",
    "state_features",
    "state_feature_indices_for_feature_version",
]
