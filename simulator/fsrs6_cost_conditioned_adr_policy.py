from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from simulator.math.fsrs import Bounds


POLICY_KIND = "fsrs6-cost-conditioned-adr"
FEATURE_VERSION_INTERVAL_MONO = "fsrs6_cost_adr_interval_mono_v1"
FEATURE_VERSION_RETENTION_MONO = "fsrs6_cost_adr_retention_mono_v1"
ACTION_HEAD_INTERVAL = "interval"
ACTION_HEAD_RETENTION = "desired_retention"
STATE_FEATURE_COUNT_COMPACT = 6
STATE_FEATURE_COUNT_HINGE = 8
COEFFICIENT_GROUP_COUNT = 4
FEATURE_COUNTS = {
    FEATURE_VERSION_INTERVAL_MONO: {
        STATE_FEATURE_COUNT_COMPACT * COEFFICIENT_GROUP_COUNT,
        STATE_FEATURE_COUNT_HINGE * COEFFICIENT_GROUP_COUNT,
    },
    FEATURE_VERSION_RETENTION_MONO: {
        STATE_FEATURE_COUNT_COMPACT * COEFFICIENT_GROUP_COUNT,
        STATE_FEATURE_COUNT_HINGE * COEFFICIENT_GROUP_COUNT,
    },
}


ActionHead = Literal["interval", "desired_retention"]


@dataclass(frozen=True, slots=True)
class FSRS6CostConditionedADRPolicy:
    coefficients: tuple[float, ...]
    action_head: ActionHead = ACTION_HEAD_INTERVAL
    feature_version: str = FEATURE_VERSION_INTERVAL_MONO
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
        return cls(
            coefficients=_float_tuple(raw.get("coefficients"), "coefficients"),
            action_head=_action_head(raw.get("action_head", ACTION_HEAD_INTERVAL)),
            feature_version=_require_str(
                raw.get("feature_version", FEATURE_VERSION_INTERVAL_MONO),
                "feature_version",
            ),
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
            title="FSRS6 cost-conditioned ADR retention baseline",
        )

    def __post_init__(self) -> None:
        _validate_feature_version(self.feature_version, len(self.coefficients))
        if self.action_head == ACTION_HEAD_INTERVAL:
            expected = FEATURE_VERSION_INTERVAL_MONO
        elif self.action_head == ACTION_HEAD_RETENTION:
            expected = FEATURE_VERSION_RETENTION_MONO
        else:
            raise ValueError(f"Unsupported action_head {self.action_head!r}.")
        if self.feature_version != expected:
            raise ValueError(
                f"action_head={self.action_head!r} expects feature_version "
                f"{expected!r}."
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
        return len(self.coefficients) // COEFFICIENT_GROUP_COUNT

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
    cost_sign: float = 1.0,
) -> float:
    phi = state_features(
        stability,
        difficulty,
        bounds,
        state_feature_count=state_feature_count,
    )
    z = normalized_cost_weight(
        cost_weight,
        cost_weight_min=cost_weight_min,
        cost_weight_max=cost_weight_max,
    )
    sqrt_z = math.sqrt(z)
    groups = [
        coefficients[idx : idx + state_feature_count]
        for idx in range(0, len(coefficients), state_feature_count)
    ]
    base = _dot(groups[0], phi)
    slope_1 = _softplus(_dot(groups[1], phi))
    slope_2 = _softplus(_dot(groups[2], phi))
    slope_3 = _softplus(_dot(groups[3], phi))
    return base + cost_sign * (slope_1 * sqrt_z + slope_2 * z + slope_3 * z * z)


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
    "FEATURE_VERSION_INTERVAL_MONO",
    "FEATURE_VERSION_RETENTION_MONO",
    "FSRS6CostConditionedADRPolicy",
    "POLICY_KIND",
    "STATE_FEATURE_COUNT_COMPACT",
    "STATE_FEATURE_COUNT_HINGE",
    "normalized_cost_weight",
    "normalized_inputs",
    "state_features",
]
