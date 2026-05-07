from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from simulator.math.fsrs import Bounds


FEATURE_VERSION_LOG_POLY = "fsrs6_adr_delta_log_poly_v1"
FEATURE_VERSION_LOG_LINEAR = "fsrs6_adr_delta_log_linear_v1"
FEATURE_VERSION = FEATURE_VERSION_LOG_POLY
FEATURE_COUNT = 10
FEATURE_COUNTS = {
    FEATURE_VERSION_LOG_POLY: 10,
    FEATURE_VERSION_LOG_LINEAR: 4,
}


@dataclass(frozen=True, slots=True)
class FSRS6ADRDeltaPolicy:
    coefficients: tuple[float, ...]
    retention_min: float = 0.50
    retention_max: float = 0.98
    bounds: Bounds = Bounds()
    title: str = "FSRS6 ADR Delta-conditioned log polynomial"
    feature_version: str = FEATURE_VERSION
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> FSRS6ADRDeltaPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(
                f"FSRS6 ADR Delta policy {policy_path} must be a JSON object."
            )
        coefficients = _float_tuple(raw.get("coefficients"), "coefficients")
        bounds_raw = raw.get("bounds", {})
        if bounds_raw is None:
            bounds_raw = {}
        if not isinstance(bounds_raw, dict):
            raise ValueError("bounds must be an object when provided.")
        retention_min = _float(raw.get("retention_min", 0.50), "retention_min")
        retention_max = _float(raw.get("retention_max", 0.98), "retention_max")
        feature_version = raw.get("feature_version", FEATURE_VERSION)
        if not isinstance(feature_version, str):
            raise ValueError("feature_version must be a string.")
        _feature_count(feature_version)
        title = raw.get("title", _default_title(feature_version))
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string.")
        return cls(
            coefficients=coefficients,
            retention_min=retention_min,
            retention_max=retention_max,
            bounds=Bounds(
                s_min=_float(bounds_raw.get("s_min", Bounds().s_min), "bounds.s_min"),
                s_max=_float(bounds_raw.get("s_max", Bounds().s_max), "bounds.s_max"),
                d_min=_float(bounds_raw.get("d_min", Bounds().d_min), "bounds.d_min"),
                d_max=_float(bounds_raw.get("d_max", Bounds().d_max), "bounds.d_max"),
            ),
            title=title.strip(),
            feature_version=feature_version,
            metadata=raw,
        )

    @classmethod
    def baseline(
        cls,
        *,
        retention_min: float = 0.50,
        retention_max: float = 0.98,
        bounds: Bounds = Bounds(),
        feature_version: str = FEATURE_VERSION,
    ) -> FSRS6ADRDeltaPolicy:
        return cls(
            coefficients=tuple(0.0 for _ in range(_feature_count(feature_version))),
            retention_min=retention_min,
            retention_max=retention_max,
            bounds=bounds,
            title=_default_title(feature_version),
            feature_version=feature_version,
        )

    def __post_init__(self) -> None:
        expected_count = _feature_count(self.feature_version)
        if len(self.coefficients) != expected_count:
            raise ValueError(
                "FSRS6 ADR Delta policy expects "
                f"{expected_count} coefficients for {self.feature_version!r}."
            )
        if not (0.0 < self.retention_min < self.retention_max < 1.0):
            raise ValueError("retention_min/max must satisfy 0 < min < max < 1.")
        if self.bounds.s_min <= 0 or self.bounds.s_max <= self.bounds.s_min:
            raise ValueError("bounds must satisfy 0 < s_min < s_max.")
        if self.bounds.d_max <= self.bounds.d_min:
            raise ValueError("bounds must satisfy d_min < d_max.")

    @property
    def feature_count(self) -> int:
        return _feature_count(self.feature_version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": "fsrs6-adr-delta",
            "feature_version": self.feature_version,
            "title": self.title,
            "coefficients": list(self.coefficients),
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
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

    def evaluate(
        self,
        stability: float,
        difficulty: float,
        desired_retention: float,
    ) -> float:
        desired = _validate_desired_retention(
            desired_retention,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
        )
        if all(coefficient == 0.0 for coefficient in self.coefficients):
            return desired
        features = dr_features(
            stability,
            difficulty,
            desired,
            bounds=self.bounds,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            feature_version=self.feature_version,
        )
        adjustment = sum(
            coef * feature for coef, feature in zip(self.coefficients, features)
        )
        logit = _retention_logit(
            desired,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
        )
        return self.retention_min + (
            self.retention_max - self.retention_min
        ) * _sigmoid(logit + adjustment)


def log_poly_dr_features(
    stability: float,
    difficulty: float,
    desired_retention: float,
    *,
    bounds: Bounds = Bounds(),
    retention_min: float = 0.50,
    retention_max: float = 0.98,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    x_s, x_d, x_dr = _normalized_dr_inputs(
        stability,
        difficulty,
        desired_retention,
        bounds=bounds,
        retention_min=retention_min,
        retention_max=retention_max,
    )
    return (
        1.0,
        x_s,
        x_d,
        x_dr,
        x_s * x_d,
        x_s * x_dr,
        x_d * x_dr,
        x_s * x_s,
        x_d * x_d,
        x_dr * x_dr,
    )


def log_linear_dr_features(
    stability: float,
    difficulty: float,
    desired_retention: float,
    *,
    bounds: Bounds = Bounds(),
    retention_min: float = 0.50,
    retention_max: float = 0.98,
) -> tuple[float, float, float, float]:
    x_s, x_d, x_dr = _normalized_dr_inputs(
        stability,
        difficulty,
        desired_retention,
        bounds=bounds,
        retention_min=retention_min,
        retention_max=retention_max,
    )
    return (1.0, x_s, x_d, x_dr)


def dr_features(
    stability: float,
    difficulty: float,
    desired_retention: float,
    *,
    bounds: Bounds = Bounds(),
    retention_min: float = 0.50,
    retention_max: float = 0.98,
    feature_version: str = FEATURE_VERSION,
) -> tuple[float, ...]:
    if feature_version == FEATURE_VERSION_LOG_POLY:
        return log_poly_dr_features(
            stability,
            difficulty,
            desired_retention,
            bounds=bounds,
            retention_min=retention_min,
            retention_max=retention_max,
        )
    if feature_version == FEATURE_VERSION_LOG_LINEAR:
        return log_linear_dr_features(
            stability,
            difficulty,
            desired_retention,
            bounds=bounds,
            retention_min=retention_min,
            retention_max=retention_max,
        )
    _feature_count(feature_version)
    raise AssertionError("unreachable feature version dispatch")


def feature_count(feature_version: str = FEATURE_VERSION) -> int:
    return _feature_count(feature_version)


def _normalized_dr_inputs(
    stability: float,
    difficulty: float,
    desired_retention: float,
    *,
    bounds: Bounds,
    retention_min: float,
    retention_max: float,
) -> tuple[float, float, float]:
    s = min(bounds.s_max, max(bounds.s_min, float(stability)))
    d = min(bounds.d_max, max(bounds.d_min, float(difficulty)))
    desired = _validate_desired_retention(
        desired_retention,
        retention_min=retention_min,
        retention_max=retention_max,
    )
    log_s_min = math.log(bounds.s_min)
    log_s_max = math.log(bounds.s_max)
    x_s = (math.log(s) - log_s_min) / (log_s_max - log_s_min)
    x_d = (d - bounds.d_min) / (bounds.d_max - bounds.d_min)
    x_dr = (desired - retention_min) / (retention_max - retention_min)
    return (
        min(1.0, max(0.0, x_s)),
        min(1.0, max(0.0, x_d)),
        min(1.0, max(0.0, x_dr)),
    )


def _feature_count(feature_version: str) -> int:
    try:
        return FEATURE_COUNTS[feature_version]
    except KeyError as exc:
        expected = ", ".join(sorted(FEATURE_COUNTS))
        raise ValueError(
            f"Unsupported FSRS6 ADR Delta feature_version {feature_version!r}; "
            f"expected one of: {expected}."
        ) from exc


def _default_title(feature_version: str) -> str:
    if feature_version == FEATURE_VERSION_LOG_LINEAR:
        return "FSRS6 ADR Delta-conditioned log linear"
    _feature_count(feature_version)
    return "FSRS6 ADR Delta-conditioned log polynomial"


def _validate_desired_retention(
    value: float,
    *,
    retention_min: float,
    retention_max: float,
) -> float:
    desired = float(value)
    if not (retention_min <= desired <= retention_max):
        raise ValueError(
            "desired_retention must be inside the policy retention bounds "
            f"[{retention_min}, {retention_max}]."
        )
    return desired


def _retention_logit(
    value: float,
    *,
    retention_min: float,
    retention_max: float,
) -> float:
    ratio = (float(value) - retention_min) / (retention_max - retention_min)
    if ratio <= 0.0:
        return -math.inf
    if ratio >= 1.0:
        return math.inf
    return math.log(ratio / (1.0 - ratio))


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return tuple(
        _float(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


__all__ = [
    "FEATURE_COUNT",
    "FEATURE_COUNTS",
    "FEATURE_VERSION",
    "FEATURE_VERSION_LOG_LINEAR",
    "FEATURE_VERSION_LOG_POLY",
    "FSRS6ADRDeltaPolicy",
    "dr_features",
    "feature_count",
    "log_linear_dr_features",
    "log_poly_dr_features",
]
