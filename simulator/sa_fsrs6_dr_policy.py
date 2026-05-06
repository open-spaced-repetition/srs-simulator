from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from simulator.math.fsrs import Bounds


FEATURE_VERSION = "sa_fsrs6_dr_log_poly_v1"
FEATURE_COUNT = 10


@dataclass(frozen=True, slots=True)
class SAFSRS6DRPolicy:
    coefficients: tuple[float, ...]
    retention_min: float = 0.50
    retention_max: float = 0.98
    bounds: Bounds = Bounds()
    title: str = "SA FSRS-6 DR-conditioned log polynomial"
    feature_version: str = FEATURE_VERSION
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> SAFSRS6DRPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(
                f"SA FSRS-6 DR policy {policy_path} must be a JSON object."
            )
        coefficients = _float_tuple(raw.get("coefficients"), "coefficients")
        bounds_raw = raw.get("bounds", {})
        if bounds_raw is None:
            bounds_raw = {}
        if not isinstance(bounds_raw, dict):
            raise ValueError("bounds must be an object when provided.")
        retention_min = _float(raw.get("retention_min", 0.50), "retention_min")
        retention_max = _float(raw.get("retention_max", 0.98), "retention_max")
        title = raw.get("title", "SA FSRS-6 DR-conditioned log polynomial")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string.")
        feature_version = raw.get("feature_version", FEATURE_VERSION)
        if feature_version != FEATURE_VERSION:
            raise ValueError(
                f"Unsupported SA FSRS-6 DR feature_version {feature_version!r}; "
                f"expected {FEATURE_VERSION!r}."
            )
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
            metadata=raw,
        )

    @classmethod
    def baseline(
        cls,
        *,
        retention_min: float = 0.50,
        retention_max: float = 0.98,
        bounds: Bounds = Bounds(),
    ) -> SAFSRS6DRPolicy:
        return cls(
            coefficients=tuple(0.0 for _ in range(FEATURE_COUNT)),
            retention_min=retention_min,
            retention_max=retention_max,
            bounds=bounds,
        )

    def __post_init__(self) -> None:
        if len(self.coefficients) != FEATURE_COUNT:
            raise ValueError(
                f"SA FSRS-6 DR policy expects {FEATURE_COUNT} coefficients."
            )
        if not (0.0 < self.retention_min < self.retention_max < 1.0):
            raise ValueError("retention_min/max must satisfy 0 < min < max < 1.")
        if self.bounds.s_min <= 0 or self.bounds.s_max <= self.bounds.s_min:
            raise ValueError("bounds must satisfy 0 < s_min < s_max.")
        if self.bounds.d_max <= self.bounds.d_min:
            raise ValueError("bounds must satisfy d_min < d_max.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": "sa-fsrs6-dr",
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
        features = log_poly_dr_features(
            stability,
            difficulty,
            desired,
            bounds=self.bounds,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
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
    x_s = min(1.0, max(0.0, x_s))
    x_d = min(1.0, max(0.0, x_d))
    x_dr = min(1.0, max(0.0, x_dr))
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
    "FEATURE_VERSION",
    "SAFSRS6DRPolicy",
    "log_poly_dr_features",
]
