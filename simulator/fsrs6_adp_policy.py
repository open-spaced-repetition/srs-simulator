from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


FEATURE_VERSION = "fsrs6_adp_weight_delta_v1"
S_MIN = 0.001
WEIGHT_COUNT = 21
DEFAULT_WEIGHT_DELTA_SCALE = 0.5

FSRS6_ADP_DEFAULT_STDDEV: tuple[float, ...] = (
    6.43,
    9.66,
    17.58,
    27.85,
    0.57,
    0.28,
    0.6,
    0.12,
    0.39,
    0.18,
    0.33,
    0.3,
    0.09,
    0.16,
    0.57,
    0.25,
    1.03,
    0.31,
    0.32,
    0.14,
    0.27,
)

FSRS6_ADP_WEIGHT_BOUNDS: tuple[tuple[float, float], ...] = (
    (S_MIN, 100.0),
    (S_MIN, 100.0),
    (S_MIN, 100.0),
    (S_MIN, 100.0),
    (1.0, 10.0),
    (0.001, 4.0),
    (0.001, 4.0),
    (0.001, 0.75),
    (0.0, 4.5),
    (0.0, 0.8),
    (0.001, 3.5),
    (0.001, 5.0),
    (0.001, 0.25),
    (0.001, 0.9),
    (0.0, 4.0),
    (0.0, 1.0),
    (1.0, 6.0),
    (0.0, 2.0),
    (0.0, 2.0),
    (0.01, 0.8),
    (0.1, 0.8),
)


@dataclass(frozen=True, slots=True)
class FSRS6ADPPolicy:
    base_weights: tuple[float, ...]
    weights: tuple[float, ...]
    delta: tuple[float, ...]
    search_vector: tuple[float, ...]
    baseline_desired_retention: float
    weight_delta_scale: float = DEFAULT_WEIGHT_DELTA_SCALE
    feature_version: str = FEATURE_VERSION
    title: str = "FSRS6 ADP adaptive parameters"

    @classmethod
    def from_json(cls, path: str | Path) -> FSRS6ADPPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"FSRS6 ADP policy {policy_path} must be a JSON object.")
        return cls(
            base_weights=_float_tuple(raw.get("base_weights"), "base_weights"),
            weights=_float_tuple(raw.get("weights"), "weights"),
            delta=_float_tuple(raw.get("delta"), "delta"),
            search_vector=_float_tuple(raw.get("search_vector"), "search_vector"),
            baseline_desired_retention=_float(
                raw.get("baseline_desired_retention"),
                "baseline_desired_retention",
            ),
            weight_delta_scale=_float(
                raw.get("weight_delta_scale", DEFAULT_WEIGHT_DELTA_SCALE),
                "weight_delta_scale",
            ),
            feature_version=_str(
                raw.get("feature_version", FEATURE_VERSION),
                "feature_version",
            ),
            title=_str(raw.get("title", "FSRS6 ADP adaptive parameters"), "title"),
        )

    @classmethod
    def from_search_vector(
        cls,
        *,
        base_weights: Sequence[float],
        search_vector: Sequence[float],
        baseline_desired_retention: float,
        weight_delta_scale: float = DEFAULT_WEIGHT_DELTA_SCALE,
        title: str = "FSRS6 ADP adaptive parameters",
    ) -> FSRS6ADPPolicy:
        base = _tuple_21(base_weights, "base_weights")
        vector = _tuple_21(search_vector, "search_vector")
        weights = decode_weight_delta(
            base,
            vector,
            weight_delta_scale=weight_delta_scale,
        )
        delta = tuple(
            weight - base_weight for weight, base_weight in zip(weights, base)
        )
        return cls(
            base_weights=base,
            weights=weights,
            delta=delta,
            search_vector=vector,
            baseline_desired_retention=float(baseline_desired_retention),
            weight_delta_scale=float(weight_delta_scale),
            title=title,
        )

    def __post_init__(self) -> None:
        _tuple_21(self.base_weights, "base_weights")
        _tuple_21(self.weights, "weights")
        _tuple_21(self.delta, "delta")
        _tuple_21(self.search_vector, "search_vector")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"Unsupported FSRS6 ADP feature_version {self.feature_version!r}."
            )
        if not (0.0 < self.baseline_desired_retention < 1.0):
            raise ValueError("baseline_desired_retention must be between 0 and 1.")
        if self.weight_delta_scale < 0.0:
            raise ValueError("weight_delta_scale must be non-negative.")
        clipped = clip_fsrs6_adp_weights(self.weights)
        if any(abs(a - b) > 1e-6 for a, b in zip(clipped, self.weights)):
            raise ValueError("weights must already satisfy FSRS6 ADP bounds.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": "fsrs6-adp",
            "feature_version": self.feature_version,
            "title": self.title,
            "baseline_desired_retention": self.baseline_desired_retention,
            "weight_delta_scale": self.weight_delta_scale,
            "base_weights": list(self.base_weights),
            "weights": list(self.weights),
            "delta": list(self.delta),
            "search_vector": list(self.search_vector),
            "weight_bounds": [list(bounds) for bounds in FSRS6_ADP_WEIGHT_BOUNDS],
            "default_stddev": list(FSRS6_ADP_DEFAULT_STDDEV),
        }

    def write_json(self, path: str | Path) -> None:
        policy_path = Path(path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        with policy_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def decode_weight_delta(
    base_weights: Sequence[float],
    search_vector: Sequence[float],
    *,
    weight_delta_scale: float = DEFAULT_WEIGHT_DELTA_SCALE,
) -> tuple[float, ...]:
    base = _tuple_21(base_weights, "base_weights")
    vector = _tuple_21(search_vector, "search_vector")
    raw = tuple(
        base_weight + offset * stddev * float(weight_delta_scale)
        for base_weight, offset, stddev in zip(base, vector, FSRS6_ADP_DEFAULT_STDDEV)
    )
    return clip_fsrs6_adp_weights(raw)


def clip_fsrs6_adp_weights(weights: Sequence[float]) -> tuple[float, ...]:
    values = _tuple_21(weights, "weights")
    return tuple(
        min(upper, max(lower, value))
        for value, (lower, upper) in zip(values, FSRS6_ADP_WEIGHT_BOUNDS)
    )


def _tuple_21(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != WEIGHT_COUNT:
        raise ValueError(f"{field_name} must contain {WEIGHT_COUNT} values.")
    return result


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return _tuple_21(value, field_name)


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


__all__ = [
    "DEFAULT_WEIGHT_DELTA_SCALE",
    "FEATURE_VERSION",
    "FSRS6_ADP_DEFAULT_STDDEV",
    "FSRS6_ADP_WEIGHT_BOUNDS",
    "FSRS6ADPPolicy",
    "S_MIN",
    "WEIGHT_COUNT",
    "clip_fsrs6_adp_weights",
    "decode_weight_delta",
]
