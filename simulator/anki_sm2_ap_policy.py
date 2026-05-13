from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


FEATURE_VERSION = "anki_sm2_ap_params_v1"
PARAM_COUNT = 7
DEFAULT_PARAMETER_DELTA_SCALE = 1.0

ANKI_SM2_AP_PARAM_NAMES: tuple[str, ...] = (
    "graduating_interval",
    "easy_interval",
    "ease_start",
    "easy_bonus",
    "hard_interval_factor",
    "new_interval_factor",
    "interval_multiplier",
)

ANKI_SM2_AP_DEFAULT_PARAMS: tuple[float, ...] = (
    1.0,
    4.0,
    2.5,
    1.3,
    1.2,
    0.0,
    1.0,
)

ANKI_SM2_AP_PARAM_BOUNDS: tuple[tuple[float, float], ...] = (
    (1.0, 100.0),
    (1.0, 100.0),
    (1.31, 5.0),
    (1.0, 5.0),
    (0.5, 1.3),
    (0.0, 1.0),
    (0.5, 2.0),
)

ANKI_SM2_AP_DEFAULT_STDDEV: tuple[float, ...] = (
    9.9,
    9.9,
    0.369,
    0.4,
    0.08,
    0.1,
    0.15,
)


@dataclass(frozen=True, slots=True)
class AnkiSM2APPolicy:
    base_params: tuple[float, ...]
    params: tuple[float, ...]
    delta: tuple[float, ...]
    search_vector: tuple[float, ...]
    parameter_delta_scale: float = DEFAULT_PARAMETER_DELTA_SCALE
    feature_version: str = FEATURE_VERSION
    title: str = "Anki SM2 AP adaptive parameters"

    @classmethod
    def from_json(cls, path: str | Path) -> AnkiSM2APPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Anki SM2 AP policy {policy_path} must be a JSON object.")
        return cls(
            base_params=_float_tuple(raw.get("base_params"), "base_params"),
            params=_float_tuple(raw.get("params"), "params"),
            delta=_float_tuple(raw.get("delta"), "delta"),
            search_vector=_float_tuple(raw.get("search_vector"), "search_vector"),
            parameter_delta_scale=_float(
                raw.get("parameter_delta_scale", DEFAULT_PARAMETER_DELTA_SCALE),
                "parameter_delta_scale",
            ),
            feature_version=_str(
                raw.get("feature_version", FEATURE_VERSION),
                "feature_version",
            ),
            title=_str(raw.get("title", "Anki SM2 AP adaptive parameters"), "title"),
        )

    @classmethod
    def from_search_vector(
        cls,
        *,
        base_params: Sequence[float] = ANKI_SM2_AP_DEFAULT_PARAMS,
        search_vector: Sequence[float],
        parameter_delta_scale: float = DEFAULT_PARAMETER_DELTA_SCALE,
        title: str = "Anki SM2 AP adaptive parameters",
    ) -> AnkiSM2APPolicy:
        base = _tuple_7(base_params, "base_params")
        vector = _tuple_7(search_vector, "search_vector")
        params = decode_parameter_delta(
            base,
            vector,
            parameter_delta_scale=parameter_delta_scale,
        )
        delta = tuple(param - base_param for param, base_param in zip(params, base))
        return cls(
            base_params=base,
            params=params,
            delta=delta,
            search_vector=vector,
            parameter_delta_scale=float(parameter_delta_scale),
            title=title,
        )

    def __post_init__(self) -> None:
        _tuple_7(self.base_params, "base_params")
        _tuple_7(self.params, "params")
        _tuple_7(self.delta, "delta")
        _tuple_7(self.search_vector, "search_vector")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"Unsupported Anki SM2 AP feature_version {self.feature_version!r}."
            )
        if self.parameter_delta_scale < 0.0:
            raise ValueError("parameter_delta_scale must be non-negative.")
        clipped = clip_anki_sm2_ap_params(self.params)
        if any(abs(a - b) > 1e-6 for a, b in zip(clipped, self.params)):
            raise ValueError("params must already satisfy Anki SM2 AP bounds.")

    def params_dict(self) -> dict[str, float]:
        return dict(zip(ANKI_SM2_AP_PARAM_NAMES, self.params, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": "anki-sm2-ap",
            "feature_version": self.feature_version,
            "title": self.title,
            "parameter_delta_scale": self.parameter_delta_scale,
            "param_names": list(ANKI_SM2_AP_PARAM_NAMES),
            "base_params": list(self.base_params),
            "params": list(self.params),
            "delta": list(self.delta),
            "search_vector": list(self.search_vector),
            "param_bounds": [list(bounds) for bounds in ANKI_SM2_AP_PARAM_BOUNDS],
            "default_stddev": list(ANKI_SM2_AP_DEFAULT_STDDEV),
        }

    def write_json(self, path: str | Path) -> None:
        policy_path = Path(path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        with policy_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def decode_parameter_delta(
    base_params: Sequence[float],
    search_vector: Sequence[float],
    *,
    parameter_delta_scale: float = DEFAULT_PARAMETER_DELTA_SCALE,
) -> tuple[float, ...]:
    base = _tuple_7(base_params, "base_params")
    vector = _tuple_7(search_vector, "search_vector")
    raw = tuple(
        base_value + offset * stddev * float(parameter_delta_scale)
        for base_value, offset, stddev in zip(
            base,
            vector,
            ANKI_SM2_AP_DEFAULT_STDDEV,
            strict=True,
        )
    )
    return clip_anki_sm2_ap_params(raw)


def clip_anki_sm2_ap_params(params: Sequence[float]) -> tuple[float, ...]:
    values = _tuple_7(params, "params")
    return tuple(
        min(upper, max(lower, value))
        for value, (lower, upper) in zip(
            values,
            ANKI_SM2_AP_PARAM_BOUNDS,
            strict=True,
        )
    )


def _tuple_7(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != PARAM_COUNT:
        raise ValueError(f"{field_name} must contain {PARAM_COUNT} values.")
    return result


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    return _tuple_7(value, field_name)


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


__all__ = [
    "ANKI_SM2_AP_DEFAULT_PARAMS",
    "ANKI_SM2_AP_DEFAULT_STDDEV",
    "ANKI_SM2_AP_PARAM_BOUNDS",
    "ANKI_SM2_AP_PARAM_NAMES",
    "DEFAULT_PARAMETER_DELTA_SCALE",
    "FEATURE_VERSION",
    "PARAM_COUNT",
    "AnkiSM2APPolicy",
    "clip_anki_sm2_ap_params",
    "decode_parameter_delta",
]
