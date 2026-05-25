from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

TargetType = Literal["memory", "time"]


@dataclass(frozen=True, slots=True)
class ConstrainedTarget:
    target_type: TargetType
    value: float
    user_id: int | None = None

    def __post_init__(self) -> None:
        if self.target_type not in {"memory", "time"}:
            raise ValueError("target_type must be 'memory' or 'time'.")
        if not math.isfinite(self.value):
            raise ValueError("target value must be finite.")
        if self.target_type == "memory" and not (0.0 < self.value < 1.0):
            raise ValueError("memory target must satisfy 0 < M0 < 1.")
        if self.target_type == "time" and self.value < 0.0:
            raise ValueError("time target must be >= 0.")

    @property
    def label(self) -> str:
        prefix = "M0" if self.target_type == "memory" else "T0"
        user_suffix = "" if self.user_id is None else f"_u{self.user_id}"
        return f"{prefix}_{self.value:.12g}{user_suffix}"


@dataclass(frozen=True, slots=True)
class EvaluatedPoint:
    user_id: int
    family: str
    theta_name: str
    theta_value: float
    memory: float
    minutes: float
    policy_ref: str | None = None
    cache_key: str | None = None
    exact: bool = False
    eval_stage: str = "confirmed"
    particles: int | None = None
    seed: int | None = None
    runtime_s: float | None = None

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError("user_id must be positive.")
        for field_name, value in (
            ("theta_value", self.theta_value),
            ("memory", self.memory),
            ("minutes", self.minutes),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if self.minutes < 0.0:
            raise ValueError("minutes must be >= 0.")


@dataclass(frozen=True, slots=True)
class FrontierSegment:
    user_id: int
    family: str
    low: EvaluatedPoint
    high: EvaluatedPoint
    slope_lambda: float | None
    certified: bool = False
    certificate_gap: float | None = None


@dataclass(frozen=True, slots=True)
class TargetAnswer:
    target: ConstrainedTarget
    family: str
    feasible: bool
    point: EvaluatedPoint | None
    achieved_memory: float | None
    achieved_minutes: float | None
    memory_slack: float | None
    time_slack: float | None
    certified: bool
    neighbor_low: EvaluatedPoint | None
    neighbor_high: EvaluatedPoint | None
    mixed_available: bool
    mixed_probability_high: float | None
    mixed_memory: float | None
    mixed_minutes: float | None
