from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import torch

from experiments.single_card_tradeoff.oracles.dp_cache import (
    OracleDPCacheConfig,
    load_cache_entry,
    resolve_oracle_dp_cache_config,
    write_cache_entry,
)
from experiments.single_card_tradeoff.core.retention_space import (
    validate_retention_values_for_model,
)
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds
from simulator.scheduler_spec import format_float


@dataclass(frozen=True)
class OracleMetrics:
    card_expected_retrievability: float
    card_minutes_per_day: float
    card_reviews_per_day: float
    card_total_reviews: float
    card_total_lapses: float
    card_total_cost_seconds: float
    observed_retention: float | None
    scalar_objective: float
    runtime_s: float


@dataclass(frozen=True)
class TransitionCache:
    interval: torch.Tensor
    prob: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    next_s_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    next_d_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    next_idx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    next_weight: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class BatchedTransitionCache:
    interval: torch.Tensor
    prob: torch.Tensor
    next_idx: torch.Tensor
    next_weight: torch.Tensor


@dataclass(frozen=True)
class StationaryActionKernelTables:
    interval: torch.Tensor
    prob: torch.Tensor
    next_idx: torch.Tensor
    next_weight: torch.Tensor


@dataclass(frozen=True)
class OracleSolution:
    metrics: OracleMetrics
    policy: torch.Tensor | None


@dataclass(frozen=True)
class AverageRewardOracleSolution:
    policy: torch.Tensor
    gains: torch.Tensor
    iterations: list[int]
    converged: list[bool]
    residuals: list[float]
    runtime_s: float


@dataclass(frozen=True)
class StationaryFiniteOracleSolution:
    policy: torch.Tensor
    metrics: list[OracleMetrics]
    objectives: torch.Tensor
    iterations: list[int]
    converged: list[bool]
    residuals: list[float]
    runtime_s: float


@dataclass(frozen=True)
class BatchedStationaryFiniteOracleSolution:
    policy: torch.Tensor
    metrics: list[list[OracleMetrics]]
    objectives: torch.Tensor
    iterations: list[list[int]]
    converged: list[list[bool]]
    residuals: list[list[float]]
    runtime_s: float


def _tensor_float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _metrics_payload(metrics: OracleMetrics) -> dict[str, float | None]:
    return {
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "observed_retention": metrics.observed_retention,
        "scalar_objective": metrics.scalar_objective,
    }


def _metrics_from_payload(
    payload: dict[str, Any],
    *,
    runtime_s: float = 0.0,
) -> OracleMetrics:
    return OracleMetrics(
        card_expected_retrievability=float(payload["card_expected_retrievability"]),
        card_minutes_per_day=float(payload["card_minutes_per_day"]),
        card_reviews_per_day=float(payload["card_reviews_per_day"]),
        card_total_reviews=float(payload["card_total_reviews"]),
        card_total_lapses=float(payload["card_total_lapses"]),
        card_total_cost_seconds=float(payload["card_total_cost_seconds"]),
        observed_retention=(
            None
            if payload.get("observed_retention") is None
            else float(payload["observed_retention"])
        ),
        scalar_objective=float(payload["scalar_objective"]),
        runtime_s=runtime_s,
    )


def scalar_objective(metrics: Any, cost_weight: float) -> float:
    return float(metrics.card_expected_retrievability) - cost_weight * float(
        metrics.card_minutes_per_day
    )


def validate_continuous_retention_bounds(
    retention_min: float,
    retention_max: float,
) -> None:
    if not math.isfinite(retention_min) or not math.isfinite(retention_max):
        raise ValueError("retention bounds must be finite.")
    if not 0.0 < retention_min <= retention_max < 1.0:
        raise ValueError("retention bounds must satisfy 0 < min <= max < 1.")


def retention_interval_float(
    *,
    s: torch.Tensor,
    retention: torch.Tensor | float,
    factor: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    retention_tensor = torch.as_tensor(retention, device=s.device, dtype=s.dtype)
    clipped = torch.clamp(retention_tensor, min=1e-7, max=1.0 - 1e-7)
    retention_factor = torch.pow(clipped, 1.0 / decay.to(dtype=s.dtype)) - 1.0
    return s / factor.to(dtype=s.dtype) * retention_factor


def retention_interval_bounds(
    *,
    s: torch.Tensor,
    retention_min: float,
    retention_max: float,
    factor: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    low_retention_interval = retention_interval_float(
        s=s,
        retention=retention_min,
        factor=factor,
        decay=decay,
    )
    high_retention_interval = retention_interval_float(
        s=s,
        retention=retention_max,
        factor=factor,
        decay=decay,
    )
    lower = torch.minimum(low_retention_interval, high_retention_interval)
    upper = torch.maximum(low_retention_interval, high_retention_interval)
    return lower, upper


def attainable_interval_mask_for_retention_bounds(
    *,
    intervals: torch.Tensor,
    s_grid: torch.Tensor,
    retention_min: float,
    retention_max: float,
    factor: torch.Tensor,
    decay: torch.Tensor,
    terminal_interval: int | None = None,
) -> torch.Tensor:
    validate_continuous_retention_bounds(retention_min, retention_max)
    lower_float, upper_float = retention_interval_bounds(
        s=s_grid,
        retention_min=retention_min,
        retention_max=retention_max,
        factor=factor,
        decay=decay,
    )
    rounded_lower = torch.clamp(torch.round(lower_float), min=1.0).to(torch.int64)
    rounded_upper = torch.clamp(torch.round(upper_float), min=1.0).to(torch.int64)
    lo = torch.minimum(rounded_lower, rounded_upper)
    hi = torch.maximum(rounded_lower, rounded_upper)
    interval_col = intervals.to(device=s_grid.device, dtype=torch.int64)[:, None]
    mask = (interval_col >= lo[None, :]) & (interval_col <= hi[None, :])
    if terminal_interval is not None:
        terminal = torch.as_tensor(
            terminal_interval,
            device=s_grid.device,
            dtype=torch.int64,
        )
        terminal_mask = (interval_col == terminal) & (hi[None, :] >= terminal)
        mask = torch.where(interval_col == terminal, terminal_mask, mask)
        mask = mask & (interval_col <= terminal)
    return mask


def canonical_retention_for_intervals(
    *,
    intervals: torch.Tensor,
    s_grid: torch.Tensor,
    factor: torch.Tensor,
    decay: torch.Tensor,
    retention_min: float,
    retention_max: float,
) -> torch.Tensor:
    validate_continuous_retention_bounds(retention_min, retention_max)
    elapsed = intervals.to(device=s_grid.device, dtype=s_grid.dtype)[:, None]
    retrievability = torch.pow(
        1.0
        + factor.to(dtype=s_grid.dtype)
        * elapsed
        / torch.clamp(s_grid[None, :], min=torch.finfo(s_grid.dtype).tiny),
        decay.to(dtype=s_grid.dtype),
    )
    return torch.clamp(
        retrievability,
        min=float(retention_min),
        max=float(retention_max),
    )


def bilinear_retention_policy_lookup(
    *,
    oracle: Any,
    policies: torch.Tensor,
    goal_indices: torch.Tensor,
    user_indices: torch.Tensor | None = None,
    s: torch.Tensor,
    d: torch.Tensor,
    retention_min: float,
    retention_max: float,
    remaining: torch.Tensor | None = None,
) -> torch.Tensor:
    validate_continuous_retention_bounds(retention_min, retention_max)
    s_count = int(oracle.s_grid.numel())
    d_count = int(oracle.d_grid.numel())
    log_s = torch.log(torch.clamp(s, oracle.bounds.s_min, oracle.bounds.s_max))
    s_pos = (log_s - oracle.log_s_min) / (oracle.log_s_max - oracle.log_s_min)
    s_pos = torch.clamp(s_pos * float(s_count - 1), 0.0, float(s_count - 1))
    s0 = torch.floor(s_pos).to(torch.int64)
    s1 = torch.clamp(s0 + 1, max=s_count - 1)
    sw = s_pos - s0.to(dtype=s_pos.dtype)

    d_pos = torch.clamp(d, oracle.bounds.d_min, oracle.bounds.d_max)
    d_pos = (d_pos - oracle.bounds.d_min) / (oracle.bounds.d_max - oracle.bounds.d_min)
    d_pos = torch.clamp(d_pos * float(d_count - 1), 0.0, float(d_count - 1))
    d0 = torch.floor(d_pos).to(torch.int64)
    d1 = torch.clamp(d0 + 1, max=d_count - 1)
    dw = d_pos - d0.to(dtype=d_pos.dtype)

    table = policies.to(device=s.device)
    if user_indices is None:
        if remaining is None:
            a00 = table[goal_indices, s0, d0]
            a10 = table[goal_indices, s1, d0]
            a01 = table[goal_indices, s0, d1]
            a11 = table[goal_indices, s1, d1]
        else:
            a00 = table[goal_indices, remaining, s0, d0]
            a10 = table[goal_indices, remaining, s1, d0]
            a01 = table[goal_indices, remaining, s0, d1]
            a11 = table[goal_indices, remaining, s1, d1]
    else:
        users = user_indices.to(device=s.device, dtype=torch.int64)
        if remaining is None:
            a00 = table[users, goal_indices, s0, d0]
            a10 = table[users, goal_indices, s1, d0]
            a01 = table[users, goal_indices, s0, d1]
            a11 = table[users, goal_indices, s1, d1]
        else:
            a00 = table[users, goal_indices, remaining, s0, d0]
            a10 = table[users, goal_indices, remaining, s1, d0]
            a01 = table[users, goal_indices, remaining, s0, d1]
            a11 = table[users, goal_indices, remaining, s1, d1]

    dtype = s.dtype
    retention = (
        a00.to(dtype=dtype) * (1.0 - sw) * (1.0 - dw)
        + a10.to(dtype=dtype) * sw * (1.0 - dw)
        + a01.to(dtype=dtype) * (1.0 - sw) * dw
        + a11.to(dtype=dtype) * sw * dw
    )
    return torch.clamp(
        retention,
        min=float(retention_min),
        max=float(retention_max),
    )


class FSRS6GridOracle:
    TRANSITION_KERNEL_VERSION = "four_corner_log_s_linear_d_v1"

    def __init__(
        self,
        *,
        days: int,
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        if days <= 1:
            raise ValueError("days must be > 1.")
        if s_grid_size < 8 or d_grid_size < 8:
            raise ValueError("grid sizes must be >= 8.")
        validate_retention_values_for_model(
            action_retentions,
            name="action retention",
        )

        self.days = int(days)
        self.horizon = int(days - 1)
        self.dtype = dtype
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.cache_config = resolve_oracle_dp_cache_config(cache_config)
        self.bounds = Bounds()
        resolved_weights = (
            DEFAULT_FSRS6_WEIGHTS if fsrs_weights is None else tuple(fsrs_weights)
        )
        if len(resolved_weights) != 21:
            raise ValueError("FSRS6 weights must contain 21 values.")
        self.weights = torch.tensor(resolved_weights, device=self.device, dtype=dtype)
        self.decay = -self.weights[20]
        self.factor = (
            torch.pow(
                torch.tensor(0.9, device=self.device, dtype=dtype),
                1.0 / self.decay,
            )
            - 1.0
        )
        self.init_d = torch.clamp(
            self.weights[4] - torch.exp(self.weights[5] * 3.0) + 1.0,
            self.bounds.d_min,
            self.bounds.d_max,
        )
        self.action_retentions = torch.tensor(
            list(action_retentions), device=self.device, dtype=dtype
        )
        self.action_retention_factor = (
            torch.pow(self.action_retentions, 1.0 / self.decay) - 1.0
        )
        resolved_first_prob = (
            DEFAULT_FIRST_RATING_PROB
            if first_rating_prob is None
            else tuple(first_rating_prob)
        )
        resolved_review_prob = (
            DEFAULT_REVIEW_RATING_PROB
            if review_rating_prob is None
            else tuple(review_rating_prob)
        )
        if len(resolved_first_prob) != 4:
            raise ValueError("first_rating_prob must contain 4 values.")
        if len(resolved_review_prob) != 3:
            raise ValueError("review_rating_prob must contain 3 values.")
        self.first_rating_prob = torch.tensor(
            resolved_first_prob, device=self.device, dtype=dtype
        )
        self.review_rating_prob = torch.tensor(
            resolved_review_prob, device=self.device, dtype=dtype
        )
        resolved_learning_costs = (
            DEFAULT_STATE_RATING_COSTS.learning
            if learning_costs is None
            else tuple(learning_costs)
        )
        resolved_review_costs = (
            DEFAULT_STATE_RATING_COSTS.review
            if review_costs is None
            else tuple(review_costs)
        )
        if len(resolved_learning_costs) != 4 or len(resolved_review_costs) != 4:
            raise ValueError("learning_costs and review_costs must contain 4 values.")
        self.learning_cost_minutes = (
            torch.tensor(
                resolved_learning_costs,
                device=self.device,
                dtype=dtype,
            )
            / 60.0
        )
        self.review_cost_minutes = (
            torch.tensor(
                resolved_review_costs,
                device=self.device,
                dtype=dtype,
            )
            / 60.0
        )

        self.log_s_min = math.log(self.bounds.s_min)
        self.log_s_max = math.log(self.bounds.s_max)
        self.s_grid = torch.exp(
            torch.linspace(
                self.log_s_min,
                self.log_s_max,
                s_grid_size,
                device=self.device,
                dtype=dtype,
            )
        )
        self.d_grid = torch.linspace(
            self.bounds.d_min,
            self.bounds.d_max,
            d_grid_size,
            device=self.device,
            dtype=dtype,
        )
        self.s_mesh = self.s_grid[:, None].expand(s_grid_size, d_grid_size)
        self.d_mesh = self.d_grid[None, :].expand(s_grid_size, d_grid_size)
        self.s_count = int(self.s_grid.numel())
        self.d_count = int(self.d_grid.numel())
        self.state_count = self.s_count * self.d_count
        self._s_grid_idx = torch.arange(self.s_grid.numel(), device=self.device)
        self.memorized_by_day = self._precompute_grid_memorized_by_day()
        self.transitions = self._precompute_transitions()

    def _single_user_payload(self) -> dict[str, Any]:
        return {
            "fsrs_weights": _tensor_float_list(self.weights),
            "first_rating_prob": _tensor_float_list(self.first_rating_prob),
            "review_rating_prob": _tensor_float_list(self.review_rating_prob),
            "learning_costs": [
                60.0 * value for value in _tensor_float_list(self.learning_cost_minutes)
            ],
            "review_costs": [
                60.0 * value for value in _tensor_float_list(self.review_cost_minutes)
            ],
        }

    def _cache_key_parts(
        self,
        *,
        oracle_kind: str,
        method: str,
        cost_weight: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "algorithm_version": 1,
            "oracle_kind": oracle_kind,
            "method": method,
            "days": self.days,
            "s_grid_size": int(self.s_grid.numel()),
            "d_grid_size": int(self.d_grid.numel()),
            "action_retentions": _tensor_float_list(self.action_retentions),
            "dtype": str(self.dtype),
            "user_config": self._single_user_payload(),
            "cost_weight": float(cost_weight),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def _grid_cache_extra(
        self,
        *,
        capture_policy: bool | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "transition_kernel": self.TRANSITION_KERNEL_VERSION,
        }
        if capture_policy is not None:
            extra["capture_policy"] = capture_policy
        return extra

    def estimate(self, cost_weight: float, *, progress: bool = False) -> OracleMetrics:
        return self.solve(
            cost_weight,
            progress=progress,
            capture_policy=False,
        ).metrics

    def estimate_many(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> list[OracleMetrics]:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        start = time.perf_counter()
        results: list[OracleMetrics | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="grid",
                    method="estimate_many",
                    cost_weight=float(weight),
                    extra=self._grid_cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None:
                missing.append((idx, float(weight)))
                continue
            results[idx] = _metrics_from_payload(entry["metrics"], runtime_s=0.0)

        if missing:
            computed = self._estimate_many_uncached(
                [weight for _, weight in missing],
                progress=progress,
            )
            for (idx, weight), metrics in zip(missing, computed, strict=True):
                results[idx] = metrics
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="grid",
                        method="estimate_many",
                        cost_weight=weight,
                        extra=self._grid_cache_extra(),
                    ),
                    data={"metrics": _metrics_payload(metrics)},
                )
        elapsed = time.perf_counter() - start
        return [
            _metrics_from_payload(_metrics_payload(metrics), runtime_s=elapsed)
            for metrics in results
            if metrics is not None
        ]

    def _estimate_many_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> list[OracleMetrics]:
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        return self._estimate_many_batch(weight_tensor, progress=progress)

    def solve(
        self,
        cost_weight: float,
        *,
        progress: bool = False,
        capture_policy: bool = False,
    ) -> OracleSolution:
        start = time.perf_counter()
        key_parts = self._cache_key_parts(
            oracle_kind="grid",
            method="solve",
            cost_weight=float(cost_weight),
            extra=self._grid_cache_extra(capture_policy=capture_policy),
        )
        entry = load_cache_entry(
            self.cache_config,
            key_parts=key_parts,
            map_location=self.device,
        )
        if entry is not None:
            policy = entry.get("policy")
            if policy is not None and not isinstance(policy, torch.Tensor):
                policy = None
            return OracleSolution(
                metrics=_metrics_from_payload(
                    entry["metrics"],
                    runtime_s=time.perf_counter() - start,
                ),
                policy=policy.to(device=self.device) if policy is not None else None,
            )

        solution = self._solve_uncached(
            cost_weight,
            progress=progress,
            capture_policy=capture_policy,
        )
        write_cache_entry(
            self.cache_config,
            key_parts=key_parts,
            data={
                "metrics": _metrics_payload(solution.metrics),
                "policy": solution.policy,
            },
        )
        return solution

    def _solve_uncached(
        self,
        cost_weight: float,
        *,
        progress: bool = False,
        capture_policy: bool = False,
    ) -> OracleSolution:
        start = time.perf_counter()
        shape = (self.horizon + 1, self.s_grid.numel(), self.d_grid.numel())
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        memorized = torch.zeros_like(value)
        minutes = torch.zeros_like(value)
        reviews = torch.zeros_like(value)
        lapses = torch.zeros_like(value)
        policy = (
            torch.zeros(shape, device=self.device, dtype=torch.int64)
            if capture_policy
            else None
        )

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Oracle w={format_float(cost_weight)}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_mem = torch.zeros_like(best_value)
                best_minutes = torch.zeros_like(best_value)
                best_reviews = torch.zeros_like(best_value)
                best_lapses = torch.zeros_like(best_value)
                best_action = torch.zeros_like(best_value, dtype=torch.int64)

                for action_idx, transition in enumerate(self.transitions):
                    candidate = self._candidate_tables(
                        transition=transition,
                        rem=rem,
                        cost_weight=cost_weight,
                        value=value,
                        memorized=memorized,
                        minutes=minutes,
                        reviews=reviews,
                        lapses=lapses,
                    )
                    candidate_value, candidate_mem, candidate_minutes = candidate[:3]
                    candidate_reviews, candidate_lapses = candidate[3:]
                    better = candidate_value > best_value
                    best_value = torch.where(better, candidate_value, best_value)
                    best_mem = torch.where(better, candidate_mem, best_mem)
                    best_minutes = torch.where(better, candidate_minutes, best_minutes)
                    best_reviews = torch.where(better, candidate_reviews, best_reviews)
                    best_lapses = torch.where(better, candidate_lapses, best_lapses)
                    best_action = torch.where(
                        better,
                        torch.full_like(best_action, action_idx),
                        best_action,
                    )

                value[rem] = best_value
                memorized[rem] = best_mem
                minutes[rem] = best_minutes
                reviews[rem] = best_reviews
                lapses[rem] = best_lapses
                if policy is not None:
                    policy[rem] = best_action
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        total_mem = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_minutes = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_reviews = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_lapses = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        mem_flat = memorized.reshape(memorized.shape[0], self.state_count)
        minutes_flat = minutes.reshape(minutes.shape[0], self.state_count)
        reviews_flat = reviews.reshape(reviews.shape[0], self.state_count)
        lapses_flat = lapses.reshape(lapses.shape[0], self.state_count)
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_mem += prob * (state_weight * mem_flat[self.horizon, state_idx]).sum()
            total_minutes += prob * (
                self.learning_cost_minutes[rating - 1]
                + (state_weight * minutes_flat[self.horizon, state_idx]).sum()
            )
            total_reviews += (
                prob * (state_weight * reviews_flat[self.horizon, state_idx]).sum()
            )
            total_lapses += (
                prob * (state_weight * lapses_flat[self.horizon, state_idx]).sum()
            )

        day_count = float(self.days)
        reviews_float = float(total_reviews.item())
        lapses_float = float(total_lapses.item())
        observed_retention = (
            1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
        )
        mem_per_day = float(total_mem.item() / day_count)
        minutes_per_day = float(total_minutes.item() / day_count)
        return OracleSolution(
            metrics=OracleMetrics(
                card_expected_retrievability=mem_per_day,
                card_minutes_per_day=minutes_per_day,
                card_reviews_per_day=reviews_float / day_count,
                card_total_reviews=reviews_float,
                card_total_lapses=lapses_float,
                card_total_cost_seconds=float(total_minutes.item() * 60.0),
                observed_retention=observed_retention,
                scalar_objective=mem_per_day - cost_weight * minutes_per_day,
                runtime_s=time.perf_counter() - start,
            ),
            policy=policy,
        )

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            key_parts = self._cache_key_parts(
                oracle_kind="grid",
                method="solve_policies",
                cost_weight=float(weight),
                extra=self._grid_cache_extra(),
            )
            entry = load_cache_entry(
                self.cache_config,
                key_parts=key_parts,
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device)

        if missing:
            computed = self._solve_policies_uncached(
                [weight for _, weight in missing],
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = computed[local_idx].to(device=self.device)
                policies[idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="grid",
                        method="solve_policies",
                        cost_weight=weight,
                        extra=self._grid_cache_extra(),
                    ),
                    data={"policy": policy},
                )

        return torch.stack(
            [policy for policy in policies if policy is not None],
            dim=0,
        ).to(device=self.device)

    def _solve_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        weight_count = int(weight_tensor.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        policy = torch.zeros(shape, device=self.device, dtype=torch.int64)
        weights = weight_tensor.view(1, 1, weight_count)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Oracle w batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_action = torch.zeros_like(policy[rem])

                for action_idx, transition in enumerate(self.transitions):
                    candidate_value = self._candidate_value_batch(
                        transition=transition,
                        rem=rem,
                        cost_weights=weights,
                        value=value,
                    )
                    better = candidate_value > best_value
                    best_value = torch.where(better, candidate_value, best_value)
                    best_action = torch.where(
                        better,
                        torch.full_like(best_action, action_idx),
                        best_action,
                    )

                value[rem] = best_value
                policy[rem] = best_action
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return policy.permute(3, 0, 1, 2).contiguous()

    def _candidate_value_batch(
        self,
        *,
        transition: TransitionCache,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        interval = transition.interval
        cont_mask = interval <= rem
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        immediate_mem = self._memorized_sum_for_days(active_days)[:, None]
        candidate_value = (
            immediate_mem.expand_as(self.s_mesh)
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                int(cost_weights.numel()),
            )
        )
        candidate_value = candidate_value.clone()

        if not cont_mask.any():
            return candidate_value

        rem_idx = future_rem[:, None].expand_as(self.s_mesh)
        rem_idx_flat = rem_idx.reshape(-1)
        cont_2d = cont_mask[:, None].expand_as(self.s_mesh)
        value_flat = value.reshape(
            value.shape[0],
            self.state_count,
            int(cost_weights.numel()),
        )
        for rating_idx, rating in enumerate(range(1, 5)):
            prob = transition.prob[rating_idx][:, None].expand_as(self.s_mesh)
            future_value = torch.zeros_like(candidate_value)
            for corner_idx in range(4):
                future_value += transition.next_weight[rating_idx][
                    corner_idx
                ].unsqueeze(2) * value_flat[
                    rem_idx_flat,
                    transition.next_idx[rating_idx][corner_idx].reshape(-1),
                ].reshape(
                    self.s_count,
                    self.d_count,
                    int(cost_weights.numel()),
                )
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_2d, prob, torch.zeros_like(prob)).unsqueeze(2)
            candidate_value += weighted * (future_value - cost_weights * review_minutes)

        return candidate_value

    def _estimate_many_batch(
        self,
        cost_weights: torch.Tensor,
        *,
        progress: bool,
    ) -> list[OracleMetrics]:
        start = time.perf_counter()
        weight_count = int(cost_weights.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        memorized = torch.zeros_like(value)
        minutes = torch.zeros_like(value)
        reviews = torch.zeros_like(value)
        lapses = torch.zeros_like(value)
        weights = cost_weights.view(1, 1, weight_count)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Oracle metrics batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_mem = torch.zeros_like(best_value)
                best_minutes = torch.zeros_like(best_value)
                best_reviews = torch.zeros_like(best_value)
                best_lapses = torch.zeros_like(best_value)

                for transition in self.transitions:
                    candidate = self._candidate_tables_batch(
                        transition=transition,
                        rem=rem,
                        cost_weights=weights,
                        value=value,
                        memorized=memorized,
                        minutes=minutes,
                        reviews=reviews,
                        lapses=lapses,
                    )
                    candidate_value, candidate_mem, candidate_minutes = candidate[:3]
                    candidate_reviews, candidate_lapses = candidate[3:]
                    better = candidate_value > best_value
                    best_value = torch.where(better, candidate_value, best_value)
                    best_mem = torch.where(better, candidate_mem, best_mem)
                    best_minutes = torch.where(
                        better,
                        candidate_minutes,
                        best_minutes,
                    )
                    best_reviews = torch.where(
                        better,
                        candidate_reviews,
                        best_reviews,
                    )
                    best_lapses = torch.where(
                        better,
                        candidate_lapses,
                        best_lapses,
                    )

                value[rem] = best_value
                memorized[rem] = best_mem
                minutes[rem] = best_minutes
                reviews[rem] = best_reviews
                lapses[rem] = best_lapses
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        total_mem = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_minutes = torch.zeros(
            weight_count,
            device=self.device,
            dtype=self.dtype,
        )
        total_reviews = torch.zeros(
            weight_count,
            device=self.device,
            dtype=self.dtype,
        )
        total_lapses = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        mem_flat = memorized.reshape(memorized.shape[0], self.state_count, weight_count)
        minutes_flat = minutes.reshape(minutes.shape[0], self.state_count, weight_count)
        reviews_flat = reviews.reshape(reviews.shape[0], self.state_count, weight_count)
        lapses_flat = lapses.reshape(lapses.shape[0], self.state_count, weight_count)
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_mem += prob * (
                state_weight[:, None] * mem_flat[self.horizon, state_idx]
            ).sum(dim=0)
            total_minutes += prob * (
                self.learning_cost_minutes[rating - 1]
                + (state_weight[:, None] * minutes_flat[self.horizon, state_idx]).sum(
                    dim=0
                )
            )
            total_reviews += prob * (
                state_weight[:, None] * reviews_flat[self.horizon, state_idx]
            ).sum(dim=0)
            total_lapses += prob * (
                state_weight[:, None] * lapses_flat[self.horizon, state_idx]
            ).sum(dim=0)

        day_count = float(self.days)
        elapsed_per_weight = (time.perf_counter() - start) / float(weight_count)
        mem_per_day = (total_mem / day_count).cpu().tolist()
        minutes_per_day = (total_minutes / day_count).cpu().tolist()
        reviews_per_day = (total_reviews / day_count).cpu().tolist()
        total_reviews_list = total_reviews.cpu().tolist()
        total_lapses_list = total_lapses.cpu().tolist()
        total_cost_seconds = (total_minutes * 60.0).cpu().tolist()
        objectives = (
            total_mem / day_count - cost_weights * (total_minutes / day_count)
        ).cpu()
        objective_list = objectives.tolist()

        metrics: list[OracleMetrics] = []
        for idx in range(weight_count):
            reviews_float = float(total_reviews_list[idx])
            lapses_float = float(total_lapses_list[idx])
            observed_retention = (
                1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
            )
            metrics.append(
                OracleMetrics(
                    card_expected_retrievability=float(mem_per_day[idx]),
                    card_minutes_per_day=float(minutes_per_day[idx]),
                    card_reviews_per_day=float(reviews_per_day[idx]),
                    card_total_reviews=reviews_float,
                    card_total_lapses=lapses_float,
                    card_total_cost_seconds=float(total_cost_seconds[idx]),
                    observed_retention=observed_retention,
                    scalar_objective=float(objective_list[idx]),
                    runtime_s=elapsed_per_weight,
                )
            )
        return metrics

    def _candidate_tables_batch(
        self,
        *,
        transition: TransitionCache,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
        memorized: torch.Tensor,
        minutes: torch.Tensor,
        reviews: torch.Tensor,
        lapses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interval = transition.interval
        weight_count = int(cost_weights.numel())
        cont_mask = interval <= rem
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        immediate_mem = self._memorized_sum_for_days(active_days)[:, None]
        candidate_value = (
            immediate_mem.expand_as(self.s_mesh)
            .unsqueeze(2)
            .expand(-1, -1, weight_count)
            .clone()
        )
        candidate_mem = candidate_value.clone()
        candidate_minutes = torch.zeros_like(candidate_value)
        candidate_reviews = torch.zeros_like(candidate_value)
        candidate_lapses = torch.zeros_like(candidate_value)

        if not cont_mask.any():
            return (
                candidate_value,
                candidate_mem,
                candidate_minutes,
                candidate_reviews,
                candidate_lapses,
            )

        rem_idx = future_rem[:, None].expand_as(self.s_mesh)
        rem_idx_flat = rem_idx.reshape(-1)
        cont_2d = cont_mask[:, None].expand_as(self.s_mesh)
        value_flat = value.reshape(value.shape[0], self.state_count, weight_count)
        memorized_flat = memorized.reshape(
            memorized.shape[0],
            self.state_count,
            weight_count,
        )
        minutes_flat = minutes.reshape(minutes.shape[0], self.state_count, weight_count)
        reviews_flat = reviews.reshape(reviews.shape[0], self.state_count, weight_count)
        lapses_flat = lapses.reshape(lapses.shape[0], self.state_count, weight_count)
        for rating_idx, rating in enumerate(range(1, 5)):
            prob = transition.prob[rating_idx][:, None].expand_as(self.s_mesh)
            future_value = torch.zeros_like(candidate_value)
            future_mem = torch.zeros_like(candidate_mem)
            future_minutes = torch.zeros_like(candidate_minutes)
            future_reviews = torch.zeros_like(candidate_reviews)
            future_lapses = torch.zeros_like(candidate_lapses)
            for corner_idx in range(4):
                next_idx = transition.next_idx[rating_idx][corner_idx].reshape(-1)
                weight = transition.next_weight[rating_idx][corner_idx].unsqueeze(2)
                future_value += weight * value_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count, weight_count)
                future_mem += weight * memorized_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count, weight_count)
                future_minutes += weight * minutes_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count, weight_count)
                future_reviews += weight * reviews_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count, weight_count)
                future_lapses += weight * lapses_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count, weight_count)
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_2d, prob, torch.zeros_like(prob)).unsqueeze(2)
            candidate_value += weighted * (future_value - cost_weights * review_minutes)
            candidate_mem += weighted * future_mem
            candidate_minutes += weighted * (future_minutes + review_minutes)
            candidate_reviews += weighted * (future_reviews + 1.0)
            candidate_lapses += weighted * (
                future_lapses + (1.0 if rating == 1 else 0.0)
            )

        return (
            candidate_value,
            candidate_mem,
            candidate_minutes,
            candidate_reviews,
            candidate_lapses,
        )

    def _candidate_tables(
        self,
        *,
        transition: TransitionCache,
        rem: int,
        cost_weight: float,
        value: torch.Tensor,
        memorized: torch.Tensor,
        minutes: torch.Tensor,
        reviews: torch.Tensor,
        lapses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interval = transition.interval
        cont_mask = interval <= rem
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        immediate_mem = self._memorized_sum_for_days(active_days)[:, None]
        candidate_value = immediate_mem.expand_as(self.s_mesh).clone()
        candidate_mem = candidate_value.clone()
        candidate_minutes = torch.zeros_like(candidate_value)
        candidate_reviews = torch.zeros_like(candidate_value)
        candidate_lapses = torch.zeros_like(candidate_value)

        if not cont_mask.any():
            return (
                candidate_value,
                candidate_mem,
                candidate_minutes,
                candidate_reviews,
                candidate_lapses,
            )

        rem_idx = future_rem[:, None].expand_as(self.s_mesh)
        rem_idx_flat = rem_idx.reshape(-1)
        cont_2d = cont_mask[:, None].expand_as(self.s_mesh)
        value_flat = value.reshape(value.shape[0], self.state_count)
        memorized_flat = memorized.reshape(memorized.shape[0], self.state_count)
        minutes_flat = minutes.reshape(minutes.shape[0], self.state_count)
        reviews_flat = reviews.reshape(reviews.shape[0], self.state_count)
        lapses_flat = lapses.reshape(lapses.shape[0], self.state_count)
        for rating_idx, rating in enumerate(range(1, 5)):
            prob = transition.prob[rating_idx][:, None].expand_as(self.s_mesh)
            future_value = torch.zeros_like(candidate_value)
            future_mem = torch.zeros_like(candidate_mem)
            future_minutes = torch.zeros_like(candidate_minutes)
            future_reviews = torch.zeros_like(candidate_reviews)
            future_lapses = torch.zeros_like(candidate_lapses)
            for corner_idx in range(4):
                next_idx = transition.next_idx[rating_idx][corner_idx].reshape(-1)
                weight = transition.next_weight[rating_idx][corner_idx]
                future_value += weight * value_flat[rem_idx_flat, next_idx].reshape(
                    self.s_count,
                    self.d_count,
                )
                future_mem += weight * memorized_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count)
                future_minutes += weight * minutes_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count)
                future_reviews += weight * reviews_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count)
                future_lapses += weight * lapses_flat[
                    rem_idx_flat,
                    next_idx,
                ].reshape(self.s_count, self.d_count)
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_2d, prob, torch.zeros_like(prob))
            candidate_value += weighted * (future_value - cost_weight * review_minutes)
            candidate_mem += weighted * future_mem
            candidate_minutes += weighted * (future_minutes + review_minutes)
            candidate_reviews += weighted * (future_reviews + 1.0)
            candidate_lapses += weighted * (
                future_lapses + (1.0 if rating == 1 else 0.0)
            )

        return (
            candidate_value,
            candidate_mem,
            candidate_minutes,
            candidate_reviews,
            candidate_lapses,
        )

    def _precompute_transitions(self) -> list[TransitionCache]:
        transitions: list[TransitionCache] = []
        for retention_factor in self.action_retention_factor:
            interval = torch.clamp(
                torch.round(self.s_grid / self.factor * retention_factor),
                min=1.0,
            ).to(torch.int64)
            elapsed = interval.to(dtype=self.dtype)
            retrievability = self._forgetting_curve(elapsed, self.s_grid)
            probs = (
                1.0 - retrievability,
                retrievability * self.review_rating_prob[0],
                retrievability * self.review_rating_prob[1],
                retrievability * self.review_rating_prob[2],
            )
            next_s_idx: list[torch.Tensor] = []
            next_d_idx: list[torch.Tensor] = []
            next_idx: list[torch.Tensor] = []
            next_weight: list[torch.Tensor] = []
            for rating in range(1, 5):
                next_s, next_d = self._next_state_grid(
                    elapsed=elapsed,
                    retrievability=retrievability,
                    rating=rating,
                )
                next_s_idx.append(self._s_to_idx(next_s))
                next_d_idx.append(self._d_to_idx(next_d))
                kernel_idx, kernel_weight = self._state_kernel(next_s, next_d)
                next_idx.append(kernel_idx)
                next_weight.append(kernel_weight)
            transitions.append(
                TransitionCache(
                    interval=interval,
                    prob=probs,
                    next_s_idx=tuple(next_s_idx),  # type: ignore[arg-type]
                    next_d_idx=tuple(next_d_idx),  # type: ignore[arg-type]
                    next_idx=tuple(next_idx),  # type: ignore[arg-type]
                    next_weight=tuple(next_weight),  # type: ignore[arg-type]
                )
            )
        return transitions

    def _next_state_grid(
        self,
        *,
        elapsed: torch.Tensor,
        retrievability: torch.Tensor,
        rating: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rating_tensor = torch.full_like(self.s_mesh, rating, dtype=torch.int64)
        s = self.s_mesh
        d = self.d_mesh
        r = retrievability[:, None].expand_as(self.s_mesh)
        if rating > 1:
            new_s = self._stability_after_success(s, r, d, rating_tensor)
        else:
            new_s = self._stability_after_failure(s, r, d)
        new_d = self._next_d(d, rating_tensor)
        return (
            torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max),
            torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max),
        )

    def _init_state_scalar(self, rating: int) -> tuple[torch.Tensor, torch.Tensor]:
        rating_f = torch.tensor(float(rating), device=self.device, dtype=self.dtype)
        s = self.weights[rating - 1]
        d = self.weights[4] - torch.exp(self.weights[5] * (rating_f - 1.0)) + 1.0
        return s, torch.clamp(d, self.bounds.d_min, self.bounds.d_max)

    def _forgetting_curve(self, elapsed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.pow(
            1.0 + self.factor * elapsed / torch.clamp(s, min=self.bounds.s_min),
            self.decay,
        )

    def _memorized_sum(self, s: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(s)
        for day_count in torch.unique(days).tolist():
            day_int = int(day_count)
            if day_int <= 0:
                continue
            idx = (days == day_int).nonzero(as_tuple=False).squeeze(1)
            times = torch.arange(
                1,
                day_int + 1,
                device=self.device,
                dtype=self.dtype,
            )
            out[idx] = self._forgetting_curve(
                times.unsqueeze(0),
                s.index_select(0, idx).unsqueeze(1),
            ).sum(dim=1)
        return out

    def _memorized_sum_for_days(self, days: torch.Tensor) -> torch.Tensor:
        return self.memorized_by_day[days.to(torch.int64), self._s_grid_idx]

    def _precompute_grid_memorized_by_day(self) -> torch.Tensor:
        table = torch.zeros(
            (self.horizon + 1, self.s_grid.numel()),
            device=self.device,
            dtype=self.dtype,
        )
        if self.horizon <= 0:
            return table
        elapsed = torch.arange(
            1,
            self.horizon + 1,
            device=self.device,
            dtype=self.dtype,
        )
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        table[1:] = torch.cumsum(retrievability, dim=0)
        return table

    def _next_d(self, d: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        rating_f = rating.to(dtype=self.dtype)
        delta_d = -self.weights[6] * (rating_f - 3.0)
        new_d = d + delta_d * (10.0 - d) / 9.0
        new_d = self.weights[7] * self.init_d + (1.0 - self.weights[7]) * new_d
        return torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)

    def _stability_after_success(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        hard_penalty = torch.where(rating == 2, self.weights[15], 1.0)
        easy_bonus = torch.where(rating == 4, self.weights[16], 1.0)
        inc = (
            torch.exp(self.weights[8])
            * (11.0 - d)
            * torch.pow(s, -self.weights[9])
            * (torch.exp((1.0 - retrievability) * self.weights[10]) - 1.0)
        )
        return s * (1.0 + inc * hard_penalty * easy_bonus)

    def _stability_after_failure(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        new_s = (
            self.weights[11]
            * torch.pow(d, -self.weights[12])
            * (torch.pow(s + 1.0, self.weights[13]) - 1.0)
            * torch.exp((1.0 - retrievability) * self.weights[14])
        )
        new_min = s / torch.exp(self.weights[17] * self.weights[18])
        return torch.minimum(new_s, new_min)

    def _state_kernel(
        self,
        s: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        s_pos = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        s_pos = torch.clamp(
            s_pos * float(self.s_count - 1), 0.0, float(self.s_count - 1)
        )
        s0 = torch.floor(s_pos).to(torch.int64)
        s1 = torch.clamp(s0 + 1, max=self.s_count - 1)
        sw = s_pos - s0.to(dtype=self.dtype)

        d_pos = torch.clamp(d, self.bounds.d_min, self.bounds.d_max)
        d_pos = (d_pos - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min)
        d_pos = torch.clamp(
            d_pos * float(self.d_count - 1), 0.0, float(self.d_count - 1)
        )
        d0 = torch.floor(d_pos).to(torch.int64)
        d1 = torch.clamp(d0 + 1, max=self.d_count - 1)
        dw = d_pos - d0.to(dtype=self.dtype)

        next_idx = torch.stack(
            (
                s0 * self.d_count + d0,
                s1 * self.d_count + d0,
                s0 * self.d_count + d1,
                s1 * self.d_count + d1,
            ),
            dim=0,
        )
        next_weight = torch.stack(
            (
                (1.0 - sw) * (1.0 - dw),
                sw * (1.0 - dw),
                (1.0 - sw) * dw,
                sw * dw,
            ),
            dim=0,
        )
        return next_idx.to(dtype=torch.int64), next_weight.to(dtype=self.dtype)

    def _initial_state_kernel(self, rating: int) -> tuple[torch.Tensor, torch.Tensor]:
        s, d = self._init_state_scalar(rating)
        return self._state_kernel(s, d)

    def _s_to_idx(self, s: torch.Tensor) -> torch.Tensor:
        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        ratio = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        return torch.clamp(
            torch.round(ratio * float(self.s_grid.numel() - 1)),
            min=0,
            max=self.s_grid.numel() - 1,
        ).to(torch.int64)

    def _d_to_idx(self, d: torch.Tensor) -> torch.Tensor:
        ratio = torch.clamp(d, self.bounds.d_min, self.bounds.d_max) - self.bounds.d_min
        ratio = ratio / (self.bounds.d_max - self.bounds.d_min)
        return torch.clamp(
            torch.round(ratio * float(self.d_grid.numel() - 1)),
            min=0,
            max=self.d_grid.numel() - 1,
        ).to(torch.int64)


class FSRS6StationaryFiniteOracle(FSRS6GridOracle):
    TRANSITION_KERNEL_VERSION = "four_corner_log_s_linear_d_v1"

    def __init__(
        self,
        *,
        days: int,
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        super().__init__(
            days=days,
            action_retentions=action_retentions,
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            dtype=dtype,
            device=device,
            fsrs_weights=fsrs_weights,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_costs=learning_costs,
            review_costs=review_costs,
            cache_config=cache_config,
        )
        self.s_count = int(self.s_grid.numel())
        self.d_count = int(self.d_grid.numel())
        self.action_count = int(self.action_retentions.numel())
        self.state_count = self.s_count * self.d_count
        self._s_grid_idx = torch.arange(self.s_count, device=self.device)
        self._flat_state_idx = torch.arange(self.state_count, device=self.device)
        self._flat_s_idx = (
            torch.arange(self.s_count, device=self.device)[:, None]
            .expand(self.s_count, self.d_count)
            .reshape(-1)
        )
        self.memorized_by_day = self._precompute_memorized_by_day()
        self._action_tables = self._precompute_stationary_action_tables()

    def solve_stationary_finite_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        metrics_by_weight: list[OracleMetrics | None] = [None for _ in cost_weights]
        objectives: list[float | None] = [None for _ in cost_weights]
        iterations: list[int | None] = [None for _ in cost_weights]
        converged: list[bool | None] = [None for _ in cost_weights]
        residuals: list[float | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="stationary_finite",
                    method="solve_stationary_finite_policies",
                    cost_weight=float(weight),
                    extra=self._stationary_cache_extra(
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    ),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device)
            metrics_by_weight[idx] = _metrics_from_payload(
                entry["metrics"],
                runtime_s=0.0,
            )
            objectives[idx] = float(entry["objective"])
            iterations[idx] = int(entry["iterations"])
            converged[idx] = bool(entry["converged"])
            residuals[idx] = float(entry["residual"])

        if missing:
            solution = self._solve_stationary_finite_policies_uncached(
                [weight for _, weight in missing],
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = solution.policy[local_idx].to(device=self.device)
                metrics = solution.metrics[local_idx]
                objective = float(solution.objectives[local_idx].item())
                iteration = solution.iterations[local_idx]
                did_converge = solution.converged[local_idx]
                residual = solution.residuals[local_idx]
                policies[idx] = policy
                metrics_by_weight[idx] = metrics
                objectives[idx] = objective
                iterations[idx] = iteration
                converged[idx] = did_converge
                residuals[idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="stationary_finite",
                        method="solve_stationary_finite_policies",
                        cost_weight=weight,
                        extra=self._stationary_cache_extra(
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                        ),
                    ),
                    data={
                        "policy": policy,
                        "metrics": _metrics_payload(metrics),
                        "objective": objective,
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )

        return StationaryFiniteOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            ).to(device=self.device, dtype=torch.int64),
            metrics=[metric for metric in metrics_by_weight if metric is not None],
            objectives=torch.tensor(
                [objective for objective in objectives if objective is not None],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[int(value) for value in iterations if value is not None],
            converged=[bool(value) for value in converged if value is not None],
            residuals=[float(value) for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def _stationary_cache_extra(
        self,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "transition_kernel": self.TRANSITION_KERNEL_VERSION,
        }
        if max_iterations is not None:
            extra["max_iterations"] = max_iterations
        if tolerance is not None:
            extra["tolerance"] = tolerance
        return extra

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        return super().solve_policies(cost_weights, progress=progress)

    def _solve_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        return super()._solve_policies_uncached(cost_weights, progress=progress)

    def _solve_stationary_finite_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        start = time.perf_counter()
        policies: list[torch.Tensor] = []
        metrics_by_weight: list[OracleMetrics] = []
        iterations: list[int] = []
        converged: list[bool] = []
        residuals: list[float] = []
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        finite_policies = self.solve_policies(cost_weights, progress=progress)
        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=len(cost_weights) * max_iterations,
                desc="Stationary finite oracle",
                unit="iter",
                leave=False,
            )
        try:
            result = self._solve_stationary_finite_policy_batch(
                cost_weights=weight_tensor,
                finite_policies=finite_policies,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress_bar=progress_bar,
            )
            (
                batched_policy,
                batch_metrics,
                batch_iterations,
                batch_converged,
                batch_residuals,
            ) = result
            policies.extend(policy for policy in batched_policy)
            metrics_by_weight.extend(batch_metrics)
            iterations.extend(batch_iterations)
            converged.extend(batch_converged)
            residuals.extend(batch_residuals)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return StationaryFiniteOracleSolution(
            policy=torch.stack(policies, dim=0).to(dtype=torch.int64),
            metrics=metrics_by_weight,
            objectives=torch.tensor(
                [metric.scalar_objective for metric in metrics_by_weight],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=iterations,
            converged=converged,
            residuals=residuals,
            runtime_s=time.perf_counter() - start,
        )

    def _solve_stationary_finite_policy_batch(
        self,
        *,
        cost_weights: torch.Tensor,
        finite_policies: torch.Tensor,
        max_iterations: int,
        tolerance: float,
        progress_bar: Any,
    ) -> tuple[
        torch.Tensor,
        list[OracleMetrics],
        list[int],
        list[bool],
        list[float],
    ]:
        policy = self._project_stationary_policies(finite_policies)
        weight_count = int(cost_weights.numel())
        value = self._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=cost_weights,
        )
        objective = self._objective_from_value_batch(
            value=value,
            cost_weights=cost_weights,
        )
        iterations = torch.zeros(
            weight_count,
            device=self.device,
            dtype=torch.int64,
        )
        converged = torch.zeros(weight_count, device=self.device, dtype=torch.bool)
        residuals = torch.full(
            (weight_count,),
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones(weight_count, device=self.device, dtype=torch.bool)

        for iteration in range(1, max_iterations + 1):
            active_count = int(active.sum().item())
            if active_count == 0:
                break
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            active_policy = policy.index_select(0, active_idx)
            active_weights = cost_weights.index_select(0, active_idx)
            active_value = value.index_select(0, active_idx)
            occupancy = self._rollout_occupancy_batch(
                policy=active_policy,
                stationary=True,
            )
            new_policy, residual, visited = self._improve_stationary_policy_batch(
                policy=active_policy,
                occupancy=occupancy,
                value=active_value,
                cost_weights=active_weights,
            )
            changed = ((new_policy != active_policy) & visited).reshape(
                active_count,
                self.state_count,
            )
            policy_changed = changed.any(dim=1)
            residuals[active_idx] = residual
            iterations[active_idx] = iteration
            if progress_bar is not None:
                progress_bar.update(active_count)

            done = (~policy_changed) | (residual <= tolerance)
            if bool(done.any().item()):
                done_idx = active_idx[done]
                converged[done_idx] = True
                active[done_idx] = False

            candidate = ~done
            if not bool(candidate.any().item()):
                continue

            candidate_idx = active_idx[candidate]
            candidate_policy = new_policy[candidate]
            candidate_weights = cost_weights.index_select(0, candidate_idx)
            candidate_value = self._evaluate_stationary_policy_value_batch(
                policy=candidate_policy,
                cost_weights=candidate_weights,
            )
            candidate_objective = self._objective_from_value_batch(
                value=candidate_value,
                cost_weights=candidate_weights,
            )
            objective_improvement = candidate_objective - objective[candidate_idx]
            accepted = objective_improvement > tolerance
            residuals[candidate_idx] = torch.where(
                accepted,
                objective_improvement,
                torch.clamp(objective_improvement, min=0.0),
            )

            if bool((~accepted).any().item()):
                rejected_idx = candidate_idx[~accepted]
                converged[rejected_idx] = True
                active[rejected_idx] = False

            if bool(accepted.any().item()):
                accepted_idx = candidate_idx[accepted]
                policy[accepted_idx] = candidate_policy[accepted]
                value[accepted_idx] = candidate_value[accepted]
                objective[accepted_idx] = candidate_objective[accepted]

        if bool(active.any().item()):
            iterations[active] = max_iterations

        metrics = self._metrics_from_occupancy_batch(
            policy=policy,
            cost_weights=cost_weights,
        )
        return (
            policy,
            metrics,
            [int(value) for value in iterations.cpu().tolist()],
            [bool(value) for value in converged.cpu().tolist()],
            [float(value) for value in residuals.cpu().tolist()],
        )

    def labels(
        self,
        *,
        policies: torch.Tensor,
        cost_weights: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
        goal_weight: torch.Tensor,
    ) -> torch.Tensor:
        s_idx = self._s_to_idx(s)
        d_idx = self._d_to_idx(d)
        goal_idx = torch.argmin(
            torch.abs(
                goal_weight.to(dtype=cost_weights.dtype)[:, None]
                - cost_weights[None, :]
            ),
            dim=1,
        )
        return policies.to(device=s.device)[goal_idx, s_idx, d_idx]

    def _state_kernel(
        self,
        s: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())

        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        s_pos = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        s_pos = torch.clamp(s_pos * float(s_count - 1), 0.0, float(s_count - 1))
        s0 = torch.floor(s_pos).to(torch.int64)
        s1 = torch.clamp(s0 + 1, max=s_count - 1)
        sw = s_pos - s0.to(dtype=self.dtype)

        d_pos = torch.clamp(d, self.bounds.d_min, self.bounds.d_max)
        d_pos = (d_pos - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min)
        d_pos = torch.clamp(d_pos * float(d_count - 1), 0.0, float(d_count - 1))
        d0 = torch.floor(d_pos).to(torch.int64)
        d1 = torch.clamp(d0 + 1, max=d_count - 1)
        dw = d_pos - d0.to(dtype=self.dtype)

        next_idx = torch.stack(
            (
                s0 * self.d_count + d0,
                s1 * self.d_count + d0,
                s0 * self.d_count + d1,
                s1 * self.d_count + d1,
            ),
            dim=0,
        )
        next_weight = torch.stack(
            (
                (1.0 - sw) * (1.0 - dw),
                sw * (1.0 - dw),
                (1.0 - sw) * dw,
                sw * dw,
            ),
            dim=0,
        )
        return next_idx.to(dtype=torch.int64), next_weight.to(dtype=self.dtype)

    def _initial_state_kernel(self, rating: int) -> tuple[torch.Tensor, torch.Tensor]:
        s0, d0 = self._init_state_scalar(rating)
        return self._state_kernel(s0, d0)

    def _solve_single_stationary_finite_policy(
        self,
        *,
        cost_weight: float,
        finite_policy: torch.Tensor,
        max_iterations: int,
        tolerance: float,
        progress_bar: Any,
    ) -> tuple[torch.Tensor, OracleMetrics, int, bool, float]:
        policy = self._project_stationary_policy(finite_policy)

        value: torch.Tensor | None = None
        objective: float | None = None
        residual = math.inf
        converged = False
        for iteration in range(1, max_iterations + 1):
            if value is None or objective is None:
                value = self._evaluate_stationary_policy_value(
                    policy=policy,
                    cost_weight=cost_weight,
                )
                objective = self._objective_from_value(
                    value=value,
                    cost_weight=cost_weight,
                )
            occupancy = self._rollout_occupancy(policy=policy, stationary=True)
            new_policy, residual, visited = self._improve_stationary_policy(
                policy=policy,
                occupancy=occupancy,
                value=value,
                cost_weight=cost_weight,
            )
            policy_changed = bool(((new_policy != policy) & visited).any().item())
            if progress_bar is not None:
                progress_bar.update(1)
            if not policy_changed or residual <= tolerance:
                converged = True
                _, metrics = self._evaluate_stationary_policy(
                    policy=policy,
                    cost_weight=cost_weight,
                )
                return policy, metrics, iteration, converged, residual
            # The occupancy-weighted greedy step is a heuristic under the stationary
            # finite-lifecycle constraint, so accept only objective-improving moves.
            new_value = self._evaluate_stationary_policy_value(
                policy=new_policy,
                cost_weight=cost_weight,
            )
            new_objective = self._objective_from_value(
                value=new_value,
                cost_weight=cost_weight,
            )
            objective_improvement = new_objective - objective
            if objective_improvement <= tolerance:
                converged = True
                _, metrics = self._evaluate_stationary_policy(
                    policy=policy,
                    cost_weight=cost_weight,
                )
                return (
                    policy,
                    metrics,
                    iteration,
                    converged,
                    max(0.0, objective_improvement),
                )
            residual = objective_improvement
            policy = new_policy
            value = new_value
            objective = new_objective

        _, metrics = self._evaluate_stationary_policy(
            policy=policy,
            cost_weight=cost_weight,
        )
        return policy, metrics, max_iterations, converged, residual

    def _project_stationary_policy(self, finite_policy: torch.Tensor) -> torch.Tensor:
        occupancy = self._rollout_occupancy(policy=finite_policy, stationary=False)
        action_weight = torch.zeros(
            (self.action_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        occupancy_flat = occupancy[1:].reshape(self.horizon, self.state_count)
        policy_flat = finite_policy[1:].reshape(self.horizon, self.state_count)
        scatter_idx = (
            policy_flat * self.state_count
            + self._flat_state_idx[None, :].expand(self.horizon, self.state_count)
        ).reshape(-1)
        action_weight.reshape(-1).scatter_add_(
            0,
            scatter_idx,
            occupancy_flat.reshape(-1),
        )

        projected = finite_policy[self.horizon].reshape(self.state_count).clone()
        visited = action_weight.sum(dim=0) > 0.0
        projected[visited] = action_weight.argmax(dim=0)[visited]
        return projected.reshape(self.s_count, self.d_count).to(dtype=torch.int64)

    def _project_stationary_policies(
        self,
        finite_policies: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(finite_policies.shape[0])
        occupancy = self._rollout_occupancy_batch(
            policy=finite_policies,
            stationary=False,
        )
        action_weight = torch.zeros(
            (weight_count, self.action_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        occupancy_flat = occupancy[:, 1:, :]
        policy_flat = finite_policies[:, 1:].reshape(
            weight_count,
            self.horizon,
            self.state_count,
        )
        batch_offsets = (
            torch.arange(weight_count, device=self.device, dtype=torch.int64)
            .view(weight_count, 1, 1)
            .mul(self.action_count * self.state_count)
        )
        state_idx = self._flat_state_idx.view(1, 1, self.state_count)
        scatter_idx = batch_offsets + policy_flat * self.state_count + state_idx
        action_weight.reshape(-1).scatter_add_(
            0,
            scatter_idx.reshape(-1),
            occupancy_flat.reshape(-1),
        )

        projected = finite_policies[:, self.horizon].reshape(
            weight_count,
            self.state_count,
        )
        action_count = action_weight.sum(dim=1)
        visited = action_count > 0.0
        best_action = action_weight.argmax(dim=1)
        projected = torch.where(visited, best_action, projected).to(dtype=torch.int64)
        return projected.reshape(weight_count, self.s_count, self.d_count)

    def _evaluate_stationary_policy(
        self,
        *,
        policy: torch.Tensor,
        cost_weight: float,
    ) -> tuple[torch.Tensor, OracleMetrics]:
        shape = (self.horizon + 1, self.state_count)
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        memorized = torch.zeros_like(value)
        minutes = torch.zeros_like(value)
        reviews = torch.zeros_like(value)
        lapses = torch.zeros_like(value)
        interval, prob, next_idx, next_weight = self._select_stationary_policy_tables(
            policy
        )

        for rem in range(1, self.horizon + 1):
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            value_rem = self._memorized_sum_flat(active_days)
            mem_rem = value_rem.clone()
            minutes_rem = torch.zeros_like(value_rem)
            reviews_rem = torch.zeros_like(value_rem)
            lapses_rem = torch.zeros_like(value_rem)

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(value_rem)
                    future_mem = torch.zeros_like(mem_rem)
                    future_minutes = torch.zeros_like(minutes_rem)
                    future_reviews = torch.zeros_like(reviews_rem)
                    future_lapses = torch.zeros_like(lapses_rem)
                    for corner_idx in range(4):
                        weight = next_weight[rating_idx, corner_idx]
                        target = next_idx[rating_idx, corner_idx]
                        future_value += weight * value[future_rem, target]
                        future_mem += weight * memorized[future_rem, target]
                        future_minutes += weight * minutes[future_rem, target]
                        future_reviews += weight * reviews[future_rem, target]
                        future_lapses += weight * lapses[future_rem, target]
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[rating_idx],
                        torch.zeros_like(prob[rating_idx]),
                    )
                    value_rem += weighted * (
                        future_value - cost_weight * review_minutes
                    )
                    mem_rem += weighted * future_mem
                    minutes_rem += weighted * (future_minutes + review_minutes)
                    reviews_rem += weighted * (future_reviews + 1.0)
                    lapses_rem += weighted * (
                        future_lapses + (1.0 if rating == 1 else 0.0)
                    )

            value[rem] = value_rem
            memorized[rem] = mem_rem
            minutes[rem] = minutes_rem
            reviews[rem] = reviews_rem
            lapses[rem] = lapses_rem

        metrics = self._metrics_from_tables(
            cost_weight=cost_weight,
            memorized=memorized,
            minutes=minutes,
            reviews=reviews,
            lapses=lapses,
        )
        return value, metrics

    def _evaluate_stationary_policy_value(
        self,
        *,
        policy: torch.Tensor,
        cost_weight: float,
    ) -> torch.Tensor:
        value = torch.zeros(
            (self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        interval, prob, next_idx, next_weight = self._select_stationary_policy_tables(
            policy
        )

        for rem in range(1, self.horizon + 1):
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            value_rem = self._memorized_sum_flat(active_days)

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[
                                rating_idx,
                                corner_idx,
                            ]
                            * value[
                                future_rem,
                                next_idx[rating_idx, corner_idx],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[rating_idx],
                        torch.zeros_like(prob[rating_idx]),
                    )
                    value_rem += weighted * (
                        future_value - cost_weight * review_minutes
                    )

            value[rem] = value_rem

        return value

    def _evaluate_stationary_policy_value_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        value = torch.zeros(
            (weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        interval, prob, next_idx, next_weight = (
            self._select_stationary_policy_tables_batch(policy)
        )
        batch_idx = torch.arange(weight_count, device=self.device)[:, None]
        weight_penalty = cost_weights.to(dtype=self.dtype)[:, None]

        for rem in range(1, self.horizon + 1):
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            value_rem = self.memorized_by_day[
                active_days,
                self._flat_s_idx[None, :].expand_as(active_days),
            ].clone()

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ]
                            * value[
                                batch_idx,
                                future_rem,
                                next_idx[:, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[:, rating_idx, :],
                        torch.zeros_like(prob[:, rating_idx, :]),
                    )
                    value_rem += weighted * (
                        future_value - weight_penalty * review_minutes
                    )

            value[:, rem] = value_rem

        return value

    def _objective_from_value(
        self,
        *,
        value: torch.Tensor,
        cost_weight: float,
    ) -> float:
        total_value = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_learning_minutes = torch.tensor(
            0.0,
            device=self.device,
            dtype=self.dtype,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_value += prob * (state_weight * value[self.horizon, state_idx]).sum()
            total_learning_minutes += prob * self.learning_cost_minutes[rating - 1]
        return float(
            (
                total_value
                - torch.as_tensor(
                    cost_weight,
                    device=self.device,
                    dtype=self.dtype,
                )
                * total_learning_minutes
            ).item()
            / float(self.days)
        )

    def _objective_from_value_batch(
        self,
        *,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        total_value = torch.zeros(
            int(cost_weights.numel()),
            device=self.device,
            dtype=self.dtype,
        )
        total_learning_minutes = torch.tensor(
            0.0,
            device=self.device,
            dtype=self.dtype,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_value += prob * (
                state_weight[None, :] * value[:, self.horizon, state_idx]
            ).sum(dim=1)
            total_learning_minutes += prob * self.learning_cost_minutes[rating - 1]
        return (total_value - cost_weights * total_learning_minutes) / float(self.days)

    def _stationary_candidate_tables(
        self,
        *,
        transition: TransitionCache,
        rem: int,
        cost_weight: float,
        value: torch.Tensor,
        memorized: torch.Tensor,
        minutes: torch.Tensor,
        reviews: torch.Tensor,
        lapses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interval = transition.interval
        cont_mask = interval <= rem
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        immediate_mem = self._memorized_sum_for_days(active_days)[:, None]
        candidate_value = immediate_mem.expand_as(self.s_mesh).clone()
        candidate_mem = candidate_value.clone()
        candidate_minutes = torch.zeros_like(candidate_value)
        candidate_reviews = torch.zeros_like(candidate_value)
        candidate_lapses = torch.zeros_like(candidate_value)

        if not cont_mask.any():
            return (
                candidate_value,
                candidate_mem,
                candidate_minutes,
                candidate_reviews,
                candidate_lapses,
            )

        rem_idx = future_rem[:, None].expand_as(self.s_mesh)
        cont_2d = cont_mask[:, None].expand_as(self.s_mesh)
        for rating_idx, rating in enumerate(range(1, 5)):
            prob = transition.prob[rating_idx][:, None].expand_as(self.s_mesh)
            s_idx = transition.next_s_idx[rating_idx]
            d_idx = transition.next_d_idx[rating_idx]
            future_value = value[rem_idx, s_idx, d_idx]
            future_mem = memorized[rem_idx, s_idx, d_idx]
            future_minutes = minutes[rem_idx, s_idx, d_idx]
            future_reviews = reviews[rem_idx, s_idx, d_idx]
            future_lapses = lapses[rem_idx, s_idx, d_idx]
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_2d, prob, torch.zeros_like(prob))
            candidate_value += weighted * (future_value - cost_weight * review_minutes)
            candidate_mem += weighted * future_mem
            candidate_minutes += weighted * (future_minutes + review_minutes)
            candidate_reviews += weighted * (future_reviews + 1.0)
            candidate_lapses += weighted * (
                future_lapses + (1.0 if rating == 1 else 0.0)
            )

        return (
            candidate_value,
            candidate_mem,
            candidate_minutes,
            candidate_reviews,
            candidate_lapses,
        )

    def _stationary_candidate_value(
        self,
        *,
        transition: TransitionCache,
        rem: int,
        cost_weight: float,
        value: torch.Tensor,
    ) -> torch.Tensor:
        interval = transition.interval[:, None].expand_as(self.s_mesh).reshape(-1)
        cont_mask = interval <= rem
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        candidate_value = self._memorized_sum_flat(active_days)

        if not cont_mask.any():
            return candidate_value.reshape(self.s_count, self.d_count)

        for rating_idx, rating in enumerate(range(1, 5)):
            prob = (
                transition.prob[rating_idx][:, None].expand_as(self.s_mesh).reshape(-1)
            )
            next_idx = (
                transition.next_s_idx[rating_idx] * self.d_count
                + transition.next_d_idx[rating_idx]
            ).reshape(-1)
            future_value = value[future_rem, next_idx]
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_mask, prob, torch.zeros_like(prob))
            candidate_value += weighted * (future_value - cost_weight * review_minutes)

        return candidate_value.reshape(self.s_count, self.d_count)

    def _improve_stationary_policy(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weight: float,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        action_scores = torch.zeros(
            (self.action_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        state_occupancy = occupancy[1:].sum(dim=0).reshape(-1)
        visited = state_occupancy > 0.0
        tables = self._action_tables
        interval = tables.interval
        prob = tables.prob
        next_idx = tables.next_idx
        next_weight = tables.next_weight

        for rem in range(1, self.horizon + 1):
            rem_occupancy = occupancy[rem].reshape(-1)
            if float(rem_occupancy.sum().item()) <= 0.0:
                continue
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            s_idx = self._flat_s_idx[None, :].expand_as(active_days)
            candidate_value = self.memorized_by_day[active_days, s_idx].clone()

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(candidate_value)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ]
                            * value[
                                future_rem,
                                next_idx[:, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[:, rating_idx, :],
                        torch.zeros_like(prob[:, rating_idx, :]),
                    )
                    candidate_value += weighted * (
                        future_value - cost_weight * review_minutes
                    )
            action_scores += rem_occupancy[None, :] * candidate_value

        best_score, best_action = action_scores.max(dim=0)
        policy_flat = policy.reshape(-1)
        current_score = action_scores.gather(0, policy_flat[None, :]).squeeze(0)
        improvement = best_score - current_score
        residual = (
            float(improvement[visited].max().item())
            if bool(visited.any().item())
            else 0.0
        )
        new_policy = torch.where(visited, best_action, policy_flat).to(
            dtype=torch.int64
        )
        return (
            new_policy.reshape(self.s_count, self.d_count),
            residual,
            visited.reshape(self.s_count, self.d_count),
        )

    def _improve_stationary_policy_batch(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[0])
        action_scores = torch.zeros(
            (weight_count, self.action_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        state_occupancy = occupancy[:, 1:, :].sum(dim=1)
        visited = state_occupancy > 0.0
        tables = self._action_tables
        interval = tables.interval
        prob = tables.prob
        next_idx = tables.next_idx
        next_weight = tables.next_weight
        batch_idx = torch.arange(weight_count, device=self.device)[:, None, None]
        weight_penalty = cost_weights.to(dtype=self.dtype).view(weight_count, 1, 1)

        for rem in range(1, self.horizon + 1):
            rem_occupancy = occupancy[:, rem, :]
            if float(rem_occupancy.sum().item()) <= 0.0:
                continue
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            s_idx = self._flat_s_idx[None, :].expand_as(active_days)
            candidate_value = self.memorized_by_day[active_days, s_idx][
                None,
                :,
                :,
            ].expand(weight_count, -1, -1)
            candidate_value = candidate_value.clone()

            if cont_mask.any():
                expanded_future_rem = future_rem[None, :, :].expand(
                    weight_count,
                    -1,
                    -1,
                )
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(candidate_value)
                    for corner_idx in range(4):
                        expanded_next_idx = next_idx[:, rating_idx, corner_idx, :][
                            None,
                            :,
                            :,
                        ].expand(weight_count, -1, -1)
                        future_value += (
                            next_weight[
                                None,
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ]
                            * value[
                                batch_idx,
                                expanded_future_rem,
                                expanded_next_idx,
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[:, rating_idx, :],
                        torch.zeros_like(prob[:, rating_idx, :]),
                    )[None, :, :]
                    candidate_value += weighted * (
                        future_value - weight_penalty * review_minutes
                    )
            action_scores += rem_occupancy[:, None, :] * candidate_value

        best_score, best_action = action_scores.max(dim=1)
        policy_flat = policy.reshape(weight_count, self.state_count)
        current_score = action_scores.gather(1, policy_flat[:, None, :]).squeeze(1)
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = masked_improvement.max(dim=1).values
        residual = torch.where(
            visited.any(dim=1),
            residual,
            torch.zeros_like(residual),
        )
        new_policy = torch.where(visited, best_action, policy_flat).to(
            dtype=torch.int64
        )
        return (
            new_policy.reshape(weight_count, self.s_count, self.d_count),
            residual,
            visited.reshape(weight_count, self.s_count, self.d_count),
        )

    def _rollout_occupancy(
        self, *, policy: torch.Tensor, stationary: bool
    ) -> torch.Tensor:
        occupancy = torch.zeros(
            (self.horizon + 1, self.s_count, self.d_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_occupancy = occupancy.reshape(-1)
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            flat_occupancy.scatter_add_(
                0,
                self.horizon * self.state_count + state_idx,
                prob * state_weight,
            )
        selected_interval: torch.Tensor | None = None
        selected_prob: torch.Tensor | None = None
        selected_next_idx: torch.Tensor | None = None
        selected_next_weight: torch.Tensor | None = None
        if stationary:
            (
                selected_interval,
                selected_prob,
                selected_next_idx,
                selected_next_weight,
            ) = self._select_stationary_policy_tables(policy)

        for rem in range(self.horizon, 0, -1):
            current = occupancy[rem].reshape(-1)
            if float(current.sum().item()) <= 0.0:
                continue
            if not stationary:
                (
                    selected_interval,
                    selected_prob,
                    selected_next_idx,
                    selected_next_weight,
                ) = self._select_stationary_policy_tables(policy[rem])
            if (
                selected_interval is None
                or selected_prob is None
                or selected_next_idx is None
                or selected_next_weight is None
            ):
                raise RuntimeError("selected policy tables were not initialized.")

            cont_mask = selected_interval <= rem
            if not bool(cont_mask.any().item()):
                continue
            source = current * cont_mask.to(dtype=self.dtype)
            if float(source.sum().item()) <= 0.0:
                continue

            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            rem_offset = future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * selected_prob[rating_idx]
                        * selected_next_weight[rating_idx, corner_idx]
                    )
                    target = rem_offset + selected_next_idx[rating_idx, corner_idx]
                    flat_occupancy.scatter_add_(0, target, amount)

        return occupancy

    def _rollout_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        stationary: bool,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        occupancy = torch.zeros(
            (weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_occupancy = occupancy.reshape(-1)
        batch_offsets = (
            torch.arange(weight_count, device=self.device, dtype=torch.int64)
            .view(weight_count, 1)
            .mul((self.horizon + 1) * self.state_count)
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            for corner_idx in range(4):
                target = (
                    batch_offsets
                    + self.horizon * self.state_count
                    + state_idx[corner_idx]
                )
                amount = (prob * state_weight[corner_idx]).expand(weight_count)
                flat_occupancy.scatter_add_(0, target.reshape(-1), amount)
        selected_interval: torch.Tensor | None = None
        selected_prob: torch.Tensor | None = None
        selected_next_idx: torch.Tensor | None = None
        selected_next_weight: torch.Tensor | None = None
        if stationary:
            (
                selected_interval,
                selected_prob,
                selected_next_idx,
                selected_next_weight,
            ) = self._select_stationary_policy_tables_batch(policy)

        for rem in range(self.horizon, 0, -1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            if not stationary:
                (
                    selected_interval,
                    selected_prob,
                    selected_next_idx,
                    selected_next_weight,
                ) = self._select_stationary_policy_tables_batch(policy[:, rem])
            if (
                selected_interval is None
                or selected_prob is None
                or selected_next_idx is None
                or selected_next_weight is None
            ):
                raise RuntimeError("selected policy tables were not initialized.")

            cont_mask = selected_interval <= rem
            if not bool(cont_mask.any().item()):
                continue
            source = current * cont_mask.to(dtype=self.dtype)
            if float(source.sum().item()) <= 0.0:
                continue

            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            target_base = batch_offsets + future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * selected_prob[:, rating_idx, :]
                        * selected_next_weight[:, rating_idx, corner_idx, :]
                    )
                    target = (
                        target_base + selected_next_idx[:, rating_idx, corner_idx, :]
                    )
                    flat_occupancy.scatter_add_(
                        0,
                        target.reshape(-1),
                        amount.reshape(-1),
                    )

        return occupancy

    def _metrics_from_tables(
        self,
        *,
        cost_weight: float,
        memorized: torch.Tensor,
        minutes: torch.Tensor,
        reviews: torch.Tensor,
        lapses: torch.Tensor,
    ) -> OracleMetrics:
        total_mem = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_minutes = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_reviews = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_lapses = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_mem += (
                prob * (state_weight * memorized[self.horizon, state_idx]).sum()
            )
            total_minutes += prob * (
                self.learning_cost_minutes[rating - 1]
                + (state_weight * minutes[self.horizon, state_idx]).sum()
            )
            total_reviews += (
                prob * (state_weight * reviews[self.horizon, state_idx]).sum()
            )
            total_lapses += (
                prob * (state_weight * lapses[self.horizon, state_idx]).sum()
            )

        day_count = float(self.days)
        reviews_float = float(total_reviews.item())
        lapses_float = float(total_lapses.item())
        observed_retention = (
            1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
        )
        mem_per_day = float(total_mem.item() / day_count)
        minutes_per_day = float(total_minutes.item() / day_count)
        return OracleMetrics(
            card_expected_retrievability=mem_per_day,
            card_minutes_per_day=minutes_per_day,
            card_reviews_per_day=reviews_float / day_count,
            card_total_reviews=reviews_float,
            card_total_lapses=lapses_float,
            card_total_cost_seconds=float(total_minutes.item() * 60.0),
            observed_retention=observed_retention,
            scalar_objective=mem_per_day - cost_weight * minutes_per_day,
            runtime_s=0.0,
        )

    def _metrics_from_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> list[OracleMetrics]:
        weight_count = int(policy.shape[0])
        occupancy = self._rollout_occupancy_batch(policy=policy, stationary=True)
        interval, prob, _, _ = self._select_stationary_policy_tables_batch(policy)
        total_mem = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_minutes = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_reviews = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_lapses = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        learning_minutes = (self.first_rating_prob * self.learning_cost_minutes).sum()
        total_minutes += learning_minutes
        expected_review_minutes = (prob * self.review_cost_minutes.view(1, 4, 1)).sum(
            dim=1
        )

        for rem in range(1, self.horizon + 1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            immediate_mem = self.memorized_by_day[
                active_days,
                self._flat_s_idx[None, :].expand_as(active_days),
            ]
            total_mem += (current * immediate_mem).sum(dim=1)

            cont_mask = interval <= rem
            source = current * cont_mask.to(dtype=self.dtype)
            if float(source.sum().item()) <= 0.0:
                continue
            total_minutes += (source * expected_review_minutes).sum(dim=1)
            total_reviews += source.sum(dim=1)
            total_lapses += (source * prob[:, 0, :]).sum(dim=1)

        day_count = float(self.days)
        mem_per_day = (total_mem / day_count).cpu().tolist()
        minutes_per_day = (total_minutes / day_count).cpu().tolist()
        reviews_per_day = (total_reviews / day_count).cpu().tolist()
        total_reviews_list = total_reviews.cpu().tolist()
        total_lapses_list = total_lapses.cpu().tolist()
        total_cost_seconds = (total_minutes * 60.0).cpu().tolist()
        objectives = (
            total_mem / day_count - cost_weights * (total_minutes / day_count)
        ).cpu()
        objective_list = objectives.tolist()

        metrics: list[OracleMetrics] = []
        for idx in range(weight_count):
            reviews_float = float(total_reviews_list[idx])
            lapses_float = float(total_lapses_list[idx])
            observed_retention = (
                1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
            )
            metrics.append(
                OracleMetrics(
                    card_expected_retrievability=float(mem_per_day[idx]),
                    card_minutes_per_day=float(minutes_per_day[idx]),
                    card_reviews_per_day=float(reviews_per_day[idx]),
                    card_total_reviews=reviews_float,
                    card_total_lapses=lapses_float,
                    card_total_cost_seconds=float(total_cost_seconds[idx]),
                    observed_retention=observed_retention,
                    scalar_objective=float(objective_list[idx]),
                    runtime_s=0.0,
                )
            )
        return metrics

    def _zero_metrics(self, cost_weight: float) -> OracleMetrics:
        return OracleMetrics(
            card_expected_retrievability=0.0,
            card_minutes_per_day=0.0,
            card_reviews_per_day=0.0,
            card_total_reviews=0.0,
            card_total_lapses=0.0,
            card_total_cost_seconds=0.0,
            observed_retention=None,
            scalar_objective=0.0 - cost_weight * 0.0,
            runtime_s=0.0,
        )

    def _memorized_sum_for_days(self, days: torch.Tensor) -> torch.Tensor:
        return self.memorized_by_day[days.to(torch.int64), self._s_grid_idx]

    def _memorized_sum_flat(self, days: torch.Tensor) -> torch.Tensor:
        return self.memorized_by_day[days.to(torch.int64), self._flat_s_idx]

    def _precompute_stationary_action_tables(self) -> StationaryActionKernelTables:
        intervals: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        next_indices: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        for transition in self.transitions:
            intervals.append(
                transition.interval[:, None].expand_as(self.s_mesh).reshape(-1)
            )
            probs.append(
                torch.stack(
                    [
                        prob[:, None].expand_as(self.s_mesh).reshape(-1)
                        for prob in transition.prob
                    ],
                    dim=0,
                )
            )
            next_indices.append(
                torch.stack(
                    [
                        kernel_idx.reshape(4, self.state_count)
                        for kernel_idx in transition.next_idx
                    ],
                    dim=0,
                ).to(dtype=torch.int64)
            )
            next_weights.append(
                torch.stack(
                    [
                        kernel_weight.reshape(4, self.state_count)
                        for kernel_weight in transition.next_weight
                    ],
                    dim=0,
                ).to(dtype=self.dtype)
            )
        return StationaryActionKernelTables(
            interval=torch.stack(intervals, dim=0).to(dtype=torch.int64),
            prob=torch.stack(probs, dim=0),
            next_idx=torch.stack(next_indices, dim=0).to(dtype=torch.int64),
            next_weight=torch.stack(next_weights, dim=0).to(dtype=self.dtype),
        )

    def _select_stationary_policy_tables(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tables = self._action_tables
        policy_flat = policy.reshape(-1).to(dtype=torch.int64)
        gather_idx = policy_flat[None, :]
        selected_interval = tables.interval.gather(0, gather_idx).squeeze(0)
        table_idx = policy_flat.view(1, 1, -1)
        selected_prob = tables.prob.gather(0, table_idx.expand(1, 4, -1)).squeeze(0)
        kernel_idx = policy_flat.view(1, 1, 1, -1).expand(1, 4, 4, -1)
        selected_next_idx = tables.next_idx.gather(0, kernel_idx).squeeze(0)
        selected_next_weight = tables.next_weight.gather(0, kernel_idx).squeeze(0)
        return selected_interval, selected_prob, selected_next_idx, selected_next_weight

    def _select_stationary_policy_tables_batch(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tables = self._action_tables
        weight_count = int(policy.shape[0])
        policy_flat = policy.reshape(weight_count, self.state_count).to(
            dtype=torch.int64
        )
        selected_interval = tables.interval.gather(0, policy_flat)
        table_idx = policy_flat.view(weight_count, 1, self.state_count)
        selected_prob = tables.prob.gather(
            0,
            table_idx.expand(weight_count, 4, self.state_count),
        )
        kernel_idx = policy_flat.view(weight_count, 1, 1, self.state_count).expand(
            weight_count,
            4,
            4,
            self.state_count,
        )
        selected_next_idx = tables.next_idx.gather(
            0,
            kernel_idx,
        )
        selected_next_weight = tables.next_weight.gather(
            0,
            kernel_idx,
        )
        return selected_interval, selected_prob, selected_next_idx, selected_next_weight

    def _precompute_memorized_by_day(self) -> torch.Tensor:
        table = torch.zeros(
            (self.horizon + 1, self.s_count),
            device=self.device,
            dtype=self.dtype,
        )
        if self.horizon <= 0:
            return table
        elapsed = torch.arange(
            1,
            self.horizon + 1,
            device=self.device,
            dtype=self.dtype,
        )
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        table[1:] = torch.cumsum(retrievability, dim=0)
        return table


class FSRS6BatchedStationaryFiniteOracle:
    TRANSITION_KERNEL_VERSION = "four_corner_log_s_linear_d_v1"

    def __init__(
        self,
        *,
        days: int,
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        fsrs_weights: Sequence[Sequence[float]],
        first_rating_prob: Sequence[Sequence[float]],
        review_rating_prob: Sequence[Sequence[float]],
        learning_costs: Sequence[Sequence[float]],
        review_costs: Sequence[Sequence[float]],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        if days <= 1:
            raise ValueError("days must be > 1.")
        if s_grid_size < 8 or d_grid_size < 8:
            raise ValueError("grid sizes must be >= 8.")
        validate_retention_values_for_model(
            action_retentions,
            name="action retention",
        )

        weights = torch.tensor(fsrs_weights, device=device, dtype=dtype)
        if weights.ndim != 2 or weights.shape[1] != 21:
            raise ValueError("fsrs_weights must have shape [user_count, 21].")
        self.user_count = int(weights.shape[0])
        if self.user_count <= 0:
            raise ValueError("fsrs_weights must contain at least one user.")

        first_prob = torch.tensor(first_rating_prob, device=device, dtype=dtype)
        review_prob = torch.tensor(review_rating_prob, device=device, dtype=dtype)
        learning_costs_t = torch.tensor(learning_costs, device=device, dtype=dtype)
        review_costs_t = torch.tensor(review_costs, device=device, dtype=dtype)
        if first_prob.shape != (self.user_count, 4):
            raise ValueError("first_rating_prob must have shape [user_count, 4].")
        if review_prob.shape != (self.user_count, 3):
            raise ValueError("review_rating_prob must have shape [user_count, 3].")
        if learning_costs_t.shape != (self.user_count, 4):
            raise ValueError("learning_costs must have shape [user_count, 4].")
        if review_costs_t.shape != (self.user_count, 4):
            raise ValueError("review_costs must have shape [user_count, 4].")

        self.days = int(days)
        self.horizon = int(days - 1)
        self.dtype = dtype
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.cache_config = resolve_oracle_dp_cache_config(cache_config)
        self.bounds = Bounds()
        self.weights = weights
        self.decay = -self.weights[:, 20]
        self.factor = (
            torch.pow(
                torch.tensor(0.9, device=self.device, dtype=dtype),
                1.0 / self.decay,
            )
            - 1.0
        )
        self.init_d = torch.clamp(
            self.weights[:, 4] - torch.exp(self.weights[:, 5] * 3.0) + 1.0,
            self.bounds.d_min,
            self.bounds.d_max,
        )
        self.action_retentions = torch.tensor(
            list(action_retentions), device=self.device, dtype=dtype
        )
        self.action_count = int(self.action_retentions.numel())
        self.action_retention_factor = (
            torch.pow(
                self.action_retentions[None, :],
                1.0 / self.decay[:, None],
            )
            - 1.0
        )
        self.first_rating_prob = first_prob
        self.review_rating_prob = review_prob
        self.learning_cost_minutes = learning_costs_t / 60.0
        self.review_cost_minutes = review_costs_t / 60.0

        self.s_count = int(s_grid_size)
        self.d_count = int(d_grid_size)
        self.state_count = self.s_count * self.d_count
        self.log_s_min = math.log(self.bounds.s_min)
        self.log_s_max = math.log(self.bounds.s_max)
        self.s_grid = torch.exp(
            torch.linspace(
                self.log_s_min,
                self.log_s_max,
                self.s_count,
                device=self.device,
                dtype=dtype,
            )
        )
        self.d_grid = torch.linspace(
            self.bounds.d_min,
            self.bounds.d_max,
            self.d_count,
            device=self.device,
            dtype=dtype,
        )
        self.s_mesh = self.s_grid[:, None].expand(self.s_count, self.d_count)
        self.d_mesh = self.d_grid[None, :].expand(self.s_count, self.d_count)
        self._flat_state_idx = torch.arange(
            self.state_count,
            device=self.device,
            dtype=torch.int64,
        )
        self._flat_s_idx = (
            torch.arange(self.s_count, device=self.device, dtype=torch.int64)[:, None]
            .expand(self.s_count, self.d_count)
            .reshape(-1)
        )
        self.memorized_by_day = self._precompute_memorized_by_day()
        self._action_tables = self._precompute_action_tables()

    def _user_payload(self, user_idx: int) -> dict[str, Any]:
        return {
            "fsrs_weights": _tensor_float_list(self.weights[user_idx]),
            "first_rating_prob": _tensor_float_list(self.first_rating_prob[user_idx]),
            "review_rating_prob": _tensor_float_list(self.review_rating_prob[user_idx]),
            "learning_costs": [
                60.0 * value
                for value in _tensor_float_list(self.learning_cost_minutes[user_idx])
            ],
            "review_costs": [
                60.0 * value
                for value in _tensor_float_list(self.review_cost_minutes[user_idx])
            ],
        }

    def _cache_key_parts(
        self,
        *,
        oracle_kind: str,
        method: str,
        user_idx: int,
        cost_weight: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "algorithm_version": 1,
            "oracle_kind": oracle_kind,
            "method": method,
            "days": self.days,
            "s_grid_size": self.s_count,
            "d_grid_size": self.d_count,
            "action_retentions": _tensor_float_list(self.action_retentions),
            "dtype": str(self.dtype),
            "user_config": self._user_payload(user_idx),
            "cost_weight": float(cost_weight),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def _stationary_cache_extra(
        self,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "transition_kernel": self.TRANSITION_KERNEL_VERSION,
        }
        if max_iterations is not None:
            extra["max_iterations"] = max_iterations
        if tolerance is not None:
            extra["tolerance"] = tolerance
        return extra

    def _cost_weight_tensor(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(cost_weights, device=self.device, dtype=self.dtype)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim > 2:
            trailing = int(math.prod(tensor.shape[2:]))
            if trailing != 1:
                raise ValueError(
                    "cost_weights broadcast tensor must have singleton trailing dims."
                )
            tensor = tensor.reshape(int(tensor.shape[0]), int(tensor.shape[1]))
        if tensor.ndim not in (1, 2):
            raise ValueError("cost_weights must be a 1D shared or 2D user grid.")
        if tensor.numel() <= 0:
            raise ValueError("cost_weights must contain at least one value.")
        if tensor.ndim == 2 and int(tensor.shape[0]) not in (1, self.user_count):
            raise ValueError(
                "2D cost_weights must have shape [user_count, weight_count] "
                "or [1, weight_count]."
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("cost_weights must be finite.")
        if bool((tensor < 0.0).any().item()):
            raise ValueError("cost_weights must be non-negative.")
        return tensor.contiguous()

    def _cost_weight_grid(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
    ) -> torch.Tensor:
        tensor = self._cost_weight_tensor(cost_weights)
        if tensor.ndim == 1:
            return tensor.view(1, int(tensor.numel())).expand(self.user_count, -1)
        if int(tensor.shape[0]) == 1:
            return tensor.expand(self.user_count, -1)
        return tensor

    def _cost_weight_count(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
    ) -> int:
        tensor = self._cost_weight_tensor(cost_weights)
        return int(tensor.numel()) if tensor.ndim == 1 else int(tensor.shape[1])

    def _user_cost_weight_list(
        self,
        cost_weights_by_user: Sequence[float],
    ) -> list[float]:
        weights = [float(weight) for weight in cost_weights_by_user]
        if not weights:
            raise ValueError("cost_weights_by_user must contain at least one value.")
        if len(weights) != self.user_count:
            raise ValueError("cost_weights_by_user length must equal user_count.")
        self._cost_weight_tensor(weights)
        return weights

    def _suboracle(
        self, user_indices: Sequence[int]
    ) -> FSRS6BatchedStationaryFiniteOracle:
        return FSRS6BatchedStationaryFiniteOracle(
            days=self.days,
            action_retentions=_tensor_float_list(self.action_retentions),
            s_grid_size=self.s_count,
            d_grid_size=self.d_count,
            fsrs_weights=[
                _tensor_float_list(self.weights[idx]) for idx in user_indices
            ],
            first_rating_prob=[
                _tensor_float_list(self.first_rating_prob[idx]) for idx in user_indices
            ],
            review_rating_prob=[
                _tensor_float_list(self.review_rating_prob[idx]) for idx in user_indices
            ],
            learning_costs=[
                [
                    60.0 * value
                    for value in _tensor_float_list(self.learning_cost_minutes[idx])
                ]
                for idx in user_indices
            ],
            review_costs=[
                [
                    60.0 * value
                    for value in _tensor_float_list(self.review_cost_minutes[idx])
                ]
                for idx in user_indices
            ],
            dtype=self.dtype,
            device=self.device,
            cache_config=OracleDPCacheConfig(enabled=False),
        )

    def solve_stationary_finite_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        weight_list = [float(weight) for weight in cost_weights]
        policies: list[list[torch.Tensor | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        metrics: list[list[OracleMetrics | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        objectives = torch.zeros(
            (self.user_count, len(weight_list)),
            device=self.device,
            dtype=self.dtype,
        )
        iterations: list[list[int | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        converged: list[list[bool | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        residuals: list[list[float | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        missing_by_user: dict[int, list[int]] = {}
        extra = self._stationary_cache_extra(
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        for user_idx in range(self.user_count):
            for weight_idx, weight in enumerate(weight_list):
                entry = load_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="stationary_finite",
                        method="solve_stationary_finite_policies",
                        user_idx=user_idx,
                        cost_weight=weight,
                        extra=extra,
                    ),
                    map_location=self.device,
                )
                if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                    missing_by_user.setdefault(user_idx, []).append(weight_idx)
                    continue
                policies[user_idx][weight_idx] = entry["policy"].to(device=self.device)
                metrics[user_idx][weight_idx] = _metrics_from_payload(
                    entry["metrics"],
                    runtime_s=0.0,
                )
                objectives[user_idx, weight_idx] = float(entry["objective"])
                iterations[user_idx][weight_idx] = int(entry["iterations"])
                converged[user_idx][weight_idx] = bool(entry["converged"])
                residuals[user_idx][weight_idx] = float(entry["residual"])

        groups: dict[tuple[int, ...], list[int]] = {}
        for user_idx, missing_weight_indices in missing_by_user.items():
            groups.setdefault(tuple(missing_weight_indices), []).append(user_idx)
        for missing_weight_indices, user_indices in groups.items():
            suboracle = self._suboracle(user_indices)
            group_weights = [weight_list[idx] for idx in missing_weight_indices]
            solution = suboracle._solve_stationary_finite_policies_uncached(
                group_weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_user_idx, user_idx in enumerate(user_indices):
                for local_weight_idx, weight_idx in enumerate(missing_weight_indices):
                    policy = solution.policy[local_user_idx, local_weight_idx].to(
                        device=self.device
                    )
                    metric = solution.metrics[local_user_idx][local_weight_idx]
                    objective = float(
                        solution.objectives[local_user_idx, local_weight_idx].item()
                    )
                    iteration = solution.iterations[local_user_idx][local_weight_idx]
                    did_converge = solution.converged[local_user_idx][local_weight_idx]
                    residual = solution.residuals[local_user_idx][local_weight_idx]
                    policies[user_idx][weight_idx] = policy
                    metrics[user_idx][weight_idx] = metric
                    objectives[user_idx, weight_idx] = objective
                    iterations[user_idx][weight_idx] = iteration
                    converged[user_idx][weight_idx] = did_converge
                    residuals[user_idx][weight_idx] = residual
                    write_cache_entry(
                        self.cache_config,
                        key_parts=self._cache_key_parts(
                            oracle_kind="stationary_finite",
                            method="solve_stationary_finite_policies",
                            user_idx=user_idx,
                            cost_weight=weight_list[weight_idx],
                            extra=extra,
                        ),
                        data={
                            "policy": policy,
                            "metrics": _metrics_payload(metric),
                            "objective": objective,
                            "iterations": iteration,
                            "converged": did_converge,
                            "residual": residual,
                        },
                    )

        return BatchedStationaryFiniteOracleSolution(
            policy=torch.stack(
                [
                    torch.stack(
                        [policy for policy in user_policies if policy is not None],
                        dim=0,
                    )
                    for user_policies in policies
                ],
                dim=0,
            )
            .to(device=self.device, dtype=torch.int64)
            .reshape(self.user_count, len(weight_list), self.s_count, self.d_count),
            metrics=[
                [metric for metric in user_metrics if metric is not None]
                for user_metrics in metrics
            ],
            objectives=objectives,
            iterations=[
                [int(value) for value in row if value is not None] for row in iterations
            ],
            converged=[
                [bool(value) for value in row if value is not None] for row in converged
            ],
            residuals=[
                [float(value) for value in row if value is not None]
                for row in residuals
            ],
            runtime_s=time.perf_counter() - start,
        )

    def solve_stationary_finite_policies_for_user_weights(
        self,
        cost_weights_by_user: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        weight_list = self._user_cost_weight_list(cost_weights_by_user)
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in range(self.user_count)]
        metrics: list[OracleMetrics | None] = [None for _ in range(self.user_count)]
        objectives = torch.zeros(
            (self.user_count, 1),
            device=self.device,
            dtype=self.dtype,
        )
        iterations: list[int | None] = [None for _ in range(self.user_count)]
        converged: list[bool | None] = [None for _ in range(self.user_count)]
        residuals: list[float | None] = [None for _ in range(self.user_count)]
        missing_user_indices: list[int] = []
        extra = self._stationary_cache_extra(
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        for user_idx, weight in enumerate(weight_list):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="stationary_finite",
                    method="solve_stationary_finite_policies",
                    user_idx=user_idx,
                    cost_weight=weight,
                    extra=extra,
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing_user_indices.append(user_idx)
                continue
            policies[user_idx] = entry["policy"].to(device=self.device)
            metrics[user_idx] = _metrics_from_payload(
                entry["metrics"],
                runtime_s=0.0,
            )
            objectives[user_idx, 0] = float(entry["objective"])
            iterations[user_idx] = int(entry["iterations"])
            converged[user_idx] = bool(entry["converged"])
            residuals[user_idx] = float(entry["residual"])

        if missing_user_indices:
            suboracle = self._suboracle(missing_user_indices)
            group_weights = [weight_list[idx] for idx in missing_user_indices]
            solution = suboracle._solve_stationary_finite_policies_uncached(
                torch.tensor(
                    group_weights,
                    device=self.device,
                    dtype=self.dtype,
                ).view(len(group_weights), 1),
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_user_idx, user_idx in enumerate(missing_user_indices):
                policy = solution.policy[local_user_idx, 0].to(device=self.device)
                metric = solution.metrics[local_user_idx][0]
                objective = float(solution.objectives[local_user_idx, 0].item())
                iteration = solution.iterations[local_user_idx][0]
                did_converge = solution.converged[local_user_idx][0]
                residual = solution.residuals[local_user_idx][0]
                policies[user_idx] = policy
                metrics[user_idx] = metric
                objectives[user_idx, 0] = objective
                iterations[user_idx] = iteration
                converged[user_idx] = did_converge
                residuals[user_idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="stationary_finite",
                        method="solve_stationary_finite_policies",
                        user_idx=user_idx,
                        cost_weight=weight_list[user_idx],
                        extra=extra,
                    ),
                    data={
                        "policy": policy,
                        "metrics": _metrics_payload(metric),
                        "objective": objective,
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )

        return BatchedStationaryFiniteOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            )
            .to(device=self.device, dtype=torch.int64)
            .reshape(self.user_count, 1, self.s_count, self.d_count),
            metrics=[[metric] for metric in metrics if metric is not None],
            objectives=objectives,
            iterations=[[int(value)] for value in iterations if value is not None],
            converged=[[bool(value)] for value in converged if value is not None],
            residuals=[[float(value)] for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def _solve_stationary_finite_policies_uncached(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        start = time.perf_counter()
        weight_tensor = self._cost_weight_tensor(cost_weights)
        weight_count = self._cost_weight_count(weight_tensor)
        if weight_tensor.ndim == 1:
            finite_policies = self.solve_policies(
                _tensor_float_list(weight_tensor),
                progress=progress,
            )
        else:
            finite_policies = self._solve_policies_uncached(
                weight_tensor,
                progress=progress,
            )
        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.user_count * weight_count * max_iterations,
                desc="Stationary finite oracle",
                unit="iter",
                leave=False,
            )
        try:
            (
                policy,
                metrics,
                iterations,
                converged,
                residuals,
                objectives,
            ) = self._solve_stationary_finite_policy_batch(
                cost_weights=weight_tensor,
                finite_policies=finite_policies,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress_bar=progress_bar,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return BatchedStationaryFiniteOracleSolution(
            policy=policy,
            metrics=metrics,
            objectives=objectives,
            iterations=iterations,
            converged=converged,
            residuals=residuals,
            runtime_s=time.perf_counter() - start,
        )

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        weight_list = [float(weight) for weight in cost_weights]
        policies: list[list[torch.Tensor | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        missing_by_user: dict[int, list[int]] = {}
        for user_idx in range(self.user_count):
            for weight_idx, weight in enumerate(weight_list):
                entry = load_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="stationary_finite",
                        method="solve_policies",
                        user_idx=user_idx,
                        cost_weight=weight,
                        extra=self._stationary_cache_extra(),
                    ),
                    map_location=self.device,
                )
                if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                    missing_by_user.setdefault(user_idx, []).append(weight_idx)
                    continue
                policies[user_idx][weight_idx] = entry["policy"].to(device=self.device)

        groups: dict[tuple[int, ...], list[int]] = {}
        for user_idx, missing_weight_indices in missing_by_user.items():
            groups.setdefault(tuple(missing_weight_indices), []).append(user_idx)
        for missing_weight_indices, user_indices in groups.items():
            suboracle = self._suboracle(user_indices)
            group_weights = [weight_list[idx] for idx in missing_weight_indices]
            computed = suboracle._solve_policies_uncached(
                group_weights,
                progress=progress,
            )
            for local_user_idx, user_idx in enumerate(user_indices):
                for local_weight_idx, weight_idx in enumerate(missing_weight_indices):
                    policy = computed[local_user_idx, local_weight_idx].to(
                        device=self.device
                    )
                    policies[user_idx][weight_idx] = policy
                    write_cache_entry(
                        self.cache_config,
                        key_parts=self._cache_key_parts(
                            oracle_kind="stationary_finite",
                            method="solve_policies",
                            user_idx=user_idx,
                            cost_weight=weight_list[weight_idx],
                            extra=self._stationary_cache_extra(),
                        ),
                        data={"policy": policy},
                    )

        return torch.stack(
            [
                torch.stack(
                    [policy for policy in user_policies if policy is not None],
                    dim=0,
                )
                for user_policies in policies
            ],
            dim=0,
        ).to(device=self.device)

    def solve_policies_for_user_weights(
        self,
        cost_weights_by_user: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_list = self._user_cost_weight_list(cost_weights_by_user)
        policies: list[torch.Tensor | None] = [None for _ in range(self.user_count)]
        missing_user_indices: list[int] = []
        for user_idx, weight in enumerate(weight_list):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="stationary_finite",
                    method="solve_policies",
                    user_idx=user_idx,
                    cost_weight=weight,
                    extra=self._stationary_cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing_user_indices.append(user_idx)
                continue
            policies[user_idx] = entry["policy"].to(device=self.device)

        if missing_user_indices:
            suboracle = self._suboracle(missing_user_indices)
            group_weights = [weight_list[idx] for idx in missing_user_indices]
            computed = suboracle._solve_policies_uncached(
                torch.tensor(
                    group_weights,
                    device=self.device,
                    dtype=self.dtype,
                ).view(len(group_weights), 1),
                progress=progress,
            )
            for local_user_idx, user_idx in enumerate(missing_user_indices):
                policy = computed[local_user_idx, 0].to(device=self.device)
                policies[user_idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="stationary_finite",
                        method="solve_policies",
                        user_idx=user_idx,
                        cost_weight=weight_list[user_idx],
                        extra=self._stationary_cache_extra(),
                    ),
                    data={"policy": policy},
                )

        return (
            torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            )
            .to(device=self.device)
            .reshape(
                self.user_count,
                1,
                self.horizon + 1,
                self.state_count,
            )
        )

    def _solve_policies_uncached(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_tensor = self._cost_weight_tensor(cost_weights)
        weight_grid = self._cost_weight_grid(weight_tensor)
        weight_count = int(weight_grid.shape[1])
        value = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.horizon + 1,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        policy = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.horizon + 1,
                self.state_count,
            ),
            device=self.device,
            dtype=torch.uint8,
        )

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Stationary finite oracle batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full(
                    (
                        self.user_count,
                        weight_count,
                        self.state_count,
                    ),
                    -math.inf,
                    device=self.device,
                    dtype=self.dtype,
                )
                best_action = torch.zeros(
                    (
                        self.user_count,
                        weight_count,
                        self.state_count,
                    ),
                    device=self.device,
                    dtype=torch.uint8,
                )
                for action_idx in range(self.action_count):
                    candidate = self._candidate_value_batch(
                        action_idx=action_idx,
                        rem=rem,
                        cost_weights=weight_grid,
                        value=value,
                    )
                    better = candidate > best_value
                    best_value = torch.where(better, candidate, best_value)
                    best_action = torch.where(
                        better,
                        torch.full_like(best_action, action_idx),
                        best_action,
                    )
                value[:, :, rem, :] = best_value
                policy[:, :, rem, :] = best_action
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return policy

    def labels(
        self,
        *,
        policies: torch.Tensor,
        cost_weights: torch.Tensor,
        user_index: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
        goal_weight: torch.Tensor,
    ) -> torch.Tensor:
        s_idx = self._s_to_idx(s)
        d_idx = self._d_to_idx(d)
        goal_idx = torch.argmin(
            torch.abs(
                goal_weight.to(dtype=cost_weights.dtype)[:, None]
                - cost_weights[None, :]
            ),
            dim=1,
        )
        return policies.to(device=s.device)[
            user_index.to(torch.int64), goal_idx, s_idx, d_idx
        ]

    def _state_kernel(
        self,
        s: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        s_pos = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        s_pos = torch.clamp(
            s_pos * float(self.s_count - 1), 0.0, float(self.s_count - 1)
        )
        s0 = torch.floor(s_pos).to(torch.int64)
        s1 = torch.clamp(s0 + 1, max=self.s_count - 1)
        sw = s_pos - s0.to(dtype=self.dtype)

        d_pos = torch.clamp(d, self.bounds.d_min, self.bounds.d_max)
        d_pos = (d_pos - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min)
        d_pos = torch.clamp(
            d_pos * float(self.d_count - 1), 0.0, float(self.d_count - 1)
        )
        d0 = torch.floor(d_pos).to(torch.int64)
        d1 = torch.clamp(d0 + 1, max=self.d_count - 1)
        dw = d_pos - d0.to(dtype=self.dtype)

        next_idx = torch.stack(
            (
                s0 * self.d_count + d0,
                s1 * self.d_count + d0,
                s0 * self.d_count + d1,
                s1 * self.d_count + d1,
            ),
            dim=0,
        )
        next_weight = torch.stack(
            (
                (1.0 - sw) * (1.0 - dw),
                sw * (1.0 - dw),
                (1.0 - sw) * dw,
                sw * dw,
            ),
            dim=0,
        )
        return next_idx.to(dtype=torch.int64), next_weight.to(dtype=self.dtype)

    def _solve_stationary_finite_policy_batch(
        self,
        *,
        cost_weights: torch.Tensor,
        finite_policies: torch.Tensor,
        max_iterations: int,
        tolerance: float,
        progress_bar: Any | None,
    ) -> tuple[
        torch.Tensor,
        list[list[OracleMetrics]],
        list[list[int]],
        list[list[bool]],
        list[list[float]],
        torch.Tensor,
    ]:
        weight_count = int(finite_policies.shape[1])
        policy = self._project_stationary_policies(finite_policies)
        value = self._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=cost_weights,
        )
        objective = self._objective_from_value_batch(
            value=value,
            cost_weights=cost_weights,
        )
        iterations = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=torch.int64,
        )
        converged = torch.zeros_like(iterations, dtype=torch.bool)
        residuals = torch.full(
            iterations.shape,
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones_like(converged)

        for iteration in range(1, max_iterations + 1):
            if not bool(active.any().item()):
                break
            occupancy = self._rollout_occupancy_batch(policy=policy, stationary=True)
            new_policy, residual, visited = self._improve_stationary_policy_batch(
                policy=policy,
                occupancy=occupancy,
                value=value,
                cost_weights=cost_weights,
            )
            policy_changed = ((new_policy != policy) & visited).any(dim=2)
            iterations[active] = iteration
            residuals[active] = residual[active]
            if progress_bar is not None:
                progress_bar.update(int(active.sum().item()))

            done = active & ((~policy_changed) | (residual <= tolerance))
            if bool(done.any().item()):
                converged[done] = True
                active[done] = False

            candidate = active & ~done
            if bool(candidate.any().item()):
                candidate_value = self._evaluate_stationary_policy_value_batch(
                    policy=new_policy,
                    cost_weights=cost_weights,
                )
                candidate_objective = self._objective_from_value_batch(
                    value=candidate_value,
                    cost_weights=cost_weights,
                )
                objective_improvement = candidate_objective - objective
                accepted = candidate & (objective_improvement > tolerance)
                residuals[candidate] = torch.where(
                    accepted[candidate],
                    objective_improvement[candidate],
                    torch.clamp(objective_improvement[candidate], min=0.0),
                )
                if bool((candidate & ~accepted).any().item()):
                    rejected = candidate & ~accepted
                    converged[rejected] = True
                    active[rejected] = False
                if bool(accepted.any().item()):
                    policy[accepted] = new_policy[accepted]
                    value[accepted] = candidate_value[accepted]
                    objective[accepted] = candidate_objective[accepted]

        if bool(active.any().item()):
            iterations[active] = max_iterations

        metrics = self._metrics_from_occupancy_batch(
            policy=policy,
            cost_weights=cost_weights,
        )
        return (
            policy.reshape(
                self.user_count,
                weight_count,
                self.s_count,
                self.d_count,
            ),
            metrics,
            [[int(value) for value in row.tolist()] for row in iterations.cpu()],
            [[bool(value) for value in row.tolist()] for row in converged.cpu()],
            [[float(value) for value in row.tolist()] for row in residuals.cpu()],
            objective,
        )

    def _candidate_value_batch(
        self,
        *,
        action_idx: int,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(value.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        tables = self._action_tables
        interval = tables.interval
        prob = tables.prob
        next_idx = tables.next_idx
        next_weight = tables.next_weight
        interval_u = interval[:, action_idx, :]
        cont_mask = interval_u <= rem
        future_rem = torch.clamp(rem - interval_u, min=0).to(torch.int64)
        active_days = torch.minimum(interval_u, torch.full_like(interval_u, rem))
        candidate_value = (
            self._memorized_sum_batch(active_days)
            .unsqueeze(1)
            .expand(
                -1,
                weight_count,
                -1,
            )
            .clone()
        )

        if not bool(cont_mask.any().item()):
            return candidate_value

        user_idx = self._user_index_view(3).expand(
            self.user_count,
            weight_count,
            self.state_count,
        )
        weight_idx = self._weight_index_view(3, weight_count).expand_as(candidate_value)
        weight_penalty = cost_weight_grid[:, :, None]
        future_rem_exp = future_rem[:, None, :].expand_as(candidate_value)
        next_idx_u = next_idx[:, action_idx, :, :, :]
        next_weight_u = next_weight[:, action_idx, :, :, :]
        prob_u = prob[:, action_idx, :, :]
        for rating_idx, rating in enumerate(range(1, 5)):
            weighted = torch.where(
                cont_mask,
                prob_u[:, rating_idx, :],
                torch.zeros_like(prob_u[:, rating_idx, :]),
            )[:, None, :]
            future_value = torch.zeros_like(candidate_value)
            for corner_idx in range(4):
                future_value += (
                    next_weight_u[:, rating_idx, corner_idx, :][
                        :,
                        None,
                        :,
                    ]
                    * value[
                        user_idx,
                        weight_idx,
                        future_rem_exp,
                        next_idx_u[:, rating_idx, corner_idx, :][
                            :,
                            None,
                            :,
                        ].expand_as(candidate_value),
                    ]
                )
            review_minutes = self.review_cost_minutes[:, rating - 1].view(
                self.user_count,
                1,
                1,
            )
            candidate_value += weighted * (
                future_value - weight_penalty * review_minutes
            )
        return candidate_value

    def _project_stationary_policies(
        self, finite_policies: torch.Tensor
    ) -> torch.Tensor:
        occupancy = self._rollout_occupancy_batch(
            policy=finite_policies, stationary=False
        )
        action_weight = torch.zeros(
            (
                self.user_count,
                int(finite_policies.shape[1]),
                self.action_count,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        batch_offset = (
            torch.arange(
                self.user_count * int(finite_policies.shape[1]),
                device=self.device,
                dtype=torch.int64,
            )
            .view(self.user_count, int(finite_policies.shape[1]), 1)
            .mul(self.action_count * self.state_count)
        )
        state_idx = self._flat_state_idx.view(1, 1, self.state_count)
        for rem in range(1, self.horizon + 1):
            occ = occupancy[:, :, rem, :]
            if not bool(occ.sum().item()):
                continue
            policy_rem = finite_policies[:, :, rem, :].to(torch.int64)
            scatter_idx = batch_offset + policy_rem * self.state_count + state_idx
            action_weight.reshape(-1).scatter_add_(
                0,
                scatter_idx.reshape(-1),
                occ.reshape(-1),
            )

        projected = finite_policies[:, :, self.horizon, :].clone()
        visited = action_weight.sum(dim=2) > 0.0
        best_action = action_weight.argmax(dim=2).to(torch.uint8)
        projected = torch.where(visited, best_action, projected)
        return projected

    def _select_stationary_policy_tables_batch(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tables = self._action_tables
        weight_count = int(policy.shape[1])
        policy_flat = policy.reshape(
            self.user_count,
            weight_count,
            self.state_count,
        ).to(torch.int64)
        interval_src = tables.interval[:, None, :, :].expand(
            self.user_count,
            weight_count,
            self.action_count,
            self.state_count,
        )
        selected_interval = interval_src.gather(
            2,
            policy_flat[:, :, None, :],
        ).squeeze(2)
        action_idx = policy_flat[:, :, None, None, :].expand(
            self.user_count,
            weight_count,
            1,
            4,
            self.state_count,
        )
        prob_src = tables.prob[:, None, :, :, :].expand(
            self.user_count,
            weight_count,
            self.action_count,
            4,
            self.state_count,
        )
        next_src = tables.next_idx[:, None, :, :, :, :].expand(
            self.user_count,
            weight_count,
            self.action_count,
            4,
            4,
            self.state_count,
        )
        selected_prob = prob_src.gather(2, action_idx).squeeze(2)
        kernel_action_idx = policy_flat[:, :, None, None, None, :].expand(
            self.user_count,
            weight_count,
            1,
            4,
            4,
            self.state_count,
        )
        selected_next_idx = next_src.gather(2, kernel_action_idx).squeeze(2)
        weight_src = tables.next_weight[:, None, :, :, :, :].expand(
            self.user_count,
            weight_count,
            self.action_count,
            4,
            4,
            self.state_count,
        )
        selected_next_weight = weight_src.gather(
            2,
            kernel_action_idx,
        ).squeeze(2)
        return selected_interval, selected_prob, selected_next_idx, selected_next_weight

    def _rollout_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        stationary: bool,
    ) -> torch.Tensor:
        if stationary:
            policy = policy.reshape(self.user_count, policy.shape[1], self.state_count)
        weight_count = int(policy.shape[1])
        occupancy = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.horizon + 1,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        batch_offset = (
            torch.arange(
                self.user_count * weight_count,
                device=self.device,
                dtype=torch.int64,
            )
            .view(self.user_count, weight_count, 1)
            .mul((self.horizon + 1) * self.state_count)
        )

        for rating in range(1, 5):
            prob = self.first_rating_prob[:, rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            state_idx, state_weight = self._state_kernel(s0, d0)
            for corner_idx in range(4):
                target = (
                    batch_offset
                    + self.horizon * self.state_count
                    + state_idx[corner_idx].view(self.user_count, 1, 1)
                )
                amount = (prob[:, None] * state_weight[corner_idx][:, None]).expand(
                    self.user_count, weight_count
                )
                occupancy.reshape(-1).scatter_add_(
                    0,
                    target.reshape(-1),
                    amount.reshape(-1),
                )

        flat_occupancy = occupancy.reshape(-1)
        selected_interval: torch.Tensor | None = None
        selected_prob: torch.Tensor | None = None
        selected_next_idx: torch.Tensor | None = None
        selected_next_weight: torch.Tensor | None = None
        if stationary:
            (
                selected_interval,
                selected_prob,
                selected_next_idx,
                selected_next_weight,
            ) = self._select_stationary_policy_tables_batch(policy)
        for rem in range(self.horizon, 0, -1):
            current = occupancy[:, :, rem, :]
            if not bool(current.sum().item()):
                continue
            if not stationary:
                (
                    selected_interval,
                    selected_prob,
                    selected_next_idx,
                    selected_next_weight,
                ) = self._select_stationary_policy_tables_batch(policy[:, :, rem, :])
            if (
                selected_interval is None
                or selected_prob is None
                or selected_next_idx is None
                or selected_next_weight is None
            ):
                raise RuntimeError("selected policy tables were not initialized.")
            cont_mask = selected_interval <= rem
            if not bool(cont_mask.any().item()):
                continue
            source = current * cont_mask.to(dtype=self.dtype)
            if not bool(source.sum().item()):
                continue
            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            base = batch_offset + future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * selected_prob[:, :, rating_idx, :]
                        * selected_next_weight[:, :, rating_idx, corner_idx, :]
                    )
                    target = base + selected_next_idx[:, :, rating_idx, corner_idx, :]
                    flat_occupancy.scatter_add_(
                        0,
                        target.reshape(-1),
                        amount.reshape(-1),
                    )

        return occupancy

    def _evaluate_stationary_policy_value_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        policy_flat = policy.reshape(
            self.user_count,
            weight_count,
            self.state_count,
        )
        value = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.horizon + 1,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        selected_interval, selected_prob, selected_next_idx, selected_next_weight = (
            self._select_stationary_policy_tables_batch(policy_flat)
        )
        user_idx = self._user_index_view(3).expand_as(selected_interval)
        weight_idx = self._weight_index_view(3, weight_count).expand_as(
            selected_interval
        )
        weight_penalty = cost_weight_grid[:, :, None]
        for rem in range(1, self.horizon + 1):
            cont_mask = selected_interval <= rem
            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            active_days = torch.minimum(
                selected_interval,
                torch.full_like(selected_interval, rem),
            )
            value_rem = self._memorized_sum_batch(active_days).clone()
            if bool(cont_mask.any().item()):
                for rating_idx, rating in enumerate(range(1, 5)):
                    weighted = torch.where(
                        cont_mask,
                        selected_prob[:, :, rating_idx, :],
                        torch.zeros_like(selected_prob[:, :, rating_idx, :]),
                    )
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            selected_next_weight[
                                :,
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ]
                            * value[
                                user_idx,
                                weight_idx,
                                future_rem,
                                selected_next_idx[:, :, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[:, rating - 1].view(
                        self.user_count,
                        1,
                        1,
                    )
                    value_rem += weighted * (
                        future_value - weight_penalty * review_minutes
                    )
            value[:, :, rem, :] = value_rem
        return value

    def _objective_from_value_batch(
        self,
        *,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(value.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        total_value = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=self.dtype,
        )
        total_learning_minutes = (
            self.first_rating_prob * self.learning_cost_minutes
        ).sum(dim=1)
        user_idx = self._user_index_view(2).expand(self.user_count, weight_count)
        weight_idx = self._weight_index_view(2, weight_count).expand(
            self.user_count,
            weight_count,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[:, rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            state_idx, state_weight = self._state_kernel(s0, d0)
            for corner_idx in range(4):
                total_value += (
                    prob[:, None]
                    * state_weight[corner_idx][:, None]
                    * value[
                        user_idx,
                        weight_idx,
                        self.horizon,
                        state_idx[corner_idx][:, None].expand(
                            self.user_count,
                            weight_count,
                        ),
                    ]
                )
        return (
            total_value - cost_weight_grid * total_learning_minutes[:, None]
        ) / float(self.days)

    def _improve_stationary_policy_batch(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        action_scores = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.action_count,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        visited = occupancy[:, :, 1:, :].sum(dim=2) > 0.0
        user_idx = self._user_index_view(4).expand(
            self.user_count,
            weight_count,
            self.action_count,
            self.state_count,
        )
        weight_idx = self._weight_index_view(4, weight_count).expand_as(user_idx)
        weight_penalty = cost_weight_grid[:, :, None, None]
        tables = self._action_tables
        interval = tables.interval
        prob = tables.prob
        next_idx = tables.next_idx
        next_weight = tables.next_weight
        for rem in range(1, self.horizon + 1):
            rem_occupancy = occupancy[:, :, rem, :]
            if not bool(rem_occupancy.sum().item()):
                continue
            interval_u = interval
            cont_mask = interval_u <= rem
            future_rem = torch.clamp(rem - interval_u, min=0).to(torch.int64)
            active_days = torch.minimum(
                interval_u,
                torch.full_like(interval_u, rem),
            )
            candidate_value = (
                self._memorized_sum_batch_actions(active_days)[
                    :,
                    None,
                    :,
                    :,
                ]
                .expand(
                    self.user_count,
                    weight_count,
                    self.action_count,
                    self.state_count,
                )
                .clone()
            )
            if bool(cont_mask.any().item()):
                future_rem_exp = future_rem[:, None, :, :].expand(
                    self.user_count,
                    weight_count,
                    self.action_count,
                    self.state_count,
                )
                for rating_idx, rating in enumerate(range(1, 5)):
                    weighted = torch.where(
                        cont_mask,
                        prob[:, :, rating_idx, :],
                        torch.zeros_like(prob[:, :, rating_idx, :]),
                    )[:, None, :, :]
                    future_value = torch.zeros_like(candidate_value)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[
                                :,
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ][:, None, :, :]
                            * value[
                                user_idx,
                                weight_idx,
                                future_rem_exp,
                                next_idx[:, :, rating_idx, corner_idx, :][
                                    :,
                                    None,
                                    :,
                                    :,
                                ].expand(
                                    self.user_count,
                                    weight_count,
                                    self.action_count,
                                    self.state_count,
                                ),
                            ]
                        )
                    review_minutes = self.review_cost_minutes[:, rating - 1].view(
                        self.user_count,
                        1,
                        1,
                        1,
                    )
                    candidate_value += weighted * (
                        future_value - weight_penalty * review_minutes
                    )
            action_scores += rem_occupancy[:, :, None, :] * candidate_value

        best_score, best_action = action_scores.max(dim=2)
        policy_flat = policy.reshape(
            self.user_count,
            weight_count,
            self.state_count,
        ).to(torch.int64)
        current_score = action_scores.gather(2, policy_flat[:, :, None, :]).squeeze(2)
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = torch.where(
            visited.any(dim=2),
            masked_improvement.max(dim=2).values,
            torch.zeros(
                (self.user_count, weight_count), device=self.device, dtype=self.dtype
            ),
        )
        new_policy = torch.where(
            visited,
            best_action.to(torch.uint8),
            policy_flat.to(torch.uint8),
        )
        return new_policy, residual, visited

    def _metrics_from_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> list[list[OracleMetrics]]:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        occupancy = self._rollout_occupancy_batch(policy=policy, stationary=True)
        selected_interval, selected_prob, _, _ = (
            self._select_stationary_policy_tables_batch(
                policy.reshape(
                    self.user_count,
                    weight_count,
                    self.state_count,
                )
            )
        )
        total_mem = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=self.dtype,
        )
        total_minutes = (
            (self.first_rating_prob * self.learning_cost_minutes)
            .sum(dim=1)[:, None]
            .expand_as(total_mem)
            .clone()
        )
        total_reviews = torch.zeros_like(total_mem)
        total_lapses = torch.zeros_like(total_mem)
        for rem in range(1, self.horizon + 1):
            current = occupancy[:, :, rem, :]
            if not bool(current.sum().item()):
                continue
            active_days = torch.minimum(
                selected_interval,
                torch.full_like(selected_interval, rem),
            )
            total_mem += (current * self._memorized_sum_batch(active_days)).sum(dim=2)
            cont_mask = selected_interval <= rem
            source = current * cont_mask.to(dtype=self.dtype)
            if not bool(source.sum().item()):
                continue
            expected_review_minutes = (
                selected_prob * self.review_cost_minutes[:, None, :, None]
            ).sum(dim=2)
            total_minutes += (source * expected_review_minutes).sum(dim=2)
            total_reviews += source.sum(dim=2)
            total_lapses += (source * selected_prob[:, :, 0, :]).sum(dim=2)

        day_count = float(self.days)
        objectives = total_mem / day_count - cost_weight_grid * (
            total_minutes / day_count
        )
        metrics: list[list[OracleMetrics]] = []
        for user_idx in range(self.user_count):
            row: list[OracleMetrics] = []
            for weight_idx in range(weight_count):
                reviews_float = float(total_reviews[user_idx, weight_idx].item())
                lapses_float = float(total_lapses[user_idx, weight_idx].item())
                observed_retention = (
                    1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
                )
                row.append(
                    OracleMetrics(
                        card_expected_retrievability=float(
                            total_mem[user_idx, weight_idx].item() / day_count
                        ),
                        card_minutes_per_day=float(
                            total_minutes[user_idx, weight_idx].item() / day_count
                        ),
                        card_reviews_per_day=float(
                            total_reviews[user_idx, weight_idx].item() / day_count
                        ),
                        card_total_reviews=reviews_float,
                        card_total_lapses=lapses_float,
                        card_total_cost_seconds=float(
                            total_minutes[user_idx, weight_idx].item() * 60.0
                        ),
                        observed_retention=observed_retention,
                        scalar_objective=float(objectives[user_idx, weight_idx].item()),
                        runtime_s=0.0,
                    )
                )
            metrics.append(row)
        return metrics

    def _memorized_sum_batch(self, days: torch.Tensor) -> torch.Tensor:
        user_idx = self._user_index_view(days.dim()).expand_as(days)
        state_idx = self._flat_s_idx.view(
            (1,) * (days.dim() - 1) + (self.state_count,)
        ).expand_as(days)
        return self.memorized_by_day[
            user_idx,
            days.to(torch.int64),
            state_idx,
        ]

    def _memorized_sum_batch_actions(self, days: torch.Tensor) -> torch.Tensor:
        return self._memorized_sum_batch(days)

    def _precompute_action_tables(self) -> BatchedTransitionCache:
        interval = torch.clamp(
            torch.round(
                self.s_grid.view(1, 1, self.s_count)
                / self.factor.view(self.user_count, 1, 1)
                * self.action_retention_factor[:, :, None]
            ),
            min=1.0,
        ).to(torch.int64)
        interval = interval[:, :, :, None].expand(
            self.user_count,
            self.action_count,
            self.s_count,
            self.d_count,
        )
        elapsed = interval.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(
            elapsed,
            self.s_mesh[None, None, :, :],
        )
        prob = torch.stack(
            [
                1.0 - retrievability,
                retrievability
                * self.review_rating_prob[:, 0].view(
                    self.user_count,
                    1,
                    1,
                    1,
                ),
                retrievability
                * self.review_rating_prob[:, 1].view(
                    self.user_count,
                    1,
                    1,
                    1,
                ),
                retrievability
                * self.review_rating_prob[:, 2].view(
                    self.user_count,
                    1,
                    1,
                    1,
                ),
            ],
            dim=2,
        )
        rating_idx = torch.full(
            (self.user_count, self.action_count, self.s_count, self.d_count),
            1,
            device=self.device,
            dtype=torch.int64,
        )
        next_idx: list[torch.Tensor] = []
        next_weight: list[torch.Tensor] = []
        for rating in range(1, 5):
            rating_idx.fill_(rating)
            if rating > 1:
                new_s = self._stability_after_success(
                    self.s_mesh[None, None, :, :],
                    retrievability,
                    self.d_mesh[None, None, :, :],
                    rating_idx,
                )
            else:
                new_s = self._stability_after_failure(
                    self.s_mesh[None, None, :, :],
                    retrievability,
                    self.d_mesh[None, None, :, :],
                )
            new_d = self._next_d(
                self.d_mesh[None, None, :, :],
                rating_idx,
            )
            kernel_idx, kernel_weight = self._state_kernel(new_s, new_d)
            next_idx.append(kernel_idx)
            next_weight.append(kernel_weight)
        return BatchedTransitionCache(
            interval=interval.reshape(
                self.user_count,
                self.action_count,
                self.state_count,
            ),
            prob=prob.reshape(
                self.user_count,
                self.action_count,
                4,
                self.state_count,
            ),
            next_idx=torch.stack(next_idx, dim=2)
            .permute(1, 3, 2, 0, 4, 5)
            .reshape(
                self.user_count,
                self.action_count,
                4,
                4,
                self.state_count,
            ),
            next_weight=torch.stack(next_weight, dim=2)
            .permute(1, 3, 2, 0, 4, 5)
            .reshape(
                self.user_count,
                self.action_count,
                4,
                4,
                self.state_count,
            ),
        )

    def _precompute_memorized_by_day(self) -> torch.Tensor:
        table = torch.zeros(
            (self.user_count, self.horizon + 1, self.s_count),
            device=self.device,
            dtype=self.dtype,
        )
        if self.horizon <= 0:
            return table
        elapsed = torch.arange(
            1,
            self.horizon + 1,
            device=self.device,
            dtype=self.dtype,
        )
        retrievability = self._forgetting_curve(
            elapsed[None, :, None],
            self.s_grid[None, None, :],
        )
        table[:, 1:] = torch.cumsum(retrievability, dim=1)
        return table

    def _user_index_view(self, ndim: int) -> torch.Tensor:
        return torch.arange(
            self.user_count,
            device=self.device,
            dtype=torch.int64,
        ).view((self.user_count,) + (1,) * (ndim - 1))

    def _weight_index_view(self, ndim: int, weight_count: int) -> torch.Tensor:
        return torch.arange(
            weight_count,
            device=self.device,
            dtype=torch.int64,
        ).view((1, weight_count) + (1,) * (ndim - 2))

    def _forgetting_curve(self, elapsed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.pow(
            1.0
            + self.factor.view((self.user_count,) + (1,) * (elapsed.dim() - 1))
            * elapsed
            / torch.clamp(s, min=self.bounds.s_min),
            self.decay.view((self.user_count,) + (1,) * (elapsed.dim() - 1)),
        )

    def _init_state_scalar(self, rating: int) -> tuple[torch.Tensor, torch.Tensor]:
        rating_f = torch.tensor(float(rating), device=self.device, dtype=self.dtype)
        s = self.weights[:, rating - 1]
        d = self.weights[:, 4] - torch.exp(self.weights[:, 5] * (rating_f - 1.0)) + 1.0
        return s, torch.clamp(d, self.bounds.d_min, self.bounds.d_max)

    def _s_to_idx(self, s: torch.Tensor) -> torch.Tensor:
        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        ratio = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        return torch.clamp(
            torch.round(ratio * float(self.s_count - 1)),
            min=0,
            max=self.s_count - 1,
        ).to(torch.int64)

    def _d_to_idx(self, d: torch.Tensor) -> torch.Tensor:
        ratio = torch.clamp(d, self.bounds.d_min, self.bounds.d_max) - self.bounds.d_min
        ratio = ratio / (self.bounds.d_max - self.bounds.d_min)
        return torch.clamp(
            torch.round(ratio * float(self.d_count - 1)),
            min=0,
            max=self.d_count - 1,
        ).to(torch.int64)

    def _next_d(self, d: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        rating_f = rating.to(dtype=self.dtype)
        delta_d = -self.weights[:, 6].view(
            (self.user_count,) + (1,) * (d.dim() - 1)
        ) * (rating_f - 3.0)
        new_d = d + delta_d * (10.0 - d) / 9.0
        init_d = self.init_d.view((self.user_count,) + (1,) * (d.dim() - 1))
        new_d = (
            self.weights[:, 7].view((self.user_count,) + (1,) * (d.dim() - 1)) * init_d
            + (1.0 - self.weights[:, 7].view((self.user_count,) + (1,) * (d.dim() - 1)))
            * new_d
        )
        return torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)

    def _stability_after_success(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        hard_penalty = torch.where(
            rating == 2,
            self.weights[:, 15].view((self.user_count,) + (1,) * (s.dim() - 1)),
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        easy_bonus = torch.where(
            rating == 4,
            self.weights[:, 16].view((self.user_count,) + (1,) * (s.dim() - 1)),
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        inc = (
            torch.exp(
                self.weights[:, 8].view((self.user_count,) + (1,) * (s.dim() - 1))
            )
            * (11.0 - d)
            * torch.pow(
                s, -self.weights[:, 9].view((self.user_count,) + (1,) * (s.dim() - 1))
            )
            * (
                torch.exp(
                    (1.0 - retrievability)
                    * self.weights[:, 10].view(
                        (self.user_count,) + (1,) * (s.dim() - 1)
                    )
                )
                - 1.0
            )
        )
        return s * (1.0 + inc * hard_penalty * easy_bonus)

    def _stability_after_failure(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        new_s = (
            self.weights[:, 11].view((self.user_count,) + (1,) * (s.dim() - 1))
            * torch.pow(
                d,
                -self.weights[:, 12].view((self.user_count,) + (1,) * (s.dim() - 1)),
            )
            * (
                torch.pow(
                    s + 1.0,
                    self.weights[:, 13].view((self.user_count,) + (1,) * (s.dim() - 1)),
                )
                - 1.0
            )
            * torch.exp(
                (1.0 - retrievability)
                * self.weights[:, 14].view((self.user_count,) + (1,) * (s.dim() - 1))
            )
        )
        new_min = s / torch.exp(
            self.weights[:, 17].view((self.user_count,) + (1,) * (s.dim() - 1))
            * self.weights[:, 18].view((self.user_count,) + (1,) * (s.dim() - 1))
        )
        return torch.minimum(new_s, new_min)


class FSRS6BatchedContinuousStationaryFiniteOracle(FSRS6BatchedStationaryFiniteOracle):
    ACTION_POLICY_LOOKUP_VERSION = "terminal_retention_min_action_v2"
    STATIONARY_POLICY_ITERATION_VERSION = "continuous_interval_greedy_v3"
    STATIONARY_IMPROVE_STATE_BLOCK_SIZE = 4096
    STATIONARY_SOLVE_WEIGHT_BLOCK_SIZE = 4

    def __init__(
        self,
        *,
        days: int,
        s_grid_size: int,
        d_grid_size: int,
        retention_min: float,
        retention_max: float,
        interval_chunk_size: int,
        fsrs_weights: Sequence[Sequence[float]],
        first_rating_prob: Sequence[Sequence[float]],
        review_rating_prob: Sequence[Sequence[float]],
        learning_costs: Sequence[Sequence[float]],
        review_costs: Sequence[Sequence[float]],
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        cache_config: OracleDPCacheConfig | None = None,
        progress_log_interval_seconds: float = 0.0,
    ) -> None:
        validate_continuous_retention_bounds(retention_min, retention_max)
        if interval_chunk_size <= 0:
            raise ValueError("interval_chunk_size must be > 0.")
        if progress_log_interval_seconds < 0.0:
            raise ValueError("progress_log_interval_seconds must be >= 0.")
        super().__init__(
            days=days,
            action_retentions=[retention_max],
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            fsrs_weights=fsrs_weights,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_costs=learning_costs,
            review_costs=review_costs,
            dtype=dtype,
            device=device,
            cache_config=cache_config,
        )
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        self.interval_chunk_size = int(interval_chunk_size)
        self.progress_log_interval_seconds = float(progress_log_interval_seconds)
        self._progress_start_s = time.perf_counter()

    def _progress_logging_enabled(self) -> bool:
        return self.progress_log_interval_seconds > 0.0

    def _progress_log(self, message: str) -> None:
        if not self._progress_logging_enabled():
            return
        elapsed_s = time.perf_counter() - self._progress_start_s
        print(f"[continuous-oracle +{elapsed_s:.1f}s] {message}", flush=True)

    def _next_progress_log_deadline(self) -> float:
        return time.perf_counter() + self.progress_log_interval_seconds

    def _progress_log_due(self, deadline: float) -> bool:
        return self._progress_logging_enabled() and time.perf_counter() >= deadline

    def _cache_key_parts(
        self,
        *,
        oracle_kind: str,
        method: str,
        user_idx: int,
        cost_weight: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "algorithm_version": 1,
            "oracle_kind": oracle_kind,
            "method": method,
            "days": self.days,
            "s_grid_size": self.s_count,
            "d_grid_size": self.d_count,
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "interval_chunk_size": self.interval_chunk_size,
            "dtype": str(self.dtype),
            "user_config": self._user_payload(user_idx),
            "cost_weight": float(cost_weight),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def _continuous_cache_extra(self) -> dict[str, Any]:
        return {
            "transition_value_lookup": FSRS6IntervalOracle.TRANSITION_VALUE_LOOKUP_VERSION,
            "action_policy_lookup": self.ACTION_POLICY_LOOKUP_VERSION,
        }

    def _stationary_cache_extra(
        self,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        extra = self._continuous_cache_extra()
        extra["policy_iteration"] = self.STATIONARY_POLICY_ITERATION_VERSION
        if max_iterations is not None:
            extra["max_iterations"] = max_iterations
        if tolerance is not None:
            extra["tolerance"] = tolerance
        return extra

    def _suboracle(
        self,
        user_indices: Sequence[int],
        *,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> "FSRS6BatchedContinuousStationaryFiniteOracle":
        return FSRS6BatchedContinuousStationaryFiniteOracle(
            days=self.days,
            s_grid_size=self.s_count,
            d_grid_size=self.d_count,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            interval_chunk_size=self.interval_chunk_size,
            fsrs_weights=[
                _tensor_float_list(self.weights[idx]) for idx in user_indices
            ],
            first_rating_prob=[
                _tensor_float_list(self.first_rating_prob[idx]) for idx in user_indices
            ],
            review_rating_prob=[
                _tensor_float_list(self.review_rating_prob[idx]) for idx in user_indices
            ],
            learning_costs=[
                [
                    60.0 * value
                    for value in _tensor_float_list(self.learning_cost_minutes[idx])
                ]
                for idx in user_indices
            ],
            review_costs=[
                [
                    60.0 * value
                    for value in _tensor_float_list(self.review_cost_minutes[idx])
                ]
                for idx in user_indices
            ],
            dtype=self.dtype,
            device=self.device,
            cache_config=cache_config or OracleDPCacheConfig(enabled=False),
            progress_log_interval_seconds=self.progress_log_interval_seconds,
        )

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        weight_list = [float(weight) for weight in cost_weights]
        policies: list[list[torch.Tensor | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        missing_by_user: dict[int, list[int]] = {}
        for user_idx in range(self.user_count):
            for weight_idx, weight in enumerate(weight_list):
                entry = load_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_retention",
                        method="solve_policies",
                        user_idx=user_idx,
                        cost_weight=weight,
                        extra=self._continuous_cache_extra(),
                    ),
                    map_location=self.device,
                )
                if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                    missing_by_user.setdefault(user_idx, []).append(weight_idx)
                    continue
                policies[user_idx][weight_idx] = entry["policy"].to(
                    device=self.device,
                    dtype=self.dtype,
                )

        groups: dict[tuple[int, ...], list[int]] = {}
        for user_idx, missing_weight_indices in missing_by_user.items():
            groups.setdefault(tuple(missing_weight_indices), []).append(user_idx)
        missing_pairs = sum(len(indices) for indices in missing_by_user.values())
        self._progress_log(
            "finite cache scan "
            f"users={self.user_count} weights={len(weight_list)} "
            f"hits={self.user_count * len(weight_list) - missing_pairs} "
            f"missing={missing_pairs} groups={len(groups)}"
        )
        for missing_weight_indices, user_indices in groups.items():
            suboracle = self._suboracle(user_indices)
            group_weights = [weight_list[idx] for idx in missing_weight_indices]
            self._progress_log(
                "finite uncached group start "
                f"users={list(user_indices)} weights={group_weights}"
            )
            group_start_s = time.perf_counter()
            computed = suboracle._solve_continuous_policies_uncached(
                group_weights,
                progress=progress,
            )
            self._progress_log(
                "finite uncached group solved "
                f"users={list(user_indices)} weights={group_weights} "
                f"runtime_s={time.perf_counter() - group_start_s:.1f}"
            )
            for local_user_idx, user_idx in enumerate(user_indices):
                for local_weight_idx, weight_idx in enumerate(missing_weight_indices):
                    policy = computed[local_user_idx, local_weight_idx].to(
                        device=self.device,
                        dtype=self.dtype,
                    )
                    policies[user_idx][weight_idx] = policy
                    write_cache_entry(
                        self.cache_config,
                        key_parts=self._cache_key_parts(
                            oracle_kind="continuous_retention",
                            method="solve_policies",
                            user_idx=user_idx,
                            cost_weight=weight_list[weight_idx],
                            extra=self._continuous_cache_extra(),
                        ),
                        data={"policy": policy},
                    )
            self._progress_log(
                "finite cache write done "
                f"users={list(user_indices)} weights={group_weights} "
                f"entries={len(user_indices) * len(missing_weight_indices)} "
                f"cache_enabled={self.cache_config.enabled}"
            )

        return torch.stack(
            [
                torch.stack(
                    [policy for policy in user_policies if policy is not None],
                    dim=0,
                )
                for user_policies in policies
            ],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)

    def solve_policies_for_user_weights(
        self,
        cost_weights_by_user: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_list = self._user_cost_weight_list(cost_weights_by_user)
        policies: list[torch.Tensor | None] = [None for _ in range(self.user_count)]
        missing_user_indices: list[int] = []
        for user_idx, weight in enumerate(weight_list):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_retention",
                    method="solve_policies",
                    user_idx=user_idx,
                    cost_weight=weight,
                    extra=self._continuous_cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing_user_indices.append(user_idx)
                continue
            policies[user_idx] = entry["policy"].to(
                device=self.device,
                dtype=self.dtype,
            )

        missing_pairs = len(missing_user_indices)
        self._progress_log(
            "finite jagged cache scan "
            f"users={self.user_count} pairs={self.user_count} "
            f"hits={self.user_count - missing_pairs} missing={missing_pairs}"
        )
        if missing_user_indices:
            suboracle = self._suboracle(missing_user_indices)
            group_weights = [weight_list[idx] for idx in missing_user_indices]
            self._progress_log(
                "finite jagged uncached start "
                f"users={list(missing_user_indices)} weights={group_weights}"
            )
            group_start_s = time.perf_counter()
            computed = suboracle._solve_continuous_policies_uncached(
                torch.tensor(
                    group_weights,
                    device=self.device,
                    dtype=self.dtype,
                ).view(len(group_weights), 1),
                progress=progress,
            )
            self._progress_log(
                "finite jagged uncached solved "
                f"users={list(missing_user_indices)} weights={group_weights} "
                f"runtime_s={time.perf_counter() - group_start_s:.1f}"
            )
            for local_user_idx, user_idx in enumerate(missing_user_indices):
                policy = computed[local_user_idx, 0].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                policies[user_idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_retention",
                        method="solve_policies",
                        user_idx=user_idx,
                        cost_weight=weight_list[user_idx],
                        extra=self._continuous_cache_extra(),
                    ),
                    data={"policy": policy},
                )
            self._progress_log(
                "finite jagged cache write done "
                f"entries={len(missing_user_indices)} "
                f"cache_enabled={self.cache_config.enabled}"
            )

        return (
            torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            )
            .to(device=self.device, dtype=self.dtype)
            .reshape(
                self.user_count,
                1,
                self.horizon + 1,
                self.s_count,
                self.d_count,
            )
        )

    def solve_stationary_finite_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        weight_list = [float(weight) for weight in cost_weights]
        policies: list[list[torch.Tensor | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        metrics: list[list[OracleMetrics | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        objectives = torch.zeros(
            (self.user_count, len(weight_list)),
            device=self.device,
            dtype=self.dtype,
        )
        iterations: list[list[int | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        converged: list[list[bool | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        residuals: list[list[float | None]] = [
            [None for _ in weight_list] for _ in range(self.user_count)
        ]
        missing_by_user: dict[int, list[int]] = {}
        extra = self._stationary_cache_extra(
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        for user_idx in range(self.user_count):
            for weight_idx, weight in enumerate(weight_list):
                entry = load_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_stationary_finite",
                        method="solve_stationary_finite_policies",
                        user_idx=user_idx,
                        cost_weight=weight,
                        extra=extra,
                    ),
                    map_location=self.device,
                )
                if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                    missing_by_user.setdefault(user_idx, []).append(weight_idx)
                    continue
                policies[user_idx][weight_idx] = entry["policy"].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                metrics[user_idx][weight_idx] = _metrics_from_payload(
                    entry["metrics"],
                    runtime_s=0.0,
                )
                objectives[user_idx, weight_idx] = float(entry["objective"])
                iterations[user_idx][weight_idx] = int(entry["iterations"])
                converged[user_idx][weight_idx] = bool(entry["converged"])
                residuals[user_idx][weight_idx] = float(entry["residual"])

        groups: dict[tuple[int, ...], list[int]] = {}
        for user_idx, missing_weight_indices in missing_by_user.items():
            groups.setdefault(tuple(missing_weight_indices), []).append(user_idx)
        missing_pairs = sum(len(indices) for indices in missing_by_user.values())
        self._progress_log(
            "stationary cache scan "
            f"users={self.user_count} weights={len(weight_list)} "
            f"hits={self.user_count * len(weight_list) - missing_pairs} "
            f"missing={missing_pairs} groups={len(groups)}"
        )
        for missing_weight_indices, user_indices in groups.items():
            suboracle = self._suboracle(user_indices, cache_config=self.cache_config)
            weight_block_size = max(1, int(self.STATIONARY_SOLVE_WEIGHT_BLOCK_SIZE))
            for block_start in range(
                0,
                len(missing_weight_indices),
                weight_block_size,
            ):
                block_weight_indices = missing_weight_indices[
                    block_start : block_start + weight_block_size
                ]
                group_weights = [weight_list[idx] for idx in block_weight_indices]
                self._progress_log(
                    "stationary uncached group start "
                    f"users={list(user_indices)} weights={group_weights}"
                )
                group_start_s = time.perf_counter()
                solution = suboracle._solve_stationary_finite_policies_uncached(
                    group_weights,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    progress=progress,
                )
                self._progress_log(
                    "stationary uncached group solved "
                    f"users={list(user_indices)} weights={group_weights} "
                    f"runtime_s={time.perf_counter() - group_start_s:.1f}"
                )
                for local_user_idx, user_idx in enumerate(user_indices):
                    for local_weight_idx, weight_idx in enumerate(block_weight_indices):
                        policy = solution.policy[local_user_idx, local_weight_idx].to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                        metric = solution.metrics[local_user_idx][local_weight_idx]
                        objective = float(
                            solution.objectives[
                                local_user_idx,
                                local_weight_idx,
                            ].item()
                        )
                        iteration = solution.iterations[local_user_idx][
                            local_weight_idx
                        ]
                        did_converge = solution.converged[local_user_idx][
                            local_weight_idx
                        ]
                        residual = solution.residuals[local_user_idx][local_weight_idx]
                        policies[user_idx][weight_idx] = policy
                        metrics[user_idx][weight_idx] = metric
                        objectives[user_idx, weight_idx] = objective
                        iterations[user_idx][weight_idx] = iteration
                        converged[user_idx][weight_idx] = did_converge
                        residuals[user_idx][weight_idx] = residual
                        write_cache_entry(
                            self.cache_config,
                            key_parts=self._cache_key_parts(
                                oracle_kind="continuous_stationary_finite",
                                method="solve_stationary_finite_policies",
                                user_idx=user_idx,
                                cost_weight=weight_list[weight_idx],
                                extra=extra,
                            ),
                            data={
                                "policy": policy,
                                "metrics": _metrics_payload(metric),
                                "objective": objective,
                                "iterations": iteration,
                                "converged": did_converge,
                                "residual": residual,
                            },
                        )
                del solution
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return BatchedStationaryFiniteOracleSolution(
            policy=torch.stack(
                [
                    torch.stack(
                        [policy for policy in user_policies if policy is not None],
                        dim=0,
                    )
                    for user_policies in policies
                ],
                dim=0,
            )
            .to(device=self.device, dtype=self.dtype)
            .reshape(self.user_count, len(weight_list), self.s_count, self.d_count),
            metrics=[
                [metric for metric in user_metrics if metric is not None]
                for user_metrics in metrics
            ],
            objectives=objectives,
            iterations=[
                [int(value) for value in row if value is not None] for row in iterations
            ],
            converged=[
                [bool(value) for value in row if value is not None] for row in converged
            ],
            residuals=[
                [float(value) for value in row if value is not None]
                for row in residuals
            ],
            runtime_s=time.perf_counter() - start,
        )

    def solve_stationary_finite_policies_for_user_weights(
        self,
        cost_weights_by_user: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        weight_list = self._user_cost_weight_list(cost_weights_by_user)
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in range(self.user_count)]
        metrics: list[OracleMetrics | None] = [None for _ in range(self.user_count)]
        objectives = torch.zeros(
            (self.user_count, 1),
            device=self.device,
            dtype=self.dtype,
        )
        iterations: list[int | None] = [None for _ in range(self.user_count)]
        converged: list[bool | None] = [None for _ in range(self.user_count)]
        residuals: list[float | None] = [None for _ in range(self.user_count)]
        missing_user_indices: list[int] = []
        extra = self._stationary_cache_extra(
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        for user_idx, weight in enumerate(weight_list):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_stationary_finite",
                    method="solve_stationary_finite_policies",
                    user_idx=user_idx,
                    cost_weight=weight,
                    extra=extra,
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing_user_indices.append(user_idx)
                continue
            policies[user_idx] = entry["policy"].to(
                device=self.device,
                dtype=self.dtype,
            )
            metrics[user_idx] = _metrics_from_payload(
                entry["metrics"],
                runtime_s=0.0,
            )
            objectives[user_idx, 0] = float(entry["objective"])
            iterations[user_idx] = int(entry["iterations"])
            converged[user_idx] = bool(entry["converged"])
            residuals[user_idx] = float(entry["residual"])

        missing_pairs = len(missing_user_indices)
        self._progress_log(
            "stationary jagged cache scan "
            f"users={self.user_count} pairs={self.user_count} "
            f"hits={self.user_count - missing_pairs} missing={missing_pairs}"
        )
        if missing_user_indices:
            suboracle = self._suboracle(
                missing_user_indices,
                cache_config=self.cache_config,
            )
            group_weights = [weight_list[idx] for idx in missing_user_indices]
            self._progress_log(
                "stationary jagged uncached start "
                f"users={list(missing_user_indices)} weights={group_weights}"
            )
            group_start_s = time.perf_counter()
            solution = suboracle._solve_stationary_finite_policies_uncached(
                torch.tensor(
                    group_weights,
                    device=self.device,
                    dtype=self.dtype,
                ).view(len(group_weights), 1),
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            self._progress_log(
                "stationary jagged uncached solved "
                f"users={list(missing_user_indices)} weights={group_weights} "
                f"runtime_s={time.perf_counter() - group_start_s:.1f}"
            )
            for local_user_idx, user_idx in enumerate(missing_user_indices):
                policy = solution.policy[local_user_idx, 0].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                metric = solution.metrics[local_user_idx][0]
                objective = float(solution.objectives[local_user_idx, 0].item())
                iteration = solution.iterations[local_user_idx][0]
                did_converge = solution.converged[local_user_idx][0]
                residual = solution.residuals[local_user_idx][0]
                policies[user_idx] = policy
                metrics[user_idx] = metric
                objectives[user_idx, 0] = objective
                iterations[user_idx] = iteration
                converged[user_idx] = did_converge
                residuals[user_idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_stationary_finite",
                        method="solve_stationary_finite_policies",
                        user_idx=user_idx,
                        cost_weight=weight_list[user_idx],
                        extra=extra,
                    ),
                    data={
                        "policy": policy,
                        "metrics": _metrics_payload(metric),
                        "objective": objective,
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return BatchedStationaryFiniteOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            )
            .to(device=self.device, dtype=self.dtype)
            .reshape(self.user_count, 1, self.s_count, self.d_count),
            metrics=[[metric] for metric in metrics if metric is not None],
            objectives=objectives,
            iterations=[[int(value)] for value in iterations if value is not None],
            converged=[[bool(value)] for value in converged if value is not None],
            residuals=[[float(value)] for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def _solve_continuous_policies_uncached(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_tensor = self._cost_weight_tensor(cost_weights)
        weight_grid = self._cost_weight_grid(weight_tensor)
        weight_count = int(weight_grid.shape[1])
        value = torch.zeros(
            (self.user_count, weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        policy = torch.full(
            (self.user_count, weight_count, self.horizon + 1, self.state_count),
            self.retention_max,
            device=self.device,
            dtype=self.dtype,
        )

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=(
                    "Continuous retention oracle "
                    f"users={self.user_count} weights={weight_count}"
                ),
                unit="day",
                leave=False,
            )
        phase_start_s = time.perf_counter()
        next_log_s = self._next_progress_log_deadline()
        self._progress_log(
            "finite DP start "
            f"users={self.user_count} weights={weight_count} "
            f"horizon={self.horizon} states={self.state_count} "
            f"chunk={self.interval_chunk_size}"
        )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full(
                    (self.user_count, weight_count, self.s_count, self.d_count),
                    -math.inf,
                    device=self.device,
                    dtype=self.dtype,
                )
                best_retention = torch.full_like(best_value, self.retention_max)
                for start in range(1, rem + 2, self.interval_chunk_size):
                    stop = min(rem + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    candidate = self._candidate_interval_value_batch(
                        intervals=intervals,
                        rem=rem,
                        cost_weights=weight_grid,
                        value=value,
                    )
                    mask = self._attainable_interval_mask(intervals, rem + 1)
                    candidate = torch.where(
                        mask[:, None, :, :, None],
                        candidate,
                        torch.full_like(candidate, -math.inf),
                    )
                    chunk_best_value, chunk_best_idx = candidate.max(dim=2)
                    chunk_interval = intervals.index_select(
                        0,
                        chunk_best_idx.reshape(-1),
                    ).reshape_as(chunk_best_idx)
                    chunk_retention = self._retention_for_interval_grid_batch(
                        chunk_interval,
                        terminal_interval=rem + 1,
                    )
                    better = chunk_best_value > best_value
                    best_value = torch.where(better, chunk_best_value, best_value)
                    best_retention = torch.where(
                        better,
                        chunk_retention,
                        best_retention,
                    )
                    if self._progress_log_due(next_log_s):
                        self._progress_log(
                            "finite DP progress "
                            f"rem={rem}/{self.horizon} interval_chunk={start}-{stop - 1} "
                            f"elapsed_s={time.perf_counter() - phase_start_s:.1f}"
                        )
                        next_log_s = self._next_progress_log_deadline()
                value[:, :, rem, :] = best_value.reshape(
                    self.user_count,
                    weight_count,
                    self.state_count,
                )
                policy[:, :, rem, :] = best_retention.reshape(
                    self.user_count,
                    weight_count,
                    self.state_count,
                )
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        self._progress_log(
            "finite DP done "
            f"users={self.user_count} weights={weight_count} "
            f"runtime_s={time.perf_counter() - phase_start_s:.1f}"
        )

        return policy.reshape(
            self.user_count,
            weight_count,
            self.horizon + 1,
            self.s_count,
            self.d_count,
        )

    def _solve_stationary_finite_policies_uncached(
        self,
        cost_weights: Sequence[float] | torch.Tensor,
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> BatchedStationaryFiniteOracleSolution:
        start = time.perf_counter()
        weight_tensor = self._cost_weight_tensor(cost_weights)
        weight_count = self._cost_weight_count(weight_tensor)
        self._progress_log(
            "stationary solve start "
            f"users={self.user_count} weights={weight_count} "
            f"horizon={self.horizon} states={self.state_count} "
            f"max_iterations={max_iterations} tolerance={tolerance:g}"
        )
        finite_start_s = time.perf_counter()
        if weight_tensor.ndim == 1:
            finite_policies = self.solve_policies(
                _tensor_float_list(weight_tensor),
                progress=progress,
            )
        elif weight_count == 1:
            finite_policies = self.solve_policies_for_user_weights(
                _tensor_float_list(weight_tensor[:, 0]),
                progress=progress,
            )
        else:
            finite_policies = self._solve_continuous_policies_uncached(
                weight_tensor,
                progress=progress,
            )
        self._progress_log(
            "stationary finite initialization done "
            f"runtime_s={time.perf_counter() - finite_start_s:.1f}"
        )
        policy = torch.clamp(
            finite_policies[:, :, self.horizon],
            min=self.retention_min,
            max=self.retention_max,
        ).contiguous()
        eval_start_s = time.perf_counter()
        value = self._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=weight_tensor,
        )
        objective = self._objective_from_value_batch(
            value=value,
            cost_weights=weight_tensor,
        )
        if self._progress_logging_enabled():
            self._progress_log(
                "stationary initial policy evaluated "
                f"runtime_s={time.perf_counter() - eval_start_s:.1f} "
                f"objective_min={float(objective.min().item()):.6g} "
                f"objective_max={float(objective.max().item()):.6g}"
            )
        iterations = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=torch.int64,
        )
        converged = torch.zeros_like(iterations, dtype=torch.bool)
        residuals = torch.full(
            iterations.shape,
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones_like(converged)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.user_count * weight_count * max_iterations,
                desc="Continuous stationary finite oracle",
                unit="iter",
                leave=False,
            )
        try:
            for iteration in range(1, max_iterations + 1):
                if not bool(active.any().item()):
                    break
                iteration_start_s = time.perf_counter()
                log_iteration = self._progress_logging_enabled()
                active_before = int(active.sum().item()) if log_iteration else 0
                occupancy_start_s = time.perf_counter()
                occupancy = self._rollout_occupancy_batch(policy=policy)
                occupancy_runtime_s = time.perf_counter() - occupancy_start_s
                improve_start_s = time.perf_counter()
                new_policy, residual, visited = self._improve_stationary_policy_batch(
                    policy=policy,
                    occupancy=occupancy,
                    value=value,
                    cost_weights=weight_tensor,
                    active=active,
                    iteration=iteration,
                )
                improve_runtime_s = time.perf_counter() - improve_start_s
                changed = (torch.abs(new_policy - policy) > 1e-12) & visited
                policy_changed = changed.reshape(
                    self.user_count,
                    weight_count,
                    self.state_count,
                ).any(dim=2)
                iterations[active] = iteration
                residuals[active] = residual[active]
                if progress_bar is not None:
                    progress_bar.update(int(active.sum().item()))

                done = active & ((~policy_changed) | (residual <= tolerance))
                if bool(done.any().item()):
                    converged[done] = True
                    active[done] = False

                candidate = active & ~done
                if not bool(candidate.any().item()):
                    if log_iteration:
                        self._progress_log(
                            "stationary iteration done "
                            f"iter={iteration}/{max_iterations} "
                            f"active_before={active_before} "
                            f"done={int(done.sum().item())} "
                            f"remaining={int(active.sum().item())} "
                            f"max_residual={float(residual.max().item()):.6g} "
                            f"occupancy_s={occupancy_runtime_s:.1f} "
                            f"improve_s={improve_runtime_s:.1f} eval_s=0.0 "
                            f"iteration_s={time.perf_counter() - iteration_start_s:.1f}"
                        )
                    continue

                eval_start_s = time.perf_counter()
                candidate_value = self._evaluate_stationary_policy_value_batch(
                    policy=new_policy,
                    cost_weights=weight_tensor,
                )
                candidate_objective = self._objective_from_value_batch(
                    value=candidate_value,
                    cost_weights=weight_tensor,
                )
                eval_runtime_s = time.perf_counter() - eval_start_s
                objective_improvement = candidate_objective - objective
                accepted = candidate & (objective_improvement > tolerance)
                residuals[candidate] = torch.where(
                    accepted[candidate],
                    objective_improvement[candidate],
                    torch.clamp(objective_improvement[candidate], min=0.0),
                )
                if bool((candidate & ~accepted).any().item()):
                    rejected = candidate & ~accepted
                    converged[rejected] = True
                    active[rejected] = False
                if bool(accepted.any().item()):
                    policy[accepted] = new_policy[accepted]
                    value[accepted] = candidate_value[accepted]
                    objective[accepted] = candidate_objective[accepted]
                if log_iteration:
                    self._progress_log(
                        "stationary iteration done "
                        f"iter={iteration}/{max_iterations} "
                        f"active_before={active_before} done={int(done.sum().item())} "
                        f"accepted={int(accepted.sum().item())} "
                        f"rejected={int((candidate & ~accepted).sum().item())} "
                        f"remaining={int(active.sum().item())} "
                        f"max_residual={float(residual.max().item()):.6g} "
                        f"max_objective_delta="
                        f"{float(objective_improvement.max().item()):.6g} "
                        f"occupancy_s={occupancy_runtime_s:.1f} "
                        f"improve_s={improve_runtime_s:.1f} "
                        f"eval_s={eval_runtime_s:.1f} "
                        f"iteration_s={time.perf_counter() - iteration_start_s:.1f}"
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if bool(active.any().item()):
            iterations[active] = max_iterations

        metrics = self._metrics_from_occupancy_batch(
            policy=policy,
            cost_weights=weight_tensor,
        )
        if self._progress_logging_enabled():
            self._progress_log(
                "stationary solve done "
                f"runtime_s={time.perf_counter() - start:.1f} "
                f"converged={int(converged.sum().item())}/{converged.numel()}"
            )
        return BatchedStationaryFiniteOracleSolution(
            policy=policy,
            metrics=metrics,
            objectives=torch.tensor(
                [[metric.scalar_objective for metric in row] for row in metrics],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[
                [int(value) for value in row.tolist()] for row in iterations.cpu()
            ],
            converged=[
                [bool(value) for value in row.tolist()] for row in converged.cpu()
            ],
            residuals=[
                [float(value) for value in row.tolist()] for row in residuals.cpu()
            ],
            runtime_s=time.perf_counter() - start,
        )

    def _candidate_interval_value_batch(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        interval, prob, next_idx, next_weight = self._interval_tables_batch(intervals)
        return self._candidate_interval_value_from_tables(
            intervals=intervals,
            rem=rem,
            cost_weights=cost_weights,
            value=value,
            interval=interval,
            prob=prob,
            next_idx=next_idx,
            next_weight=next_weight,
        )

    def _candidate_interval_value_from_tables(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
        interval: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        next_weight: torch.Tensor,
    ) -> torch.Tensor:
        interval_count = int(intervals.numel())
        weight_count = int(value.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        active_days = torch.minimum(intervals, torch.full_like(intervals, rem))
        immediate_mem = self._memorized_sum_interval_candidates(active_days)
        candidate_flat = (
            immediate_mem[:, None, :, :, None]
            .expand(
                self.user_count,
                weight_count,
                interval_count,
                self.s_count,
                self.d_count,
            )
            .clone()
            .reshape(self.user_count, weight_count, interval_count, self.state_count)
        )

        cont_mask = interval <= rem
        if not bool(cont_mask.any().item()):
            return candidate_flat.reshape(
                self.user_count,
                weight_count,
                interval_count,
                self.s_count,
                self.d_count,
            )

        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        user_idx = self._user_index_view(4).expand(
            self.user_count,
            weight_count,
            interval_count,
            self.state_count,
        )
        weight_idx = self._weight_index_view(4, weight_count).expand_as(user_idx)
        future_rem_exp = future_rem[:, None, :, :].expand_as(user_idx)
        cont_weight = cont_mask.to(dtype=self.dtype)[:, None, :, :]
        cost_weight = cost_weight_grid[:, :, None, None]
        for rating_idx, rating in enumerate(range(1, 5)):
            weighted = prob[:, None, :, rating_idx, :] * cont_weight
            future_value = torch.zeros_like(candidate_flat)
            for corner_idx in range(4):
                future_value += (
                    next_weight[:, None, :, rating_idx, corner_idx, :]
                    * value[
                        user_idx,
                        weight_idx,
                        future_rem_exp,
                        next_idx[:, None, :, rating_idx, corner_idx, :].expand_as(
                            user_idx
                        ),
                    ]
                )
            review_minutes = self.review_cost_minutes[:, rating - 1].view(
                self.user_count,
                1,
                1,
                1,
            )
            candidate_flat += weighted * (future_value - cost_weight * review_minutes)
        return candidate_flat.reshape(
            self.user_count,
            weight_count,
            interval_count,
            self.s_count,
            self.d_count,
        )

    def _interval_tables_batch(
        self,
        intervals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interval_count = int(intervals.numel())
        interval = intervals.view(1, interval_count, 1, 1).expand(
            self.user_count,
            interval_count,
            self.s_count,
            self.d_count,
        )
        elapsed = interval.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(
            elapsed,
            self.s_mesh.view(1, 1, self.s_count, self.d_count),
        )
        prob = torch.stack(
            [
                1.0 - retrievability,
                retrievability
                * self.review_rating_prob[:, 0].view(self.user_count, 1, 1, 1),
                retrievability
                * self.review_rating_prob[:, 1].view(self.user_count, 1, 1, 1),
                retrievability
                * self.review_rating_prob[:, 2].view(self.user_count, 1, 1, 1),
            ],
            dim=2,
        ).reshape(self.user_count, interval_count, 4, self.state_count)

        next_indices: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        s = self.s_mesh.view(1, 1, self.s_count, self.d_count)
        d = self.d_mesh.view(1, 1, self.s_count, self.d_count)
        for rating in range(1, 5):
            rating_tensor = torch.full(
                (self.user_count, interval_count, self.s_count, self.d_count),
                rating,
                device=self.device,
                dtype=torch.int64,
            )
            if rating > 1:
                new_s = self._stability_after_success(
                    s,
                    retrievability,
                    d,
                    rating_tensor,
                )
            else:
                new_s = self._stability_after_failure(s, retrievability, d)
            new_d = self._next_d(d, rating_tensor)
            kernel_idx, kernel_weight = self._state_kernel(new_s, new_d)
            next_indices.append(kernel_idx)
            next_weights.append(kernel_weight)

        next_idx = torch.stack(next_indices, dim=2).permute(1, 3, 2, 0, 4, 5)
        next_weight = torch.stack(next_weights, dim=2).permute(1, 3, 2, 0, 4, 5)
        return (
            interval.reshape(self.user_count, interval_count, self.state_count),
            prob,
            next_idx.reshape(self.user_count, interval_count, 4, 4, self.state_count),
            next_weight.reshape(
                self.user_count,
                interval_count,
                4,
                4,
                self.state_count,
            ).to(dtype=self.dtype),
        )

    def _interpolate_interval_value_batch(
        self,
        *,
        value: torch.Tensor,
        rem_idx: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(value.shape[1])
        interval_count = int(rem_idx.numel())
        state_idx, state_weight = self._state_kernel(s, d)
        user_idx = self._user_index_view(5).expand(
            self.user_count,
            weight_count,
            interval_count,
            self.s_count,
            self.d_count,
        )
        weight_idx = self._weight_index_view(5, weight_count).expand_as(user_idx)
        rem_exp = rem_idx.view(1, 1, interval_count, 1, 1).expand_as(user_idx)
        future = torch.zeros(
            (
                self.user_count,
                weight_count,
                interval_count,
                self.s_count,
                self.d_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        for corner_idx in range(4):
            future += (
                state_weight[corner_idx][:, None, :, :, :]
                * value[
                    user_idx,
                    weight_idx,
                    rem_exp,
                    state_idx[corner_idx][:, None, :, :, :].expand_as(user_idx),
                ]
            )
        return future

    def _memorized_sum_interval_candidates(
        self,
        days: torch.Tensor,
    ) -> torch.Tensor:
        interval_count = int(days.numel())
        user_idx = self._user_index_view(3).expand(
            self.user_count,
            interval_count,
            self.s_count,
        )
        day_idx = days.view(1, interval_count, 1).expand_as(user_idx)
        s_idx = torch.arange(
            self.s_count,
            device=self.device,
            dtype=torch.int64,
        ).view(1, 1, self.s_count)
        return self.memorized_by_day[
            user_idx,
            day_idx,
            s_idx.expand_as(user_idx),
        ]

    def _attainable_interval_mask(
        self,
        intervals: torch.Tensor,
        terminal_interval: int,
    ) -> torch.Tensor:
        s = self.s_grid.view(1, self.s_count).expand(self.user_count, self.s_count)
        lower_float, upper_float = retention_interval_bounds(
            s=s,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            factor=self.factor.view(self.user_count, 1),
            decay=self.decay.view(self.user_count, 1),
        )
        rounded_lower = torch.clamp(torch.round(lower_float), min=1.0).to(torch.int64)
        rounded_upper = torch.clamp(torch.round(upper_float), min=1.0).to(torch.int64)
        lo = torch.minimum(rounded_lower, rounded_upper)
        hi = torch.maximum(rounded_lower, rounded_upper)
        interval_col = intervals.to(device=self.device, dtype=torch.int64).view(
            1,
            int(intervals.numel()),
            1,
        )
        mask = (interval_col >= lo[:, None, :]) & (interval_col <= hi[:, None, :])
        terminal = torch.as_tensor(
            terminal_interval,
            device=self.device,
            dtype=torch.int64,
        )
        terminal_mask = (interval_col == terminal) & (hi[:, None, :] >= terminal)
        mask = torch.where(interval_col == terminal, terminal_mask, mask)
        return mask & (interval_col <= terminal)

    def _retention_for_interval_grid_batch(
        self,
        interval: torch.Tensor,
        *,
        terminal_interval: int | None = None,
    ) -> torch.Tensor:
        s = self.s_grid.view(1, 1, self.s_count, 1).expand_as(interval)
        retention = self._forgetting_curve(interval.to(dtype=self.dtype), s)
        retention = torch.clamp(
            retention,
            min=self.retention_min,
            max=self.retention_max,
        )
        if terminal_interval is not None:
            terminal = interval.to(dtype=torch.int64) >= int(terminal_interval)
            retention = torch.where(
                terminal,
                torch.full_like(retention, self.retention_min),
                retention,
            )
        return retention

    def _intervals_for_retention_policy(self, policy: torch.Tensor) -> torch.Tensor:
        retention = torch.clamp(policy, min=1e-7, max=1.0 - 1e-7)
        retention_factor = (
            torch.pow(
                retention,
                1.0 / self.decay.view(self.user_count, 1, 1, 1),
            )
            - 1.0
        )
        interval = (
            self.s_mesh.view(1, 1, self.s_count, self.d_count)
            / self.factor.view(self.user_count, 1, 1, 1)
            * retention_factor
        )
        return torch.clamp(
            torch.round(interval),
            min=1.0,
            max=float(self.horizon + 1),
        ).to(torch.int64)

    def _next_state_interval_candidates(
        self,
        *,
        elapsed: torch.Tensor,
        retrievability: torch.Tensor,
        rating: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rating_tensor = torch.full(
            (self.user_count, int(elapsed.numel()), self.s_count, self.d_count),
            rating,
            device=self.device,
            dtype=torch.int64,
        )
        s = self.s_mesh.view(1, 1, self.s_count, self.d_count).expand_as(retrievability)
        d = self.d_mesh.view(1, 1, self.s_count, self.d_count).expand_as(retrievability)
        if rating > 1:
            new_s = self._stability_after_success(
                s,
                retrievability,
                d,
                rating_tensor,
            )
        else:
            new_s = self._stability_after_failure(s, retrievability, d)
        new_d = self._next_d(d, rating_tensor)
        return (
            torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max),
            torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max),
        )

    def _policy_tables_batch(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy = policy.reshape(
            self.user_count,
            int(policy.shape[1]),
            self.s_count,
            self.d_count,
        )
        weight_count = int(policy.shape[1])
        interval = self._intervals_for_retention_policy(policy)
        s = self.s_mesh.view(1, 1, self.s_count, self.d_count).expand_as(policy)
        d = self.d_mesh.view(1, 1, self.s_count, self.d_count).expand_as(policy)
        elapsed = interval.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(elapsed, s)
        prob = torch.stack(
            [
                1.0 - retrievability,
                retrievability
                * self.review_rating_prob[:, 0].view(self.user_count, 1, 1, 1),
                retrievability
                * self.review_rating_prob[:, 1].view(self.user_count, 1, 1, 1),
                retrievability
                * self.review_rating_prob[:, 2].view(self.user_count, 1, 1, 1),
            ],
            dim=2,
        ).reshape(self.user_count, weight_count, 4, self.state_count)

        next_indices: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        for rating in range(1, 5):
            rating_tensor = torch.full(
                (self.user_count, weight_count, self.s_count, self.d_count),
                rating,
                device=self.device,
                dtype=torch.int64,
            )
            if rating > 1:
                new_s = self._stability_after_success(
                    s,
                    retrievability,
                    d,
                    rating_tensor,
                )
            else:
                new_s = self._stability_after_failure(s, retrievability, d)
            new_d = self._next_d(d, rating_tensor)
            kernel_idx, kernel_weight = self._state_kernel(new_s, new_d)
            next_indices.append(kernel_idx)
            next_weights.append(kernel_weight)

        next_idx = torch.stack(next_indices, dim=2).permute(1, 3, 2, 0, 4, 5)
        next_weight = torch.stack(next_weights, dim=2).permute(1, 3, 2, 0, 4, 5)
        return (
            interval.reshape(self.user_count, weight_count, self.state_count),
            prob,
            next_idx.reshape(self.user_count, weight_count, 4, 4, self.state_count).to(
                dtype=torch.int64
            ),
            next_weight.reshape(
                self.user_count,
                weight_count,
                4,
                4,
                self.state_count,
            ).to(dtype=self.dtype),
        )

    def _rollout_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        stationary: bool = True,
    ) -> torch.Tensor:
        if not stationary:
            raise ValueError("continuous batched occupancy requires stationary=True.")
        weight_count = int(policy.shape[1])
        occupancy = torch.zeros(
            (
                self.user_count,
                weight_count,
                self.horizon + 1,
                self.state_count,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        batch_offset = (
            torch.arange(
                self.user_count * weight_count,
                device=self.device,
                dtype=torch.int64,
            )
            .view(self.user_count, weight_count, 1)
            .mul((self.horizon + 1) * self.state_count)
        )

        for rating in range(1, 5):
            prob = self.first_rating_prob[:, rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            state_idx, state_weight = self._state_kernel(s0, d0)
            for corner_idx in range(4):
                target = (
                    batch_offset
                    + self.horizon * self.state_count
                    + state_idx[corner_idx].view(self.user_count, 1, 1)
                )
                amount = (prob[:, None] * state_weight[corner_idx][:, None]).expand(
                    self.user_count,
                    weight_count,
                )
                occupancy.reshape(-1).scatter_add_(
                    0,
                    target.reshape(-1),
                    amount.reshape(-1),
                )

        flat_occupancy = occupancy.reshape(-1)
        selected_interval, selected_prob, selected_next_idx, selected_next_weight = (
            self._policy_tables_batch(policy)
        )
        for rem in range(self.horizon, 0, -1):
            current = occupancy[:, :, rem, :]
            if not bool(current.sum().item()):
                continue
            cont_mask = selected_interval <= rem
            if not bool(cont_mask.any().item()):
                continue
            source = current * cont_mask.to(dtype=self.dtype)
            if not bool(source.sum().item()):
                continue
            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            base = batch_offset + future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * selected_prob[:, :, rating_idx, :]
                        * selected_next_weight[:, :, rating_idx, corner_idx, :]
                    )
                    target = base + selected_next_idx[:, :, rating_idx, corner_idx, :]
                    flat_occupancy.scatter_add_(
                        0,
                        target.reshape(-1),
                        amount.reshape(-1),
                    )

        return occupancy

    def _evaluate_stationary_policy_value_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        value = torch.zeros(
            (self.user_count, weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        selected_interval, selected_prob, selected_next_idx, selected_next_weight = (
            self._policy_tables_batch(policy)
        )
        user_idx = self._user_index_view(3).expand_as(selected_interval)
        weight_idx = self._weight_index_view(3, weight_count).expand_as(
            selected_interval
        )
        weight_penalty = cost_weight_grid[:, :, None]
        for rem in range(1, self.horizon + 1):
            cont_mask = selected_interval <= rem
            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            active_days = torch.minimum(
                selected_interval,
                torch.full_like(selected_interval, rem),
            )
            value_rem = self._memorized_sum_batch(active_days).clone()
            if bool(cont_mask.any().item()):
                for rating_idx, rating in enumerate(range(1, 5)):
                    weighted = torch.where(
                        cont_mask,
                        selected_prob[:, :, rating_idx, :],
                        torch.zeros_like(selected_prob[:, :, rating_idx, :]),
                    )
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            selected_next_weight[
                                :,
                                :,
                                rating_idx,
                                corner_idx,
                                :,
                            ]
                            * value[
                                user_idx,
                                weight_idx,
                                future_rem,
                                selected_next_idx[:, :, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[:, rating - 1].view(
                        self.user_count,
                        1,
                        1,
                    )
                    value_rem += weighted * (
                        future_value - weight_penalty * review_minutes
                    )
            value[:, :, rem, :] = value_rem
        return value

    def _objective_from_value_batch(
        self,
        *,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(value.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        total_value = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=self.dtype,
        )
        total_learning_minutes = (
            self.first_rating_prob * self.learning_cost_minutes
        ).sum(dim=1)
        user_idx = self._user_index_view(2).expand(self.user_count, weight_count)
        weight_idx = self._weight_index_view(2, weight_count).expand(
            self.user_count,
            weight_count,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[:, rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            state_idx, state_weight = self._state_kernel(s0, d0)
            for corner_idx in range(4):
                total_value += (
                    prob[:, None]
                    * state_weight[corner_idx][:, None]
                    * value[
                        user_idx,
                        weight_idx,
                        self.horizon,
                        state_idx[corner_idx][:, None].expand(
                            self.user_count,
                            weight_count,
                        ),
                    ]
                )
        return (
            total_value - cost_weight_grid * total_learning_minutes[:, None]
        ) / float(self.days)

    def _active_cell_attainable_interval_mask(
        self,
        *,
        intervals: torch.Tensor,
        user_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> torch.Tensor:
        interval_count = int(intervals.numel())
        if int(user_idx.numel()) == 0:
            return torch.zeros(
                (0, interval_count),
                device=self.device,
                dtype=torch.bool,
            )
        s_idx = torch.div(state_idx, self.d_count, rounding_mode="floor")
        full_mask = self._attainable_interval_mask(intervals, self.horizon + 1)
        interval_idx = torch.arange(
            interval_count,
            device=self.device,
            dtype=torch.int64,
        )
        return full_mask[
            user_idx[:, None].expand(-1, interval_count),
            interval_idx[None, :].expand(int(user_idx.numel()), interval_count),
            s_idx[:, None].expand(-1, interval_count),
        ]

    def _active_cell_occupancy_block(
        self,
        *,
        occupancy: torch.Tensor,
        user_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> torch.Tensor:
        return occupancy[user_idx, :, :, state_idx]

    def _active_cell_policy_interval_block(
        self,
        *,
        current_interval: torch.Tensor,
        user_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> torch.Tensor:
        return current_interval[user_idx, :, state_idx]

    def _active_cell_immediate_prefix_tables(
        self,
        *,
        block_occupancy: torch.Tensor,
        user_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx = torch.div(state_idx, self.d_count, rounding_mode="floor")
        memorized = self.memorized_by_day[
            user_idx,
            :,
            s_idx,
        ]
        weight_count = int(block_occupancy.shape[1])
        if self.horizon <= 0:
            empty = torch.zeros(
                (int(user_idx.numel()), weight_count, 0),
                device=self.device,
                dtype=self.dtype,
            )
            totals = torch.zeros(
                (int(user_idx.numel()), weight_count),
                device=self.device,
                dtype=self.dtype,
            )
            return memorized, empty, empty, totals, totals
        occupancy_by_rem = block_occupancy[:, :, 1:]
        weighted_memorized = occupancy_by_rem * memorized[:, None, 1:]
        prefix_weighted = torch.cumsum(weighted_memorized, dim=2)
        prefix_occupancy = torch.cumsum(occupancy_by_rem, dim=2)
        return (
            memorized,
            prefix_weighted,
            prefix_occupancy,
            prefix_weighted[:, :, -1],
            prefix_occupancy[:, :, -1],
        )

    def _active_cell_immediate_scores_for_intervals(
        self,
        *,
        intervals: torch.Tensor,
        memorized: torch.Tensor,
        prefix_weighted: torch.Tensor,
        prefix_occupancy: torch.Tensor,
        total_weighted: torch.Tensor,
        total_occupancy: torch.Tensor,
    ) -> torch.Tensor:
        interval_count = int(intervals.numel())
        if self.horizon <= 0:
            return torch.zeros(
                (
                    int(memorized.shape[0]),
                    int(total_occupancy.shape[1]),
                    interval_count,
                ),
                device=self.device,
                dtype=self.dtype,
            )
        before_idx = torch.clamp(intervals - 2, min=0, max=self.horizon - 1)
        prefix_value = prefix_weighted.index_select(2, before_idx)
        prefix_count = prefix_occupancy.index_select(2, before_idx)
        has_prefix = (intervals > 1).view(1, 1, interval_count)
        prefix_value = torch.where(
            has_prefix,
            prefix_value,
            torch.zeros_like(prefix_value),
        )
        prefix_count = torch.where(
            has_prefix,
            prefix_count,
            torch.zeros_like(prefix_count),
        )
        memorized_idx = torch.clamp(intervals, max=self.horizon)
        memorized_at_interval = memorized.index_select(1, memorized_idx)
        score = (
            prefix_value
            + (total_occupancy[:, :, None] - prefix_count)
            * memorized_at_interval[:, None, :]
        )
        terminal = (intervals > self.horizon).view(1, 1, interval_count)
        return torch.where(
            terminal,
            total_weighted[:, :, None].expand_as(score),
            score,
        )

    def _active_cell_interval_tables(
        self,
        *,
        intervals: torch.Tensor,
        user_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = int(user_idx.numel())
        interval_count = int(intervals.numel())
        s_idx = torch.div(state_idx, self.d_count, rounding_mode="floor")
        d_idx = state_idx.remainder(self.d_count)
        weights = self.weights.index_select(0, user_idx)
        s = self.s_grid.index_select(0, s_idx).view(block_size, 1)
        d = self.d_grid.index_select(0, d_idx).view(block_size, 1)
        elapsed = intervals.to(device=self.device, dtype=self.dtype).view(
            1,
            interval_count,
        )
        factor = self.factor.index_select(0, user_idx).view(block_size, 1)
        decay = self.decay.index_select(0, user_idx).view(block_size, 1)
        retrievability = torch.pow(
            1.0 + factor * elapsed / torch.clamp(s, min=self.bounds.s_min),
            decay,
        )
        review_prob = self.review_rating_prob.index_select(0, user_idx)
        prob = torch.stack(
            [
                1.0 - retrievability,
                retrievability * review_prob[:, 0:1],
                retrievability * review_prob[:, 1:2],
                retrievability * review_prob[:, 2:3],
            ],
            dim=2,
        )
        s_exp = s.expand(block_size, interval_count)
        d_exp = d.expand(block_size, interval_count)
        next_indices: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        init_d = self.init_d.index_select(0, user_idx).view(block_size, 1)
        for rating_idx, rating in enumerate(range(1, 5)):
            if rating > 1:
                hard_penalty = (
                    weights[:, 15:16]
                    if rating == 2
                    else torch.ones(
                        (block_size, 1),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                easy_bonus = (
                    weights[:, 16:17]
                    if rating == 4
                    else torch.ones(
                        (block_size, 1),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                inc = (
                    torch.exp(weights[:, 8:9])
                    * (11.0 - d_exp)
                    * torch.pow(s_exp, -weights[:, 9:10])
                    * (
                        torch.exp(
                            (1.0 - retrievability) * weights[:, 10:11],
                        )
                        - 1.0
                    )
                )
                new_s = s_exp * (1.0 + inc * hard_penalty * easy_bonus)
            else:
                new_s = (
                    weights[:, 11:12]
                    * torch.pow(d_exp, -weights[:, 12:13])
                    * (torch.pow(s_exp + 1.0, weights[:, 13:14]) - 1.0)
                    * torch.exp((1.0 - retrievability) * weights[:, 14:15])
                )
                new_min = s_exp / torch.exp(weights[:, 17:18] * weights[:, 18:19])
                new_s = torch.minimum(new_s, new_min)

            rating_delta = float(rating) - 3.0
            delta_d = -weights[:, 6:7] * rating_delta
            new_d = d_exp + delta_d * (10.0 - d_exp) / 9.0
            new_d = weights[:, 7:8] * init_d + (1.0 - weights[:, 7:8]) * new_d
            new_d = torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)
            kernel_idx, kernel_weight = self._state_kernel(new_s, new_d)
            next_indices.append(kernel_idx)
            next_weights.append(kernel_weight)

        next_idx = torch.stack(next_indices, dim=0).permute(2, 3, 0, 1)
        next_weight = torch.stack(next_weights, dim=0).permute(2, 3, 0, 1)
        return prob, next_idx.to(dtype=torch.int64), next_weight.to(dtype=self.dtype)

    def _active_cell_continuation_scores_for_intervals(
        self,
        *,
        block_occupancy: torch.Tensor,
        user_idx: torch.Tensor,
        intervals: torch.Tensor,
        valid: torch.Tensor,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        next_weight: torch.Tensor,
        active_rems: Sequence[int],
    ) -> torch.Tensor:
        block_size = int(user_idx.numel())
        weight_count = int(block_occupancy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        interval_count = int(intervals.numel())
        score = torch.zeros(
            (block_size, weight_count, interval_count),
            device=self.device,
            dtype=self.dtype,
        )
        if block_size == 0 or interval_count == 0:
            return score
        user_exp = user_idx.view(block_size, 1, 1).expand(
            block_size,
            weight_count,
            interval_count,
        )
        weight_idx = torch.arange(
            weight_count,
            device=self.device,
            dtype=torch.int64,
        ).view(1, weight_count, 1)
        weight_exp = weight_idx.expand(block_size, weight_count, interval_count)
        cost_weight = cost_weight_grid.index_select(0, user_idx)[:, :, None]
        review_minutes = self.review_cost_minutes.index_select(0, user_idx)
        interval_row = intervals.view(1, interval_count)
        valid_float = valid.to(dtype=self.dtype)
        first_interval = int(intervals[0].item())
        for rem in active_rems:
            if rem < first_interval:
                continue
            future_rem = torch.clamp(rem - intervals, min=0).to(torch.int64)
            future_rem_exp = future_rem.view(1, 1, interval_count).expand(
                block_size,
                weight_count,
                interval_count,
            )
            cont_weight = valid_float * (interval_row <= rem).to(
                device=self.device, dtype=self.dtype
            )
            occupancy_rem = block_occupancy[:, :, rem]
            for rating_idx, rating in enumerate(range(1, 5)):
                future_value = torch.zeros_like(score)
                for corner_idx in range(4):
                    state_exp = next_idx[:, :, rating_idx, corner_idx].view(
                        block_size,
                        1,
                        interval_count,
                    )
                    future_value += (
                        next_weight[:, None, :, rating_idx, corner_idx]
                        * value[
                            user_exp,
                            weight_exp,
                            future_rem_exp,
                            state_exp.expand(
                                block_size,
                                weight_count,
                                interval_count,
                            ),
                        ]
                    )
                weighted_prob = prob[:, :, rating_idx] * cont_weight
                score += (
                    occupancy_rem[:, :, None]
                    * weighted_prob[:, None, :]
                    * (
                        future_value
                        - cost_weight * review_minutes[:, rating - 1].view(-1, 1, 1)
                    )
                )
        return score

    def _improve_stationary_policy_batch(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
        active: torch.Tensor | None = None,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[1])
        visited_flat = occupancy[:, :, 1:, :].sum(dim=2) > 0.0
        if active is not None:
            visited_flat = visited_flat & active[:, :, None]
        visited_cells = visited_flat.any(dim=1)
        active_weight_count = (
            int(active.sum().item())
            if active is not None
            else self.user_count * weight_count
        )
        active_user_idx, active_state_idx = torch.nonzero(
            visited_cells,
            as_tuple=True,
        )
        active_cell_count = int(active_user_idx.numel())
        block_size = int(self.STATIONARY_IMPROVE_STATE_BLOCK_SIZE)
        block_count = (
            math.ceil(active_cell_count / block_size) if active_cell_count else 0
        )
        if self._progress_logging_enabled():
            total_cells = max(1, self.user_count * self.state_count)
            iter_label = "?" if iteration is None else str(iteration)
            self._progress_log(
                "stationary improve start "
                f"iter={iter_label} visited_state_density="
                f"{active_cell_count / float(total_cells):.6f} "
                f"active_cells={active_cell_count}/{total_cells} "
                f"active_weights={active_weight_count}/{self.user_count * weight_count} "
                f"active_state_blocks={block_count} block_size={block_size}"
            )

        best_score = torch.full(
            (self.user_count, weight_count, self.state_count),
            -math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        best_interval = torch.ones(
            (self.user_count, weight_count, self.state_count),
            device=self.device,
            dtype=torch.int64,
        )
        current_score = torch.zeros_like(best_score)
        current_interval = self._intervals_for_retention_policy(policy).reshape(
            self.user_count,
            weight_count,
            self.state_count,
        )
        improve_start_s = time.perf_counter()
        next_log_s = self._next_progress_log_deadline()
        total_chunks = math.ceil((self.horizon + 1) / self.interval_chunk_size)
        if active_cell_count:
            weight_scatter = torch.arange(
                weight_count,
                device=self.device,
                dtype=torch.int64,
            ).view(1, weight_count)
            for block_number, block_start in enumerate(
                range(0, active_cell_count, block_size),
                start=1,
            ):
                block_stop = min(active_cell_count, block_start + block_size)
                block_user = active_user_idx[block_start:block_stop]
                block_state = active_state_idx[block_start:block_stop]
                current_interval_block = self._active_cell_policy_interval_block(
                    current_interval=current_interval,
                    user_idx=block_user,
                    state_idx=block_state,
                )
                block_occupancy = self._active_cell_occupancy_block(
                    occupancy=occupancy,
                    user_idx=block_user,
                    state_idx=block_state,
                )
                active_rems = [
                    int(rem)
                    for rem in (
                        torch.nonzero(
                            block_occupancy[:, :, 1:].sum(dim=(0, 1)) > 0.0,
                            as_tuple=False,
                        )
                        .flatten()
                        .add(1)
                        .cpu()
                        .tolist()
                    )
                ]
                (
                    memorized,
                    prefix_weighted,
                    prefix_occupancy,
                    total_weighted,
                    total_occupancy,
                ) = self._active_cell_immediate_prefix_tables(
                    block_occupancy=block_occupancy,
                    user_idx=block_user,
                    state_idx=block_state,
                )
                block_best_score = torch.full(
                    (int(block_user.numel()), weight_count),
                    -math.inf,
                    device=self.device,
                    dtype=self.dtype,
                )
                block_best_interval = torch.ones(
                    (int(block_user.numel()), weight_count),
                    device=self.device,
                    dtype=torch.int64,
                )
                block_current_score = torch.zeros_like(block_best_score)

                for chunk_number, start in enumerate(
                    range(1, self.horizon + 2, self.interval_chunk_size),
                    start=1,
                ):
                    chunk_start_s = time.perf_counter()
                    stop = min(self.horizon + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    valid = self._active_cell_attainable_interval_mask(
                        intervals=intervals,
                        user_idx=block_user,
                        state_idx=block_state,
                    )
                    if not bool(valid.any().item()):
                        continue
                    score = self._active_cell_immediate_scores_for_intervals(
                        intervals=intervals,
                        memorized=memorized,
                        prefix_weighted=prefix_weighted,
                        prefix_occupancy=prefix_occupancy,
                        total_weighted=total_weighted,
                        total_occupancy=total_occupancy,
                    )
                    prob, next_idx, next_weight = self._active_cell_interval_tables(
                        intervals=intervals,
                        user_idx=block_user,
                        state_idx=block_state,
                    )
                    score += self._active_cell_continuation_scores_for_intervals(
                        block_occupancy=block_occupancy,
                        user_idx=block_user,
                        intervals=intervals,
                        valid=valid,
                        cost_weights=cost_weights,
                        value=value,
                        prob=prob,
                        next_idx=next_idx,
                        next_weight=next_weight,
                        active_rems=active_rems,
                    )
                    valid_expanded = valid[:, None, :]
                    masked_score = torch.where(
                        valid_expanded,
                        score,
                        torch.full_like(score, -math.inf),
                    )
                    chunk_max_score = masked_score.max(dim=2).values
                    tie_tolerance = (
                        torch.finfo(self.dtype).eps
                        * 64.0
                        * torch.clamp(torch.abs(chunk_max_score), min=1.0)
                    )
                    chunk_best_idx = (
                        (masked_score >= (chunk_max_score - tie_tolerance)[:, :, None])
                        .to(torch.int64)
                        .argmax(dim=2)
                    )
                    chunk_best_score = masked_score.gather(
                        2,
                        chunk_best_idx[:, :, None],
                    ).squeeze(2)
                    chunk_interval = intervals.index_select(
                        0,
                        chunk_best_idx.reshape(-1),
                    ).reshape_as(chunk_best_idx)
                    update_tolerance = (
                        torch.finfo(self.dtype).eps
                        * 64.0
                        * torch.clamp(
                            torch.maximum(
                                torch.abs(chunk_best_score),
                                torch.abs(block_best_score),
                            ),
                            min=1.0,
                        )
                    )
                    better = torch.isneginf(block_best_score) | (
                        chunk_best_score > (block_best_score + update_tolerance)
                    )
                    block_best_score = torch.where(
                        better,
                        chunk_best_score,
                        block_best_score,
                    )
                    block_best_interval = torch.where(
                        better,
                        chunk_interval,
                        block_best_interval,
                    )
                    current_match = current_interval_block[
                        :, :, None
                    ] == intervals.view(1, 1, int(intervals.numel()))
                    score_for_current = torch.where(
                        valid_expanded,
                        score,
                        torch.zeros_like(score),
                    )
                    block_current_score += (
                        score_for_current * current_match.to(dtype=self.dtype)
                    ).sum(dim=2)
                    if self._progress_log_due(next_log_s):
                        iter_label = "?" if iteration is None else str(iteration)
                        self._progress_log(
                            "stationary improve progress "
                            f"iter={iter_label} block={block_number}/{block_count} "
                            f"chunk={chunk_number}/{total_chunks} "
                            f"intervals={start}-{stop - 1} "
                            f"valid_pairs={int(valid.sum().item())} "
                            f"chunk_s={time.perf_counter() - chunk_start_s:.1f} "
                            f"elapsed_s={time.perf_counter() - improve_start_s:.1f}"
                        )
                        next_log_s = self._next_progress_log_deadline()

                block_weight_idx = weight_scatter.expand(
                    int(block_user.numel()),
                    weight_count,
                )
                block_user_exp = block_user.view(-1, 1).expand_as(block_weight_idx)
                block_state_exp = block_state.view(-1, 1).expand_as(block_weight_idx)
                best_score[block_user_exp, block_weight_idx, block_state_exp] = (
                    block_best_score
                )
                best_interval[block_user_exp, block_weight_idx, block_state_exp] = (
                    block_best_interval
                )
                current_score[block_user_exp, block_weight_idx, block_state_exp] = (
                    block_current_score
                )

        visited = visited_flat.reshape(
            self.user_count,
            weight_count,
            self.s_count,
            self.d_count,
        )
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited_flat,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = torch.where(
            visited_flat.any(dim=2),
            masked_improvement.max(dim=2).values,
            torch.zeros(
                (self.user_count, weight_count),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        best_interval_grid = best_interval.reshape(
            self.user_count,
            weight_count,
            self.s_count,
            self.d_count,
        )
        new_policy = torch.where(
            visited,
            self._retention_for_interval_grid_batch(
                best_interval_grid,
                terminal_interval=self.horizon + 1,
            ),
            policy,
        )
        return new_policy, residual, visited

    def stationary_policy_interval_q_gaps(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: Sequence[float] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_tensor = torch.as_tensor(
            cost_weights,
            device=self.device,
            dtype=self.dtype,
        )
        weight_count = int(weight_tensor.numel())
        if weight_count <= 0:
            raise ValueError("cost_weights must contain at least one value.")
        policy = policy.to(device=self.device, dtype=self.dtype).reshape(
            self.user_count,
            weight_count,
            self.s_count,
            self.d_count,
        )
        value = self._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=weight_tensor,
        )
        occupancy = self._rollout_occupancy_batch(policy=policy)
        visited_flat = occupancy[:, :, 1:, :].sum(dim=2) > 0.0
        visited_cells = visited_flat.any(dim=1)
        active_user_idx, active_state_idx = torch.nonzero(
            visited_cells,
            as_tuple=True,
        )
        active_cell_count = int(active_user_idx.numel())
        block_size = int(self.STATIONARY_IMPROVE_STATE_BLOCK_SIZE)
        block_count = (
            math.ceil(active_cell_count / block_size) if active_cell_count else 0
        )
        if self._progress_logging_enabled():
            total_cells = max(1, self.user_count * self.state_count)
            self._progress_log(
                "stationary q-gap start "
                f"visited_state_density={active_cell_count / float(total_cells):.6f} "
                f"active_cells={active_cell_count}/{total_cells} "
                f"active_state_blocks={block_count} block_size={block_size}"
            )

        best_score = torch.full(
            (self.user_count, weight_count, self.state_count),
            -math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        second_score = torch.full_like(best_score, -math.inf)
        qgap_start_s = time.perf_counter()
        next_log_s = self._next_progress_log_deadline()
        total_chunks = math.ceil((self.horizon + 1) / self.interval_chunk_size)
        if active_cell_count:
            weight_scatter = torch.arange(
                weight_count,
                device=self.device,
                dtype=torch.int64,
            ).view(1, weight_count)
            for block_number, block_start in enumerate(
                range(0, active_cell_count, block_size),
                start=1,
            ):
                block_stop = min(active_cell_count, block_start + block_size)
                block_user = active_user_idx[block_start:block_stop]
                block_state = active_state_idx[block_start:block_stop]
                block_occupancy = self._active_cell_occupancy_block(
                    occupancy=occupancy,
                    user_idx=block_user,
                    state_idx=block_state,
                )
                active_rems = [
                    int(rem)
                    for rem in (
                        torch.nonzero(
                            block_occupancy[:, :, 1:].sum(dim=(0, 1)) > 0.0,
                            as_tuple=False,
                        )
                        .flatten()
                        .add(1)
                        .cpu()
                        .tolist()
                    )
                ]
                (
                    memorized,
                    prefix_weighted,
                    prefix_occupancy,
                    total_weighted,
                    total_occupancy,
                ) = self._active_cell_immediate_prefix_tables(
                    block_occupancy=block_occupancy,
                    user_idx=block_user,
                    state_idx=block_state,
                )
                block_best_score = torch.full(
                    (int(block_user.numel()), weight_count),
                    -math.inf,
                    device=self.device,
                    dtype=self.dtype,
                )
                block_second_score = torch.full_like(block_best_score, -math.inf)

                for chunk_number, start in enumerate(
                    range(1, self.horizon + 2, self.interval_chunk_size),
                    start=1,
                ):
                    chunk_start_s = time.perf_counter()
                    stop = min(self.horizon + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    valid = self._active_cell_attainable_interval_mask(
                        intervals=intervals,
                        user_idx=block_user,
                        state_idx=block_state,
                    )
                    if not bool(valid.any().item()):
                        continue
                    score = self._active_cell_immediate_scores_for_intervals(
                        intervals=intervals,
                        memorized=memorized,
                        prefix_weighted=prefix_weighted,
                        prefix_occupancy=prefix_occupancy,
                        total_weighted=total_weighted,
                        total_occupancy=total_occupancy,
                    )
                    prob, next_idx, next_weight = self._active_cell_interval_tables(
                        intervals=intervals,
                        user_idx=block_user,
                        state_idx=block_state,
                    )
                    score += self._active_cell_continuation_scores_for_intervals(
                        block_occupancy=block_occupancy,
                        user_idx=block_user,
                        intervals=intervals,
                        valid=valid,
                        cost_weights=weight_tensor,
                        value=value,
                        prob=prob,
                        next_idx=next_idx,
                        next_weight=next_weight,
                        active_rems=active_rems,
                    )
                    masked_score = torch.where(
                        valid[:, None, :],
                        score,
                        torch.full_like(score, -math.inf),
                    )
                    candidates = torch.cat(
                        [
                            block_best_score[:, :, None],
                            block_second_score[:, :, None],
                            masked_score,
                        ],
                        dim=2,
                    )
                    top2 = torch.topk(candidates, k=2, dim=2).values
                    block_best_score = top2[:, :, 0]
                    block_second_score = top2[:, :, 1]
                    if self._progress_log_due(next_log_s):
                        self._progress_log(
                            "stationary q-gap progress "
                            f"block={block_number}/{block_count} "
                            f"chunk={chunk_number}/{total_chunks} "
                            f"intervals={start}-{stop - 1} "
                            f"valid_pairs={int(valid.sum().item())} "
                            f"chunk_s={time.perf_counter() - chunk_start_s:.1f} "
                            f"elapsed_s={time.perf_counter() - qgap_start_s:.1f}"
                        )
                        next_log_s = self._next_progress_log_deadline()

                block_weight_idx = weight_scatter.expand(
                    int(block_user.numel()),
                    weight_count,
                )
                block_user_exp = block_user.view(-1, 1).expand_as(block_weight_idx)
                block_state_exp = block_state.view(-1, 1).expand_as(block_weight_idx)
                best_score[block_user_exp, block_weight_idx, block_state_exp] = (
                    block_best_score
                )
                second_score[block_user_exp, block_weight_idx, block_state_exp] = (
                    block_second_score
                )

        finite_gap = torch.isfinite(best_score) & torch.isfinite(second_score)
        q_gap = torch.where(
            finite_gap,
            torch.clamp(best_score - second_score, min=0.0),
            torch.zeros_like(best_score),
        )
        q_gap = torch.where(visited_flat, q_gap, torch.zeros_like(q_gap))
        if self._progress_logging_enabled():
            positive = q_gap[q_gap > 0.0]
            mean_positive = float(positive.mean().item()) if positive.numel() else 0.0
            self._progress_log(
                "stationary q-gap done "
                f"runtime_s={time.perf_counter() - qgap_start_s:.1f} "
                f"positive_cells={int(positive.numel())} "
                f"mean_positive={mean_positive:.6g}"
            )
        return (
            q_gap.reshape(self.user_count, weight_count, self.s_count, self.d_count),
            visited_flat.reshape(
                self.user_count,
                weight_count,
                self.s_count,
                self.d_count,
            ),
        )

    def _improve_stationary_policy_batch_dense_reference(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
        active: torch.Tensor | None = None,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        best_score = torch.full(
            (self.user_count, weight_count, self.s_count, self.d_count),
            -math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        best_interval = torch.ones(
            (self.user_count, weight_count, self.s_count, self.d_count),
            device=self.device,
            dtype=torch.int64,
        )
        current_interval = self._intervals_for_retention_policy(policy)
        current_score = torch.zeros_like(best_score)
        improve_start_s = time.perf_counter()
        next_log_s = self._next_progress_log_deadline()
        total_chunks = math.ceil((self.horizon + 1) / self.interval_chunk_size)
        chunk_number = 0

        for start in range(1, self.horizon + 2, self.interval_chunk_size):
            chunk_number += 1
            chunk_start_s = time.perf_counter()
            stop = min(self.horizon + 2, start + self.interval_chunk_size)
            intervals = torch.arange(
                start,
                stop,
                device=self.device,
                dtype=torch.int64,
            )
            interval_count = int(intervals.numel())
            interval, prob, next_idx, next_weight = self._interval_tables_batch(
                intervals
            )
            score = torch.zeros(
                (
                    self.user_count,
                    weight_count,
                    interval_count,
                    self.s_count,
                    self.d_count,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            valid = self._attainable_interval_mask(intervals, self.horizon + 1)
            for rem in range(1, self.horizon + 1):
                rem_occupancy = occupancy[:, :, rem, :].reshape(
                    self.user_count,
                    weight_count,
                    self.s_count,
                    self.d_count,
                )
                if not bool(rem_occupancy.sum().item()):
                    continue
                candidate = self._candidate_interval_value_from_tables(
                    intervals=intervals,
                    rem=rem,
                    cost_weights=cost_weight_grid,
                    value=value,
                    interval=interval,
                    prob=prob,
                    next_idx=next_idx,
                    next_weight=next_weight,
                )
                candidate = torch.where(
                    valid[:, None, :, :, None],
                    candidate,
                    torch.zeros_like(candidate),
                )
                score += rem_occupancy[:, :, None, :, :] * candidate
                if self._progress_log_due(next_log_s):
                    active_mass = float(rem_occupancy.sum().item())
                    iter_label = "?" if iteration is None else str(iteration)
                    self._progress_log(
                        "stationary improve progress "
                        f"iter={iter_label} chunk={chunk_number}/{total_chunks} "
                        f"intervals={start}-{stop - 1} rem={rem}/{self.horizon} "
                        f"active_mass={active_mass:.6g} "
                        f"chunk_s={time.perf_counter() - chunk_start_s:.1f} "
                        f"elapsed_s={time.perf_counter() - improve_start_s:.1f}"
                    )
                    next_log_s = self._next_progress_log_deadline()

            masked_score = torch.where(
                valid[:, None, :, :, None],
                score,
                torch.full_like(score, -math.inf),
            )
            chunk_best_score, chunk_best_idx = masked_score.max(dim=2)
            chunk_interval = intervals.index_select(
                0,
                chunk_best_idx.reshape(-1),
            ).reshape_as(chunk_best_idx)
            better = chunk_best_score > best_score
            best_score = torch.where(better, chunk_best_score, best_score)
            best_interval = torch.where(better, chunk_interval, best_interval)

            current_match = current_interval[:, :, None, :, :] == intervals.view(
                1,
                1,
                interval_count,
                1,
                1,
            )
            current_score += (score * current_match.to(dtype=self.dtype)).sum(dim=2)
            if self._progress_log_due(next_log_s):
                iter_label = "?" if iteration is None else str(iteration)
                self._progress_log(
                    "stationary improve chunk done "
                    f"iter={iter_label} chunk={chunk_number}/{total_chunks} "
                    f"intervals={start}-{stop - 1} "
                    f"chunk_s={time.perf_counter() - chunk_start_s:.1f} "
                    f"elapsed_s={time.perf_counter() - improve_start_s:.1f}"
                )
                next_log_s = self._next_progress_log_deadline()

        visited = occupancy[:, :, 1:, :].sum(dim=2) > 0.0
        if active is not None:
            visited = visited & active[:, :, None]
        visited = visited.reshape(
            self.user_count,
            weight_count,
            self.s_count,
            self.d_count,
        )
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = torch.where(
            visited.reshape(self.user_count, weight_count, self.state_count).any(dim=2),
            masked_improvement.reshape(
                self.user_count,
                weight_count,
                self.state_count,
            )
            .max(dim=2)
            .values,
            torch.zeros(
                (self.user_count, weight_count),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        new_policy = torch.where(
            visited,
            self._retention_for_interval_grid_batch(
                best_interval,
                terminal_interval=self.horizon + 1,
            ),
            policy,
        )
        return new_policy, residual, visited

    def _metrics_from_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> list[list[OracleMetrics]]:
        weight_count = int(policy.shape[1])
        cost_weight_grid = self._cost_weight_grid(cost_weights)
        occupancy = self._rollout_occupancy_batch(policy=policy)
        selected_interval, selected_prob, _, _ = self._policy_tables_batch(policy)
        total_mem = torch.zeros(
            (self.user_count, weight_count),
            device=self.device,
            dtype=self.dtype,
        )
        total_minutes = (
            (self.first_rating_prob * self.learning_cost_minutes)
            .sum(dim=1)[:, None]
            .expand_as(total_mem)
            .clone()
        )
        total_reviews = torch.zeros_like(total_mem)
        total_lapses = torch.zeros_like(total_mem)
        expected_review_minutes = (
            selected_prob * self.review_cost_minutes[:, None, :, None]
        ).sum(dim=2)
        for rem in range(1, self.horizon + 1):
            current = occupancy[:, :, rem, :]
            if not bool(current.sum().item()):
                continue
            active_days = torch.minimum(
                selected_interval,
                torch.full_like(selected_interval, rem),
            )
            total_mem += (current * self._memorized_sum_batch(active_days)).sum(dim=2)
            cont_mask = selected_interval <= rem
            source = current * cont_mask.to(dtype=self.dtype)
            if not bool(source.sum().item()):
                continue
            total_minutes += (source * expected_review_minutes).sum(dim=2)
            total_reviews += source.sum(dim=2)
            total_lapses += (source * selected_prob[:, :, 0, :]).sum(dim=2)

        day_count = float(self.days)
        objectives = total_mem / day_count - cost_weight_grid * (
            total_minutes / day_count
        )
        metrics: list[list[OracleMetrics]] = []
        for user_idx in range(self.user_count):
            row: list[OracleMetrics] = []
            for weight_idx in range(weight_count):
                reviews_float = float(total_reviews[user_idx, weight_idx].item())
                lapses_float = float(total_lapses[user_idx, weight_idx].item())
                observed_retention = (
                    1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
                )
                row.append(
                    OracleMetrics(
                        card_expected_retrievability=float(
                            total_mem[user_idx, weight_idx].item() / day_count
                        ),
                        card_minutes_per_day=float(
                            total_minutes[user_idx, weight_idx].item() / day_count
                        ),
                        card_reviews_per_day=float(
                            total_reviews[user_idx, weight_idx].item() / day_count
                        ),
                        card_total_reviews=reviews_float,
                        card_total_lapses=lapses_float,
                        card_total_cost_seconds=float(
                            total_minutes[user_idx, weight_idx].item() * 60.0
                        ),
                        observed_retention=observed_retention,
                        scalar_objective=float(objectives[user_idx, weight_idx].item()),
                        runtime_s=0.0,
                    )
                )
            metrics.append(row)
        return metrics


class FSRS6AverageRewardOracle(FSRS6GridOracle):
    def __init__(
        self,
        *,
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        super().__init__(
            days=2,
            action_retentions=action_retentions,
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            dtype=dtype,
            device=device,
            fsrs_weights=fsrs_weights,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_costs=learning_costs,
            review_costs=review_costs,
            cache_config=cache_config,
        )
        self.state_count = int(self.s_grid.numel() * self.d_grid.numel())
        self._action_tables = self._precompute_average_reward_action_tables()

    def solve_average_reward_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> AverageRewardOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        gains: list[torch.Tensor | None] = [None for _ in cost_weights]
        iterations: list[int | None] = [None for _ in cost_weights]
        converged: list[bool | None] = [None for _ in cost_weights]
        residuals: list[float | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="average_reward",
                    method="solve_average_reward_policies",
                    cost_weight=float(weight),
                    extra={
                        "max_iterations": max_iterations,
                        "tolerance": tolerance,
                    },
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device)
            gains[idx] = torch.as_tensor(
                entry["gain"],
                device=self.device,
                dtype=self.dtype,
            )
            iterations[idx] = int(entry["iterations"])
            converged[idx] = bool(entry["converged"])
            residuals[idx] = float(entry["residual"])

        if missing:
            solution = self._solve_average_reward_policies_uncached(
                [weight for _, weight in missing],
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = solution.policy[local_idx].to(device=self.device)
                gain = solution.gains[local_idx].to(device=self.device)
                iteration = solution.iterations[local_idx]
                did_converge = solution.converged[local_idx]
                residual = solution.residuals[local_idx]
                policies[idx] = policy
                gains[idx] = gain
                iterations[idx] = iteration
                converged[idx] = did_converge
                residuals[idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="average_reward",
                        method="solve_average_reward_policies",
                        cost_weight=weight,
                        extra={
                            "max_iterations": max_iterations,
                            "tolerance": tolerance,
                        },
                    ),
                    data={
                        "policy": policy,
                        "gain": float(gain.item()),
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )

        return AverageRewardOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            ).to(device=self.device, dtype=torch.int64),
            gains=torch.stack(
                [gain for gain in gains if gain is not None],
                dim=0,
            ).to(device=self.device, dtype=self.dtype),
            iterations=[int(value) for value in iterations if value is not None],
            converged=[bool(value) for value in converged if value is not None],
            residuals=[float(value) for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def _solve_average_reward_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> AverageRewardOracleSolution:
        start = time.perf_counter()
        policies: list[torch.Tensor] = []
        gains: list[torch.Tensor] = []
        iterations: list[int] = []
        converged: list[bool] = []
        residuals: list[float] = []
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=len(cost_weights) * max_iterations,
                desc="Average-reward oracle",
                unit="iter",
                leave=False,
            )
        try:
            result = self._solve_policy_batch(
                cost_weights=weight_tensor,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress_bar=progress_bar,
            )
            (
                batched_policy,
                batched_gain,
                batch_iterations,
                batch_converged,
                batch_residuals,
            ) = result
            policies.extend(
                policy.reshape(self.s_grid.numel(), self.d_grid.numel())
                for policy in batched_policy
            )
            gains.extend(gain for gain in batched_gain)
            iterations.extend(batch_iterations)
            converged.extend(batch_converged)
            residuals.extend(batch_residuals)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return AverageRewardOracleSolution(
            policy=torch.stack(policies, dim=0).to(dtype=torch.int64),
            gains=torch.stack(gains).to(dtype=self.dtype),
            iterations=iterations,
            converged=converged,
            residuals=residuals,
            runtime_s=time.perf_counter() - start,
        )

    def _solve_policy_batch(
        self,
        *,
        cost_weights: torch.Tensor,
        max_iterations: int,
        tolerance: float,
        progress_bar: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[bool], list[float]]:
        interval, immediate_mem, review_cost, prob, next_idx = self._action_tables
        weight_count = int(cost_weights.numel())
        reward = (
            immediate_mem[None, :, :]
            - cost_weights.view(
                weight_count,
                1,
                1,
            )
            * review_cost[None, :, :]
        )
        policy = torch.argmax(reward / interval[None, :, :], dim=1).to(torch.int64)
        h = torch.zeros(
            (weight_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        gain = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        iterations = torch.zeros(
            weight_count,
            device=self.device,
            dtype=torch.int64,
        )
        converged = torch.zeros(weight_count, device=self.device, dtype=torch.bool)
        residuals = torch.full(
            (weight_count,),
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones(weight_count, device=self.device, dtype=torch.bool)

        for iteration in range(1, max_iterations + 1):
            active_count = int(active.sum().item())
            if active_count == 0:
                break
            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            active_policy = policy.index_select(0, active_idx)
            active_reward = reward.index_select(0, active_idx)
            selected = self._select_policy_tables_batch(
                policy=active_policy,
                interval=interval,
                reward=active_reward,
                prob=prob,
                next_idx=next_idx,
            )
            selected_interval, selected_reward, selected_prob, selected_next = selected
            active_gain, active_h = self._evaluate_policy_batch(
                interval=selected_interval,
                reward=selected_reward,
                prob=selected_prob,
                next_idx=selected_next,
                initial_h=h.index_select(0, active_idx),
                tolerance=tolerance,
            )
            scores = (
                active_reward
                - active_gain.view(active_count, 1, 1) * interval[None, :, :]
                + self._expected_bias_batch(
                    prob=prob,
                    next_idx=next_idx,
                    h=active_h,
                )
            )
            new_policy = torch.argmax(scores, dim=1).to(torch.int64)
            current_score = scores.gather(1, active_policy[:, None, :]).squeeze(1)
            residual = scores.max(dim=1).values.sub(current_score).max(dim=1).values
            policy_changed = (new_policy != active_policy).any(dim=1)

            policy[active_idx] = new_policy
            h[active_idx] = active_h
            gain[active_idx] = active_gain
            iterations[active_idx] = iteration
            residuals[active_idx] = residual
            if progress_bar is not None:
                progress_bar.update(active_count)

            done = (~policy_changed) & (residual <= tolerance)
            if bool(done.any().item()):
                done_idx = active_idx[done]
                converged[done_idx] = True
                active[done_idx] = False

        if bool(active.any().item()):
            iterations[active] = max_iterations

        return (
            policy,
            gain,
            [int(value) for value in iterations.cpu().tolist()],
            [bool(value) for value in converged.cpu().tolist()],
            [float(value) for value in residuals.cpu().tolist()],
        )

    def labels(
        self,
        *,
        policies: torch.Tensor,
        cost_weights: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
        goal_weight: torch.Tensor,
    ) -> torch.Tensor:
        s_idx = self._s_to_idx(s)
        d_idx = self._d_to_idx(d)
        goal_idx = torch.argmin(
            torch.abs(
                goal_weight.to(dtype=cost_weights.dtype)[:, None]
                - cost_weights[None, :]
            ),
            dim=1,
        )
        return policies.to(device=s.device)[goal_idx, s_idx, d_idx]

    def _solve_single_policy(
        self,
        *,
        cost_weight: float,
        max_iterations: int,
        tolerance: float,
        progress_bar: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, int, bool, float]:
        interval, immediate_mem, review_cost, prob, next_idx = self._action_tables
        reward = immediate_mem - float(cost_weight) * review_cost
        policy = torch.argmax(reward / interval, dim=0).to(torch.int64)
        h = torch.zeros(self.state_count, device=self.device, dtype=self.dtype)
        gain = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        residual = math.inf
        converged = False

        for iteration in range(1, max_iterations + 1):
            selected = self._select_policy_tables(
                policy=policy,
                interval=interval,
                reward=reward,
                prob=prob,
                next_idx=next_idx,
            )
            selected_interval, selected_reward, selected_prob, selected_next = selected
            gain, h = self._evaluate_policy(
                interval=selected_interval,
                reward=selected_reward,
                prob=selected_prob,
                next_idx=selected_next,
                initial_h=h,
                tolerance=tolerance,
            )
            scores = (
                reward
                - gain * interval
                + self._expected_bias(
                    prob=prob,
                    next_idx=next_idx,
                    h=h,
                )
            )
            new_policy = torch.argmax(scores, dim=0).to(torch.int64)
            current_score = scores.gather(0, policy[None, :]).squeeze(0)
            residual = float((scores.max(dim=0).values - current_score).max().item())
            policy_changed = bool((new_policy != policy).any().item())
            policy = new_policy
            if progress_bar is not None:
                progress_bar.update(1)
            if not policy_changed and residual <= tolerance:
                converged = True
                return policy, gain, iteration, converged, residual

        return policy, gain, max_iterations, converged, residual

    def _precompute_average_reward_action_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        intervals: list[torch.Tensor] = []
        immediate_mem: list[torch.Tensor] = []
        review_cost: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        next_indices: list[torch.Tensor] = []
        d_count = int(self.d_grid.numel())
        for transition in self.transitions:
            interval = transition.interval.to(dtype=self.dtype)
            interval_flat = interval[:, None].expand_as(self.s_mesh).reshape(-1)
            intervals.append(interval_flat)
            immediate = self._memorized_sum_integral(self.s_grid, transition.interval)
            immediate_mem.append(immediate[:, None].expand_as(self.s_mesh).reshape(-1))
            rating_probs = torch.stack(
                [
                    prob[:, None].expand_as(self.s_mesh).reshape(-1)
                    for prob in transition.prob
                ],
                dim=1,
            )
            cost = torch.zeros(self.state_count, device=self.device, dtype=self.dtype)
            for rating_idx, rating in enumerate(range(1, 5)):
                cost += (
                    rating_probs[:, rating_idx] * self.review_cost_minutes[rating - 1]
                )
            review_cost.append(cost)
            probs.append(rating_probs)
            next_indices.append(
                torch.stack(
                    [
                        (
                            transition.next_s_idx[rating_idx].reshape(-1) * d_count
                            + transition.next_d_idx[rating_idx].reshape(-1)
                        )
                        for rating_idx in range(4)
                    ],
                    dim=1,
                ).to(torch.int64)
            )
        return (
            torch.stack(intervals, dim=0),
            torch.stack(immediate_mem, dim=0),
            torch.stack(review_cost, dim=0),
            torch.stack(probs, dim=0),
            torch.stack(next_indices, dim=0),
        )

    def _memorized_sum_integral(
        self,
        s: torch.Tensor,
        days: torch.Tensor,
    ) -> torch.Tensor:
        days_f = days.to(dtype=self.dtype)
        safe_s = torch.clamp(s, min=self.bounds.s_min)
        rate = self.factor / safe_s
        upper = days_f + 0.5
        lower = torch.full_like(upper, 0.5)
        exponent = self.decay + 1.0
        integral = (
            torch.pow(1.0 + rate * upper, exponent)
            - torch.pow(1.0 + rate * lower, exponent)
        ) / (rate * exponent)
        return torch.where(days_f > 0.0, integral, torch.zeros_like(integral))

    def _select_policy_tables(
        self,
        *,
        policy: torch.Tensor,
        interval: torch.Tensor,
        reward: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_idx = torch.arange(self.state_count, device=self.device)
        return (
            interval[policy, state_idx],
            reward[policy, state_idx],
            prob[policy, state_idx],
            next_idx[policy, state_idx],
        )

    def _select_policy_tables_batch(
        self,
        *,
        policy: torch.Tensor,
        interval: torch.Tensor,
        reward: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[0])
        state_idx = torch.arange(self.state_count, device=self.device)
        selected_interval = interval[policy, state_idx[None, :]]
        selected_reward = reward.gather(1, policy[:, None, :]).squeeze(1)
        policy_idx = policy[:, None, :, None]
        selected_prob = (
            prob[None, :, :, :]
            .expand(
                weight_count,
                -1,
                -1,
                -1,
            )
            .gather(
                1,
                policy_idx.expand(weight_count, 1, self.state_count, 4),
            )
        )
        selected_next_idx = (
            next_idx[None, :, :, :]
            .expand(
                weight_count,
                -1,
                -1,
                -1,
            )
            .gather(
                1,
                policy_idx.expand(weight_count, 1, self.state_count, 4),
            )
        )
        return (
            selected_interval,
            selected_reward,
            selected_prob.squeeze(1),
            selected_next_idx.squeeze(1),
        )

    def _evaluate_policy(
        self,
        *,
        interval: torch.Tensor,
        reward: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        initial_h: torch.Tensor,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stationary = self._stationary_distribution(
            prob=prob,
            next_idx=next_idx,
            tolerance=tolerance,
        )
        gain = torch.sum(stationary * reward) / torch.sum(stationary * interval)
        h = initial_h
        eval_tolerance = max(float(tolerance), 1e-11)
        for _ in range(4096):
            h_next = (
                reward
                - gain * interval
                + torch.sum(
                    prob * h.index_select(0, next_idx.reshape(-1)).reshape_as(prob),
                    dim=1,
                )
            )
            h_next = h_next - h_next[0]
            diff = torch.max(torch.abs(h_next - h))
            h = h_next
            if float(diff.item()) <= eval_tolerance:
                break
        return gain, h

    def _evaluate_policy_batch(
        self,
        *,
        interval: torch.Tensor,
        reward: torch.Tensor,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        initial_h: torch.Tensor,
        tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stationary = self._stationary_distribution_batch(
            prob=prob,
            next_idx=next_idx,
            tolerance=tolerance,
        )
        gain = torch.sum(stationary * reward, dim=1) / torch.sum(
            stationary * interval,
            dim=1,
        )
        h = initial_h
        eval_tolerance = max(float(tolerance), 1e-11)
        for _ in range(4096):
            future_h = h.gather(
                1,
                next_idx.reshape(next_idx.shape[0], -1),
            ).reshape_as(prob)
            h_next = (
                reward
                - gain[:, None] * interval
                + torch.sum(
                    prob * future_h,
                    dim=2,
                )
            )
            h_next = h_next - h_next[:, :1]
            diff = torch.max(torch.abs(h_next - h), dim=1).values
            h = h_next
            if bool((diff <= eval_tolerance).all().item()):
                break
        return gain, h

    def _stationary_distribution(
        self,
        *,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        pi = torch.full(
            (self.state_count,),
            1.0 / float(self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_next = next_idx.reshape(-1)
        stationary_tolerance = max(float(tolerance), 1e-12)
        for _ in range(4096):
            new_pi = torch.zeros_like(pi)
            new_pi.scatter_add_(0, flat_next, (pi[:, None] * prob).reshape(-1))
            new_pi = new_pi / torch.clamp(new_pi.sum(), min=1e-30)
            diff = torch.max(torch.abs(new_pi - pi))
            pi = new_pi
            if float(diff.item()) <= stationary_tolerance:
                break
        return pi

    def _stationary_distribution_batch(
        self,
        *,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        weight_count = int(prob.shape[0])
        pi = torch.full(
            (weight_count, self.state_count),
            1.0 / float(self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_pi = pi.reshape(-1)
        batch_offsets = (
            torch.arange(weight_count, device=self.device, dtype=torch.int64)
            .view(weight_count, 1, 1)
            .mul(self.state_count)
        )
        target = batch_offsets + next_idx
        stationary_tolerance = max(float(tolerance), 1e-12)
        for _ in range(4096):
            new_pi = torch.zeros_like(pi)
            new_pi.reshape(-1).scatter_add_(
                0,
                target.reshape(-1),
                (pi[:, :, None] * prob).reshape(-1),
            )
            new_pi = new_pi / torch.clamp(
                new_pi.sum(dim=1, keepdim=True),
                min=1e-30,
            )
            diff = torch.max(torch.abs(new_pi - pi), dim=1).values
            pi = new_pi
            flat_pi = pi.reshape(-1)
            if bool((diff <= stationary_tolerance).all().item()):
                break
        return flat_pi.reshape(weight_count, self.state_count)

    def _expected_bias(
        self,
        *,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(
            prob * h.index_select(0, next_idx.reshape(-1)).reshape_as(prob),
            dim=2,
        )

    def _expected_bias_batch(
        self,
        *,
        prob: torch.Tensor,
        next_idx: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(h.shape[0])
        future_h = h.gather(
            1,
            next_idx.reshape(1, -1).expand(weight_count, -1),
        ).reshape(weight_count, prob.shape[0], self.state_count, 4)
        return torch.sum(prob[None, :, :, :] * future_h, dim=3)


class FSRS6IntervalOracle(FSRS6GridOracle):
    TRANSITION_VALUE_LOOKUP_VERSION = "bilinear_log_s_linear_d_v1"

    def __init__(
        self,
        *,
        days: int,
        s_grid_size: int,
        d_grid_size: int,
        interval_chunk_size: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        if interval_chunk_size <= 0:
            raise ValueError("interval_chunk_size must be > 0.")
        super().__init__(
            days=days,
            action_retentions=[0.9],
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            dtype=dtype,
            device=device,
            fsrs_weights=fsrs_weights,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_costs=learning_costs,
            review_costs=review_costs,
            cache_config=cache_config,
        )
        self.interval_chunk_size = int(interval_chunk_size)
        self.memorized_by_day = self._precompute_memorized_by_day()

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="interval",
                    method="solve_policies",
                    cost_weight=float(weight),
                    extra=self._cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device)

        if missing:
            computed = self._solve_interval_policies_uncached(
                [weight for _, weight in missing],
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = computed[local_idx].to(device=self.device)
                policies[idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="interval",
                        method="solve_policies",
                        cost_weight=weight,
                        extra=self._cache_extra(),
                    ),
                    data={"policy": policy},
                )

        return torch.stack(
            [policy for policy in policies if policy is not None],
            dim=0,
        ).to(device=self.device)

    def _solve_interval_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        weight_count = int(weight_tensor.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        policy = torch.ones(shape, device=self.device, dtype=torch.int64)
        weights = weight_tensor.view(1, 1, 1, weight_count)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Interval oracle w batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_interval = torch.ones_like(policy[rem])

                for start in range(1, rem + 2, self.interval_chunk_size):
                    stop = min(rem + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    candidate_value = self._candidate_interval_value_batch(
                        intervals=intervals,
                        rem=rem,
                        cost_weights=weights,
                        value=value,
                    )
                    chunk_best_value, chunk_best_idx = candidate_value.max(dim=0)
                    chunk_best_interval = intervals.index_select(
                        0,
                        chunk_best_idx.reshape(-1),
                    ).reshape_as(chunk_best_idx)
                    better = chunk_best_value > best_value
                    best_value = torch.where(better, chunk_best_value, best_value)
                    best_interval = torch.where(
                        better,
                        chunk_best_interval,
                        best_interval,
                    )

                value[rem] = best_value
                policy[rem] = best_interval
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return policy.permute(3, 0, 1, 2).contiguous()

    def _cache_extra(self) -> dict[str, Any]:
        return {
            "interval_chunk_size": self.interval_chunk_size,
            "transition_value_lookup": self.TRANSITION_VALUE_LOOKUP_VERSION,
        }

    def _candidate_interval_value_batch(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(cost_weights.numel())
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())
        interval_count = int(intervals.numel())
        active_days = torch.minimum(intervals, torch.full_like(intervals, rem))
        immediate_mem = self._memorized_sum_for_interval_candidates(active_days)
        candidate_value = (
            immediate_mem[:, :, None, None]
            .expand(interval_count, s_count, d_count, weight_count)
            .clone()
        )

        cont_mask = intervals <= rem
        if not bool(cont_mask.any().item()):
            return candidate_value

        elapsed = intervals.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        future_rem = torch.clamp(rem - intervals, min=0).to(torch.int64)
        future_rem_idx = future_rem[:, None, None].expand(
            interval_count,
            s_count,
            d_count,
        )
        cont_weight = cont_mask.to(dtype=self.dtype)[:, None]

        for rating_idx, rating in enumerate(range(1, 5)):
            if rating == 1:
                prob = 1.0 - retrievability
            else:
                prob = retrievability * self.review_rating_prob[rating_idx - 1]
            next_s, next_d = self._next_state_interval_candidates(
                elapsed=elapsed,
                retrievability=retrievability,
                rating=rating,
            )
            future_value = self._interpolate_interval_value(
                value=value,
                rem_idx=future_rem_idx,
                s=next_s,
                d=next_d,
            )
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = (prob * cont_weight)[:, :, None, None]
            candidate_value += weighted * (future_value - cost_weights * review_minutes)

        return candidate_value

    def _interpolate_interval_value(
        self,
        *,
        value: torch.Tensor,
        rem_idx: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())

        log_s = torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
        s_pos = (log_s - self.log_s_min) / (self.log_s_max - self.log_s_min)
        s_pos = torch.clamp(s_pos * float(s_count - 1), 0.0, float(s_count - 1))
        s0 = torch.floor(s_pos).to(torch.int64)
        s1 = torch.clamp(s0 + 1, max=s_count - 1)
        sw = (s_pos - s0.to(dtype=self.dtype))[..., None]

        d_pos = torch.clamp(d, self.bounds.d_min, self.bounds.d_max)
        d_pos = (d_pos - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min)
        d_pos = torch.clamp(d_pos * float(d_count - 1), 0.0, float(d_count - 1))
        d0 = torch.floor(d_pos).to(torch.int64)
        d1 = torch.clamp(d0 + 1, max=d_count - 1)
        dw = (d_pos - d0.to(dtype=self.dtype))[..., None]

        v00 = value[rem_idx, s0, d0]
        v10 = value[rem_idx, s1, d0]
        v01 = value[rem_idx, s0, d1]
        v11 = value[rem_idx, s1, d1]
        return (
            v00 * (1.0 - sw) * (1.0 - dw)
            + v10 * sw * (1.0 - dw)
            + v01 * (1.0 - sw) * dw
            + v11 * sw * dw
        )

    def _memorized_sum_for_interval_candidates(
        self,
        days: torch.Tensor,
    ) -> torch.Tensor:
        return self.memorized_by_day.index_select(0, days.to(torch.int64))

    def _precompute_memorized_by_day(self) -> torch.Tensor:
        table = torch.zeros(
            (self.horizon + 1, int(self.s_grid.numel())),
            device=self.device,
            dtype=self.dtype,
        )
        if self.horizon <= 0:
            return table
        elapsed = torch.arange(
            1,
            self.horizon + 1,
            device=self.device,
            dtype=self.dtype,
        )
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        table[1:] = torch.cumsum(retrievability, dim=0)
        return table

    def _next_state_interval_candidates(
        self,
        *,
        elapsed: torch.Tensor,
        retrievability: torch.Tensor,
        rating: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interval_count = int(elapsed.numel())
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())
        s = self.s_grid.view(1, s_count, 1).expand(
            interval_count,
            s_count,
            d_count,
        )
        d = self.d_grid.view(1, 1, d_count).expand(
            interval_count,
            s_count,
            d_count,
        )
        r = retrievability[:, :, None].expand(interval_count, s_count, d_count)
        rating_tensor = torch.full(
            (interval_count, s_count, d_count),
            rating,
            device=self.device,
            dtype=torch.int64,
        )
        if rating > 1:
            new_s = self._stability_after_success(s, r, d, rating_tensor)
        else:
            new_s = self._stability_after_failure(s, r, d)
        new_d = self._next_d(d, rating_tensor)
        return (
            torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max),
            torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max),
        )


class FSRS6ContinuousRetentionOracle(FSRS6IntervalOracle):
    ACTION_POLICY_LOOKUP_VERSION = "terminal_retention_min_action_v2"

    def __init__(
        self,
        *,
        days: int,
        s_grid_size: int,
        d_grid_size: int,
        retention_min: float = 0.5,
        retention_max: float = 0.98,
        interval_chunk_size: int = 64,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        validate_continuous_retention_bounds(retention_min, retention_max)
        super().__init__(
            days=days,
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            interval_chunk_size=interval_chunk_size,
            dtype=dtype,
            device=device,
            fsrs_weights=fsrs_weights,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_costs=learning_costs,
            review_costs=review_costs,
            cache_config=cache_config,
        )
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        self.s_count = int(self.s_grid.numel())
        self.d_count = int(self.d_grid.numel())
        self.state_count = self.s_count * self.d_count
        self._flat_s_idx = (
            torch.arange(self.s_count, device=self.device)[:, None]
            .expand(self.s_count, self.d_count)
            .reshape(-1)
        )

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_retention",
                    method="solve_policies",
                    cost_weight=float(weight),
                    extra=self._cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device, dtype=self.dtype)

        if missing:
            computed = self._solve_continuous_policies_uncached(
                [weight for _, weight in missing],
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = computed[local_idx].to(device=self.device, dtype=self.dtype)
                policies[idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_retention",
                        method="solve_policies",
                        cost_weight=weight,
                        extra=self._cache_extra(),
                    ),
                    data={"policy": policy},
                )

        return torch.stack(
            [policy for policy in policies if policy is not None],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)

    def _cache_extra(self) -> dict[str, Any]:
        return {
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "interval_chunk_size": self.interval_chunk_size,
            "transition_value_lookup": self.TRANSITION_VALUE_LOOKUP_VERSION,
            "action_policy_lookup": self.ACTION_POLICY_LOOKUP_VERSION,
        }

    def _solve_continuous_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        weight_count = int(weight_tensor.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        policy = torch.full(
            shape,
            self.retention_max,
            device=self.device,
            dtype=self.dtype,
        )
        weights = weight_tensor.view(1, 1, 1, weight_count)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Continuous retention oracle w batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_retention = torch.full_like(policy[rem], self.retention_max)

                for start in range(1, rem + 2, self.interval_chunk_size):
                    stop = min(rem + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    candidate_value = self._candidate_interval_value_batch(
                        intervals=intervals,
                        rem=rem,
                        cost_weights=weights,
                        value=value,
                    )
                    candidate_value = self._mask_unattainable_candidates(
                        intervals=intervals,
                        rem=rem,
                        candidate_value=candidate_value,
                    )
                    chunk_best_value, chunk_best_idx = candidate_value.max(dim=0)
                    chunk_best_interval = intervals.index_select(
                        0,
                        chunk_best_idx.reshape(-1),
                    ).reshape_as(chunk_best_idx)
                    chunk_retention = self._retention_for_interval_grid(
                        chunk_best_interval,
                        terminal_interval=rem + 1,
                    )
                    better = chunk_best_value > best_value
                    best_value = torch.where(better, chunk_best_value, best_value)
                    best_retention = torch.where(
                        better,
                        chunk_retention,
                        best_retention,
                    )

                value[rem] = best_value
                policy[rem] = best_retention
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return policy.permute(3, 0, 1, 2).contiguous()

    def _mask_unattainable_candidates(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        candidate_value: torch.Tensor,
    ) -> torch.Tensor:
        mask = attainable_interval_mask_for_retention_bounds(
            intervals=intervals,
            s_grid=self.s_grid,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            factor=self.factor,
            decay=self.decay,
            terminal_interval=rem + 1,
        )
        expanded = mask[:, :, None, None]
        return torch.where(
            expanded,
            candidate_value,
            torch.full_like(candidate_value, -math.inf),
        )

    def _canonical_retention_for_intervals(
        self,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        return canonical_retention_for_intervals(
            intervals=intervals,
            s_grid=self.s_grid,
            factor=self.factor,
            decay=self.decay,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
        )

    def _retention_for_interval_grid(
        self,
        interval: torch.Tensor,
        *,
        terminal_interval: int | None = None,
    ) -> torch.Tensor:
        s = self.s_grid[:, None, None].expand_as(interval).to(dtype=self.dtype)
        retention = self._forgetting_curve(interval.to(dtype=self.dtype), s)
        retention = torch.clamp(
            retention,
            min=self.retention_min,
            max=self.retention_max,
        )
        if terminal_interval is not None:
            terminal = interval.to(dtype=torch.int64) >= int(terminal_interval)
            retention = torch.where(
                terminal,
                torch.full_like(retention, self.retention_min),
                retention,
            )
        return retention


class FSRS6ContinuousUniformTerminationOracle(FSRS6ContinuousRetentionOracle):
    """Continuous-retention oracle for a hidden uniformly distributed terminal day."""

    TERMINATION_DISTRIBUTION_VERSION = "hidden_uniform_1_to_remaining_v1"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cumulative_memorized_by_day = torch.cumsum(self.memorized_by_day, dim=0)

    def solve_policies(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_uniform_termination",
                    method="solve_policies",
                    cost_weight=float(weight),
                    extra=self._cache_extra(),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device, dtype=self.dtype)

        if missing:
            computed = self._solve_continuous_policies_uncached(
                [weight for _, weight in missing],
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = computed[local_idx].to(device=self.device, dtype=self.dtype)
                policies[idx] = policy
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_uniform_termination",
                        method="solve_policies",
                        cost_weight=weight,
                        extra=self._cache_extra(),
                    ),
                    data={"policy": policy},
                )

        return torch.stack(
            [policy for policy in policies if policy is not None],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)

    def _cache_extra(self) -> dict[str, Any]:
        extra = super()._cache_extra()
        extra["termination_distribution"] = self.TERMINATION_DISTRIBUTION_VERSION
        return extra

    def _solve_continuous_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool = False,
    ) -> torch.Tensor:
        weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        weight_count = int(weight_tensor.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, device=self.device, dtype=self.dtype)
        policy = torch.full(
            shape,
            self.retention_max,
            device=self.device,
            dtype=self.dtype,
        )
        weights = weight_tensor.view(1, 1, 1, weight_count)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=self.horizon,
                desc=f"Uniform-H continuous oracle w batch={weight_count}",
                unit="day",
                leave=False,
            )
        try:
            for rem in range(1, self.horizon + 1):
                best_value = torch.full_like(value[rem], -math.inf)
                best_retention = torch.full_like(policy[rem], self.retention_max)

                for start in range(1, rem + 2, self.interval_chunk_size):
                    stop = min(rem + 2, start + self.interval_chunk_size)
                    intervals = torch.arange(
                        start,
                        stop,
                        device=self.device,
                        dtype=torch.int64,
                    )
                    candidate_value = self._candidate_uniform_interval_value_batch(
                        intervals=intervals,
                        rem=rem,
                        cost_weights=weights,
                        value=value,
                    )
                    candidate_value = self._mask_unattainable_candidates(
                        intervals=intervals,
                        rem=rem,
                        candidate_value=candidate_value,
                    )
                    chunk_best_value, chunk_best_idx = candidate_value.max(dim=0)
                    chunk_best_interval = intervals.index_select(
                        0,
                        chunk_best_idx.reshape(-1),
                    ).reshape_as(chunk_best_idx)
                    chunk_retention = self._retention_for_interval_grid(
                        chunk_best_interval,
                        terminal_interval=rem + 1,
                    )
                    better = chunk_best_value > best_value
                    best_value = torch.where(better, chunk_best_value, best_value)
                    best_retention = torch.where(
                        better,
                        chunk_retention,
                        best_retention,
                    )

                value[rem] = best_value
                policy[rem] = best_retention
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return policy.permute(3, 0, 1, 2).contiguous()

    def _candidate_uniform_interval_value_batch(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(cost_weights.numel())
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())
        interval_count = int(intervals.numel())
        immediate_mem = self._uniform_termination_memorized_by_interval(
            intervals=intervals,
            rem=rem,
        )
        candidate_value = (
            immediate_mem[:, :, None, None]
            .expand(interval_count, s_count, d_count, weight_count)
            .clone()
        )

        review_mask = intervals <= rem
        if not bool(review_mask.any().item()):
            return candidate_value

        elapsed = intervals.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        future_rem = torch.clamp(rem - intervals, min=0).to(torch.int64)
        future_rem_idx = future_rem[:, None, None].expand(
            interval_count,
            s_count,
            d_count,
        )
        rem_float = float(rem)
        review_weight = (
            torch.clamp(rem - intervals + 1, min=0).to(dtype=self.dtype) / rem_float
        )
        future_weight = (
            torch.clamp(rem - intervals, min=0).to(dtype=self.dtype) / rem_float
        )

        for rating_idx, rating in enumerate(range(1, 5)):
            if rating == 1:
                prob = 1.0 - retrievability
            else:
                prob = retrievability * self.review_rating_prob[rating_idx - 1]
            next_s, next_d = self._next_state_interval_candidates(
                elapsed=elapsed,
                retrievability=retrievability,
                rating=rating,
            )
            future_value = self._interpolate_interval_value(
                value=value,
                rem_idx=future_rem_idx,
                s=next_s,
                d=next_d,
            )
            review_minutes = self.review_cost_minutes[rating - 1]
            prob_expanded = prob[:, :, None, None]
            candidate_value += prob_expanded * (
                future_weight[:, None, None, None] * future_value
                - review_weight[:, None, None, None] * cost_weights * review_minutes
            )

        return candidate_value

    def _uniform_termination_memorized_by_interval(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
    ) -> torch.Tensor:
        clamped = torch.clamp(intervals.to(torch.int64), max=rem)
        prefix_idx = torch.clamp(intervals.to(torch.int64) - 1, max=rem)
        terminal_count = torch.clamp(rem - intervals.to(torch.int64) + 1, min=0).to(
            dtype=self.dtype
        )
        return (
            self.cumulative_memorized_by_day.index_select(0, prefix_idx)
            + terminal_count[:, None] * self.memorized_by_day.index_select(0, clamped)
        ) / float(rem)


class FSRS6ContinuousStationaryFiniteOracle(FSRS6ContinuousRetentionOracle):
    STATIONARY_POLICY_ITERATION_VERSION = "continuous_interval_greedy_v3"

    def solve_stationary_finite_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        metrics_by_weight: list[OracleMetrics | None] = [None for _ in cost_weights]
        objectives: list[float | None] = [None for _ in cost_weights]
        iterations: list[int | None] = [None for _ in cost_weights]
        converged: list[bool | None] = [None for _ in cost_weights]
        residuals: list[float | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_stationary_finite",
                    method="solve_stationary_finite_policies",
                    cost_weight=float(weight),
                    extra=self._stationary_cache_extra(
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    ),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device, dtype=self.dtype)
            metrics_by_weight[idx] = _metrics_from_payload(
                entry["metrics"],
                runtime_s=0.0,
            )
            objectives[idx] = float(entry["objective"])
            iterations[idx] = int(entry["iterations"])
            converged[idx] = bool(entry["converged"])
            residuals[idx] = float(entry["residual"])

        if missing:
            solution = self._solve_stationary_finite_policies_uncached(
                [weight for _, weight in missing],
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = solution.policy[local_idx].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                metrics = solution.metrics[local_idx]
                objective = float(solution.objectives[local_idx].item())
                iteration = solution.iterations[local_idx]
                did_converge = solution.converged[local_idx]
                residual = solution.residuals[local_idx]
                policies[idx] = policy
                metrics_by_weight[idx] = metrics
                objectives[idx] = objective
                iterations[idx] = iteration
                converged[idx] = did_converge
                residuals[idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_stationary_finite",
                        method="solve_stationary_finite_policies",
                        cost_weight=weight,
                        extra=self._stationary_cache_extra(
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                        ),
                    ),
                    data={
                        "policy": policy,
                        "metrics": _metrics_payload(metrics),
                        "objective": objective,
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )

        return StationaryFiniteOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            ).to(device=self.device, dtype=self.dtype),
            metrics=[metric for metric in metrics_by_weight if metric is not None],
            objectives=torch.tensor(
                [objective for objective in objectives if objective is not None],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[int(value) for value in iterations if value is not None],
            converged=[bool(value) for value in converged if value is not None],
            residuals=[float(value) for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def _stationary_cache_extra(
        self,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        extra = self._cache_extra()
        extra["policy_iteration"] = self.STATIONARY_POLICY_ITERATION_VERSION
        if max_iterations is not None:
            extra["max_iterations"] = max_iterations
        if tolerance is not None:
            extra["tolerance"] = tolerance
        return extra

    def _solve_stationary_finite_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int,
        tolerance: float,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        start = time.perf_counter()
        cost_weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        finite_policies = self.solve_policies(cost_weights, progress=progress)
        policy = torch.clamp(
            finite_policies[:, self.horizon],
            min=self.retention_min,
            max=self.retention_max,
        ).contiguous()
        weight_count = int(cost_weight_tensor.numel())
        value = self._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=cost_weight_tensor,
        )
        objective = self._objective_from_value_batch(
            value=value,
            cost_weights=cost_weight_tensor,
        )
        iterations = torch.zeros(weight_count, device=self.device, dtype=torch.int64)
        converged = torch.zeros(weight_count, device=self.device, dtype=torch.bool)
        residuals = torch.full(
            (weight_count,),
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones(weight_count, device=self.device, dtype=torch.bool)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=weight_count * max_iterations,
                desc="Continuous stationary finite oracle",
                unit="iter",
                leave=False,
            )
        try:
            for iteration in range(1, max_iterations + 1):
                if not bool(active.any().item()):
                    break
                active_idx = active.nonzero(as_tuple=False).squeeze(1)
                active_count = int(active_idx.numel())
                active_policy = policy.index_select(0, active_idx)
                active_weights = cost_weight_tensor.index_select(0, active_idx)
                active_value = value.index_select(0, active_idx)
                occupancy = self._rollout_occupancy_batch(
                    policy=active_policy,
                    stationary=True,
                )
                new_policy, residual, visited = self._improve_stationary_policy_batch(
                    policy=active_policy,
                    occupancy=occupancy,
                    value=active_value,
                    cost_weights=active_weights,
                )
                changed = (
                    (torch.abs(new_policy - active_policy) > 1e-12) & visited
                ).reshape(active_count, self.state_count)
                policy_changed = changed.any(dim=1)
                residuals[active_idx] = residual
                iterations[active_idx] = iteration
                if progress_bar is not None:
                    progress_bar.update(active_count)

                done = (~policy_changed) | (residual <= tolerance)
                if bool(done.any().item()):
                    done_idx = active_idx[done]
                    converged[done_idx] = True
                    active[done_idx] = False

                candidate = ~done
                if not bool(candidate.any().item()):
                    continue

                candidate_idx = active_idx[candidate]
                candidate_policy = new_policy[candidate]
                candidate_weights = cost_weight_tensor.index_select(0, candidate_idx)
                candidate_value = self._evaluate_stationary_policy_value_batch(
                    policy=candidate_policy,
                    cost_weights=candidate_weights,
                )
                candidate_objective = self._objective_from_value_batch(
                    value=candidate_value,
                    cost_weights=candidate_weights,
                )
                objective_improvement = candidate_objective - objective[candidate_idx]
                accepted = objective_improvement > tolerance
                residuals[candidate_idx] = torch.where(
                    accepted,
                    objective_improvement,
                    torch.clamp(objective_improvement, min=0.0),
                )

                if bool((~accepted).any().item()):
                    rejected_idx = candidate_idx[~accepted]
                    converged[rejected_idx] = True
                    active[rejected_idx] = False

                if bool(accepted.any().item()):
                    accepted_idx = candidate_idx[accepted]
                    policy[accepted_idx] = candidate_policy[accepted]
                    value[accepted_idx] = candidate_value[accepted]
                    objective[accepted_idx] = candidate_objective[accepted]
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if bool(active.any().item()):
            iterations[active] = max_iterations

        metrics = self._metrics_from_occupancy_batch(
            policy=policy,
            cost_weights=cost_weight_tensor,
        )
        return StationaryFiniteOracleSolution(
            policy=policy,
            metrics=metrics,
            objectives=torch.tensor(
                [metric.scalar_objective for metric in metrics],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[int(value) for value in iterations.cpu().tolist()],
            converged=[bool(value) for value in converged.cpu().tolist()],
            residuals=[float(value) for value in residuals.cpu().tolist()],
            runtime_s=time.perf_counter() - start,
        )

    def labels(
        self,
        *,
        policies: torch.Tensor,
        cost_weights: torch.Tensor,
        s: torch.Tensor,
        d: torch.Tensor,
        goal_weight: torch.Tensor,
    ) -> torch.Tensor:
        goal_idx = torch.argmin(
            torch.abs(
                goal_weight.to(dtype=cost_weights.dtype)[:, None]
                - cost_weights[None, :]
            ),
            dim=1,
        )
        return bilinear_retention_policy_lookup(
            oracle=self,
            policies=policies,
            goal_indices=goal_idx,
            s=s,
            d=d,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
        )

    def _evaluate_stationary_policy_value_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        value = torch.zeros(
            (weight_count, self.horizon + 1, self.s_count, self.d_count),
            device=self.device,
            dtype=self.dtype,
        )
        interval, prob, next_idx, next_weight = self._policy_tables_batch(policy)
        batch_idx = torch.arange(weight_count, device=self.device)[:, None]
        weight_penalty = cost_weights.to(dtype=self.dtype)[:, None]
        flat_s_idx = self._flat_s_idx[None, :].expand(weight_count, self.state_count)
        value_flat = value.reshape(
            weight_count,
            self.horizon + 1,
            self.state_count,
        )

        for rem in range(1, self.horizon + 1):
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            value_rem = self.memorized_by_day[active_days, flat_s_idx].clone()

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[:, rating_idx, corner_idx, :]
                            * value_flat[
                                batch_idx,
                                future_rem,
                                next_idx[:, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    weighted = torch.where(
                        cont_mask,
                        prob[:, rating_idx, :],
                        torch.zeros_like(prob[:, rating_idx, :]),
                    )
                    value_rem += weighted * (
                        future_value - weight_penalty * review_minutes
                    )

            value[:, rem] = value_rem.reshape(weight_count, self.s_count, self.d_count)

        return value

    def _objective_from_value_batch(
        self,
        *,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(cost_weights.numel())
        value_flat = value.reshape(weight_count, self.horizon + 1, self.state_count)
        total_value = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_learning_minutes = torch.tensor(
            0.0,
            device=self.device,
            dtype=self.dtype,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            total_value += prob * (
                state_weight[None, :] * value_flat[:, self.horizon, state_idx]
            ).sum(dim=1)
            total_learning_minutes += prob * self.learning_cost_minutes[rating - 1]
        return (total_value - cost_weights * total_learning_minutes) / float(self.days)

    def _improve_stationary_policy_batch(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[0])
        interval_count = self.horizon + 1
        action_scores = torch.zeros(
            (weight_count, interval_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        valid_actions = (
            self._stationary_attainable_interval_mask(
                torch.arange(1, self.horizon + 2, device=self.device, dtype=torch.int64)
            )
            .reshape(interval_count, self.s_count, 1)
            .expand(
                interval_count,
                self.s_count,
                self.d_count,
            )
            .reshape(interval_count, self.state_count)
        )
        value_for_lookup = value.permute(1, 2, 3, 0).contiguous()
        weights = cost_weights.to(dtype=self.dtype).view(1, 1, 1, weight_count)

        for rem in range(1, self.horizon + 1):
            rem_occupancy = occupancy[:, rem, :].reshape(
                weight_count,
                self.s_count,
                self.d_count,
            )
            if float(rem_occupancy.sum().item()) <= 0.0:
                continue
            for start in range(1, self.horizon + 2, self.interval_chunk_size):
                stop = min(self.horizon + 2, start + self.interval_chunk_size)
                intervals = torch.arange(
                    start,
                    stop,
                    device=self.device,
                    dtype=torch.int64,
                )
                candidate_value = self._candidate_interval_value_batch(
                    intervals=intervals,
                    rem=rem,
                    cost_weights=weights,
                    value=value_for_lookup,
                )
                mask = self._stationary_attainable_interval_mask(intervals)
                candidate_value = torch.where(
                    mask[:, :, None, None],
                    candidate_value,
                    torch.zeros_like(candidate_value),
                )
                score = rem_occupancy[:, None, :, :] * candidate_value.permute(
                    3, 0, 1, 2
                )
                action_scores[:, start - 1 : stop - 1] += score.reshape(
                    weight_count,
                    int(intervals.numel()),
                    self.state_count,
                )

        action_scores = torch.where(
            valid_actions[None, :, :],
            action_scores,
            torch.full_like(action_scores, -math.inf),
        )
        best_score, best_idx = action_scores.max(dim=1)
        best_interval = (best_idx + 1).reshape(
            weight_count,
            self.s_count,
            self.d_count,
        )
        current_interval = self._intervals_for_retention_policy(policy)
        current_score = action_scores.gather(
            1,
            (current_interval.reshape(weight_count, self.state_count) - 1)[:, None, :],
        ).squeeze(1)
        state_occupancy = occupancy[:, 1:, :].sum(dim=1)
        visited = state_occupancy > 0.0
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = masked_improvement.max(dim=1).values
        residual = torch.where(
            visited.any(dim=1),
            residual,
            torch.zeros_like(residual),
        )
        new_policy = torch.where(
            visited.reshape(weight_count, self.s_count, self.d_count),
            self._retention_for_interval_grid_batch(
                best_interval,
                terminal_interval=self.horizon + 1,
            ),
            policy,
        )
        return (
            new_policy,
            residual,
            visited.reshape(weight_count, self.s_count, self.d_count),
        )

    def _rollout_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        stationary: bool,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        occupancy = torch.zeros(
            (weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_occupancy = occupancy.reshape(-1)
        batch_offsets = (
            torch.arange(weight_count, device=self.device, dtype=torch.int64)
            .view(weight_count, 1)
            .mul((self.horizon + 1) * self.state_count)
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            for corner_idx in range(4):
                target = (
                    batch_offsets
                    + self.horizon * self.state_count
                    + state_idx[corner_idx]
                )
                amount = (prob * state_weight[corner_idx]).expand(weight_count)
                flat_occupancy.scatter_add_(0, target.reshape(-1), amount)

        selected_interval: torch.Tensor | None = None
        selected_prob: torch.Tensor | None = None
        selected_next_idx: torch.Tensor | None = None
        selected_next_weight: torch.Tensor | None = None
        if stationary:
            (
                selected_interval,
                selected_prob,
                selected_next_idx,
                selected_next_weight,
            ) = self._policy_tables_batch(policy)

        for rem in range(self.horizon, 0, -1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            if not stationary:
                (
                    selected_interval,
                    selected_prob,
                    selected_next_idx,
                    selected_next_weight,
                ) = self._policy_tables_batch(policy[:, rem])
            if (
                selected_interval is None
                or selected_prob is None
                or selected_next_idx is None
                or selected_next_weight is None
            ):
                raise RuntimeError("selected policy tables were not initialized.")

            cont_mask = selected_interval <= rem
            if not bool(cont_mask.any().item()):
                continue
            source = current * cont_mask.to(dtype=self.dtype)
            if float(source.sum().item()) <= 0.0:
                continue

            future_rem = torch.clamp(rem - selected_interval, min=0).to(torch.int64)
            target_base = batch_offsets + future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * selected_prob[:, rating_idx, :]
                        * selected_next_weight[:, rating_idx, corner_idx, :]
                    )
                    target = (
                        target_base + selected_next_idx[:, rating_idx, corner_idx, :]
                    )
                    flat_occupancy.scatter_add_(
                        0,
                        target.reshape(-1),
                        amount.reshape(-1),
                    )

        return occupancy

    def _metrics_from_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> list[OracleMetrics]:
        weight_count = int(policy.shape[0])
        occupancy = self._rollout_occupancy_batch(policy=policy, stationary=True)
        interval, prob, _, _ = self._policy_tables_batch(policy)
        total_mem = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_minutes = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_reviews = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_lapses = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        learning_minutes = (self.first_rating_prob * self.learning_cost_minutes).sum()
        total_minutes += learning_minutes
        expected_review_minutes = (prob * self.review_cost_minutes.view(1, 4, 1)).sum(
            dim=1
        )
        flat_s_idx = self._flat_s_idx[None, :].expand(weight_count, self.state_count)

        for rem in range(1, self.horizon + 1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            immediate_mem = self.memorized_by_day[active_days, flat_s_idx]
            total_mem += (current * immediate_mem).sum(dim=1)

            cont_mask = interval <= rem
            source = current * cont_mask.to(dtype=self.dtype)
            if float(source.sum().item()) <= 0.0:
                continue
            total_minutes += (source * expected_review_minutes).sum(dim=1)
            total_reviews += source.sum(dim=1)
            total_lapses += (source * prob[:, 0, :]).sum(dim=1)

        day_count = float(self.days)
        objectives = total_mem / day_count - cost_weights * (total_minutes / day_count)
        metrics: list[OracleMetrics] = []
        for idx in range(weight_count):
            reviews_float = float(total_reviews[idx].item())
            lapses_float = float(total_lapses[idx].item())
            observed_retention = (
                1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
            )
            metrics.append(
                OracleMetrics(
                    card_expected_retrievability=float(
                        (total_mem[idx] / day_count).item()
                    ),
                    card_minutes_per_day=float((total_minutes[idx] / day_count).item()),
                    card_reviews_per_day=reviews_float / day_count,
                    card_total_reviews=reviews_float,
                    card_total_lapses=lapses_float,
                    card_total_cost_seconds=float((total_minutes[idx] * 60.0).item()),
                    observed_retention=observed_retention,
                    scalar_objective=float(objectives[idx].item()),
                    runtime_s=0.0,
                )
            )
        return metrics

    def _policy_tables_batch(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[0])
        interval = self._intervals_for_retention_policy(policy)
        s = self.s_mesh[None, :, :].expand(weight_count, self.s_count, self.d_count)
        d = self.d_mesh[None, :, :].expand(weight_count, self.s_count, self.d_count)
        elapsed = interval.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(elapsed, s)
        probs = torch.stack(
            (
                1.0 - retrievability,
                retrievability * self.review_rating_prob[0],
                retrievability * self.review_rating_prob[1],
                retrievability * self.review_rating_prob[2],
            ),
            dim=1,
        ).reshape(weight_count, 4, self.state_count)

        next_indices: list[torch.Tensor] = []
        next_weights: list[torch.Tensor] = []
        for rating in range(1, 5):
            rating_tensor = torch.full(
                (weight_count, self.s_count, self.d_count),
                rating,
                device=self.device,
                dtype=torch.int64,
            )
            if rating > 1:
                new_s = self._stability_after_success(
                    s,
                    retrievability,
                    d,
                    rating_tensor,
                )
            else:
                new_s = self._stability_after_failure(s, retrievability, d)
            new_d = self._next_d(d, rating_tensor)
            kernel_idx, kernel_weight = self._state_kernel(new_s, new_d)
            next_indices.append(kernel_idx)
            next_weights.append(kernel_weight)

        next_idx = torch.stack(next_indices, dim=0).permute(2, 0, 1, 3, 4)
        next_weight = torch.stack(next_weights, dim=0).permute(2, 0, 1, 3, 4)
        return (
            interval.reshape(weight_count, self.state_count),
            probs,
            next_idx.reshape(weight_count, 4, 4, self.state_count).to(
                dtype=torch.int64
            ),
            next_weight.reshape(weight_count, 4, 4, self.state_count).to(
                dtype=self.dtype
            ),
        )

    def _intervals_for_retention_policy(self, policy: torch.Tensor) -> torch.Tensor:
        s = self.s_mesh[None, :, :].expand_as(policy)
        interval = retention_interval_float(
            s=s,
            retention=policy,
            factor=self.factor,
            decay=self.decay,
        )
        return torch.clamp(
            torch.round(interval),
            min=1.0,
            max=float(self.horizon + 1),
        ).to(torch.int64)

    def _stationary_attainable_interval_mask(
        self,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        return attainable_interval_mask_for_retention_bounds(
            intervals=intervals,
            s_grid=self.s_grid,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            factor=self.factor,
            decay=self.decay,
            terminal_interval=self.horizon + 1,
        )

    def _retention_for_interval_grid_batch(
        self,
        interval: torch.Tensor,
        *,
        terminal_interval: int | None = None,
    ) -> torch.Tensor:
        s = self.s_grid[None, :, None].expand_as(interval).to(dtype=self.dtype)
        retention = self._forgetting_curve(interval.to(dtype=self.dtype), s)
        retention = torch.clamp(
            retention,
            min=self.retention_min,
            max=self.retention_max,
        )
        if terminal_interval is not None:
            terminal = interval.to(dtype=torch.int64) >= int(terminal_interval)
            retention = torch.where(
                terminal,
                torch.full_like(retention, self.retention_min),
                retention,
            )
        return retention


class FSRS6ContinuousStationaryUniformTerminationOracle(
    FSRS6ContinuousStationaryFiniteOracle
):
    """Best stationary continuous-retention policy for hidden Uniform-H terminal day."""

    STATIONARY_UNIFORM_POLICY_ITERATION_VERSION = "uniform_h_stationary_greedy_v1"
    TERMINATION_DISTRIBUTION_VERSION = (
        FSRS6ContinuousUniformTerminationOracle.TERMINATION_DISTRIBUTION_VERSION
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cumulative_memorized_by_day = torch.cumsum(self.memorized_by_day, dim=0)

    def solve_stationary_uniform_termination_policies(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        if not cost_weights:
            raise ValueError("cost_weights must contain at least one value.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        start = time.perf_counter()
        policies: list[torch.Tensor | None] = [None for _ in cost_weights]
        metrics_by_weight: list[OracleMetrics | None] = [None for _ in cost_weights]
        objectives: list[float | None] = [None for _ in cost_weights]
        iterations: list[int | None] = [None for _ in cost_weights]
        converged: list[bool | None] = [None for _ in cost_weights]
        residuals: list[float | None] = [None for _ in cost_weights]
        missing: list[tuple[int, float]] = []
        for idx, weight in enumerate(cost_weights):
            entry = load_cache_entry(
                self.cache_config,
                key_parts=self._cache_key_parts(
                    oracle_kind="continuous_stationary_uniform_termination",
                    method="solve_stationary_uniform_termination_policies",
                    cost_weight=float(weight),
                    extra=self._stationary_uniform_cache_extra(
                        max_iterations=max_iterations,
                        tolerance=tolerance,
                    ),
                ),
                map_location=self.device,
            )
            if entry is None or not isinstance(entry.get("policy"), torch.Tensor):
                missing.append((idx, float(weight)))
                continue
            policies[idx] = entry["policy"].to(device=self.device, dtype=self.dtype)
            metrics_by_weight[idx] = _metrics_from_payload(
                entry["metrics"],
                runtime_s=0.0,
            )
            objectives[idx] = float(entry["objective"])
            iterations[idx] = int(entry["iterations"])
            converged[idx] = bool(entry["converged"])
            residuals[idx] = float(entry["residual"])

        if missing:
            solution = self._solve_stationary_uniform_termination_policies_uncached(
                [weight for _, weight in missing],
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            for local_idx, (idx, weight) in enumerate(missing):
                policy = solution.policy[local_idx].to(
                    device=self.device,
                    dtype=self.dtype,
                )
                metrics = solution.metrics[local_idx]
                objective = float(solution.objectives[local_idx].item())
                iteration = solution.iterations[local_idx]
                did_converge = solution.converged[local_idx]
                residual = solution.residuals[local_idx]
                policies[idx] = policy
                metrics_by_weight[idx] = metrics
                objectives[idx] = objective
                iterations[idx] = iteration
                converged[idx] = did_converge
                residuals[idx] = residual
                write_cache_entry(
                    self.cache_config,
                    key_parts=self._cache_key_parts(
                        oracle_kind="continuous_stationary_uniform_termination",
                        method="solve_stationary_uniform_termination_policies",
                        cost_weight=weight,
                        extra=self._stationary_uniform_cache_extra(
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                        ),
                    ),
                    data={
                        "policy": policy,
                        "metrics": _metrics_payload(metrics),
                        "objective": objective,
                        "iterations": iteration,
                        "converged": did_converge,
                        "residual": residual,
                    },
                )

        return StationaryFiniteOracleSolution(
            policy=torch.stack(
                [policy for policy in policies if policy is not None],
                dim=0,
            ).to(device=self.device, dtype=self.dtype),
            metrics=[metric for metric in metrics_by_weight if metric is not None],
            objectives=torch.tensor(
                [objective for objective in objectives if objective is not None],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[int(value) for value in iterations if value is not None],
            converged=[bool(value) for value in converged if value is not None],
            residuals=[float(value) for value in residuals if value is not None],
            runtime_s=time.perf_counter() - start,
        )

    def evaluate_stationary_uniform_termination_policy(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: Sequence[float] | torch.Tensor,
    ) -> list[OracleMetrics]:
        weight_tensor = torch.as_tensor(
            cost_weights,
            device=self.device,
            dtype=self.dtype,
        )
        return self._metrics_from_uniform_termination_occupancy_batch(
            policy=policy.to(device=self.device, dtype=self.dtype),
            cost_weights=weight_tensor,
        )

    def _stationary_uniform_cache_extra(
        self,
        *,
        max_iterations: int | None = None,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        extra = self._cache_extra()
        extra["termination_distribution"] = self.TERMINATION_DISTRIBUTION_VERSION
        extra["policy_iteration"] = self.STATIONARY_UNIFORM_POLICY_ITERATION_VERSION
        if max_iterations is not None:
            extra["max_iterations"] = max_iterations
        if tolerance is not None:
            extra["tolerance"] = tolerance
        return extra

    def _solve_stationary_uniform_termination_policies_uncached(
        self,
        cost_weights: Sequence[float],
        *,
        max_iterations: int,
        tolerance: float,
        progress: bool = False,
    ) -> StationaryFiniteOracleSolution:
        start = time.perf_counter()
        cost_weight_tensor = torch.tensor(
            list(cost_weights), device=self.device, dtype=self.dtype
        )
        policy = self._initial_stationary_uniform_policy(
            cost_weights,
            progress=progress,
        )
        weight_count = int(cost_weight_tensor.numel())
        value = self._evaluate_stationary_uniform_termination_policy_value_batch(
            policy=policy,
            cost_weights=cost_weight_tensor,
        )
        objective = self._objective_from_value_batch(
            value=value,
            cost_weights=cost_weight_tensor,
        )
        iterations = torch.zeros(weight_count, device=self.device, dtype=torch.int64)
        converged = torch.zeros(weight_count, device=self.device, dtype=torch.bool)
        residuals = torch.full(
            (weight_count,),
            math.inf,
            device=self.device,
            dtype=self.dtype,
        )
        active = torch.ones(weight_count, device=self.device, dtype=torch.bool)

        progress_bar = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=weight_count * max_iterations,
                desc="Continuous stationary Uniform-H oracle",
                unit="iter",
                leave=False,
            )
        try:
            for iteration in range(1, max_iterations + 1):
                if not bool(active.any().item()):
                    break
                active_idx = active.nonzero(as_tuple=False).squeeze(1)
                active_count = int(active_idx.numel())
                active_policy = policy.index_select(0, active_idx)
                active_weights = cost_weight_tensor.index_select(0, active_idx)
                active_value = value.index_select(0, active_idx)
                occupancy = self._rollout_uniform_termination_occupancy_batch(
                    policy=active_policy,
                )
                new_policy, residual, visited = (
                    self._improve_stationary_uniform_termination_policy_batch(
                        policy=active_policy,
                        occupancy=occupancy,
                        value=active_value,
                        cost_weights=active_weights,
                    )
                )
                changed = (
                    (torch.abs(new_policy - active_policy) > 1e-12) & visited
                ).reshape(active_count, self.state_count)
                policy_changed = changed.any(dim=1)
                residuals[active_idx] = residual
                iterations[active_idx] = iteration
                if progress_bar is not None:
                    progress_bar.update(active_count)

                done = (~policy_changed) | (residual <= tolerance)
                if bool(done.any().item()):
                    done_idx = active_idx[done]
                    converged[done_idx] = True
                    active[done_idx] = False

                candidate = ~done
                if not bool(candidate.any().item()):
                    continue

                candidate_idx = active_idx[candidate]
                candidate_policy = new_policy[candidate]
                candidate_weights = cost_weight_tensor.index_select(0, candidate_idx)
                candidate_value = (
                    self._evaluate_stationary_uniform_termination_policy_value_batch(
                        policy=candidate_policy,
                        cost_weights=candidate_weights,
                    )
                )
                candidate_objective = self._objective_from_value_batch(
                    value=candidate_value,
                    cost_weights=candidate_weights,
                )
                objective_improvement = candidate_objective - objective[candidate_idx]
                accepted = objective_improvement > tolerance
                residuals[candidate_idx] = torch.where(
                    accepted,
                    objective_improvement,
                    torch.clamp(objective_improvement, min=0.0),
                )

                if bool((~accepted).any().item()):
                    rejected_idx = candidate_idx[~accepted]
                    converged[rejected_idx] = True
                    active[rejected_idx] = False

                if bool(accepted.any().item()):
                    accepted_idx = candidate_idx[accepted]
                    policy[accepted_idx] = candidate_policy[accepted]
                    value[accepted_idx] = candidate_value[accepted]
                    objective[accepted_idx] = candidate_objective[accepted]
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if bool(active.any().item()):
            iterations[active] = max_iterations

        metrics = self._metrics_from_uniform_termination_occupancy_batch(
            policy=policy,
            cost_weights=cost_weight_tensor,
        )
        return StationaryFiniteOracleSolution(
            policy=policy,
            metrics=metrics,
            objectives=torch.tensor(
                [metric.scalar_objective for metric in metrics],
                device=self.device,
                dtype=self.dtype,
            ),
            iterations=[int(value) for value in iterations.cpu().tolist()],
            converged=[bool(value) for value in converged.cpu().tolist()],
            residuals=[float(value) for value in residuals.cpu().tolist()],
            runtime_s=time.perf_counter() - start,
        )

    def _initial_stationary_uniform_policy(
        self,
        cost_weights: Sequence[float],
        *,
        progress: bool,
    ) -> torch.Tensor:
        uniform_oracle = FSRS6ContinuousUniformTerminationOracle(
            days=self.days,
            s_grid_size=self.s_count,
            d_grid_size=self.d_count,
            retention_min=self.retention_min,
            retention_max=self.retention_max,
            interval_chunk_size=self.interval_chunk_size,
            dtype=self.dtype,
            device=self.device,
            fsrs_weights=_tensor_float_list(self.weights),
            first_rating_prob=_tensor_float_list(self.first_rating_prob),
            review_rating_prob=_tensor_float_list(self.review_rating_prob),
            learning_costs=[
                60.0 * value for value in _tensor_float_list(self.learning_cost_minutes)
            ],
            review_costs=[
                60.0 * value for value in _tensor_float_list(self.review_cost_minutes)
            ],
            cache_config=self.cache_config,
        )
        finite_policies = uniform_oracle.solve_policies(
            cost_weights,
            progress=progress,
        )
        return torch.clamp(
            finite_policies[:, self.horizon],
            min=self.retention_min,
            max=self.retention_max,
        ).contiguous()

    def _evaluate_stationary_uniform_termination_policy_value_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        value = torch.zeros(
            (weight_count, self.horizon + 1, self.s_count, self.d_count),
            device=self.device,
            dtype=self.dtype,
        )
        interval, prob, next_idx, next_weight = self._policy_tables_batch(policy)
        batch_idx = torch.arange(weight_count, device=self.device)[:, None]
        weight_penalty = cost_weights.to(dtype=self.dtype)[:, None]
        value_flat = value.reshape(
            weight_count,
            self.horizon + 1,
            self.state_count,
        )

        for rem in range(1, self.horizon + 1):
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            review_weight = torch.clamp(rem - interval + 1, min=0).to(
                dtype=self.dtype
            ) / float(rem)
            future_weight = torch.clamp(rem - interval, min=0).to(
                dtype=self.dtype
            ) / float(rem)
            value_rem = self._uniform_termination_memorized_for_state_intervals(
                interval=interval,
                rem=rem,
            )

            if bool(review_weight.any().item()):
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = torch.zeros_like(value_rem)
                    for corner_idx in range(4):
                        future_value += (
                            next_weight[:, rating_idx, corner_idx, :]
                            * value_flat[
                                batch_idx,
                                future_rem,
                                next_idx[:, rating_idx, corner_idx, :],
                            ]
                        )
                    review_minutes = self.review_cost_minutes[rating - 1]
                    value_rem += prob[:, rating_idx, :] * (
                        future_weight * future_value
                        - review_weight * weight_penalty * review_minutes
                    )

            value[:, rem] = value_rem.reshape(weight_count, self.s_count, self.d_count)

        return value

    def _improve_stationary_uniform_termination_policy_batch(
        self,
        *,
        policy: torch.Tensor,
        occupancy: torch.Tensor,
        value: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_count = int(policy.shape[0])
        interval_count = self.horizon + 1
        action_scores = torch.zeros(
            (weight_count, interval_count, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        valid_actions = (
            self._stationary_attainable_interval_mask(
                torch.arange(1, self.horizon + 2, device=self.device, dtype=torch.int64)
            )
            .reshape(interval_count, self.s_count, 1)
            .expand(interval_count, self.s_count, self.d_count)
            .reshape(interval_count, self.state_count)
        )
        value_for_lookup = value.permute(1, 2, 3, 0).contiguous()
        weights = cost_weights.to(dtype=self.dtype).view(1, 1, 1, weight_count)

        for rem in range(1, self.horizon + 1):
            rem_occupancy = occupancy[:, rem, :].reshape(
                weight_count,
                self.s_count,
                self.d_count,
            )
            if float(rem_occupancy.sum().item()) <= 0.0:
                continue
            for start in range(1, self.horizon + 2, self.interval_chunk_size):
                stop = min(self.horizon + 2, start + self.interval_chunk_size)
                intervals = torch.arange(
                    start,
                    stop,
                    device=self.device,
                    dtype=torch.int64,
                )
                candidate_value = self._candidate_uniform_interval_value_batch(
                    intervals=intervals,
                    rem=rem,
                    cost_weights=weights,
                    value=value_for_lookup,
                )
                mask = self._stationary_attainable_interval_mask(intervals)
                candidate_value = torch.where(
                    mask[:, :, None, None],
                    candidate_value,
                    torch.zeros_like(candidate_value),
                )
                score = rem_occupancy[:, None, :, :] * candidate_value.permute(
                    3,
                    0,
                    1,
                    2,
                )
                action_scores[:, start - 1 : stop - 1] += score.reshape(
                    weight_count,
                    int(intervals.numel()),
                    self.state_count,
                )

        action_scores = torch.where(
            valid_actions[None, :, :],
            action_scores,
            torch.full_like(action_scores, -math.inf),
        )
        best_score, best_idx = action_scores.max(dim=1)
        best_interval = (best_idx + 1).reshape(
            weight_count,
            self.s_count,
            self.d_count,
        )
        current_interval = self._intervals_for_retention_policy(policy)
        current_score = action_scores.gather(
            1,
            (current_interval.reshape(weight_count, self.state_count) - 1)[:, None, :],
        ).squeeze(1)
        state_occupancy = occupancy[:, 1:, :].sum(dim=1)
        visited = state_occupancy > 0.0
        improvement = best_score - current_score
        masked_improvement = torch.where(
            visited,
            improvement,
            torch.full_like(improvement, -math.inf),
        )
        residual = masked_improvement.max(dim=1).values
        residual = torch.where(
            visited.any(dim=1),
            residual,
            torch.zeros_like(residual),
        )
        new_policy = torch.where(
            visited.reshape(weight_count, self.s_count, self.d_count),
            self._retention_for_interval_grid_batch(
                best_interval,
                terminal_interval=self.horizon + 1,
            ),
            policy,
        )
        return (
            new_policy,
            residual,
            visited.reshape(weight_count, self.s_count, self.d_count),
        )

    def _rollout_uniform_termination_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(policy.shape[0])
        occupancy = torch.zeros(
            (weight_count, self.horizon + 1, self.state_count),
            device=self.device,
            dtype=self.dtype,
        )
        flat_occupancy = occupancy.reshape(-1)
        batch_offsets = (
            torch.arange(weight_count, device=self.device, dtype=torch.int64)
            .view(weight_count, 1)
            .mul((self.horizon + 1) * self.state_count)
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            state_idx, state_weight = self._initial_state_kernel(rating)
            for corner_idx in range(4):
                target = (
                    batch_offsets
                    + self.horizon * self.state_count
                    + state_idx[corner_idx]
                )
                amount = (prob * state_weight[corner_idx]).expand(weight_count)
                flat_occupancy.scatter_add_(0, target.reshape(-1), amount)

        interval, prob, next_idx, next_weight = self._policy_tables_batch(policy)

        for rem in range(self.horizon, 0, -1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            future_weight = torch.clamp(rem - interval, min=0).to(
                dtype=self.dtype
            ) / float(rem)
            source = current * future_weight
            if float(source.sum().item()) <= 0.0:
                continue

            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            target_base = batch_offsets + future_rem * self.state_count
            for rating_idx in range(4):
                for corner_idx in range(4):
                    amount = (
                        source
                        * prob[:, rating_idx, :]
                        * next_weight[:, rating_idx, corner_idx, :]
                    )
                    target = target_base + next_idx[:, rating_idx, corner_idx, :]
                    flat_occupancy.scatter_add_(
                        0,
                        target.reshape(-1),
                        amount.reshape(-1),
                    )

        return occupancy

    def _metrics_from_uniform_termination_occupancy_batch(
        self,
        *,
        policy: torch.Tensor,
        cost_weights: torch.Tensor,
    ) -> list[OracleMetrics]:
        weight_count = int(policy.shape[0])
        occupancy = self._rollout_uniform_termination_occupancy_batch(policy=policy)
        interval, prob, _, _ = self._policy_tables_batch(policy)
        total_mem = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_minutes = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_reviews = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        total_lapses = torch.zeros(weight_count, device=self.device, dtype=self.dtype)
        learning_minutes = (self.first_rating_prob * self.learning_cost_minutes).sum()
        total_minutes += learning_minutes
        expected_review_minutes = (prob * self.review_cost_minutes.view(1, 4, 1)).sum(
            dim=1
        )

        for rem in range(1, self.horizon + 1):
            current = occupancy[:, rem, :]
            if float(current.sum().item()) <= 0.0:
                continue
            immediate_mem = self._uniform_termination_memorized_for_state_intervals(
                interval=interval,
                rem=rem,
            )
            total_mem += (current * immediate_mem).sum(dim=1)

            review_weight = torch.clamp(rem - interval + 1, min=0).to(
                dtype=self.dtype
            ) / float(rem)
            source = current * review_weight
            if float(source.sum().item()) <= 0.0:
                continue
            total_minutes += (source * expected_review_minutes).sum(dim=1)
            total_reviews += source.sum(dim=1)
            total_lapses += (source * prob[:, 0, :]).sum(dim=1)

        day_count = float(self.days)
        objectives = total_mem / day_count - cost_weights * (total_minutes / day_count)
        metrics: list[OracleMetrics] = []
        for idx in range(weight_count):
            reviews_float = float(total_reviews[idx].item())
            lapses_float = float(total_lapses[idx].item())
            observed_retention = (
                1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
            )
            metrics.append(
                OracleMetrics(
                    card_expected_retrievability=float(
                        (total_mem[idx] / day_count).item()
                    ),
                    card_minutes_per_day=float((total_minutes[idx] / day_count).item()),
                    card_reviews_per_day=reviews_float / day_count,
                    card_total_reviews=reviews_float,
                    card_total_lapses=lapses_float,
                    card_total_cost_seconds=float((total_minutes[idx] * 60.0).item()),
                    observed_retention=observed_retention,
                    scalar_objective=float(objectives[idx].item()),
                    runtime_s=0.0,
                )
            )
        return metrics

    def _candidate_uniform_interval_value_batch(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
        cost_weights: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        weight_count = int(cost_weights.numel())
        s_count = int(self.s_grid.numel())
        d_count = int(self.d_grid.numel())
        interval_count = int(intervals.numel())
        immediate_mem = self._uniform_termination_memorized_by_interval(
            intervals=intervals,
            rem=rem,
        )
        candidate_value = (
            immediate_mem[:, :, None, None]
            .expand(interval_count, s_count, d_count, weight_count)
            .clone()
        )

        review_mask = intervals <= rem
        if not bool(review_mask.any().item()):
            return candidate_value

        elapsed = intervals.to(dtype=self.dtype)
        retrievability = self._forgetting_curve(
            elapsed[:, None],
            self.s_grid[None, :],
        )
        future_rem = torch.clamp(rem - intervals, min=0).to(torch.int64)
        future_rem_idx = future_rem[:, None, None].expand(
            interval_count,
            s_count,
            d_count,
        )
        rem_float = float(rem)
        review_weight = (
            torch.clamp(rem - intervals + 1, min=0).to(dtype=self.dtype) / rem_float
        )
        future_weight = (
            torch.clamp(rem - intervals, min=0).to(dtype=self.dtype) / rem_float
        )

        for rating_idx, rating in enumerate(range(1, 5)):
            if rating == 1:
                prob = 1.0 - retrievability
            else:
                prob = retrievability * self.review_rating_prob[rating_idx - 1]
            next_s, next_d = self._next_state_interval_candidates(
                elapsed=elapsed,
                retrievability=retrievability,
                rating=rating,
            )
            future_value = self._interpolate_interval_value(
                value=value,
                rem_idx=future_rem_idx,
                s=next_s,
                d=next_d,
            )
            review_minutes = self.review_cost_minutes[rating - 1]
            prob_expanded = prob[:, :, None, None]
            candidate_value += prob_expanded * (
                future_weight[:, None, None, None] * future_value
                - review_weight[:, None, None, None] * cost_weights * review_minutes
            )

        return candidate_value

    def _uniform_termination_memorized_by_interval(
        self,
        *,
        intervals: torch.Tensor,
        rem: int,
    ) -> torch.Tensor:
        clamped = torch.clamp(intervals.to(torch.int64), max=rem)
        prefix_idx = torch.clamp(intervals.to(torch.int64) - 1, max=rem)
        terminal_count = torch.clamp(rem - intervals.to(torch.int64) + 1, min=0).to(
            dtype=self.dtype
        )
        return (
            self.cumulative_memorized_by_day.index_select(0, prefix_idx)
            + terminal_count[:, None] * self.memorized_by_day.index_select(0, clamped)
        ) / float(rem)

    def _uniform_termination_memorized_for_state_intervals(
        self,
        *,
        interval: torch.Tensor,
        rem: int,
    ) -> torch.Tensor:
        interval_idx = interval.to(torch.int64)
        flat_s_idx = self._flat_s_idx[None, :].expand_as(interval_idx)
        clamped = torch.clamp(interval_idx, max=rem)
        prefix_idx = torch.clamp(interval_idx - 1, max=rem)
        terminal_count = torch.clamp(rem - interval_idx + 1, min=0).to(dtype=self.dtype)
        return (
            self.cumulative_memorized_by_day[prefix_idx, flat_s_idx]
            + terminal_count * self.memorized_by_day[clamped, flat_s_idx]
        ) / float(rem)
