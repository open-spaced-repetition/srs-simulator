from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
    validate_retention_values_for_model,
)
from experiments.single_card_tradeoff.tradeoff import DEFAULT_TARGET_RETENTIONS
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds
from simulator.scheduler_spec import format_float

DEFAULT_COST_WEIGHTS = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]


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


def parse_csv_floats(value: str, *, name: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value '{item}'.") from exc
        if not math.isfinite(parsed):
            raise SystemExit(f"{name} values must be finite.")
        values.append(parsed)
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def scalar_objective(metrics: Any, cost_weight: float) -> float:
    return float(metrics.card_expected_retrievability) - cost_weight * float(
        metrics.card_minutes_per_day
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a single-card FSRS6 oracle frontier with grid DP.",
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help="Comma-separated scalarization weights for oracle frontier points.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to the oracle.",
    )
    parser.add_argument(
        "--s-grid-size",
        type=int,
        default=64,
        help="Number of log-spaced stability grid points.",
    )
    parser.add_argument(
        "--d-grid-size",
        type=int,
        default=32,
        help="Number of linearly spaced difficulty grid points.",
    )
    parser.add_argument(
        "--baseline-particles",
        type=int,
        default=10_000,
        help="Particles for static-FSRS baseline rows. Set 0 to skip baselines.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/oracle_frontier.csv"),
    )
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


class FSRS6GridOracle:
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
        self._s_grid_idx = torch.arange(self.s_grid.numel(), device=self.device)
        self.memorized_by_day = self._precompute_memorized_by_day()
        self.transitions = self._precompute_transitions()

    def estimate(self, cost_weight: float, *, progress: bool = False) -> OracleMetrics:
        return self.solve(
            cost_weight,
            progress=progress,
            capture_policy=False,
        ).metrics

    def solve(
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
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            s_idx = self._s_to_idx(s0)
            d_idx = self._d_to_idx(d0)
            total_mem += prob * memorized[self.horizon, s_idx, d_idx]
            total_minutes += prob * (
                self.learning_cost_minutes[rating - 1]
                + minutes[self.horizon, s_idx, d_idx]
            )
            total_reviews += prob * reviews[self.horizon, s_idx, d_idx]
            total_lapses += prob * lapses[self.horizon, s_idx, d_idx]

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
        cont_2d = cont_mask[:, None].expand_as(self.s_mesh)
        for rating_idx, rating in enumerate(range(1, 5)):
            prob = transition.prob[rating_idx][:, None].expand_as(self.s_mesh)
            s_idx = transition.next_s_idx[rating_idx]
            d_idx = transition.next_d_idx[rating_idx]
            future_value = value[rem_idx, s_idx, d_idx]
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = torch.where(cont_2d, prob, torch.zeros_like(prob)).unsqueeze(2)
            candidate_value += weighted * (future_value - cost_weights * review_minutes)

        return candidate_value

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
            for rating in range(1, 5):
                next_s, next_d = self._next_state_grid(
                    elapsed=elapsed,
                    retrievability=retrievability,
                    rating=rating,
                )
                next_s_idx.append(self._s_to_idx(next_s))
                next_d_idx.append(self._d_to_idx(next_d))
            transitions.append(
                TransitionCache(
                    interval=interval,
                    prob=probs,
                    next_s_idx=tuple(next_s_idx),  # type: ignore[arg-type]
                    next_d_idx=tuple(next_d_idx),  # type: ignore[arg-type]
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

    def _precompute_memorized_by_day(self) -> torch.Tensor:
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
        policies: list[torch.Tensor] = []
        metrics_by_weight: list[OracleMetrics] = []
        iterations: list[int] = []
        converged: list[bool] = []
        residuals: list[float] = []
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
            for weight_idx, cost_weight in enumerate(cost_weights):
                result = self._solve_single_stationary_finite_policy(
                    cost_weight=float(cost_weight),
                    finite_policy=finite_policies[weight_idx],
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    progress_bar=progress_bar,
                )
                policy, metrics, iteration_count, is_converged, residual = result
                policies.append(policy)
                metrics_by_weight.append(metrics)
                iterations.append(iteration_count)
                converged.append(is_converged)
                residuals.append(residual)
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
        interval, prob, next_idx = self._select_stationary_policy_tables(policy)

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
                    future_value = value[future_rem, next_idx[rating_idx]]
                    future_mem = memorized[future_rem, next_idx[rating_idx]]
                    future_minutes = minutes[future_rem, next_idx[rating_idx]]
                    future_reviews = reviews[future_rem, next_idx[rating_idx]]
                    future_lapses = lapses[future_rem, next_idx[rating_idx]]
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
        interval, prob, next_idx = self._select_stationary_policy_tables(policy)

        for rem in range(1, self.horizon + 1):
            cont_mask = interval <= rem
            future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
            active_days = torch.minimum(interval, torch.full_like(interval, rem))
            value_rem = self._memorized_sum_flat(active_days)

            if cont_mask.any():
                for rating_idx, rating in enumerate(range(1, 5)):
                    future_value = value[future_rem, next_idx[rating_idx]]
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
            s0, d0 = self._init_state_scalar(rating)
            s_idx = self._s_to_idx(s0)
            d_idx = self._d_to_idx(d0)
            state_idx = s_idx * self.d_count + d_idx
            total_value += prob * value[self.horizon, state_idx]
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
        interval, prob, next_idx = self._action_tables

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
                    future_value = value[future_rem, next_idx[:, rating_idx, :]]
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

    def _rollout_occupancy(
        self, *, policy: torch.Tensor, stationary: bool
    ) -> torch.Tensor:
        occupancy = torch.zeros(
            (self.horizon + 1, self.s_count, self.d_count),
            device=self.device,
            dtype=self.dtype,
        )
        for rating in range(1, 5):
            prob = self.first_rating_prob[rating - 1]
            s0, d0 = self._init_state_scalar(rating)
            s_idx = self._s_to_idx(s0)
            d_idx = self._d_to_idx(d0)
            occupancy[self.horizon, s_idx, d_idx] += prob

        flat_occupancy = occupancy.reshape(-1)
        selected_interval: torch.Tensor | None = None
        selected_prob: torch.Tensor | None = None
        selected_next_idx: torch.Tensor | None = None
        if stationary:
            selected_interval, selected_prob, selected_next_idx = (
                self._select_stationary_policy_tables(policy)
            )

        for rem in range(self.horizon, 0, -1):
            current = occupancy[rem].reshape(-1)
            if float(current.sum().item()) <= 0.0:
                continue
            if not stationary:
                selected_interval, selected_prob, selected_next_idx = (
                    self._select_stationary_policy_tables(policy[rem])
                )
            if (
                selected_interval is None
                or selected_prob is None
                or selected_next_idx is None
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
                amount = source * selected_prob[rating_idx]
                target = rem_offset + selected_next_idx[rating_idx]
                flat_occupancy.scatter_add_(0, target, amount)

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
            s0, d0 = self._init_state_scalar(rating)
            s_idx = self._s_to_idx(s0)
            d_idx = self._d_to_idx(d0)
            state_idx = s_idx * self.d_count + d_idx
            total_mem += prob * memorized[self.horizon, state_idx]
            total_minutes += prob * (
                self.learning_cost_minutes[rating - 1]
                + minutes[self.horizon, state_idx]
            )
            total_reviews += prob * reviews[self.horizon, state_idx]
            total_lapses += prob * lapses[self.horizon, state_idx]

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

    def _precompute_stationary_action_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intervals: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        next_indices: list[torch.Tensor] = []
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
                        (
                            transition.next_s_idx[rating_idx] * self.d_count
                            + transition.next_d_idx[rating_idx]
                        ).reshape(-1)
                        for rating_idx in range(4)
                    ],
                    dim=0,
                ).to(dtype=torch.int64)
            )
        return (
            torch.stack(intervals, dim=0).to(dtype=torch.int64),
            torch.stack(probs, dim=0),
            torch.stack(next_indices, dim=0).to(dtype=torch.int64),
        )

    def _select_stationary_policy_tables(
        self,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interval, prob, next_idx = self._action_tables
        policy_flat = policy.reshape(-1).to(dtype=torch.int64)
        gather_idx = policy_flat[None, :]
        selected_interval = interval.gather(0, gather_idx).squeeze(0)
        table_idx = policy_flat.view(1, 1, -1)
        selected_prob = prob.gather(0, table_idx.expand(1, 4, -1)).squeeze(0)
        selected_next_idx = next_idx.gather(0, table_idx.expand(1, 4, -1)).squeeze(0)
        return selected_interval, selected_prob, selected_next_idx

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
        policies: list[torch.Tensor] = []
        gains: list[torch.Tensor] = []
        iterations: list[int] = []
        converged: list[bool] = []
        residuals: list[float] = []
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
            for cost_weight in cost_weights:
                result = self._solve_single_policy(
                    cost_weight=float(cost_weight),
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    progress_bar=progress_bar,
                )
                policy, gain, iteration_count, is_converged, residual = result
                policies.append(
                    policy.reshape(self.s_grid.numel(), self.d_grid.numel())
                )
                gains.append(gain)
                iterations.append(iteration_count)
                converged.append(is_converged)
                residuals.append(residual)
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


class FSRS6IntervalOracle(FSRS6GridOracle):
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
            s_idx, d_idx = self._next_state_interval_candidates(
                elapsed=elapsed,
                retrievability=retrievability,
                rating=rating,
            )
            future_value = value[future_rem_idx, s_idx, d_idx]
            review_minutes = self.review_cost_minutes[rating - 1]
            weighted = (prob * cont_weight)[:, :, None, None]
            candidate_value += weighted * (future_value - cost_weights * review_minutes)

        return candidate_value

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
            self._s_to_idx(torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max)),
            self._d_to_idx(torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)),
        )


def row_from_metrics(
    *,
    args: argparse.Namespace,
    scheduler: str,
    scheduler_spec: str,
    metrics: Any,
    goal_cost_weight: float | None,
    desired_retention: float | None,
    runtime_s: float,
    scalar: float | None,
    delta_vs_best_fsrs: float | None,
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    return {
        "environment": args.env,
        "scheduler": scheduler,
        "scheduler_spec": scheduler_spec,
        "goal_cost_weight": goal_cost_weight,
        "desired_retention": desired_retention,
        "fixed_interval": None,
        "seed": args.seed,
        "days": args.days,
        "particles": 0 if scheduler == "oracle_grid" else args.baseline_particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "scalar_objective": scalar,
        "delta_vs_best_fsrs": delta_vs_best_fsrs,
        "runtime_s": runtime_s,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "scheduler",
        "scheduler_spec",
        "goal_cost_weight",
        "desired_retention",
        "fixed_interval",
        "seed",
        "days",
        "particles",
        "deck_scale",
        "card_expected_retrievability",
        "card_minutes_per_day",
        "card_reviews_per_day",
        "card_total_reviews",
        "card_total_lapses",
        "card_total_cost_seconds",
        "observed_retention",
        "deck_expected_memorized",
        "deck_minutes_per_day",
        "deck_reviews_per_day",
        "scalar_objective",
        "delta_vs_best_fsrs",
        "runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scheduler"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in groups.items():
        if label == "fsrs6_static":
            group = sorted(group, key=lambda row: float(row["desired_retention"]))
        else:
            group = sorted(group, key=lambda row: float(row["goal_cost_weight"]))
        ax.plot(
            [row["deck_expected_memorized"] for row in group],
            [row["deck_minutes_per_day"] for row in group],
            marker="o",
            linewidth=1.4 if label == "oracle_grid" else 1.0,
            alpha=0.9 if label == "oracle_grid" else 0.5,
            label=label,
        )
    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title("FSRS6 single-card oracle frontier estimate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    from experiments.single_card_tradeoff.uvfa_ppo import evaluate_static_fsrs

    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.baseline_particles < 0:
        raise SystemExit("--baseline-particles must be >= 0.")
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    fsrs_config = load_single_card_fsrs6_config(args)
    oracle = FSRS6GridOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
        fsrs_weights=fsrs_config.fsrs_weights,
        first_rating_prob=fsrs_config.first_rating_prob,
        review_rating_prob=fsrs_config.review_rating_prob,
        learning_costs=fsrs_config.learning_costs,
        review_costs=fsrs_config.review_costs,
    )
    rows: list[dict[str, Any]] = []
    fsrs_metrics_by_weight: dict[float, float] = {}

    if args.baseline_particles > 0:
        baseline_ns = argparse.Namespace(days=args.days)
        for retention in action_retentions:
            metrics = evaluate_static_fsrs(
                args=baseline_ns,
                device=torch.device("cpu"),
                retention=retention,
                particles=args.baseline_particles,
                seed=args.seed + 10_000 + int(round(retention * 10_000)),
                fsrs_config=fsrs_config,
            )
            for cost_weight in cost_weights:
                scalar = scalar_objective(metrics, cost_weight)
                current = fsrs_metrics_by_weight.get(cost_weight)
                if current is None or scalar > current:
                    fsrs_metrics_by_weight[cost_weight] = scalar
            rows.append(
                row_from_metrics(
                    args=args,
                    scheduler="fsrs6_static",
                    scheduler_spec=f"fsrs@{format_float(retention)}",
                    metrics=metrics,
                    goal_cost_weight=None,
                    desired_retention=retention,
                    runtime_s=0.0,
                    scalar=None,
                    delta_vs_best_fsrs=None,
                )
            )

    for cost_weight in cost_weights:
        metrics = oracle.estimate(cost_weight, progress=not args.no_progress)
        best_fsrs = fsrs_metrics_by_weight.get(cost_weight)
        delta = metrics.scalar_objective - best_fsrs if best_fsrs is not None else None
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="oracle_grid",
                scheduler_spec=f"oracle_grid_{args.s_grid_size}x{args.d_grid_size}",
                metrics=metrics,
                goal_cost_weight=cost_weight,
                desired_retention=None,
                runtime_s=metrics.runtime_s,
                scalar=metrics.scalar_objective,
                delta_vs_best_fsrs=delta,
            )
        )
        print(
            " ".join(
                [
                    f"oracle w={format_float(cost_weight)}",
                    f"card_mem={metrics.card_expected_retrievability:.4f}",
                    f"card_min/day={metrics.card_minutes_per_day:.6f}",
                    f"scalar={metrics.scalar_objective:.6f}",
                    f"delta_fsrs={delta:.6f}" if delta is not None else "delta_fsrs=NA",
                    f"runtime_s={metrics.runtime_s:.2f}",
                ]
            )
        )

    write_csv(args.out, rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        write_plot(plot_path, rows)
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {args.out}")


if __name__ == "__main__":
    main()
