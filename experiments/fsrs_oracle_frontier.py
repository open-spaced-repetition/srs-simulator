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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff import DEFAULT_TARGET_RETENTIONS
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
        default=Path("logs/single_card_tradeoff/oracle_frontier.csv"),
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
    ) -> None:
        if days <= 1:
            raise ValueError("days must be > 1.")
        if s_grid_size < 8 or d_grid_size < 8:
            raise ValueError("grid sizes must be >= 8.")
        if any(value <= 0.0 or value >= 1.0 for value in action_retentions):
            raise ValueError("action retentions must be within (0, 1).")

        self.days = int(days)
        self.horizon = int(days - 1)
        self.dtype = dtype
        self.bounds = Bounds()
        self.weights = torch.tensor(DEFAULT_FSRS6_WEIGHTS, dtype=dtype)
        self.decay = -self.weights[20]
        self.factor = torch.pow(torch.tensor(0.9, dtype=dtype), 1.0 / self.decay) - 1.0
        self.init_d = torch.clamp(
            self.weights[4] - torch.exp(self.weights[5] * 3.0) + 1.0,
            self.bounds.d_min,
            self.bounds.d_max,
        )
        self.action_retentions = torch.tensor(list(action_retentions), dtype=dtype)
        self.action_retention_factor = (
            torch.pow(self.action_retentions, 1.0 / self.decay) - 1.0
        )
        self.first_rating_prob = torch.tensor(DEFAULT_FIRST_RATING_PROB, dtype=dtype)
        self.review_rating_prob = torch.tensor(DEFAULT_REVIEW_RATING_PROB, dtype=dtype)
        self.learning_cost_minutes = (
            torch.tensor(DEFAULT_STATE_RATING_COSTS.learning, dtype=dtype) / 60.0
        )
        self.review_cost_minutes = (
            torch.tensor(DEFAULT_STATE_RATING_COSTS.review, dtype=dtype) / 60.0
        )

        self.log_s_min = math.log(self.bounds.s_min)
        self.log_s_max = math.log(self.bounds.s_max)
        self.s_grid = torch.exp(
            torch.linspace(self.log_s_min, self.log_s_max, s_grid_size, dtype=dtype)
        )
        self.d_grid = torch.linspace(
            self.bounds.d_min,
            self.bounds.d_max,
            d_grid_size,
            dtype=dtype,
        )
        self.s_mesh = self.s_grid[:, None].expand(s_grid_size, d_grid_size)
        self.d_mesh = self.d_grid[None, :].expand(s_grid_size, d_grid_size)
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
        value = torch.zeros(shape, dtype=self.dtype)
        memorized = torch.zeros_like(value)
        minutes = torch.zeros_like(value)
        reviews = torch.zeros_like(value)
        lapses = torch.zeros_like(value)
        policy = torch.zeros(shape, dtype=torch.int64) if capture_policy else None

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

        total_mem = torch.tensor(0.0, dtype=self.dtype)
        total_minutes = torch.tensor(0.0, dtype=self.dtype)
        total_reviews = torch.tensor(0.0, dtype=self.dtype)
        total_lapses = torch.tensor(0.0, dtype=self.dtype)
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
        weight_tensor = torch.tensor(list(cost_weights), dtype=self.dtype)
        weight_count = int(weight_tensor.numel())
        shape = (
            self.horizon + 1,
            self.s_grid.numel(),
            self.d_grid.numel(),
            weight_count,
        )
        value = torch.zeros(shape, dtype=self.dtype)
        policy = torch.zeros(shape, dtype=torch.int64)
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
        immediate_mem = self._memorized_sum(self.s_grid, active_days)[:, None]
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
        immediate_mem = self._memorized_sum(self.s_grid, active_days)[:, None]
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
        rating_f = torch.tensor(float(rating), dtype=self.dtype)
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
            times = torch.arange(1, day_int + 1, dtype=self.dtype)
            out[idx] = self._forgetting_curve(
                times.unsqueeze(0),
                s.index_select(0, idx).unsqueeze(1),
            ).sum(dim=1)
        return out

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
        "environment": "fsrs6_default",
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
    from experiments.uvfa_ppo_single_card import evaluate_static_fsrs

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
    oracle = FSRS6GridOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
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
