from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from experiments.single_card_tradeoff.core.retention_space import (
    validate_retention_values_for_model,
)
from experiments.single_card_tradeoff.core.types import SimMetrics
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds


class FSRS6SingleCardBatch:
    def __init__(
        self,
        *,
        days: int,
        env_count: int,
        cost_weights: Sequence[float],
        action_retentions: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        exact_memory: bool,
        goal_norm_max: float | None = None,
        obs_mode: str = "basic",
        max_interval_days: int | None = None,
        fsrs_weights: Sequence[float] | None = None,
        first_rating_prob: Sequence[float] | None = None,
        review_rating_prob: Sequence[float] | None = None,
        learning_costs: Sequence[float] | None = None,
        review_costs: Sequence[float] | None = None,
    ) -> None:
        if days <= 1:
            raise ValueError("days must be > 1.")
        if env_count <= 0:
            raise ValueError("env_count must be > 0.")
        if any(weight < 0.0 for weight in cost_weights):
            raise ValueError("cost weights must be >= 0.")
        validate_retention_values_for_model(
            action_retentions,
            name="action retention",
        )
        if obs_mode not in {
            "basic",
            "rich",
            "belief",
            "oracle",
            "oracle_rho",
            "oracle_rho4",
            "oracle_rho3",
            "oracle_stationary",
        }:
            raise ValueError(
                "obs_mode must be 'basic', 'rich', 'belief', 'oracle', "
                "'oracle_rho', 'oracle_rho4', 'oracle_rho3', or "
                "'oracle_stationary'."
            )

        self.days = int(days)
        self.env_count = int(env_count)
        self.device = device
        self.dtype = dtype
        self.exact_memory = exact_memory
        self.obs_mode = obs_mode
        self.max_interval_days = (
            int(max_interval_days) if max_interval_days is not None else self.days * 4
        )
        if self.max_interval_days < 1:
            raise ValueError("max_interval_days must be >= 1.")
        self.bounds = Bounds()
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        resolved_weights = (
            DEFAULT_FSRS6_WEIGHTS if fsrs_weights is None else tuple(fsrs_weights)
        )
        if len(resolved_weights) != 21:
            raise ValueError("FSRS6 weights must contain 21 values.")
        self.weights = torch.tensor(resolved_weights, device=device, dtype=dtype)
        self.decay = -self.weights[20]
        self.factor = (
            torch.pow(torch.tensor(0.9, device=device, dtype=dtype), 1.0 / self.decay)
            - 1.0
        )
        self.init_d = torch.clamp(
            self.weights[4] - torch.exp(self.weights[5] * 3.0) + 1.0,
            self.bounds.d_min,
            self.bounds.d_max,
        )
        self.cost_weight_values = torch.tensor(
            list(cost_weights), device=device, dtype=dtype
        )
        self._goal_norm_max = max(
            1.0,
            float(goal_norm_max)
            if goal_norm_max is not None
            else float(max(cost_weights)),
        )
        self.action_retentions = torch.tensor(
            list(action_retentions), device=device, dtype=dtype
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
            resolved_first_prob, device=device, dtype=dtype
        )
        self.review_rating_prob = torch.tensor(
            resolved_review_prob, device=device, dtype=dtype
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
        self.learning_costs = torch.tensor(
            resolved_learning_costs, device=device, dtype=dtype
        )
        self.review_costs = torch.tensor(
            resolved_review_costs, device=device, dtype=dtype
        )

        self.s = torch.empty(env_count, device=device, dtype=dtype)
        self.d = torch.empty_like(self.s)
        self.day = torch.empty(env_count, device=device, dtype=torch.int64)
        self.pending_cost_seconds = torch.empty_like(self.s)
        self.goal_weight = torch.empty_like(self.s)
        self.last_interval = torch.empty_like(self.s)
        self.last_rating = torch.empty(env_count, device=device, dtype=torch.int64)
        self.done = torch.empty(env_count, device=device, dtype=torch.bool)
        self.total_memorized = torch.empty_like(self.s)
        self.total_cost_seconds = torch.empty_like(self.s)
        self.total_reviews = torch.empty(env_count, device=device, dtype=torch.int64)
        self.total_lapses = torch.empty(env_count, device=device, dtype=torch.int64)
        self.reset_all()

    @property
    def obs_dim(self) -> int:
        if self.obs_mode == "basic":
            return 7
        if self.obs_mode == "oracle":
            return 4
        if self.obs_mode == "oracle_rho":
            return 5
        if self.obs_mode == "oracle_rho4":
            return 4
        if self.obs_mode == "oracle_rho3":
            return 3
        if self.obs_mode == "oracle_stationary":
            return 3
        if self.obs_mode == "rich":
            return 13
        return 10

    @property
    def action_count(self) -> int:
        return int(self.action_retentions.numel())

    @property
    def max_goal_weight(self) -> float:
        return self._goal_norm_max

    def reset_all(self, goal_weight: float | None = None) -> torch.Tensor:
        idx = torch.arange(self.env_count, device=self.device)
        self.reset_indices(idx, goal_weight=goal_weight)
        return self.obs()

    def reset_indices(
        self,
        idx: torch.Tensor,
        *,
        goal_weight: float | None = None,
    ) -> None:
        if idx.numel() == 0:
            return
        count = int(idx.numel())
        if goal_weight is None:
            goal_idx = torch.randint(
                self.cost_weight_values.numel(),
                (count,),
                device=self.device,
                generator=self.generator,
            )
            goals = self.cost_weight_values.index_select(0, goal_idx)
        else:
            goals = torch.full(
                (count,), goal_weight, device=self.device, dtype=self.dtype
            )

        first_weights = self.first_rating_prob.expand(count, -1)
        rating = (
            torch.multinomial(
                first_weights,
                num_samples=1,
                replacement=True,
                generator=self.generator,
            )
            .squeeze(1)
            .to(torch.int64)
            + 1
        )
        s_init, d_init = self._init_state(rating)
        self.s[idx] = s_init
        self.d[idx] = d_init
        self.day[idx] = 0
        self.pending_cost_seconds[idx] = self.learning_costs.index_select(0, rating - 1)
        self.goal_weight[idx] = goals
        self.last_interval[idx] = 0.0
        self.last_rating[idx] = rating
        self.done[idx] = False
        self.total_memorized[idx] = 0.0
        self.total_cost_seconds[idx] = 0.0
        self.total_reviews[idx] = 0
        self.total_lapses[idx] = 0

    def obs(self) -> torch.Tensor:
        log_s_min = math.log(self.bounds.s_min)
        log_s_max = math.log(self.bounds.s_max)
        log_s = torch.log(torch.clamp(self.s, min=self.bounds.s_min))
        s_norm = (log_s - log_s_min) / (log_s_max - log_s_min)
        d_norm = (self.d - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min)
        day_norm = self.day.to(dtype=self.dtype) / float(self.days - 1)
        remaining = torch.clamp((self.days - 1) - self.day, min=0).to(dtype=self.dtype)
        remaining_norm = remaining / float(self.days - 1)
        log_remaining_norm = torch.log1p(remaining) / math.log1p(float(self.days - 1))
        rho = torch.log1p(remaining) - log_s
        rho_min = -log_s_max
        rho_max = math.log1p(float(self.days - 1)) - log_s_min
        rho_norm = (rho - rho_min) / (rho_max - rho_min)
        interval_norm = torch.log1p(
            torch.clamp(self.last_interval, min=0.0)
        ) / math.log1p(float(self.days * 4))
        rating_norm = (self.last_rating.to(dtype=self.dtype) - 1.0) / 3.0
        max_goal = max(1.0, self.max_goal_weight)
        goal_norm = torch.log1p(self.goal_weight) / math.log1p(max_goal)
        goal_linear = self.goal_weight / max_goal
        pending_norm = self.pending_cost_seconds / 60.0
        review_count_norm = self.total_reviews.to(dtype=self.dtype) / float(self.days)
        if self.obs_mode == "basic":
            return torch.stack(
                [
                    s_norm,
                    d_norm,
                    day_norm,
                    interval_norm,
                    rating_norm,
                    goal_norm,
                    pending_norm,
                ],
                dim=1,
            )

        if self.obs_mode == "oracle":
            return torch.stack(
                [
                    s_norm,
                    d_norm,
                    log_remaining_norm,
                    goal_norm,
                ],
                dim=1,
            )

        if self.obs_mode == "oracle_rho":
            return torch.stack(
                [
                    s_norm,
                    d_norm,
                    log_remaining_norm,
                    goal_norm,
                    rho_norm,
                ],
                dim=1,
            )

        if self.obs_mode == "oracle_rho4":
            return torch.stack(
                [
                    rho_norm,
                    d_norm,
                    goal_norm,
                    s_norm,
                ],
                dim=1,
            )

        if self.obs_mode == "oracle_rho3":
            return torch.stack(
                [
                    rho_norm,
                    d_norm,
                    goal_norm,
                ],
                dim=1,
            )

        if self.obs_mode == "oracle_stationary":
            return torch.stack(
                [
                    s_norm,
                    d_norm,
                    goal_norm,
                ],
                dim=1,
            )

        rating = self.last_rating.to(dtype=self.dtype)
        if self.obs_mode == "belief":
            return torch.stack(
                [
                    day_norm,
                    remaining_norm,
                    log_remaining_norm,
                    interval_norm,
                    review_count_norm,
                    (rating == 1.0).to(dtype=self.dtype),
                    (rating == 2.0).to(dtype=self.dtype),
                    (rating == 3.0).to(dtype=self.dtype),
                    (rating == 4.0).to(dtype=self.dtype),
                    pending_norm,
                ],
                dim=1,
            )

        return torch.stack(
            [
                s_norm,
                d_norm,
                day_norm,
                remaining_norm,
                log_remaining_norm,
                interval_norm,
                (rating == 1.0).to(dtype=self.dtype),
                (rating == 2.0).to(dtype=self.dtype),
                (rating == 3.0).to(dtype=self.dtype),
                (rating == 4.0).to(dtype=self.dtype),
                goal_norm,
                goal_linear,
                pending_norm,
            ],
            dim=1,
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_action = action.index_select(0, active).to(torch.int64)
        intervals = self._intervals_for_action(
            self.s.index_select(0, active), active_action
        )
        return self._step_active_intervals(active, intervals)

    def step_log_interval(
        self, log_interval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_log_interval = log_interval.index_select(0, active).to(dtype=self.dtype)
        intervals = self._intervals_for_log_interval(active_log_interval)
        return self._step_active_intervals(active, intervals)

    def step_intervals(
        self, intervals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_intervals = intervals.index_select(0, active).to(torch.int64)
        active_intervals = torch.clamp(
            active_intervals,
            min=1,
            max=self.max_interval_days,
        )
        return self._step_active_intervals(active, active_intervals)

    def step_retentions(
        self, retentions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_retentions = retentions.index_select(0, active).to(dtype=self.dtype)
        intervals = self._intervals_for_retentions(
            self.s.index_select(0, active),
            active_retentions,
        )
        return self._step_active_intervals(active, intervals)

    def _step_active_intervals(
        self,
        active: torch.Tensor,
        intervals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
        remaining = (self.days - 1) - self.day.index_select(0, active)
        memorized_days = torch.minimum(intervals, remaining)
        memorized = self._memorized_sum(self.s.index_select(0, active), memorized_days)
        cost_seconds = self.pending_cost_seconds.index_select(0, active)
        goals = self.goal_weight.index_select(0, active)
        reward_active = (memorized - goals * (cost_seconds / 60.0)) / float(self.days)
        reward[active] = reward_active
        self.total_memorized[active] += memorized
        self.total_cost_seconds[active] += cost_seconds

        next_day = self.day.index_select(0, active) + intervals
        terminal = next_day > (self.days - 1)
        if terminal.any():
            self.done[active[terminal]] = True

        continuing = ~terminal
        if continuing.any():
            cont_idx = active[continuing]
            elapsed = intervals[continuing].to(dtype=self.dtype)
            retrievability = self._forgetting_curve(
                elapsed,
                self.s.index_select(0, cont_idx),
            )
            fail = (
                torch.rand(
                    retrievability.shape,
                    device=self.device,
                    dtype=self.dtype,
                    generator=self.generator,
                )
                > retrievability
            )
            success_rating = (
                torch.multinomial(
                    self.review_rating_prob.expand(int(cont_idx.numel()), -1),
                    num_samples=1,
                    replacement=True,
                    generator=self.generator,
                )
                .squeeze(1)
                .to(torch.int64)
                + 2
            )
            rating = torch.where(fail, torch.ones_like(success_rating), success_rating)
            self._update_review(cont_idx, elapsed, rating, retrievability)
            self.day[cont_idx] = next_day[continuing]
            self.pending_cost_seconds[cont_idx] = self.review_costs.index_select(
                0, rating - 1
            )
            self.last_interval[cont_idx] = elapsed
            self.last_rating[cont_idx] = rating
            self.total_reviews[cont_idx] += 1
            self.total_lapses[cont_idx] += (rating == 1).to(torch.int64)

        return self.obs(), reward, self.done.clone()

    def metrics(self) -> SimMetrics:
        particles = float(self.env_count)
        days = float(self.days)
        total_reviews = float(self.total_reviews.sum().item())
        total_lapses = float(self.total_lapses.sum().item())
        total_cost = float(self.total_cost_seconds.sum().item())
        observed_retention = (
            1.0 - total_lapses / total_reviews if total_reviews > 0.0 else None
        )
        return SimMetrics(
            card_expected_retrievability=float(
                self.total_memorized.sum().item() / days / particles
            ),
            card_minutes_per_day=total_cost / days / 60.0 / particles,
            card_reviews_per_day=total_reviews / days / particles,
            card_total_reviews=total_reviews / particles,
            card_total_lapses=total_lapses / particles,
            card_total_cost_seconds=total_cost / particles,
            observed_retention=observed_retention,
        )

    def _init_state(self, rating: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rating_f = rating.to(dtype=self.dtype)
        s = self.weights.index_select(0, torch.clamp(rating - 1, min=0, max=3))
        d = self.weights[4] - torch.exp(self.weights[5] * (rating_f - 1.0)) + 1.0
        return s, torch.clamp(d, self.bounds.d_min, self.bounds.d_max)

    def _forgetting_curve(self, elapsed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.pow(
            1.0 + self.factor * elapsed / torch.clamp(s, min=self.bounds.s_min),
            self.decay,
        )

    def _intervals_for_action(
        self, s: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        retention_factor = self.action_retention_factor.index_select(0, action)
        interval = s / self.factor * retention_factor
        return torch.clamp(torch.round(interval), min=1.0).to(torch.int64)

    def _intervals_for_retentions(
        self, s: torch.Tensor, retention: torch.Tensor
    ) -> torch.Tensor:
        clipped_retention = torch.clamp(retention, min=1e-7, max=1.0 - 1e-7)
        retention_factor = torch.pow(clipped_retention, 1.0 / self.decay) - 1.0
        interval = s / self.factor * retention_factor
        return torch.clamp(
            torch.round(interval),
            min=1.0,
            max=float(self.max_interval_days),
        ).to(torch.int64)

    def intervals_for_retention(
        self, s: torch.Tensor, retention: float
    ) -> torch.Tensor:
        retention_tensor = torch.tensor(retention, device=self.device, dtype=self.dtype)
        return self._intervals_for_retentions(s, retention_tensor.expand_as(s))

    def _intervals_for_log_interval(self, log_interval: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(
            log_interval,
            min=0.0,
            max=math.log(float(self.max_interval_days)),
        )
        return torch.clamp(
            torch.round(torch.exp(clipped)),
            min=1.0,
            max=float(self.max_interval_days),
        ).to(torch.int64)

    def _memorized_sum(self, s: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        if self.exact_memory:
            return self._memorized_sum_exact(s, days)
        return self._memorized_sum_integral(s, days)

    def _memorized_sum_integral(
        self, s: torch.Tensor, days: torch.Tensor
    ) -> torch.Tensor:
        days_f = days.to(dtype=self.dtype)
        positive = days_f > 0
        safe_s = torch.clamp(s, min=self.bounds.s_min)
        rate = self.factor / safe_s
        upper = days_f + 0.5
        lower = torch.full_like(upper, 0.5)
        exponent = self.decay + 1.0
        integral = (
            torch.pow(1.0 + rate * upper, exponent)
            - torch.pow(1.0 + rate * lower, exponent)
        ) / (rate * exponent)
        return torch.where(positive, integral, torch.zeros_like(integral))

    def _memorized_sum_exact(self, s: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(s)
        unique_days = torch.unique(days)
        max_cells = 4_000_000
        for day_count in unique_days.tolist():
            day_int = int(day_count)
            if day_int <= 0:
                continue
            idx = (days == day_int).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            chunk = max(1, max_cells // day_int)
            times = torch.arange(
                1,
                day_int + 1,
                device=self.device,
                dtype=self.dtype,
            )
            for start in range(0, int(idx.numel()), chunk):
                sub_idx = idx[start : start + chunk]
                values = self._forgetting_curve(
                    times.unsqueeze(0),
                    s.index_select(0, sub_idx).unsqueeze(1),
                ).sum(dim=1)
                out[sub_idx] = values
        return out

    def _update_review(
        self,
        idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        retrievability: torch.Tensor,
    ) -> None:
        current_s = self.s.index_select(0, idx)
        current_d = self.d.index_select(0, idx)
        short_term = elapsed < 1.0
        success = rating > 1
        new_s = current_s
        new_s = torch.where(
            short_term,
            self._stability_short_term(current_s, rating),
            new_s,
        )
        new_s = torch.where(
            (~short_term) & success,
            self._stability_after_success(current_s, retrievability, current_d, rating),
            new_s,
        )
        new_s = torch.where(
            (~short_term) & (~success),
            self._stability_after_failure(current_s, retrievability, current_d),
            new_s,
        )
        self.s[idx] = torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max)
        self.d[idx] = self._next_d(current_d, rating)

    def _next_d(self, d: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        rating_f = rating.to(dtype=self.dtype)
        delta_d = -self.weights[6] * (rating_f - 3.0)
        new_d = d + delta_d * (10.0 - d) / 9.0
        new_d = self.weights[7] * self.init_d + (1.0 - self.weights[7]) * new_d
        return torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)

    def _stability_short_term(
        self, s: torch.Tensor, rating: torch.Tensor
    ) -> torch.Tensor:
        rating_f = rating.to(dtype=self.dtype)
        sinc = torch.exp(self.weights[17] * (rating_f - 3.0 + self.weights[18]))
        sinc = sinc * torch.pow(s, -self.weights[19])
        safe = torch.maximum(
            sinc, torch.tensor(1.0, device=self.device, dtype=self.dtype)
        )
        return s * torch.where(rating >= 3, safe, sinc)

    def _stability_after_success(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        hard_penalty = torch.where(
            rating == 2,
            self.weights[15],
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        easy_bonus = torch.where(
            rating == 4,
            self.weights[16],
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
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
