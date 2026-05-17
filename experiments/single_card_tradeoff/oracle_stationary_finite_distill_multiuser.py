from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import csv
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.auc_outputs import (  # noqa: E402
    MULTI_SCHEDULER_AUC_FIELDS,
    SINGLE_SCHEDULER_AUC_FIELDS,
    write_auc_summary as write_filtered_auc_summary,
    write_mean_auc_summary,
)
from experiments.single_card_tradeoff.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.oracle_frontier import (  # noqa: E402
    FSRS6BatchedStationaryFiniteOracle,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_HIDDEN_SIZE,
    DEFAULT_DISTILL_NETWORK_DEPTH,
    DEFAULT_DISTILL_SUPERVISION,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.tradeoff import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
    _build_regret_auc_rows,
    _write_csv,
    _write_regret_auc_csv,
)
from experiments.single_card_tradeoff.uvfa_ppo import (  # noqa: E402
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_TRAIN_ENVS,
    PolicyValueNet,
    SimMetrics,
    parse_csv_floats,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED  # noqa: E402
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS  # noqa: E402
from simulator.math.fsrs import Bounds  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402

DEFAULT_USER_IDS = tuple(range(1, 9))
DEFAULT_TRAIN_ENVS_PER_USER = DEFAULT_TRAIN_ENVS
DEFAULT_AGREEMENT_ENVS_PER_USER = 512
DEFAULT_AGREEMENT_STEPS = 256
DEFAULT_EVAL_PARTICLES = 10_000
DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE = 0
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/stationary_finite_distill_first8_users_batched"
)
BASELINE_SCHEDULER = "fsrs6"
EXACT_STATIONARY_FINITE_SCHEDULER = "fsrs6_oracle_stationary_finite"
PER_USER_SCHEDULER = "fsrs6_oracle_stationary_finite_distill_per_user"


@dataclass(frozen=True)
class MultiUserTrainStats:
    user_ids: list[int]
    params: int
    epochs: int
    steps_per_epoch: int
    train_envs_per_user: int
    train_transitions: int
    setup_runtime_s: float
    teacher_runtime_s: float
    train_runtime_s: float
    agreement_runtime_s: float
    final_ce_loss: float
    final_teacher_action_agreement: float
    eval_teacher_action_agreement: float
    eval_teacher_action_agreement_by_user: list[float]


@dataclass(frozen=True)
class SingleUserTrainStats:
    user_id: int
    user_index: int
    params_per_user: int
    ensemble_trainable_params: int
    epochs: int
    steps_per_epoch: int
    train_envs: int
    train_transitions: int
    train_runtime_s: float
    agreement_runtime_s: float
    final_ce_loss: float
    final_teacher_action_agreement: float
    eval_teacher_action_agreement: float
    supervision: str = DEFAULT_DISTILL_SUPERVISION
    table_samples_per_weight: int = DEFAULT_TABLE_SAMPLES_PER_WEIGHT


@dataclass(frozen=True)
class BatchedPolicyEnsemble:
    base_model: PolicyValueNet
    params: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    params_per_user: int


class MultiUserFSRS6SingleCardBatch:
    def __init__(
        self,
        *,
        days: int,
        user_indices: Sequence[int],
        configs: Sequence[SingleCardFSRS6Config],
        cost_weights: Sequence[float],
        action_retentions: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        exact_memory: bool,
        goal_norm_max: float | None = None,
        obs_mode: str = "oracle_stationary",
        reset_on_init: bool = True,
    ) -> None:
        if days <= 1:
            raise ValueError("days must be > 1.")
        if not configs:
            raise ValueError("configs must contain at least one user.")
        if not user_indices:
            raise ValueError("user_indices must contain at least one row.")
        if any(weight < 0.0 for weight in cost_weights):
            raise ValueError("cost weights must be >= 0.")

        self.days = int(days)
        self.user_count = len(configs)
        self.env_count = len(user_indices)
        self.device = device
        self.dtype = dtype
        self.exact_memory = exact_memory
        self.obs_mode = obs_mode
        self.bounds = Bounds()
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        user_tensor = torch.tensor(user_indices, device=device, dtype=torch.int64)
        if bool((user_tensor < 0).any().item()) or bool(
            (user_tensor >= self.user_count).any().item()
        ):
            raise ValueError("user_indices contains an out-of-range user index.")
        self.user_index = user_tensor

        weights_by_user = [
            tuple(config.fsrs_weights) if config.fsrs_weights else DEFAULT_FSRS6_WEIGHTS
            for config in configs
        ]
        if any(len(weights) != 21 for weights in weights_by_user):
            raise ValueError("each FSRS6 user config must contain 21 weights.")
        self.weights = torch.tensor(weights_by_user, device=device, dtype=dtype)
        self.decay = -self.weights[:, 20]
        self.factor = (
            torch.pow(
                torch.tensor(0.9, device=device, dtype=dtype),
                1.0 / self.decay,
            )
            - 1.0
        )
        self.init_d = torch.clamp(
            self.weights[:, 4] - torch.exp(self.weights[:, 5] * 3.0) + 1.0,
            self.bounds.d_min,
            self.bounds.d_max,
        )

        self.first_rating_prob = torch.tensor(
            [config.first_rating_prob for config in configs],
            device=device,
            dtype=dtype,
        )
        self.review_rating_prob = torch.tensor(
            [config.review_rating_prob for config in configs],
            device=device,
            dtype=dtype,
        )
        self.learning_costs = torch.tensor(
            [config.learning_costs for config in configs],
            device=device,
            dtype=dtype,
        )
        self.review_costs = torch.tensor(
            [config.review_costs for config in configs],
            device=device,
            dtype=dtype,
        )
        if self.learning_costs.shape[1] != 4 or self.review_costs.shape[1] != 4:
            raise ValueError("learning_costs and review_costs must contain 4 values.")

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

        self.s = torch.empty(self.env_count, device=device, dtype=dtype)
        self.d = torch.empty_like(self.s)
        self.day = torch.empty(self.env_count, device=device, dtype=torch.int64)
        self.pending_cost_seconds = torch.empty_like(self.s)
        self.goal_weight = torch.empty_like(self.s)
        self.last_interval = torch.empty_like(self.s)
        self.last_rating = torch.empty(self.env_count, device=device, dtype=torch.int64)
        self.done = torch.empty(self.env_count, device=device, dtype=torch.bool)
        self.total_memorized = torch.empty_like(self.s)
        self.total_cost_seconds = torch.empty_like(self.s)
        self.total_reviews = torch.empty(
            self.env_count, device=device, dtype=torch.int64
        )
        self.total_lapses = torch.empty_like(self.total_reviews)
        if reset_on_init:
            self.reset_all()

    @property
    def obs_dim(self) -> int:
        if self.obs_mode != "oracle_stationary":
            raise ValueError("multi-user distill currently supports oracle_stationary.")
        return 3

    @property
    def action_count(self) -> int:
        return int(self.action_retentions.numel())

    @property
    def max_goal_weight(self) -> float:
        return self._goal_norm_max

    def user_row_indices(self) -> list[torch.Tensor]:
        return [
            (self.user_index == user_idx).nonzero(as_tuple=False).squeeze(1)
            for user_idx in range(self.user_count)
        ]

    def reset_all(
        self, goal_values: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        idx = torch.arange(self.env_count, device=self.device)
        self.reset_indices(idx, goal_values=goal_values)
        return self.obs()

    def reset_indices(
        self,
        idx: torch.Tensor,
        *,
        goal_values: float | torch.Tensor | None = None,
    ) -> None:
        if idx.numel() == 0:
            return
        count = int(idx.numel())
        users = self.user_index.index_select(0, idx)
        if goal_values is None:
            goal_idx = torch.randint(
                self.cost_weight_values.numel(),
                (count,),
                device=self.device,
                generator=self.generator,
            )
            goals = self.cost_weight_values.index_select(0, goal_idx)
        elif isinstance(goal_values, torch.Tensor):
            goals = goal_values.to(device=self.device, dtype=self.dtype).index_select(
                0, idx
            )
        else:
            goals = torch.full(
                (count,), float(goal_values), device=self.device, dtype=self.dtype
            )

        first_weights = self.first_rating_prob.index_select(0, users)
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
        s_init, d_init = self._init_state(users, rating)
        self.s[idx] = s_init
        self.d[idx] = d_init
        self.day[idx] = 0
        self.pending_cost_seconds[idx] = self._gather_rating_cost(
            self.learning_costs,
            users,
            rating,
        )
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
        goal_norm = torch.log1p(self.goal_weight) / math.log1p(
            max(1.0, self.max_goal_weight)
        )
        return torch.stack([s_norm, d_norm, goal_norm], dim=1)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_action = action.index_select(0, active).to(torch.int64)
        active_users = self.user_index.index_select(0, active)
        intervals = self._intervals_for_action(
            self.s.index_select(0, active),
            active_action,
            active_users,
        )
        return self._step_active_intervals(active, intervals, active_users)

    def step_retention(
        self,
        retention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active = (~self.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
            return self.obs(), reward, self.done.clone()

        active_retention = retention.index_select(0, active).to(dtype=self.dtype)
        active_users = self.user_index.index_select(0, active)
        intervals = self._intervals_for_retention(
            self.s.index_select(0, active),
            active_retention,
            active_users,
        )
        return self._step_active_intervals(active, intervals, active_users)

    def _step_active_intervals(
        self,
        active: torch.Tensor,
        intervals: torch.Tensor,
        active_users: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reward = torch.zeros(self.env_count, device=self.device, dtype=self.dtype)
        remaining = (self.days - 1) - self.day.index_select(0, active)
        memorized_days = torch.minimum(intervals, remaining)
        memorized = self._memorized_sum(
            self.s.index_select(0, active),
            memorized_days,
            active_users,
        )
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
            cont_users = active_users[continuing]
            elapsed = intervals[continuing].to(dtype=self.dtype)
            retrievability = self._forgetting_curve(
                elapsed,
                self.s.index_select(0, cont_idx),
                cont_users,
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
                    self.review_rating_prob.index_select(0, cont_users),
                    num_samples=1,
                    replacement=True,
                    generator=self.generator,
                )
                .squeeze(1)
                .to(torch.int64)
                + 2
            )
            rating = torch.where(fail, torch.ones_like(success_rating), success_rating)
            self._update_review(cont_idx, cont_users, elapsed, rating, retrievability)
            self.day[cont_idx] = next_day[continuing]
            self.pending_cost_seconds[cont_idx] = self._gather_rating_cost(
                self.review_costs,
                cont_users,
                rating,
            )
            self.last_interval[cont_idx] = elapsed
            self.last_rating[cont_idx] = rating
            self.total_reviews[cont_idx] += 1
            self.total_lapses[cont_idx] += (rating == 1).to(torch.int64)

        return self.obs(), reward, self.done.clone()

    def metrics_by_group(
        self,
        *,
        group_index: torch.Tensor,
        group_count: int,
        particles_per_group: int,
    ) -> list[SimMetrics]:
        group_index = group_index.to(device=self.device, dtype=torch.int64)
        days = float(self.days)
        particles = float(particles_per_group)
        mem = torch.bincount(
            group_index,
            weights=self.total_memorized.to(dtype=torch.float64),
            minlength=group_count,
        )
        cost = torch.bincount(
            group_index,
            weights=self.total_cost_seconds.to(dtype=torch.float64),
            minlength=group_count,
        )
        reviews = torch.bincount(
            group_index,
            weights=self.total_reviews.to(dtype=torch.float64),
            minlength=group_count,
        )
        lapses = torch.bincount(
            group_index,
            weights=self.total_lapses.to(dtype=torch.float64),
            minlength=group_count,
        )
        out: list[SimMetrics] = []
        for group in range(group_count):
            review_count = float(reviews[group].item())
            lapse_count = float(lapses[group].item())
            total_cost = float(cost[group].item())
            out.append(
                SimMetrics(
                    card_expected_retrievability=float(
                        mem[group].item() / days / particles
                    ),
                    card_minutes_per_day=total_cost / days / 60.0 / particles,
                    card_reviews_per_day=review_count / days / particles,
                    card_total_reviews=review_count / particles,
                    card_total_lapses=lapse_count / particles,
                    card_total_cost_seconds=total_cost / particles,
                    observed_retention=(
                        1.0 - lapse_count / review_count if review_count > 0.0 else None
                    ),
                )
            )
        return out

    def _weights_for(self, users: torch.Tensor) -> torch.Tensor:
        return self.weights.index_select(0, users)

    def _factor_for(self, users: torch.Tensor) -> torch.Tensor:
        return self.factor.index_select(0, users)

    def _decay_for(self, users: torch.Tensor) -> torch.Tensor:
        return self.decay.index_select(0, users)

    def _gather_rating_cost(
        self,
        costs: torch.Tensor,
        users: torch.Tensor,
        rating: torch.Tensor,
    ) -> torch.Tensor:
        user_costs = costs.index_select(0, users)
        return user_costs.gather(1, (rating - 1).view(-1, 1)).squeeze(1)

    def _init_state(
        self,
        users: torch.Tensor,
        rating: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._weights_for(users)
        rating_f = rating.to(dtype=self.dtype)
        s = weights.gather(1, torch.clamp(rating - 1, min=0, max=3).view(-1, 1))
        s = s.squeeze(1)
        d = weights[:, 4] - torch.exp(weights[:, 5] * (rating_f - 1.0)) + 1.0
        return s, torch.clamp(d, self.bounds.d_min, self.bounds.d_max)

    def _forgetting_curve(
        self,
        elapsed: torch.Tensor,
        s: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        return torch.pow(
            1.0
            + self._factor_for(users) * elapsed / torch.clamp(s, min=self.bounds.s_min),
            self._decay_for(users),
        )

    def _intervals_for_action(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        retention = self.action_retentions.index_select(0, action)
        return self._intervals_for_retention(s, retention, users)

    def _intervals_for_retention(
        self,
        s: torch.Tensor,
        retention: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        retention_factor = torch.pow(retention, 1.0 / self._decay_for(users)) - 1.0
        interval = s / self._factor_for(users) * retention_factor
        return torch.clamp(torch.round(interval), min=1.0).to(torch.int64)

    def _memorized_sum(
        self,
        s: torch.Tensor,
        days: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        if self.exact_memory:
            return self._memorized_sum_exact(s, days, users)
        return self._memorized_sum_integral(s, days, users)

    def _memorized_sum_integral(
        self,
        s: torch.Tensor,
        days: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        days_f = days.to(dtype=self.dtype)
        positive = days_f > 0
        safe_s = torch.clamp(s, min=self.bounds.s_min)
        rate = self._factor_for(users) / safe_s
        upper = days_f + 0.5
        lower = torch.full_like(upper, 0.5)
        exponent = self._decay_for(users) + 1.0
        integral = (
            torch.pow(1.0 + rate * upper, exponent)
            - torch.pow(1.0 + rate * lower, exponent)
        ) / (rate * exponent)
        return torch.where(positive, integral, torch.zeros_like(integral))

    def _memorized_sum_exact(
        self,
        s: torch.Tensor,
        days: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
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
                sub_users = users.index_select(0, sub_idx)
                sub_s = torch.clamp(
                    s.index_select(0, sub_idx),
                    min=self.bounds.s_min,
                ).unsqueeze(1)
                values = torch.pow(
                    1.0
                    + self._factor_for(sub_users).unsqueeze(1)
                    * times.unsqueeze(0)
                    / sub_s,
                    self._decay_for(sub_users).unsqueeze(1),
                ).sum(dim=1)
                out[sub_idx] = values
        return out

    def _update_review(
        self,
        idx: torch.Tensor,
        users: torch.Tensor,
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
            self._stability_short_term(current_s, rating, users),
            new_s,
        )
        new_s = torch.where(
            (~short_term) & success,
            self._stability_after_success(
                current_s,
                retrievability,
                current_d,
                rating,
                users,
            ),
            new_s,
        )
        new_s = torch.where(
            (~short_term) & (~success),
            self._stability_after_failure(
                current_s,
                retrievability,
                current_d,
                users,
            ),
            new_s,
        )
        self.s[idx] = torch.clamp(new_s, self.bounds.s_min, self.bounds.s_max)
        self.d[idx] = self._next_d(current_d, rating, users)

    def _next_d(
        self,
        d: torch.Tensor,
        rating: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._weights_for(users)
        rating_f = rating.to(dtype=self.dtype)
        delta_d = -weights[:, 6] * (rating_f - 3.0)
        new_d = d + delta_d * (10.0 - d) / 9.0
        init_d = self.init_d.index_select(0, users)
        new_d = weights[:, 7] * init_d + (1.0 - weights[:, 7]) * new_d
        return torch.clamp(new_d, self.bounds.d_min, self.bounds.d_max)

    def _stability_short_term(
        self,
        s: torch.Tensor,
        rating: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._weights_for(users)
        rating_f = rating.to(dtype=self.dtype)
        sinc = torch.exp(weights[:, 17] * (rating_f - 3.0 + weights[:, 18]))
        sinc = sinc * torch.pow(s, -weights[:, 19])
        safe = torch.maximum(
            sinc,
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        return s * torch.where(rating >= 3, safe, sinc)

    def _stability_after_success(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
        rating: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._weights_for(users)
        hard_penalty = torch.where(
            rating == 2,
            weights[:, 15],
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        easy_bonus = torch.where(
            rating == 4,
            weights[:, 16],
            torch.tensor(1.0, device=self.device, dtype=self.dtype),
        )
        inc = (
            torch.exp(weights[:, 8])
            * (11.0 - d)
            * torch.pow(s, -weights[:, 9])
            * (torch.exp((1.0 - retrievability) * weights[:, 10]) - 1.0)
        )
        return s * (1.0 + inc * hard_penalty * easy_bonus)

    def _stability_after_failure(
        self,
        s: torch.Tensor,
        retrievability: torch.Tensor,
        d: torch.Tensor,
        users: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._weights_for(users)
        new_s = (
            weights[:, 11]
            * torch.pow(d, -weights[:, 12])
            * (torch.pow(s + 1.0, weights[:, 13]) - 1.0)
            * torch.exp((1.0 - retrievability) * weights[:, 14])
        )
        new_min = s / torch.exp(weights[:, 17] * weights[:, 18])
        return torch.minimum(new_s, new_min)


class BatchedStationaryFiniteOracleGuide:
    def __init__(
        self,
        *,
        days: int,
        cost_weights: Sequence[float],
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        device: torch.device,
        max_iterations: int,
        tolerance: float,
        progress: bool,
        configs: Sequence[SingleCardFSRS6Config],
        user_batch_size: int,
    ) -> None:
        if not configs:
            raise ValueError("configs must contain at least one user.")
        self.device = device
        self.cost_weights = torch.tensor(
            list(cost_weights), device=device, dtype=torch.float32
        )
        self.s_count = int(s_grid_size)
        self.d_count = int(d_grid_size)
        self.bounds = Bounds()
        self.log_s_min = math.log(self.bounds.s_min)
        self.log_s_max = math.log(self.bounds.s_max)

        chunk_size = len(configs) if user_batch_size <= 0 else user_batch_size
        policy_chunks: list[torch.Tensor] = []
        objective_chunks: list[torch.Tensor] = []
        metrics: list[list[Any]] = []
        iterations: list[list[int]] = []
        converged: list[list[bool]] = []
        residuals: list[list[float]] = []

        for start in range(0, len(configs), chunk_size):
            chunk_configs = configs[start : start + chunk_size]
            oracle = FSRS6BatchedStationaryFiniteOracle(
                days=days,
                action_retentions=action_retentions,
                s_grid_size=s_grid_size,
                d_grid_size=d_grid_size,
                device=device,
                fsrs_weights=[
                    tuple(config.fsrs_weights)
                    if config.fsrs_weights
                    else DEFAULT_FSRS6_WEIGHTS
                    for config in chunk_configs
                ],
                first_rating_prob=[
                    tuple(config.first_rating_prob) for config in chunk_configs
                ],
                review_rating_prob=[
                    tuple(config.review_rating_prob) for config in chunk_configs
                ],
                learning_costs=[
                    tuple(config.learning_costs) for config in chunk_configs
                ],
                review_costs=[tuple(config.review_costs) for config in chunk_configs],
            )
            solution = oracle.solve_stationary_finite_policies(
                cost_weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            failed = [
                f"user={chunk_configs[user_idx].user_id}:w={format_float(weight)}"
                for user_idx, row in enumerate(solution.converged)
                for weight, did_converge in zip(cost_weights, row, strict=True)
                if not did_converge
            ]
            if failed:
                raise RuntimeError(
                    "Stationary finite oracle did not converge for " + ",".join(failed)
                )
            policy_chunks.append(solution.policy.to(device=device, dtype=torch.uint8))
            objective_chunks.append(solution.objectives.to(device=device))
            metrics.extend(solution.metrics)
            iterations.extend(solution.iterations)
            converged.extend(solution.converged)
            residuals.extend(solution.residuals)

        self.policy = torch.cat(policy_chunks, dim=0)
        self.objectives = torch.cat(objective_chunks, dim=0)
        self.metrics = metrics
        self.iterations = iterations
        self.converged = converged
        self.residuals = residuals

    def labels(self, env: MultiUserFSRS6SingleCardBatch) -> torch.Tensor:
        s_idx = self._s_to_idx(env.s)
        d_idx = self._d_to_idx(env.d)
        goal_idx = torch.argmin(
            torch.abs(
                env.goal_weight.to(dtype=self.cost_weights.dtype)[:, None]
                - self.cost_weights[None, :]
            ),
            dim=1,
        )
        return self.policy.to(device=env.device)[
            env.user_index,
            goal_idx,
            s_idx,
            d_idx,
        ].to(torch.int64)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train stationary finite FSRS-6 distill policies for multiple users "
            "in one process with a batched exact teacher."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.set_defaults(env="fsrs6")
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
        help="Comma-separated benchmark user IDs to include in the training batch.",
    )
    parser.add_argument(
        "--per-user-models",
        dest="per_user_models",
        action="store_true",
        help=(
            "Train one independent distill model per user with stacked "
            "parameters and vmap in one process, while still solving the exact "
            "teachers as a batched DP."
        ),
    )
    parser.add_argument(
        "--shared-model",
        dest="per_user_models",
        action="store_false",
        help=(
            "Train one shared distill model across all users. This is mainly a "
            "diagnostic baseline because the policy observation does not identify "
            "the user (default)."
        ),
    )
    parser.set_defaults(per_user_models=False)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value)
            for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights for the oracle teacher.",
    )
    parser.add_argument(
        "--eval-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights for final policy evaluation.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to the oracle teacher.",
    )
    parser.add_argument(
        "--train-envs-per-user",
        type=int,
        default=DEFAULT_TRAIN_ENVS_PER_USER,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--per-user-supervision",
        choices=["uniform_table", "rollout"],
        default=DEFAULT_DISTILL_SUPERVISION,
        help=(
            "Supervision distribution for --per-user-models. uniform_table samples "
            "each exact stationary policy table cost weight equally; rollout keeps "
            "the older teacher-forcing event distribution."
        ),
    )
    parser.add_argument(
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
        help=(
            "Uniform exact policy table samples per user and cost weight per train "
            "step for --per-user-models."
        ),
    )
    parser.add_argument(
        "--network", choices=["mlp", "residual"], default=DEFAULT_NETWORK
    )
    parser.add_argument(
        "--network-depth", type=int, default=DEFAULT_DISTILL_NETWORK_DEPTH
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_DISTILL_HIDDEN_SIZE)
    parser.add_argument(
        "--oracle-s-grid-size", type=int, default=DEFAULT_ORACLE_S_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-d-grid-size", type=int, default=DEFAULT_ORACLE_D_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-stationary-finite-max-iterations",
        type=int,
        default=DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--oracle-stationary-finite-tolerance",
        type=float,
        default=DEFAULT_STATIONARY_FINITE_TOLERANCE,
    )
    parser.add_argument(
        "--oracle-teacher-user-batch-size",
        type=int,
        default=DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
        help=(
            "Number of users per batched exact teacher solve. 0 means solve all "
            "users in one single-process GPU batch."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--eval-group-batch-size",
        type=int,
        default=0,
        help=(
            "Number of retention or cost-weight evaluation groups to roll out "
            "together. 0 means batch all groups at once."
        ),
    )
    parser.add_argument(
        "--agreement-envs-per-user",
        type=int,
        default=DEFAULT_AGREEMENT_ENVS_PER_USER,
    )
    parser.add_argument("--agreement-steps", type=int, default=DEFAULT_AGREEMENT_STEPS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help=(
            "Model checkpoint path for --shared-model. Defaults to "
            "<out-dir>/multiuser_policy.pt. Ignored for per-user training."
        ),
    )
    parser.add_argument(
        "--eval-exact-vs-distill",
        action="store_true",
        help=(
            "Skip training and evaluate exact stationary finite policies against "
            "per-user distill checkpoints and the fsrs6 static-retention baseline."
        ),
    )
    parser.add_argument(
        "--distill-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing user_<id>_policy.pt checkpoints for "
            "--eval-exact-vs-distill. Defaults to --out-dir."
        ),
    )
    add_run_monitoring_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_user_ids(raw: str) -> list[int]:
    user_ids = [int(item) for item in raw.split(",") if item.strip()]
    if not user_ids:
        raise SystemExit("--user-ids must contain at least one user.")
    if any(user_id <= 0 for user_id in user_ids):
        raise SystemExit("--user-ids must be positive integers.")
    return user_ids


def load_user_configs(
    args: argparse.Namespace, user_ids: Sequence[int]
) -> list[SingleCardFSRS6Config]:
    configs = []
    for user_id in user_ids:
        user_args = argparse.Namespace(**vars(args))
        user_args.env = "fsrs6"
        user_args.user_id = user_id
        configs.append(load_single_card_fsrs6_config(user_args, environment="fsrs6"))
    return configs


def repeated_user_indices(user_count: int, rows_per_user: int) -> list[int]:
    return [user_idx for user_idx in range(user_count) for _ in range(rows_per_user)]


def teacher_labels(
    env: MultiUserFSRS6SingleCardBatch,
    guide: BatchedStationaryFiniteOracleGuide,
) -> torch.Tensor:
    return guide.labels(env)


def build_guide(
    args: argparse.Namespace,
    *,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[BatchedStationaryFiniteOracleGuide, float]:
    start = time.perf_counter()
    guide = BatchedStationaryFiniteOracleGuide(
        days=args.days,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        max_iterations=args.oracle_stationary_finite_max_iterations,
        tolerance=args.oracle_stationary_finite_tolerance,
        progress=not args.no_progress,
        configs=configs,
        user_batch_size=args.oracle_teacher_user_batch_size,
    )
    return guide, time.perf_counter() - start


def train_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    guide: BatchedStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[PolicyValueNet, float, float, float]:
    torch.manual_seed(args.seed)
    user_indices = repeated_user_indices(len(configs), args.train_envs_per_user)
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    model = PolicyValueNet(
        env.obs_dim,
        env.action_count,
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = env.obs()
    final_loss = 0.0
    final_agreement = 0.0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        for _ in range(args.steps_per_epoch):
            label = teacher_labels(env, guide)
            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                batch_total = int(label.numel())
                correct += int((pred == label).sum().item())
                total += batch_total
                loss_sum += float(loss.item()) * batch_total
                next_obs, _, done = env.step(label)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs
        final_loss = loss_sum / float(max(1, total))
        final_agreement = correct / float(max(1, total))
        if not args.no_progress:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"ce={final_loss:.5f} teacher_action_agreement={final_agreement:.4f}",
                flush=True,
            )
    return model, time.perf_counter() - start, final_loss, final_agreement


def train_single_user_model(
    args: argparse.Namespace,
    *,
    user_idx: int,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    guide: BatchedStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[PolicyValueNet, float, float, float]:
    torch.manual_seed(args.seed + user_idx)
    user_indices = [user_idx] * args.train_envs_per_user
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed + user_idx * 10_000,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    model = PolicyValueNet(
        env.obs_dim,
        env.action_count,
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = env.obs()
    final_loss = 0.0
    final_agreement = 0.0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        for _ in range(args.steps_per_epoch):
            label = teacher_labels(env, guide)
            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                batch_total = int(label.numel())
                correct += int((pred == label).sum().item())
                total += batch_total
                loss_sum += float(loss.item()) * batch_total
                next_obs, _, done = env.step(label)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs
        final_loss = loss_sum / float(max(1, total))
        final_agreement = correct / float(max(1, total))
        if not args.no_progress:
            print(
                f"user={configs[user_idx].user_id} epoch={epoch + 1}/{args.epochs} "
                f"ce={final_loss:.5f} teacher_action_agreement={final_agreement:.4f}",
                flush=True,
            )
    return model, time.perf_counter() - start, final_loss, final_agreement


def build_batched_per_user_ensemble(
    args: argparse.Namespace,
    *,
    user_count: int,
    obs_dim: int,
    action_count: int,
    params_per_user: int,
    device: torch.device,
) -> BatchedPolicyEnsemble:
    models = []
    for user_idx in range(user_count):
        torch.manual_seed(args.seed + user_idx)
        models.append(
            PolicyValueNet(
                obs_dim,
                action_count,
                args.hidden_size,
                architecture=args.network,
                depth=args.network_depth,
            ).to(device)
        )
    params, buffers = torch.func.stack_module_state(models)
    base_model = models[0]
    base_model.requires_grad_(False)
    return BatchedPolicyEnsemble(
        base_model=base_model,
        params=params,
        buffers=buffers,
        params_per_user=params_per_user,
    )


def batched_ensemble_forward(
    ensemble: BatchedPolicyEnsemble,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def call_one(
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        single_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = torch.func.functional_call(
            ensemble.base_model,
            (params, buffers),
            (single_obs,),
        )
        return logits, values

    return torch.func.vmap(call_one, in_dims=(0, 0, 0))(
        ensemble.params,
        ensemble.buffers,
        obs,
    )


def clip_stacked_grad_norm_(
    params: dict[str, torch.Tensor],
    *,
    max_norm: float,
) -> None:
    if max_norm <= 0.0 or not params:
        return
    user_count = next(iter(params.values())).shape[0]
    total_sq = torch.zeros(
        user_count,
        device=next(iter(params.values())).device,
        dtype=torch.float32,
    )
    for param in params.values():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += (
            grad.to(dtype=torch.float32).square().reshape(user_count, -1).sum(dim=1)
        )
    total_norm = torch.sqrt(total_sq)
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for param in params.values():
        if param.grad is None:
            continue
        view_shape = (user_count, *([1] * (param.grad.ndim - 1)))
        param.grad.mul_(scale.reshape(view_shape).to(dtype=param.grad.dtype))


def ensemble_state_dict_for_user(
    ensemble: BatchedPolicyEnsemble,
    user_idx: int,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in ensemble.params.items():
        state[name] = tensor[user_idx].detach().cpu().clone()
    for name, tensor in ensemble.buffers.items():
        state[name] = tensor[user_idx].detach().cpu().clone()
    return state


def materialize_ensemble_model(
    args: argparse.Namespace,
    *,
    ensemble: BatchedPolicyEnsemble,
    user_idx: int,
    obs_dim: int,
    action_count: int,
) -> PolicyValueNet:
    model = PolicyValueNet(
        obs_dim,
        action_count,
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    )
    model.load_state_dict(ensemble_state_dict_for_user(ensemble, user_idx))
    return model


def load_per_user_distill_ensemble(
    *,
    distill_dir: Path,
    user_ids: Sequence[int],
    device: torch.device,
) -> tuple[BatchedPolicyEnsemble, list[float], list[float]]:
    models: list[PolicyValueNet] = []
    first_checkpoint: dict[str, Any] | None = None
    action_retentions: list[float] | None = None
    cost_weights: list[float] | None = None
    params_per_user: int | None = None
    for user_id in user_ids:
        checkpoint_path = distill_dir / f"user_{user_id}_policy.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing distill checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("policy_type") != "fsrs6_oracle_stationary_finite_distill":
            raise ValueError(f"Unexpected policy_type in {checkpoint_path}.")
        checkpoint_actions = [float(value) for value in checkpoint["action_retentions"]]
        checkpoint_costs = [float(value) for value in checkpoint["cost_weights"]]
        if action_retentions is None:
            action_retentions = checkpoint_actions
            cost_weights = checkpoint_costs
            first_checkpoint = checkpoint
        elif (
            checkpoint_actions != action_retentions or checkpoint_costs != cost_weights
        ):
            raise ValueError(f"Distill checkpoint grid mismatch in {checkpoint_path}.")
        model = PolicyValueNet(
            int(checkpoint["obs_dim"]),
            len(checkpoint_actions),
            int(checkpoint["hidden_size"]),
            architecture=str(checkpoint["network"]),
            depth=int(checkpoint["network_depth"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model.to(device).eval())
        checkpoint_params = checkpoint.get("params_per_user")
        if params_per_user is None and checkpoint_params is not None:
            params_per_user = int(checkpoint_params)
    if first_checkpoint is None or action_retentions is None or cost_weights is None:
        raise ValueError("user_ids must contain at least one user.")
    if params_per_user is None:
        params_per_user = sum(param.numel() for param in models[0].parameters())
    params, buffers = torch.func.stack_module_state(models)
    base_model = models[0]
    base_model.requires_grad_(False)
    ensemble = BatchedPolicyEnsemble(
        base_model=base_model,
        params=params,
        buffers=buffers,
        params_per_user=params_per_user,
    )
    return ensemble, action_retentions, cost_weights


def sample_batched_uniform_table_batch(
    guide: BatchedStationaryFiniteOracleGuide,
    *,
    cost_weights: Sequence[float],
    samples_per_weight: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    user_count, weight_count, s_count, d_count = guide.policy.shape
    user_idx = torch.arange(user_count, device=device)[:, None, None].expand(
        user_count,
        weight_count,
        samples_per_weight,
    )
    weight_idx = torch.arange(weight_count, device=device)[None, :, None].expand(
        user_count,
        weight_count,
        samples_per_weight,
    )
    s_idx = torch.randint(
        s_count,
        (user_count, weight_count, samples_per_weight),
        device=device,
        generator=generator,
    )
    d_idx = torch.randint(
        d_count,
        (user_count, weight_count, samples_per_weight),
        device=device,
        generator=generator,
    )
    labels = guide.policy.to(device=device)[user_idx, weight_idx, s_idx, d_idx].reshape(
        user_count,
        -1,
    )
    max_goal = max(1.0, max(cost_weights))
    goal_norm = torch.log1p(
        torch.tensor(cost_weights, device=device, dtype=torch.float32)
    ) / torch.log1p(torch.tensor(max_goal, device=device, dtype=torch.float32))
    obs = torch.stack(
        [
            (s_idx.to(dtype=torch.float32) / float(s_count - 1)).reshape(
                user_count,
                -1,
            ),
            (d_idx.to(dtype=torch.float32) / float(d_count - 1)).reshape(
                user_count,
                -1,
            ),
            goal_norm[None, :, None]
            .expand(user_count, weight_count, samples_per_weight)
            .reshape(user_count, -1),
        ],
        dim=2,
    )
    return obs, labels.to(torch.int64)


def train_batched_per_user_models(
    args: argparse.Namespace,
    *,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    guide: BatchedStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    params_per_user: int,
) -> tuple[BatchedPolicyEnsemble, float, list[float], list[float]]:
    user_count = len(configs)
    user_indices = repeated_user_indices(user_count, args.train_envs_per_user)
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    ensemble = build_batched_per_user_ensemble(
        args,
        user_count=user_count,
        obs_dim=env.obs_dim,
        action_count=env.action_count,
        params_per_user=params_per_user,
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(ensemble.params.values()),
        lr=args.learning_rate,
        eps=1e-5,
    )
    table_generator = torch.Generator(device=device)
    table_generator.manual_seed(args.seed + 90_000)

    obs = env.obs().reshape(user_count, args.train_envs_per_user, env.obs_dim)
    final_loss_by_user = [0.0 for _ in range(user_count)]
    final_agreement_by_user = [0.0 for _ in range(user_count)]
    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss_sum_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
        correct_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
        total_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
        for _ in range(args.steps_per_epoch):
            if args.per_user_supervision == "uniform_table":
                obs, label = sample_batched_uniform_table_batch(
                    guide,
                    cost_weights=cost_weights,
                    samples_per_weight=args.table_samples_per_weight,
                    device=device,
                    generator=table_generator,
                )
            else:
                label = teacher_labels(env, guide).reshape(
                    user_count,
                    args.train_envs_per_user,
                )
            logits, _ = batched_ensemble_forward(ensemble, obs)
            per_item_loss = nn.functional.cross_entropy(
                logits.reshape(-1, env.action_count),
                label.reshape(-1),
                reduction="none",
            ).reshape(user_count, -1)
            loss_by_user = per_item_loss.mean(dim=1)
            loss = loss_by_user.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_stacked_grad_norm_(
                ensemble.params,
                max_norm=args.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits.detach(), dim=2)
                correct_by_user += (pred == label).sum(dim=1).to(dtype=torch.float64)
                total_by_user += float(label.shape[1])
                loss_sum_by_user += (
                    per_item_loss.detach().sum(dim=1).to(dtype=torch.float64)
                )
                if args.per_user_supervision == "rollout":
                    next_obs, _, done = env.step(label.reshape(-1))
                    if done.any():
                        env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                        next_obs = env.obs()
                    obs = next_obs.reshape(
                        user_count, args.train_envs_per_user, env.obs_dim
                    )
        final_loss_tensor = loss_sum_by_user / torch.clamp(total_by_user, min=1.0)
        final_agreement_tensor = correct_by_user / torch.clamp(total_by_user, min=1.0)
        final_loss_by_user = [float(value) for value in final_loss_tensor.tolist()]
        final_agreement_by_user = [
            float(value) for value in final_agreement_tensor.tolist()
        ]
        if not args.no_progress:
            mean_loss = sum(final_loss_by_user) / float(user_count)
            mean_agreement = sum(final_agreement_by_user) / float(user_count)
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"mean_ce={mean_loss:.5f} "
                f"mean_teacher_action_agreement={mean_agreement:.4f}",
                flush=True,
            )
    return (
        ensemble,
        time.perf_counter() - start,
        final_loss_by_user,
        final_agreement_by_user,
    )


@torch.inference_mode()
def estimate_agreement(
    args: argparse.Namespace,
    *,
    model: PolicyValueNet,
    guide: BatchedStationaryFiniteOracleGuide,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[float, list[float], float]:
    user_indices = repeated_user_indices(len(configs), args.agreement_envs_per_user)
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed + 60_000,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    user_count = len(configs)
    correct_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
    total_by_user = torch.zeros_like(correct_by_user)
    model.eval()
    obs = env.obs()
    start = time.perf_counter()
    for _ in range(args.agreement_steps):
        label = teacher_labels(env, guide)
        logits, _ = model(obs)
        pred = torch.argmax(logits, dim=1)
        matches = (pred == label).to(dtype=torch.float64)
        correct_by_user += torch.bincount(
            env.user_index,
            weights=matches,
            minlength=user_count,
        )
        total_by_user += torch.bincount(
            env.user_index,
            weights=torch.ones_like(matches),
            minlength=user_count,
        )
        next_obs, _, done = env.step(label)
        if done.any():
            env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
            next_obs = env.obs()
        obs = next_obs
    runtime_s = time.perf_counter() - start
    per_user = (correct_by_user / torch.clamp(total_by_user, min=1.0)).tolist()
    overall = float(correct_by_user.sum().item() / max(1.0, total_by_user.sum().item()))
    return overall, [float(value) for value in per_user], runtime_s


@torch.inference_mode()
def estimate_single_user_agreement(
    args: argparse.Namespace,
    *,
    user_idx: int,
    model: PolicyValueNet,
    guide: BatchedStationaryFiniteOracleGuide,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[float, float]:
    if args.agreement_steps <= 0:
        return 0.0, 0.0
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=[user_idx] * args.agreement_envs_per_user,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed + 60_000 + user_idx * 10_000,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    correct = 0
    total = 0
    model.eval()
    obs = env.obs()
    start = time.perf_counter()
    for _ in range(args.agreement_steps):
        label = teacher_labels(env, guide)
        logits, _ = model(obs)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == label).sum().item())
        total += int(label.numel())
        next_obs, _, done = env.step(label)
        if done.any():
            env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
            next_obs = env.obs()
        obs = next_obs
    return correct / float(max(1, total)), time.perf_counter() - start


@torch.inference_mode()
def estimate_batched_per_user_agreement(
    args: argparse.Namespace,
    *,
    ensemble: BatchedPolicyEnsemble,
    guide: BatchedStationaryFiniteOracleGuide,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[float, list[float], float]:
    if args.agreement_steps <= 0:
        return 0.0, [0.0 for _ in configs], 0.0
    user_count = len(configs)
    user_indices = repeated_user_indices(user_count, args.agreement_envs_per_user)
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=args.seed + 60_000,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
    )
    correct_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
    total_by_user = torch.zeros_like(correct_by_user)
    model_dtype = next(iter(ensemble.params.values())).dtype
    obs = (
        env.obs()
        .to(dtype=model_dtype)
        .reshape(
            user_count,
            args.agreement_envs_per_user,
            env.obs_dim,
        )
    )
    start = time.perf_counter()
    for _ in range(args.agreement_steps):
        label = teacher_labels(env, guide).reshape(
            user_count,
            args.agreement_envs_per_user,
        )
        logits, _ = batched_ensemble_forward(ensemble, obs)
        pred = torch.argmax(logits, dim=2)
        correct_by_user += (pred == label).sum(dim=1).to(dtype=torch.float64)
        total_by_user += float(args.agreement_envs_per_user)
        next_obs, _, done = env.step(label.reshape(-1))
        if done.any():
            env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
            next_obs = env.obs()
        obs = next_obs.to(dtype=model_dtype).reshape(
            user_count,
            args.agreement_envs_per_user,
            env.obs_dim,
        )
    runtime_s = time.perf_counter() - start
    per_user = (correct_by_user / torch.clamp(total_by_user, min=1.0)).tolist()
    overall = float(correct_by_user.sum().item() / max(1.0, total_by_user.sum().item()))
    return overall, [float(value) for value in per_user], runtime_s


@torch.inference_mode()
def estimate_batched_per_user_table_agreement(
    *,
    ensemble: BatchedPolicyEnsemble,
    guide: BatchedStationaryFiniteOracleGuide,
    device: torch.device,
    cost_weights: Sequence[float],
) -> tuple[float, list[float], float]:
    user_count, weight_count, s_count, d_count = guide.policy.shape
    s_grid, d_grid = torch.meshgrid(
        torch.arange(s_count, device=device),
        torch.arange(d_count, device=device),
        indexing="ij",
    )
    state_count = s_count * d_count
    max_goal = max(1.0, max(cost_weights))
    goal_norm = torch.log1p(
        torch.tensor(cost_weights, device=device, dtype=torch.float32)
    ) / torch.log1p(torch.tensor(max_goal, device=device, dtype=torch.float32))
    correct_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
    total_by_user = torch.zeros_like(correct_by_user)
    model_dtype = next(iter(ensemble.params.values())).dtype
    start = time.perf_counter()
    for weight_idx in range(weight_count):
        obs = torch.stack(
            [
                s_grid.reshape(-1).to(dtype=torch.float32) / float(s_count - 1),
                d_grid.reshape(-1).to(dtype=torch.float32) / float(d_count - 1),
                torch.full(
                    (state_count,),
                    float(goal_norm[weight_idx].item()),
                    device=device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        obs = obs[None, :, :].expand(user_count, state_count, 3).to(dtype=model_dtype)
        logits, _ = batched_ensemble_forward(ensemble, obs)
        labels = guide.policy[:, weight_idx].to(device=device, dtype=torch.int64)
        pred = torch.argmax(logits, dim=2).reshape(user_count, s_count, d_count)
        correct_by_user += (
            (pred == labels).reshape(user_count, -1).sum(dim=1).to(dtype=torch.float64)
        )
        total_by_user += float(state_count)
    runtime_s = time.perf_counter() - start
    per_user = (correct_by_user / torch.clamp(total_by_user, min=1.0)).tolist()
    overall = float(correct_by_user.sum().item() / max(1.0, total_by_user.sum().item()))
    return overall, [float(value) for value in per_user], runtime_s


def metric_row(
    args: argparse.Namespace,
    *,
    user_id: int,
    scheduler: str,
    scheduler_spec: str,
    desired_retention: float | None,
    goal_cost_weight: float | None,
    metrics: SimMetrics,
    runtime_s: float,
    engine: str = "multiuser_single_batch",
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    return {
        "environment": f"fsrs6_user_{user_id}",
        "scheduler": scheduler,
        "scheduler_spec": scheduler_spec,
        "desired_retention": desired_retention,
        "fixed_interval": None,
        "goal_cost_weight": goal_cost_weight,
        "seed": args.seed,
        "days": args.days,
        "particles": args.eval_particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "card_final_projected_retrievability": None,
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "total_reviews": metrics.card_total_reviews * args.eval_particles,
        "total_lapses": metrics.card_total_lapses * args.eval_particles,
        "total_cost_seconds": metrics.card_total_cost_seconds * args.eval_particles,
        "runtime_s": runtime_s,
        "engine": engine,
        "fuzz": False,
    }


def _eval_group_chunks(
    values: Sequence[float],
    group_batch_size: int,
) -> list[tuple[int, list[float]]]:
    if not values:
        return []
    chunk_size = len(values) if group_batch_size <= 0 else group_batch_size
    return [
        (start, list(values[start : start + chunk_size]))
        for start in range(0, len(values), chunk_size)
    ]


def _batched_eval_layout(
    *,
    user_count: int,
    group_count: int,
    particles_per_group: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    user_indices: list[int] = []
    group_indices: list[int] = []
    local_group_indices: list[int] = []
    for user_idx in range(user_count):
        for group_idx in range(group_count):
            user_indices.extend([user_idx] * particles_per_group)
            group_indices.extend(
                [user_idx * group_count + group_idx] * particles_per_group
            )
            local_group_indices.extend([group_idx] * particles_per_group)
    return (
        user_indices,
        torch.tensor(group_indices, device=device, dtype=torch.int64),
        torch.tensor(local_group_indices, device=device, dtype=torch.int64),
    )


@torch.inference_mode()
def evaluate_static_retentions_by_user(
    args: argparse.Namespace,
    *,
    retentions: Sequence[float],
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
) -> list[tuple[float, list[SimMetrics], float]]:
    user_count = len(configs)
    results: list[tuple[float, list[SimMetrics], float] | None] = [
        None for _ in retentions
    ]
    for start_idx, batch_retentions in _eval_group_chunks(
        retentions,
        args.eval_group_batch_size,
    ):
        group_count = len(batch_retentions)
        user_indices, group_index, local_group_idx = _batched_eval_layout(
            user_count=user_count,
            group_count=group_count,
            particles_per_group=args.eval_particles,
            device=device,
        )
        env = MultiUserFSRS6SingleCardBatch(
            days=args.days,
            user_indices=user_indices,
            configs=configs,
            cost_weights=[0.0],
            action_retentions=batch_retentions,
            device=device,
            dtype=torch.float64,
            seed=args.seed + 50_000 + int(round(float(batch_retentions[0]) * 10_000.0)),
            exact_memory=True,
            goal_norm_max=1.0,
        )
        action = local_group_idx
        start = time.perf_counter()
        while not bool(env.done.all().item()):
            env.step(action)
        elapsed_s = time.perf_counter() - start
        runtime_s = elapsed_s / float(max(1, user_count * group_count))
        metrics_flat = env.metrics_by_group(
            group_index=group_index,
            group_count=user_count * group_count,
            particles_per_group=args.eval_particles,
        )
        for local_idx, retention in enumerate(batch_retentions):
            metrics_by_user = [
                metrics_flat[user_idx * group_count + local_idx]
                for user_idx in range(user_count)
            ]
            results[start_idx + local_idx] = (
                retention,
                metrics_by_user,
                runtime_s,
            )
    return [result for result in results if result is not None]


@torch.inference_mode()
def evaluate_exact_stationary_finite_by_user(
    args: argparse.Namespace,
    *,
    guide: BatchedStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    goal_norm_max: float,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
) -> list[tuple[float, list[SimMetrics], float]]:
    user_count = len(configs)
    results: list[tuple[float, list[SimMetrics], float] | None] = [
        None for _ in cost_weights
    ]
    for start_idx, batch_weights in _eval_group_chunks(
        cost_weights,
        args.eval_group_batch_size,
    ):
        group_count = len(batch_weights)
        user_indices, group_index, local_group_idx = _batched_eval_layout(
            user_count=user_count,
            group_count=group_count,
            particles_per_group=args.eval_particles,
            device=device,
        )
        env = MultiUserFSRS6SingleCardBatch(
            days=args.days,
            user_indices=user_indices,
            configs=configs,
            cost_weights=batch_weights,
            action_retentions=action_retentions,
            device=device,
            dtype=torch.float64,
            seed=args.seed + 80_000 + int(round(float(batch_weights[0]) * 10.0)),
            exact_memory=True,
            goal_norm_max=goal_norm_max,
            reset_on_init=False,
        )
        goal_values = torch.tensor(
            batch_weights,
            device=device,
            dtype=torch.float64,
        ).index_select(0, local_group_idx)
        env.reset_all(goal_values=goal_values)
        start = time.perf_counter()
        while not bool(env.done.all().item()):
            env.step(teacher_labels(env, guide))
        elapsed_s = time.perf_counter() - start
        runtime_s = elapsed_s / float(max(1, user_count * group_count))
        metrics_flat = env.metrics_by_group(
            group_index=group_index,
            group_count=user_count * group_count,
            particles_per_group=args.eval_particles,
        )
        for local_idx, cost_weight in enumerate(batch_weights):
            metrics_by_user = [
                metrics_flat[user_idx * group_count + local_idx]
                for user_idx in range(user_count)
            ]
            results[start_idx + local_idx] = (
                cost_weight,
                metrics_by_user,
                runtime_s,
            )
    return [result for result in results if result is not None]


@torch.inference_mode()
def evaluate_policies_by_user(
    args: argparse.Namespace,
    *,
    model: PolicyValueNet,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    goal_norm_max: float,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
) -> list[tuple[float, list[SimMetrics], float]]:
    user_count = len(configs)
    results: list[tuple[float, list[SimMetrics], float] | None] = [
        None for _ in cost_weights
    ]
    model_dtype = next(model.parameters()).dtype
    model.eval()
    for start_idx, batch_weights in _eval_group_chunks(
        cost_weights,
        args.eval_group_batch_size,
    ):
        group_count = len(batch_weights)
        user_indices, group_index, local_group_idx = _batched_eval_layout(
            user_count=user_count,
            group_count=group_count,
            particles_per_group=args.eval_particles,
            device=device,
        )
        env = MultiUserFSRS6SingleCardBatch(
            days=args.days,
            user_indices=user_indices,
            configs=configs,
            cost_weights=batch_weights,
            action_retentions=action_retentions,
            device=device,
            dtype=torch.float64,
            seed=args.seed + 70_000 + int(round(float(batch_weights[0]) * 10.0)),
            exact_memory=True,
            goal_norm_max=goal_norm_max,
            reset_on_init=False,
        )
        goal_values = torch.tensor(
            batch_weights,
            device=device,
            dtype=torch.float64,
        ).index_select(0, local_group_idx)
        env.reset_all(goal_values=goal_values)
        start = time.perf_counter()
        while not bool(env.done.all().item()):
            obs = env.obs().to(dtype=model_dtype)
            logits, _ = model(obs)
            action = torch.argmax(logits, dim=1)
            env.step(action)
        elapsed_s = time.perf_counter() - start
        runtime_s = elapsed_s / float(max(1, user_count * group_count))
        metrics_flat = env.metrics_by_group(
            group_index=group_index,
            group_count=user_count * group_count,
            particles_per_group=args.eval_particles,
        )
        for local_idx, cost_weight in enumerate(batch_weights):
            metrics_by_user = [
                metrics_flat[user_idx * group_count + local_idx]
                for user_idx in range(user_count)
            ]
            results[start_idx + local_idx] = (
                cost_weight,
                metrics_by_user,
                runtime_s,
            )
    return [result for result in results if result is not None]


@torch.inference_mode()
def evaluate_batched_per_user_policies(
    args: argparse.Namespace,
    *,
    ensemble: BatchedPolicyEnsemble,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    goal_norm_max: float,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
) -> list[tuple[float, list[SimMetrics], float]]:
    user_count = len(configs)
    results: list[tuple[float, list[SimMetrics], float] | None] = [
        None for _ in cost_weights
    ]
    model_dtype = next(iter(ensemble.params.values())).dtype
    for start_idx, batch_weights in _eval_group_chunks(
        cost_weights,
        args.eval_group_batch_size,
    ):
        group_count = len(batch_weights)
        user_indices, group_index, local_group_idx = _batched_eval_layout(
            user_count=user_count,
            group_count=group_count,
            particles_per_group=args.eval_particles,
            device=device,
        )
        env = MultiUserFSRS6SingleCardBatch(
            days=args.days,
            user_indices=user_indices,
            configs=configs,
            cost_weights=batch_weights,
            action_retentions=action_retentions,
            device=device,
            dtype=torch.float64,
            seed=args.seed + 70_000 + int(round(float(batch_weights[0]) * 10.0)),
            exact_memory=True,
            goal_norm_max=goal_norm_max,
            reset_on_init=False,
        )
        goal_values = torch.tensor(
            batch_weights,
            device=device,
            dtype=torch.float64,
        ).index_select(0, local_group_idx)
        env.reset_all(goal_values=goal_values)
        start = time.perf_counter()
        while not bool(env.done.all().item()):
            obs = (
                env.obs()
                .to(dtype=model_dtype)
                .reshape(
                    user_count,
                    group_count * args.eval_particles,
                    env.obs_dim,
                )
            )
            logits, _ = batched_ensemble_forward(ensemble, obs)
            action = torch.argmax(logits, dim=2).reshape(-1)
            env.step(action)
        elapsed_s = time.perf_counter() - start
        runtime_s = elapsed_s / float(max(1, user_count * group_count))
        metrics_flat = env.metrics_by_group(
            group_index=group_index,
            group_count=user_count * group_count,
            particles_per_group=args.eval_particles,
        )
        for local_idx, cost_weight in enumerate(batch_weights):
            metrics_by_user = [
                metrics_flat[user_idx * group_count + local_idx]
                for user_idx in range(user_count)
            ]
            results[start_idx + local_idx] = (
                cost_weight,
                metrics_by_user,
                runtime_s,
            )
    return [result for result in results if result is not None]


def save_checkpoint(
    path: Path,
    *,
    model: PolicyValueNet,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    guide: BatchedStationaryFiniteOracleGuide,
    stats: MultiUserTrainStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "fsrs6_oracle_stationary_finite_distill",
            "training_scope": "multiuser_single_batch",
            "model_state_dict": model.state_dict(),
            "cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "obs_dim": model.obs_dim,
            "obs_mode": "oracle_stationary",
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "guide_policy": "stationary_finite_oracle",
            "user_ids": list(user_ids),
            "user_configs": [config.checkpoint_payload() for config in configs],
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_stationary_finite_max_iterations": (
                args.oracle_stationary_finite_max_iterations
            ),
            "oracle_stationary_finite_tolerance": (
                args.oracle_stationary_finite_tolerance
            ),
            "oracle_teacher_user_batch_size": args.oracle_teacher_user_batch_size,
            "oracle_stationary_finite_objectives_by_user": guide.objectives.tolist(),
            "oracle_stationary_finite_iterations_by_user": [
                list(row) for row in guide.iterations
            ],
            "oracle_stationary_finite_residuals_by_user": [
                list(row) for row in guide.residuals
            ],
            "oracle_stationary_finite_converged_by_user": [
                list(row) for row in guide.converged
            ],
            "distill_epochs": stats.epochs,
            "distill_steps_per_epoch": stats.steps_per_epoch,
            "train_envs_per_user": stats.train_envs_per_user,
            "train_transitions": stats.train_transitions,
            "train_setup_runtime_s": stats.setup_runtime_s,
            "teacher_runtime_s": stats.teacher_runtime_s,
            "train_runtime_s": stats.train_runtime_s,
            "agreement_runtime_s": stats.agreement_runtime_s,
            "final_ce_loss": stats.final_ce_loss,
            "final_teacher_action_agreement": stats.final_teacher_action_agreement,
            "eval_teacher_action_agreement": stats.eval_teacher_action_agreement,
            "eval_teacher_action_agreement_by_user": (
                stats.eval_teacher_action_agreement_by_user
            ),
        },
        path,
    )


def save_single_user_checkpoint(
    path: Path,
    *,
    model: PolicyValueNet,
    args: argparse.Namespace,
    user_id: int,
    user_idx: int,
    config: SingleCardFSRS6Config,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    guide: BatchedStationaryFiniteOracleGuide,
    stats: SingleUserTrainStats,
    teacher_runtime_s: float,
    training_scope: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": "fsrs6_oracle_stationary_finite_distill",
            "training_scope": training_scope,
            "model_state_dict": model.state_dict(),
            "cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "obs_dim": model.obs_dim,
            "obs_mode": "oracle_stationary",
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "guide_policy": "stationary_finite_oracle",
            "user_ids": [user_id],
            "user_index": user_idx,
            "user_configs": [config.checkpoint_payload()],
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_stationary_finite_max_iterations": (
                args.oracle_stationary_finite_max_iterations
            ),
            "oracle_stationary_finite_tolerance": (
                args.oracle_stationary_finite_tolerance
            ),
            "oracle_teacher_user_batch_size": args.oracle_teacher_user_batch_size,
            "oracle_stationary_finite_objectives": guide.objectives[user_idx].tolist(),
            "oracle_stationary_finite_iterations": list(guide.iterations[user_idx]),
            "oracle_stationary_finite_residuals": list(guide.residuals[user_idx]),
            "oracle_stationary_finite_converged": list(guide.converged[user_idx]),
            "distill_epochs": stats.epochs,
            "distill_steps_per_epoch": stats.steps_per_epoch,
            "distill_supervision": stats.supervision,
            "table_samples_per_weight": stats.table_samples_per_weight,
            "train_envs": stats.train_envs,
            "train_transitions": stats.train_transitions,
            "params_per_user": stats.params_per_user,
            "ensemble_trainable_params": stats.ensemble_trainable_params,
            "teacher_runtime_s": teacher_runtime_s,
            "train_runtime_s": stats.train_runtime_s,
            "agreement_runtime_s": stats.agreement_runtime_s,
            "final_ce_loss": stats.final_ce_loss,
            "final_teacher_action_agreement": stats.final_teacher_action_agreement,
            "eval_teacher_action_agreement": stats.eval_teacher_action_agreement,
        },
        path,
    )


def write_train_summary(path: Path, stats: MultiUserTrainStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "user_ids",
                "params",
                "epochs",
                "steps_per_epoch",
                "train_envs_per_user",
                "train_transitions",
                "setup_runtime_s",
                "teacher_runtime_s",
                "train_runtime_s",
                "agreement_runtime_s",
                "total_train_runtime_s",
                "final_ce_loss",
                "final_teacher_action_agreement",
                "eval_teacher_action_agreement",
                "eval_teacher_action_agreement_by_user",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "user_ids": ",".join(str(user_id) for user_id in stats.user_ids),
                "params": stats.params,
                "epochs": stats.epochs,
                "steps_per_epoch": stats.steps_per_epoch,
                "train_envs_per_user": stats.train_envs_per_user,
                "train_transitions": stats.train_transitions,
                "setup_runtime_s": stats.setup_runtime_s,
                "teacher_runtime_s": stats.teacher_runtime_s,
                "train_runtime_s": stats.train_runtime_s,
                "agreement_runtime_s": stats.agreement_runtime_s,
                "total_train_runtime_s": (
                    stats.setup_runtime_s
                    + stats.teacher_runtime_s
                    + stats.train_runtime_s
                    + stats.agreement_runtime_s
                ),
                "final_ce_loss": stats.final_ce_loss,
                "final_teacher_action_agreement": (
                    stats.final_teacher_action_agreement
                ),
                "eval_teacher_action_agreement": stats.eval_teacher_action_agreement,
                "eval_teacher_action_agreement_by_user": ",".join(
                    f"{value:.6f}"
                    for value in stats.eval_teacher_action_agreement_by_user
                ),
            }
        )


def write_single_user_train_summary(
    path: Path,
    *,
    stats_by_user: Sequence[SingleUserTrainStats],
    teacher_runtime_s: float,
    setup_runtime_s: float,
    eval_runtime_s: float,
    training_scope: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "training_scope",
            "user_id",
            "params_per_user",
            "ensemble_trainable_params",
            "epochs",
            "steps_per_epoch",
            "supervision",
            "table_samples_per_weight",
            "train_envs",
            "train_transitions",
            "setup_runtime_s",
            "teacher_runtime_s",
            "train_runtime_s",
            "agreement_runtime_s",
            "eval_runtime_s",
            "total_runtime_s",
            "final_ce_loss",
            "final_teacher_action_agreement",
            "eval_teacher_action_agreement",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_by_user:
            writer.writerow(
                {
                    "training_scope": training_scope,
                    "user_id": stats.user_id,
                    "params_per_user": stats.params_per_user,
                    "ensemble_trainable_params": stats.ensemble_trainable_params,
                    "epochs": stats.epochs,
                    "steps_per_epoch": stats.steps_per_epoch,
                    "supervision": stats.supervision,
                    "table_samples_per_weight": stats.table_samples_per_weight,
                    "train_envs": stats.train_envs,
                    "train_transitions": stats.train_transitions,
                    "setup_runtime_s": setup_runtime_s,
                    "teacher_runtime_s": teacher_runtime_s,
                    "train_runtime_s": stats.train_runtime_s,
                    "agreement_runtime_s": stats.agreement_runtime_s,
                    "eval_runtime_s": eval_runtime_s,
                    "total_runtime_s": (
                        setup_runtime_s
                        + teacher_runtime_s
                        + stats.train_runtime_s
                        + stats.agreement_runtime_s
                        + eval_runtime_s
                    ),
                    "final_ce_loss": stats.final_ce_loss,
                    "final_teacher_action_agreement": (
                        stats.final_teacher_action_agreement
                    ),
                    "eval_teacher_action_agreement": (
                        stats.eval_teacher_action_agreement
                    ),
                }
            )


def write_auc_summary(
    path: Path,
    auc_rows: list[dict[str, Any]],
    *,
    scheduler: str = "fsrs6_oracle_stationary_finite_distill",
) -> None:
    write_filtered_auc_summary(
        path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER],
        schedulers=[scheduler],
        fieldnames=SINGLE_SCHEDULER_AUC_FIELDS,
    )


def evaluate_exact_vs_distill(
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
    configs: Sequence[SingleCardFSRS6Config],
    eval_cost_weights: Sequence[float],
    device: torch.device,
) -> None:
    distill_dir = args.distill_dir or args.out_dir
    ensemble, action_retentions, distill_cost_weights = load_per_user_distill_ensemble(
        distill_dir=distill_dir,
        user_ids=user_ids,
        device=device,
    )
    guide, teacher_runtime_s = build_guide(
        args,
        device=device,
        configs=configs,
        cost_weights=eval_cost_weights,
        action_retentions=action_retentions,
    )

    rows: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    for retention, metrics_by_user, runtime_s in evaluate_static_retentions_by_user(
        args,
        retentions=action_retentions,
        device=device,
        configs=configs,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler=BASELINE_SCHEDULER,
                    scheduler_spec=BASELINE_SCHEDULER,
                    desired_retention=retention,
                    goal_cost_weight=None,
                    metrics=metrics,
                    runtime_s=runtime_s,
                )
            )
    for (
        cost_weight,
        metrics_by_user,
        runtime_s,
    ) in evaluate_exact_stationary_finite_by_user(
        args,
        guide=guide,
        cost_weights=eval_cost_weights,
        action_retentions=action_retentions,
        goal_norm_max=max(eval_cost_weights),
        device=device,
        configs=configs,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler=EXACT_STATIONARY_FINITE_SCHEDULER,
                    scheduler_spec=EXACT_STATIONARY_FINITE_SCHEDULER,
                    desired_retention=None,
                    goal_cost_weight=cost_weight,
                    metrics=metrics,
                    runtime_s=runtime_s,
                    engine="multiuser_exact_policy_table_batch",
                )
            )
    for cost_weight, metrics_by_user, runtime_s in evaluate_batched_per_user_policies(
        args,
        ensemble=ensemble,
        cost_weights=eval_cost_weights,
        action_retentions=action_retentions,
        goal_norm_max=max(distill_cost_weights),
        device=device,
        configs=configs,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler=PER_USER_SCHEDULER,
                    scheduler_spec=PER_USER_SCHEDULER,
                    desired_retention=None,
                    goal_cost_weight=cost_weight,
                    metrics=metrics,
                    runtime_s=runtime_s,
                    engine="per_user_batched_uniform_table_supervision",
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_runtime_s = time.perf_counter() - eval_start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.csv"
    regret_path = args.out_dir / "regret_auc.csv"
    summary_path = args.out_dir / "summary.csv"
    mean_summary_path = args.out_dir / "mean_summary.csv"
    _write_csv(results_path, rows)
    auc_rows = _build_regret_auc_rows(rows)
    _write_regret_auc_csv(regret_path, auc_rows)
    write_filtered_auc_summary(
        summary_path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER],
        schedulers=[EXACT_STATIONARY_FINITE_SCHEDULER, PER_USER_SCHEDULER],
        fieldnames=MULTI_SCHEDULER_AUC_FIELDS,
    )
    write_mean_auc_summary(
        mean_summary_path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER],
        schedulers=[EXACT_STATIONARY_FINITE_SCHEDULER, PER_USER_SCHEDULER],
        include_baseline_scheduler=False,
    )

    print(f"Wrote CSV: {results_path}")
    print(f"Wrote regret AUC CSV: {regret_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote mean summary CSV: {mean_summary_path}")
    print(
        "Exact-vs-distill stationary finite evaluation: "
        f"users={','.join(str(user_id) for user_id in user_ids)} "
        f"distill_dir={distill_dir} device={device} "
        f"teacher_s={teacher_runtime_s:.2f} eval_s={eval_runtime_s:.2f}"
    )


def main() -> None:
    args = parse_args()
    user_ids = parse_user_ids(args.user_ids)
    if args.env != "fsrs6":
        raise SystemExit("multi-user stationary finite distill requires --env fsrs6.")
    if args.train_envs_per_user <= 0:
        raise SystemExit("--train-envs-per-user must be > 0.")
    if args.eval_particles <= 0:
        raise SystemExit("--eval-particles must be > 0.")
    if args.eval_group_batch_size < 0:
        raise SystemExit("--eval-group-batch-size must be >= 0.")
    if args.agreement_envs_per_user <= 0:
        raise SystemExit("--agreement-envs-per-user must be > 0.")
    if args.oracle_teacher_user_batch_size < 0:
        raise SystemExit("--oracle-teacher-user-batch-size must be >= 0.")
    if args.table_samples_per_weight <= 0:
        raise SystemExit("--table-samples-per-weight must be > 0.")

    device = resolve_torch_device(args.torch_device)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    eval_cost_weights = parse_csv_floats(
        args.eval_cost_weights,
        name="--eval-cost-weights",
    )
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    configs = load_user_configs(args, user_ids)

    if args.eval_exact_vs_distill:
        evaluate_exact_vs_distill(
            args,
            user_ids=user_ids,
            configs=configs,
            eval_cost_weights=eval_cost_weights,
            device=device,
        )
        return

    setup_start = time.perf_counter()
    params_probe = PolicyValueNet(
        3,
        len(action_retentions),
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    )
    params = sum(param.numel() for param in params_probe.parameters())
    setup_runtime_s = time.perf_counter() - setup_start

    guide, teacher_runtime_s = build_guide(
        args,
        device=device,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    if args.per_user_models:
        if args.model_out is not None:
            print(
                "--model-out is ignored with --per-user-models; "
                "writing one checkpoint per user under --out-dir.",
                flush=True,
            )

        ensemble, train_runtime_s, final_loss_by_user, final_agreement_by_user = (
            train_batched_per_user_models(
                args,
                device=device,
                configs=configs,
                guide=guide,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
                params_per_user=params,
            )
        )
        if args.per_user_supervision == "uniform_table":
            eval_agreement, eval_agreement_by_user, agreement_runtime_s = (
                estimate_batched_per_user_table_agreement(
                    ensemble=ensemble,
                    guide=guide,
                    device=device,
                    cost_weights=cost_weights,
                )
            )
            train_samples_per_user = args.table_samples_per_weight * len(cost_weights)
            training_scope = "per_user_batched_uniform_table_supervision"
        else:
            eval_agreement, eval_agreement_by_user, agreement_runtime_s = (
                estimate_batched_per_user_agreement(
                    args,
                    ensemble=ensemble,
                    guide=guide,
                    device=device,
                    configs=configs,
                    cost_weights=cost_weights,
                    action_retentions=action_retentions,
                )
            )
            train_samples_per_user = args.train_envs_per_user
            training_scope = "per_user_batched_single_process"

        ensemble_trainable_params = params * len(user_ids)
        stats_by_user: list[SingleUserTrainStats] = []
        model_paths: list[Path] = []
        for user_idx, user_id in enumerate(user_ids):
            user_stats = SingleUserTrainStats(
                user_id=user_id,
                user_index=user_idx,
                params_per_user=params,
                ensemble_trainable_params=ensemble_trainable_params,
                epochs=args.epochs,
                steps_per_epoch=args.steps_per_epoch,
                train_envs=train_samples_per_user,
                train_transitions=(
                    args.epochs * args.steps_per_epoch * train_samples_per_user
                ),
                train_runtime_s=train_runtime_s,
                agreement_runtime_s=agreement_runtime_s,
                final_ce_loss=final_loss_by_user[user_idx],
                final_teacher_action_agreement=final_agreement_by_user[user_idx],
                eval_teacher_action_agreement=eval_agreement_by_user[user_idx],
                supervision=args.per_user_supervision,
                table_samples_per_weight=args.table_samples_per_weight,
            )
            model_path = args.out_dir / f"user_{user_id}_policy.pt"
            model = materialize_ensemble_model(
                args,
                ensemble=ensemble,
                user_idx=user_idx,
                obs_dim=3,
                action_count=len(action_retentions),
            )
            save_single_user_checkpoint(
                model_path,
                model=model,
                args=args,
                user_id=user_id,
                user_idx=user_idx,
                config=configs[user_idx],
                cost_weights=cost_weights,
                action_retentions=action_retentions,
                guide=guide,
                stats=user_stats,
                teacher_runtime_s=teacher_runtime_s,
                training_scope=training_scope,
            )
            stats_by_user.append(user_stats)
            model_paths.append(model_path)

        rows: list[dict[str, Any]] = []
        eval_start = time.perf_counter()
        for retention, metrics_by_user, runtime_s in evaluate_static_retentions_by_user(
            args,
            retentions=action_retentions,
            device=device,
            configs=configs,
        ):
            for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
                rows.append(
                    metric_row(
                        args,
                        user_id=user_id,
                        scheduler=BASELINE_SCHEDULER,
                        scheduler_spec=BASELINE_SCHEDULER,
                        desired_retention=retention,
                        goal_cost_weight=None,
                        metrics=metrics,
                        runtime_s=runtime_s,
                    )
                )
        for (
            cost_weight,
            metrics_by_user,
            runtime_s,
        ) in evaluate_batched_per_user_policies(
            args,
            ensemble=ensemble,
            cost_weights=eval_cost_weights,
            action_retentions=action_retentions,
            goal_norm_max=max(cost_weights),
            device=device,
            configs=configs,
        ):
            for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
                rows.append(
                    metric_row(
                        args,
                        user_id=user_id,
                        scheduler=PER_USER_SCHEDULER,
                        scheduler_spec=PER_USER_SCHEDULER,
                        desired_retention=None,
                        goal_cost_weight=cost_weight,
                        metrics=metrics,
                        runtime_s=runtime_s,
                        engine=training_scope,
                    )
                )
        eval_runtime_s = time.perf_counter() - eval_start

        results_path = args.out_dir / "results.csv"
        regret_path = args.out_dir / "regret_auc.csv"
        summary_path = args.out_dir / "summary.csv"
        train_summary_path = args.out_dir / "train_summary.csv"
        _write_csv(results_path, rows)
        auc_rows = _build_regret_auc_rows(rows)
        _write_regret_auc_csv(regret_path, auc_rows)
        write_auc_summary(summary_path, auc_rows, scheduler=PER_USER_SCHEDULER)
        write_single_user_train_summary(
            train_summary_path,
            stats_by_user=stats_by_user,
            teacher_runtime_s=teacher_runtime_s,
            setup_runtime_s=setup_runtime_s,
            eval_runtime_s=eval_runtime_s,
            training_scope=training_scope,
        )

        mean_loss = sum(stats.final_ce_loss for stats in stats_by_user) / float(
            len(stats_by_user)
        )
        mean_train_agreement = sum(
            stats.final_teacher_action_agreement for stats in stats_by_user
        ) / float(len(stats_by_user))
        mean_eval_agreement = sum(
            stats.eval_teacher_action_agreement for stats in stats_by_user
        ) / float(len(stats_by_user))

        print(
            "Wrote per-user models: " + ", ".join(str(path) for path in model_paths),
            flush=True,
        )
        print(f"Wrote CSV: {results_path}")
        print(f"Wrote regret AUC CSV: {regret_path}")
        print(f"Wrote summary CSV: {summary_path}")
        print(f"Wrote train summary CSV: {train_summary_path}")
        print(
            "Per-user stationary finite distill: "
            f"users={','.join(str(user_id) for user_id in user_ids)} "
            f"params_each={params} ensemble_params={ensemble_trainable_params} "
            f"device={device} "
            f"teacher_s={teacher_runtime_s:.2f} "
            f"train_s={train_runtime_s:.2f} "
            f"agreement_s={agreement_runtime_s:.2f} "
            f"eval_s={eval_runtime_s:.2f} "
            f"mean_final_ce={mean_loss:.5f} "
            f"mean_train_agreement={mean_train_agreement:.4f} "
            f"mean_eval_agreement={mean_eval_agreement:.4f}"
        )
        return

    model, train_runtime_s, final_loss, final_agreement = train_model(
        args,
        device=device,
        configs=configs,
        guide=guide,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    eval_agreement, eval_agreement_by_user, agreement_runtime_s = estimate_agreement(
        args,
        model=model,
        guide=guide,
        device=device,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    stats = MultiUserTrainStats(
        user_ids=list(user_ids),
        params=params,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        train_envs_per_user=args.train_envs_per_user,
        train_transitions=(
            args.epochs
            * args.steps_per_epoch
            * args.train_envs_per_user
            * len(user_ids)
        ),
        setup_runtime_s=setup_runtime_s,
        teacher_runtime_s=teacher_runtime_s,
        train_runtime_s=train_runtime_s,
        agreement_runtime_s=agreement_runtime_s,
        final_ce_loss=final_loss,
        final_teacher_action_agreement=final_agreement,
        eval_teacher_action_agreement=eval_agreement,
        eval_teacher_action_agreement_by_user=eval_agreement_by_user,
    )

    model_out = args.model_out or (args.out_dir / "multiuser_policy.pt")
    save_checkpoint(
        model_out,
        model=model,
        args=args,
        user_ids=user_ids,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        guide=guide,
        stats=stats,
    )

    rows: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    for retention, metrics_by_user, runtime_s in evaluate_static_retentions_by_user(
        args,
        retentions=action_retentions,
        device=device,
        configs=configs,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler=BASELINE_SCHEDULER,
                    scheduler_spec=BASELINE_SCHEDULER,
                    desired_retention=retention,
                    goal_cost_weight=None,
                    metrics=metrics,
                    runtime_s=runtime_s,
                )
            )
    for cost_weight, metrics_by_user, runtime_s in evaluate_policies_by_user(
        args,
        model=model,
        cost_weights=eval_cost_weights,
        action_retentions=action_retentions,
        goal_norm_max=max(cost_weights),
        device=device,
        configs=configs,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler="fsrs6_oracle_stationary_finite_distill",
                    scheduler_spec="fsrs6_oracle_stationary_finite_distill",
                    desired_retention=None,
                    goal_cost_weight=cost_weight,
                    metrics=metrics,
                    runtime_s=runtime_s,
                )
            )
    eval_runtime_s = time.perf_counter() - eval_start

    results_path = args.out_dir / "results.csv"
    regret_path = args.out_dir / "regret_auc.csv"
    summary_path = args.out_dir / "summary.csv"
    train_summary_path = args.out_dir / "train_summary.csv"
    _write_csv(results_path, rows)
    auc_rows = _build_regret_auc_rows(rows)
    _write_regret_auc_csv(regret_path, auc_rows)
    write_auc_summary(summary_path, auc_rows)
    write_train_summary(train_summary_path, stats)

    print(f"Wrote model: {model_out}")
    print(f"Wrote CSV: {results_path}")
    print(f"Wrote regret AUC CSV: {regret_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote train summary CSV: {train_summary_path}")
    print(
        "Multi-user stationary finite distill: "
        f"users={','.join(str(user_id) for user_id in user_ids)} "
        f"params={params} device={device} "
        f"teacher_s={teacher_runtime_s:.2f} train_s={train_runtime_s:.2f} "
        f"agreement_s={agreement_runtime_s:.2f} eval_s={eval_runtime_s:.2f} "
        f"final_ce={final_loss:.5f} train_agreement={final_agreement:.4f} "
        f"eval_agreement={eval_agreement:.4f}"
    )


if __name__ == "__main__":
    main()
