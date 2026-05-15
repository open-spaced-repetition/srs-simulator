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
from torch import nn
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.tradeoff import (  # noqa: E402
    DEFAULT_FIXED_INTERVALS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
    SingleCardFSRS6Config,
)
from experiments.single_card_tradeoff.oracle_frontier import FSRS6GridOracle  # noqa: E402
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds
from simulator.scheduler_spec import format_float

DEFAULT_COST_WEIGHTS = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
DEFAULT_TRAIN_ENVS = 1024
DEFAULT_UPDATES = 36
DEFAULT_ROLLOUT_STEPS = 64
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINIBATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_GAMMA = 1.0
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_ADVANTAGE_NORMALIZATION = "goal"
DEFAULT_OBS_MODE = "rich"
DEFAULT_NETWORK = "residual"
DEFAULT_NETWORK_DEPTH = 3
DEFAULT_GUIDE_POLICY = "oracle"
DEFAULT_ORACLE_S_GRID_SIZE = 64
DEFAULT_ORACLE_D_GRID_SIZE = 32
DEFAULT_CLIP_COEF = 0.2
DEFAULT_PRIOR_COEF = 0.05
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VALUE_COEF = 0.5
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_WARMUP_EPOCHS = 16
DEFAULT_WARMUP_STEPS = 8


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a UVFA PPO scheduler on the single-card lifecycle tradeoff."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help=(
            "Comma-separated scalarization goals. A policy row optimizes "
            "card_expected_retrievability - weight * card_minutes_per_day."
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to PPO.",
    )
    parser.add_argument(
        "--fixed-intervals",
        default=",".join(format_float(value) for value in DEFAULT_FIXED_INTERVALS),
        help="Fixed-interval baseline points.",
    )
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument(
        "--advantage-normalization",
        choices=["global", "goal"],
        default=DEFAULT_ADVANTAGE_NORMALIZATION,
        help=(
            "Normalize PPO advantages globally or separately per UVFA cost-weight "
            "goal. Per-goal normalization keeps one goal from dominating updates."
        ),
    )
    parser.add_argument(
        "--obs-mode",
        choices=[
            "basic",
            "rich",
            "oracle",
            "oracle_rho",
            "oracle_rho4",
            "oracle_rho3",
        ],
        default=DEFAULT_OBS_MODE,
        help=(
            "Observation features for the UVFA policy. 'oracle' uses only "
            "stability, difficulty, remaining horizon, and goal cost weight; "
            "oracle_rho variants add or substitute the log remaining/stability ratio."
        ),
    )
    parser.add_argument(
        "--network",
        choices=["mlp", "residual"],
        default=DEFAULT_NETWORK,
        help="Policy/value architecture.",
    )
    parser.add_argument(
        "--network-depth",
        type=int,
        default=DEFAULT_NETWORK_DEPTH,
        help="Hidden blocks for --network residual; ignored by the legacy MLP.",
    )
    parser.add_argument(
        "--guide-policy",
        choices=["oracle", "static", "none"],
        default=DEFAULT_GUIDE_POLICY,
        help=(
            "Teacher used for actor warmup and policy regularization. "
            "'oracle' uses a finite-horizon FSRS grid oracle; 'static' uses the "
            "previous static-FSRS target prior."
        ),
    )
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=DEFAULT_ORACLE_S_GRID_SIZE,
        help="Stability grid size for --guide-policy oracle.",
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=DEFAULT_ORACLE_D_GRID_SIZE,
        help="Difficulty grid size for --guide-policy oracle.",
    )
    parser.add_argument("--clip-coef", type=float, default=DEFAULT_CLIP_COEF)
    parser.add_argument(
        "--prior-coef",
        type=float,
        default=DEFAULT_PRIOR_COEF,
        help=(
            "Small supervised regularization toward --guide-policy during PPO "
            "updates. This keeps nearby UVFA goals separated while PPO still "
            "optimizes returns."
        ),
    )
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    parser.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help=(
            "Actor-only supervised warmup epochs from --guide-policy. The PPO "
            "phase still optimizes the policy after this initialization."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Event steps sampled per warmup epoch.",
    )
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument(
        "--baseline-particles",
        type=int,
        default=None,
        help="Particles for baselines. Defaults to --eval-particles.",
    )
    parser.add_argument(
        "--baseline",
        choices=["fixed", "fsrs", "overall"],
        default="fixed",
        help=(
            "Baseline family used for the printed pass/fail summary. CSV always "
            "includes fixed and static-FSRS reference rows."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/uvfa_ppo_results.csv"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/uvfa_ppo_policy.pt"),
    )
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


@dataclass
class SimMetrics:
    card_expected_retrievability: float
    card_minutes_per_day: float
    card_reviews_per_day: float
    card_total_reviews: float
    card_total_lapses: float
    card_total_cost_seconds: float
    observed_retention: float | None


def fsrs_config_kwargs(
    fsrs_config: SingleCardFSRS6Config | None,
) -> dict[str, Any]:
    if fsrs_config is None:
        return {}
    return {
        "fsrs_weights": fsrs_config.fsrs_weights,
        "first_rating_prob": fsrs_config.first_rating_prob,
        "review_rating_prob": fsrs_config.review_rating_prob,
        "learning_costs": fsrs_config.learning_costs,
        "review_costs": fsrs_config.review_costs,
    }


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
        if any(retention <= 0.0 or retention >= 1.0 for retention in action_retentions):
            raise ValueError("action retentions must be within (0, 1).")
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


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 0.5 * self.net(hidden)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_count: int,
        hidden_size: int,
        architecture: str = "mlp",
        depth: int = 3,
    ) -> None:
        super().__init__()
        if architecture not in {"mlp", "residual"}:
            raise ValueError("architecture must be 'mlp' or 'residual'.")
        if depth <= 0:
            raise ValueError("depth must be > 0.")
        self.obs_dim = obs_dim
        self.architecture = architecture
        if architecture == "mlp":
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        else:
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.SiLU(),
                *[ResidualBlock(hidden_size) for _ in range(depth)],
                nn.LayerNorm(hidden_size),
            )
        self.policy = nn.Linear(hidden_size, action_count)
        self.value = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


@dataclass
class TrainStats:
    updates: int
    transitions: int
    runtime_s: float


def normalize_advantages(
    advantages: torch.Tensor,
    goals: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "global":
        return (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    if mode != "goal":
        raise ValueError("advantage normalization mode must be 'global' or 'goal'.")

    normalized = torch.empty_like(advantages)
    for goal in torch.unique(goals):
        mask = goals == goal
        goal_advantages = advantages[mask]
        normalized[mask] = (goal_advantages - goal_advantages.mean()) / (
            goal_advantages.std(unbiased=False) + 1e-8
        )
    return normalized


def static_retention_prior(goal_weight: torch.Tensor) -> torch.Tensor:
    """Map scalarization weights to a strong static-FSRS retention prior."""
    retention = torch.full_like(goal_weight, 0.85)
    retention = torch.where(
        goal_weight <= 96.0, torch.full_like(retention, 0.90), retention
    )
    retention = torch.where(
        goal_weight <= 48.0, torch.full_like(retention, 0.93), retention
    )
    retention = torch.where(
        goal_weight <= 20.0, torch.full_like(retention, 0.96), retention
    )
    retention = torch.where(
        goal_weight <= 10.0, torch.full_like(retention, 0.98), retention
    )
    return retention


def nearest_retention_action(
    *,
    target_retention: torch.Tensor,
    action_retentions: torch.Tensor,
) -> torch.Tensor:
    distance = torch.abs(target_retention[:, None] - action_retentions[None, :])
    return torch.argmin(distance, dim=1)


class PolicyGuide:
    def labels(self, env: FSRS6SingleCardBatch) -> torch.Tensor:
        raise NotImplementedError


class StaticRetentionGuide(PolicyGuide):
    def labels(self, env: FSRS6SingleCardBatch) -> torch.Tensor:
        return nearest_retention_action(
            target_retention=static_retention_prior(env.goal_weight),
            action_retentions=env.action_retentions,
        )


class OracleGridGuide(PolicyGuide):
    def __init__(
        self,
        *,
        days: int,
        cost_weights: Sequence[float],
        action_retentions: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        device: torch.device,
        progress: bool,
        fsrs_config: SingleCardFSRS6Config | None = None,
    ) -> None:
        self.days = int(days)
        self.horizon = int(days - 1)
        self.device = device
        self.cost_weights = torch.tensor(
            list(cost_weights), device=device, dtype=torch.float32
        )
        self.oracle = FSRS6GridOracle(
            days=days,
            action_retentions=action_retentions,
            s_grid_size=s_grid_size,
            d_grid_size=d_grid_size,
            **fsrs_config_kwargs(fsrs_config),
        )
        self.policy_tables: list[torch.Tensor] = []
        self.metrics_by_weight: dict[float, float] = {}
        for cost_weight in cost_weights:
            solution = self.oracle.solve(
                float(cost_weight),
                progress=progress,
                capture_policy=True,
            )
            if solution.policy is None:
                raise RuntimeError("Oracle policy table was not captured.")
            self.policy_tables.append(solution.policy.to(device=device))
            self.metrics_by_weight[float(cost_weight)] = (
                solution.metrics.scalar_objective
            )

    def labels(self, env: FSRS6SingleCardBatch) -> torch.Tensor:
        remaining = torch.clamp((env.days - 1) - env.day, min=0, max=self.horizon)
        s_idx = self._s_to_idx(env.s)
        d_idx = self._d_to_idx(env.d)
        goal_idx = torch.argmin(
            torch.abs(
                env.goal_weight.to(dtype=self.cost_weights.dtype)[:, None]
                - self.cost_weights[None, :]
            ),
            dim=1,
        )
        labels = torch.empty(env.env_count, device=env.device, dtype=torch.int64)
        for idx in torch.unique(goal_idx).tolist():
            goal_mask = goal_idx == int(idx)
            labels[goal_mask] = self.policy_tables[int(idx)][
                remaining[goal_mask],
                s_idx[goal_mask],
                d_idx[goal_mask],
            ]
        return labels

    def _s_to_idx(self, s: torch.Tensor) -> torch.Tensor:
        log_s = torch.log(
            torch.clamp(s, self.oracle.bounds.s_min, self.oracle.bounds.s_max)
        )
        ratio = (log_s - self.oracle.log_s_min) / (
            self.oracle.log_s_max - self.oracle.log_s_min
        )
        return torch.clamp(
            torch.round(ratio * float(self.oracle.s_grid.numel() - 1)),
            min=0,
            max=self.oracle.s_grid.numel() - 1,
        ).to(torch.int64)

    def _d_to_idx(self, d: torch.Tensor) -> torch.Tensor:
        ratio = torch.clamp(d, self.oracle.bounds.d_min, self.oracle.bounds.d_max)
        ratio = (ratio - self.oracle.bounds.d_min) / (
            self.oracle.bounds.d_max - self.oracle.bounds.d_min
        )
        return torch.clamp(
            torch.round(ratio * float(self.oracle.d_grid.numel() - 1)),
            min=0,
            max=self.oracle.d_grid.numel() - 1,
        ).to(torch.int64)


def build_policy_guide(
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> PolicyGuide | None:
    if args.guide_policy == "none":
        return None
    if args.guide_policy == "static":
        return StaticRetentionGuide()
    return OracleGridGuide(
        days=args.days,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        progress=not args.no_progress,
        fsrs_config=fsrs_config,
    )


def warmup_policy(
    *,
    args: argparse.Namespace,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    env: FSRS6SingleCardBatch,
    obs: torch.Tensor,
    guide: PolicyGuide | None,
) -> torch.Tensor:
    if args.warmup_epochs <= 0 or guide is None:
        return obs
    for _ in range(args.warmup_epochs):
        for _ in range(args.warmup_steps):
            label = guide.labels(env)
            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                next_obs, _, done = env.step(label)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs
    return obs


def train_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    policy_guide: PolicyGuide | None = None,
    build_guide_if_missing: bool = True,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> tuple[PolicyValueNet, TrainStats]:
    torch.manual_seed(args.seed)
    dtype = torch.float32
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=args.train_envs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=dtype,
        seed=args.seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
        obs_mode=args.obs_mode,
        **fsrs_config_kwargs(fsrs_config),
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
    start = time.perf_counter()
    guide = policy_guide
    if guide is None and build_guide_if_missing:
        guide = build_policy_guide(
            args=args,
            device=device,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            fsrs_config=fsrs_config,
        )
    obs = warmup_policy(
        args=args,
        model=model,
        optimizer=optimizer,
        env=env,
        obs=obs,
        guide=guide,
    )

    for update in range(args.updates):
        obs_buf = torch.empty(
            (args.rollout_steps, args.train_envs, env.obs_dim),
            device=device,
            dtype=dtype,
        )
        action_buf = torch.empty(
            (args.rollout_steps, args.train_envs), device=device, dtype=torch.int64
        )
        logprob_buf = torch.empty(
            (args.rollout_steps, args.train_envs), device=device, dtype=dtype
        )
        guide_label_buf = torch.empty(
            (args.rollout_steps, args.train_envs), device=device, dtype=torch.int64
        )
        goal_buf = torch.empty_like(logprob_buf)
        reward_buf = torch.empty_like(logprob_buf)
        done_buf = torch.empty_like(logprob_buf)
        value_buf = torch.empty_like(logprob_buf)

        for step in range(args.rollout_steps):
            guide_label = (
                guide.labels(env)
                if guide is not None
                else torch.zeros(args.train_envs, device=device, dtype=torch.int64)
            )
            with torch.no_grad():
                logits, value = model(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            next_obs, reward, done = env.step(action)
            obs_buf[step] = obs
            action_buf[step] = action
            logprob_buf[step] = logprob
            guide_label_buf[step] = guide_label
            goal_buf[step] = env.goal_weight
            reward_buf[step] = reward
            done_buf[step] = done.to(dtype=dtype)
            value_buf[step] = value
            if done.any():
                env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                next_obs = env.obs()
            obs = next_obs

        with torch.no_grad():
            _, next_value = model(obs)
            advantages = torch.zeros_like(reward_buf)
            last_gae = torch.zeros(args.train_envs, device=device, dtype=dtype)
            for step in reversed(range(args.rollout_steps)):
                if step == args.rollout_steps - 1:
                    next_nonterminal = 1.0 - done_buf[step]
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[step]
                    next_values = value_buf[step + 1]
                delta = (
                    reward_buf[step]
                    + args.gamma * next_values * next_nonterminal
                    - value_buf[step]
                )
                last_gae = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                )
                advantages[step] = last_gae
            returns = advantages + value_buf

        flat_obs = obs_buf.reshape((-1, env.obs_dim))
        flat_actions = action_buf.reshape(-1)
        flat_logprobs = logprob_buf.reshape(-1)
        flat_guide_labels = guide_label_buf.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_goals = goal_buf.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = value_buf.reshape(-1)
        flat_advantages = normalize_advantages(
            flat_advantages,
            flat_goals,
            mode=args.advantage_normalization,
        )

        batch_size = int(flat_obs.shape[0])
        permutation = torch.randperm(batch_size, device=device)
        for _ in range(args.ppo_epochs):
            for start_idx in range(0, batch_size, args.minibatch_size):
                mb_idx = permutation[start_idx : start_idx + args.minibatch_size]
                logits, new_value = model(flat_obs.index_select(0, mb_idx))
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(flat_actions.index_select(0, mb_idx))
                entropy = dist.entropy().mean()
                prior_loss = torch.tensor(0.0, device=device, dtype=dtype)
                if args.prior_coef > 0.0 and guide is not None:
                    prior_label = flat_guide_labels.index_select(0, mb_idx)
                    prior_loss = nn.functional.cross_entropy(logits, prior_label)
                old_logprob = flat_logprobs.index_select(0, mb_idx)
                logratio = new_logprob - old_logprob
                ratio = logratio.exp()
                mb_adv = flat_advantages.index_select(0, mb_idx)
                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()

                old_value = flat_values.index_select(0, mb_idx)
                value_target = flat_returns.index_select(0, mb_idx)
                value_clipped = old_value + torch.clamp(
                    new_value - old_value,
                    -args.clip_coef,
                    args.clip_coef,
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (new_value - value_target).pow(2),
                        (value_clipped - value_target).pow(2),
                    ).mean()
                )
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    + args.prior_coef * prior_loss
                    - args.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if not args.no_progress:
            mean_reward = float(reward_buf.mean().item())
            print(
                f"update={update + 1}/{args.updates} "
                f"mean_step_reward={mean_reward:.5f}",
                flush=True,
            )

    runtime_s = time.perf_counter() - start
    return model, TrainStats(
        updates=args.updates,
        transitions=args.updates * args.rollout_steps * args.train_envs,
        runtime_s=runtime_s,
    )


@torch.inference_mode()
def evaluate_policy(
    model: PolicyValueNet,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weight: float,
    action_retentions: Sequence[float],
    particles: int,
    seed: int,
    goal_norm_max: float,
    obs_mode: str | None = None,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> SimMetrics:
    resolved_obs_mode = obs_mode or getattr(args, "obs_mode", "basic")
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=particles,
        cost_weights=[cost_weight],
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode=resolved_obs_mode,
        **fsrs_config_kwargs(fsrs_config),
    )
    model.eval()
    while not bool(env.done.all().item()):
        obs = env.obs().to(dtype=next(model.parameters()).dtype)
        logits, _ = model(obs)
        action = torch.argmax(logits, dim=1)
        env.step(action)
    return env.metrics()


@torch.inference_mode()
def evaluate_static_fsrs(
    *,
    args: argparse.Namespace,
    device: torch.device,
    retention: float,
    particles: int,
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> SimMetrics:
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=particles,
        cost_weights=[0.0],
        action_retentions=[retention],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        **fsrs_config_kwargs(fsrs_config),
    )
    while not bool(env.done.all().item()):
        active = (~env.done).nonzero(as_tuple=False).squeeze(1)
        action = torch.zeros(env.env_count, device=device, dtype=torch.int64)
        env.step(action)
        if active.numel() == 0:
            break
    return env.metrics()


@torch.inference_mode()
def evaluate_fixed_interval(
    *,
    args: argparse.Namespace,
    device: torch.device,
    interval: float,
    particles: int,
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> SimMetrics:
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=particles,
        cost_weights=[0.0],
        action_retentions=[0.9],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        **fsrs_config_kwargs(fsrs_config),
    )
    interval_days = max(1, int(round(interval)))
    while not bool(env.done.all().item()):
        active = (~env.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            break
        intervals = torch.full(
            (int(active.numel()),), interval_days, device=device, dtype=torch.int64
        )
        remaining = (env.days - 1) - env.day.index_select(0, active)
        memorized_days = torch.minimum(intervals, remaining)
        memorized = env._memorized_sum(env.s.index_select(0, active), memorized_days)
        cost_seconds = env.pending_cost_seconds.index_select(0, active)
        env.total_memorized[active] += memorized
        env.total_cost_seconds[active] += cost_seconds
        next_day = env.day.index_select(0, active) + intervals
        terminal = next_day > (env.days - 1)
        if terminal.any():
            env.done[active[terminal]] = True
        continuing = ~terminal
        if continuing.any():
            cont_idx = active[continuing]
            elapsed = intervals[continuing].to(dtype=env.dtype)
            retrievability = env._forgetting_curve(
                elapsed,
                env.s.index_select(0, cont_idx),
            )
            fail = (
                torch.rand(
                    retrievability.shape,
                    device=device,
                    dtype=env.dtype,
                    generator=env.generator,
                )
                > retrievability
            )
            success_rating = (
                torch.multinomial(
                    env.review_rating_prob.expand(int(cont_idx.numel()), -1),
                    num_samples=1,
                    replacement=True,
                    generator=env.generator,
                )
                .squeeze(1)
                .to(torch.int64)
                + 2
            )
            rating = torch.where(fail, torch.ones_like(success_rating), success_rating)
            env._update_review(cont_idx, elapsed, rating, retrievability)
            env.day[cont_idx] = next_day[continuing]
            env.pending_cost_seconds[cont_idx] = env.review_costs.index_select(
                0, rating - 1
            )
            env.last_interval[cont_idx] = elapsed
            env.last_rating[cont_idx] = rating
            env.total_reviews[cont_idx] += 1
            env.total_lapses[cont_idx] += (rating == 1).to(torch.int64)
    return env.metrics()


def scalar_objective(metrics: SimMetrics, cost_weight: float) -> float:
    return (
        metrics.card_expected_retrievability
        - cost_weight * metrics.card_minutes_per_day
    )


def row_from_metrics(
    *,
    args: argparse.Namespace,
    scheduler: str,
    scheduler_spec: str,
    metrics: SimMetrics,
    particles: int,
    seed: int,
    runtime_s: float,
    goal_cost_weight: float | None = None,
    desired_retention: float | None = None,
    fixed_interval: float | None = None,
    scalar: float | None = None,
    delta_vs_fixed: float | None = None,
    delta_vs_fsrs: float | None = None,
    delta_vs_overall: float | None = None,
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    return {
        "environment": getattr(args, "env", "fsrs6_default"),
        "scheduler": scheduler,
        "scheduler_spec": scheduler_spec,
        "goal_cost_weight": goal_cost_weight,
        "desired_retention": desired_retention,
        "fixed_interval": fixed_interval,
        "seed": seed,
        "days": args.days,
        "particles": particles,
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
        "delta_vs_best_fixed": delta_vs_fixed,
        "delta_vs_best_fsrs": delta_vs_fsrs,
        "delta_vs_best_overall": delta_vs_overall,
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
        "delta_vs_best_fixed",
        "delta_vs_best_fsrs",
        "delta_vs_best_overall",
        "runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        candidate_mem = float(candidate["deck_expected_memorized"])
        candidate_minutes = float(candidate["deck_minutes_per_day"])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_mem = float(other["deck_expected_memorized"])
            other_minutes = float(other["deck_minutes_per_day"])
            no_worse = other_mem >= candidate_mem and other_minutes <= candidate_minutes
            strictly_better = (
                other_mem > candidate_mem or other_minutes < candidate_minutes
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            float(row["deck_expected_memorized"]),
            float(row["deck_minutes_per_day"]),
        ),
    )


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    curve_rows = [
        row for row in rows if row["scheduler"] in {"uvfa_ppo", "fixed", "fsrs6_static"}
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in curve_rows:
        groups.setdefault(str(row["scheduler"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in groups.items():
        if label == "fixed":
            group = sorted(group, key=lambda row: float(row["fixed_interval"]))
        elif label == "fsrs6_static":
            group = sorted(group, key=lambda row: float(row["desired_retention"]))
        else:
            group = sorted(group, key=lambda row: float(row["goal_cost_weight"]))
        ax.plot(
            [row["deck_expected_memorized"] for row in group],
            [row["deck_minutes_per_day"] for row in group],
            marker="o",
            linewidth=1.2 if label == "uvfa_ppo" else 1.0,
            alpha=0.9 if label == "uvfa_ppo" else 0.5,
            label=label,
        )

    frontier = pareto_frontier(curve_rows)
    if frontier:
        ax.plot(
            [row["deck_expected_memorized"] for row in frontier],
            [row["deck_minutes_per_day"] for row in frontier],
            color="black",
            marker="o",
            linewidth=2.0,
            markersize=4,
            label="Pareto frontier",
        )
    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title("UVFA PPO single-card tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_model(
    path: Path,
    *,
    model: PolicyValueNet,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    train_stats: TrainStats,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> None:
    config_payload = fsrs_config.checkpoint_payload() if fsrs_config is not None else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "obs_dim": model.obs_dim,
            "obs_mode": args.obs_mode,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "guide_policy": args.guide_policy,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "train_updates": train_stats.updates,
            "train_transitions": train_stats.transitions,
            "train_runtime_s": train_stats.runtime_s,
            **config_payload,
        },
        path,
    )


def best_objective(
    metrics_by_name: dict[str, SimMetrics],
    *,
    cost_weight: float,
    prefix: str,
) -> tuple[str, float]:
    candidates = {
        name: scalar_objective(metrics, cost_weight)
        for name, metrics in metrics_by_name.items()
        if name.startswith(prefix)
    }
    if not candidates:
        raise ValueError(f"No candidates for prefix {prefix}.")
    return max(candidates.items(), key=lambda item: item[1])


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.train_envs <= 0 or args.eval_particles <= 0:
        raise SystemExit("--train-envs and --eval-particles must be > 0.")
    if args.updates < 0:
        raise SystemExit("--updates must be >= 0.")
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be > 0.")
    if args.network_depth <= 0:
        raise SystemExit("--network-depth must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    fixed_intervals = parse_csv_floats(args.fixed_intervals, name="--fixed-intervals")
    baseline_particles = args.baseline_particles or args.eval_particles
    fsrs_config = load_single_card_fsrs6_config(args)

    model, train_stats = train_policy(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        fsrs_config=fsrs_config,
    )
    save_model(
        args.model_out,
        model=model,
        args=args,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        train_stats=train_stats,
        fsrs_config=fsrs_config,
    )

    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, SimMetrics] = {}

    for retention in action_retentions:
        start = time.perf_counter()
        metrics = evaluate_static_fsrs(
            args=args,
            device=device,
            retention=retention,
            particles=baseline_particles,
            seed=args.seed + 10_000 + int(round(retention * 10_000)),
            fsrs_config=fsrs_config,
        )
        runtime_s = time.perf_counter() - start
        name = f"fsrs@{format_float(retention)}"
        baseline_metrics[name] = metrics
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="fsrs6_static",
                scheduler_spec=name,
                desired_retention=retention,
                metrics=metrics,
                particles=baseline_particles,
                seed=args.seed,
                runtime_s=runtime_s,
            )
        )

    for interval in fixed_intervals:
        start = time.perf_counter()
        metrics = evaluate_fixed_interval(
            args=args,
            device=device,
            interval=interval,
            particles=baseline_particles,
            seed=args.seed + 20_000 + int(round(interval)),
            fsrs_config=fsrs_config,
        )
        runtime_s = time.perf_counter() - start
        name = f"fixed@{format_float(interval)}"
        baseline_metrics[name] = metrics
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="fixed",
                scheduler_spec=name,
                fixed_interval=interval,
                metrics=metrics,
                particles=baseline_particles,
                seed=args.seed,
                runtime_s=runtime_s,
            )
        )

    ppo_deltas: list[float] = []
    for cost_weight in cost_weights:
        start = time.perf_counter()
        metrics = evaluate_policy(
            model,
            args=args,
            device=device,
            cost_weight=cost_weight,
            action_retentions=action_retentions,
            particles=args.eval_particles,
            seed=args.seed + 30_000 + int(round(cost_weight * 10)),
            goal_norm_max=max(cost_weights),
            fsrs_config=fsrs_config,
        )
        runtime_s = time.perf_counter() - start
        ppo_scalar = scalar_objective(metrics, cost_weight)
        _, best_fixed = best_objective(
            baseline_metrics, cost_weight=cost_weight, prefix="fixed@"
        )
        _, best_fsrs = best_objective(
            baseline_metrics, cost_weight=cost_weight, prefix="fsrs@"
        )
        best_overall = max(best_fixed, best_fsrs)
        if args.baseline == "fixed":
            ppo_deltas.append(ppo_scalar - best_fixed)
        elif args.baseline == "fsrs":
            ppo_deltas.append(ppo_scalar - best_fsrs)
        else:
            ppo_deltas.append(ppo_scalar - best_overall)
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="uvfa_ppo",
                scheduler_spec="uvfa_ppo",
                goal_cost_weight=cost_weight,
                metrics=metrics,
                particles=args.eval_particles,
                seed=args.seed,
                runtime_s=runtime_s,
                scalar=ppo_scalar,
                delta_vs_fixed=ppo_scalar - best_fixed,
                delta_vs_fsrs=ppo_scalar - best_fsrs,
                delta_vs_overall=ppo_scalar - best_overall,
            )
        )

    write_csv(args.out, rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        write_plot(plot_path, rows)
        print(f"Wrote plot: {plot_path}")

    passed = all(delta > 0.0 for delta in ppo_deltas)
    print(f"Wrote CSV: {args.out}")
    print(f"Wrote model: {args.model_out}")
    print(
        f"Training: updates={train_stats.updates} "
        f"transitions={train_stats.transitions} runtime_s={train_stats.runtime_s:.2f}"
    )
    baseline_label = {
        "fixed": "best fixed interval",
        "fsrs": "best static FSRS target",
        "overall": "best fixed/static-FSRS",
    }[args.baseline]
    for row in rows:
        if row["scheduler"] != "uvfa_ppo":
            continue
        print(
            " ".join(
                [
                    f"goal={row['goal_cost_weight']}",
                    f"card_mem={row['card_expected_retrievability']:.4f}",
                    f"card_min/day={row['card_minutes_per_day']:.6f}",
                    f"delta_fixed={row['delta_vs_best_fixed']:.6f}",
                    f"delta_fsrs={row['delta_vs_best_fsrs']:.6f}",
                    f"delta_overall={row['delta_vs_best_overall']:.6f}",
                ]
            )
        )
    if passed:
        print(f"PASS: UVFA PPO beat {baseline_label} for every evaluated goal.")
    else:
        print(f"FAIL: UVFA PPO did not beat {baseline_label} for every evaluated goal.")


if __name__ == "__main__":
    main()
