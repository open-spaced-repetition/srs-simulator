from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from experiments.single_card_tradeoff.models.policy_net import ResidualBlock
from experiments.single_card_tradeoff.models.single_card_env import (
    FSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.oracles import FSRS6IntervalOracle

__all__ = [
    "IntervalAwareRetentionLossConfig",
    "RetentionDistillLoss",
    "IntervalDistillNet",
    "RetentionDistillNet",
    "RecurrentIntervalPolicyValueNet",
    "auxiliary_action_labels",
    "continuous_intervals_for_retentions",
    "interval_underprediction_weights",
    "interval_aware_retention_loss",
    "interval_distill_loss",
    "interval_oracle_labels",
    "omega_features",
    "oracle_d_to_idx",
    "oracle_s_to_idx",
    "predicted_intervals",
    "predicted_retentions",
    "retention_distill_loss",
    "retention_logits_for_retentions",
    "rounded_intervals_for_retentions",
    "target_retentions_for_intervals",
    "weighted_log_interval_smooth_l1",
]


@dataclass(frozen=True)
class IntervalAwareRetentionLossConfig:
    interval_weight: float
    retention_logit_weight: float
    underprediction_weight: float
    terminal_underprediction_weight: float


@dataclass(frozen=True)
class RetentionDistillLoss:
    total: torch.Tensor
    interval: torch.Tensor
    retention_logit: torch.Tensor


class IntervalDistillNet(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_size: int,
        architecture: str = "residual",
        depth: int = 3,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        if architecture not in {"mlp", "residual"}:
            raise ValueError("architecture must be 'mlp' or 'residual'.")
        if depth <= 0:
            raise ValueError("depth must be > 0.")
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.architecture = architecture
        self.depth = int(depth)
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
        self.output = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(obs)).squeeze(-1)


class RetentionDistillNet(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_size: int,
        action_count: int,
        architecture: str = "residual",
        depth: int = 2,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        if action_count <= 0:
            raise ValueError("action_count must be > 0.")
        if architecture not in {"mlp", "residual"}:
            raise ValueError("architecture must be 'mlp' or 'residual'.")
        if depth <= 0:
            raise ValueError("depth must be > 0.")
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.action_count = int(action_count)
        self.architecture = architecture
        self.depth = int(depth)
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
        self.retention = nn.Linear(hidden_size, 1)
        self.auxiliary_action = nn.Linear(hidden_size, action_count)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.retention.weight, gain=0.01)
        nn.init.orthogonal_(self.auxiliary_action.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.retention(hidden).squeeze(-1), self.auxiliary_action(hidden)


class RecurrentIntervalPolicyValueNet(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_size: int,
        initial_mean_interval: float,
        initial_log_std: float,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be > 0.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        if initial_mean_interval <= 0.0:
            raise ValueError("initial_mean_interval must be > 0.")
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.GRUCell(obs_dim, hidden_size)
        joint_dim = hidden_size + 2
        self.body = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Linear(joint_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_size, 1)
        self.value = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.tensor(float(initial_log_std)))
        self._init_weights(initial_mean_interval=initial_mean_interval)

    def _init_weights(self, *, initial_mean_interval: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        with torch.no_grad():
            self.mean.bias.fill_(math.log(initial_mean_interval))
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def encode(self, obs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs, hidden)

    def dist_value(
        self,
        hidden: torch.Tensor,
        goal_weight: torch.Tensor,
        *,
        max_goal_weight: float,
    ) -> tuple[Normal, torch.Tensor]:
        omega = omega_features(goal_weight, max_goal_weight=max_goal_weight)
        joint = torch.cat([hidden, omega], dim=1)
        features = self.body(joint)
        mean = self.mean(features)
        std = torch.exp(torch.clamp(self.log_std, min=-5.0, max=2.0)).expand_as(mean)
        return Normal(mean, std), self.value(features).squeeze(-1)


def omega_features(
    goal_weight: torch.Tensor, *, max_goal_weight: float
) -> torch.Tensor:
    max_goal = max(1.0, float(max_goal_weight))
    goal = goal_weight.to(dtype=torch.float32)
    return torch.stack(
        [
            torch.log1p(goal) / math.log1p(max_goal),
            goal / max_goal,
        ],
        dim=1,
    )


def oracle_s_to_idx(oracle: FSRS6IntervalOracle, s: torch.Tensor) -> torch.Tensor:
    log_s = torch.log(torch.clamp(s, oracle.bounds.s_min, oracle.bounds.s_max))
    ratio = (log_s - oracle.log_s_min) / (oracle.log_s_max - oracle.log_s_min)
    return torch.clamp(
        torch.round(ratio * float(oracle.s_grid.numel() - 1)),
        min=0,
        max=oracle.s_grid.numel() - 1,
    ).to(torch.int64)


def oracle_d_to_idx(oracle: FSRS6IntervalOracle, d: torch.Tensor) -> torch.Tensor:
    ratio = torch.clamp(d, oracle.bounds.d_min, oracle.bounds.d_max)
    ratio = (ratio - oracle.bounds.d_min) / (oracle.bounds.d_max - oracle.bounds.d_min)
    return torch.clamp(
        torch.round(ratio * float(oracle.d_grid.numel() - 1)),
        min=0,
        max=oracle.d_grid.numel() - 1,
    ).to(torch.int64)


def interval_oracle_labels(
    *,
    env: FSRS6SingleCardBatch,
    oracle: FSRS6IntervalOracle,
    policies: torch.Tensor,
    cost_weights: torch.Tensor,
) -> torch.Tensor:
    remaining = torch.clamp((env.days - 1) - env.day, min=0, max=oracle.horizon)
    s_idx = oracle_s_to_idx(oracle, env.s)
    d_idx = oracle_d_to_idx(oracle, env.d)
    goal_idx = torch.argmin(
        torch.abs(
            env.goal_weight.to(dtype=cost_weights.dtype)[:, None]
            - cost_weights[None, :]
        ),
        dim=1,
    )
    labels = torch.empty(env.env_count, device=env.device, dtype=torch.int64)
    for idx in torch.unique(goal_idx).tolist():
        goal_mask = goal_idx == int(idx)
        labels[goal_mask] = policies[int(idx)][
            remaining[goal_mask],
            s_idx[goal_mask],
            d_idx[goal_mask],
        ]
    return labels


def _goal_norm(env: FSRS6SingleCardBatch) -> torch.Tensor:
    return torch.log1p(env.goal_weight) / math.log1p(max(1.0, env.max_goal_weight))


def predicted_intervals(
    *,
    env: FSRS6SingleCardBatch,
    log_interval: torch.Tensor,
    log_interval_bias: float,
    terminal_snap_ratio: float,
) -> torch.Tensor:
    adjusted = log_interval.to(dtype=env.dtype)
    if log_interval_bias:
        adjusted = adjusted + float(log_interval_bias) * _goal_norm(env)
    clipped = torch.clamp(
        adjusted,
        min=0.0,
        max=math.log(float(env.max_interval_days)),
    )
    intervals = torch.clamp(
        torch.round(torch.exp(clipped)),
        min=1.0,
        max=float(env.max_interval_days),
    ).to(torch.int64)
    if terminal_snap_ratio > 0.0:
        remaining = torch.clamp((env.days - 1) - env.day, min=0).to(torch.int64)
        snap = intervals.to(dtype=env.dtype) >= (
            remaining.to(dtype=env.dtype) * float(terminal_snap_ratio)
        )
        intervals = torch.where(snap, remaining + 1, intervals)
    return intervals


def predicted_retentions(
    raw_retention: torch.Tensor,
    *,
    retention_min: float,
    retention_max: float,
) -> torch.Tensor:
    unit = torch.sigmoid(raw_retention)
    return float(retention_min) + unit * (float(retention_max) - float(retention_min))


def retention_logits_for_retentions(
    retention: torch.Tensor,
    *,
    retention_min: float,
    retention_max: float,
) -> torch.Tensor:
    unit = (retention - float(retention_min)) / (
        float(retention_max) - float(retention_min)
    )
    return torch.logit(torch.clamp(unit, min=1e-6, max=1.0 - 1e-6))


def continuous_intervals_for_retentions(
    *,
    env: FSRS6SingleCardBatch,
    s: torch.Tensor,
    retention: torch.Tensor,
) -> torch.Tensor:
    clipped_retention = torch.clamp(retention, min=1e-7, max=1.0 - 1e-7)
    retention_factor = torch.pow(clipped_retention, 1.0 / env.decay) - 1.0
    interval = s / env.factor * retention_factor
    return torch.clamp(
        interval,
        min=1.0,
        max=float(env.max_interval_days),
    )


def rounded_intervals_for_retentions(
    *,
    env: FSRS6SingleCardBatch,
    retention: torch.Tensor,
    terminal_snap_ratio: float,
) -> torch.Tensor:
    intervals = env._intervals_for_retentions(env.s, retention)
    if terminal_snap_ratio > 0.0:
        remaining = torch.clamp((env.days - 1) - env.day, min=0).to(torch.int64)
        snap = intervals.to(dtype=env.dtype) >= (
            remaining.to(dtype=env.dtype) * float(terminal_snap_ratio)
        )
        intervals = torch.where(snap, remaining + 1, intervals)
    return intervals


def target_retentions_for_intervals(
    *,
    env: FSRS6SingleCardBatch,
    intervals: torch.Tensor,
    retention_min: float,
    retention_max: float,
) -> torch.Tensor:
    elapsed = intervals.to(dtype=env.dtype)
    retention = env._forgetting_curve(elapsed, env.s)
    return torch.clamp(
        retention,
        min=float(retention_min),
        max=float(retention_max),
    )


def auxiliary_action_labels(
    *,
    target_retention: torch.Tensor,
    action_retentions: torch.Tensor,
) -> torch.Tensor:
    return torch.argmin(
        torch.abs(target_retention[:, None] - action_retentions[None, :]),
        dim=1,
    )


def interval_aware_retention_loss(
    *,
    pred_log_interval: torch.Tensor,
    target_log_interval: torch.Tensor,
    labels: torch.Tensor,
    env: FSRS6SingleCardBatch,
    underprediction_loss_weight: float,
    terminal_underprediction_loss_weight: float,
) -> torch.Tensor:
    weights = interval_underprediction_weights(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        labels=labels,
        env=env,
        underprediction_loss_weight=underprediction_loss_weight,
        terminal_underprediction_loss_weight=terminal_underprediction_loss_weight,
    )
    return weighted_log_interval_smooth_l1(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        weights=weights,
    )


def interval_distill_loss(
    *,
    pred_log_interval: torch.Tensor,
    target_log_interval: torch.Tensor,
    labels: torch.Tensor,
    env: FSRS6SingleCardBatch,
    underprediction_loss_weight: float,
    terminal_underprediction_loss_weight: float,
) -> torch.Tensor:
    weights = interval_underprediction_weights(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        labels=labels,
        env=env,
        underprediction_loss_weight=underprediction_loss_weight,
        terminal_underprediction_loss_weight=terminal_underprediction_loss_weight,
    )
    return weighted_log_interval_smooth_l1(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        weights=weights,
    )


def interval_underprediction_weights(
    *,
    pred_log_interval: torch.Tensor,
    target_log_interval: torch.Tensor,
    labels: torch.Tensor,
    env: FSRS6SingleCardBatch,
    underprediction_loss_weight: float,
    terminal_underprediction_loss_weight: float,
) -> torch.Tensor:
    target_log_interval = target_log_interval.to(
        device=pred_log_interval.device,
        dtype=pred_log_interval.dtype,
    )
    under = (pred_log_interval < target_log_interval).to(dtype=pred_log_interval.dtype)
    weights = torch.ones_like(pred_log_interval)
    if underprediction_loss_weight:
        goal_norm = _goal_norm(env).to(device=weights.device, dtype=weights.dtype)
        weights = weights + (float(underprediction_loss_weight) * goal_norm * under)
    if terminal_underprediction_loss_weight:
        remaining = torch.clamp((env.days - 1) - env.day, min=0).to(torch.int64)
        labels = labels.to(device=remaining.device)
        terminal = (labels == (remaining + 1)).to(dtype=weights.dtype)
        weights = weights + (
            float(terminal_underprediction_loss_weight) * terminal * under
        )
    return weights


def weighted_log_interval_smooth_l1(
    *,
    pred_log_interval: torch.Tensor,
    target_log_interval: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    target_log_interval = target_log_interval.to(
        device=pred_log_interval.device,
        dtype=pred_log_interval.dtype,
    )
    weights = weights.to(device=pred_log_interval.device, dtype=pred_log_interval.dtype)
    elementwise = nn.functional.smooth_l1_loss(
        pred_log_interval,
        target_log_interval,
        reduction="none",
    )
    return torch.mean(elementwise * weights)


def retention_distill_loss(
    *,
    pred_logit: torch.Tensor,
    target_interval: torch.Tensor,
    env: FSRS6SingleCardBatch,
    config: IntervalAwareRetentionLossConfig,
    retention_min: float,
    retention_max: float,
) -> RetentionDistillLoss:
    pred_retention = predicted_retentions(
        pred_logit,
        retention_min=retention_min,
        retention_max=retention_max,
    )
    target_retention = target_retentions_for_intervals(
        env=env,
        intervals=target_interval,
        retention_min=retention_min,
        retention_max=retention_max,
    )
    pred_log_interval = torch.log(
        continuous_intervals_for_retentions(
            env=env,
            s=env.s,
            retention=pred_retention,
        )
    )
    target_log_interval = torch.log(
        target_interval.to(
            device=pred_log_interval.device,
            dtype=pred_log_interval.dtype,
        )
    )
    interval_weights = interval_underprediction_weights(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        labels=target_interval,
        env=env,
        underprediction_loss_weight=config.underprediction_weight,
        terminal_underprediction_loss_weight=config.terminal_underprediction_weight,
    )
    interval = weighted_log_interval_smooth_l1(
        pred_log_interval=pred_log_interval,
        target_log_interval=target_log_interval,
        weights=interval_weights,
    )
    target_logit = retention_logits_for_retentions(
        target_retention.to(device=pred_logit.device, dtype=pred_logit.dtype),
        retention_min=retention_min,
        retention_max=retention_max,
    )
    retention_logit = nn.functional.smooth_l1_loss(pred_logit, target_logit)
    total = (
        float(config.interval_weight) * interval
        + float(config.retention_logit_weight) * retention_logit
    )
    return RetentionDistillLoss(
        total=total,
        interval=interval,
        retention_logit=retention_logit,
    )
