from __future__ import annotations

import math

import torch
from torch import nn


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


class QuadraticFeatureMap(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.output_dim = 1 + obs_dim + (obs_dim * (obs_dim + 1)) // 2

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = [
            torch.ones((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)
        ]
        features.append(obs)
        for left_idx in range(self.obs_dim):
            for right_idx in range(left_idx, self.obs_dim):
                features.append(
                    (obs[..., left_idx] * obs[..., right_idx]).unsqueeze(-1)
                )
        return torch.cat(features, dim=-1)


class ZeroValueHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*hidden.shape[:-1], 1), device=hidden.device, dtype=hidden.dtype
        )


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
        if architecture not in {"mlp", "residual", "linear", "quadratic"}:
            raise ValueError(
                "architecture must be 'mlp', 'residual', 'linear', or 'quadratic'."
            )
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
            feature_size = hidden_size
            self.policy = nn.Linear(feature_size, action_count)
            self.value = nn.Linear(feature_size, 1)
        elif architecture == "residual":
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.SiLU(),
                *[ResidualBlock(hidden_size) for _ in range(depth)],
                nn.LayerNorm(hidden_size),
            )
            feature_size = hidden_size
            self.policy = nn.Linear(feature_size, action_count)
            self.value = nn.Linear(feature_size, 1)
        elif architecture == "linear":
            self.body = nn.Identity()
            feature_size = obs_dim
            self.policy = nn.Linear(feature_size, action_count)
            self.value = ZeroValueHead()
        else:
            quadratic_body = QuadraticFeatureMap(obs_dim)
            self.body = quadratic_body
            feature_size = quadratic_body.output_dim
            self.policy = nn.Linear(feature_size, action_count, bias=False)
            self.value = ZeroValueHead()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        if isinstance(self.value, nn.Linear):
            nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.policy(hidden), self.value(hidden).squeeze(-1)
