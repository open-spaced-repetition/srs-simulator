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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    DEFAULT_FIXED_INTERVALS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
    SingleCardFSRS6Config,
)
from experiments.single_card_tradeoff.oracles import FSRS6GridOracle  # noqa: E402
from experiments.single_card_tradeoff.oracles.dp_cache import (  # noqa: E402
    OracleDPCacheConfig,
)
from experiments.single_card_tradeoff.models.policy_net import (  # noqa: E402,F401
    PolicyValueNet,
    QuadraticFeatureMap,
    ResidualBlock,
    ZeroValueHead,
)
from experiments.single_card_tradeoff.models.single_card_env import (  # noqa: E402
    FSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.core.types import SimMetrics  # noqa: E402
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
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
        cache_config: OracleDPCacheConfig | None = None,
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
            cache_config=cache_config,
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
    cache_config: OracleDPCacheConfig | None = None,
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
        cache_config=cache_config,
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
    cache_config: OracleDPCacheConfig | None = None,
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
            cache_config=cache_config,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model_checkpoint_payload(
            model=model,
            args=args,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            train_stats=train_stats,
            fsrs_config=fsrs_config,
        ),
        path,
    )


def model_checkpoint_payload(
    *,
    model: PolicyValueNet,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    train_stats: TrainStats,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> dict[str, Any]:
    config_payload = fsrs_config.checkpoint_payload() if fsrs_config is not None else {}
    return {
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
    }


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
    cache_config = configure_oracle_dp_cache_from_args(args)
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
    validate_retention_values(action_retentions, name="--action-retentions")
    fixed_intervals = parse_csv_floats(args.fixed_intervals, name="--fixed-intervals")
    baseline_particles = args.baseline_particles or args.eval_particles
    fsrs_config = load_single_card_fsrs6_config(args)

    model, train_stats = train_policy(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        fsrs_config=fsrs_config,
        cache_config=cache_config,
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
