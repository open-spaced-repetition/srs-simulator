from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff import DEFAULT_FIXED_INTERVALS
from experiments.uvfa_ppo_single_card import (
    DEFAULT_COST_WEIGHTS,
    FSRS6SingleCardBatch,
    SimMetrics,
    TrainStats,
    best_objective,
    evaluate_fixed_interval,
    evaluate_static_fsrs,
    normalize_advantages,
    parse_csv_floats,
    row_from_metrics,
    scalar_objective,
    write_csv,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recurrent UVFA PPO scheduler whose continuous action is "
            "the log review interval."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help=(
            "Comma-separated UVFA scalarization weights. A policy row optimizes "
            "card_expected_retrievability - weight * card_minutes_per_day."
        ),
    )
    parser.add_argument(
        "--fixed-intervals",
        default=",".join(format_float(value) for value in DEFAULT_FIXED_INTERVALS),
        help="Fixed-interval baseline points.",
    )
    parser.add_argument("--train-envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=36)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--advantage-normalization",
        choices=["global", "goal"],
        default="goal",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--initial-mean-interval",
        type=float,
        default=32.0,
        help="Initial actor mean, in physical days, before log-space conversion.",
    )
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=0.7,
        help="Initial Gaussian log standard deviation for log-interval actions.",
    )
    parser.add_argument(
        "--max-interval-days",
        type=int,
        default=None,
        help="Clamp exp(action) to this many days. Defaults to days * 4.",
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
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/single_card_tradeoff/uvfa_ppo_rnn_interval_results.csv"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("logs/single_card_tradeoff/uvfa_ppo_rnn_interval_policy.pt"),
    )
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


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


def recurrent_forward_sequence(
    *,
    model: RecurrentIntervalPolicyValueNet,
    obs: torch.Tensor,
    goals: torch.Tensor,
    done: torch.Tensor,
    initial_hidden: torch.Tensor,
    max_goal_weight: float,
) -> tuple[Normal, torch.Tensor]:
    hidden = initial_hidden
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    zero_hidden = torch.zeros_like(hidden)
    for step in range(obs.shape[0]):
        if step > 0:
            reset = done[step - 1].to(dtype=torch.bool).unsqueeze(1)
            hidden = torch.where(reset, zero_hidden, hidden)
        hidden = model.encode(obs[step], hidden)
        dist, value = model.dist_value(
            hidden,
            goals[step],
            max_goal_weight=max_goal_weight,
        )
        means.append(dist.mean.squeeze(1))
        stds.append(dist.stddev.squeeze(1))
        values.append(value)
    mean = torch.stack(means)
    std = torch.stack(stds)
    return Normal(mean, std), torch.stack(values)


def train_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cost_weights: Sequence[float],
) -> tuple[RecurrentIntervalPolicyValueNet, TrainStats]:
    torch.manual_seed(args.seed)
    dtype = torch.float32
    max_interval_days = args.max_interval_days or args.days * 4
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=args.train_envs,
        cost_weights=cost_weights,
        action_retentions=[0.9],
        device=device,
        dtype=dtype,
        seed=args.seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
        obs_mode="belief",
        max_interval_days=max_interval_days,
    )
    model = RecurrentIntervalPolicyValueNet(
        obs_dim=env.obs_dim,
        hidden_size=args.hidden_size,
        initial_mean_interval=args.initial_mean_interval,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    obs = env.obs()
    hidden_state = model.initial_state(
        args.train_envs,
        device=device,
        dtype=dtype,
    )
    start = time.perf_counter()

    for update in range(args.updates):
        rollout_initial_hidden = hidden_state.detach().clone()
        obs_buf = torch.empty(
            (args.rollout_steps, args.train_envs, env.obs_dim),
            device=device,
            dtype=dtype,
        )
        action_buf = torch.empty(
            (args.rollout_steps, args.train_envs), device=device, dtype=dtype
        )
        logprob_buf = torch.empty_like(action_buf)
        goal_buf = torch.empty_like(action_buf)
        reward_buf = torch.empty_like(action_buf)
        done_buf = torch.empty_like(action_buf)
        value_buf = torch.empty_like(action_buf)

        for step in range(args.rollout_steps):
            with torch.no_grad():
                hidden = model.encode(obs, hidden_state)
                dist, value = model.dist_value(
                    hidden,
                    env.goal_weight,
                    max_goal_weight=max(cost_weights),
                )
                action = dist.sample().squeeze(1)
                logprob = dist.log_prob(action.unsqueeze(1)).squeeze(1)
            next_obs, reward, done = env.step_log_interval(action)
            obs_buf[step] = obs
            action_buf[step] = action
            logprob_buf[step] = logprob
            goal_buf[step] = env.goal_weight
            reward_buf[step] = reward
            done_buf[step] = done.to(dtype=dtype)
            value_buf[step] = value

            hidden_state = hidden.detach()
            if done.any():
                done_idx = done.nonzero(as_tuple=False).squeeze(1)
                env.reset_indices(done_idx)
                hidden_state[done_idx] = 0.0
                next_obs = env.obs()
            obs = next_obs

        with torch.no_grad():
            next_hidden = model.encode(obs, hidden_state)
            _, next_value = model.dist_value(
                next_hidden,
                env.goal_weight,
                max_goal_weight=max(cost_weights),
            )
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

        flat_advantages = normalize_advantages(
            advantages.reshape(-1),
            goal_buf.reshape(-1),
            mode=args.advantage_normalization,
        ).reshape_as(advantages)

        for _ in range(args.ppo_epochs):
            dist, new_value = recurrent_forward_sequence(
                model=model,
                obs=obs_buf,
                goals=goal_buf,
                done=done_buf,
                initial_hidden=rollout_initial_hidden,
                max_goal_weight=max(cost_weights),
            )
            new_logprob = dist.log_prob(action_buf)
            entropy = dist.entropy().mean()
            logratio = new_logprob - logprob_buf
            ratio = logratio.exp()
            pg_loss_1 = -flat_advantages * ratio
            pg_loss_2 = -flat_advantages * torch.clamp(
                ratio,
                1.0 - args.clip_coef,
                1.0 + args.clip_coef,
            )
            policy_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()

            value_clipped = value_buf + torch.clamp(
                new_value - value_buf,
                -args.clip_coef,
                args.clip_coef,
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (new_value - returns).pow(2),
                    (value_clipped - returns).pow(2),
                ).mean()
            )
            loss = (
                policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        if not args.no_progress:
            print(
                f"update={update + 1}/{args.updates} "
                f"mean_step_reward={float(reward_buf.mean().item()):.5f} "
                f"log_std={float(model.log_std.item()):.3f}",
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
    model: RecurrentIntervalPolicyValueNet,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weight: float,
    particles: int,
    seed: int,
    goal_norm_max: float,
) -> SimMetrics:
    model_dtype = next(model.parameters()).dtype
    max_interval_days = args.max_interval_days or args.days * 4
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=particles,
        cost_weights=[cost_weight],
        action_retentions=[0.9],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode="belief",
        max_interval_days=max_interval_days,
    )
    hidden_state = model.initial_state(
        particles,
        device=device,
        dtype=model_dtype,
    )
    model.eval()
    while not bool(env.done.all().item()):
        obs = env.obs().to(dtype=model_dtype)
        hidden = model.encode(obs, hidden_state)
        dist, _ = model.dist_value(
            hidden,
            env.goal_weight.to(dtype=model_dtype),
            max_goal_weight=goal_norm_max,
        )
        action = dist.mean.squeeze(1)
        _, _, done = env.step_log_interval(action.to(dtype=env.dtype))
        hidden_state = hidden.detach()
        if done.any():
            hidden_state[done.nonzero(as_tuple=False).squeeze(1)] = 0.0
    return env.metrics()


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
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
            linewidth=1.4 if label == "uvfa_ppo_rnn_interval" else 1.0,
            alpha=0.9 if label == "uvfa_ppo_rnn_interval" else 0.5,
            label=label,
        )
    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title("Recurrent UVFA PPO log-interval tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_model(
    path: Path,
    *,
    model: RecurrentIntervalPolicyValueNet,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    train_stats: TrainStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": "uvfa_ppo_rnn_interval",
            "action_mode": "log_interval",
            "cost_weights": list(cost_weights),
            "days": args.days,
            "obs_dim": model.obs_dim,
            "obs_mode": "belief",
            "hidden_size": model.hidden_size,
            "rnn_type": "gru",
            "initial_mean_interval": args.initial_mean_interval,
            "initial_log_std": args.initial_log_std,
            "max_interval_days": args.max_interval_days or args.days * 4,
            "train_updates": train_stats.updates,
            "train_transitions": train_stats.transitions,
            "train_runtime_s": train_stats.runtime_s,
        },
        path,
    )


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
    if args.hidden_size <= 0:
        raise SystemExit("--hidden-size must be > 0.")
    if args.initial_mean_interval <= 0.0:
        raise SystemExit("--initial-mean-interval must be > 0.")
    if args.max_interval_days is not None and args.max_interval_days < 1:
        raise SystemExit("--max-interval-days must be >= 1.")

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    fixed_intervals = parse_csv_floats(args.fixed_intervals, name="--fixed-intervals")
    baseline_particles = args.baseline_particles or args.eval_particles

    model, train_stats = train_policy(args, device=device, cost_weights=cost_weights)
    save_model(
        args.model_out,
        model=model,
        args=args,
        cost_weights=cost_weights,
        train_stats=train_stats,
    )

    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, SimMetrics] = {}
    for interval in fixed_intervals:
        start = time.perf_counter()
        metrics = evaluate_fixed_interval(
            args=args,
            device=device,
            interval=interval,
            particles=baseline_particles,
            seed=args.seed + 20_000 + int(round(interval)),
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

    static_reference_retentions = [0.5, 0.7, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99]
    for retention in static_reference_retentions:
        start = time.perf_counter()
        metrics = evaluate_static_fsrs(
            args=args,
            device=device,
            retention=retention,
            particles=baseline_particles,
            seed=args.seed + 10_000 + int(round(retention * 10_000)),
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

    ppo_deltas: list[float] = []
    for cost_weight in cost_weights:
        start = time.perf_counter()
        metrics = evaluate_policy(
            model,
            args=args,
            device=device,
            cost_weight=cost_weight,
            particles=args.eval_particles,
            seed=args.seed + 30_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=max(cost_weights),
        )
        runtime_s = time.perf_counter() - start
        ppo_scalar = scalar_objective(metrics, cost_weight)
        _, best_fixed = best_objective(
            baseline_metrics,
            cost_weight=cost_weight,
            prefix="fixed@",
        )
        _, best_fsrs = best_objective(
            baseline_metrics,
            cost_weight=cost_weight,
            prefix="fsrs@",
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
                scheduler="uvfa_ppo_rnn_interval",
                scheduler_spec="uvfa_ppo_rnn_interval",
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
    for row in rows:
        if row["scheduler"] != "uvfa_ppo_rnn_interval":
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
    baseline_label = {
        "fixed": "best fixed interval",
        "fsrs": "best static FSRS target",
        "overall": "best fixed/static-FSRS",
    }[args.baseline]
    if passed:
        print(f"PASS: recurrent UVFA PPO beat {baseline_label} for every goal.")
    else:
        print(f"FAIL: recurrent UVFA PPO did not beat {baseline_label} for every goal.")


if __name__ == "__main__":
    main()
