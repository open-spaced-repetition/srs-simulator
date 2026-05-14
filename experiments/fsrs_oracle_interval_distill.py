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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.fsrs_oracle_frontier import FSRS6IntervalOracle
from experiments.uvfa_ppo_single_card import (
    DEFAULT_COST_WEIGHTS,
    FSRS6SingleCardBatch,
    ResidualBlock,
    SimMetrics,
    parse_csv_floats,
    scalar_objective,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float


@dataclass(frozen=True)
class DistillTrainStats:
    epochs: int
    steps_per_epoch: int
    transitions: int
    final_loss: float
    mean_loss: float
    runtime_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the FSRS6 integer-interval oracle into a 4D log-interval MLP."
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
            "Comma-separated scalarization weights. A policy row optimizes "
            "card_expected_retrievability - weight * card_minutes_per_day."
        ),
    )
    parser.add_argument("--train-envs", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument(
        "--network",
        choices=["mlp", "residual"],
        default="residual",
        help="Regression network architecture.",
    )
    parser.add_argument(
        "--network-depth",
        type=int,
        default=3,
        help="Hidden blocks for --network residual; ignored by the MLP.",
    )
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=64,
        help="Stability grid size for the interval oracle.",
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=32,
        help="Difficulty grid size for the interval oracle.",
    )
    parser.add_argument(
        "--oracle-interval-chunk-size",
        type=int,
        default=64,
        help="Interval candidates per Bellman-backup chunk.",
    )
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/single_card_tradeoff/fsrs6_oracle_interval_distill.csv"),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_interval_distill_policy.pt"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


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


def resolve_torch_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def train_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    oracle: FSRS6IntervalOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
) -> tuple[IntervalDistillNet, DistillTrainStats]:
    torch.manual_seed(args.seed)
    dtype = torch.float32
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
        obs_mode="oracle",
    )
    model = IntervalDistillNet(
        obs_dim=env.obs_dim,
        hidden_size=args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    policy_tables = policies.to(device=device)
    cost_weight_tensor = torch.tensor(cost_weights, device=device, dtype=dtype)
    obs = env.obs()
    losses: list[float] = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            labels = interval_oracle_labels(
                env=env,
                oracle=oracle,
                policies=policy_tables,
                cost_weights=cost_weight_tensor,
            )
            target = torch.log(labels.to(dtype=dtype))
            pred = model(obs)
            loss = nn.functional.smooth_l1_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            epoch_loss += loss_value
            with torch.no_grad():
                next_obs, _, done = env.step_intervals(labels)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs
        if not args.no_progress:
            mean_epoch_loss = epoch_loss / float(max(1, args.steps_per_epoch))
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={mean_epoch_loss:.6f}",
                flush=True,
            )

    runtime_s = time.perf_counter() - start
    return model, DistillTrainStats(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        transitions=args.epochs * args.steps_per_epoch * args.train_envs,
        final_loss=losses[-1] if losses else math.nan,
        mean_loss=sum(losses) / len(losses) if losses else math.nan,
        runtime_s=runtime_s,
    )


@torch.inference_mode()
def evaluate_interval_agreement(
    *,
    args: argparse.Namespace,
    model: IntervalDistillNet,
    device: torch.device,
    oracle: FSRS6IntervalOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    seed: int,
) -> dict[str, float]:
    dtype = torch.float64
    weight_count = len(cost_weights)
    env_count = args.eval_particles * weight_count
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=[0.9],
        device=device,
        dtype=dtype,
        seed=seed,
        exact_memory=True,
        goal_norm_max=max(cost_weights),
        obs_mode="oracle",
    )
    for weight_idx, cost_weight in enumerate(cost_weights):
        start = weight_idx * args.eval_particles
        stop = start + args.eval_particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        env.reset_indices(idx, goal_weight=cost_weight)

    policy_tables = policies.to(device=device)
    cost_weight_tensor = torch.tensor(cost_weights, device=device, dtype=dtype)
    model_dtype = next(model.parameters()).dtype
    total_count = 0.0
    total_loss = 0.0
    total_abs_log = 0.0
    total_sq_log = 0.0
    total_abs_interval = 0.0
    total_agree = 0.0
    start_time = time.perf_counter()
    model.eval()
    while not bool(env.done.all().item()):
        active = (~env.done).nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            break
        labels = interval_oracle_labels(
            env=env,
            oracle=oracle,
            policies=policy_tables,
            cost_weights=cost_weight_tensor,
        )
        obs = env.obs().to(dtype=model_dtype)
        pred_log = model(obs).to(dtype=dtype)
        target_log = torch.log(labels.to(dtype=dtype))
        active_pred = pred_log.index_select(0, active)
        active_target = target_log.index_select(0, active)
        active_labels = labels.index_select(0, active)
        log_error = active_pred - active_target
        clipped_pred = torch.clamp(
            active_pred,
            min=0.0,
            max=math.log(float(env.max_interval_days)),
        )
        pred_interval = torch.clamp(
            torch.round(torch.exp(clipped_pred)),
            min=1.0,
            max=float(env.max_interval_days),
        ).to(torch.int64)
        count = float(active.numel())
        total_count += count
        total_loss += nn.functional.smooth_l1_loss(
            active_pred,
            active_target,
            reduction="sum",
        ).item()
        total_abs_log += torch.abs(log_error).sum().item()
        total_sq_log += torch.square(log_error).sum().item()
        total_abs_interval += (
            torch.abs(pred_interval - active_labels).to(dtype=dtype).sum().item()
        )
        total_agree += (pred_interval == active_labels).to(dtype=dtype).sum().item()
        env.step_intervals(labels)

    denom = max(1.0, total_count)
    return {
        "eval_smooth_l1_loss": total_loss / denom,
        "eval_log_interval_mae": total_abs_log / denom,
        "eval_log_interval_rmse": math.sqrt(total_sq_log / denom),
        "eval_interval_mae_days": total_abs_interval / denom,
        "eval_rounded_interval_agreement": total_agree / denom,
        "eval_runtime_s": time.perf_counter() - start_time,
    }


@torch.inference_mode()
def evaluate_policy(
    model: IntervalDistillNet,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weight: float,
    particles: int,
    seed: int,
    goal_norm_max: float,
) -> SimMetrics:
    model_dtype = next(model.parameters()).dtype
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
        obs_mode="oracle",
    )
    model.eval()
    while not bool(env.done.all().item()):
        obs = env.obs().to(dtype=model_dtype)
        pred_log_interval = model(obs)
        env.step_log_interval(pred_log_interval.to(dtype=env.dtype))
    return env.metrics()


def row_from_metrics(
    *,
    args: argparse.Namespace,
    cost_weight: float,
    metrics: SimMetrics,
    runtime_s: float,
    scalar: float,
    train_stats: DistillTrainStats,
    eval_stats: dict[str, float],
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    row: dict[str, Any] = {
        "scheduler": "fsrs6_oracle_interval_distill",
        "goal_cost_weight": cost_weight,
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
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "scalar_objective": scalar,
        "runtime_s": runtime_s,
        "train_final_loss": train_stats.final_loss,
        "train_mean_loss": train_stats.mean_loss,
        "train_runtime_s": train_stats.runtime_s,
    }
    row.update(eval_stats)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scheduler",
        "goal_cost_weight",
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
        "runtime_s",
        "train_final_loss",
        "train_mean_loss",
        "train_runtime_s",
        "eval_smooth_l1_loss",
        "eval_log_interval_mae",
        "eval_log_interval_rmse",
        "eval_interval_mae_days",
        "eval_rounded_interval_agreement",
        "eval_runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_model(
    path: Path,
    *,
    model: IntervalDistillNet,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    train_stats: DistillTrainStats,
    eval_stats: dict[str, float],
    oracle_solve_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": "fsrs6_oracle_interval_distill",
            "action_mode": "log_interval",
            "obs_mode": "oracle",
            "obs_dim": model.obs_dim,
            "cost_weights": list(cost_weights),
            "days": args.days,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_interval_chunk_size": args.oracle_interval_chunk_size,
            "train_epochs": train_stats.epochs,
            "train_steps_per_epoch": train_stats.steps_per_epoch,
            "train_transitions": train_stats.transitions,
            "train_final_loss": train_stats.final_loss,
            "train_mean_loss": train_stats.mean_loss,
            "train_runtime_s": train_stats.runtime_s,
            "oracle_solve_runtime_s": oracle_solve_runtime_s,
            **eval_stats,
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
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0.")
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch must be > 0.")
    if args.hidden_size <= 0:
        raise SystemExit("--hidden-size must be > 0.")
    if args.network_depth <= 0:
        raise SystemExit("--network-depth must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")

    device = resolve_torch_device(args.torch_device)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")

    oracle = FSRS6IntervalOracle(
        days=args.days,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        interval_chunk_size=args.oracle_interval_chunk_size,
        device=device,
    )
    oracle_start = time.perf_counter()
    policies = oracle.solve_policies(cost_weights, progress=not args.no_progress)
    oracle_solve_runtime_s = time.perf_counter() - oracle_start

    model, train_stats = train_model(
        args,
        device=device,
        oracle=oracle,
        policies=policies,
        cost_weights=cost_weights,
    )
    eval_stats = evaluate_interval_agreement(
        args=args,
        model=model,
        device=device,
        oracle=oracle,
        policies=policies,
        cost_weights=cost_weights,
        seed=args.seed + 70_000,
    )
    save_model(
        args.model_out,
        model=model,
        args=args,
        cost_weights=cost_weights,
        train_stats=train_stats,
        eval_stats=eval_stats,
        oracle_solve_runtime_s=oracle_solve_runtime_s,
    )

    rows: list[dict[str, Any]] = []
    for cost_weight in cost_weights:
        start = time.perf_counter()
        metrics = evaluate_policy(
            model,
            args=args,
            device=device,
            cost_weight=cost_weight,
            particles=args.eval_particles,
            seed=args.seed + 80_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=max(cost_weights),
        )
        runtime_s = time.perf_counter() - start
        scalar = scalar_objective(metrics, cost_weight)
        rows.append(
            row_from_metrics(
                args=args,
                cost_weight=cost_weight,
                metrics=metrics,
                runtime_s=runtime_s,
                scalar=scalar,
                train_stats=train_stats,
                eval_stats=eval_stats,
            )
        )

    write_csv(args.out, rows)
    print(f"Wrote CSV: {args.out}")
    print(f"Wrote model: {args.model_out}")
    print(
        f"Interval oracle solve runtime_s={oracle_solve_runtime_s:.2f} device={device}"
    )
    print(
        " ".join(
            [
                f"train_final_loss={train_stats.final_loss:.6f}",
                f"eval_log_mae={eval_stats['eval_log_interval_mae']:.6f}",
                f"eval_log_rmse={eval_stats['eval_log_interval_rmse']:.6f}",
                "rounded_agreement="
                f"{eval_stats['eval_rounded_interval_agreement']:.4f}",
            ]
        )
    )
    for row in rows:
        print(
            " ".join(
                [
                    f"goal={row['goal_cost_weight']}",
                    f"card_mem={row['card_expected_retrievability']:.4f}",
                    f"card_min/day={row['card_minutes_per_day']:.6f}",
                    f"scalar={row['scalar_objective']:.6f}",
                    f"runtime_s={row['runtime_s']:.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
