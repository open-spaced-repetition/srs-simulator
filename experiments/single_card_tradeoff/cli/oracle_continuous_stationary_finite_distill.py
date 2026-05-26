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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.config import (
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.core.defaults import (
    DEFAULT_FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.retention_space import (
    MIN_TARGET_RETENTION,
    validate_retention_values,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
)
from experiments.single_card_tradeoff.cli.uvfa_ppo import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_TRAIN_ENVS,
    FSRS6SingleCardBatch,
    SimMetrics,
    fsrs_config_kwargs,
    parse_csv_floats,
    scalar_objective,
)
from experiments.single_card_tradeoff.models.policy_runtime import (
    RetentionDistillNet,
    predicted_retentions,
    retention_logits_for_retentions,
)
from experiments.single_card_tradeoff.oracles import (
    FSRS6ContinuousStationaryFiniteOracle,
    FSRS6ContinuousStationaryUniformTerminationOracle,
    retention_interval_float,
)
from experiments.single_card_tradeoff.oracles.dp_cache import OracleDPCacheConfig
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float


POLICY_TYPE = "fsrs6_oracle_continuous_stationary_finite_distill"
DEFAULT_DISTILL_EPOCHS = 128
DEFAULT_STEPS_PER_EPOCH = 64
DEFAULT_EVAL_PARTICLES = 10_000
DEFAULT_OBS_MODE = "oracle_stationary"
DEFAULT_HIDDEN_SIZE = 8
DEFAULT_NETWORK_DEPTH = 2
DEFAULT_RETENTION_MIN = 0.5
DEFAULT_RETENTION_MAX = 0.98
DEFAULT_INTERVAL_LOSS_WEIGHT = 1.0
DEFAULT_RETENTION_LOGIT_LOSS_WEIGHT = 1.0
DEFAULT_TABLE_SAMPLES_PER_WEIGHT = 256
TEACHER_POLICY_STATIONARY_FINITE = "continuous_stationary_finite"
TEACHER_POLICY_STATIONARY_UNIFORM_H = "continuous_stationary_uniform_h"


@dataclass(frozen=True)
class DistillTrainStats:
    epochs: int
    steps_per_epoch: int
    transitions: int
    final_loss: float
    final_interval_loss: float
    final_retention_loss: float
    mean_loss: float
    mean_interval_loss: float
    runtime_s: float


class ContinuousStationaryFiniteOracleGuide:
    def __init__(
        self,
        *,
        days: int,
        cost_weights: Sequence[float],
        s_grid_size: int,
        d_grid_size: int,
        retention_min: float,
        retention_max: float,
        interval_chunk_size: int,
        device: torch.device,
        max_iterations: int,
        tolerance: float,
        progress: bool,
        teacher_policy: str = TEACHER_POLICY_STATIONARY_FINITE,
        fsrs_config: SingleCardFSRS6Config | None = None,
        cache_config: OracleDPCacheConfig | None = None,
    ) -> None:
        self.device = device
        self.cost_weights = torch.tensor(
            list(cost_weights), device=device, dtype=torch.float32
        )
        self.teacher_policy = teacher_policy
        oracle_kwargs: dict[str, Any] = {
            "days": days,
            "s_grid_size": s_grid_size,
            "d_grid_size": d_grid_size,
            "retention_min": retention_min,
            "retention_max": retention_max,
            "interval_chunk_size": interval_chunk_size,
            "device": device,
            "cache_config": cache_config,
            **fsrs_config_kwargs(fsrs_config),
        }
        if teacher_policy == TEACHER_POLICY_STATIONARY_UNIFORM_H:
            uniform_oracle = FSRS6ContinuousStationaryUniformTerminationOracle(
                **oracle_kwargs
            )
            solution = uniform_oracle.solve_stationary_uniform_termination_policies(
                cost_weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            self.oracle = uniform_oracle
        elif teacher_policy == TEACHER_POLICY_STATIONARY_FINITE:
            finite_oracle = FSRS6ContinuousStationaryFiniteOracle(**oracle_kwargs)
            solution = finite_oracle.solve_stationary_finite_policies(
                cost_weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            self.oracle = finite_oracle
        else:
            raise ValueError(f"Unsupported teacher_policy: {teacher_policy}")
        if not all(solution.converged):
            failed = [
                format_float(weight)
                for weight, converged in zip(
                    cost_weights,
                    solution.converged,
                    strict=True,
                )
                if not converged
            ]
            raise RuntimeError(
                "Continuous stationary oracle did not converge for cost "
                "weights: " + ",".join(failed)
            )
        self.policy = solution.policy.to(device=device, dtype=torch.float32)
        self.objectives = solution.objectives.to(device=device)
        self.iterations = solution.iterations
        self.residuals = solution.residuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the continuous stationary finite-lifecycle FSRS6 oracle into "
            "a scalar desired-retention policy."
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
        default=",".join(
            format_float(value)
            for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights for the oracle teacher.",
    )
    parser.add_argument(
        "--teacher-policy",
        choices=[
            TEACHER_POLICY_STATIONARY_FINITE,
            TEACHER_POLICY_STATIONARY_UNIFORM_H,
        ],
        default=TEACHER_POLICY_STATIONARY_FINITE,
        help=(
            "Oracle teacher to distill. continuous_stationary_uniform_h solves "
            "the best stationary policy under hidden H~Uniform{1,days}."
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Retention anchors used only to size the auxiliary checkpoint head.",
    )
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--obs-mode",
        choices=["oracle_stationary"],
        default=DEFAULT_OBS_MODE,
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--network", choices=["mlp", "residual"], default="residual")
    parser.add_argument("--network-depth", type=int, default=DEFAULT_NETWORK_DEPTH)
    parser.add_argument("--retention-min", type=float, default=DEFAULT_RETENTION_MIN)
    parser.add_argument("--retention-max", type=float, default=DEFAULT_RETENTION_MAX)
    parser.add_argument(
        "--interval-loss-weight",
        type=float,
        default=DEFAULT_INTERVAL_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--retention-logit-loss-weight",
        type=float,
        default=DEFAULT_RETENTION_LOGIT_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--oracle-s-grid-size", type=int, default=DEFAULT_ORACLE_S_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-d-grid-size", type=int, default=DEFAULT_ORACLE_D_GRID_SIZE
    )
    parser.add_argument("--oracle-interval-chunk-size", type=int, default=64)
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
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    )
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/"
            "fsrs6_oracle_continuous_stationary_finite_distill_results.csv"
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY,
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_torch_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_uniform_table_batch(
    guide: ContinuousStationaryFiniteOracleGuide,
    *,
    cost_weights: Sequence[float],
    samples_per_weight: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_count, s_count, d_count = guide.policy.shape
    weight_idx = torch.arange(weight_count, device=device)[:, None].expand(
        weight_count,
        samples_per_weight,
    )
    s_idx = torch.randint(
        s_count,
        (weight_count, samples_per_weight),
        device=device,
        generator=generator,
    )
    d_idx = torch.randint(
        d_count,
        (weight_count, samples_per_weight),
        device=device,
        generator=generator,
    )
    target_retention = guide.policy[weight_idx, s_idx, d_idx].reshape(-1)
    max_goal = max(1.0, max(cost_weights))
    goal_norm = torch.log1p(
        torch.tensor(cost_weights, device=device, dtype=torch.float32)
    ) / torch.log1p(torch.tensor(max_goal, device=device, dtype=torch.float32))
    obs = torch.stack(
        [
            (s_idx.to(dtype=torch.float32) / float(s_count - 1)).reshape(-1),
            (d_idx.to(dtype=torch.float32) / float(d_count - 1)).reshape(-1),
            goal_norm[:, None].expand(weight_count, samples_per_weight).reshape(-1),
        ],
        dim=1,
    )
    s_values = guide.oracle.s_grid.index_select(0, s_idx.reshape(-1)).to(
        dtype=torch.float32
    )
    return obs, target_retention, s_values


def _continuous_log_intervals(
    *,
    guide: ContinuousStationaryFiniteOracleGuide,
    s: torch.Tensor,
    retention: torch.Tensor,
) -> torch.Tensor:
    interval = retention_interval_float(
        s=s,
        retention=retention,
        factor=guide.oracle.factor.to(device=s.device, dtype=s.dtype),
        decay=guide.oracle.decay.to(device=s.device, dtype=s.dtype),
    )
    interval = torch.clamp(interval, min=1.0, max=float(guide.oracle.horizon + 1))
    return torch.log(interval)


def train_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    guide: ContinuousStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[RetentionDistillNet, DistillTrainStats]:
    torch.manual_seed(args.seed)
    model = RetentionDistillNet(
        obs_dim=3,
        hidden_size=args.hidden_size,
        action_count=len(action_retentions),
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 91_000)
    losses: list[float] = []
    interval_losses: list[float] = []
    final_interval_loss = math.nan
    final_retention_loss = math.nan
    start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            obs, target_retention, s_values = sample_uniform_table_batch(
                guide,
                cost_weights=cost_weights,
                samples_per_weight=args.table_samples_per_weight,
                device=device,
                generator=generator,
            )
            target_logit = retention_logits_for_retentions(
                target_retention,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            pred_logit, _ = model(obs)
            pred_retention = predicted_retentions(
                pred_logit,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            pred_log_interval = _continuous_log_intervals(
                guide=guide,
                s=s_values,
                retention=pred_retention,
            )
            target_log_interval = _continuous_log_intervals(
                guide=guide,
                s=s_values,
                retention=target_retention,
            )
            interval_loss = nn.functional.smooth_l1_loss(
                pred_log_interval,
                target_log_interval,
            )
            retention_loss = nn.functional.smooth_l1_loss(pred_logit, target_logit)
            loss = (
                float(args.interval_loss_weight) * interval_loss
                + float(args.retention_logit_loss_weight) * retention_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            interval_losses.append(float(interval_loss.item()))
            epoch_loss += loss_value
            final_interval_loss = float(interval_loss.item())
            final_retention_loss = float(retention_loss.item())
        if not args.no_progress:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"loss={epoch_loss / float(max(1, args.steps_per_epoch)):.6f}",
                flush=True,
            )
    return model, DistillTrainStats(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        transitions=(
            args.epochs
            * args.steps_per_epoch
            * args.table_samples_per_weight
            * len(cost_weights)
        ),
        final_loss=losses[-1] if losses else math.nan,
        final_interval_loss=final_interval_loss,
        final_retention_loss=final_retention_loss,
        mean_loss=sum(losses) / len(losses) if losses else math.nan,
        mean_interval_loss=sum(interval_losses) / len(interval_losses)
        if interval_losses
        else math.nan,
        runtime_s=time.perf_counter() - start,
    )


@torch.inference_mode()
def evaluate_table_fit(
    *,
    model: RetentionDistillNet,
    guide: ContinuousStationaryFiniteOracleGuide,
    device: torch.device,
    cost_weights: Sequence[float],
    retention_min: float,
    retention_max: float,
) -> dict[str, float]:
    weight_count, s_count, d_count = guide.policy.shape
    s_grid, d_grid = torch.meshgrid(
        torch.arange(s_count, device=device),
        torch.arange(d_count, device=device),
        indexing="ij",
    )
    max_goal = max(1.0, max(cost_weights))
    goal_norm = torch.log1p(
        torch.tensor(cost_weights, device=device, dtype=torch.float32)
    ) / torch.log1p(torch.tensor(max_goal, device=device, dtype=torch.float32))
    total_count = 0.0
    total_retention_abs = 0.0
    total_log_interval_abs = 0.0
    model.eval()
    for weight_idx in range(weight_count):
        obs = torch.stack(
            [
                s_grid.reshape(-1).to(dtype=torch.float32) / float(s_count - 1),
                d_grid.reshape(-1).to(dtype=torch.float32) / float(d_count - 1),
                torch.full(
                    (s_count * d_count,),
                    float(goal_norm[weight_idx].item()),
                    device=device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        raw, _ = model(obs)
        pred = predicted_retentions(
            raw,
            retention_min=retention_min,
            retention_max=retention_max,
        )
        target = guide.policy[weight_idx].reshape(-1)
        s_values = guide.oracle.s_grid.index_select(0, s_grid.reshape(-1)).to(
            dtype=torch.float32
        )
        total_count += float(target.numel())
        total_retention_abs += torch.abs(pred - target).sum().item()
        total_log_interval_abs += (
            torch.abs(
                _continuous_log_intervals(guide=guide, s=s_values, retention=pred)
                - _continuous_log_intervals(guide=guide, s=s_values, retention=target)
            )
            .sum()
            .item()
        )
    denom = max(1.0, total_count)
    return {
        "eval_retention_mae": total_retention_abs / denom,
        "eval_log_interval_mae": total_log_interval_abs / denom,
    }


@torch.inference_mode()
def evaluate_policy(
    model: RetentionDistillNet,
    *,
    args: argparse.Namespace,
    device: torch.device,
    cost_weight: float,
    action_retentions: Sequence[float],
    particles: int,
    seed: int,
    goal_norm_max: float,
    fsrs_config: SingleCardFSRS6Config | None,
) -> SimMetrics:
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
        obs_mode=args.obs_mode,
        **fsrs_config_kwargs(fsrs_config),
    )
    model_dtype = next(model.parameters()).dtype
    model.eval()
    while not bool(env.done.all().item()):
        raw_retention, _ = model(env.obs().to(dtype=model_dtype))
        retention = predicted_retentions(
            raw_retention.to(dtype=env.dtype),
            retention_min=args.retention_min,
            retention_max=args.retention_max,
        )
        env.step_retentions(retention)
    return env.metrics()


def row_from_metrics(
    *,
    args: argparse.Namespace,
    cost_weight: float,
    metrics: SimMetrics,
    runtime_s: float,
    train_stats: DistillTrainStats,
    eval_stats: dict[str, float],
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    row: dict[str, Any] = {
        "scheduler": POLICY_TYPE,
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
        "scalar_objective": scalar_objective(metrics, cost_weight),
        "runtime_s": runtime_s,
        "train_final_loss": train_stats.final_loss,
        "train_final_interval_loss": train_stats.final_interval_loss,
        "train_final_retention_loss": train_stats.final_retention_loss,
        "train_mean_loss": train_stats.mean_loss,
        "train_mean_interval_loss": train_stats.mean_interval_loss,
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
        "train_final_interval_loss",
        "train_final_retention_loss",
        "train_mean_loss",
        "train_mean_interval_loss",
        "train_runtime_s",
        "eval_retention_mae",
        "eval_log_interval_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_model(
    path: Path,
    *,
    model: RetentionDistillNet,
    guide: ContinuousStationaryFiniteOracleGuide,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    train_stats: DistillTrainStats,
    eval_stats: dict[str, float],
    teacher_runtime_s: float,
    fsrs_config: SingleCardFSRS6Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    oracle_metadata: dict[str, Any] = {
        "teacher_policy": f"{guide.teacher_policy}_oracle",
        "oracle_stationary_finite_max_iterations": (
            args.oracle_stationary_finite_max_iterations
        ),
        "oracle_stationary_finite_tolerance": (args.oracle_stationary_finite_tolerance),
        "oracle_stationary_finite_objectives": [
            float(value) for value in guide.objectives.tolist()
        ],
        "oracle_stationary_finite_iterations": list(guide.iterations),
        "oracle_stationary_finite_residuals": list(guide.residuals),
    }
    if guide.teacher_policy == TEACHER_POLICY_STATIONARY_UNIFORM_H:
        oracle_metadata.update(
            {
                "termination_distribution": (
                    FSRS6ContinuousStationaryUniformTerminationOracle.TERMINATION_DISTRIBUTION_VERSION
                ),
                "stationary_uniform_policy_iteration": (
                    FSRS6ContinuousStationaryUniformTerminationOracle.STATIONARY_UNIFORM_POLICY_ITERATION_VERSION
                ),
            }
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": POLICY_TYPE,
            "action_mode": "desired_retention",
            "obs_mode": args.obs_mode,
            "obs_dim": model.obs_dim,
            "cost_weights": list(cost_weights),
            "teacher_cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "retention_min": args.retention_min,
            "retention_max": args.retention_max,
            "interval_loss_weight": args.interval_loss_weight,
            "retention_logit_loss_weight": args.retention_logit_loss_weight,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_interval_chunk_size": args.oracle_interval_chunk_size,
            **oracle_metadata,
            **fsrs_config.checkpoint_payload(),
            "train_epochs": train_stats.epochs,
            "train_steps_per_epoch": train_stats.steps_per_epoch,
            "train_transitions": train_stats.transitions,
            "train_final_loss": train_stats.final_loss,
            "train_final_interval_loss": train_stats.final_interval_loss,
            "train_final_retention_loss": train_stats.final_retention_loss,
            "train_mean_loss": train_stats.mean_loss,
            "train_mean_interval_loss": train_stats.mean_interval_loss,
            "train_runtime_s": train_stats.runtime_s,
            "teacher_runtime_s": teacher_runtime_s,
            **eval_stats,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cache_config = configure_oracle_dp_cache_from_args(args)
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
    if not MIN_TARGET_RETENTION <= args.retention_min <= args.retention_max < 1.0:
        raise SystemExit("--retention-min/max must satisfy 0.5 <= min <= max < 1.")
    if args.interval_loss_weight < 0.0:
        raise SystemExit("--interval-loss-weight must be >= 0.")
    if args.retention_logit_loss_weight < 0.0:
        raise SystemExit("--retention-logit-loss-weight must be >= 0.")
    if args.interval_loss_weight == 0.0 and args.retention_logit_loss_weight == 0.0:
        raise SystemExit("At least one loss weight must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")
    if args.oracle_stationary_finite_max_iterations <= 0:
        raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
    if args.oracle_stationary_finite_tolerance <= 0.0:
        raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")
    if args.table_samples_per_weight <= 0:
        raise SystemExit("--table-samples-per-weight must be > 0.")

    device = resolve_torch_device(args.torch_device)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    fsrs_config = load_single_card_fsrs6_config(args)
    teacher_start = time.perf_counter()
    guide = ContinuousStationaryFiniteOracleGuide(
        days=args.days,
        cost_weights=cost_weights,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        interval_chunk_size=args.oracle_interval_chunk_size,
        device=device,
        max_iterations=args.oracle_stationary_finite_max_iterations,
        tolerance=args.oracle_stationary_finite_tolerance,
        progress=not args.no_progress,
        teacher_policy=args.teacher_policy,
        fsrs_config=fsrs_config,
        cache_config=cache_config,
    )
    teacher_runtime_s = time.perf_counter() - teacher_start
    model, train_stats = train_model(
        args,
        device=device,
        guide=guide,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    eval_stats = evaluate_table_fit(
        model=model,
        guide=guide,
        device=device,
        cost_weights=cost_weights,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
    )
    save_model(
        args.model_out,
        model=model,
        guide=guide,
        args=args,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        train_stats=train_stats,
        eval_stats=eval_stats,
        teacher_runtime_s=teacher_runtime_s,
        fsrs_config=fsrs_config,
    )

    rows: list[dict[str, Any]] = []
    for cost_weight in cost_weights:
        start = time.perf_counter()
        metrics = evaluate_policy(
            model,
            args=args,
            device=device,
            cost_weight=cost_weight,
            action_retentions=action_retentions,
            particles=args.eval_particles,
            seed=args.seed + 82_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=max(cost_weights),
            fsrs_config=fsrs_config,
        )
        rows.append(
            row_from_metrics(
                args=args,
                cost_weight=cost_weight,
                metrics=metrics,
                runtime_s=time.perf_counter() - start,
                train_stats=train_stats,
                eval_stats=eval_stats,
            )
        )
    write_csv(args.out, rows)
    print(f"Wrote CSV: {args.out}")
    print(f"Wrote model: {args.model_out}")
    print(
        " ".join(
            [
                f"teacher_runtime_s={teacher_runtime_s:.2f}",
                f"train_final_loss={train_stats.final_loss:.6f}",
                f"retention_mae={eval_stats['eval_retention_mae']:.6f}",
                f"log_interval_mae={eval_stats['eval_log_interval_mae']:.6f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
