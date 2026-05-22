from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

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
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
    SingleCardFSRS6Config,
)
from experiments.single_card_tradeoff.oracles import FSRS6IntervalOracle
from experiments.single_card_tradeoff.core.defaults import (
    DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.retention_space import (
    MIN_TARGET_RETENTION,
    validate_retention_values,
)
from experiments.single_card_tradeoff.cli.uvfa_ppo import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_TRAIN_ENVS,
    FSRS6SingleCardBatch,
    SimMetrics,
    fsrs_config_kwargs,
    parse_csv_floats,
    scalar_objective,
)
from experiments.single_card_tradeoff.models.policy_runtime import (
    IntervalAwareRetentionLossConfig,
    RetentionDistillNet,
    auxiliary_action_labels,
    continuous_intervals_for_retentions,
    interval_oracle_labels,
    predicted_retentions,
    retention_distill_loss,
    retention_logits_for_retentions,
    rounded_intervals_for_retentions,
    target_retentions_for_intervals,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float

DEFAULT_DISTILL_EPOCHS = 64
DEFAULT_STEPS_PER_EPOCH = 64
DEFAULT_EVAL_PARTICLES = 10_000
DEFAULT_OBS_MODE = "oracle_rho4"
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_NETWORK_DEPTH = 2
DEFAULT_RETENTION_MIN = MIN_TARGET_RETENTION
DEFAULT_RETENTION_MAX = 0.999
DEFAULT_INTERVAL_LOSS_WEIGHT = 1.0
DEFAULT_RETENTION_LOGIT_LOSS_WEIGHT = 0.0
DEFAULT_AUXILIARY_ACTION_LOSS_WEIGHT = 0.05
DEFAULT_UNDERPREDICTION_LOSS_WEIGHT = 6.0
DEFAULT_TERMINAL_UNDERPREDICTION_LOSS_WEIGHT = 12.0
DEFAULT_STUDENT_ROLLOUT_PROB = 0.75
DEFAULT_STUDENT_ROLLOUT_WARMUP_EPOCHS = 8
DEFAULT_TERMINAL_SNAP_RATIO = 0.85


@dataclass(frozen=True)
class DistillTrainStats:
    epochs: int
    steps_per_epoch: int
    transitions: int
    final_loss: float
    final_interval_loss: float
    final_retention_loss: float
    final_auxiliary_loss: float
    mean_loss: float
    mean_interval_loss: float
    runtime_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the FSRS6 integer-interval oracle into a continuous desired-"
            "retention policy."
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
            format_float(value) for value in DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete retentions used only for the equal-budget auxiliary head.",
    )
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--obs-mode",
        choices=[
            "basic",
            "rich",
            "belief",
            "oracle",
            "oracle_rho",
            "oracle_rho4",
            "oracle_rho3",
        ],
        default=DEFAULT_OBS_MODE,
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--network",
        choices=["mlp", "residual"],
        default="residual",
    )
    parser.add_argument("--network-depth", type=int, default=DEFAULT_NETWORK_DEPTH)
    parser.add_argument("--retention-min", type=float, default=DEFAULT_RETENTION_MIN)
    parser.add_argument("--retention-max", type=float, default=DEFAULT_RETENTION_MAX)
    parser.add_argument(
        "--interval-loss-weight",
        type=float,
        default=DEFAULT_INTERVAL_LOSS_WEIGHT,
        help=(
            "Weight for smooth-L1 loss on the log interval implied by the "
            "predicted desired retention."
        ),
    )
    parser.add_argument(
        "--retention-logit-loss-weight",
        type=float,
        default=DEFAULT_RETENTION_LOGIT_LOSS_WEIGHT,
        help="Weight for auxiliary smooth-L1 loss on desired-retention logits.",
    )
    parser.add_argument(
        "--auxiliary-action-loss-weight",
        type=float,
        default=DEFAULT_AUXILIARY_ACTION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--underprediction-loss-weight",
        type=float,
        default=DEFAULT_UNDERPREDICTION_LOSS_WEIGHT,
        help=(
            "Extra log-interval loss multiplier when the predicted retention "
            "implies an interval shorter than the teacher interval, scaled by "
            "normalized cost weight."
        ),
    )
    parser.add_argument(
        "--terminal-underprediction-loss-weight",
        type=float,
        default=DEFAULT_TERMINAL_UNDERPREDICTION_LOSS_WEIGHT,
        help=(
            "Extra log-interval loss multiplier when the teacher chose the "
            "terminal remaining+1 interval and the model predicts shorter."
        ),
    )
    parser.add_argument(
        "--student-rollout-prob",
        type=float,
        default=DEFAULT_STUDENT_ROLLOUT_PROB,
        help=(
            "Probability of stepping the training environment with the interval "
            "implied by the student's retention after warmup."
        ),
    )
    parser.add_argument(
        "--student-rollout-warmup-epochs",
        type=int,
        default=DEFAULT_STUDENT_ROLLOUT_WARMUP_EPOCHS,
        help="Teacher-forced warmup epochs before student-rollout sampling.",
    )
    parser.add_argument(
        "--terminal-snap-ratio",
        type=float,
        default=DEFAULT_TERMINAL_SNAP_RATIO,
        help=(
            "When the rounded interval implied by retention is at least this "
            "fraction of the remaining horizon, execute remaining+1. Use 0 to "
            "disable."
        ),
    )
    parser.add_argument("--oracle-s-grid-size", type=int, default=64)
    parser.add_argument("--oracle-d-grid-size", type=int, default=32)
    parser.add_argument("--oracle-interval-chunk-size", type=int, default=64)
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_results.csv"
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/fsrs6_oracle_retention_distill_policy.pt"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_torch_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    args: argparse.Namespace,
    *,
    device: torch.device,
    oracle: FSRS6IntervalOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None,
) -> tuple[RetentionDistillNet, DistillTrainStats]:
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
    model = RetentionDistillNet(
        obs_dim=env.obs_dim,
        hidden_size=args.hidden_size,
        action_count=len(action_retentions),
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    policy_tables = policies.to(device=device)
    cost_weight_tensor = torch.tensor(cost_weights, device=device, dtype=dtype)
    action_retention_tensor = torch.tensor(
        action_retentions,
        device=device,
        dtype=dtype,
    )
    obs = env.obs()
    loss_config = IntervalAwareRetentionLossConfig(
        interval_weight=args.interval_loss_weight,
        retention_logit_weight=args.retention_logit_loss_weight,
        underprediction_weight=args.underprediction_loss_weight,
        terminal_underprediction_weight=args.terminal_underprediction_loss_weight,
    )
    losses: list[float] = []
    interval_losses: list[float] = []
    final_interval_loss = math.nan
    final_retention_loss = math.nan
    final_auxiliary_loss = math.nan
    start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            interval_labels = interval_oracle_labels(
                env=env,
                oracle=oracle,
                policies=policy_tables,
                cost_weights=cost_weight_tensor,
            )
            pred_logit, aux_logits = model(obs)
            distill = retention_distill_loss(
                pred_logit=pred_logit,
                target_interval=interval_labels,
                env=env,
                config=loss_config,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            target_retention = target_retentions_for_intervals(
                env=env,
                intervals=interval_labels,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            aux_labels = auxiliary_action_labels(
                target_retention=target_retention,
                action_retentions=action_retention_tensor,
            )
            auxiliary_loss = nn.functional.cross_entropy(aux_logits, aux_labels)
            loss = distill.total + (
                float(args.auxiliary_action_loss_weight) * auxiliary_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_value = float(loss.item())
            losses.append(loss_value)
            interval_losses.append(float(distill.interval.item()))
            epoch_loss += loss_value
            final_interval_loss = float(distill.interval.item())
            final_retention_loss = float(distill.retention_logit.item())
            final_auxiliary_loss = float(auxiliary_loss.item())
            with torch.no_grad():
                rollout_prob = (
                    args.student_rollout_prob
                    if epoch >= args.student_rollout_warmup_epochs
                    else 0.0
                )
                if rollout_prob > 0.0:
                    student_logit, _ = model(obs)
                    student_retention = predicted_retentions(
                        student_logit,
                        retention_min=args.retention_min,
                        retention_max=args.retention_max,
                    )
                    student_intervals = rounded_intervals_for_retentions(
                        env=env,
                        retention=student_retention,
                        terminal_snap_ratio=args.terminal_snap_ratio,
                    )
                    use_student = (
                        torch.rand(
                            (env.env_count,),
                            device=device,
                            generator=env.generator,
                        )
                        < rollout_prob
                    )
                    step_intervals = torch.where(
                        use_student,
                        student_intervals,
                        interval_labels,
                    )
                else:
                    step_intervals = interval_labels
                next_obs, _, done = env.step_intervals(step_intervals)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs
        if not args.no_progress:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"loss={epoch_loss / float(max(1, args.steps_per_epoch)):.6f}",
                flush=True,
            )
    return model, DistillTrainStats(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        transitions=args.epochs * args.steps_per_epoch * args.train_envs,
        final_loss=losses[-1] if losses else math.nan,
        final_interval_loss=final_interval_loss,
        final_retention_loss=final_retention_loss,
        final_auxiliary_loss=final_auxiliary_loss,
        mean_loss=sum(losses) / len(losses) if losses else math.nan,
        mean_interval_loss=sum(interval_losses) / len(interval_losses)
        if interval_losses
        else math.nan,
        runtime_s=time.perf_counter() - start,
    )


@torch.inference_mode()
def evaluate_retention_agreement(
    *,
    args: argparse.Namespace,
    model: RetentionDistillNet,
    device: torch.device,
    oracle: FSRS6IntervalOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None,
) -> dict[str, float]:
    dtype = torch.float64
    weight_count = len(cost_weights)
    env_count = args.eval_particles * weight_count
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=dtype,
        seed=args.seed + 70_000,
        exact_memory=True,
        goal_norm_max=max(cost_weights),
        obs_mode=args.obs_mode,
        **fsrs_config_kwargs(fsrs_config),
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
    total_abs_log_interval = 0.0
    total_abs_retention = 0.0
    total_abs_logit = 0.0
    total_interval_agree = 0.0
    total_interval_abs = 0.0
    start_time = time.perf_counter()
    model.eval()
    while not bool(env.done.all().item()):
        active = (~env.done).nonzero(as_tuple=False).squeeze(1)
        interval_labels = interval_oracle_labels(
            env=env,
            oracle=oracle,
            policies=policy_tables,
            cost_weights=cost_weight_tensor,
        )
        target_retention = target_retentions_for_intervals(
            env=env,
            intervals=interval_labels,
            retention_min=args.retention_min,
            retention_max=args.retention_max,
        )
        target_logit = retention_logits_for_retentions(
            target_retention,
            retention_min=args.retention_min,
            retention_max=args.retention_max,
        )
        pred_logit, _ = model(env.obs().to(dtype=model_dtype))
        pred_retention = predicted_retentions(
            pred_logit.to(dtype=dtype),
            retention_min=args.retention_min,
            retention_max=args.retention_max,
        )
        pred_continuous_interval = continuous_intervals_for_retentions(
            env=env,
            s=env.s,
            retention=pred_retention,
        )
        pred_intervals = rounded_intervals_for_retentions(
            env=env,
            retention=pred_retention,
            terminal_snap_ratio=args.terminal_snap_ratio,
        )
        active_pred_logit = pred_logit.to(dtype=dtype).index_select(0, active)
        active_target_logit = target_logit.index_select(0, active)
        active_pred_retention = pred_retention.index_select(0, active)
        active_target_retention = target_retention.index_select(0, active)
        active_pred_log_interval = torch.log(
            pred_continuous_interval.index_select(0, active)
        )
        active_target_log_interval = torch.log(
            interval_labels.to(dtype=dtype).index_select(0, active)
        )
        active_pred_interval = pred_intervals.index_select(0, active)
        active_label_interval = interval_labels.index_select(0, active)
        count = float(active.numel())
        total_count += count
        total_loss += nn.functional.smooth_l1_loss(
            active_pred_logit,
            active_target_logit,
            reduction="sum",
        ).item()
        total_abs_logit += (
            torch.abs(active_pred_logit - active_target_logit).sum().item()
        )
        total_abs_retention += (
            torch.abs(active_pred_retention - active_target_retention).sum().item()
        )
        total_abs_log_interval += (
            torch.abs(active_pred_log_interval - active_target_log_interval)
            .sum()
            .item()
        )
        total_interval_abs += (
            torch.abs(active_pred_interval - active_label_interval)
            .to(dtype=dtype)
            .sum()
            .item()
        )
        total_interval_agree += (
            (active_pred_interval == active_label_interval).to(dtype=dtype).sum().item()
        )
        env.step_intervals(interval_labels)
    denom = max(1.0, total_count)
    return {
        "eval_retention_smooth_l1_loss": total_loss / denom,
        "eval_retention_logit_mae": total_abs_logit / denom,
        "eval_retention_mae": total_abs_retention / denom,
        "eval_log_interval_mae": total_abs_log_interval / denom,
        "eval_interval_mae_days": total_interval_abs / denom,
        "eval_rounded_interval_agreement": total_interval_agree / denom,
        "eval_runtime_s": time.perf_counter() - start_time,
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
        intervals = rounded_intervals_for_retentions(
            env=env,
            retention=retention,
            terminal_snap_ratio=args.terminal_snap_ratio,
        )
        env.step_intervals(intervals)
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
        "scheduler": "fsrs6_oracle_retention_distill",
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
        "train_final_interval_loss": train_stats.final_interval_loss,
        "train_final_retention_loss": train_stats.final_retention_loss,
        "train_final_auxiliary_loss": train_stats.final_auxiliary_loss,
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
        "train_final_auxiliary_loss",
        "train_mean_loss",
        "train_mean_interval_loss",
        "train_runtime_s",
        "eval_retention_smooth_l1_loss",
        "eval_retention_logit_mae",
        "eval_retention_mae",
        "eval_log_interval_mae",
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
    model: RetentionDistillNet,
    args: argparse.Namespace,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    train_stats: DistillTrainStats,
    eval_stats: dict[str, float],
    oracle_solve_runtime_s: float,
    fsrs_config: SingleCardFSRS6Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": "fsrs6_oracle_retention_distill",
            "action_mode": "desired_retention",
            "obs_mode": args.obs_mode,
            "obs_dim": model.obs_dim,
            "cost_weights": list(cost_weights),
            "action_retentions": list(action_retentions),
            "days": args.days,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "retention_min": args.retention_min,
            "retention_max": args.retention_max,
            "interval_loss_weight": args.interval_loss_weight,
            "retention_logit_loss_weight": args.retention_logit_loss_weight,
            "auxiliary_action_loss_weight": args.auxiliary_action_loss_weight,
            "underprediction_loss_weight": args.underprediction_loss_weight,
            "terminal_underprediction_loss_weight": (
                args.terminal_underprediction_loss_weight
            ),
            "student_rollout_prob": args.student_rollout_prob,
            "student_rollout_warmup_epochs": args.student_rollout_warmup_epochs,
            "terminal_snap_ratio": args.terminal_snap_ratio,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_interval_chunk_size": args.oracle_interval_chunk_size,
            **fsrs_config.checkpoint_payload(),
            "train_epochs": train_stats.epochs,
            "train_steps_per_epoch": train_stats.steps_per_epoch,
            "train_transitions": train_stats.transitions,
            "train_final_loss": train_stats.final_loss,
            "train_final_interval_loss": train_stats.final_interval_loss,
            "train_final_retention_loss": train_stats.final_retention_loss,
            "train_final_auxiliary_loss": train_stats.final_auxiliary_loss,
            "train_mean_loss": train_stats.mean_loss,
            "train_mean_interval_loss": train_stats.mean_interval_loss,
            "train_runtime_s": train_stats.runtime_s,
            "oracle_solve_runtime_s": oracle_solve_runtime_s,
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
    if not MIN_TARGET_RETENTION <= args.retention_min < args.retention_max < 1.0:
        raise SystemExit("--retention-min and --retention-max must be within [0.5, 1).")
    if args.interval_loss_weight < 0.0:
        raise SystemExit("--interval-loss-weight must be >= 0.")
    if args.retention_logit_loss_weight < 0.0:
        raise SystemExit("--retention-logit-loss-weight must be >= 0.")
    if args.auxiliary_action_loss_weight < 0.0:
        raise SystemExit("--auxiliary-action-loss-weight must be >= 0.")
    if args.underprediction_loss_weight < 0.0:
        raise SystemExit("--underprediction-loss-weight must be >= 0.")
    if args.terminal_underprediction_loss_weight < 0.0:
        raise SystemExit("--terminal-underprediction-loss-weight must be >= 0.")
    if not 0.0 <= args.student_rollout_prob <= 1.0:
        raise SystemExit("--student-rollout-prob must be within [0, 1].")
    if args.student_rollout_warmup_epochs < 0:
        raise SystemExit("--student-rollout-warmup-epochs must be >= 0.")
    if args.terminal_snap_ratio < 0.0:
        raise SystemExit("--terminal-snap-ratio must be >= 0.")
    if (
        args.interval_loss_weight == 0.0
        and args.retention_logit_loss_weight == 0.0
        and args.auxiliary_action_loss_weight == 0.0
    ):
        raise SystemExit("At least one loss weight must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")

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
    oracle = FSRS6IntervalOracle(
        days=args.days,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        interval_chunk_size=args.oracle_interval_chunk_size,
        device=device,
        cache_config=cache_config,
        **fsrs_config_kwargs(fsrs_config),
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
        action_retentions=action_retentions,
        fsrs_config=fsrs_config,
    )
    eval_stats = evaluate_retention_agreement(
        args=args,
        model=model,
        device=device,
        oracle=oracle,
        policies=policies,
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
        eval_stats=eval_stats,
        oracle_solve_runtime_s=oracle_solve_runtime_s,
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
            seed=args.seed + 80_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=max(cost_weights),
            fsrs_config=fsrs_config,
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
        f"Retention oracle solve runtime_s={oracle_solve_runtime_s:.2f} device={device}"
    )
    print(
        " ".join(
            [
                f"train_final_loss={train_stats.final_loss:.6f}",
                f"interval_loss={train_stats.final_interval_loss:.6f}",
                f"retention_mae={eval_stats['eval_retention_mae']:.6f}",
                f"log_interval_mae={eval_stats['eval_log_interval_mae']:.6f}",
                "rounded_interval_agreement="
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
