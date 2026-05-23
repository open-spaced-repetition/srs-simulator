from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import json
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

from experiments.single_card_tradeoff.cli.oracle_continuous_stationary_finite_distill import (  # noqa: E402
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INTERVAL_LOSS_WEIGHT,
    DEFAULT_NETWORK_DEPTH,
    DEFAULT_RETENTION_LOGIT_LOSS_WEIGHT,
    DEFAULT_RETENTION_MAX,
    DEFAULT_RETENTION_MIN,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    POLICY_TYPE,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    DEFAULT_EVAL_PARTICLES,
    DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
    DEFAULT_USER_IDS,
    MultiUserFSRS6SingleCardBatch,
    _batched_eval_layout,
    _eval_group_chunks,
    clip_stacked_grad_norm_,
    evaluate_static_retentions_by_user,
    load_user_configs,
    metric_row,
    parse_user_ids,
)
from experiments.single_card_tradeoff.cli.uvfa_ppo import (  # noqa: E402
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_TRAIN_ENVS,
    SimMetrics,
    parse_csv_floats,
)
from experiments.single_card_tradeoff.core.auc_outputs import (  # noqa: E402
    SINGLE_SCHEDULER_AUC_FIELDS,
    write_auc_summary as write_filtered_auc_summary,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.results import (  # noqa: E402
    build_regret_auc_rows as _build_regret_auc_rows,
    write_csv as _write_csv,
    write_regret_auc_csv as _write_regret_auc_csv,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    MIN_TARGET_RETENTION,
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.models.policy_runtime import (  # noqa: E402
    RetentionDistillNet,
    predicted_retentions,
    retention_logits_for_retentions,
)
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    FSRS6BatchedContinuousStationaryFiniteOracle,
)
from experiments.single_card_tradeoff.oracles.dp_cache import (  # noqa: E402
    OracleDPCacheConfig,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED  # noqa: E402
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS  # noqa: E402
from simulator.math.fsrs import Bounds  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/"
    "continuous_stationary_finite_distill_first8_users_batched"
)
BASELINE_SCHEDULER = "fsrs6"
PER_USER_SCHEDULER = POLICY_TYPE
DEFAULT_TRAIN_ENVS_PER_USER = DEFAULT_TRAIN_ENVS


def _diagnostic_log(
    *,
    enabled: bool,
    start_s: float,
    message: str,
) -> None:
    if not enabled:
        return
    print(
        f"[continuous-distill +{time.perf_counter() - start_s:.1f}s] {message}",
        flush=True,
    )


def _progress_log_enabled(args: argparse.Namespace) -> bool:
    return float(args.progress_log_interval_seconds) > 0.0


def _next_progress_deadline(args: argparse.Namespace) -> float:
    return time.perf_counter() + float(args.progress_log_interval_seconds)


def _progress_log_due(args: argparse.Namespace, deadline: float) -> bool:
    return _progress_log_enabled(args) and time.perf_counter() >= deadline


@dataclass(frozen=True)
class ContinuousSingleUserTrainStats:
    user_id: int
    user_index: int
    params_per_user: int
    ensemble_trainable_params: int
    epochs: int
    steps_per_epoch: int
    train_samples: int
    train_transitions: int
    train_runtime_s: float
    final_loss: float
    final_interval_loss: float
    final_retention_loss: float
    eval_retention_mae: float
    eval_log_interval_mae: float
    table_samples_per_weight: int


@dataclass(frozen=True)
class BatchedRetentionPolicyEnsemble:
    base_model: RetentionDistillNet
    params: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    params_per_user: int


class BatchedContinuousStationaryFiniteOracleGuide:
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
        progress_log_interval_seconds: float,
        configs: Sequence[SingleCardFSRS6Config],
        user_batch_size: int,
        cache_config: OracleDPCacheConfig | None = None,
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
        self.days = int(days)
        self.horizon = int(days - 1)
        self.retention_min = float(retention_min)
        self.retention_max = float(retention_max)
        guide_start_s = time.perf_counter()
        log_enabled = progress_log_interval_seconds > 0.0

        chunk_size = len(configs) if user_batch_size <= 0 else user_batch_size
        policy_chunks: list[torch.Tensor] = []
        objective_chunks: list[torch.Tensor] = []
        factor_chunks: list[torch.Tensor] = []
        decay_chunks: list[torch.Tensor] = []
        metrics: list[list[Any]] = []
        iterations: list[list[int]] = []
        converged: list[list[bool]] = []
        residuals: list[list[float]] = []
        first_oracle: FSRS6BatchedContinuousStationaryFiniteOracle | None = None

        for start in range(0, len(configs), chunk_size):
            chunk_configs = configs[start : start + chunk_size]
            user_ids = [config.user_id for config in chunk_configs]
            _diagnostic_log(
                enabled=log_enabled,
                start_s=guide_start_s,
                message=(
                    "teacher chunk start "
                    f"users={user_ids} weights={list(cost_weights)} "
                    f"days={days} grid={s_grid_size}x{d_grid_size} "
                    f"chunk={interval_chunk_size}"
                ),
            )
            chunk_start_s = time.perf_counter()
            oracle = FSRS6BatchedContinuousStationaryFiniteOracle(
                days=days,
                s_grid_size=s_grid_size,
                d_grid_size=d_grid_size,
                retention_min=retention_min,
                retention_max=retention_max,
                interval_chunk_size=interval_chunk_size,
                device=device,
                cache_config=cache_config,
                progress_log_interval_seconds=progress_log_interval_seconds,
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
            if first_oracle is None:
                first_oracle = oracle
            solution = oracle.solve_stationary_finite_policies(
                cost_weights,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress=progress,
            )
            _diagnostic_log(
                enabled=log_enabled,
                start_s=guide_start_s,
                message=(
                    "teacher chunk solved "
                    f"users={user_ids} runtime_s={time.perf_counter() - chunk_start_s:.1f}"
                ),
            )
            failed = [
                f"user={chunk_configs[user_idx].user_id}:w={format_float(weight)}"
                for user_idx, row in enumerate(solution.converged)
                for weight, did_converge in zip(cost_weights, row, strict=True)
                if not did_converge
            ]
            if failed:
                raise RuntimeError(
                    "Continuous stationary finite oracle did not converge for "
                    + ",".join(failed)
                )
            policy_chunks.append(solution.policy.to(device=device, dtype=torch.float32))
            objective_chunks.append(solution.objectives.to(device=device))
            factor_chunks.append(oracle.factor.to(device=device, dtype=torch.float32))
            decay_chunks.append(oracle.decay.to(device=device, dtype=torch.float32))
            metrics.extend(solution.metrics)
            iterations.extend(solution.iterations)
            converged.extend(solution.converged)
            residuals.extend(solution.residuals)

        if first_oracle is None:
            raise ValueError("configs must contain at least one user.")
        self.policy = torch.cat(policy_chunks, dim=0)
        self.objectives = torch.cat(objective_chunks, dim=0)
        self.factor = torch.cat(factor_chunks, dim=0)
        self.decay = torch.cat(decay_chunks, dim=0)
        self.s_grid = first_oracle.s_grid.to(device=device, dtype=torch.float32)
        self.metrics = metrics
        self.iterations = iterations
        self.converged = converged
        self.residuals = residuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train continuous stationary finite FSRS-6 retention distill policies "
            "for multiple users in one process with a batched exact teacher."
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
        action="store_true",
        help=(
            "Compatibility flag with stationary finite distill; continuous "
            "multi-user distill always trains one model per user."
        ),
    )
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
        help="Retention anchors used only to size the auxiliary checkpoint head.",
    )
    parser.add_argument(
        "--train-envs-per-user",
        type=int,
        default=DEFAULT_TRAIN_ENVS_PER_USER,
        help="Kept for budget parity metadata; uniform-table training does not use it.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument(
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--network", choices=["mlp", "residual"], default=DEFAULT_NETWORK
    )
    parser.add_argument("--network-depth", type=int, default=DEFAULT_NETWORK_DEPTH)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
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
        "--oracle-teacher-user-batch-size",
        type=int,
        default=DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
        help=(
            "Number of users per batched exact teacher solve. 0 means solve all "
            "users in one single-process GPU batch."
        ),
    )
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--eval-group-batch-size",
        type=int,
        default=0,
        help=(
            "Number of cost-weight evaluation groups to roll out together. "
            "0 means batch all groups at once."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_run_monitoring_args(parser)
    parser.add_argument(
        "--progress-log-interval-seconds",
        type=float,
        default=30.0,
        help=(
            "Print continuous teacher/training diagnostic progress logs at this "
            "interval. Set to 0 to disable. This is independent of tqdm progress "
            "bars and still works with --no-progress."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_guide(
    args: argparse.Namespace,
    *,
    device: torch.device,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    cache_config: OracleDPCacheConfig | None = None,
) -> tuple[BatchedContinuousStationaryFiniteOracleGuide, float]:
    start = time.perf_counter()
    guide = BatchedContinuousStationaryFiniteOracleGuide(
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
        progress_log_interval_seconds=args.progress_log_interval_seconds,
        configs=configs,
        user_batch_size=args.oracle_teacher_user_batch_size,
        cache_config=cache_config,
    )
    return guide, time.perf_counter() - start


def sample_batched_uniform_table_batch(
    guide: BatchedContinuousStationaryFiniteOracleGuide,
    *,
    cost_weights: Sequence[float],
    samples_per_weight: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    target_retention = guide.policy.to(device=device)[
        user_idx,
        weight_idx,
        s_idx,
        d_idx,
    ].reshape(user_count, -1)
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
    s_values = guide.s_grid.index_select(0, s_idx.reshape(-1)).reshape(
        user_count,
        -1,
    )
    return obs, target_retention, s_values


def log_intervals_for_retentions(
    *,
    guide: BatchedContinuousStationaryFiniteOracleGuide,
    s: torch.Tensor,
    retention: torch.Tensor,
) -> torch.Tensor:
    retention = torch.clamp(retention, min=1e-7, max=1.0 - 1e-7)
    retention_factor = (
        torch.pow(retention, 1.0 / guide.decay[:, None].to(dtype=retention.dtype)) - 1.0
    )
    interval = (
        s
        / guide.factor[:, None].to(device=s.device, dtype=s.dtype)
        * retention_factor.to(dtype=s.dtype)
    )
    interval = torch.clamp(interval, min=1.0, max=float(guide.horizon + 1))
    return torch.log(interval)


def build_batched_retention_ensemble(
    args: argparse.Namespace,
    *,
    user_count: int,
    obs_dim: int,
    action_count: int,
    params_per_user: int,
    device: torch.device,
) -> BatchedRetentionPolicyEnsemble:
    models = []
    for user_idx in range(user_count):
        torch.manual_seed(args.seed + user_idx)
        models.append(
            RetentionDistillNet(
                obs_dim=obs_dim,
                hidden_size=args.hidden_size,
                action_count=action_count,
                architecture=args.network,
                depth=args.network_depth,
            ).to(device)
        )
    params, buffers = torch.func.stack_module_state(models)
    base_model = models[0]
    base_model.requires_grad_(False)
    return BatchedRetentionPolicyEnsemble(
        base_model=base_model,
        params=params,
        buffers=buffers,
        params_per_user=params_per_user,
    )


def batched_retention_ensemble_forward(
    ensemble: BatchedRetentionPolicyEnsemble,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def call_one(
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        single_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_retention, auxiliary = torch.func.functional_call(
            ensemble.base_model,
            (params, buffers),
            (single_obs,),
        )
        return raw_retention, auxiliary

    return torch.func.vmap(call_one, in_dims=(0, 0, 0))(
        ensemble.params,
        ensemble.buffers,
        obs,
    )


def ensemble_state_dict_for_user(
    ensemble: BatchedRetentionPolicyEnsemble,
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
    ensemble: BatchedRetentionPolicyEnsemble,
    user_idx: int,
    obs_dim: int,
    action_count: int,
) -> RetentionDistillNet:
    model = RetentionDistillNet(
        obs_dim=obs_dim,
        hidden_size=args.hidden_size,
        action_count=action_count,
        architecture=args.network,
        depth=args.network_depth,
    )
    model.load_state_dict(ensemble_state_dict_for_user(ensemble, user_idx))
    return model


def train_batched_per_user_models(
    args: argparse.Namespace,
    *,
    device: torch.device,
    user_count: int,
    guide: BatchedContinuousStationaryFiniteOracleGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    params_per_user: int,
) -> tuple[
    BatchedRetentionPolicyEnsemble, float, list[float], list[float], list[float]
]:
    ensemble = build_batched_retention_ensemble(
        args,
        user_count=user_count,
        obs_dim=3,
        action_count=len(action_retentions),
        params_per_user=params_per_user,
        device=device,
    )
    optimizer = torch.optim.Adam(
        list(ensemble.params.values()),
        lr=args.learning_rate,
        eps=1e-5,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 91_000)
    final_loss_by_user = [math.nan for _ in range(user_count)]
    final_interval_loss_by_user = [math.nan for _ in range(user_count)]
    final_retention_loss_by_user = [math.nan for _ in range(user_count)]
    start = time.perf_counter()
    next_log_s = _next_progress_deadline(args)
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=start,
        message=(
            "train start "
            f"users={user_count} epochs={args.epochs} "
            f"steps_per_epoch={args.steps_per_epoch} "
            f"samples_per_weight={args.table_samples_per_weight} "
            f"teacher_weights={len(cost_weights)}"
        ),
    )
    for epoch in range(args.epochs):
        total_by_user = torch.zeros(user_count, device=device, dtype=torch.float64)
        loss_sum_by_user = torch.zeros_like(total_by_user)
        interval_sum_by_user = torch.zeros_like(total_by_user)
        retention_sum_by_user = torch.zeros_like(total_by_user)
        for step in range(args.steps_per_epoch):
            obs, target_retention, s_values = sample_batched_uniform_table_batch(
                guide,
                cost_weights=cost_weights,
                samples_per_weight=args.table_samples_per_weight,
                device=device,
                generator=generator,
            )
            pred_logit, _ = batched_retention_ensemble_forward(ensemble, obs)
            target_logit = retention_logits_for_retentions(
                target_retention,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            pred_retention = predicted_retentions(
                pred_logit,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
            )
            pred_log_interval = log_intervals_for_retentions(
                guide=guide,
                s=s_values,
                retention=pred_retention,
            )
            target_log_interval = log_intervals_for_retentions(
                guide=guide,
                s=s_values,
                retention=target_retention,
            )
            interval_item = nn.functional.smooth_l1_loss(
                pred_log_interval,
                target_log_interval,
                reduction="none",
            )
            retention_item = nn.functional.smooth_l1_loss(
                pred_logit,
                target_logit,
                reduction="none",
            )
            interval_loss_by_user = interval_item.mean(dim=1)
            retention_loss_by_user = retention_item.mean(dim=1)
            loss_by_user = (
                float(args.interval_loss_weight) * interval_loss_by_user
                + float(args.retention_logit_loss_weight) * retention_loss_by_user
            )
            loss = loss_by_user.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_stacked_grad_norm_(ensemble.params, max_norm=args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                count = float(target_retention.shape[1])
                total_by_user += count
                loss_sum_by_user += (
                    loss_by_user.detach().to(dtype=torch.float64) * count
                )
                interval_sum_by_user += (
                    interval_loss_by_user.detach().to(dtype=torch.float64) * count
                )
                retention_sum_by_user += (
                    retention_loss_by_user.detach().to(dtype=torch.float64) * count
                )
            if _progress_log_due(args, next_log_s):
                partial_loss = loss_sum_by_user / torch.clamp(total_by_user, min=1.0)
                partial_interval = interval_sum_by_user / torch.clamp(
                    total_by_user,
                    min=1.0,
                )
                partial_retention = retention_sum_by_user / torch.clamp(
                    total_by_user,
                    min=1.0,
                )
                _diagnostic_log(
                    enabled=True,
                    start_s=start,
                    message=(
                        "train progress "
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={step + 1}/{args.steps_per_epoch} "
                        f"mean_loss={float(partial_loss.mean().item()):.6f} "
                        f"mean_interval={float(partial_interval.mean().item()):.6f} "
                        f"mean_retention={float(partial_retention.mean().item()):.6f}"
                    ),
                )
                next_log_s = _next_progress_deadline(args)
        final_loss = loss_sum_by_user / torch.clamp(total_by_user, min=1.0)
        final_interval = interval_sum_by_user / torch.clamp(total_by_user, min=1.0)
        final_retention = retention_sum_by_user / torch.clamp(total_by_user, min=1.0)
        final_loss_by_user = [float(value) for value in final_loss.tolist()]
        final_interval_loss_by_user = [
            float(value) for value in final_interval.tolist()
        ]
        final_retention_loss_by_user = [
            float(value) for value in final_retention.tolist()
        ]
        if not args.no_progress:
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"mean_loss={sum(final_loss_by_user) / float(user_count):.6f} "
                f"mean_interval={sum(final_interval_loss_by_user) / float(user_count):.6f} "
                f"mean_retention={sum(final_retention_loss_by_user) / float(user_count):.6f}",
                flush=True,
            )
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=start,
        message=(
            "train done "
            f"runtime_s={time.perf_counter() - start:.1f} "
            f"mean_loss={sum(final_loss_by_user) / float(user_count):.6f} "
            f"mean_interval={sum(final_interval_loss_by_user) / float(user_count):.6f} "
            f"mean_retention={sum(final_retention_loss_by_user) / float(user_count):.6f}"
        ),
    )
    return (
        ensemble,
        time.perf_counter() - start,
        final_loss_by_user,
        final_interval_loss_by_user,
        final_retention_loss_by_user,
    )


@torch.inference_mode()
def evaluate_batched_per_user_table_fit(
    *,
    ensemble: BatchedRetentionPolicyEnsemble,
    guide: BatchedContinuousStationaryFiniteOracleGuide,
    device: torch.device,
    cost_weights: Sequence[float],
    retention_min: float,
    retention_max: float,
) -> tuple[list[float], list[float], float]:
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
    total_count = torch.zeros(user_count, device=device, dtype=torch.float64)
    retention_abs = torch.zeros_like(total_count)
    log_interval_abs = torch.zeros_like(total_count)
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
        raw_retention, _ = batched_retention_ensemble_forward(ensemble, obs)
        pred = predicted_retentions(
            raw_retention,
            retention_min=retention_min,
            retention_max=retention_max,
        )
        target = guide.policy[:, weight_idx].reshape(user_count, -1).to(device=device)
        s_values = guide.s_grid.index_select(0, s_grid.reshape(-1))
        s_values = s_values[None, :].expand(user_count, state_count)
        total_count += float(state_count)
        retention_abs += torch.abs(pred - target).sum(dim=1).to(dtype=torch.float64)
        log_interval_abs += (
            torch.abs(
                log_intervals_for_retentions(guide=guide, s=s_values, retention=pred)
                - log_intervals_for_retentions(
                    guide=guide, s=s_values, retention=target
                )
            )
            .sum(dim=1)
            .to(dtype=torch.float64)
        )
    denom = torch.clamp(total_count, min=1.0)
    return (
        [float(value) for value in (retention_abs / denom).tolist()],
        [float(value) for value in (log_interval_abs / denom).tolist()],
        time.perf_counter() - start,
    )


@torch.inference_mode()
def evaluate_batched_per_user_policies(
    args: argparse.Namespace,
    *,
    ensemble: BatchedRetentionPolicyEnsemble,
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
            seed=args.seed + 76_000 + int(round(float(batch_weights[0]) * 10.0)),
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
            active = (~env.done).nonzero(as_tuple=False).squeeze(1)
            obs = env.obs().index_select(0, active).to(dtype=model_dtype)
            active_users = env.user_index.index_select(0, active)
            retention = torch.empty(env.env_count, device=device, dtype=env.dtype)
            for user_idx in range(user_count):
                local = (active_users == user_idx).nonzero(as_tuple=False).squeeze(1)
                if local.numel() == 0:
                    continue
                user_obs = obs.index_select(0, local)
                raw_retention, _ = torch.func.functional_call(
                    ensemble.base_model,
                    (
                        {
                            name: tensor[user_idx]
                            for name, tensor in ensemble.params.items()
                        },
                        {
                            name: tensor[user_idx]
                            for name, tensor in ensemble.buffers.items()
                        },
                    ),
                    (user_obs,),
                )
                retention[active.index_select(0, local)] = predicted_retentions(
                    raw_retention.reshape(-1).to(dtype=env.dtype),
                    retention_min=args.retention_min,
                    retention_max=args.retention_max,
                )
            env.step_retention(retention)
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


def save_single_user_checkpoint(
    path: Path,
    *,
    model: RetentionDistillNet,
    args: argparse.Namespace,
    user_id: int,
    user_idx: int,
    config: SingleCardFSRS6Config,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    guide: BatchedContinuousStationaryFiniteOracleGuide,
    stats: ContinuousSingleUserTrainStats,
    teacher_runtime_s: float,
    table_eval_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": POLICY_TYPE,
            "action_mode": "desired_retention",
            "training_scope": "per_user_batched_uniform_table_supervision",
            "obs_mode": "oracle_stationary",
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
            "teacher_policy": "continuous_stationary_finite_oracle",
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
            "user_ids": [user_id],
            "user_index": user_idx,
            "user_configs": [config.checkpoint_payload()],
            **config.checkpoint_payload(),
            "train_epochs": stats.epochs,
            "train_steps_per_epoch": stats.steps_per_epoch,
            "table_samples_per_weight": stats.table_samples_per_weight,
            "train_samples": stats.train_samples,
            "train_transitions": stats.train_transitions,
            "params_per_user": stats.params_per_user,
            "ensemble_trainable_params": stats.ensemble_trainable_params,
            "train_final_loss": stats.final_loss,
            "train_final_interval_loss": stats.final_interval_loss,
            "train_final_retention_loss": stats.final_retention_loss,
            "train_runtime_s": stats.train_runtime_s,
            "teacher_runtime_s": teacher_runtime_s,
            "table_eval_runtime_s": table_eval_runtime_s,
            "eval_retention_mae": stats.eval_retention_mae,
            "eval_log_interval_mae": stats.eval_log_interval_mae,
        },
        path,
    )


def write_train_summary(
    path: Path,
    *,
    stats_by_user: Sequence[ContinuousSingleUserTrainStats],
    setup_runtime_s: float,
    teacher_runtime_s: float,
    table_eval_runtime_s: float,
    rollout_eval_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "training_scope",
        "user_id",
        "params_per_user",
        "ensemble_trainable_params",
        "epochs",
        "steps_per_epoch",
        "table_samples_per_weight",
        "train_samples",
        "train_transitions",
        "setup_runtime_s",
        "teacher_runtime_s",
        "train_runtime_s",
        "table_eval_runtime_s",
        "rollout_eval_runtime_s",
        "total_runtime_s",
        "final_loss",
        "final_interval_loss",
        "final_retention_loss",
        "eval_retention_mae",
        "eval_log_interval_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_by_user:
            writer.writerow(
                {
                    "training_scope": "per_user_batched_uniform_table_supervision",
                    "user_id": stats.user_id,
                    "params_per_user": stats.params_per_user,
                    "ensemble_trainable_params": stats.ensemble_trainable_params,
                    "epochs": stats.epochs,
                    "steps_per_epoch": stats.steps_per_epoch,
                    "table_samples_per_weight": stats.table_samples_per_weight,
                    "train_samples": stats.train_samples,
                    "train_transitions": stats.train_transitions,
                    "setup_runtime_s": setup_runtime_s,
                    "teacher_runtime_s": teacher_runtime_s,
                    "train_runtime_s": stats.train_runtime_s,
                    "table_eval_runtime_s": table_eval_runtime_s,
                    "rollout_eval_runtime_s": rollout_eval_runtime_s,
                    "total_runtime_s": (
                        setup_runtime_s
                        + teacher_runtime_s
                        + stats.train_runtime_s
                        + table_eval_runtime_s
                        + rollout_eval_runtime_s
                    ),
                    "final_loss": stats.final_loss,
                    "final_interval_loss": stats.final_interval_loss,
                    "final_retention_loss": stats.final_retention_loss,
                    "eval_retention_mae": stats.eval_retention_mae,
                    "eval_log_interval_mae": stats.eval_log_interval_mae,
                }
            )


def write_run_config_snapshot(
    path: Path,
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    device: torch.device,
    cost_weights: Sequence[float],
    eval_cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    payload = {
        "experiment": "oracle_continuous_stationary_finite_distill_multiuser",
        "command": sys.argv,
        "device": str(device),
        "env": args.env,
        "user_ids": list(user_ids),
        "review_markov_transition": False,
        "days": args.days,
        "deck_scale": args.deck_scale,
        "seed": args.seed,
        "training_cost_weights": list(cost_weights),
        "eval_cost_weights": list(eval_cost_weights),
        "action_retentions": list(action_retentions),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "table_samples_per_weight": args.table_samples_per_weight,
        "train_envs_per_user": args.train_envs_per_user,
        "eval_particles": args.eval_particles,
        "network": args.network,
        "hidden_size": args.hidden_size,
        "network_depth": args.network_depth,
        "retention_min": args.retention_min,
        "retention_max": args.retention_max,
        "interval_loss_weight": args.interval_loss_weight,
        "retention_logit_loss_weight": args.retention_logit_loss_weight,
        "oracle_s_grid_size": args.oracle_s_grid_size,
        "oracle_d_grid_size": args.oracle_d_grid_size,
        "oracle_interval_chunk_size": args.oracle_interval_chunk_size,
        "oracle_stationary_finite_max_iterations": (
            args.oracle_stationary_finite_max_iterations
        ),
        "oracle_stationary_finite_tolerance": args.oracle_stationary_finite_tolerance,
        "oracle_teacher_user_batch_size": args.oracle_teacher_user_batch_size,
        "progress_log_interval_seconds": args.progress_log_interval_seconds,
        "button_usage": str(args.button_usage) if args.button_usage else None,
        "benchmark_partition": args.benchmark_partition,
        "srs_benchmark_root": (
            str(args.srs_benchmark_root) if args.srs_benchmark_root else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_auc_summary(path: Path, auc_rows: list[dict[str, Any]]) -> None:
    write_filtered_auc_summary(
        path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER],
        schedulers=[PER_USER_SCHEDULER],
        fieldnames=SINGLE_SCHEDULER_AUC_FIELDS,
    )


def main() -> None:
    run_start_s = time.perf_counter()
    args = parse_args()
    cache_config = configure_oracle_dp_cache_from_args(args)
    user_ids = parse_user_ids(args.user_ids)
    if args.env != "fsrs6":
        raise SystemExit(
            "multi-user continuous stationary finite distill requires --env fsrs6."
        )
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.train_envs_per_user <= 0:
        raise SystemExit("--train-envs-per-user must be > 0.")
    if args.eval_particles <= 0:
        raise SystemExit("--eval-particles must be > 0.")
    if args.eval_group_batch_size < 0:
        raise SystemExit("--eval-group-batch-size must be >= 0.")
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0.")
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch must be > 0.")
    if args.table_samples_per_weight <= 0:
        raise SystemExit("--table-samples-per-weight must be > 0.")
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
    if args.oracle_teacher_user_batch_size < 0:
        raise SystemExit("--oracle-teacher-user-batch-size must be >= 0.")
    if args.progress_log_interval_seconds < 0.0:
        raise SystemExit("--progress-log-interval-seconds must be >= 0.")

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
    if any(weight < 0.0 for weight in cost_weights + eval_cost_weights):
        raise SystemExit("--cost-weights and --eval-cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    write_run_config_snapshot(
        args.out_dir / "run_config.json",
        args=args,
        user_ids=user_ids,
        device=device,
        cost_weights=cost_weights,
        eval_cost_weights=eval_cost_weights,
        action_retentions=action_retentions,
    )
    configs = load_user_configs(args, user_ids)
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message=(
            "run configured "
            f"users={list(user_ids)} train_weights={cost_weights} "
            f"eval_weights={eval_cost_weights} device={device} "
            f"out_dir={args.out_dir}"
        ),
    )

    setup_start = time.perf_counter()
    params_probe = RetentionDistillNet(
        obs_dim=3,
        hidden_size=args.hidden_size,
        action_count=len(action_retentions),
        architecture=args.network,
        depth=args.network_depth,
    )
    params = sum(param.numel() for param in params_probe.parameters())
    setup_runtime_s = time.perf_counter() - setup_start

    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message="teacher build start",
    )
    guide, teacher_runtime_s = build_guide(
        args,
        device=device,
        configs=configs,
        cost_weights=cost_weights,
        cache_config=cache_config,
    )
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message=f"teacher build done runtime_s={teacher_runtime_s:.1f}",
    )
    (
        ensemble,
        train_runtime_s,
        final_loss_by_user,
        final_interval_by_user,
        final_retention_by_user,
    ) = train_batched_per_user_models(
        args,
        device=device,
        user_count=len(configs),
        guide=guide,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        params_per_user=params,
    )
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message="table fit eval start",
    )
    eval_retention_mae, eval_log_interval_mae, table_eval_runtime_s = (
        evaluate_batched_per_user_table_fit(
            ensemble=ensemble,
            guide=guide,
            device=device,
            cost_weights=cost_weights,
            retention_min=args.retention_min,
            retention_max=args.retention_max,
        )
    )
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message=f"table fit eval done runtime_s={table_eval_runtime_s:.1f}",
    )

    train_samples_per_user = args.table_samples_per_weight * len(cost_weights)
    ensemble_trainable_params = params * len(user_ids)
    stats_by_user: list[ContinuousSingleUserTrainStats] = []
    model_paths: list[Path] = []
    for user_idx, user_id in enumerate(user_ids):
        stats = ContinuousSingleUserTrainStats(
            user_id=user_id,
            user_index=user_idx,
            params_per_user=params,
            ensemble_trainable_params=ensemble_trainable_params,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            train_samples=train_samples_per_user,
            train_transitions=(
                args.epochs * args.steps_per_epoch * train_samples_per_user
            ),
            train_runtime_s=train_runtime_s,
            final_loss=final_loss_by_user[user_idx],
            final_interval_loss=final_interval_by_user[user_idx],
            final_retention_loss=final_retention_by_user[user_idx],
            eval_retention_mae=eval_retention_mae[user_idx],
            eval_log_interval_mae=eval_log_interval_mae[user_idx],
            table_samples_per_weight=args.table_samples_per_weight,
        )
        model = materialize_ensemble_model(
            args,
            ensemble=ensemble,
            user_idx=user_idx,
            obs_dim=3,
            action_count=len(action_retentions),
        )
        model_path = args.out_dir / f"user_{user_id}_policy.pt"
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
            stats=stats,
            teacher_runtime_s=teacher_runtime_s,
            table_eval_runtime_s=table_eval_runtime_s,
        )
        stats_by_user.append(stats)
        model_paths.append(model_path)

    rows: list[dict[str, Any]] = []
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message="rollout eval start",
    )
    rollout_eval_start = time.perf_counter()
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
                    engine="per_user_batched_uniform_table_supervision",
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    rollout_eval_runtime_s = time.perf_counter() - rollout_eval_start
    _diagnostic_log(
        enabled=_progress_log_enabled(args),
        start_s=run_start_s,
        message=f"rollout eval done runtime_s={rollout_eval_runtime_s:.1f}",
    )

    results_path = args.out_dir / "results.csv"
    regret_path = args.out_dir / "regret_auc.csv"
    summary_path = args.out_dir / "summary.csv"
    train_summary_path = args.out_dir / "train_summary.csv"
    _write_csv(results_path, rows)
    auc_rows = _build_regret_auc_rows(rows)
    _write_regret_auc_csv(regret_path, auc_rows)
    write_auc_summary(summary_path, auc_rows)
    write_train_summary(
        train_summary_path,
        stats_by_user=stats_by_user,
        setup_runtime_s=setup_runtime_s,
        teacher_runtime_s=teacher_runtime_s,
        table_eval_runtime_s=table_eval_runtime_s,
        rollout_eval_runtime_s=rollout_eval_runtime_s,
    )

    mean_loss = sum(stats.final_loss for stats in stats_by_user) / float(
        len(stats_by_user)
    )
    mean_interval = sum(stats.final_interval_loss for stats in stats_by_user) / float(
        len(stats_by_user)
    )
    mean_retention = sum(stats.final_retention_loss for stats in stats_by_user) / float(
        len(stats_by_user)
    )
    print("Wrote per-user models: " + ", ".join(str(path) for path in model_paths))
    print(f"Wrote CSV: {results_path}")
    print(f"Wrote same-target time saved AUC CSV: {regret_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote train summary CSV: {train_summary_path}")
    print(
        "Per-user continuous stationary finite distill: "
        f"users={','.join(str(user_id) for user_id in user_ids)} "
        f"params_each={params} ensemble_params={ensemble_trainable_params} "
        f"device={device} teacher_s={teacher_runtime_s:.2f} "
        f"train_s={train_runtime_s:.2f} table_eval_s={table_eval_runtime_s:.2f} "
        f"rollout_eval_s={rollout_eval_runtime_s:.2f} "
        f"mean_final_loss={mean_loss:.6f} "
        f"mean_interval={mean_interval:.6f} "
        f"mean_retention={mean_retention:.6f}"
    )


if __name__ == "__main__":
    main()
