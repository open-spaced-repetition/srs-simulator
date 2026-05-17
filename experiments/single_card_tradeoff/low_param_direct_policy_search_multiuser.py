from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
from collections.abc import Sequence
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    BASELINE_SCHEDULER,
    DEFAULT_USER_IDS,
    PER_USER_SCHEDULER,
    BatchedPolicyEnsemble,
    MultiUserFSRS6SingleCardBatch,
    _batched_eval_layout,
    _eval_group_chunks,
    evaluate_batched_per_user_policies,
    evaluate_static_retentions_by_user,
    load_user_configs,
    metric_row,
    parse_user_ids,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.tradeoff import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
    _build_regret_auc_rows,
    _write_csv,
    _write_regret_auc_csv,
)
from experiments.single_card_tradeoff.uvfa_ppo import (  # noqa: E402
    PolicyValueNet,
    parse_csv_floats,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402

DIRECT_POLICY_SCHEDULER = "fsrs6_low_param_direct_policy_search"
DEFAULT_DISTILL_DIR = Path(
    "artifacts/single_card_tradeoff/"
    "stationary_finite_distill_first8_users_per_user_uniform_table_supervision_"
    "fsrs6_baseline_gpu"
)
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/low_param_direct_policy_search_first8_users"
)
PARAMETER_NAMES = (
    "bias",
    "stability",
    "difficulty",
    "stability_difficulty",
    "cost_bias",
    "cost_stability",
    "cost_difficulty",
)
PARAMETER_COUNT = len(PARAMETER_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize tiny stationary FSRS-6 desired-retention policies with "
            "direct evolutionary search."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.set_defaults(env="fsrs6")
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
        help="Comma-separated benchmark user IDs to optimize in one batch.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--train-cost-weights",
        default=",".join(
            format_float(value)
            for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights used by direct search.",
    )
    parser.add_argument(
        "--eval-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights used for final evaluation.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help=(
            "Static-retention baseline action grid. The distill checkpoint action "
            "grid is used when --distill-dir is provided."
        ),
    )
    parser.add_argument("--min-retention", type=float, default=0.5)
    parser.add_argument("--max-retention", type=float, default=0.98)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=64)
    parser.add_argument("--train-particles", type=int, default=64)
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument("--initial-std", type=float, default=1.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--theta-clip", type=float, default=8.0)
    parser.add_argument(
        "--train-exact-memory",
        action="store_true",
        help="Use exact daily memorized sums during CEM training rollouts.",
    )
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
        "--distill-dir",
        type=Path,
        default=DEFAULT_DISTILL_DIR,
        help=(
            "Directory containing per-user stationary finite distill checkpoints "
            "for the distill baseline."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.env != "fsrs6":
        raise SystemExit("low-parameter direct policy search requires --env fsrs6.")
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.population_size < 2:
        raise SystemExit("--population-size must be >= 2.")
    if args.elite_count <= 0 or args.elite_count > args.population_size:
        raise SystemExit("--elite-count must be in [1, population-size].")
    if args.generations <= 0:
        raise SystemExit("--generations must be > 0.")
    if args.train_particles <= 0:
        raise SystemExit("--train-particles must be > 0.")
    if args.eval_particles <= 0:
        raise SystemExit("--eval-particles must be > 0.")
    if args.eval_group_batch_size < 0:
        raise SystemExit("--eval-group-batch-size must be >= 0.")
    if not (0.0 < args.min_retention < args.max_retention < 1.0):
        raise SystemExit(
            "--min-retention and --max-retention must satisfy 0 < min < max < 1."
        )
    if args.initial_std <= 0.0:
        raise SystemExit("--initial-std must be > 0.")
    if args.min_std <= 0.0:
        raise SystemExit("--min-std must be > 0.")
    if args.max_std < args.min_std:
        raise SystemExit("--max-std must be >= --min-std.")
    if not (0.0 < args.cem_alpha <= 1.0):
        raise SystemExit("--cem-alpha must be in (0, 1].")
    if args.theta_clip <= 0.0:
        raise SystemExit("--theta-clip must be > 0.")


def initial_theta(
    *,
    user_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    theta = torch.zeros((user_count, PARAMETER_COUNT), device=device, dtype=dtype)
    theta[:, 0] = 2.0
    theta[:, 1] = 0.4
    theta[:, 2] = -0.4
    theta[:, 4] = 2.0
    return theta


def low_param_retention(
    theta: torch.Tensor,
    obs: torch.Tensor,
    *,
    min_retention: float,
    max_retention: float,
) -> torch.Tensor:
    obs = obs.to(dtype=theta.dtype)
    s_norm = obs[:, 0]
    d_norm = obs[:, 1]
    w_norm = obs[:, 2]
    (
        bias,
        stability,
        difficulty,
        interaction,
        cost_bias,
        cost_stability,
        cost_difficulty,
    ) = theta.unbind(dim=1)
    base_logit = (
        bias + stability * s_norm + difficulty * d_norm + interaction * s_norm * d_norm
    )
    cost_slope = torch.nn.functional.softplus(
        cost_bias + cost_stability * s_norm + cost_difficulty * d_norm
    )
    logit = base_logit - cost_slope * w_norm
    span = max_retention - min_retention
    return min_retention + span * torch.sigmoid(logit)


def _candidate_eval_layout(
    *,
    user_count: int,
    candidate_count: int,
    weight_count: int,
    particles_per_group: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    user_indices: list[int] = []
    candidate_indices: list[int] = []
    weight_indices: list[int] = []
    group_indices: list[int] = []
    for user_idx in range(user_count):
        for candidate_idx in range(candidate_count):
            for weight_idx in range(weight_count):
                group_idx = (user_idx * candidate_count + candidate_idx) * weight_count
                group_idx += weight_idx
                user_indices.extend([user_idx] * particles_per_group)
                candidate_indices.extend([candidate_idx] * particles_per_group)
                weight_indices.extend([weight_idx] * particles_per_group)
                group_indices.extend([group_idx] * particles_per_group)
    return (
        user_indices,
        torch.tensor(candidate_indices, device=device, dtype=torch.int64),
        torch.tensor(weight_indices, device=device, dtype=torch.int64),
        torch.tensor(group_indices, device=device, dtype=torch.int64),
    )


@torch.inference_mode()
def evaluate_candidate_objectives(
    args: argparse.Namespace,
    *,
    theta: torch.Tensor,
    cost_weights: Sequence[float],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    user_count, candidate_count, _ = theta.shape
    weight_count = len(cost_weights)
    user_indices, candidate_idx, weight_idx, group_index = _candidate_eval_layout(
        user_count=user_count,
        candidate_count=candidate_count,
        weight_count=weight_count,
        particles_per_group=args.train_particles,
        device=device,
    )
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=[args.min_retention, args.max_retention],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=args.train_exact_memory,
        goal_norm_max=max(cost_weights),
        reset_on_init=False,
    )
    goal_values = torch.tensor(
        cost_weights,
        device=device,
        dtype=torch.float64,
    ).index_select(0, weight_idx)
    env.reset_all(goal_values=goal_values)
    while not bool(env.done.all().item()):
        selected_theta = theta[env.user_index, candidate_idx].to(dtype=torch.float64)
        retention = low_param_retention(
            selected_theta,
            env.obs(),
            min_retention=args.min_retention,
            max_retention=args.max_retention,
        )
        env.step_retention(retention)

    scalar_return = env.total_memorized - env.goal_weight * (
        env.total_cost_seconds / 60.0
    )
    objective = torch.bincount(
        group_index,
        weights=scalar_return.to(dtype=torch.float64),
        minlength=user_count * candidate_count * weight_count,
    ).reshape(user_count, candidate_count, weight_count)
    objective = objective / float(args.days * args.train_particles)
    return objective.mean(dim=2)


def sample_population(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    population_size: int,
    theta_clip: float,
    generator: torch.Generator,
) -> torch.Tensor:
    user_count, parameter_count = mean.shape
    noise = torch.randn(
        (user_count, population_size - 1, parameter_count),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    sampled = mean[:, None, :] + noise * std[:, None, :]
    candidates = torch.cat([mean[:, None, :], sampled], dim=1)
    return torch.clamp(candidates, min=-theta_clip, max=theta_clip)


def optimize_direct_policy(
    args: argparse.Namespace,
    *,
    configs: Sequence[SingleCardFSRS6Config],
    train_cost_weights: Sequence[float],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    user_count = len(configs)
    mean = initial_theta(
        user_count=user_count,
        device=device,
        dtype=torch.float64,
    )
    std = torch.full_like(mean, float(args.initial_std))
    best_theta = mean.clone()
    best_score = torch.full(
        (user_count,), -math.inf, device=device, dtype=torch.float64
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 120_000)
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for generation in range(1, args.generations + 1):
        candidates = sample_population(
            mean=mean,
            std=std,
            population_size=args.population_size,
            theta_clip=args.theta_clip,
            generator=generator,
        )
        scores = evaluate_candidate_objectives(
            args,
            theta=candidates,
            cost_weights=train_cost_weights,
            configs=configs,
            device=device,
            seed=args.seed + 130_000 + generation,
        )
        elite_scores, elite_idx = torch.topk(scores, k=args.elite_count, dim=1)
        gather_idx = elite_idx[:, :, None].expand(-1, -1, PARAMETER_COUNT)
        elite_theta = candidates.gather(1, gather_idx)
        generation_best_score, generation_best_idx = torch.max(scores, dim=1)
        improved = generation_best_score > best_score
        if bool(improved.any().item()):
            best_score = torch.where(improved, generation_best_score, best_score)
            best_theta[improved] = candidates[
                improved,
                generation_best_idx[improved],
            ]

        elite_mean = elite_theta.mean(dim=1)
        elite_std = torch.clamp(
            elite_theta.std(dim=1, unbiased=False),
            min=args.min_std,
            max=args.max_std,
        )
        mean = (1.0 - args.cem_alpha) * mean + args.cem_alpha * elite_mean
        std = (1.0 - args.cem_alpha) * std + args.cem_alpha * elite_std
        std = torch.clamp(std, min=args.min_std, max=args.max_std)

        mean_score = scores.mean(dim=1)
        for user_idx, config in enumerate(configs):
            history.append(
                {
                    "generation": generation,
                    "user_id": config.user_id,
                    "best_objective": float(best_score[user_idx].item()),
                    "generation_best_objective": float(
                        generation_best_score[user_idx].item()
                    ),
                    "elite_mean_objective": float(elite_scores[user_idx].mean().item()),
                    "population_mean_objective": float(mean_score[user_idx].item()),
                    "mean_std": float(std[user_idx].mean().item()),
                }
            )
        if not args.no_progress:
            print(
                f"generation={generation}/{args.generations} "
                f"mean_best_objective={best_score.mean().item():.6f} "
                f"mean_std={std.mean().item():.4f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    return best_theta, history, time.perf_counter() - start


@torch.inference_mode()
def evaluate_direct_policy_by_user(
    args: argparse.Namespace,
    *,
    theta: torch.Tensor,
    cost_weights: Sequence[float],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
) -> list[tuple[float, list[Any], float]]:
    user_count = len(configs)
    results: list[tuple[float, list[Any], float] | None] = [None for _ in cost_weights]
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
            action_retentions=[args.min_retention, args.max_retention],
            device=device,
            dtype=torch.float64,
            seed=args.seed + 150_000 + int(round(float(batch_weights[0]) * 10.0)),
            exact_memory=True,
            goal_norm_max=max(cost_weights),
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
            selected_theta = theta[env.user_index].to(dtype=torch.float64)
            retention = low_param_retention(
                selected_theta,
                env.obs(),
                min_retention=args.min_retention,
                max_retention=args.max_retention,
            )
            env.step_retention(retention)
        if device.type == "cuda":
            torch.cuda.synchronize()
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


def load_distill_ensemble(
    *,
    distill_dir: Path,
    user_ids: Sequence[int],
    device: torch.device,
) -> tuple[BatchedPolicyEnsemble, list[float], list[float]]:
    models: list[PolicyValueNet] = []
    first_checkpoint: dict[str, Any] | None = None
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
        if first_checkpoint is None:
            first_checkpoint = checkpoint
        model = PolicyValueNet(
            int(checkpoint["obs_dim"]),
            len(checkpoint["action_retentions"]),
            int(checkpoint["hidden_size"]),
            architecture=str(checkpoint["network"]),
            depth=int(checkpoint["network_depth"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model.to(device).eval())
    if first_checkpoint is None:
        raise ValueError("user_ids must contain at least one user.")
    params, buffers = torch.func.stack_module_state(models)
    base_model = models[0]
    base_model.requires_grad_(False)
    ensemble = BatchedPolicyEnsemble(
        base_model=base_model,
        params=params,
        buffers=buffers,
        params_per_user=int(first_checkpoint["params_per_user"]),
    )
    action_retentions = [
        float(value) for value in first_checkpoint["action_retentions"]
    ]
    cost_weights = [float(value) for value in first_checkpoint["cost_weights"]]
    return ensemble, action_retentions, cost_weights


def write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "generation",
            "user_id",
            "best_objective",
            "generation_best_objective",
            "elite_mean_objective",
            "population_mean_objective",
            "mean_std",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_auc_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    baselines: set[str],
    scheduler: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["baseline_scheduler"] in baselines and row["scheduler"] == scheduler
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "environment",
            "baseline_scheduler",
            "scheduler",
            "span_coverage_percent",
            "time_regret_auc",
            "baseline_time_auc",
            "relative_regret_auc_percent",
            "covered_target_count",
            "target_count",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row[field] for field in fieldnames})


def write_mean_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    baselines: Sequence[str],
    scheduler: str,
) -> list[dict[str, Any]]:
    mean_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        selected = [
            row
            for row in rows
            if row["baseline_scheduler"] == baseline and row["scheduler"] == scheduler
        ]
        if not selected:
            continue
        mean_rows.append(
            {
                "baseline_scheduler": baseline,
                "scheduler": scheduler,
                "user_count": len(selected),
                "mean_span_coverage_percent": sum(
                    float(row["span_coverage_percent"]) for row in selected
                )
                / float(len(selected)),
                "mean_time_regret_auc": sum(
                    float(row["time_regret_auc"]) for row in selected
                )
                / float(len(selected)),
                "mean_relative_regret_auc_percent": sum(
                    float(row["relative_regret_auc_percent"]) for row in selected
                )
                / float(len(selected)),
                "covered_target_count_sum": sum(
                    int(row["covered_target_count"]) for row in selected
                ),
                "target_count_sum": sum(int(row["target_count"]) for row in selected),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "baseline_scheduler",
            "scheduler",
            "user_count",
            "mean_span_coverage_percent",
            "mean_time_regret_auc",
            "mean_relative_regret_auc_percent",
            "covered_target_count_sum",
            "target_count_sum",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in mean_rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return mean_rows


def save_policy(
    path: Path,
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    configs: Sequence[SingleCardFSRS6Config],
    theta: torch.Tensor,
    train_cost_weights: Sequence[float],
    eval_cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    train_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": DIRECT_POLICY_SCHEDULER,
            "policy_family": "bilinear_monotone",
            "parameter_names": list(PARAMETER_NAMES),
            "params_per_user": PARAMETER_COUNT,
            "ensemble_trainable_params": PARAMETER_COUNT * len(user_ids),
            "theta_by_user": theta.detach().cpu(),
            "user_ids": list(user_ids),
            "user_configs": [config.checkpoint_payload() for config in configs],
            "days": args.days,
            "train_cost_weights": list(train_cost_weights),
            "eval_cost_weights": list(eval_cost_weights),
            "action_retentions": list(action_retentions),
            "min_retention": args.min_retention,
            "max_retention": args.max_retention,
            "generations": args.generations,
            "population_size": args.population_size,
            "elite_count": args.elite_count,
            "train_particles": args.train_particles,
            "train_exact_memory": args.train_exact_memory,
            "train_runtime_s": train_runtime_s,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    user_ids = parse_user_ids(args.user_ids)
    device = resolve_torch_device(args.torch_device)
    train_cost_weights = parse_csv_floats(
        args.train_cost_weights,
        name="--train-cost-weights",
    )
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

    distill_ensemble, distill_action_retentions, distill_train_cost_weights = (
        load_distill_ensemble(
            distill_dir=args.distill_dir,
            user_ids=user_ids,
            device=device,
        )
    )
    baseline_retentions = distill_action_retentions or action_retentions

    theta, history, train_runtime_s = optimize_direct_policy(
        args,
        configs=configs,
        train_cost_weights=train_cost_weights,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    for retention, metrics_by_user, runtime_s in evaluate_static_retentions_by_user(
        args,
        retentions=baseline_retentions,
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

    for cost_weight, metrics_by_user, runtime_s in evaluate_batched_per_user_policies(
        args,
        ensemble=distill_ensemble,
        cost_weights=eval_cost_weights,
        action_retentions=distill_action_retentions,
        goal_norm_max=max(distill_train_cost_weights),
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

    for cost_weight, metrics_by_user, runtime_s in evaluate_direct_policy_by_user(
        args,
        theta=theta,
        cost_weights=eval_cost_weights,
        configs=configs,
        device=device,
    ):
        for user_id, metrics in zip(user_ids, metrics_by_user, strict=True):
            rows.append(
                metric_row(
                    args,
                    user_id=user_id,
                    scheduler=DIRECT_POLICY_SCHEDULER,
                    scheduler_spec=DIRECT_POLICY_SCHEDULER,
                    desired_retention=None,
                    goal_cost_weight=cost_weight,
                    metrics=metrics,
                    runtime_s=runtime_s,
                    engine="low_param_cem_direct_search",
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
    history_path = args.out_dir / "train_history.csv"
    policy_path = args.out_dir / "policy.pt"
    metadata_path = args.out_dir / "metadata.json"

    _write_csv(results_path, rows)
    auc_rows = _build_regret_auc_rows(rows)
    _write_regret_auc_csv(regret_path, auc_rows)
    write_auc_summary(
        summary_path,
        auc_rows,
        baselines={BASELINE_SCHEDULER, PER_USER_SCHEDULER},
        scheduler=DIRECT_POLICY_SCHEDULER,
    )
    mean_rows = write_mean_summary(
        mean_summary_path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER, PER_USER_SCHEDULER],
        scheduler=DIRECT_POLICY_SCHEDULER,
    )
    write_history(history_path, history)
    save_policy(
        policy_path,
        args=args,
        user_ids=user_ids,
        configs=configs,
        theta=theta,
        train_cost_weights=train_cost_weights,
        eval_cost_weights=eval_cost_weights,
        action_retentions=baseline_retentions,
        train_runtime_s=train_runtime_s,
    )
    metadata = {
        "scheduler": DIRECT_POLICY_SCHEDULER,
        "policy_family": "bilinear_monotone",
        "params_per_user": PARAMETER_COUNT,
        "ensemble_trainable_params": PARAMETER_COUNT * len(user_ids),
        "device": str(device),
        "user_ids": list(user_ids),
        "days": args.days,
        "train_cost_weights": list(train_cost_weights),
        "eval_cost_weights": list(eval_cost_weights),
        "baseline_action_retentions": list(baseline_retentions),
        "distill_dir": str(args.distill_dir),
        "train_runtime_s": train_runtime_s,
        "eval_runtime_s": eval_runtime_s,
        "total_runtime_s": train_runtime_s + eval_runtime_s,
        "mean_summary": mean_rows,
        "theta_by_user": {
            str(user_id): {
                name: float(value)
                for name, value in zip(
                    PARAMETER_NAMES,
                    theta[user_idx].detach().cpu().tolist(),
                    strict=True,
                )
            }
            for user_idx, user_id in enumerate(user_ids)
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote policy: {policy_path}")
    print(f"Wrote CSV: {results_path}")
    print(f"Wrote regret AUC CSV: {regret_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote mean summary CSV: {mean_summary_path}")
    print(f"Wrote train history CSV: {history_path}")
    print(
        "Low-parameter direct policy search: "
        f"users={','.join(str(user_id) for user_id in user_ids)} "
        f"params_each={PARAMETER_COUNT} "
        f"ensemble_params={PARAMETER_COUNT * len(user_ids)} "
        f"device={device} train_s={train_runtime_s:.2f} eval_s={eval_runtime_s:.2f}"
    )
    for row in mean_rows:
        print(
            f"vs={row['baseline_scheduler']} "
            f"coverage={row['mean_span_coverage_percent']:.2f}% "
            f"time_regret_auc={row['mean_time_regret_auc']:.4f} "
            f"relative_regret={row['mean_relative_regret_auc_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
