from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.rl_scheduler.policy_search_common import (  # noqa: E402
    TrainingProgress,
    _git_commit,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    MultiUserFSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    SUPPORTED_SINGLE_CARD_ENVS,
    SingleCardFSRS6Config,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    MIN_TARGET_RETENTION,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.core.types import SimMetrics  # noqa: E402
from simulator.batched_engine.multiuser_types import (  # noqa: E402
    MultiUserBehavior,
    MultiUserCost,
)
from simulator.behavior import StochasticBehavior  # noqa: E402
from simulator.button_usage import load_button_usage_config, normalize_button_usage  # noqa: E402
from simulator.cost import StateRatingCosts, StatefulCostModel  # noqa: E402
from simulator.core import new_first_priority  # noqa: E402
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402
from simulator.fsrs6_adr_policy import (  # noqa: E402
    FEATURE_VERSION_LOG_LINEAR,
    FEATURE_VERSION_LOG_POLY,
    FEATURE_VERSION_LOG_POLY_TIME,
    FSRS6ADRPolicy,
    feature_count,
)
from simulator.math.fsrs import Bounds  # noqa: E402
from simulator.scheduler_catalog import fsrs6_adr_variant_for_feature_version  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402
from simulator.batched_sweep.fsrs6_adr_policy import format_float_token  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/fsrs6_adr_single_card_direct_multiuser"
)
DEFAULT_COST_WEIGHTS = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
DEFAULT_FEATURE_VERSION = FEATURE_VERSION_LOG_POLY
DEFAULT_INITIAL_DESIRED_RETENTION = 0.90
DEFAULT_POPULATION_SIZE = 32
DEFAULT_ELITE_COUNT = 8
DEFAULT_GENERATIONS = 64
DEFAULT_TRAIN_PARTICLES = 64
DEFAULT_EVAL_PARTICLES = 10_000
DEFAULT_INITIAL_STD = 1.5
DEFAULT_MIN_STD = 0.05
DEFAULT_MAX_STD = 3.0
DEFAULT_CEM_ALPHA = 0.7
DEFAULT_THETA_CLIP = 8.0


@dataclass(frozen=True, slots=True)
class ADRTrainJob:
    user_id: int
    lambda_value: float
    config_index: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ADRTrainResult:
    job: ADRTrainJob
    baseline_train_metrics: SimMetrics
    baseline_eval_metrics: SimMetrics
    train_best_metrics: SimMetrics
    eval_best_metrics: SimMetrics
    baseline_train_objective: float
    baseline_eval_objective: float
    train_best_objective: float
    eval_best_objective: float
    best_coefficients: torch.Tensor
    history: list[dict[str, float]]
    train_runtime_s: float
    eval_runtime_s: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one independent FSRS6 ADR policy per (user, cost weight) "
            "with direct CEM search on the single-card tradeoff."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--env",
        choices=sorted(SUPPORTED_SINGLE_CARD_ENVS),
        default="fsrs6_default",
    )
    parser.add_argument(
        "--user-ids",
        default="1",
        help="Comma-separated user IDs to train in one process.",
    )
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help="Comma-separated scalarization weights; one policy is trained per weight.",
    )
    parser.add_argument(
        "--feature-version",
        choices=[
            FEATURE_VERSION_LOG_POLY,
            FEATURE_VERSION_LOG_LINEAR,
            FEATURE_VERSION_LOG_POLY_TIME,
        ],
        default=DEFAULT_FEATURE_VERSION,
    )
    parser.add_argument(
        "--initial-desired-retention",
        type=float,
        default=DEFAULT_INITIAL_DESIRED_RETENTION,
        help="Constant-retention initialization used for the first population mean.",
    )
    parser.add_argument(
        "--retention-min",
        type=float,
        default=MIN_TARGET_RETENTION,
    )
    parser.add_argument("--retention-max", type=float, default=0.98)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--train-particles", type=int, default=DEFAULT_TRAIN_PARTICLES)
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument("--initial-std", type=float, default=DEFAULT_INITIAL_STD)
    parser.add_argument("--min-std", type=float, default=DEFAULT_MIN_STD)
    parser.add_argument("--max-std", type=float, default=DEFAULT_MAX_STD)
    parser.add_argument("--cem-alpha", type=float, default=DEFAULT_CEM_ALPHA)
    parser.add_argument("--theta-clip", type=float, default=DEFAULT_THETA_CLIP)
    parser.add_argument(
        "--job-batch-size",
        type=int,
        default=0,
        help=(
            "Number of (user, cost-weight) jobs to evaluate together. 0 means "
            "all jobs in one GPU batch."
        ),
    )
    parser.add_argument(
        "--scheduler-priority",
        choices=[
            "low_retrievability",
            "high_retrievability",
            "low_difficulty",
            "high_difficulty",
        ],
        default="low_retrievability",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--button-usage", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default="0")
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--review-markov-transition",
        action="store_true",
        help="Use button-usage Markov review transitions when available.",
    )
    parser.add_argument(
        "--train-exact-memory",
        action="store_true",
        help="Use exact memorized-day accumulation during training rollouts.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_run_monitoring_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.env not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(f"Unsupported environment: {args.env}.")
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
    if args.job_batch_size < 0:
        raise SystemExit("--job-batch-size must be >= 0.")
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
    if not (0.0 < args.retention_min < args.retention_max < 1.0):
        raise SystemExit(
            "--retention-min and --retention-max must satisfy 0 < min < max < 1."
        )
    if not (args.retention_min <= args.initial_desired_retention <= args.retention_max):
        raise SystemExit(
            "--initial-desired-retention must be within the policy retention range."
        )


def parse_user_ids(raw: str) -> list[int]:
    user_ids = [int(item) for item in raw.split(",") if item.strip()]
    if not user_ids:
        raise SystemExit("--user-ids must contain at least one user.")
    if any(user_id <= 0 for user_id in user_ids):
        raise SystemExit("--user-ids must be positive integers.")
    if len(set(user_ids)) != len(user_ids):
        raise SystemExit("--user-ids must not contain duplicates.")
    return user_ids


def parse_float_list(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value '{token}'.") from exc
        if not math.isfinite(value):
            raise SystemExit(f"{name} values must be finite.")
        values.append(value)
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def load_user_configs(
    args: argparse.Namespace, user_ids: Sequence[int]
) -> list[SingleCardFSRS6Config]:
    configs: list[SingleCardFSRS6Config] = []
    for user_id in user_ids:
        user_args = argparse.Namespace(
            env=args.env,
            user_id=user_id,
            benchmark_result=args.benchmark_result,
            benchmark_partition=args.benchmark_partition,
            srs_benchmark_root=args.srs_benchmark_root,
            button_usage=args.button_usage,
        )
        configs.append(load_single_card_fsrs6_config(user_args, environment=args.env))
    return configs


def build_jobs(
    *,
    user_ids: Sequence[int],
    cost_weights: Sequence[float],
    configs: Sequence[SingleCardFSRS6Config],
    out_dir: Path,
) -> list[ADRTrainJob]:
    config_index_by_user_id = {
        config.user_id: index for index, config in enumerate(configs)
    }
    jobs: list[ADRTrainJob] = []
    seen: set[tuple[int, float]] = set()
    for user_id in user_ids:
        config_index = config_index_by_user_id.get(user_id)
        if config_index is None:
            raise ValueError(f"Missing configuration for user {user_id}.")
        for lambda_value in cost_weights:
            key = (user_id, float(lambda_value))
            if key in seen:
                raise ValueError(
                    f"Duplicate job for user={user_id}, lambda={lambda_value}."
                )
            seen.add(key)
            jobs.append(
                ADRTrainJob(
                    user_id=user_id,
                    lambda_value=float(lambda_value),
                    config_index=config_index,
                    output_dir=(
                        out_dir
                        / "train-overfit"
                        / "train_outputs"
                        / f"user_{user_id}"
                        / f"lambda_{format_float_token(float(lambda_value))}"
                    ),
                )
            )
    return jobs


def _make_behavior(
    config: SingleCardFSRS6Config,
    *,
    review_markov_transition: bool,
) -> tuple[StochasticBehavior, StatefulCostModel]:
    button_usage = (
        load_button_usage_config(Path(config.button_usage), config.user_id)
        if config.button_usage is not None
        else None
    )
    usage = normalize_button_usage(button_usage)
    behavior = StochasticBehavior(
        attendance_prob=1.0,
        lazy_good_bias=0.0,
        max_new_per_day=None,
        max_reviews_per_day=None,
        max_cost_per_day=None,
        priority_fn=new_first_priority,
        first_rating_prob=usage["first_rating_prob"],
        review_rating_prob=usage["review_rating_prob"],
        learning_rating_prob=usage["learning_rating_prob"],
        relearning_rating_prob=usage["relearning_rating_prob"],
        review_markov_transition=(
            usage.get("long_term_transition") if review_markov_transition else None
        ),
    )
    cost_model = StatefulCostModel(
        state_costs=StateRatingCosts(
            learning=usage["learn_costs"],
            review=usage["review_costs"],
            relearning=usage["review_costs"],
        )
    )
    return behavior, cost_model


def _make_multiuser_behavior_cost(
    configs: Sequence[SingleCardFSRS6Config],
    *,
    review_markov_transition: bool,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    behaviors: list[StochasticBehavior] = []
    cost_models: list[StatefulCostModel] = []
    for config in configs:
        behavior, cost_model = _make_behavior(
            config,
            review_markov_transition=review_markov_transition,
        )
        behaviors.append(behavior)
        cost_models.append(cost_model)

    if any(behavior.review_markov_success is not None for behavior in behaviors):
        markov_rows = []
        for behavior in behaviors:
            if behavior.review_markov_success is not None:
                markov_rows.append(behavior.review_markov_success)
            else:
                fallback = list(behavior.success_dist.success_weights)
                markov_rows.append([fallback, fallback, fallback, fallback])
        markov_tensor = torch.tensor(markov_rows, device=device, dtype=dtype)
    else:
        markov_tensor = None

    multi_behavior = MultiUserBehavior(
        attendance_prob=torch.full((rows,), 1.0, device=device, dtype=dtype),
        lazy_good_bias=torch.zeros(rows, device=device, dtype=dtype),
        max_new_per_day=torch.full(
            (rows,), 1_000_000, device=device, dtype=torch.int64
        ),
        max_reviews_per_day=torch.full(
            (rows,), 1_000_000, device=device, dtype=torch.int64
        ),
        max_cost_per_day=torch.full((rows,), math.inf, device=device, dtype=dtype),
        success_weights=torch.tensor(
            [behavior.success_dist.success_weights for behavior in behaviors],
            device=device,
            dtype=dtype,
        ),
        learning_success_weights=torch.tensor(
            [behavior.learning_success_dist.success_weights for behavior in behaviors],
            device=device,
            dtype=dtype,
        ),
        relearning_success_weights=torch.tensor(
            [
                behavior.relearning_success_dist.success_weights
                for behavior in behaviors
            ],
            device=device,
            dtype=dtype,
        ),
        first_rating_prob=torch.tensor(
            [behavior.first_rating_prob for behavior in behaviors],
            device=device,
            dtype=dtype,
        ),
        review_markov_success_weights=markov_tensor,
    )
    multi_cost = MultiUserCost(
        base=torch.zeros(rows, device=device, dtype=dtype),
        penalty=torch.zeros(rows, device=device, dtype=dtype),
        learn_costs=torch.tensor(
            [cost_model.state_costs.learning for cost_model in cost_models],
            device=device,
            dtype=dtype,
        ),
        review_costs=torch.tensor(
            [cost_model.state_costs.review for cost_model in cost_models],
            device=device,
            dtype=dtype,
        ),
        learning_review_costs=torch.tensor(
            [cost_model.state_costs.learning for cost_model in cost_models],
            device=device,
            dtype=dtype,
        ),
        relearning_review_costs=torch.tensor(
            [cost_model.state_costs.relearning for cost_model in cost_models],
            device=device,
            dtype=dtype,
        ),
    )
    return multi_behavior, multi_cost


def _initial_coefficients(
    *,
    job_count: int,
    feature_version: str,
    retention_min: float,
    retention_max: float,
    desired_retention: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    baseline = FSRS6ADRPolicy.baseline(
        desired_retention=desired_retention,
        retention_min=retention_min,
        retention_max=retention_max,
        feature_version=feature_version,
    )
    return (
        torch.tensor(
            baseline.coefficients,
            device=device,
            dtype=dtype,
        )
        .unsqueeze(0)
        .repeat(job_count, 1)
    )


def _sample_population(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    population_size: int,
    theta_clip: float,
    generator: torch.Generator,
) -> torch.Tensor:
    job_count, parameter_count = mean.shape
    noise = torch.randn(
        (job_count, population_size - 1, parameter_count),
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    sampled = mean[:, None, :] + noise * std[:, None, :]
    candidates = torch.cat([mean[:, None, :], sampled], dim=1)
    return torch.clamp(candidates, min=-theta_clip, max=theta_clip)


def _evaluate_job_population(
    *,
    jobs: Sequence[ADRTrainJob],
    configs: Sequence[SingleCardFSRS6Config],
    coefficients: torch.Tensor,
    days: int,
    particles_per_group: int,
    feature_version: str,
    retention_min: float,
    retention_max: float,
    exact_memory: bool,
    job_batch_size: int,
    seed: int,
    review_markov_transition: bool,
    scheduler_priority: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[list[SimMetrics]]]:
    job_count, candidate_count, parameter_count = coefficients.shape
    if job_count != len(jobs):
        raise ValueError("job count does not match the coefficients tensor.")
    if job_count == 0:
        raise ValueError("At least one job is required.")
    if candidate_count == 0:
        raise ValueError("At least one candidate is required.")
    if parameter_count != feature_count(feature_version):
        raise ValueError(
            "Coefficient count does not match the requested feature version."
        )

    objective = torch.empty(
        (job_count, candidate_count), device=device, dtype=torch.float64
    )
    metrics_by_job: list[list[SimMetrics]] = [
        [None for _candidate in range(candidate_count)]  # type: ignore[list-item]
        for _job in range(job_count)
    ]
    chunk_size = job_count if job_batch_size <= 0 else job_batch_size
    bounds = Bounds()
    log_s_min = math.log(bounds.s_min)
    log_s_span = math.log(bounds.s_max / bounds.s_min)
    d_span = bounds.d_max - bounds.d_min

    for chunk_start in range(0, job_count, chunk_size):
        chunk_stop = min(job_count, chunk_start + chunk_size)
        job_chunk = jobs[chunk_start:chunk_stop]
        coeff_chunk = coefficients[chunk_start:chunk_stop]
        chunk_job_count = len(job_chunk)
        lane_count = chunk_job_count * candidate_count * particles_per_group
        user_indices: list[int] = []
        group_index: list[int] = []
        goal_values = torch.empty(lane_count, device=device, dtype=dtype)
        for local_job_index, job in enumerate(job_chunk):
            for candidate_index in range(candidate_count):
                group = local_job_index * candidate_count + candidate_index
                user_indices.extend([local_job_index] * particles_per_group)
                group_index.extend([group] * particles_per_group)
                start = (
                    local_job_index * candidate_count + candidate_index
                ) * particles_per_group
                goal_values[start : start + particles_per_group] = job.lambda_value

        env = MultiUserFSRS6SingleCardBatch(
            days=days,
            user_indices=user_indices,
            configs=[configs[job.config_index] for job in job_chunk],
            cost_weights=[job.lambda_value for job in job_chunk],
            action_retentions=[retention_min, retention_max],
            device=device,
            dtype=dtype,
            seed=seed + chunk_start,
            exact_memory=exact_memory,
            goal_norm_max=max(1.0, max(job.lambda_value for job in job_chunk)),
            reset_on_init=False,
        )
        env.reset_all(goal_values=goal_values)

        flat_coefficients = coeff_chunk.repeat_interleave(particles_per_group, dim=1)
        flat_coefficients = flat_coefficients.reshape(-1, parameter_count)

        while not bool(env.done.all().item()):
            retention = _retention_for_state(
                flat_coefficients,
                env.s,
                env.d,
                day=env.day,
                days=days,
                retention_min=retention_min,
                retention_max=retention_max,
                feature_version=feature_version,
                log_s_min=log_s_min,
                log_s_span=log_s_span,
                d_span=d_span,
            )
            env.step_retention(retention)

        group_metrics = env.metrics_by_group(
            group_index=torch.tensor(group_index, device=device, dtype=torch.int64),
            group_count=chunk_job_count * candidate_count,
            particles_per_group=particles_per_group,
        )
        for local_job_index, job in enumerate(job_chunk):
            for candidate_index in range(candidate_count):
                group = local_job_index * candidate_count + candidate_index
                metric = group_metrics[group]
                metrics_by_job[chunk_start + local_job_index][candidate_index] = metric
                objective[chunk_start + local_job_index, candidate_index] = (
                    metric.card_expected_retrievability
                    - job.lambda_value * metric.card_minutes_per_day
                )

    return objective, metrics_by_job


def _retention_for_state(
    coefficients: torch.Tensor,
    s: torch.Tensor,
    d: torch.Tensor,
    *,
    day: torch.Tensor,
    days: int,
    retention_min: float,
    retention_max: float,
    feature_version: str,
    log_s_min: float,
    log_s_span: float,
    d_span: float,
) -> torch.Tensor:
    s_clamped = torch.clamp(s, min=Bounds().s_min, max=Bounds().s_max)
    d_clamped = torch.clamp(d, min=Bounds().d_min, max=Bounds().d_max)
    s_norm = (torch.log(s_clamped) - log_s_min) / log_s_span
    d_norm = (d_clamped - Bounds().d_min) / d_span
    s_norm = torch.clamp(s_norm, 0.0, 1.0)
    d_norm = torch.clamp(d_norm, 0.0, 1.0)
    logit = (
        coefficients[:, 0] + coefficients[:, 1] * s_norm + coefficients[:, 2] * d_norm
    )
    if feature_version == FEATURE_VERSION_LOG_LINEAR:
        pass
    elif feature_version == FEATURE_VERSION_LOG_POLY:
        logit = (
            logit
            + coefficients[:, 3] * s_norm * d_norm
            + coefficients[:, 4] * s_norm * s_norm
            + coefficients[:, 5] * d_norm * d_norm
        )
    elif feature_version == FEATURE_VERSION_LOG_POLY_TIME:
        remaining_time_norm = torch.clamp(
            (float(days) - day.to(dtype=coefficients.dtype)) / float(days),
            0.0,
            1.0,
        )
        logit = (
            logit
            + coefficients[:, 3] * remaining_time_norm
            + coefficients[:, 4] * s_norm * d_norm
            + coefficients[:, 5] * s_norm * remaining_time_norm
            + coefficients[:, 6] * d_norm * remaining_time_norm
            + coefficients[:, 7] * s_norm * s_norm
            + coefficients[:, 8] * d_norm * d_norm
            + coefficients[:, 9] * remaining_time_norm * remaining_time_norm
        )
    else:
        raise ValueError(f"Unsupported feature_version: {feature_version}")
    return retention_min + (retention_max - retention_min) * torch.sigmoid(logit)


def _evaluate_single_policy(
    *,
    jobs: Sequence[ADRTrainJob],
    configs: Sequence[SingleCardFSRS6Config],
    coefficients: torch.Tensor,
    days: int,
    particles_per_group: int,
    feature_version: str,
    retention_min: float,
    retention_max: float,
    exact_memory: bool,
    job_batch_size: int,
    seed: int,
    review_markov_transition: bool,
    scheduler_priority: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[SimMetrics], list[float]]:
    objective, metrics_by_job = _evaluate_job_population(
        jobs=jobs,
        configs=configs,
        coefficients=coefficients[:, None, :],
        days=days,
        particles_per_group=particles_per_group,
        feature_version=feature_version,
        retention_min=retention_min,
        retention_max=retention_max,
        exact_memory=exact_memory,
        job_batch_size=job_batch_size,
        seed=seed,
        review_markov_transition=review_markov_transition,
        scheduler_priority=scheduler_priority,
        device=device,
        dtype=dtype,
    )
    metrics = [rows[0] for rows in metrics_by_job]
    objectives = [float(value) for value in objective[:, 0].tolist()]
    return metrics, objectives


def _scalar_objective(metrics: SimMetrics, lambda_value: float) -> float:
    return (
        metrics.card_expected_retrievability
        - lambda_value * metrics.card_minutes_per_day
    )


def optimize_jobs(
    *,
    args: argparse.Namespace,
    jobs: Sequence[ADRTrainJob],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, float]], float]:
    parameter_count = feature_count(args.feature_version)
    mean = _initial_coefficients(
        job_count=len(jobs),
        feature_version=args.feature_version,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        desired_retention=args.initial_desired_retention,
        device=device,
        dtype=torch.float64,
    )
    std = torch.full_like(mean, float(args.initial_std))
    best_coefficients = mean.clone()
    best_score = torch.full(
        (len(jobs),), float("-inf"), device=device, dtype=torch.float64
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 120_000)
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for generation in range(1, args.generations + 1):
        candidates = _sample_population(
            mean=mean,
            std=std,
            population_size=args.population_size,
            theta_clip=args.theta_clip,
            generator=generator,
        )
        scores, _metrics = _evaluate_job_population(
            jobs=jobs,
            configs=configs,
            coefficients=candidates,
            days=args.days,
            particles_per_group=args.train_particles,
            feature_version=args.feature_version,
            retention_min=args.retention_min,
            retention_max=args.retention_max,
            exact_memory=args.train_exact_memory,
            job_batch_size=args.job_batch_size,
            seed=args.seed + 130_000 + generation,
            review_markov_transition=args.review_markov_transition,
            scheduler_priority=args.scheduler_priority,
            device=device,
            dtype=torch.float64,
        )
        elite_scores, elite_idx = torch.topk(scores, k=args.elite_count, dim=1)
        gather_idx = elite_idx.unsqueeze(-1).expand(-1, -1, parameter_count)
        elite_theta = candidates.gather(1, gather_idx)
        generation_best_score, generation_best_idx = torch.max(scores, dim=1)
        improved = generation_best_score > best_score
        if bool(improved.any().item()):
            best_score = torch.where(improved, generation_best_score, best_score)
            best_coefficients[improved] = candidates[
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
        population_mean = scores.mean(dim=1)
        for job_index, job in enumerate(jobs):
            history.append(
                {
                    "generation": float(generation),
                    "user_id": float(job.user_id),
                    "lambda_value": float(job.lambda_value),
                    "best_objective": float(best_score[job_index].item()),
                    "generation_best_objective": float(
                        generation_best_score[job_index].item()
                    ),
                    "elite_mean_objective": float(
                        elite_scores[job_index].mean().item()
                    ),
                    "population_mean_objective": float(
                        population_mean[job_index].item()
                    ),
                    "mean_std": float(std[job_index].mean().item()),
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
    return best_coefficients, history, time.perf_counter() - start


def _job_label(job: ADRTrainJob) -> str:
    return f"user_{job.user_id}/lambda_{format_float_token(job.lambda_value)}"


def _policy_title(
    *,
    scheduler_name: str,
    user_id: int,
    lambda_value: float,
    initial_desired_retention: float,
) -> str:
    return (
        f"{scheduler_name}_single_card_u{user_id}_lambda_{format_float(lambda_value)}_"
        f"initdr_{format_float(initial_desired_retention)}"
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_path_string(path: Path | None, *, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _metrics_dict(metrics: SimMetrics) -> dict[str, Any]:
    return asdict(metrics)


def _write_policy_manifest(
    *,
    path: Path,
    feature_version: str,
    jobs: Sequence[ADRTrainJob],
    policies: Sequence[Path],
    scheduler_name: str,
    action_space: str,
) -> None:
    lines = [
        'family = "single_card_tradeoff"',
        "schema_version = 1",
        f'generated_at = "{_now_iso()}"',
        f'feature_version = "{feature_version}"',
        f'scheduler_name = "{scheduler_name}"',
        f'action_space = "{action_space}"',
        "",
    ]
    for job, policy_path in zip(jobs, policies, strict=True):
        lines.extend(
            [
                "[[policies]]",
                f"user_id = {job.user_id}",
                f"lambda_value = {format_float(job.lambda_value)}",
                f'path = "{_relative_path_string(policy_path, base=path.parent) or policy_path.name}"',
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _baseline_policy(
    *,
    args: argparse.Namespace,
) -> FSRS6ADRPolicy:
    return FSRS6ADRPolicy.baseline(
        desired_retention=args.initial_desired_retention,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        feature_version=args.feature_version,
    )


def _build_result(
    *,
    job: ADRTrainJob,
    baseline_train_metrics: SimMetrics,
    baseline_eval_metrics: SimMetrics,
    train_best_metrics: SimMetrics,
    eval_best_metrics: SimMetrics,
    best_coefficients: torch.Tensor,
    history: list[dict[str, float]],
    train_runtime_s: float,
    eval_runtime_s: float,
) -> ADRTrainResult:
    baseline_train_objective = _scalar_objective(
        baseline_train_metrics,
        job.lambda_value,
    )
    baseline_eval_objective = _scalar_objective(
        baseline_eval_metrics,
        job.lambda_value,
    )
    train_best_objective = _scalar_objective(train_best_metrics, job.lambda_value)
    eval_best_objective = _scalar_objective(eval_best_metrics, job.lambda_value)
    return ADRTrainResult(
        job=job,
        baseline_train_metrics=baseline_train_metrics,
        baseline_eval_metrics=baseline_eval_metrics,
        train_best_metrics=train_best_metrics,
        eval_best_metrics=eval_best_metrics,
        baseline_train_objective=baseline_train_objective,
        baseline_eval_objective=baseline_eval_objective,
        train_best_objective=train_best_objective,
        eval_best_objective=eval_best_objective,
        best_coefficients=best_coefficients.detach().cpu(),
        history=history,
        train_runtime_s=train_runtime_s,
        eval_runtime_s=eval_runtime_s,
        passed=eval_best_objective >= baseline_eval_objective,
    )


def _write_job_artifacts(
    *,
    result: ADRTrainResult,
    config: SingleCardFSRS6Config,
    args: argparse.Namespace,
    scheduler_name: str,
    action_space: str,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = FSRS6ADRPolicy(
        coefficients=tuple(float(value) for value in result.best_coefficients.tolist()),
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        baseline_desired_retention=None,
        feature_version=args.feature_version,
        title=_policy_title(
            scheduler_name=scheduler_name,
            user_id=result.job.user_id,
            lambda_value=result.job.lambda_value,
            initial_desired_retention=args.initial_desired_retention,
        ),
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)

    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "job": {
                "user_id": result.job.user_id,
                "lambda_value": result.job.lambda_value,
                "config_index": result.job.config_index,
                "output_dir": _relative_path_string(output_dir, base=output_root),
            },
            "feature_version": args.feature_version,
            "scheduler_name": scheduler_name,
            "action_space": action_space,
            "initial_desired_retention": args.initial_desired_retention,
            "retention_min": args.retention_min,
            "retention_max": args.retention_max,
            "train_particles": args.train_particles,
            "eval_particles": args.eval_particles,
            "population_size": args.population_size,
            "elite_count": args.elite_count,
            "generations": args.generations,
            "baseline_train_metrics": _metrics_dict(result.baseline_train_metrics),
            "baseline_eval_metrics": _metrics_dict(result.baseline_eval_metrics),
            "train_best_metrics": _metrics_dict(result.train_best_metrics),
            "eval_best_metrics": _metrics_dict(result.eval_best_metrics),
            "baseline_train_objective": result.baseline_train_objective,
            "baseline_eval_objective": result.baseline_eval_objective,
            "train_best_objective": result.train_best_objective,
            "eval_best_objective": result.eval_best_objective,
            "objective_improvement": result.eval_best_objective
            - result.baseline_eval_objective,
            "best_coefficients": [
                float(value) for value in result.best_coefficients.tolist()
            ],
            "train_runtime_s": result.train_runtime_s,
            "eval_runtime_s": result.eval_runtime_s,
            "passed": result.passed,
        },
    )

    metadata_path = output_dir / "metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "artifact_kind": "scheduler-policy",
            "artifact_id": (
                f"single-card-adr-{result.job.user_id}-"
                f"lambda-{format_float(result.job.lambda_value)}"
            ),
            "family": "single_card_tradeoff",
            "training_scope": "single_card_tradeoff_direct",
            "training_objective": "card_expected_retrievability_minus_lambda_minutes_per_day",
            "scheduler_name": scheduler_name,
            "environment": args.env,
            "engine": "batched",
            "training_engine": "single_card_tradeoff_direct",
            "review_markov_transition": args.review_markov_transition,
            "training_user_ids": [result.job.user_id],
            "validation_user_ids": [],
            "seed": args.seed,
            "policy_path": "policy.json",
            "feature_version": args.feature_version,
            "action_space": action_space,
            "created_at": _now_iso(),
            "code_commit": _git_commit(),
            "lambda_value": result.job.lambda_value,
            "baseline_desired_retention": None,
            "initial_desired_retention": args.initial_desired_retention,
            "config_snapshot_path": None,
            "training_command_path": None,
            "metrics_path": "metrics.json",
            "capabilities": ["event", "batched"],
            "train_particles": args.train_particles,
            "eval_particles": args.eval_particles,
            "population_size": args.population_size,
            "elite_count": args.elite_count,
            "generations": args.generations,
            "train_runtime_s": result.train_runtime_s,
            "eval_runtime_s": result.eval_runtime_s,
            "passed": result.passed,
            "train_best_objective": result.train_best_objective,
            "eval_best_objective": result.eval_best_objective,
            "objective_improvement": result.eval_best_objective
            - result.baseline_eval_objective,
        },
    )

    return policy_path, metrics_path, metadata_path


def _write_root_summary(
    *,
    path: Path,
    results: Sequence[ADRTrainResult],
    scheduler_name: str,
    action_space: str,
    feature_version: str,
    args: argparse.Namespace,
    output_root: Path,
    policy_paths: Sequence[Path],
    metadata_paths: Sequence[Path],
) -> None:
    fieldnames = [
        "user_id",
        "lambda_value",
        "scheduler_name",
        "action_space",
        "feature_version",
        "policy_path",
        "metadata_path",
        "baseline_train_objective",
        "baseline_eval_objective",
        "train_best_objective",
        "eval_best_objective",
        "objective_improvement",
        "train_best_card_expected_retrievability",
        "train_best_card_minutes_per_day",
        "eval_best_card_expected_retrievability",
        "eval_best_card_minutes_per_day",
        "train_runtime_s",
        "eval_runtime_s",
        "passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result, policy_path, metadata_path in zip(
            results, policy_paths, metadata_paths, strict=True
        ):
            writer.writerow(
                {
                    "user_id": result.job.user_id,
                    "lambda_value": format_float(result.job.lambda_value),
                    "scheduler_name": scheduler_name,
                    "action_space": action_space,
                    "feature_version": feature_version,
                    "policy_path": _relative_path_string(policy_path, base=output_root),
                    "metadata_path": _relative_path_string(
                        metadata_path, base=output_root
                    ),
                    "baseline_train_objective": result.baseline_train_objective,
                    "baseline_eval_objective": result.baseline_eval_objective,
                    "train_best_objective": result.train_best_objective,
                    "eval_best_objective": result.eval_best_objective,
                    "objective_improvement": (
                        result.eval_best_objective - result.baseline_eval_objective
                    ),
                    "train_best_card_expected_retrievability": (
                        result.train_best_metrics.card_expected_retrievability
                    ),
                    "train_best_card_minutes_per_day": (
                        result.train_best_metrics.card_minutes_per_day
                    ),
                    "eval_best_card_expected_retrievability": (
                        result.eval_best_metrics.card_expected_retrievability
                    ),
                    "eval_best_card_minutes_per_day": (
                        result.eval_best_metrics.card_minutes_per_day
                    ),
                    "train_runtime_s": result.train_runtime_s,
                    "eval_runtime_s": result.eval_runtime_s,
                    "passed": result.passed,
                }
            )


def _write_history(
    *,
    path: Path,
    history: Sequence[dict[str, float]],
) -> None:
    fieldnames = [
        "generation",
        "user_id",
        "lambda_value",
        "best_objective",
        "generation_best_objective",
        "elite_mean_objective",
        "population_mean_objective",
        "mean_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in fieldnames})


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def main() -> int:
    args = parse_args()
    validate_args(args)
    user_ids = parse_user_ids(args.user_ids)
    cost_weights = parse_float_list(args.cost_weights, name="--cost-weights")
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    configs = load_user_configs(args, user_ids)
    jobs = build_jobs(
        user_ids=user_ids,
        cost_weights=cost_weights,
        configs=configs,
        out_dir=args.out_dir,
    )
    device = (
        torch.device(args.torch_device)
        if args.torch_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    progress = TrainingProgress(args.out_dir / "training_progress.jsonl")
    progress.write(
        "started",
        device=device,
        env=args.env,
        feature_version=args.feature_version,
        user_ids=user_ids,
        cost_weights=[float(value) for value in cost_weights],
        population_size=args.population_size,
        elite_count=args.elite_count,
        generations=args.generations,
        train_particles=args.train_particles,
        eval_particles=args.eval_particles,
        job_batch_size=args.job_batch_size,
    )

    baseline_policy = _baseline_policy(args=args)
    scheduler_variant = fsrs6_adr_variant_for_feature_version(args.feature_version)
    scheduler_name = scheduler_variant.scheduler_name
    action_space = scheduler_variant.action_space

    progress.write(
        "config_loaded",
        device=device,
        scheduler_name=scheduler_name,
        action_space=action_space,
        feature_version=args.feature_version,
        job_count=len(jobs),
        config_count=len(configs),
    )

    baseline_coefficients = (
        torch.tensor(
            baseline_policy.coefficients,
            device=device,
            dtype=torch.float64,
        )
        .unsqueeze(0)
        .repeat(len(jobs), 1)
    )

    baseline_train_eval_start = time.perf_counter()
    baseline_train_metrics, _baseline_train_objectives = _evaluate_single_policy(
        jobs=jobs,
        configs=configs,
        coefficients=baseline_coefficients,
        days=args.days,
        particles_per_group=args.train_particles,
        feature_version=args.feature_version,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        exact_memory=args.train_exact_memory,
        job_batch_size=args.job_batch_size,
        seed=args.seed + 50_000,
        review_markov_transition=args.review_markov_transition,
        scheduler_priority=args.scheduler_priority,
        device=device,
        dtype=torch.float64,
    )
    baseline_train_eval_runtime_s = time.perf_counter() - baseline_train_eval_start

    baseline_eval_start = time.perf_counter()
    baseline_eval_metrics, _baseline_eval_objectives = _evaluate_single_policy(
        jobs=jobs,
        configs=configs,
        coefficients=baseline_coefficients,
        days=args.days,
        particles_per_group=args.eval_particles,
        feature_version=args.feature_version,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        exact_memory=True,
        job_batch_size=args.job_batch_size,
        seed=args.seed + 60_000,
        review_markov_transition=args.review_markov_transition,
        scheduler_priority=args.scheduler_priority,
        device=device,
        dtype=torch.float64,
    )
    baseline_eval_runtime_s = time.perf_counter() - baseline_eval_start

    best_coefficients, history, train_runtime_s = optimize_jobs(
        args=args,
        jobs=jobs,
        configs=configs,
        device=device,
    )

    train_best_eval_start = time.perf_counter()
    train_best_metrics, _train_best_objectives = _evaluate_single_policy(
        jobs=jobs,
        configs=configs,
        coefficients=best_coefficients,
        days=args.days,
        particles_per_group=args.train_particles,
        feature_version=args.feature_version,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        exact_memory=args.train_exact_memory,
        job_batch_size=args.job_batch_size,
        seed=args.seed + 70_000,
        review_markov_transition=args.review_markov_transition,
        scheduler_priority=args.scheduler_priority,
        device=device,
        dtype=torch.float64,
    )
    train_best_eval_runtime_s = time.perf_counter() - train_best_eval_start

    eval_best_eval_start = time.perf_counter()
    eval_best_metrics, _eval_best_objectives = _evaluate_single_policy(
        jobs=jobs,
        configs=configs,
        coefficients=best_coefficients,
        days=args.days,
        particles_per_group=args.eval_particles,
        feature_version=args.feature_version,
        retention_min=args.retention_min,
        retention_max=args.retention_max,
        exact_memory=True,
        job_batch_size=args.job_batch_size,
        seed=args.seed + 80_000,
        review_markov_transition=args.review_markov_transition,
        scheduler_priority=args.scheduler_priority,
        device=device,
        dtype=torch.float64,
    )
    eval_best_eval_runtime_s = time.perf_counter() - eval_best_eval_start
    eval_runtime_s = (
        baseline_train_eval_runtime_s
        + baseline_eval_runtime_s
        + train_best_eval_runtime_s
        + eval_best_eval_runtime_s
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    results = [
        _build_result(
            job=job,
            baseline_train_metrics=baseline_train_metrics[index],
            baseline_eval_metrics=baseline_eval_metrics[index],
            train_best_metrics=train_best_metrics[index],
            eval_best_metrics=eval_best_metrics[index],
            best_coefficients=best_coefficients[index],
            history=[
                row
                for row in history
                if int(row["user_id"]) == job.user_id
                and math.isclose(row["lambda_value"], job.lambda_value, abs_tol=1e-12)
            ],
            train_runtime_s=train_runtime_s,
            eval_runtime_s=eval_runtime_s,
        )
        for index, job in enumerate(jobs)
    ]

    for result in results:
        result.job.output_dir.mkdir(parents=True, exist_ok=True)

    policy_paths: list[Path] = []
    metadata_paths: list[Path] = []
    for result, config in zip(
        results, (configs[result.job.config_index] for result in results), strict=True
    ):
        policy_path, _metrics_path, metadata_path = _write_job_artifacts(
            result=result,
            config=config,
            args=args,
            scheduler_name=scheduler_name,
            action_space=action_space,
            output_root=args.out_dir,
        )
        policy_paths.append(policy_path)
        metadata_paths.append(metadata_path)

    history_path = args.out_dir / "train_history.csv"
    summary_path = args.out_dir / "summary.csv"
    manifest_path = args.out_dir / "policy_manifest.toml"
    metadata_path = args.out_dir / "metadata.json"

    _write_history(path=history_path, history=history)
    _write_root_summary(
        path=summary_path,
        results=results,
        scheduler_name=scheduler_name,
        action_space=action_space,
        feature_version=args.feature_version,
        args=args,
        output_root=args.out_dir,
        policy_paths=policy_paths,
        metadata_paths=metadata_paths,
    )
    _write_policy_manifest(
        path=manifest_path,
        feature_version=args.feature_version,
        jobs=jobs,
        policies=policy_paths,
        scheduler_name=scheduler_name,
        action_space=action_space,
    )

    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "artifact_kind": "single-card-adr-direct-train-run",
            "family": "single_card_tradeoff",
            "scheduler_name": scheduler_name,
            "action_space": action_space,
            "feature_version": args.feature_version,
            "environment": args.env,
            "user_ids": user_ids,
            "cost_weights": [float(value) for value in cost_weights],
            "job_count": len(jobs),
            "config_count": len(configs),
            "days": args.days,
            "seed": args.seed,
            "code_commit": _git_commit(),
            "retention_min": args.retention_min,
            "retention_max": args.retention_max,
            "initial_desired_retention": args.initial_desired_retention,
            "population_size": args.population_size,
            "elite_count": args.elite_count,
            "generations": args.generations,
            "train_particles": args.train_particles,
            "eval_particles": args.eval_particles,
            "job_batch_size": args.job_batch_size,
            "train_runtime_s": train_runtime_s,
            "eval_runtime_s": eval_runtime_s,
            "summary_path": _relative_path_string(summary_path, base=args.out_dir),
            "history_path": _relative_path_string(history_path, base=args.out_dir),
            "policy_manifest_path": _relative_path_string(
                manifest_path, base=args.out_dir
            ),
            "jobs": [
                {
                    "user_id": result.job.user_id,
                    "lambda_value": result.job.lambda_value,
                    "output_dir": _relative_path_string(
                        result.job.output_dir, base=args.out_dir
                    ),
                    "policy_path": _relative_path_string(
                        policy_path, base=args.out_dir
                    ),
                    "metadata_path": _relative_path_string(
                        metadata_path, base=args.out_dir
                    ),
                    "baseline_eval_objective": result.baseline_eval_objective,
                    "eval_best_objective": result.eval_best_objective,
                    "objective_improvement": (
                        result.eval_best_objective - result.baseline_eval_objective
                    ),
                    "passed": result.passed,
                }
                for result, policy_path, metadata_path in zip(
                    results, policy_paths, metadata_paths, strict=True
                )
            ],
        },
    )

    progress.write(
        "completed",
        device=device,
        train_runtime_s=train_runtime_s,
        eval_runtime_s=eval_runtime_s,
        job_count=len(jobs),
        policy_count=len(policy_paths),
    )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote history: {history_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metadata: {metadata_path}")
    for policy_path in policy_paths:
        print(f"Wrote policy: {policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
