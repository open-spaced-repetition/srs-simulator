from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    MultiUserFSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.cli.uvfa_ppo import (  # noqa: E402
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    DEFAULT_FIXED_INTERVALS,
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.core.target_search.family_search import (  # noqa: E402
    adaptive_theta_candidates,
    missing_values_from_values,
    unique_sorted,
)
from experiments.single_card_tradeoff.core.target_search.frontier import (  # noqa: E402
    empirical_frontier,
    frontier_segments,
    target_answers,
)
from experiments.single_card_tradeoff.core.target_search.io import (  # noqa: E402
    answer_row,
    point_row,
    read_points_csv,
    segment_row,
)
from experiments.single_card_tradeoff.core.target_search.oracle_refinement import (  # noqa: E402
    apply_target_certifications,
    certify_oracle_segments,
    oracle_refinement_candidates,
    target_certification_map,
)
from experiments.single_card_tradeoff.core.target_search.types import (  # noqa: E402
    ConstrainedTarget,
    EvaluatedPoint,
)
from experiments.single_card_tradeoff.core.types import SimMetrics  # noqa: E402
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    BatchedStationaryFiniteOracleSolution,
    FSRS6BatchedContinuousStationaryFiniteOracle,
    FSRS6BatchedStationaryFiniteOracle,
    OracleMetrics,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402

DEFAULT_OUT_DIR = Path("artifacts/single_card_tradeoff/target_search")
ROLLOUT_FAMILIES = ("fsrs6", "fixed")
ORACLE_FAMILIES = (
    "fsrs6_oracle_stationary_finite",
    "fsrs6_oracle_continuous_stationary_finite",
)
FAMILY_CHOICES = (*ROLLOUT_FAMILIES, *ORACLE_FAMILIES)
DEFAULT_EXPLORE_PARTICLES = 1024
DEFAULT_CONFIRM_PARTICLES = 10_000
DEFAULT_CERTIFICATE_TOLERANCE = 1e-9
DEFAULT_ORACLE_CONTINUOUS_INTERVAL_CHUNK_SIZE = 64


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find best feasible deterministic single-card policies for batches "
            "of target memory or target time constraints."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument(
        "--user-ids",
        default=None,
        help="Comma-separated user IDs. Defaults to --user-id or 1.",
    )
    parser.add_argument(
        "--family",
        choices=FAMILY_CHOICES,
        default="fsrs6",
        help="One-dimensional policy family to search.",
    )
    parser.add_argument(
        "--target-memories",
        default="",
        help="Comma-separated M0 targets for min-time constrained search.",
    )
    parser.add_argument(
        "--target-times",
        default="",
        help="Comma-separated T0 minute/day targets for max-memory search.",
    )
    parser.add_argument(
        "--theta-grid",
        default=None,
        help=(
            "Initial theta grid. Defaults to target retentions for --family fsrs6, "
            "fixed intervals for --family fixed, and scalarization cost weights "
            "for oracle families."
        ),
    )
    parser.add_argument("--theta-min", type=float, default=None)
    parser.add_argument("--theta-max", type=float, default=None)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--explore-particles", type=int, default=DEFAULT_EXPLORE_PARTICLES
    )
    parser.add_argument("--confirm-particles", type=int, default=None)
    parser.add_argument(
        "--particles",
        type=int,
        default=None,
        help="Alias for --confirm-particles when that flag is omitted.",
    )
    parser.add_argument("--eval-group-batch-size", type=int, default=0)
    parser.add_argument("--max-refinement-rounds", type=int, default=4)
    parser.add_argument("--candidates-per-bracket", type=int, default=3)
    parser.add_argument(
        "--candidates-per-round",
        type=int,
        default=None,
        help=(
            "Maximum exact-oracle lambda_AB candidates per refinement round. "
            "Defaults to --candidates-per-bracket."
        ),
    )
    parser.add_argument("--memory-margin", type=float, default=0.0)
    parser.add_argument("--time-margin", type=float, default=0.0)
    parser.add_argument(
        "--init-points",
        action="append",
        type=Path,
        default=[],
        help=(
            "Existing target-search points.csv to warm-start from. May be "
            "passed multiple times."
        ),
    )
    parser.add_argument(
        "--certificate-tolerance",
        type=float,
        default=DEFAULT_CERTIFICATE_TOLERANCE,
        help="Tolerance for exact-oracle lambda_AB segment certificates.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions for stationary finite oracle family.",
    )
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=DEFAULT_ORACLE_S_GRID_SIZE,
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=DEFAULT_ORACLE_D_GRID_SIZE,
    )
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
    parser.add_argument("--oracle-continuous-retention-min", type=float, default=0.5)
    parser.add_argument("--oracle-continuous-retention-max", type=float, default=0.98)
    parser.add_argument(
        "--oracle-continuous-interval-chunk-size",
        type=int,
        default=DEFAULT_ORACLE_CONTINUOUS_INTERVAL_CHUNK_SIZE,
    )
    parser.add_argument(
        "--progress-log-interval-seconds",
        type=float,
        default=0.0,
        help="Diagnostic log cadence for continuous stationary oracle internals.",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Select deterministic policies. Mixed fields remain diagnostics.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_run_monitoring_args(parser)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def _parse_csv_floats(raw: str | None, *, name: str) -> list[float]:
    if raw is None or raw.strip() == "":
        return []
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value {token!r}.") from exc
        if not math.isfinite(value):
            raise SystemExit(f"{name} values must be finite.")
        values.append(value)
    return values


def _parse_user_ids(args: argparse.Namespace) -> list[int]:
    raw = args.user_ids
    if raw is None or raw.strip() == "":
        raw = str(args.user_id or 1)
    user_ids = [int(item) for item in raw.split(",") if item.strip()]
    if not user_ids:
        raise SystemExit("--user-ids must contain at least one user.")
    if any(user_id <= 0 for user_id in user_ids):
        raise SystemExit("--user-ids must be positive integers.")
    if len(set(user_ids)) != len(user_ids):
        raise SystemExit("--user-ids must not contain duplicates.")
    return user_ids


def _is_oracle_family(family: str) -> bool:
    return family in ORACLE_FAMILIES


def _validate_args(args: argparse.Namespace) -> None:
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.explore_particles <= 0:
        raise SystemExit("--explore-particles must be > 0.")
    confirm_particles = _confirm_particles(args)
    if confirm_particles <= 0:
        raise SystemExit("--confirm-particles/--particles must be > 0.")
    if args.eval_group_batch_size < 0:
        raise SystemExit("--eval-group-batch-size must be >= 0.")
    if args.max_refinement_rounds < 0:
        raise SystemExit("--max-refinement-rounds must be >= 0.")
    if args.candidates_per_bracket < 0:
        raise SystemExit("--candidates-per-bracket must be >= 0.")
    if args.candidates_per_round is not None and args.candidates_per_round < 0:
        raise SystemExit("--candidates-per-round must be >= 0.")
    if args.memory_margin < 0.0:
        raise SystemExit("--memory-margin must be >= 0.")
    if args.time_margin < 0.0:
        raise SystemExit("--time-margin must be >= 0.")
    for path in args.init_points:
        if not path.exists():
            raise SystemExit(f"--init-points path does not exist: {path}")
    if args.certificate_tolerance < 0.0:
        raise SystemExit("--certificate-tolerance must be >= 0.")
    if _is_oracle_family(args.family):
        if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
            raise SystemExit("--oracle-*-grid-size values must be >= 8.")
        if args.oracle_stationary_finite_max_iterations <= 0:
            raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
        if args.oracle_stationary_finite_tolerance <= 0.0:
            raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")
        if args.oracle_continuous_interval_chunk_size <= 0:
            raise SystemExit("--oracle-continuous-interval-chunk-size must be > 0.")
        if args.progress_log_interval_seconds < 0.0:
            raise SystemExit("--progress-log-interval-seconds must be >= 0.")
        if not (
            0.0
            < args.oracle_continuous_retention_min
            <= args.oracle_continuous_retention_max
            < 1.0
        ):
            raise SystemExit(
                "continuous retention bounds must satisfy 0 < min <= max < 1."
            )
    if not _parse_csv_floats(args.target_memories, name="--target-memories") and not (
        _parse_csv_floats(args.target_times, name="--target-times")
    ):
        raise SystemExit(
            "Provide at least one --target-memories or --target-times value."
        )


def _confirm_particles(args: argparse.Namespace) -> int:
    if args.confirm_particles is not None:
        return int(args.confirm_particles)
    if args.particles is not None:
        return int(args.particles)
    return DEFAULT_CONFIRM_PARTICLES


def _theta_defaults(family: str) -> tuple[list[float], str, float, float]:
    if family == "fsrs6":
        values = [float(value) for value in DEFAULT_TARGET_RETENTIONS]
        return values, "continuous", min(values), max(values)
    if family == "fixed":
        values = [float(value) for value in DEFAULT_FIXED_INTERVALS]
        return values, "integer", 1.0, max(values)
    if _is_oracle_family(family):
        values = [float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS]
        return values, "continuous", 0.0, max(values)
    raise ValueError(f"Unsupported family: {family}")


def _theta_grid(args: argparse.Namespace) -> tuple[list[float], str, float, float]:
    default_values, theta_kind, default_min, default_max = _theta_defaults(args.family)
    values = _parse_csv_floats(args.theta_grid, name="--theta-grid")
    if not values:
        values = default_values
    theta_min = default_min if args.theta_min is None else float(args.theta_min)
    theta_max = default_max if args.theta_max is None else float(args.theta_max)
    if theta_max <= theta_min:
        raise SystemExit("--theta-max must be greater than --theta-min.")
    values = [value for value in values if theta_min <= value <= theta_max]
    if not values:
        raise SystemExit("--theta-grid has no values inside theta bounds.")
    if args.family == "fsrs6" and any(not (0.0 < value < 1.0) for value in values):
        raise SystemExit("fsrs6 theta values must satisfy 0 < retention < 1.")
    if args.family == "fixed" and any(value < 1.0 for value in values):
        raise SystemExit("fixed theta values must be >= 1 day.")
    if _is_oracle_family(args.family) and any(value < 0.0 for value in values):
        raise SystemExit("oracle cost weights must be >= 0.")
    if theta_kind == "integer":
        values = [float(round(value)) for value in values]
    return unique_sorted(values), theta_kind, theta_min, theta_max


def _targets(
    args: argparse.Namespace, user_ids: Sequence[int]
) -> list[ConstrainedTarget]:
    memory_values = _parse_csv_floats(args.target_memories, name="--target-memories")
    time_values = _parse_csv_floats(args.target_times, name="--target-times")
    targets: list[ConstrainedTarget] = []
    for user_id in user_ids:
        targets.extend(
            ConstrainedTarget("memory", value, user_id=user_id)
            for value in memory_values
        )
        targets.extend(
            ConstrainedTarget("time", value, user_id=user_id) for value in time_values
        )
    return targets


def _load_user_configs(
    args: argparse.Namespace,
    user_ids: Sequence[int],
) -> list[SingleCardFSRS6Config]:
    configs: list[SingleCardFSRS6Config] = []
    for user_id in user_ids:
        user_args = argparse.Namespace(**vars(args))
        user_args.user_id = user_id
        configs.append(load_single_card_fsrs6_config(user_args, environment=args.env))
    return configs


def _group_chunks(values: Sequence[float], group_batch_size: int) -> list[list[float]]:
    if not values:
        return []
    chunk_size = len(values) if group_batch_size <= 0 else group_batch_size
    return [
        list(values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]


def _batched_layout(
    *,
    user_count: int,
    group_count: int,
    particles_per_group: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    user_indices: list[int] = []
    group_indices: list[int] = []
    local_group_indices: list[int] = []
    for user_idx in range(user_count):
        for group_idx in range(group_count):
            user_indices.extend([user_idx] * particles_per_group)
            group_indices.extend(
                [user_idx * group_count + group_idx] * particles_per_group
            )
            local_group_indices.extend([group_idx] * particles_per_group)
    return (
        user_indices,
        torch.tensor(group_indices, device=device, dtype=torch.int64),
        torch.tensor(local_group_indices, device=device, dtype=torch.int64),
    )


def _metrics_to_point(
    *,
    user_id: int,
    family: str,
    theta: float,
    metrics: SimMetrics | OracleMetrics,
    eval_stage: str,
    particles: int | None,
    seed: int | None,
    runtime_s: float,
    exact: bool = False,
    policy_ref: str | None = None,
) -> EvaluatedPoint:
    if family == "fsrs6":
        theta_name = "desired_retention"
    elif family == "fixed":
        theta_name = "fixed_interval"
    else:
        theta_name = "goal_cost_weight"
    return EvaluatedPoint(
        user_id=user_id,
        family=family,
        theta_name=theta_name,
        theta_value=float(theta),
        memory=metrics.card_expected_retrievability,
        minutes=metrics.card_minutes_per_day,
        policy_ref=policy_ref,
        cache_key=None,
        exact=exact,
        eval_stage=eval_stage,
        particles=particles,
        seed=seed,
        runtime_s=runtime_s,
    )


def _point_key(point: EvaluatedPoint, *, digits: int = 12) -> tuple[int, str, float]:
    return (point.user_id, point.family, round(point.theta_value, digits))


def _merge_points_by_key(points: Sequence[EvaluatedPoint]) -> list[EvaluatedPoint]:
    merged: dict[tuple[int, str, float], EvaluatedPoint] = {}
    for point in points:
        key = _point_key(point)
        current = merged.get(key)
        if current is None:
            merged[key] = point
            continue
        if _point_rank(point) >= _point_rank(current):
            merged[key] = point
    return list(merged.values())


def _point_rank(point: EvaluatedPoint) -> tuple[int, int, int]:
    stage_rank = {"exact": 4, "confirmed": 3, "explore": 2}.get(point.eval_stage, 1)
    exact_rank = 1 if point.exact else 0
    particle_rank = point.particles if point.particles is not None else 0
    return (exact_rank, stage_rank, particle_rank)


def _load_initial_points(
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    theta_min: float,
    theta_max: float,
) -> list[EvaluatedPoint]:
    if not args.init_points:
        return []
    users = set(user_ids)
    points: list[EvaluatedPoint] = []
    for path in args.init_points:
        loaded = read_points_csv(path)
        points.extend(
            point
            for point in loaded
            if point.family == args.family
            and point.user_id in users
            and theta_min <= point.theta_value <= theta_max
        )
    merged = _merge_points_by_key(points)
    if merged and not args.no_progress:
        print(
            f"warm_start: loaded {len(merged)} points from "
            f"{len(args.init_points)} file(s)",
            flush=True,
        )
    return merged


def _missing_thetas_for_all_users(
    theta_values: Sequence[float],
    points: Sequence[EvaluatedPoint],
    *,
    user_ids: Sequence[int],
    family: str,
    require_exact: bool,
) -> list[float]:
    present: set[tuple[int, float]] = set()
    requested_users = set(user_ids)
    for point in points:
        if point.family != family or point.user_id not in requested_users:
            continue
        if require_exact and not point.exact:
            continue
        present.add((point.user_id, round(point.theta_value, 12)))
    missing: list[float] = []
    for theta in unique_sorted(theta_values):
        key = round(theta, 12)
        if all((user_id, key) in present for user_id in user_ids):
            continue
        missing.append(theta)
    return missing


@torch.inference_mode()
def evaluate_family_points(
    *,
    family: str,
    theta_values: Sequence[float],
    user_ids: Sequence[int],
    configs: Sequence[SingleCardFSRS6Config],
    days: int,
    particles: int,
    group_batch_size: int,
    seed: int,
    device: torch.device,
    eval_stage: str,
    progress: bool,
) -> list[EvaluatedPoint]:
    points: list[EvaluatedPoint] = []
    user_count = len(configs)
    for batch_values in _group_chunks(theta_values, group_batch_size):
        group_count = len(batch_values)
        user_indices, group_index, local_group_idx = _batched_layout(
            user_count=user_count,
            group_count=group_count,
            particles_per_group=particles,
            device=device,
        )
        env = MultiUserFSRS6SingleCardBatch(
            days=days,
            user_indices=user_indices,
            configs=configs,
            cost_weights=[0.0],
            action_retentions=batch_values if family == "fsrs6" else [0.5, 0.98],
            device=device,
            dtype=torch.float64,
            seed=seed + int(round(float(batch_values[0]) * 10_000.0)),
            exact_memory=True,
            goal_norm_max=1.0,
        )
        start = time.perf_counter()
        if family == "fsrs6":
            action = local_group_idx
            while not bool(env.done.all().item()):
                env.step(action)
        elif family == "fixed":
            intervals_by_group = torch.tensor(
                [max(1, int(round(value))) for value in batch_values],
                device=device,
                dtype=torch.int64,
            )
            intervals = intervals_by_group.index_select(0, local_group_idx)
            while not bool(env.done.all().item()):
                env.step_intervals(intervals)
        else:
            raise ValueError(f"Unsupported family: {family}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start
        runtime_s = elapsed_s / float(max(1, user_count * group_count))
        metrics_flat = env.metrics_by_group(
            group_index=group_index,
            group_count=user_count * group_count,
            particles_per_group=particles,
        )
        for local_idx, theta in enumerate(batch_values):
            for user_idx, user_id in enumerate(user_ids):
                metrics = metrics_flat[user_idx * group_count + local_idx]
                points.append(
                    _metrics_to_point(
                        user_id=user_id,
                        family=family,
                        theta=theta,
                        metrics=metrics,
                        eval_stage=eval_stage,
                        particles=particles,
                        seed=seed,
                        runtime_s=runtime_s,
                    )
                )
        if progress:
            print(
                f"{eval_stage}: evaluated {family} theta "
                f"{format_float(batch_values[0])}..{format_float(batch_values[-1])} "
                f"for {len(user_ids)} users with particles={particles}",
                flush=True,
            )
    return points


def _oracle_action_retentions(args: argparse.Namespace) -> list[float]:
    action_retentions = _parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    return action_retentions


def _build_oracle(
    *,
    args: argparse.Namespace,
    family: str,
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
) -> FSRS6BatchedStationaryFiniteOracle | FSRS6BatchedContinuousStationaryFiniteOracle:
    cache_config = configure_oracle_dp_cache_from_args(args)
    common = {
        "days": int(args.days),
        "s_grid_size": int(args.oracle_s_grid_size),
        "d_grid_size": int(args.oracle_d_grid_size),
        "fsrs_weights": [config.fsrs_weights for config in configs],
        "first_rating_prob": [config.first_rating_prob for config in configs],
        "review_rating_prob": [config.review_rating_prob for config in configs],
        "learning_costs": [config.learning_costs for config in configs],
        "review_costs": [config.review_costs for config in configs],
        "dtype": torch.float64,
        "device": device,
        "cache_config": cache_config,
    }
    if family == "fsrs6_oracle_stationary_finite":
        return FSRS6BatchedStationaryFiniteOracle(
            action_retentions=_oracle_action_retentions(args),
            **common,
        )
    if family == "fsrs6_oracle_continuous_stationary_finite":
        return FSRS6BatchedContinuousStationaryFiniteOracle(
            retention_min=float(args.oracle_continuous_retention_min),
            retention_max=float(args.oracle_continuous_retention_max),
            interval_chunk_size=int(args.oracle_continuous_interval_chunk_size),
            progress_log_interval_seconds=float(args.progress_log_interval_seconds),
            **common,
        )
    raise ValueError(f"Unsupported oracle family: {family}")


def _oracle_solution_points(
    *,
    family: str,
    theta_values: Sequence[float],
    user_ids: Sequence[int],
    solution: BatchedStationaryFiniteOracleSolution,
) -> list[EvaluatedPoint]:
    weight_count = len(theta_values)
    runtime_s = solution.runtime_s / float(max(1, len(user_ids) * weight_count))
    points: list[EvaluatedPoint] = []
    for user_idx, user_id in enumerate(user_ids):
        for weight_idx, cost_weight in enumerate(theta_values):
            did_converge = bool(solution.converged[user_idx][weight_idx])
            policy_ref = f"{family}:user={user_id}:lambda={format_float(cost_weight)}"
            points.append(
                _metrics_to_point(
                    user_id=user_id,
                    family=family,
                    theta=cost_weight,
                    metrics=solution.metrics[user_idx][weight_idx],
                    eval_stage="exact" if did_converge else "oracle_unconverged",
                    particles=None,
                    seed=None,
                    runtime_s=runtime_s,
                    exact=did_converge,
                    policy_ref=policy_ref,
                )
            )
    return points


@torch.inference_mode()
def evaluate_oracle_points(
    *,
    family: str,
    theta_values: Sequence[float],
    user_ids: Sequence[int],
    configs: Sequence[SingleCardFSRS6Config],
    args: argparse.Namespace,
    device: torch.device,
    progress: bool,
) -> list[EvaluatedPoint]:
    points: list[EvaluatedPoint] = []
    for batch_values in _group_chunks(theta_values, args.eval_group_batch_size):
        oracle = _build_oracle(
            args=args,
            family=family,
            configs=configs,
            device=device,
        )
        solution = oracle.solve_stationary_finite_policies(
            batch_values,
            max_iterations=int(args.oracle_stationary_finite_max_iterations),
            tolerance=float(args.oracle_stationary_finite_tolerance),
            progress=progress,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        points.extend(
            _oracle_solution_points(
                family=family,
                theta_values=batch_values,
                user_ids=user_ids,
                solution=solution,
            )
        )
        if progress:
            converged = sum(
                1 for row in solution.converged for did_converge in row if did_converge
            )
            total = sum(len(row) for row in solution.converged)
            print(
                f"exact: solved {family} lambda "
                f"{format_float(batch_values[0])}..{format_float(batch_values[-1])} "
                f"for {len(user_ids)} users converged={converged}/{total}",
                flush=True,
            )
    return points


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path,
    *,
    points: Sequence[EvaluatedPoint],
    frontier: Sequence[EvaluatedPoint],
    selected: Sequence[EvaluatedPoint],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    by_user: dict[int, list[EvaluatedPoint]] = {}
    for point in points:
        by_user.setdefault(point.user_id, []).append(point)
    for user_id, user_points in sorted(by_user.items()):
        ordered = sorted(user_points, key=lambda point: point.memory)
        ax.scatter(
            [point.memory for point in ordered],
            [point.minutes for point in ordered],
            s=22,
            alpha=0.45,
            label=f"user {user_id} points",
        )
    frontier_by_user: dict[int, list[EvaluatedPoint]] = {}
    for point in frontier:
        frontier_by_user.setdefault(point.user_id, []).append(point)
    for user_id, user_points in sorted(frontier_by_user.items()):
        ordered = sorted(user_points, key=lambda point: point.memory)
        ax.plot(
            [point.memory for point in ordered],
            [point.minutes for point in ordered],
            linewidth=1.6,
            label=f"user {user_id} frontier",
        )
    if selected:
        ax.scatter(
            [point.memory for point in selected],
            [point.minutes for point in selected],
            marker="x",
            s=64,
            color="black",
            label="selected",
        )
    ax.set_xlabel("M: card expected retrievability")
    ax.set_ylabel("T: card minutes per day")
    ax.set_title(title)
    if all(point.minutes > 0.0 for point in points):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _oracle_candidates_per_round(args: argparse.Namespace) -> int:
    if args.candidates_per_round is not None:
        return int(args.candidates_per_round)
    return int(args.candidates_per_bracket)


def _run_oracle_target_search(
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    targets: Sequence[ConstrainedTarget],
    configs: Sequence[SingleCardFSRS6Config],
    theta_grid: Sequence[float],
    theta_kind: str,
    theta_min: float,
    theta_max: float,
    device: torch.device,
    initial_points: Sequence[EvaluatedPoint],
) -> dict[str, Any]:
    current_thetas = unique_sorted(
        [*theta_grid, *[point.theta_value for point in initial_points]]
    )
    all_points = list(initial_points)
    missing_initial = _missing_thetas_for_all_users(
        current_thetas,
        all_points,
        user_ids=user_ids,
        family=args.family,
        require_exact=True,
    )
    if missing_initial:
        all_points.extend(
            evaluate_oracle_points(
                family=args.family,
                theta_values=missing_initial,
                user_ids=user_ids,
                configs=configs,
                args=args,
                device=device,
                progress=not args.no_progress,
            )
        )
    else:
        all_points = _merge_points_by_key(all_points)

    for round_index in range(1, args.max_refinement_rounds + 1):
        candidates = oracle_refinement_candidates(
            all_points,
            targets,
            family=args.family,
            theta_min=theta_min,
            theta_max=theta_max,
            existing_theta_values=current_thetas,
            max_candidates=_oracle_candidates_per_round(args),
        )
        if not candidates:
            break
        current_thetas.extend(candidates)
        if not args.no_progress:
            print(
                f"oracle_refinement_round={round_index} lambda_AB="
                f"{','.join(format_float(value) for value in candidates)}",
                flush=True,
            )
        missing_candidates = _missing_thetas_for_all_users(
            candidates,
            all_points,
            user_ids=user_ids,
            family=args.family,
            require_exact=True,
        )
        if missing_candidates:
            all_points.extend(
                evaluate_oracle_points(
                    family=args.family,
                    theta_values=missing_candidates,
                    user_ids=user_ids,
                    configs=configs,
                    args=args,
                    device=device,
                    progress=not args.no_progress,
                )
            )
        all_points = _merge_points_by_key(all_points)

    frontier = empirical_frontier([point for point in all_points if point.exact])
    segments = certify_oracle_segments(
        frontier_segments(frontier),
        all_points,
        family=args.family,
        tolerance=float(args.certificate_tolerance),
    )
    answers = target_answers(
        frontier,
        targets,
        family=args.family,
        memory_margin=args.memory_margin,
        time_margin=args.time_margin,
        certified=False,
    )
    answers = apply_target_certifications(
        answers,
        target_certification_map(segments, targets, family=args.family),
    )
    selected = [answer.point for answer in answers if answer.point is not None]

    return _write_outputs(
        args=args,
        user_ids=user_ids,
        points=all_points,
        frontier=frontier,
        answers=answers,
        segments=segments,
        selected=[point for point in selected if point is not None],
        theta_grid=theta_grid,
        theta_values_confirmed=unique_sorted(current_thetas),
        theta_kind=theta_kind,
        theta_min=theta_min,
        theta_max=theta_max,
        device=device,
        extra_metadata={
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_stationary_finite_max_iterations": (
                args.oracle_stationary_finite_max_iterations
            ),
            "oracle_stationary_finite_tolerance": (
                args.oracle_stationary_finite_tolerance
            ),
            "certificate_tolerance": args.certificate_tolerance,
            "certified_segments": sum(1 for segment in segments if segment.certified),
            "segment_count": len(segments),
        },
    )


def _write_outputs(
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    points: Sequence[EvaluatedPoint],
    frontier: Sequence[EvaluatedPoint],
    answers: Sequence[Any],
    segments: Sequence[Any],
    selected: Sequence[EvaluatedPoint],
    theta_grid: Sequence[float],
    theta_values_confirmed: Sequence[float],
    theta_kind: str,
    theta_min: float,
    theta_max: float,
    device: torch.device,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    points_path = args.out_dir / "points.csv"
    frontier_path = args.out_dir / "frontier.csv"
    answers_path = args.out_dir / "target_answers.csv"
    segments_path = args.out_dir / "segments.csv"
    metadata_path = args.out_dir / "metadata.json"
    plot_path = args.out_dir / "target_search.png"
    oracle_family = _is_oracle_family(args.family)
    _write_csv(points_path, [point_row(point) for point in points])
    _write_csv(frontier_path, [point_row(point) for point in frontier])
    _write_csv(answers_path, [answer_row(answer) for answer in answers])
    _write_csv(segments_path, [segment_row(segment) for segment in segments])
    metadata = {
        "family": args.family,
        "environment": args.env,
        "user_ids": list(user_ids),
        "target_memories": _parse_csv_floats(
            args.target_memories,
            name="--target-memories",
        ),
        "target_times": _parse_csv_floats(args.target_times, name="--target-times"),
        "theta_grid_initial": list(theta_grid),
        "theta_values_confirmed": list(theta_values_confirmed),
        "theta_kind": theta_kind,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "days": args.days,
        "explore_particles": None if oracle_family else args.explore_particles,
        "confirm_particles": None if oracle_family else _confirm_particles(args),
        "seed": args.seed,
        "device": str(device),
        "memory_margin": args.memory_margin,
        "time_margin": args.time_margin,
        "certification_scope": "oracle_target_local"
        if oracle_family
        else "family_constrained_rollout",
        "globally_certified": False,
        "oracle_certificates_available": oracle_family,
        "family_constrained": not oracle_family,
        "init_points": [str(path) for path in args.init_points],
        "deterministic_only": True,
        "outputs": {
            "points": str(points_path),
            "frontier": str(frontier_path),
            "target_answers": str(answers_path),
            "segments": str(segments_path),
            "plot": None if args.no_plot else str(plot_path),
        },
        "feasible_targets": sum(1 for answer in answers if answer.feasible),
        "target_count": len(answers),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        _write_plot(
            plot_path,
            points=points,
            frontier=frontier,
            selected=selected,
            title=f"{args.family} constrained target search",
        )

    print(f"Wrote points: {points_path}")
    print(f"Wrote frontier: {frontier_path}")
    print(f"Wrote target answers: {answers_path}")
    print(f"Wrote segments: {segments_path}")
    print(f"Wrote metadata: {metadata_path}")
    if not args.no_plot:
        print(f"Wrote plot: {plot_path}")
    return metadata


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    user_ids = _parse_user_ids(args)
    targets = _targets(args, user_ids)
    theta_grid, theta_kind, theta_min, theta_max = _theta_grid(args)
    confirm_particles = _confirm_particles(args)
    device = resolve_torch_device(args.torch_device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    configs = _load_user_configs(args, user_ids)
    initial_points = _load_initial_points(
        args=args,
        user_ids=user_ids,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    if _is_oracle_family(args.family):
        return _run_oracle_target_search(
            args=args,
            user_ids=user_ids,
            targets=targets,
            configs=configs,
            theta_grid=theta_grid,
            theta_kind=theta_kind,
            theta_min=theta_min,
            theta_max=theta_max,
            device=device,
            initial_points=initial_points,
        )

    missing_initial = _missing_thetas_for_all_users(
        theta_grid,
        initial_points,
        user_ids=user_ids,
        family=args.family,
        require_exact=False,
    )
    all_explore_points = list(initial_points)
    if missing_initial:
        all_explore_points.extend(
            evaluate_family_points(
                family=args.family,
                theta_values=missing_initial,
                user_ids=user_ids,
                configs=configs,
                days=args.days,
                particles=args.explore_particles,
                group_batch_size=args.eval_group_batch_size,
                seed=args.seed + 10_000,
                device=device,
                eval_stage="explore",
                progress=not args.no_progress,
            )
        )
    all_explore_points = _merge_points_by_key(all_explore_points)
    current_thetas = unique_sorted(
        [*theta_grid, *[point.theta_value for point in initial_points]]
    )
    for round_index in range(1, args.max_refinement_rounds + 1):
        candidates = adaptive_theta_candidates(
            all_explore_points,
            targets,
            family=args.family,
            theta_kind=theta_kind,
            theta_min=theta_min,
            theta_max=theta_max,
            candidates_per_bracket=args.candidates_per_bracket,
        )
        candidates = missing_values_from_values(current_thetas, candidates)
        if not candidates:
            break
        current_thetas.extend(candidates)
        if not args.no_progress:
            print(
                f"refinement_round={round_index} candidates="
                f"{','.join(format_float(value) for value in candidates)}",
                flush=True,
            )
        all_explore_points.extend(
            evaluate_family_points(
                family=args.family,
                theta_values=candidates,
                user_ids=user_ids,
                configs=configs,
                days=args.days,
                particles=args.explore_particles,
                group_batch_size=args.eval_group_batch_size,
                seed=args.seed + 20_000 + round_index * 1_000,
                device=device,
                eval_stage="explore",
                progress=not args.no_progress,
            )
        )

    confirm_thetas = unique_sorted(current_thetas)
    confirmed_points = evaluate_family_points(
        family=args.family,
        theta_values=confirm_thetas,
        user_ids=user_ids,
        configs=configs,
        days=args.days,
        particles=confirm_particles,
        group_batch_size=args.eval_group_batch_size,
        seed=args.seed + 90_000,
        device=device,
        eval_stage="confirmed",
        progress=not args.no_progress,
    )

    frontier = empirical_frontier(confirmed_points)
    answers = target_answers(
        confirmed_points,
        targets,
        family=args.family,
        memory_margin=args.memory_margin,
        time_margin=args.time_margin,
        certified=False,
    )
    segments = frontier_segments(frontier)
    selected = [answer.point for answer in answers if answer.point is not None]

    return _write_outputs(
        args=args,
        user_ids=user_ids,
        points=_merge_points_by_key([*all_explore_points, *confirmed_points]),
        frontier=frontier,
        answers=answers,
        segments=segments,
        selected=[point for point in selected if point is not None],
        theta_grid=theta_grid,
        theta_values_confirmed=confirm_thetas,
        theta_kind=theta_kind,
        theta_min=theta_min,
        theta_max=theta_max,
        device=device,
    )


def main() -> None:
    run_search(parse_args())


if __name__ == "__main__":
    main()
