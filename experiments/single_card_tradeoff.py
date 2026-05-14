from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
import csv
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import simulate as simulate_cli
from experiments.retention_sweep.cli_utils import (
    add_benchmark_args,
    add_fuzz_arg,
    add_retention_range_args,
    add_torch_device_arg,
    parse_csv,
)
from simulator import simulate as simulate_event
from simulator.behavior import StochasticBehavior
from simulator.button_usage import load_button_usage_config, normalize_button_usage
from simulator.core import new_first_priority
from simulator.cost import StatefulCostModel, StateRatingCosts
from simulator.defaults import (
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
)
from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.retention_sweep.grid import dr_values
from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.schedulers.fixed import FixedBatchSchedulerOps
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.vectorized import simulate as simulate_vectorized
from simulator.vectorized.multiuser_engine import simulate_multiuser
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost

DEFAULT_FIXED_INTERVALS = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
DEFAULT_TARGET_RETENTIONS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.93,
    0.96,
    0.98,
    0.99,
]
DEFAULT_UVFA_PPO_POLICY = Path("logs/single_card_tradeoff/uvfa_ppo_policy.pt")
UVFA_PPO_SCHEDULER = "uvfa_ppo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an iid single-card lifecycle tradeoff experiment without daily "
            "study-budget constraints."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Single-card lifecycle length in days.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=10_000,
        help=(
            "Monte Carlo particles for the single-card lifecycle. Each particle "
            "is one independent card learned on day 0."
        ),
    )
    parser.add_argument(
        "--deck-scale",
        type=int,
        default=DEFAULT_DECK_SIZE,
        help="Scale single-card metrics by this many iid cards in the CSV output.",
    )
    parser.add_argument(
        "--env",
        default="fsrs6_default",
        help="Comma-separated environments. Defaults avoid external benchmark data.",
    )
    parser.add_argument(
        "--sched",
        default="fsrs6_default",
        help=(
            "Comma-separated schedulers. Desired-retention schedulers are swept; "
            "use fixed@<days> for fixed intervals."
        ),
    )
    add_retention_range_args(parser)
    parser.add_argument(
        "--target-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help=(
            "Comma-separated desired-retention targets for DR schedulers. "
            "Pass an empty string to use --start-retention/--end-retention/--step."
        ),
    )
    parser.add_argument(
        "--fixed-intervals",
        default=None,
        help=(
            "Comma-separated fixed intervals to run when --sched contains plain "
            "'fixed'. Defaults to 8,16,32,64,128,256,512. "
            "Ignored for fixed@<days> specs."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-policy",
        type=Path,
        default=DEFAULT_UVFA_PPO_POLICY,
        help=(
            "Path to a UVFA PPO policy checkpoint when --sched contains uvfa_ppo. "
            "Create one with experiments/uvfa_ppo_single_card.py."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-cost-weights",
        default=None,
        help=(
            "Comma-separated scalarization weights for uvfa_ppo. Defaults to "
            "the cost_weights saved in --uvfa-ppo-policy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed reused for every point in the sweep.",
    )
    parser.add_argument(
        "--scheduler-priority",
        default=DEFAULT_SCHEDULER_PRIORITY,
        help="FSRS6 priority hint passed to the scheduler.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Load benchmark weights and button usage for this user ID when requested.",
    )
    add_benchmark_args(parser)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=None,
        help=(
            "Optional Anki button usage JSONL. If omitted, built-in rating and "
            "cost defaults are used."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["vectorized", "event"],
        default="vectorized",
        help=(
            "Simulation engine. Vectorized treats particles as iid card samples; "
            "event is mainly for debugging small particle counts."
        ),
    )
    add_torch_device_arg(parser)
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=0,
        help=(
            "Desired-retention targets per vectorized batch for supported FSRS6 "
            "runs. 0 batches all targets together; 1 disables target batching."
        ),
    )
    add_fuzz_arg(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/single_card_tradeoff/results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Plot output path. Defaults to the CSV path with .png suffix.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the Pareto-style PNG plot.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm simulation progress bars.",
    )
    return parser.parse_args()


def _fixed_intervals(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_FIXED_INTERVALS)
    intervals: list[float] = []
    for item in parse_csv(value):
        try:
            interval = float(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid fixed interval '{item}'.") from exc
        if interval <= 0.0:
            raise SystemExit("Fixed intervals must be > 0.")
        intervals.append(interval)
    if not intervals:
        raise SystemExit("--fixed-intervals must include at least one value.")
    return intervals


def _parse_float_list(value: str, *, label: str) -> list[float]:
    values: list[float] = []
    for item in parse_csv(value):
        try:
            parsed = float(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {label} '{item}'.") from exc
        if not math.isfinite(parsed):
            raise SystemExit(f"{label} values must be finite.")
        values.append(parsed)
    if not values:
        raise SystemExit(f"{label} must include at least one value.")
    return values


def _run_specs(args: argparse.Namespace) -> list[tuple[str, str, float | None]]:
    specs: list[tuple[str, str, float | None]] = []
    for raw in parse_csv(args.sched) or ["fsrs6_default"]:
        try:
            name, fixed_interval, raw_spec = parse_scheduler_spec(raw)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if name not in simulate_cli.SCHEDULER_FACTORIES and name != UVFA_PPO_SCHEDULER:
            raise SystemExit(f"Unknown scheduler '{name}'.")
        if name == "fixed" and fixed_interval is None:
            for interval in _fixed_intervals(args.fixed_intervals):
                specs.append((name, f"fixed@{format_float(interval)}", interval))
            continue
        specs.append((name, raw_spec, fixed_interval))
    return specs


def _retention_grid(args: argparse.Namespace) -> list[float]:
    target_retention_arg = getattr(args, "target_retentions", None)
    if target_retention_arg is not None and target_retention_arg.strip():
        values: list[float] = []
        for item in parse_csv(target_retention_arg):
            try:
                value = round(float(item), 2)
            except ValueError as exc:
                raise SystemExit(f"Invalid target retention '{item}'.") from exc
            if not (0.0 < value < 1.0):
                raise SystemExit("Target retentions must be within (0, 1).")
            values.append(value)
        if not values:
            raise SystemExit("--target-retentions must include at least one value.")
        return values
    try:
        return dr_values(args.start_retention, args.end_retention, args.step)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _make_behavior(
    args: argparse.Namespace,
) -> tuple[StochasticBehavior, StatefulCostModel]:
    button_usage = (
        load_button_usage_config(args.button_usage, args.user_id or 1)
        if args.button_usage is not None
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
        review_markov_transition=usage.get("long_term_transition"),
    )
    cost_model = StatefulCostModel(
        state_costs=StateRatingCosts(
            learning=usage["learn_costs"],
            review=usage["review_costs"],
            relearning=usage["review_costs"],
        )
    )
    return behavior, cost_model


def _row_from_stats(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    fixed_interval: float | None,
    desired_retention: float | None,
    seed: int,
    stats: Any,
    runtime_s: float,
) -> dict[str, Any]:
    particle_count = float(args.particles)
    day_count = float(args.days)
    card_minutes_per_day = stats.total_cost / day_count / 60.0 / particle_count
    card_expected_retrievability = (
        sum(stats.daily_memorized) / day_count / particle_count
    )
    card_reviews_per_day = sum(stats.daily_reviews) / day_count / particle_count
    card_total_reviews = stats.total_reviews / particle_count
    card_total_lapses = stats.total_lapses / particle_count
    observed_retention = (
        1.0 - stats.total_lapses / stats.total_reviews
        if stats.total_reviews > 0
        else None
    )
    deck_scale = float(args.deck_scale)
    return {
        "environment": environment_name,
        "scheduler": scheduler_name,
        "scheduler_spec": scheduler_spec,
        "desired_retention": desired_retention,
        "fixed_interval": fixed_interval,
        "goal_cost_weight": None,
        "seed": seed,
        "days": args.days,
        "particles": args.particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": card_expected_retrievability,
        "card_minutes_per_day": card_minutes_per_day,
        "card_reviews_per_day": card_reviews_per_day,
        "card_total_reviews": card_total_reviews,
        "card_total_lapses": card_total_lapses,
        "card_total_cost_seconds": stats.total_cost / particle_count,
        "card_final_projected_retrievability": (
            stats.total_projected_retrievability / particle_count
        ),
        "observed_retention": observed_retention,
        "deck_expected_memorized": card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": card_reviews_per_day * deck_scale,
        "total_reviews": stats.total_reviews,
        "total_lapses": stats.total_lapses,
        "total_cost_seconds": stats.total_cost,
        "runtime_s": runtime_s,
        "engine": args.engine,
        "fuzz": bool(args.fuzz),
    }


def _run_point(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    fixed_interval: float | None,
    desired_retention: float | None,
    seed: int,
) -> dict[str, Any]:
    run_args = argparse.Namespace(**vars(args))
    run_args.env = environment_name
    run_args.environment = environment_name
    run_args.scheduler = scheduler_name
    run_args.sched = scheduler_spec
    run_args.scheduler_spec = scheduler_spec
    run_args.fixed_interval = fixed_interval
    run_args.desired_retention = desired_retention
    run_args.short_term_source = None
    run_args.short_term = False
    run_args.sspmmc_policy = None
    run_args.lstm_interval_mode = "integer"
    run_args.lstm_min_interval = 1.0

    env = simulate_cli.ENVIRONMENT_FACTORIES[environment_name](run_args)
    scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](run_args)
    behavior, cost_model = _make_behavior(args)

    start = time.perf_counter()
    if args.engine == "vectorized":
        stats = simulate_vectorized(
            days=args.days,
            deck_size=args.particles,
            environment=env,
            scheduler=scheduler,
            behavior=behavior,
            cost_model=cost_model,
            seed=seed,
            device=args.torch_device,
            fuzz=args.fuzz,
            progress=not args.no_progress,
        )
    else:
        rng = random.Random(seed)
        stats = simulate_event(
            days=args.days,
            deck_size=args.particles,
            environment=env,
            scheduler=scheduler,
            behavior=behavior,
            cost_model=cost_model,
            seed_fn=rng.random,
            fuzz=args.fuzz,
            progress=not args.no_progress,
        )
    runtime_s = time.perf_counter() - start

    return _row_from_stats(
        args,
        environment_name=environment_name,
        scheduler_name=scheduler_name,
        scheduler_spec=scheduler_spec,
        fixed_interval=fixed_interval,
        desired_retention=desired_retention,
        seed=seed,
        stats=stats,
        runtime_s=runtime_s,
    )


def _chunks(values: Sequence[float], size: int) -> list[list[float]]:
    if size <= 0:
        return [list(values)]
    return [list(values[idx : idx + size]) for idx in range(0, len(values), size)]


def _target_batch_supported(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    desired_values: Sequence[float | None],
) -> bool:
    return (
        args.engine == "vectorized"
        and args.target_batch_size != 1
        and environment_name in {"fsrs6", "fsrs6_default"}
        and scheduler_name in {"fsrs6", "fsrs6_default"}
        and len(desired_values) > 1
        and all(value is not None for value in desired_values)
    )


def _fixed_batch_supported(
    args: argparse.Namespace,
    *,
    environment_name: str,
    fixed_specs: Sequence[tuple[str, float]],
) -> bool:
    return (
        args.engine == "vectorized"
        and args.target_batch_size != 1
        and environment_name in {"fsrs6", "fsrs6_default"}
        and len(fixed_specs) > 1
    )


def _fsrs6_weights(obj: Any, *, label: str) -> tuple[float, ...]:
    params = getattr(obj, "params", None)
    weights = getattr(params, "weights", None)
    if weights is None or len(weights) != 21:
        raise ValueError(f"{label} must expose FSRS6 params.weights.")
    return tuple(float(weight) for weight in weights)


def _repeat_rows(
    values: Sequence[float] | Sequence[Sequence[float]],
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.tensor(values, device=device, dtype=dtype)
    return tensor.unsqueeze(0).repeat(rows, *([1] * tensor.ndim))


def _make_multiuser_behavior_cost(
    args: argparse.Namespace,
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    behavior, cost_model = _make_behavior(args)
    markov_success = behavior.review_markov_success
    markov_tensor = (
        _repeat_rows(markov_success, rows=rows, device=device, dtype=dtype)
        if markov_success is not None
        else None
    )
    state_costs = cost_model.state_costs
    multi_behavior = MultiUserBehavior(
        attendance_prob=torch.full((rows,), 1.0, device=device, dtype=dtype),
        lazy_good_bias=torch.zeros(rows, device=device, dtype=dtype),
        max_new_per_day=torch.full(
            (rows,), args.particles, device=device, dtype=torch.int64
        ),
        max_reviews_per_day=torch.full(
            (rows,), args.particles, device=device, dtype=torch.int64
        ),
        max_cost_per_day=torch.full((rows,), math.inf, device=device, dtype=dtype),
        success_weights=_repeat_rows(
            behavior.success_dist.success_weights,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        learning_success_weights=_repeat_rows(
            behavior.learning_success_dist.success_weights,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        relearning_success_weights=_repeat_rows(
            behavior.relearning_success_dist.success_weights,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        first_rating_prob=_repeat_rows(
            behavior.first_rating_prob,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        review_markov_success_weights=markov_tensor,
    )
    multi_cost = MultiUserCost(
        base=torch.zeros(rows, device=device, dtype=dtype),
        penalty=torch.zeros(rows, device=device, dtype=dtype),
        learn_costs=_repeat_rows(
            state_costs.learning,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        review_costs=_repeat_rows(
            state_costs.review,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        learning_review_costs=_repeat_rows(
            state_costs.learning,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
        relearning_review_costs=_repeat_rows(
            state_costs.relearning,
            rows=rows,
            device=device,
            dtype=dtype,
        ),
    )
    return multi_behavior, multi_cost


def _run_target_batch(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    desired_values: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    run_args = argparse.Namespace(**vars(args))
    run_args.env = environment_name
    run_args.environment = environment_name
    run_args.scheduler = scheduler_name
    run_args.sched = scheduler_spec
    run_args.scheduler_spec = scheduler_spec
    run_args.fixed_interval = None
    run_args.desired_retention = float(desired_values[0])
    run_args.short_term_source = None
    run_args.short_term = False
    run_args.sspmmc_policy = None
    run_args.lstm_interval_mode = "integer"
    run_args.lstm_min_interval = 1.0

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    dtype = torch.float64
    target_count = len(desired_values)

    env = simulate_cli.ENVIRONMENT_FACTORIES[environment_name](run_args)
    scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](run_args)
    env_weights = torch.tensor(
        [_fsrs6_weights(env, label=environment_name) for _ in range(target_count)],
        device=device,
        dtype=dtype,
    )
    sched_weights = torch.tensor(
        [_fsrs6_weights(scheduler, label=scheduler_name) for _ in range(target_count)],
        device=device,
        dtype=dtype,
    )
    env_ops = FSRS6BatchEnvOps(
        weights=env_weights,
        bounds=Bounds(),
        device=device,
        dtype=dtype,
    )
    sched_ops = FSRS6BatchSchedulerOps(
        weights=sched_weights,
        desired_retention=torch.tensor(desired_values, device=device, dtype=dtype),
        bounds=Bounds(),
        priority_mode=args.scheduler_priority,
        device=device,
        dtype=dtype,
    )
    behavior, cost_model = _make_multiuser_behavior_cost(
        args, rows=target_count, device=device, dtype=dtype
    )

    start = time.perf_counter()
    stats_by_target = simulate_multiuser(
        days=args.days,
        deck_size=args.particles,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        priority_mode="new-first",
        seed=seed,
        device=device,
        dtype=dtype,
        fuzz=args.fuzz,
        progress=not args.no_progress,
        progress_label=f"{environment_name}/{scheduler_spec} targets",
    )
    runtime_per_target = (time.perf_counter() - start) / max(1, target_count)
    return [
        _row_from_stats(
            args,
            environment_name=environment_name,
            scheduler_name=scheduler_name,
            scheduler_spec=scheduler_spec,
            fixed_interval=None,
            desired_retention=desired_retention,
            seed=seed,
            stats=stats,
            runtime_s=runtime_per_target,
        )
        for desired_retention, stats in zip(desired_values, stats_by_target)
    ]


def _run_fixed_batch(
    args: argparse.Namespace,
    *,
    environment_name: str,
    fixed_specs: Sequence[tuple[str, float]],
    seed: int,
) -> list[dict[str, Any]]:
    run_args = argparse.Namespace(**vars(args))
    run_args.env = environment_name
    run_args.environment = environment_name
    run_args.scheduler = "fixed"
    run_args.sched = fixed_specs[0][0]
    run_args.scheduler_spec = fixed_specs[0][0]
    run_args.fixed_interval = fixed_specs[0][1]
    run_args.desired_retention = None
    run_args.short_term_source = None
    run_args.short_term = False
    run_args.sspmmc_policy = None
    run_args.lstm_interval_mode = "integer"
    run_args.lstm_min_interval = 1.0

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    dtype = torch.float64
    row_count = len(fixed_specs)
    intervals = [interval for _, interval in fixed_specs]

    env = simulate_cli.ENVIRONMENT_FACTORIES[environment_name](run_args)
    env_weights = torch.tensor(
        [_fsrs6_weights(env, label=environment_name) for _ in range(row_count)],
        device=device,
        dtype=dtype,
    )
    env_ops = FSRS6BatchEnvOps(
        weights=env_weights,
        bounds=Bounds(),
        device=device,
        dtype=dtype,
    )
    sched_ops = FixedBatchSchedulerOps(
        interval=torch.tensor(intervals, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    behavior, cost_model = _make_multiuser_behavior_cost(
        args, rows=row_count, device=device, dtype=dtype
    )

    start = time.perf_counter()
    stats_by_interval = simulate_multiuser(
        days=args.days,
        deck_size=args.particles,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        priority_mode="new-first",
        seed=seed,
        device=device,
        dtype=dtype,
        fuzz=args.fuzz,
        progress=not args.no_progress,
        progress_label=f"{environment_name}/fixed intervals",
    )
    runtime_per_interval = (time.perf_counter() - start) / max(1, row_count)
    return [
        _row_from_stats(
            args,
            environment_name=environment_name,
            scheduler_name="fixed",
            scheduler_spec=scheduler_spec,
            fixed_interval=interval,
            desired_retention=None,
            seed=seed,
            stats=stats,
            runtime_s=runtime_per_interval,
        )
        for (scheduler_spec, interval), stats in zip(fixed_specs, stats_by_interval)
    ]


def _load_uvfa_ppo_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float]:
    if not args.uvfa_ppo_policy.exists():
        raise SystemExit(
            "UVFA PPO policy not found: "
            f"{args.uvfa_ppo_policy}. Train one with "
            "`uv run experiments/uvfa_ppo_single_card.py --model-out "
            f"{args.uvfa_ppo_policy}` or pass --uvfa-ppo-policy."
        )

    from experiments.uvfa_ppo_single_card import PolicyValueNet

    checkpoint = torch.load(args.uvfa_ppo_policy, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Invalid UVFA PPO checkpoint: {args.uvfa_ppo_policy}")
    raw_actions = checkpoint.get("action_retentions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise SystemExit("UVFA PPO checkpoint is missing action_retentions.")
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit("UVFA PPO checkpoint is missing cost_weights.")
    action_retentions = [float(value) for value in raw_actions]
    policy_cost_weights = [float(value) for value in raw_cost_weights]
    obs_dim = int(checkpoint.get("obs_dim", 7))
    hidden_size = int(checkpoint.get("hidden_size", 96))
    model = PolicyValueNet(
        obs_dim=obs_dim,
        action_count=len(action_retentions),
        hidden_size=hidden_size,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit("UVFA PPO checkpoint is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, action_retentions, policy_cost_weights, max(policy_cost_weights)


def _uvfa_ppo_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "uvfa_ppo_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="UVFA PPO cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("UVFA PPO cost weights must be >= 0.")
    return values


def _row_from_uvfa_metrics(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    goal_cost_weight: float,
    seed: int,
    metrics: Any,
    runtime_s: float,
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    total_reviews = metrics.card_total_reviews * args.particles
    total_lapses = metrics.card_total_lapses * args.particles
    total_cost_seconds = metrics.card_total_cost_seconds * args.particles
    return {
        "environment": environment_name,
        "scheduler": UVFA_PPO_SCHEDULER,
        "scheduler_spec": scheduler_spec,
        "desired_retention": None,
        "fixed_interval": None,
        "goal_cost_weight": goal_cost_weight,
        "seed": seed,
        "days": args.days,
        "particles": args.particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "card_final_projected_retrievability": None,
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "total_reviews": total_reviews,
        "total_lapses": total_lapses,
        "total_cost_seconds": total_cost_seconds,
        "runtime_s": runtime_s,
        "engine": UVFA_PPO_SCHEDULER,
        "fuzz": False,
    }


def _run_uvfa_ppo(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name != "fsrs6_default":
        raise SystemExit("uvfa_ppo currently supports only --env fsrs6_default.")
    if args.engine != "vectorized":
        raise SystemExit("uvfa_ppo is supported only with --engine vectorized.")
    if args.fuzz:
        raise SystemExit("uvfa_ppo does not support --fuzz.")

    from experiments.uvfa_ppo_single_card import evaluate_policy

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    model, action_retentions, policy_cost_weights, goal_norm_max = (
        _load_uvfa_ppo_policy(
            args,
            device=device,
        )
    )
    cost_weights = _uvfa_ppo_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
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
            particles=args.particles,
            seed=seed + 30_000 + int(round(cost_weight * 10.0)),
            goal_norm_max=goal_norm_max,
        )
        runtime_s = time.perf_counter() - start
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "scheduler",
        "scheduler_spec",
        "desired_retention",
        "fixed_interval",
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
        "card_final_projected_retrievability",
        "observed_retention",
        "deck_expected_memorized",
        "deck_minutes_per_day",
        "deck_reviews_per_day",
        "total_reviews",
        "total_lapses",
        "total_cost_seconds",
        "runtime_s",
        "engine",
        "fuzz",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _point_label(row: dict[str, Any]) -> str:
    goal_cost_weight = row.get("goal_cost_weight")
    if goal_cost_weight is not None and goal_cost_weight != "":
        return f"w={format_float(float(goal_cost_weight))}"
    desired_retention = row["desired_retention"]
    if desired_retention is not None:
        return format_float(float(desired_retention))
    fixed_interval = row["fixed_interval"]
    if fixed_interval is not None:
        return f"{format_float(float(fixed_interval))}d"
    return row["scheduler_spec"]


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _plot_group_key(row: dict[str, Any]) -> tuple[str, str]:
    scheduler = str(row["scheduler"])
    scheduler_label = "fixed" if scheduler == "fixed" else str(row["scheduler_spec"])
    return str(row["environment"]), scheduler_label


def _plot_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    goal_cost_weight = row.get("goal_cost_weight")
    if goal_cost_weight is not None and goal_cost_weight != "":
        return 0.5, float(goal_cost_weight)
    fixed_interval = row["fixed_interval"]
    if fixed_interval is not None:
        return 1.0, float(fixed_interval)
    desired_retention = row["desired_retention"]
    if desired_retention is not None:
        return 0.0, float(desired_retention)
    return 2.0, 0.0


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = _plot_group_key(row)
        groups.setdefault(key, []).append(row)

    frontier = _pareto_frontier(rows)
    frontier_ids = {id(row) for row in frontier}
    label_rows: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    for (environment, scheduler_label), group in groups.items():
        group = sorted(group, key=_plot_sort_key)
        x = [row["deck_expected_memorized"] for row in group]
        y = [row["deck_minutes_per_day"] for row in group]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.0,
            alpha=0.45,
            label=f"{environment}/{scheduler_label}",
        )
        for row in group:
            if row["scheduler"] == "fixed" or id(row) in frontier_ids:
                label_rows.append(row)

    if frontier:
        ax.plot(
            [row["deck_expected_memorized"] for row in frontier],
            [row["deck_minutes_per_day"] for row in frontier],
            color="black",
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=f"Pareto frontier ({len(frontier)} points)",
        )

    ax.margins(x=0.04, y=0.08)
    texts = []
    label_x = []
    label_y = []
    for row in label_rows:
        x = float(row["deck_expected_memorized"])
        y = float(row["deck_minutes_per_day"])
        label_x.append(x)
        label_y.append(y)
        texts.append(
            ax.text(
                x,
                y,
                _point_label(row),
                fontsize=8,
                zorder=6,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 0.4,
                },
            )
        )
    if texts:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                x=label_x,
                y=label_y,
                target_x=label_x,
                target_y=label_y,
                ax=ax,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.45",
                    "lw": 0.5,
                    "alpha": 0.75,
                    "shrinkA": 3,
                    "shrinkB": 2,
                },
                force_text=(0.35, 0.5),
                force_static=(0.2, 0.35),
                expand=(1.08, 1.2),
                ensure_inside_axes=True,
            )
        except ImportError:
            pass

    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title("Single-card lifecycle Pareto frontier")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        target = _point_label(row)
        print(
            " ".join(
                [
                    f"{row['environment']}/{row['scheduler_spec']}",
                    f"target={target}",
                    f"card_mem={row['card_expected_retrievability']:.4f}",
                    f"card_min/day={row['card_minutes_per_day']:.6f}",
                    f"deck_mem={row['deck_expected_memorized']:.1f}",
                    f"deck_min/day={row['deck_minutes_per_day']:.2f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise SystemExit("--days must be > 0.")
    if args.particles <= 0:
        raise SystemExit("--particles must be > 0.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.target_batch_size < 0:
        raise SystemExit("--target-batch-size must be >= 0.")

    environments = parse_csv(args.env) or ["fsrs6_default"]
    for environment in environments:
        if environment not in simulate_cli.ENVIRONMENT_FACTORIES:
            raise SystemExit(f"Unknown environment '{environment}'.")
    scheduler_specs = _run_specs(args)
    retention_values = _retention_grid(args)

    rows: list[dict[str, Any]] = []
    for environment in environments:
        fixed_specs = [
            (scheduler_spec, normalize_fixed_interval(fixed_interval))
            for scheduler_name, scheduler_spec, fixed_interval in scheduler_specs
            if scheduler_name == "fixed"
        ]
        batched_fixed_specs: set[str] = set()
        if _fixed_batch_supported(
            args, environment_name=environment, fixed_specs=fixed_specs
        ):
            rows.extend(
                _run_fixed_batch(
                    args,
                    environment_name=environment,
                    fixed_specs=fixed_specs,
                    seed=args.seed,
                )
            )
            batched_fixed_specs = {scheduler_spec for scheduler_spec, _ in fixed_specs}

        for scheduler_name, scheduler_spec, fixed_interval in scheduler_specs:
            if scheduler_name == "fixed" and scheduler_spec in batched_fixed_specs:
                continue
            if scheduler_name == UVFA_PPO_SCHEDULER:
                rows.extend(
                    _run_uvfa_ppo(
                        args,
                        environment_name=environment,
                        scheduler_spec=scheduler_spec,
                        seed=args.seed,
                    )
                )
                continue
            desired_values: Sequence[float | None]
            if scheduler_uses_desired_retention(scheduler_name):
                desired_values = retention_values
            else:
                desired_values = [None]
            if _target_batch_supported(
                args,
                environment_name=environment,
                scheduler_name=scheduler_name,
                desired_values=desired_values,
            ):
                target_values: list[float] = []
                for value in desired_values:
                    if value is None:
                        raise AssertionError("Target-batched runs require DR values.")
                    target_values.append(float(value))
                for chunk in _chunks(target_values, args.target_batch_size):
                    rows.extend(
                        _run_target_batch(
                            args,
                            environment_name=environment,
                            scheduler_name=scheduler_name,
                            scheduler_spec=scheduler_spec,
                            desired_values=chunk,
                            seed=args.seed,
                        )
                    )
                continue
            for desired_retention in desired_values:
                rows.append(
                    _run_point(
                        args,
                        environment_name=environment,
                        scheduler_name=scheduler_name,
                        scheduler_spec=scheduler_spec,
                        fixed_interval=normalize_fixed_interval(fixed_interval)
                        if scheduler_name == "fixed"
                        else None,
                        desired_retention=desired_retention,
                        seed=args.seed,
                    )
                )

    _write_csv(args.out, rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        _write_plot(plot_path, rows)
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {args.out}")
    _print_summary(rows)


if __name__ == "__main__":
    main()
