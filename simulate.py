from __future__ import annotations

import argparse
import math
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from simulator import simulate
from simulator.behavior import StochasticBehavior
from simulator.button_usage import (
    DEFAULT_BUTTON_USAGE_PATH,
    load_button_usage_config,
    normalize_button_usage,
)
from simulator.cost import StatefulCostModel, StateRatingCosts
from simulator.benchmark_loader import load_benchmark_weights, parse_result_overrides
from simulator.models import FSRS3Model, FSRS6Model, LSTMModel
from simulator.schedulers import (
    FSRS3Scheduler,
    FSRS6Scheduler,
    HLRScheduler,
    DASHScheduler,
    LSTMScheduler,
    FixedIntervalScheduler,
    AnkiSM2Scheduler,
    MemriseScheduler,
    SSPMMCScheduler,
)
from simulator.core import Action, Event, new_first_priority, review_first_priority
from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.vectorized import simulate as simulate_vectorized
from simulator.short_term import ShortTermScheduler


def _resolve_benchmark_weights(
    args, environment: str, expected_len: int
) -> tuple[float, ...] | None:
    overrides = parse_result_overrides(args.benchmark_result)
    weights = load_benchmark_weights(
        repo_root=Path(__file__).resolve().parent,
        benchmark_root=args.srs_benchmark_root,
        environment=environment,
        user_id=args.user_id or 1,
        partition_key=args.benchmark_partition,
        overrides=overrides,
    )
    if len(weights) != expected_len:
        raise ValueError(
            f"{environment} expects {expected_len} weights, got {len(weights)}."
        )
    return tuple(float(x) for x in weights)


def _parse_steps(value: str | None) -> list[float]:
    if not value:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def _resolve_short_term_config(
    args: argparse.Namespace,
) -> tuple[str | None, list[float], list[float]]:
    short_term_source = getattr(args, "short_term_source", None)
    legacy_short_term = bool(getattr(args, "short_term", False))
    if legacy_short_term:
        if short_term_source and short_term_source != "steps":
            raise SystemExit(
                "Use --short-term-source=steps or drop --short-term "
                "to avoid conflicting short-term options."
            )
        short_term_source = short_term_source or "steps"
    learning_raw = getattr(args, "learning_steps", None)
    relearning_raw = getattr(args, "relearning_steps", None)
    if short_term_source == "steps":
        if learning_raw is None:
            learning_raw = "1,10"
        if relearning_raw is None:
            relearning_raw = "10"
    learning_steps = _parse_steps(learning_raw)
    relearning_steps = _parse_steps(relearning_raw)
    if short_term_source is None:
        if learning_steps or relearning_steps:
            raise SystemExit("Learning steps require --short-term-source=steps.")
        return None, learning_steps, relearning_steps
    if short_term_source == "scheduler" and (learning_steps or relearning_steps):
        raise SystemExit(
            "--short-term-source=scheduler cannot be combined with "
            "--learning-steps or --relearning-steps."
        )
    return short_term_source, learning_steps, relearning_steps


def _lstm_interval_mode(args: argparse.Namespace) -> str:
    return getattr(args, "lstm_interval_mode", None) or "integer"


def _lstm_min_interval(args: argparse.Namespace) -> float:
    value = getattr(args, "lstm_min_interval", None)
    return 1.0 if value is None else float(value)


ENVIRONMENT_FACTORIES = {
    "lstm": lambda args: LSTMModel(
        user_id=args.user_id or 1,
        benchmark_root=args.srs_benchmark_root,
    ),
    "fsrs6": lambda args: FSRS6Model(
        weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21)
    ),
    "fsrs3": lambda args: FSRS3Model(
        weights=_resolve_benchmark_weights(args, "fsrs3", expected_len=13)
    ),
}


def _require_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "SSP-MMC scheduler requires --sspmmc-policy pointing to a metadata JSON."
        )
    return path


SCHEDULER_FACTORIES = {
    "fsrs6": lambda args: FSRS6Scheduler(
        weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21),
        desired_retention=args.desired_retention,
        priority_mode=args.scheduler_priority,
    ),
    "fsrs3": lambda args: FSRS3Scheduler(
        weights=_resolve_benchmark_weights(args, "fsrs3", expected_len=13),
        desired_retention=args.desired_retention,
    ),
    "hlr": lambda args: HLRScheduler(
        weights=_resolve_benchmark_weights(args, "hlr", expected_len=3),
        desired_retention=args.desired_retention,
    ),
    "dash": lambda args: DASHScheduler(
        weights=_resolve_benchmark_weights(args, "dash", expected_len=9),
        desired_retention=args.desired_retention,
    ),
    "lstm": lambda args: LSTMScheduler(
        user_id=args.user_id or 1,
        benchmark_root=args.srs_benchmark_root,
        desired_retention=args.desired_retention,
        interval_mode=_lstm_interval_mode(args),
        min_interval=_lstm_min_interval(args),
    ),
    "fixed": lambda args: FixedIntervalScheduler(
        interval=normalize_fixed_interval(getattr(args, "fixed_interval", None))
    ),
    "anki_sm2": lambda args: AnkiSM2Scheduler(),
    "memrise": lambda args: MemriseScheduler(),
    "sspmmc": lambda args: SSPMMCScheduler(
        policy_json=_require_policy(args.sspmmc_policy),
        fsrs_weights=None,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize spaced repetition simulation metrics."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to store simulation logs.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing simulation logs (meta + totals) to disk.",
    )
    parser.add_argument(
        "--log-reviews",
        action="store_true",
        help="Include per-event logs (learn/review) in the JSONL output (can be large).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the simulation progress bar.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting the dashboard.",
    )
    parser.add_argument(
        "--fuzz",
        action="store_true",
        help="Apply scheduler interval fuzz (Anki-style).",
    )
    parser.add_argument(
        "--short-term",
        action="store_true",
        help="(Deprecated) Alias for --short-term-source=steps.",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "scheduler"],
        default=None,
        help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or scheduler (LSTM-only short-term intervals)."
        ),
    )
    parser.add_argument(
        "--learning-steps",
        default=None,
        help="Comma-separated learning steps (minutes) for short-term steps mode.",
    )
    parser.add_argument(
        "--relearning-steps",
        default=None,
        help="Comma-separated relearning steps (minutes) for short-term steps mode.",
    )
    parser.add_argument(
        "--short-term-threshold",
        type=float,
        default=0.5,
        help="Short-term threshold (days) for LSTM interval conversion.",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized"],
        default="vectorized",
        help=(
            "Simulation engine: vectorized (default) or event "
            "(FSRS6 environment + FSRS6/FSRS3/HLR/fixed/Memrise/Anki SM-2/SSPMMC/LSTM "
            "schedulers, or LSTM environment + FSRS6/FSRS3/HLR/fixed/Memrise/"
            "Anki SM-2/SSPMMC/LSTM schedulers)."
        ),
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for vectorized engine (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--days", type=int, default=365 * 5, help="Number of simulated days."
    )
    parser.add_argument("--deck", type=int, default=10000, help="Deck size.")
    parser.add_argument(
        "--learn-limit",
        type=int,
        default=10,
        help="Max new cards per day (behavior limit).",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=9999,
        help="Max reviews per day (behavior limit).",
    )
    parser.add_argument(
        "--cost-limit-minutes",
        type=float,
        default=720,
        help="Daily study time limit in minutes (behavior limit).",
    )
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default="review-first",
        help="Card action priority: review-first favors due cards, new-first favors introductions.",
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENT_FACTORIES),
        default="fsrs6",
        help="Memory model to simulate.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Load benchmark weights for this user ID.",
    )
    parser.add_argument(
        "--benchmark-partition",
        default="0",
        help="Partition key inside benchmark result parameters.",
    )
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help=(
            "Override benchmark result base names, e.g. "
            "fsrs6=FSRS-6-short,fsrs3=FSRSv3."
        ),
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help="Path to the srs-benchmark repo (used for LSTM weights).",
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL for per-user costs/probabilities.",
    )
    parser.add_argument(
        "--sched",
        default="fsrs6",
        help=(
            "Scheduler under evaluation "
            f"({', '.join(sorted(SCHEDULER_FACTORIES))}); "
            "use fixed@<days> for fixed intervals."
        ),
    )
    parser.add_argument(
        "--desired-retention",
        type=float,
        default=0.9,
        help="Desired retention target passed to the scheduler.",
    )
    parser.add_argument(
        "--scheduler-priority",
        choices=sorted(FSRS6Scheduler.PRIORITY_MODES),
        default="low_retrievability",
        help="FSRS6 priority hint (ignored by other schedulers).",
    )
    parser.add_argument(
        "--sspmmc-policy",
        type=Path,
        default=None,
        help="Path to an SSP-MMC policy metadata JSON when using --sched sspmmc.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    try:
        scheduler_name, fixed_interval, _ = parse_scheduler_spec(args.sched)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if scheduler_name not in SCHEDULER_FACTORIES:
        raise SystemExit(f"Unknown scheduler '{scheduler_name}'.")
    args.scheduler_spec = args.sched
    args.scheduler = scheduler_name
    args.fixed_interval = fixed_interval

    short_term_source, learning_steps, relearning_steps = _resolve_short_term_config(
        args
    )
    args.short_term_source = short_term_source
    args.short_term = bool(short_term_source)

    if short_term_source in {"steps", "scheduler"} and args.engine != "event":
        raise SystemExit("Short-term scheduling requires --engine event.")
    if short_term_source == "scheduler":
        if args.scheduler != "lstm":
            raise SystemExit("--short-term-source=scheduler requires --sched lstm.")
        args.lstm_interval_mode = "float"
        args.lstm_min_interval = 0.0

    priority_fn = (
        review_first_priority if args.priority == "review-first" else new_first_priority
    )

    rng = random.Random(args.seed)
    env = ENVIRONMENT_FACTORIES[args.env](args)
    agent = SCHEDULER_FACTORIES[args.scheduler](args)
    if short_term_source == "steps":
        agent = ShortTermScheduler(
            agent,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            threshold_days=args.short_term_threshold,
            allow_short_term_interval=False,
        )
    elif short_term_source == "scheduler":
        agent = ShortTermScheduler(
            agent,
            learning_steps=[],
            relearning_steps=[],
            threshold_days=args.short_term_threshold,
            allow_short_term_interval=True,
        )
    cost_limit = (
        args.cost_limit_minutes * 60.0 if args.cost_limit_minutes is not None else None
    )
    button_usage = (
        load_button_usage_config(args.button_usage, args.user_id or 1)
        if args.button_usage is not None
        else None
    )
    usage = normalize_button_usage(button_usage)
    behavior = StochasticBehavior(
        attendance_prob=1.0,
        lazy_good_bias=0.0,
        max_new_per_day=args.learn_limit,
        max_reviews_per_day=args.review_limit,
        max_cost_per_day=cost_limit,
        priority_fn=priority_fn,
        first_rating_prob=usage["first_rating_prob"],
        review_rating_prob=usage["review_rating_prob"],
        learning_rating_prob=usage["learning_rating_prob"],
        relearning_rating_prob=usage["relearning_rating_prob"],
    )
    cost_model = StatefulCostModel(
        state_costs=StateRatingCosts(
            learning=usage["learn_costs"],
            review=usage["review_costs"],
        )
    )
    start_time = time.perf_counter()
    if args.engine == "vectorized":
        if args.log_reviews:
            sys.stderr.write(
                "Vectorized engine does not emit per-event logs; "
                "--log-reviews ignored.\n"
            )
        try:
            stats = simulate_vectorized(
                days=args.days,
                deck_size=args.deck,
                environment=env,
                scheduler=agent,
                behavior=behavior,
                cost_model=cost_model,
                seed=args.seed,
                device=args.torch_device,
                fuzz=args.fuzz,
                progress=not args.no_progress,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        stats = simulate(
            days=args.days,
            deck_size=args.deck,
            environment=env,
            scheduler=agent,
            behavior=behavior,
            cost_model=cost_model,
            fuzz=args.fuzz,
            seed_fn=rng.random,
            progress=not args.no_progress,
        )
    elapsed = time.perf_counter() - start_time
    sys.stderr.write(f"Simulation time: {elapsed:.2f}s\n")
    if not args.no_log:
        _write_log(args, stats)

    if args.no_plot:
        return

    plot_simulation(stats)


def plot_simulation(stats) -> None:
    days = list(range(len(stats.daily_reviews)))

    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax[0].plot(days, stats.daily_reviews, label="Reviews/day", color="tab:blue")
    ax[0].plot(days, stats.daily_new, label="New/day", color="tab:green")
    ax[0].set_ylabel("Count")
    ax[0].legend()
    ax[0].set_title("Workload")

    valid_retentions = [r for r in stats.daily_retention if not math.isnan(r)]
    mean_ret = (
        sum(valid_retentions) / len(valid_retentions) if valid_retentions else 0.0
    )
    ax[1].plot(
        days,
        [c / 60.0 for c in stats.daily_cost],
        label="Study minutes",
        color="tab:red",
    )
    ax[1].set_ylabel("Minutes")
    ax[1].legend()
    ax[1].set_title("Daily workload cost")

    ax[2].plot(days, stats.daily_retention, label="Daily retention", color="tab:purple")
    ax[2].axhline(
        mean_ret,
        color="tab:gray",
        linestyle="--",
        label=f"Mean retention={mean_ret:.3f}",
    )
    ax[2].set_ylabel("Retention")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend()
    ax[2].set_title("Observed retention (1 - lapses/reviews)")

    # Event raster plot
    event_x: list[int] = []
    event_y: list[int] = []
    event_colors: list[str] = []
    per_day_counts: dict[int, int] = {}
    phase_colors = {
        "new": "tab:blue",
        "learning": "tab:orange",
        "review": "tab:green",
        "relearning": "tab:red",
    }
    for event in stats.events:
        y = per_day_counts.get(event.day, 0)
        per_day_counts[event.day] = y + 1
        event_x.append(event.day)
        event_y.append(y)
        phase = getattr(event, "phase", None) or (
            "new" if event.action == Action.LEARN else "review"
        )
        event_colors.append(phase_colors.get(phase, "tab:gray"))
    max_events = max(per_day_counts.values()) if per_day_counts else 0

    ax[3].scatter(event_x, event_y, c=event_colors, s=8)
    ax[3].set_ylim(-1, max(max_events, 1) + 1)
    ax[3].set_xlabel("Day")
    ax[3].set_ylabel("Event order")
    ax[3].set_title("Daily event raster")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="New",
            markerfacecolor=phase_colors["new"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Learning",
            markerfacecolor=phase_colors["learning"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Review",
            markerfacecolor=phase_colors["review"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Relearning",
            markerfacecolor=phase_colors["relearning"],
            markersize=6,
        ),
    ]
    ax[3].legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.show()


def _write_log(args: argparse.Namespace, stats) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)

    desired_retention = (
        args.desired_retention
        if scheduler_uses_desired_retention(args.scheduler)
        else None
    )
    fixed_interval = (
        normalize_fixed_interval(getattr(args, "fixed_interval", None))
        if args.scheduler == "fixed"
        else None
    )
    cost_limit = format_float(args.cost_limit_minutes)
    review_limit = args.review_limit if args.review_limit is not None else "none"
    env_name = (
        getattr(args, "env", None) or getattr(args, "environment", None) or "unknown"
    )
    parts = [f"env={env_name}", f"sched={args.scheduler}"]
    if getattr(args, "fuzz", False):
        parts.append("fuzz=1")
    short_term_source = getattr(args, "short_term_source", None)
    if getattr(args, "short_term", False) and short_term_source is None:
        short_term_source = "steps"
    if short_term_source:
        parts.append(f"st={short_term_source}")
    if fixed_interval is not None:
        parts.append(f"ivl={format_float(fixed_interval)}")
    if args.sspmmc_policy:
        parts.append(f"policy={args.sspmmc_policy.stem}")
    parts.extend(
        [
            f"user={args.user_id or 1}",
            f"days={args.days}",
            f"deck={args.deck}",
            f"learn={args.learn_limit}",
            f"review={review_limit}",
            f"costm={cost_limit}",
            f"prio={args.priority}",
            f"ret={format_float(desired_retention)}",
            f"sprio={args.scheduler_priority}",
            f"seed={args.seed}",
        ]
    )
    filename = args.log_dir / f"log_{'_'.join(parts)}.jsonl"
    meta = {
        "engine": args.engine,
        "days": args.days,
        "deck_size": args.deck,
        "learn_limit": args.learn_limit,
        "review_limit": args.review_limit,
        "cost_limit_minutes": args.cost_limit_minutes,
        "priority": args.priority,
        "environment": env_name,
        "scheduler": args.scheduler,
        "scheduler_spec": getattr(args, "scheduler_spec", args.scheduler),
        "user_id": args.user_id or 1,
        "button_usage": str(args.button_usage) if args.button_usage else None,
        "desired_retention": desired_retention,
        "scheduler_priority": args.scheduler_priority,
        "sspmmc_policy": str(args.sspmmc_policy) if args.sspmmc_policy else None,
        "fixed_interval": fixed_interval,
        "seed": args.seed,
        "fuzz": bool(getattr(args, "fuzz", False)),
        "short_term": bool(short_term_source),
        "short_term_source": short_term_source,
        "learning_steps": _parse_steps(getattr(args, "learning_steps", None)),
        "relearning_steps": _parse_steps(getattr(args, "relearning_steps", None)),
        "short_term_threshold": getattr(args, "short_term_threshold", None),
    }
    with filename.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "meta", "data": meta}) + "\n")
        accum_cost = []
        running = 0.0
        for daily in stats.daily_cost:
            running += daily
            accum_cost.append(running)
        time_average = (
            sum(stats.daily_cost) / len(stats.daily_cost) / 60.0
            if stats.daily_cost
            else 0.0
        )
        accum_time_average = (
            sum(accum_cost) / len(accum_cost) / 3600.0 if accum_cost else 0.0
        )
        memorized_average = (
            sum(stats.daily_memorized) / len(stats.daily_memorized)
            if stats.daily_memorized
            else 0.0
        )
        avg_accum_memorized_per_hour = (
            round(memorized_average / accum_time_average, 2)
            if accum_time_average > 0
            else None
        )
        reviews_average = (
            sum(stats.daily_reviews) / len(stats.daily_reviews)
            if stats.daily_reviews
            else 0.0
        )
        totals = {
            "avg_accum_memorized_per_hour": avg_accum_memorized_per_hour,
            "memorized_average": round(memorized_average),
            "reviews_average": round(reviews_average, 2),
            "time_average": round(time_average, 2),
            "total_reviews": stats.total_reviews,
            "total_lapses": stats.total_lapses,
            "total_cost": round(stats.total_cost),
            "mean_daily_reviews": round(reviews_average, 2),
            "total_projected_retrievability": round(
                stats.total_projected_retrievability
            ),
        }
        if stats.total_projected_retrievability > 0:
            totals["cost_per_projected_retrievability"] = round(
                stats.total_cost / stats.total_projected_retrievability, 2
            )
        else:
            totals["cost_per_projected_retrievability"] = None
        fh.write(json.dumps({"type": "totals", "data": totals}) + "\n")
        if args.log_reviews:
            for event in stats.events:
                fh.write(json.dumps({"type": "event", "data": event.to_dict()}) + "\n")


if __name__ == "__main__":
    main()
