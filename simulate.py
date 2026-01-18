from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from simulator import simulate
from simulator.behavior import StochasticBehavior
from simulator.cost import StatefulCostModel
from simulator.models import (
    FSRS3Model,
    FSRS6Model,
    LSTMModel,
)
from simulator.schedulers import (
    FSRS3Scheduler,
    FSRS6Scheduler,
    HLRScheduler,
    DASHScheduler,
    FixedIntervalScheduler,
    SSPMMCScheduler,
)
from simulator.core import Action, Event, new_first_priority, review_first_priority

ENVIRONMENT_FACTORIES = {
    "lstm": lambda args: LSTMModel(),
    "fsrs6": lambda args: FSRS6Model(),
    "fsrs3": lambda args: FSRS3Model(),
}


def _require_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "SSP-MMC scheduler requires --sspmmc-policy pointing to a metadata JSON."
        )
    return path


SCHEDULER_FACTORIES = {
    "fsrs6": lambda args: FSRS6Scheduler(
        desired_retention=args.desired_retention,
        priority_mode=args.scheduler_priority,
    ),
    "fsrs3": lambda args: FSRS3Scheduler(desired_retention=args.desired_retention),
    "hlr": lambda args: HLRScheduler(desired_retention=args.desired_retention),
    "dash": lambda args: DASHScheduler(desired_retention=args.desired_retention),
    "fixed": lambda args: FixedIntervalScheduler(),
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
        help="Disable writing simulation logs to disk.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the simulation progress bar.",
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
        "--environment",
        choices=sorted(ENVIRONMENT_FACTORIES),
        default="fsrs6",
        help="Memory model to simulate.",
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(SCHEDULER_FACTORIES),
        default="fsrs6",
        help="Scheduler under evaluation.",
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
        help="Path to an SSP-MMC policy metadata JSON when using --scheduler sspmmc.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    priority_fn = (
        review_first_priority if args.priority == "review-first" else new_first_priority
    )

    rng = random.Random(args.seed)
    env = ENVIRONMENT_FACTORIES[args.environment](args)
    agent = SCHEDULER_FACTORIES[args.scheduler](args)
    cost_limit = (
        args.cost_limit_minutes * 60.0 if args.cost_limit_minutes is not None else None
    )
    behavior = StochasticBehavior(
        attendance_prob=1.0,
        lazy_good_bias=0.0,
        max_new_per_day=args.learn_limit,
        max_reviews_per_day=args.review_limit,
        max_cost_per_day=cost_limit,
        priority_fn=priority_fn,
    )
    cost_model = StatefulCostModel()
    stats = simulate(
        days=args.days,
        deck_size=args.deck,
        environment=env,
        scheduler=agent,
        behavior=behavior,
        cost_model=cost_model,
        seed_fn=rng.random,
        progress=not args.no_progress,
    )
    if not args.no_log:
        _write_log(args, stats)

    days = list(range(args.days))

    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax[0].plot(days, stats.daily_reviews, label="Reviews/day", color="tab:blue")
    ax[0].plot(days, stats.daily_new, label="New/day", color="tab:green")
    ax[0].set_ylabel("Count")
    ax[0].legend()
    ax[0].set_title("Workload")

    valid_retentions = [
        r for r, cnt in zip(stats.daily_retention, stats.daily_reviews) if cnt > 0
    ]
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
    for event in stats.events:
        y = per_day_counts.get(event.day, 0)
        per_day_counts[event.day] = y + 1
        event_x.append(event.day)
        event_y.append(y)
        event_colors.append("tab:blue" if event.action == Action.LEARN else "tab:green")
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
            markerfacecolor="tab:blue",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Review",
            markerfacecolor="tab:green",
            markersize=6,
        ),
    ]
    ax[3].legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.show()


def _write_log(args: argparse.Namespace, stats) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    cost_limit = (
        f"{args.cost_limit_minutes:.2f}"
        if args.cost_limit_minutes is not None
        else "none"
    )
    review_limit = args.review_limit if args.review_limit is not None else "none"
    parts = [
        f"env{args.environment}",
        f"sched{args.scheduler}",
        f"days{args.days}",
        f"deck{args.deck}",
        f"learn{args.learn_limit}",
        f"review{review_limit}",
        f"cost{cost_limit}",
        f"priority{args.priority}",
        f"seed{args.seed}",
    ]
    filename = args.log_dir / f"log_{'_'.join(map(str, parts))}.jsonl"
    meta = {
        "days": args.days,
        "deck_size": args.deck,
        "learn_limit": args.learn_limit,
        "review_limit": args.review_limit,
        "cost_limit_minutes": args.cost_limit_minutes,
        "priority": args.priority,
        "environment": args.environment,
        "scheduler": args.scheduler,
        "desired_retention": args.desired_retention,
        "scheduler_priority": args.scheduler_priority,
        "sspmmc_policy": str(args.sspmmc_policy) if args.sspmmc_policy else None,
        "seed": args.seed,
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
        for event in stats.events:
            fh.write(json.dumps({"type": "event", "data": event.to_dict()}) + "\n")


if __name__ == "__main__":
    main()
