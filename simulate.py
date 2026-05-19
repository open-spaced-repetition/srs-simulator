from __future__ import annotations

import argparse
import csv
import hashlib
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
    AnkiSM2APScheduler,
    MemriseScheduler,
    FSRS6ADRScheduler,
    FSRS6OracleStationaryFiniteDistillScheduler,
    FSRS6APScheduler,
    SSPMMCScheduler,
)
from simulator.core import Action, new_first_priority, review_first_priority
from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_PRIORITY,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
    DEFAULT_SHORT_TERM_LOOPS_LIMIT,
)
from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.short_term import ShortTermScheduler
from simulator.short_term_config import (
    parse_steps as _parse_steps,
    resolve_short_term_config as _resolve_short_term_config,
)


def _resolve_benchmark_weights(
    args, environment: str, expected_len: int
) -> tuple[float, ...] | None:
    overrides = parse_result_overrides(args.benchmark_result)
    short_term = bool(getattr(args, "short_term_source", None))
    weights = load_benchmark_weights(
        repo_root=Path(__file__).resolve().parent,
        benchmark_root=args.srs_benchmark_root,
        environment=environment,
        user_id=args.user_id or 1,
        partition_key=args.benchmark_partition,
        overrides=overrides,
        short_term=short_term,
    )
    if len(weights) != expected_len:
        raise ValueError(
            f"{environment} expects {expected_len} weights, got {len(weights)}."
        )
    return tuple(float(x) for x in weights)


def _lstm_interval_mode(args: argparse.Namespace) -> str:
    return getattr(args, "lstm_interval_mode", None) or "integer"


def _lstm_min_interval(args: argparse.Namespace) -> float:
    value = getattr(args, "lstm_min_interval", None)
    return 1.0 if value is None else float(value)


ENVIRONMENT_FACTORIES = {
    "lstm": lambda args: LSTMModel(
        user_id=args.user_id or 1,
        benchmark_root=args.srs_benchmark_root,
        short_term=bool(getattr(args, "short_term_source", None)),
    ),
    "fsrs6": lambda args: FSRS6Model(
        weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21)
    ),
    "fsrs6_default": lambda args: FSRS6Model(weights=None),
    "fsrs3": lambda args: FSRS3Model(
        weights=_resolve_benchmark_weights(args, "fsrs3", expected_len=13)
    ),
    "fsrs3_default": lambda args: FSRS3Model(weights=None),
}


def _require_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "SSP-MMC scheduler requires --sspmmc-policy pointing to a metadata JSON."
        )
    return path


def _require_fsrs6_adr_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "FSRS6 ADR scheduler requires --fsrs6-adr-policy pointing to a policy JSON."
        )
    return path


def _require_fsrs6_oracle_stationary_finite_distill_policy(
    path: Path | None,
) -> Path:
    if path is None:
        raise ValueError(
            "FSRS6 oracle stationary finite distill scheduler requires "
            "--fsrs6-oracle-stationary-finite-distill-policy pointing to a policy JSON."
        )
    return path


def _require_fsrs6_ap_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "FSRS6 AP scheduler requires --fsrs6-ap-policy pointing to a policy JSON."
        )
    return path


def _require_anki_sm2_ap_policy(path: Path | None) -> Path:
    if path is None:
        raise ValueError(
            "Anki SM2 AP scheduler requires --anki-sm2-ap-policy pointing to a policy JSON."
        )
    return path


SCHEDULER_FACTORIES = {
    "fsrs6": lambda args: FSRS6Scheduler(
        weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21),
        desired_retention=args.desired_retention,
        priority_mode=args.scheduler_priority,
    ),
    "fsrs6_default": lambda args: FSRS6Scheduler(
        weights=None,
        desired_retention=args.desired_retention,
        priority_mode=args.scheduler_priority,
    ),
    "fsrs3": lambda args: FSRS3Scheduler(
        weights=_resolve_benchmark_weights(args, "fsrs3", expected_len=13),
        desired_retention=args.desired_retention,
    ),
    "fsrs3_default": lambda args: FSRS3Scheduler(
        weights=None,
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
        short_term=bool(getattr(args, "short_term_source", None)),
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
    "fsrs6_adr": lambda args: FSRS6ADRScheduler(
        policy_json=_require_fsrs6_adr_policy(args.fsrs6_adr_policy),
        fsrs_weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21),
        priority_mode=args.scheduler_priority,
        simulation_days=args.days,
    ),
    "fsrs6_adr_time": lambda args: FSRS6ADRScheduler(
        policy_json=_require_fsrs6_adr_policy(args.fsrs6_adr_policy),
        fsrs_weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21),
        priority_mode=args.scheduler_priority,
        simulation_days=args.days,
    ),
    "fsrs6_default_adr": lambda args: FSRS6ADRScheduler(
        policy_json=_require_fsrs6_adr_policy(args.fsrs6_adr_policy),
        fsrs_weights=None,
        priority_mode=args.scheduler_priority,
        simulation_days=args.days,
    ),
    "fsrs6_oracle_stationary_finite_distill": lambda args: (
        FSRS6OracleStationaryFiniteDistillScheduler(
            policy_json=_require_fsrs6_oracle_stationary_finite_distill_policy(
                args.fsrs6_oracle_stationary_finite_distill_policy
            ),
            fsrs_weights=_resolve_benchmark_weights(args, "fsrs6", expected_len=21),
            priority_mode=args.scheduler_priority,
        )
    ),
    "fsrs6_ap": lambda args: FSRS6APScheduler(
        policy_json=_require_fsrs6_ap_policy(args.fsrs6_ap_policy),
        priority_mode=args.scheduler_priority,
    ),
    "anki_sm2_ap": lambda args: AnkiSM2APScheduler(
        policy_json=_require_anki_sm2_ap_policy(args.anki_sm2_ap_policy),
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
        "--short-term-source",
        choices=["steps", "sched"],
        default=None,
        help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or sched (LSTM-only short-term intervals)."
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
        "--short-term-loops-limit",
        type=int,
        default=DEFAULT_SHORT_TERM_LOOPS_LIMIT,
        help=(
            "Max short-term review loops per day (per user). "
            "Remaining short-term cards carry over to the next day."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["event"],
        default="event",
        help=(
            "Simulation engine: event-driven simulator with per-event logging support."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of simulated days.",
    )
    parser.add_argument(
        "--deck", type=int, default=DEFAULT_DECK_SIZE, help="Deck size."
    )
    parser.add_argument(
        "--learn-limit",
        type=int,
        default=DEFAULT_LEARN_LIMIT,
        help="Max new cards per day (behavior limit).",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help="Max reviews per day (behavior limit).",
    )
    parser.add_argument(
        "--cost-limit-minutes",
        type=float,
        default=DEFAULT_COST_LIMIT_MINUTES,
        help="Daily study time limit in minutes (behavior limit).",
    )
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default=DEFAULT_PRIORITY,
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
        help=(
            "Path to Anki button usage JSONL for per-user costs/probabilities. "
            "Review Markov transitions require --review-markov-transition."
        ),
    )
    parser.add_argument(
        "--review-markov-transition",
        action="store_true",
        help=(
            "Use long_term_transition from button usage data for review button "
            "behavior. Defaults to marginal review probabilities only."
        ),
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
        default=DEFAULT_SCHEDULER_PRIORITY,
        help="FSRS6 priority hint (ignored by other schedulers).",
    )
    parser.add_argument(
        "--sspmmc-policy",
        type=Path,
        default=None,
        help="Path to an SSP-MMC policy metadata JSON when using --sched sspmmc.",
    )
    parser.add_argument(
        "--fsrs6-adr-policy",
        type=Path,
        default=None,
        help=(
            "Path to an FSRS6 ADR policy JSON when using --sched fsrs6_adr, "
            "fsrs6_adr_time, or fsrs6_default_adr."
        ),
    )
    parser.add_argument(
        "--fsrs6-oracle-stationary-finite-distill-policy",
        type=Path,
        default=None,
        help=(
            "Path to an FSRS6 oracle stationary finite distill policy JSON when "
            "using --sched fsrs6_oracle_stationary_finite_distill."
        ),
    )
    parser.add_argument(
        "--fsrs6-ap-policy",
        type=Path,
        default=None,
        help="Path to an FSRS6 AP policy JSON when using --sched fsrs6_ap.",
    )
    parser.add_argument(
        "--anki-sm2-ap-policy",
        type=Path,
        default=None,
        help="Path to an Anki SM2 AP policy JSON when using --sched anki_sm2_ap.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
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

    if short_term_source in {"steps", "sched"} and args.engine != "event":
        raise SystemExit("Short-term scheduling requires --engine event.")
    if short_term_source == "sched":
        if args.scheduler != "lstm":
            raise SystemExit("--short-term-source=sched requires --sched lstm.")
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
    elif short_term_source == "sched":
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
        review_markov_transition=(
            usage.get("long_term_transition") if args.review_markov_transition else None
        ),
    )
    if short_term_source:
        state_rating_costs = usage["state_rating_costs"]
        cost_model = StatefulCostModel(
            state_costs=StateRatingCosts(
                learning=state_rating_costs[0],
                review=state_rating_costs[1],
                relearning=state_rating_costs[2],
            )
        )
    else:
        cost_model = StatefulCostModel(
            state_costs=StateRatingCosts(
                learning=usage["learn_costs"],
                review=usage["review_costs"],
                relearning=usage["review_costs"],
            )
        )
    start_time = time.perf_counter()
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
        short_term_loops_limit=args.short_term_loops_limit,
    )
    elapsed = time.perf_counter() - start_time
    sys.stderr.write(f"Simulation time: {elapsed:.2f}s\n")
    timing = getattr(stats, "timing", None)
    if timing:
        long_s = timing.get("long_reviews_s", 0.0)
        short_s = timing.get("short_reviews_s", 0.0)
        sys.stderr.write(f"Review timing: long={long_s:.2f}s, short={short_s:.2f}s\n")
        short_loops = timing.get("short_review_loops")
        loop_days = timing.get("short_review_loop_days")
        if short_loops is not None and loop_days is not None:
            avg_per_day = short_loops / args.days if args.days else 0.0
            avg_active = short_loops / loop_days if loop_days > 0 else 0.0
            sys.stderr.write(
                "Short-term loops: "
                f"total={int(short_loops)}, "
                f"avg/day={avg_per_day:.2f}, "
                f"avg/active-day={avg_active:.2f}\n"
            )
    _print_review_summary(stats)
    if not args.no_log:
        _write_log(args, stats)

    if args.no_plot:
        return

    plot_simulation(stats, args)


def _format_plot_footer(args: argparse.Namespace) -> str:
    env_name = (
        getattr(args, "env", None) or getattr(args, "environment", None) or "unknown"
    )
    sched_label = (
        getattr(args, "scheduler_spec", None)
        or getattr(args, "sched", None)
        or getattr(args, "scheduler", None)
        or "unknown"
    )
    review_limit = args.review_limit if args.review_limit is not None else "none"
    cost_limit = format_float(args.cost_limit_minutes)
    desired_retention = (
        format_float(args.desired_retention)
        if scheduler_uses_desired_retention(args.scheduler)
        else "n/a"
    )
    resolved_short_term, learning_steps, relearning_steps = _resolve_short_term_config(
        args
    )
    short_term_source = resolved_short_term or "off"
    short_term_threshold = getattr(args, "short_term_threshold", None)
    short_term_max_loops = getattr(args, "short_term_loops_limit", None)
    fixed_interval = (
        normalize_fixed_interval(getattr(args, "fixed_interval", None))
        if args.scheduler == "fixed"
        else None
    )
    header = [
        f"environment={env_name}",
        f"scheduler={sched_label}",
        f"engine={args.engine}",
        f"short-term-source={short_term_source}",
    ]
    core = [
        f"user={args.user_id or 1}",
        f"days={args.days}",
        f"deck={args.deck}",
        f"learn-limit={args.learn_limit}",
        f"review-limit={review_limit}",
        f"cost-limit-minutes={cost_limit}",
        f"desired-retention={desired_retention}",
        f"priority={args.priority}",
        f"scheduler-priority={args.scheduler_priority}",
        f"seed={args.seed}",
    ]
    if getattr(args, "fuzz", False):
        core.append("fuzz=on")
    extra: list[str] = []
    if fixed_interval is not None:
        extra.append(f"fixed-interval={format_float(fixed_interval)}")
    sspmmc_policy = getattr(args, "sspmmc_policy", None)
    fsrs6_adr_policy = getattr(args, "fsrs6_adr_policy", None)
    fsrs6_oracle_distill_policy = getattr(
        args, "fsrs6_oracle_stationary_finite_distill_policy", None
    )
    fsrs6_ap_policy = getattr(args, "fsrs6_ap_policy", None)
    anki_sm2_ap_policy = getattr(args, "anki_sm2_ap_policy", None)
    if sspmmc_policy:
        extra.append(f"sspmmc-policy={sspmmc_policy.stem}")
    if fsrs6_adr_policy:
        extra.append(f"fsrs6-adr-policy={fsrs6_adr_policy.stem}")
    if fsrs6_oracle_distill_policy:
        extra.append(
            "fsrs6-oracle-stationary-finite-distill-policy="
            f"{fsrs6_oracle_distill_policy.stem}"
        )
    if fsrs6_ap_policy:
        extra.append(f"fsrs6-ap-policy={fsrs6_ap_policy.stem}")
    if anki_sm2_ap_policy:
        extra.append(f"anki-sm2-ap-policy={anki_sm2_ap_policy.stem}")
    if short_term_source != "off":
        extra.append(f"learning-steps={','.join(str(step) for step in learning_steps)}")
        extra.append(
            f"relearning-steps={','.join(str(step) for step in relearning_steps)}"
        )
        if short_term_threshold is not None:
            extra.append(f"short-term-threshold={format_float(short_term_threshold)}")
        if short_term_max_loops is not None:
            extra.append(f"short-term-loops-limit={short_term_max_loops}")
    lines = [" ".join(header), " ".join(core)]
    if extra:
        lines.append(" ".join(extra))
    return "\n".join(lines)


def _log_filename_token(value: object) -> str:
    text = str(value).strip()
    return "".join(char if char.isalnum() or char in "._=-" else "-" for char in text)


_LOG_FILENAME_COMPONENT_LIMIT = 240


def _simulation_log_filename(log_dir: Path, parts: list[str]) -> Path:
    def _build(candidate_parts: list[str]) -> str:
        return f"log_{'_'.join(candidate_parts)}.jsonl"

    candidate_parts = list(parts)
    filename = _build(candidate_parts)
    if len(filename) <= _LOG_FILENAME_COMPONENT_LIMIT:
        return log_dir / filename

    policy_detail_prefixes = ("policy-dr=", "lambda=")
    shortened = [
        part for part in candidate_parts if not part.startswith(policy_detail_prefixes)
    ]
    if len(shortened) != len(candidate_parts):
        candidate_parts = shortened
        filename = _build(candidate_parts)
        if len(filename) <= _LOG_FILENAME_COMPONENT_LIMIT:
            return log_dir / filename

    policy_parts = [part for part in candidate_parts if part.startswith("policy=")]
    if policy_parts:
        digest = hashlib.sha1("_".join(parts).encode("utf-8")).hexdigest()[:10]
        shortened = [part for part in candidate_parts if not part.startswith("policy=")]
        shortened.append(f"policyid={digest}")
        candidate_parts = shortened
        filename = _build(candidate_parts)
        if len(filename) <= _LOG_FILENAME_COMPONENT_LIMIT:
            return log_dir / filename

    for prefix in (
        "goalw=",
        "days=",
        "deck=",
        "learn=",
        "review=",
        "costm=",
        "sprio=",
    ):
        shortened = [part for part in candidate_parts if not part.startswith(prefix)]
        if len(shortened) == len(candidate_parts):
            continue
        candidate_parts = shortened
        filename = _build(candidate_parts)
        if len(filename) <= _LOG_FILENAME_COMPONENT_LIMIT:
            return log_dir / filename

    raise ValueError(
        "Simulation log filename exceeds the filesystem component limit after "
        "shortening non-filter fields."
    )


def plot_simulation(stats, args: argparse.Namespace) -> None:
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

    footer = _format_plot_footer(args)
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    plt.show()


def _print_review_summary(stats) -> None:
    daily = list(stats.daily_reviews or [])
    if not daily:
        return
    total = int(stats.total_reviews)
    days = len(daily)
    mean_daily = total / days if days else 0.0
    max_daily = max(daily)
    nonzero = [count for count in daily if count > 0]
    days_with_reviews = len(nonzero)
    mean_on_review_days = sum(nonzero) / days_with_reviews if days_with_reviews else 0.0
    sys.stderr.write(
        "Review volume: "
        f"total={total}, "
        f"mean/day={mean_daily:.2f}, "
        f"max/day={max_daily}, "
        f"days-with-reviews={days_with_reviews}, "
        f"mean-on-review-days={mean_on_review_days:.2f}\n"
    )


def _write_daily_csv(path: Path, stats) -> None:
    headers = [
        "day",
        "reviews",
        "new",
        "retention",
        "cost",
        "memorized",
        "phase_reviews",
        "phase_lapses",
        "short_loops",
    ]
    daily_map = {
        "reviews": stats.daily_reviews,
        "new": stats.daily_new,
        "retention": stats.daily_retention,
        "cost": stats.daily_cost,
        "memorized": stats.daily_memorized,
        "phase_reviews": stats.daily_phase_reviews,
        "phase_lapses": stats.daily_phase_lapses,
        "short_loops": stats.daily_short_loops,
    }
    days = len(stats.daily_reviews or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for day in range(days):
            row: list[object] = [day]
            for key in headers[1:]:
                series = daily_map.get(key)
                if series is None or day >= len(series):
                    row.append("")
                else:
                    row.append(series[day])
            writer.writerow(row)


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
    parts = [f"env={env_name}", f"engine={args.engine}", f"sched={args.scheduler}"]
    run_id = getattr(args, "run_id", None)
    if run_id:
        parts.append(f"run={_log_filename_token(run_id)}")
    if getattr(args, "fuzz", False):
        parts.append("fuzz=1")
    short_term_source = getattr(args, "short_term_source", None)
    short_term_max_loops = getattr(args, "short_term_loops_limit", None)
    if short_term_source:
        parts.append(f"st={short_term_source}")
        if short_term_max_loops is not None:
            parts.append(f"stloops={short_term_max_loops}")
    if fixed_interval is not None:
        parts.append(f"ivl={format_float(fixed_interval)}")
    sspmmc_policy = getattr(args, "sspmmc_policy", None)
    fsrs6_adr_policy = getattr(args, "fsrs6_adr_policy", None)
    fsrs6_oracle_distill_policy = getattr(
        args, "fsrs6_oracle_stationary_finite_distill_policy", None
    )
    fsrs6_ap_policy = getattr(args, "fsrs6_ap_policy", None)
    anki_sm2_ap_policy = getattr(args, "anki_sm2_ap_policy", None)
    if sspmmc_policy:
        parts.append(f"policy={sspmmc_policy.stem}")
    if fsrs6_adr_policy:
        parts.append(f"policy={fsrs6_adr_policy.stem}")
        adr_baseline_dr = getattr(args, "fsrs6_adr_baseline_desired_retention", None)
        adr_lambda = getattr(args, "fsrs6_adr_lambda_value", None)
        if adr_baseline_dr is not None:
            parts.append(f"policy-dr={format_float(adr_baseline_dr)}")
        if adr_lambda is not None:
            parts.append(f"lambda={format_float(adr_lambda)}")
    if fsrs6_oracle_distill_policy:
        parts.append(f"policy={fsrs6_oracle_distill_policy.stem}")
        goal_cost_weight = getattr(
            args,
            "fsrs6_oracle_stationary_finite_distill_goal_cost_weight",
            None,
        )
        if goal_cost_weight is not None:
            parts.append(f"goalw={format_float(goal_cost_weight)}")
    if fsrs6_ap_policy:
        parts.append(f"policy={fsrs6_ap_policy.stem}")
        ap_baseline_dr = getattr(args, "fsrs6_ap_baseline_desired_retention", None)
        ap_lambda = getattr(args, "fsrs6_ap_lambda_value", None)
        if ap_baseline_dr is not None:
            parts.append(f"policy-dr={format_float(ap_baseline_dr)}")
        if ap_lambda is not None:
            parts.append(f"lambda={format_float(ap_lambda)}")
    if anki_sm2_ap_policy:
        parts.append(f"policy={anki_sm2_ap_policy.stem}")
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
    filename = _simulation_log_filename(args.log_dir, parts)
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
        "run_id": str(run_id) if run_id else None,
        "user_id": args.user_id or 1,
        "button_usage": str(args.button_usage) if args.button_usage else None,
        "review_markov_transition": bool(
            getattr(args, "review_markov_transition", False)
        ),
        "desired_retention": desired_retention,
        "scheduler_priority": args.scheduler_priority,
        "sspmmc_policy": str(sspmmc_policy) if sspmmc_policy else None,
        "fsrs6_adr_policy": str(fsrs6_adr_policy) if fsrs6_adr_policy else None,
        "fsrs6_adr_baseline_desired_retention": getattr(
            args, "fsrs6_adr_baseline_desired_retention", None
        ),
        "fsrs6_adr_lambda_value": getattr(args, "fsrs6_adr_lambda_value", None),
        "fsrs6_oracle_stationary_finite_distill_policy": str(
            fsrs6_oracle_distill_policy
        )
        if fsrs6_oracle_distill_policy
        else None,
        "fsrs6_oracle_stationary_finite_distill_goal_cost_weight": getattr(
            args,
            "fsrs6_oracle_stationary_finite_distill_goal_cost_weight",
            None,
        ),
        "fsrs6_ap_policy": str(fsrs6_ap_policy) if fsrs6_ap_policy else None,
        "fsrs6_ap_baseline_desired_retention": getattr(
            args, "fsrs6_ap_baseline_desired_retention", None
        ),
        "fsrs6_ap_lambda_value": getattr(args, "fsrs6_ap_lambda_value", None),
        "anki_sm2_ap_policy": str(anki_sm2_ap_policy) if anki_sm2_ap_policy else None,
        "fixed_interval": fixed_interval,
        "seed": args.seed,
        "fuzz": bool(getattr(args, "fuzz", False)),
        "short_term": bool(short_term_source),
        "short_term_source": short_term_source,
        "learning_steps": _parse_steps(getattr(args, "learning_steps", None)),
        "relearning_steps": _parse_steps(getattr(args, "relearning_steps", None)),
        "short_term_threshold": getattr(args, "short_term_threshold", None),
        "short_term_loops_limit": short_term_max_loops,
    }
    if getattr(args, "write_daily_csv", True):
        csv_filename = filename.with_suffix(".csv")
        _write_daily_csv(csv_filename, stats)
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
