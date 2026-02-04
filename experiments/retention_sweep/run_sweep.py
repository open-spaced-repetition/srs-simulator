from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, List
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from experiments.retention_sweep.cli_utils import (
    add_benchmark_args,
    add_button_usage_arg,
    add_common_sim_args,
    add_env_sched_args,
    add_fuzz_arg,
    add_log_args,
    add_retention_range_args,
    add_short_term_args,
    add_torch_device_arg,
    parse_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a desired-retention sweep for simulate.py.",
        allow_abbrev=False,
    )
    add_retention_range_args(parser)
    add_env_sched_args(
        parser,
        env_default="lstm",
        sched_default="fsrs6",
        env_help="Comma-separated list of environments to sweep.",
        sched_help=(
            "Comma-separated list of schedulers to sweep "
            "(include sspmmc to run policies; use fixed@<days> for fixed intervals)."
        ),
    )
    add_common_sim_args(
        parser,
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Load benchmark weights for this user ID.",
    )
    add_benchmark_args(
        parser,
        benchmark_result_help=(
            "Override benchmark result base names, e.g. "
            "fsrs6=FSRS-6-short,fsrs3=FSRSv3."
        ),
        benchmark_partition_help="Partition key inside benchmark result parameters.",
        srs_benchmark_help="Path to the srs-benchmark repo (used for LSTM weights).",
    )
    add_button_usage_arg(parser, default_path=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument(
        "--sspmmc-policy",
        type=Path,
        default=None,
        help="Path to an SSP-MMC policy metadata JSON when using sspmmc.",
    )
    parser.add_argument(
        "--sspmmc-policy-dir",
        type=Path,
        default=None,
        help=(
            "Directory with SSP-MMC policy metadata JSON files "
            "(defaults to ../SSP-MMC-FSRS/outputs/policies/user_<id>)."
        ),
    )
    parser.add_argument(
        "--sspmmc-policies",
        default=None,
        help="Comma-separated list of SSP-MMC policy metadata JSON paths.",
    )
    parser.add_argument(
        "--sspmmc-policy-glob",
        default="*.json",
        help="Glob pattern to select policies in --sspmmc-policy-dir.",
    )
    parser.add_argument(
        "--sspmmc-max",
        type=int,
        default=None,
        help="Maximum number of SSP-MMC policies to simulate.",
    )
    parser.add_argument(
        "--sspmmc-desired-retention",
        type=float,
        default=0.9,
        help="Desired retention value logged for SSP-MMC runs.",
    )
    add_log_args(
        parser, log_dir_default=None, include_no_log=True, include_no_progress=True
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots during the sweep (slower, opens windows).",
    )
    add_fuzz_arg(parser)
    add_short_term_args(
        parser,
        choices=["steps", "sched"],
        source_help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or sched (LSTM-only short-term intervals)."
        ),
        learning_help="Comma-separated learning steps (minutes) for short-term steps mode.",
        relearning_help="Comma-separated relearning steps (minutes) for short-term steps mode.",
        threshold_help="Short-term threshold (days) for LSTM interval conversion.",
    )
    parser.add_argument(
        "--log-reviews",
        action="store_true",
        help="Include per-event logs (learn/review) in the JSONL output (can be large).",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized"],
        default="vectorized",
        help=(
            "Simulation engine: vectorized (default) or event "
            "(FSRS6 environment + FSRS6 scheduler, or LSTM environment + "
            "FSRS6/FSRS3/HLR/fixed/Memrise/Anki SM-2/SSPMMC/LSTM schedulers)."
        ),
    )
    add_torch_device_arg(parser)
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable total sweep time summary output.",
    )
    parser.add_argument(
        "--progress-events",
        action="store_true",
        help="Emit JSON progress events to stdout (for parent aggregation).",
    )
    return parser.parse_args()


def _dr_values(start: float, end: float, step: float) -> List[float]:
    values: List[float] = []
    if step == 0:
        return values
    value = round(start, 2)
    end = round(end, 2)
    epsilon = abs(step) * 1e-6
    if step > 0:
        while value <= end + epsilon:
            values.append(value)
            next_value = round(value + step, 2)
            if next_value == value:
                raise SystemExit(
                    "Retention step is too small after rounding; "
                    "use a step of at least 0.01."
                )
            value = next_value
    else:
        while value >= end - epsilon:
            values.append(value)
            next_value = round(value + step, 2)
            if next_value == value:
                raise SystemExit(
                    "Retention step is too small after rounding; "
                    "use a step of at least 0.01."
                )
            value = next_value
    return values


def _progress_label(args: argparse.Namespace) -> str:
    label = f"{args.environment}/{args.scheduler}"
    if args.scheduler == "fixed":
        interval = getattr(args, "fixed_interval", None)
        if interval is not None:
            label = f"{label} ivl={format_float(interval)}"
        return f"u{args.user_id} {label}" if args.user_id is not None else label
    if args.scheduler == "sspmmc":
        if args.sspmmc_policy:
            label = f"{label}:{args.sspmmc_policy.stem}"
    elif scheduler_uses_desired_retention(args.scheduler):
        label = f"{label} dr={args.desired_retention:.2f}"
    return f"u{args.user_id} {label}" if args.user_id is not None else label


def _make_progress_callback(
    args: argparse.Namespace,
) -> tuple[Callable[[int, int], None], Callable[[], None]]:
    if args.progress_events:
        label = _progress_label(args)
        last_completed = -1

        def _emit_progress(completed: int, total: int) -> None:
            nonlocal last_completed
            if completed == last_completed:
                return
            last_completed = completed
            event = {
                "type": "progress",
                "completed": completed,
                "total": total,
                "label": label,
            }
            print(json.dumps(event), flush=True)

        def _noop_close() -> None:
            return None

        _emit_progress(0, args.days)
        return _emit_progress, _noop_close

    if args.no_progress:

        def _noop_update(_completed: int, _total: int) -> None:
            return None

        def _noop_close() -> None:
            return None

        return _noop_update, _noop_close

    bar = tqdm(
        total=args.days,
        desc=_progress_label(args),
        unit="day",
        file=sys.stderr,
        ascii=True,
        position=0,
        leave=False,
    )

    def _update_progress(completed: int, total: int) -> None:
        if total > 0 and bar.total != total:
            bar.total = total
        delta = completed - bar.n
        if delta > 0:
            bar.update(delta)
        elif delta < 0:
            bar.n = completed
            bar.refresh()

    return _update_progress, bar.close


def _resolve_policy_paths(
    args: argparse.Namespace, repo_root: Path, run_sspmmc: bool
) -> List[Path]:
    if args.sspmmc_policies:
        paths = [Path(path) for path in parse_csv(args.sspmmc_policies)]
        return [path.resolve() for path in paths]

    if args.sspmmc_policy:
        return [args.sspmmc_policy.resolve()]

    policy_dir = args.sspmmc_policy_dir
    if policy_dir is None and run_sspmmc:
        user_id = args.user_id or 1
        candidate = (
            repo_root.parent
            / "SSP-MMC-FSRS"
            / "outputs"
            / "policies"
            / f"user_{user_id}"
        )
        if candidate.exists():
            policy_dir = candidate

    if policy_dir is None:
        return []

    policy_dir = policy_dir.resolve()
    paths = sorted(policy_dir.glob(args.sspmmc_policy_glob))
    if args.sspmmc_max is not None:
        paths = paths[: args.sspmmc_max]
    return paths


def _run_once(
    run_args: argparse.Namespace,
    priority_fn,
    run_simulation,
    simulate_cli,
    behavior_cls,
    cost_model_cls,
) -> None:
    progress_callback, progress_close = _make_progress_callback(run_args)
    rng = random.Random(run_args.seed)
    short_term_source, learning_steps, relearning_steps = (
        simulate_cli._resolve_short_term_config(run_args)
    )
    run_args.short_term_source = short_term_source
    run_args.short_term = bool(short_term_source)
    if short_term_source in {"steps", "sched"} and run_args.engine not in {
        "event",
        "vectorized",
    }:
        raise SystemExit("Short-term scheduling requires --engine event or vectorized.")
    if short_term_source == "sched":
        if run_args.scheduler != "lstm":
            raise SystemExit("--short-term-source=sched requires --sched lstm.")
        run_args.lstm_interval_mode = "float"
        run_args.lstm_min_interval = 0.0

    env = simulate_cli.ENVIRONMENT_FACTORIES[run_args.environment](run_args)
    agent = simulate_cli.SCHEDULER_FACTORIES[run_args.scheduler](run_args)
    if run_args.engine == "event":
        if short_term_source in {"steps", "sched"}:
            from simulator.short_term import ShortTermScheduler

            agent = ShortTermScheduler(
                agent,
                learning_steps=learning_steps if short_term_source == "steps" else [],
                relearning_steps=relearning_steps
                if short_term_source == "steps"
                else [],
                threshold_days=getattr(run_args, "short_term_threshold", 0.5),
                allow_short_term_interval=short_term_source == "sched",
            )
    cost_limit = (
        run_args.cost_limit_minutes * 60.0
        if run_args.cost_limit_minutes is not None
        else None
    )
    from simulator.button_usage import load_button_usage_config, normalize_button_usage
    from simulator.cost import StateRatingCosts

    button_usage = (
        load_button_usage_config(run_args.button_usage, run_args.user_id or 1)
        if run_args.button_usage is not None
        else None
    )
    usage = normalize_button_usage(button_usage)
    behavior = behavior_cls(
        attendance_prob=1.0,
        lazy_good_bias=0.0,
        max_new_per_day=run_args.learn_limit,
        max_reviews_per_day=run_args.review_limit,
        max_cost_per_day=cost_limit,
        priority_fn=priority_fn,
        first_rating_prob=usage["first_rating_prob"],
        review_rating_prob=usage["review_rating_prob"],
        learning_rating_prob=usage["learning_rating_prob"],
        relearning_rating_prob=usage["relearning_rating_prob"],
    )
    if short_term_source:
        state_rating_costs = usage["state_rating_costs"]
        cost_model = cost_model_cls(
            state_costs=StateRatingCosts(
                learning=state_rating_costs[0],
                review=state_rating_costs[1],
                relearning=state_rating_costs[2],
            )
        )
    else:
        cost_model = cost_model_cls(
            state_costs=StateRatingCosts(
                learning=usage["learn_costs"],
                review=usage["review_costs"],
                relearning=usage["review_costs"],
            )
        )
    try:
        if run_args.engine == "vectorized":
            stats = _run_vectorized(
                run_args,
                env,
                agent,
                behavior,
                cost_model,
                progress_callback,
                short_term_source,
                learning_steps,
                relearning_steps,
            )
        else:
            stats = run_simulation(
                days=run_args.days,
                deck_size=run_args.deck,
                environment=env,
                scheduler=agent,
                behavior=behavior,
                cost_model=cost_model,
                fuzz=run_args.fuzz,
                seed_fn=rng.random,
                progress=False,
                progress_callback=progress_callback,
                short_term_loops_limit=getattr(
                    run_args, "short_term_loops_limit", None
                ),
            )
    finally:
        progress_close()
    if not run_args.no_log:
        simulate_cli._write_log(run_args, stats)
    if run_args.plot:
        simulate_cli.plot_simulation(stats, run_args)


def _run_vectorized(
    run_args: argparse.Namespace,
    env,
    agent,
    behavior,
    cost_model,
    progress_callback,
    short_term_source,
    learning_steps,
    relearning_steps,
):
    from simulator.vectorized import simulate as simulate_vectorized

    try:
        return simulate_vectorized(
            days=run_args.days,
            deck_size=run_args.deck,
            environment=env,
            scheduler=agent,
            behavior=behavior,
            cost_model=cost_model,
            seed=run_args.seed,
            device=run_args.torch_device,
            fuzz=run_args.fuzz,
            progress=False,
            progress_callback=progress_callback,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            short_term_threshold=getattr(run_args, "short_term_threshold", 0.5),
            short_term_loops_limit=getattr(run_args, "short_term_loops_limit", None),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()

    if args.step == 0:
        raise SystemExit("--step must be non-zero.")
    if not (0.0 < args.start_retention <= 1.0):
        raise SystemExit("--start-retention must be within (0, 1].")
    if not (0.0 < args.end_retention <= 1.0):
        raise SystemExit("--end-retention must be within (0, 1].")
    if args.step > 0 and args.start_retention > args.end_retention:
        raise SystemExit(
            "--start-retention must be <= --end-retention when --step is positive."
        )
    if args.step < 0 and args.start_retention < args.end_retention:
        raise SystemExit(
            "--start-retention must be >= --end-retention when --step is negative."
        )

    if not args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")

    repo_root = REPO_ROOT

    import simulate as simulate_cli
    from simulator import simulate as run_simulation
    from simulator.behavior import StochasticBehavior
    from simulator.cost import StatefulCostModel
    from simulator.core import new_first_priority, review_first_priority

    user_id = args.user_id or 1
    log_dir = args.log_dir or (
        repo_root / "logs" / "retention_sweep" / f"user_{user_id}"
    )
    envs = parse_csv(args.env)
    if not envs:
        raise SystemExit("No environments specified. Use --env.")
    schedulers = parse_csv(args.sched) or ["fsrs6"]
    try:
        scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name, _, _ in scheduler_specs:
        if name not in simulate_cli.SCHEDULER_FACTORIES:
            raise SystemExit(f"Unknown scheduler '{name}'.")
    dr_schedulers: List[str] = []
    non_dr_schedulers: List[str] = []
    fixed_schedulers = [spec for spec in scheduler_specs if spec[0] == "fixed"]
    has_sspmmc = any(spec[0] == "sspmmc" for spec in scheduler_specs)
    for name, _, _ in scheduler_specs:
        if name in {"sspmmc", "fixed"}:
            continue
        if scheduler_uses_desired_retention(name):
            if name not in dr_schedulers:
                dr_schedulers.append(name)
        else:
            if name not in non_dr_schedulers:
                non_dr_schedulers.append(name)
    run_dr = bool(dr_schedulers)
    run_sspmmc = has_sspmmc
    run_fixed = bool(fixed_schedulers)
    run_non_dr = bool(non_dr_schedulers)
    if not run_dr and not run_sspmmc and not run_fixed and not run_non_dr:
        raise SystemExit("No schedulers specified. Use --sched to select runs.")

    sspmmc_policies = _resolve_policy_paths(args, repo_root, run_sspmmc)
    if run_sspmmc and not sspmmc_policies:
        raise SystemExit("No SSP-MMC policies found. Provide --sspmmc-policy-dir.")

    priority_fn = (
        review_first_priority if args.priority == "review-first" else new_first_priority
    )

    try:
        for environment in envs:
            if run_dr:
                for scheduler in dr_schedulers:
                    for dr in _dr_values(
                        args.start_retention, args.end_retention, args.step
                    ):
                        run_args = argparse.Namespace(**vars(args))
                        run_args.environment = environment
                        run_args.scheduler = scheduler
                        run_args.fixed_interval = None
                        run_args.desired_retention = dr
                        run_args.scheduler_spec = scheduler
                        run_args.log_dir = log_dir
                        _run_once(
                            run_args,
                            priority_fn,
                            run_simulation,
                            simulate_cli,
                            StochasticBehavior,
                            StatefulCostModel,
                        )

            if run_non_dr:
                for scheduler in non_dr_schedulers:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.environment = environment
                    run_args.scheduler = scheduler
                    run_args.fixed_interval = None
                    run_args.desired_retention = None
                    run_args.scheduler_spec = scheduler
                    run_args.log_dir = log_dir
                    _run_once(
                        run_args,
                        priority_fn,
                        run_simulation,
                        simulate_cli,
                        StochasticBehavior,
                        StatefulCostModel,
                    )

            if run_fixed:
                for scheduler, fixed_interval, raw in fixed_schedulers:
                    fixed_interval = normalize_fixed_interval(fixed_interval)
                    run_args = argparse.Namespace(**vars(args))
                    run_args.environment = environment
                    run_args.scheduler = scheduler
                    run_args.fixed_interval = fixed_interval
                    run_args.desired_retention = None
                    run_args.scheduler_spec = raw
                    run_args.log_dir = log_dir
                    _run_once(
                        run_args,
                        priority_fn,
                        run_simulation,
                        simulate_cli,
                        StochasticBehavior,
                        StatefulCostModel,
                    )

            if run_sspmmc:
                for policy_path in sspmmc_policies:
                    run_args = argparse.Namespace(**vars(args))
                    run_args.environment = environment
                    run_args.scheduler = "sspmmc"
                    run_args.sspmmc_policy = policy_path
                    run_args.desired_retention = args.sspmmc_desired_retention
                    run_args.scheduler_spec = "sspmmc"
                    run_args.log_dir = log_dir
                    _run_once(
                        run_args,
                        priority_fn,
                        run_simulation,
                        simulate_cli,
                        StochasticBehavior,
                        StatefulCostModel,
                    )
    finally:
        elapsed = time.perf_counter() - start_time
        if not args.no_summary:
            print(f"Total sweep time: {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
