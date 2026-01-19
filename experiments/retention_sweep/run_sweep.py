from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a desired-retention sweep for simulate.py.",
    )
    parser.add_argument("--start", type=int, default=70, help="Start retention x100.")
    parser.add_argument("--end", type=int, default=99, help="End retention x100.")
    parser.add_argument(
        "--step", type=int, default=1, help="Step size for retention x100."
    )
    parser.add_argument(
        "--environment",
        default="lstm",
        help="Environment name passed to simulate.py.",
    )
    parser.add_argument(
        "--environments",
        default=None,
        help="Comma-separated list of environments to sweep.",
    )
    parser.add_argument(
        "--schedulers",
        default="fsrs6",
        help="Comma-separated list of schedulers to sweep (include sspmmc to run policies).",
    )
    parser.add_argument("--days", type=int, default=1825, help="Simulation days.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        "--priority",
        choices=["review-first", "new-first"],
        default="review-first",
        help="Action priority passed to simulate.py.",
    )
    parser.add_argument(
        "--scheduler-priority",
        default="low_retrievability",
        help="FSRS6 priority hint passed to simulate.py.",
    )
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
            "(defaults to ../SSP-MMC-FSRS/outputs/policies)."
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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store logs (defaults to logs/retention_sweep).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable the simulate.py progress bar.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots during the sweep (slower, opens windows).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing logs to disk.",
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_policy_paths(
    args: argparse.Namespace, repo_root: Path, run_sspmmc: bool
) -> List[Path]:
    if args.sspmmc_policies:
        paths = [Path(path) for path in _parse_csv(args.sspmmc_policies)]
        return [path.resolve() for path in paths]

    if args.sspmmc_policy:
        return [args.sspmmc_policy.resolve()]

    policy_dir = args.sspmmc_policy_dir
    if policy_dir is None and run_sspmmc:
        candidate = repo_root.parent / "SSP-MMC-FSRS" / "outputs" / "policies"
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
    rng = random.Random(run_args.seed)
    env = simulate_cli.ENVIRONMENT_FACTORIES[run_args.environment](run_args)
    agent = simulate_cli.SCHEDULER_FACTORIES[run_args.scheduler](run_args)
    cost_limit = (
        run_args.cost_limit_minutes * 60.0
        if run_args.cost_limit_minutes is not None
        else None
    )
    behavior = behavior_cls(
        attendance_prob=1.0,
        lazy_good_bias=0.0,
        max_new_per_day=run_args.learn_limit,
        max_reviews_per_day=run_args.review_limit,
        max_cost_per_day=cost_limit,
        priority_fn=priority_fn,
    )
    cost_model = cost_model_cls()
    stats = run_simulation(
        days=run_args.days,
        deck_size=run_args.deck,
        environment=env,
        scheduler=agent,
        behavior=behavior,
        cost_model=cost_model,
        seed_fn=rng.random,
        progress=run_args.progress,
    )
    if not run_args.no_log:
        simulate_cli._write_log(run_args, stats)
    if run_args.plot:
        simulate_cli.plot_simulation(stats)


def main() -> None:
    args = parse_args()

    if args.step == 0:
        raise SystemExit("--step must be non-zero.")
    if args.start > args.end and args.step > 0:
        raise SystemExit("--start must be <= --end when --step is positive.")

    if not args.plot:
        os.environ.setdefault("MPLBACKEND", "Agg")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import simulate as simulate_cli
    from simulator import simulate as run_simulation
    from simulator.behavior import StochasticBehavior
    from simulator.cost import StatefulCostModel
    from simulator.core import new_first_priority, review_first_priority

    user_id = args.user_id or 1
    log_dir = args.log_dir or (
        repo_root / "logs" / "retention_sweep" / f"user_{user_id}"
    )
    envs = _parse_csv(args.environments) or [args.environment]
    schedulers = _parse_csv(args.schedulers) or ["fsrs6"]
    dr_schedulers = [scheduler for scheduler in schedulers if scheduler != "sspmmc"]
    has_sspmmc = "sspmmc" in schedulers
    run_dr = bool(dr_schedulers)
    run_sspmmc = has_sspmmc
    if not run_dr and not run_sspmmc:
        raise SystemExit("No schedulers specified. Use --schedulers to select runs.")

    sspmmc_policies = _resolve_policy_paths(args, repo_root, run_sspmmc)
    if run_sspmmc and not sspmmc_policies:
        raise SystemExit("No SSP-MMC policies found. Provide --sspmmc-policy-dir.")

    priority_fn = (
        review_first_priority if args.priority == "review-first" else new_first_priority
    )

    for environment in envs:
        if run_dr:
            for scheduler in dr_schedulers:
                for i in range(args.start, args.end + 1, args.step):
                    dr = i / 100.0
                    run_args = argparse.Namespace(**vars(args))
                    run_args.environment = environment
                    run_args.scheduler = scheduler
                    run_args.desired_retention = dr
                    run_args.log_dir = log_dir
                    print(
                        f"Running env={environment} scheduler={scheduler} "
                        f"desired_retention={dr:.2f}"
                    )
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
                run_args.log_dir = log_dir
                print(f"Running env={environment} sspmmc_policy={policy_path}")
                _run_once(
                    run_args,
                    priority_fn,
                    run_simulation,
                    simulate_cli,
                    StochasticBehavior,
                    StatefulCostModel,
                )


if __name__ == "__main__":
    main()
