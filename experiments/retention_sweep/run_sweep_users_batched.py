from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import DEFAULT_MAX_LANES_PER_BATCH
from simulator.batched_sweep.config import load_batched_sweep_config
from simulator.batched_sweep.plan import build_batched_sweep_plan
from simulator.batched_sweep.execution import run_batches

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
    add_user_range_args,
    has_flag,
    parse_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run multi-user retention sweeps with batched vectorized simulation.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Batched sweep TOML config. Direct CLI flags override config values.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier included in written JSONL log filenames.",
    )
    add_user_range_args(parser)
    parser.set_defaults(user_ids=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Number of users per outer batch. Defaults to auto-sized batches "
            "derived from --max-lanes-per-batch."
        ),
    )
    parser.add_argument(
        "--max-lanes-per-batch",
        type=int,
        default=DEFAULT_MAX_LANES_PER_BATCH,
        help=(
            "Maximum expanded simulation lanes per in-process batch. "
            f"Defaults to {DEFAULT_MAX_LANES_PER_BATCH}."
        ),
    )
    add_env_sched_args(
        parser,
        env_default="lstm",
        sched_default="fsrs6,anki_sm2,memrise",
        env_help="Comma-separated environments to sweep (lstm, fsrs6, fsrs6_default).",
        sched_help=(
            "Comma-separated schedulers to sweep "
            "(fsrs6, fsrs6_default, fsrs3, fsrs3_default, lstm, "
            "anki_sm2, memrise, fixed, fsrs6_adr, fsrs6_adp)."
        ),
    )
    add_retention_range_args(parser)
    add_common_sim_args(
        parser,
    )
    add_button_usage_arg(parser, default_path=DEFAULT_BUTTON_USAGE_PATH)
    add_benchmark_args(parser)
    parser.add_argument(
        "--fsrs6-adr-policy",
        type=Path,
        default=None,
        help="Path to an FSRS6 ADR policy JSON when using --sched fsrs6_adr.",
    )
    parser.add_argument(
        "--fsrs6-adr-policy-root",
        type=Path,
        default=None,
        help=(
            "Root containing trained FSRS6 ADR policy artifacts, usually "
            "train-overfit/train_outputs."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-train-run-root",
        type=Path,
        default=None,
        help=(
            "Training run root; treated as "
            "<root>/train-overfit/train_outputs for FSRS6 ADR policy discovery."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-policy-manifest",
        type=Path,
        default=None,
        help="TOML manifest with [[policies]] FSRS6 ADR entries.",
    )
    parser.add_argument(
        "--fsrs6-adr-lambda-values",
        default=None,
        help=(
            "Optional comma-separated lambda values to select from an FSRS6 ADR "
            "policy root or manifest."
        ),
    )
    parser.add_argument(
        "--fsrs6-adp-policy",
        type=Path,
        default=None,
        help="Path to an FSRS6 ADP policy JSON when using --sched fsrs6_adp.",
    )
    parser.add_argument(
        "--fsrs6-adp-policy-root",
        type=Path,
        default=None,
        help=(
            "Root containing trained FSRS6 ADP policy artifacts, usually "
            "train-overfit/train_outputs."
        ),
    )
    parser.add_argument(
        "--fsrs6-adp-train-run-root",
        type=Path,
        default=None,
        help=(
            "Training run root; treated as "
            "<root>/train-overfit/train_outputs for FSRS6 ADP policy discovery."
        ),
    )
    parser.add_argument(
        "--fsrs6-adp-policy-manifest",
        type=Path,
        default=None,
        help="TOML manifest with [[policies]] FSRS6 ADP entries.",
    )
    parser.add_argument(
        "--fsrs6-adp-lambda-values",
        default=None,
        help=(
            "Optional comma-separated lambda values to select from an FSRS6 ADP "
            "policy root or manifest."
        ),
    )
    add_log_args(
        parser, log_dir_default=None, include_no_log=True, include_no_progress=True
    )
    parser.add_argument(
        "--log-layout",
        choices=["user", "sweep"],
        default="user",
        help=(
            "Log directory layout. user writes <log-dir>/user_<id>/sched_... "
            "(default); sweep writes <log-dir>/sched_.../user_<id>."
        ),
    )
    parser.add_argument(
        "--diagnostic-csv-logs",
        action="store_true",
        help=(
            "Write per-user daily CSV sidecars and batch GPU CSV logs for "
            "diagnosing simulation behavior. Batched retention sweeps skip CSVs "
            "by default to limit disk usage."
        ),
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
        threshold_help="Short-term cutoff in days (used by sched mode).",
    )
    add_torch_device_arg(parser)
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute batches across "
            "(e.g. 0,1). Each batch is assigned a device round-robin."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print expanded lane counts without simulation.",
    )
    args = parser.parse_args(argv)
    if args.config is not None:
        config = load_batched_sweep_config(args.config, repo_root=REPO_ROOT)
        args = _merge_config_args(cli_args=args, config_args=config.args, argv=argv)
    if isinstance(args.fsrs6_adr_lambda_values, str):
        args.fsrs6_adr_lambda_values = tuple(
            float(item) for item in parse_csv(args.fsrs6_adr_lambda_values)
        )
    if isinstance(args.fsrs6_adp_lambda_values, str):
        args.fsrs6_adp_lambda_values = tuple(
            float(item) for item in parse_csv(args.fsrs6_adp_lambda_values)
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    envs = parse_csv(args.env)
    schedulers = parse_csv(args.sched)
    plan = build_batched_sweep_plan(
        repo_root=REPO_ROOT,
        args=args,
        envs=envs,
        schedulers=schedulers,
    )
    if args.dry_run:
        _print_dry_run(plan)
        return 0
    overall = None
    if not args.no_progress:
        overall = tqdm(
            total=plan.total_user_days,
            desc="Overall",
            unit="user-day",
            leave=True,
        )

    run_batches(
        args=args,
        ctx=plan.ctx,
        batches=plan.batches,
        batches_by_env=plan.batches_by_env,
        devices=plan.devices,
        device=plan.device,
        overall=overall,
    )
    if overall is not None:
        overall.close()
    return 0


def _merge_config_args(
    *,
    cli_args: argparse.Namespace,
    config_args: argparse.Namespace,
    argv: list[str],
) -> argparse.Namespace:
    user_range_overridden = has_flag(argv, "--start-user") or has_flag(
        argv, "--end-user"
    )
    if not user_range_overridden:
        cli_args.user_ids = getattr(config_args, "user_ids", None)
        cli_args.start_user = config_args.start_user
        cli_args.end_user = config_args.end_user
    else:
        cli_args.user_ids = None

    flag_map = {
        "batch_size": ("--batch-size",),
        "max_lanes_per_batch": ("--max-lanes-per-batch",),
        "env": ("--env",),
        "sched": ("--sched",),
        "run_id": ("--run-id",),
        "start_retention": ("--start-retention",),
        "end_retention": ("--end-retention",),
        "step": ("--step",),
        "days": ("--days",),
        "deck": ("--deck",),
        "learn_limit": ("--learn-limit",),
        "review_limit": ("--review-limit",),
        "cost_limit_minutes": ("--cost-limit-minutes",),
        "seed": ("--seed",),
        "priority": ("--priority",),
        "scheduler_priority": ("--scheduler-priority",),
        "button_usage": ("--button-usage",),
        "benchmark_result": ("--benchmark-result",),
        "benchmark_partition": ("--benchmark-partition",),
        "srs_benchmark_root": ("--srs-benchmark-root",),
        "fsrs6_adr_policy": ("--fsrs6-adr-policy",),
        "fsrs6_adr_policy_root": ("--fsrs6-adr-policy-root",),
        "fsrs6_adr_train_run_root": ("--fsrs6-adr-train-run-root",),
        "fsrs6_adr_policy_manifest": ("--fsrs6-adr-policy-manifest",),
        "fsrs6_adr_lambda_values": ("--fsrs6-adr-lambda-values",),
        "fsrs6_adp_policy": ("--fsrs6-adp-policy",),
        "fsrs6_adp_policy_root": ("--fsrs6-adp-policy-root",),
        "fsrs6_adp_train_run_root": ("--fsrs6-adp-train-run-root",),
        "fsrs6_adp_policy_manifest": ("--fsrs6-adp-policy-manifest",),
        "fsrs6_adp_lambda_values": ("--fsrs6-adp-lambda-values",),
        "log_dir": ("--log-dir",),
        "log_layout": ("--log-layout",),
        "no_log": ("--no-log",),
        "no_progress": ("--no-progress",),
        "diagnostic_csv_logs": ("--diagnostic-csv-logs",),
        "fuzz": ("--fuzz",),
        "short_term_source": ("--short-term-source",),
        "learning_steps": ("--learning-steps",),
        "relearning_steps": ("--relearning-steps",),
        "short_term_threshold": ("--short-term-threshold",),
        "short_term_loops_limit": ("--short-term-loops-limit",),
        "torch_device": ("--torch-device",),
        "cuda_devices": ("--cuda-devices",),
        "dry_run": ("--dry-run",),
    }
    for attr, flags in flag_map.items():
        if not any(has_flag(argv, flag) for flag in flags):
            setattr(cli_args, attr, getattr(config_args, attr))
    cli_args.env_batch_overrides = getattr(config_args, "env_batch_overrides", {})
    return cli_args


def _print_dry_run(plan) -> None:
    print("Batched sweep dry run")
    print(f"user batches: {len(plan.batches)}")
    for environment in plan.ctx.envs:
        batches = plan.batches_by_env.get(environment, plan.batches)
        max_lanes = plan.max_lanes_per_batch_by_env.get(environment)
        print(
            f"{environment} user batches: {len(batches)} "
            f"(max lanes: {max_lanes if max_lanes is not None else 'unlimited'})"
        )
    print(f"expanded lanes: {plan.total_lanes}")
    print(f"user-days: {plan.total_user_days}")
    print(f"envs: {','.join(plan.ctx.envs)}")
    print(f"schedulers: {','.join(plan.ctx.schedulers)}")
    print(f"log layout: {plan.ctx.log_layout}")
    print(f"log root: {plan.ctx.log_root}")
    if plan.example_log_dir is not None:
        print(f"example log dir: {plan.example_log_dir}")
    if plan.ctx.fsrs6_adr_policy_specs:
        print(f"fsrs6_adr policies: {len(plan.ctx.fsrs6_adr_policy_specs)}")


if __name__ == "__main__":
    raise SystemExit(main())
