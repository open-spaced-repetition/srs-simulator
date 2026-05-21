from __future__ import annotations

import argparse
import math
from pathlib import Path

import simulate as simulate_cli

from experiments.retention_sweep.cli_utils import (
    add_benchmark_args,
    add_fuzz_arg,
    add_retention_range_args,
    add_torch_device_arg,
    parse_csv,
)
from experiments.single_card_tradeoff.core.defaults import (
    DEFAULT_FIXED_INTERVALS,
    DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT,
    DEFAULT_FSRS6_ORACLE_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_INFINITE_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_INTERVAL_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_RETENTION_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_POLICY,
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
    DEFAULT_UVFA_PPO_POLICY,
    DEFAULT_UVFA_PPO_RNN_INTERVAL_POLICY,
    FSRS6_ADR_SCHEDULERS,
    FSRS6_ORACLE_DISTILL_SCHEDULER,
    FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
    FSRS6_ORACLE_INFINITE_SCHEDULER,
    FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER,
    FSRS6_ORACLE_INTERVAL_SCHEDULER,
    FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER,
    FSRS6_ORACLE_SCHEDULER,
    FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
    FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER,
    MIN_TARGET_RETENTION,
    UVFA_PPO_RNN_INTERVAL_SCHEDULER,
    UVFA_PPO_SCHEDULER,
)
from experiments.single_card_tradeoff.core.retention_space import (
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.run_monitoring import (
    add_run_monitoring_args,
)
from simulator.defaults import (
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
)
from simulator.retention_sweep.grid import dr_values
from simulator.scheduler_spec import format_float, parse_scheduler_spec


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
            f"Values must be >= {format_float(MIN_TARGET_RETENTION)}. "
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
        "--fsrs6-adr-policy",
        type=Path,
        default=None,
        help=(
            "Path to one FSRS6 ADR policy JSON when --sched contains fsrs6_adr, "
            "fsrs6_adr_time, or fsrs6_default_adr."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-policy-root",
        type=Path,
        default=None,
        help=(
            "Root containing expanded FSRS6 ADR policy JSONs, usually "
            "train-overfit/train_outputs."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-train-run-root",
        type=Path,
        default=None,
        help=(
            "Experiment run root for expanded FSRS6 ADR policies. When no ADR "
            "policy source is passed, tradeoff.py uses the local "
            f"{DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT} artifact if it exists."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-policy-manifest",
        type=Path,
        default=None,
        help="TOML manifest with explicit FSRS6 ADR policy entries.",
    )
    parser.add_argument(
        "--fsrs6-adr-lambda-values",
        default=None,
        help=(
            "Comma-separated lambda values to keep when using an expanded FSRS6 "
            "ADR policy source."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-policy",
        type=Path,
        default=DEFAULT_UVFA_PPO_POLICY,
        help=(
            "Path to a UVFA PPO policy checkpoint when --sched contains uvfa_ppo. "
            "Create one with python -m experiments.single_card_tradeoff.cli.uvfa_ppo."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for uvfa_ppo. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--uvfa-ppo-policy."
        ),
    )
    parser.add_argument(
        "--oracle-distill-policy",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_DISTILL_POLICY,
        help=(
            "Path to an FSRS-6 oracle-distilled policy checkpoint when --sched "
            "contains fsrs6_oracle_distill. Create one with "
            "python -m experiments.single_card_tradeoff.cli.oracle_distill."
        ),
    )
    parser.add_argument(
        "--oracle-distill-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for fsrs6_oracle_distill. "
            "Defaults to 0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--oracle-distill-policy."
        ),
    )
    parser.add_argument(
        "--oracle-retention-distill-policy",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_RETENTION_DISTILL_POLICY,
        help=(
            "Path to an FSRS6 oracle desired-retention distillation checkpoint "
            "when --sched contains fsrs6_oracle_retention_distill."
        ),
    )
    parser.add_argument(
        "--oracle-retention-distill-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for "
            "fsrs6_oracle_retention_distill. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--oracle-retention-distill-policy."
        ),
    )
    parser.add_argument(
        "--oracle-infinite-distill-policy",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_INFINITE_DISTILL_POLICY,
        help=(
            "Path to an FSRS6 average-reward oracle distillation checkpoint "
            "when --sched contains fsrs6_oracle_infinite_distill."
        ),
    )
    parser.add_argument(
        "--oracle-infinite-distill-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for "
            "fsrs6_oracle_infinite_distill. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--oracle-infinite-distill-policy."
        ),
    )
    parser.add_argument(
        "--oracle-stationary-finite-distill-policy",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_POLICY,
        help=(
            "Path to an FSRS6 stationary finite-lifecycle oracle distillation "
            "checkpoint when --sched contains "
            "fsrs6_oracle_stationary_finite_distill."
        ),
    )
    parser.add_argument(
        "--oracle-stationary-finite-distill-policy-template",
        default=None,
        help=(
            "Per-user FSRS6 stationary finite-lifecycle oracle distillation "
            "checkpoint template for multi-user runs. Must contain "
            "{user_id}, for example artifacts/.../user_{user_id}_policy.pt. "
            "When omitted, --oracle-stationary-finite-distill-policy is shared."
        ),
    )
    parser.add_argument(
        "--oracle-stationary-finite-distill-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for "
            "fsrs6_oracle_stationary_finite_distill. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--oracle-stationary-finite-distill-policy."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-rnn-interval-policy",
        type=Path,
        default=DEFAULT_UVFA_PPO_RNN_INTERVAL_POLICY,
        help=(
            "Path to a recurrent UVFA PPO log-interval checkpoint when --sched "
            "contains uvfa_ppo_rnn_interval. Create one with "
            "python -m experiments.single_card_tradeoff.cli.uvfa_ppo_rnn_interval."
        ),
    )
    parser.add_argument(
        "--uvfa-ppo-rnn-interval-cost-weights",
        default=None,
        help=(
            "Comma-separated scalarization weights for uvfa_ppo_rnn_interval. "
            "Defaults to the cost_weights saved in --uvfa-ppo-rnn-interval-policy."
        ),
    )
    parser.add_argument(
        "--oracle-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for fsrs6_oracle. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024."
        ),
    )
    parser.add_argument(
        "--oracle-action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help=(
            "Discrete desired-retention actions available to fsrs6_oracle. "
            f"Values must be >= {format_float(MIN_TARGET_RETENTION)}."
        ),
    )
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=64,
        help="Stability grid size for fsrs6_oracle.",
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=32,
        help="Difficulty grid size for fsrs6_oracle.",
    )
    parser.add_argument(
        "--oracle-interval-chunk-size",
        type=int,
        default=64,
        help="Interval candidates per Bellman-backup chunk for fsrs6_oracle_interval.",
    )
    parser.add_argument(
        "--oracle-infinite-max-iterations",
        type=int,
        default=128,
        help="Policy-iteration limit for fsrs6_oracle_infinite.",
    )
    parser.add_argument(
        "--oracle-infinite-tolerance",
        type=float,
        default=1e-10,
        help="Convergence tolerance for fsrs6_oracle_infinite.",
    )
    parser.add_argument(
        "--oracle-stationary-finite-max-iterations",
        type=int,
        default=128,
        help="Policy-iteration limit for fsrs6_oracle_stationary_finite.",
    )
    parser.add_argument(
        "--oracle-stationary-finite-tolerance",
        type=float,
        default=1e-10,
        help="Convergence tolerance for fsrs6_oracle_stationary_finite.",
    )
    parser.add_argument(
        "--oracle-interval-distill-policy",
        type=Path,
        default=DEFAULT_FSRS6_ORACLE_INTERVAL_DISTILL_POLICY,
        help=(
            "Path to an FSRS6 oracle interval distillation checkpoint when --sched "
            "contains fsrs6_oracle_interval_distill."
        ),
    )
    parser.add_argument(
        "--oracle-interval-distill-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help=(
            "Comma-separated scalarization weights for "
            "fsrs6_oracle_interval_distill. Defaults to "
            "0,1,2,4,8,16,32,48,64,96,128,192,256,320,384,512,1024. "
            "Pass an empty string to use the cost_weights saved in "
            "--oracle-interval-distill-policy."
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
    parser.add_argument(
        "--user-ids",
        default=None,
        help=(
            "Comma-separated user IDs to evaluate in one vectorized run. "
            "Use --user-id for the current single-user behavior."
        ),
    )
    add_benchmark_args(parser)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=None,
        help=(
            "Optional Anki button usage JSONL. If omitted, built-in rating and "
            "cost defaults are used. Review Markov transitions require "
            "--review-markov-transition."
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
        "--engine",
        choices=["vectorized", "event"],
        default="vectorized",
        help=(
            "Simulation engine. Vectorized treats particles as iid card samples; "
            "event is mainly for debugging small particle counts."
        ),
    )
    add_torch_device_arg(parser)
    add_run_monitoring_args(parser)
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
        default=Path("artifacts/single_card_tradeoff/results.csv"),
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
        "--regret-auc-out",
        type=Path,
        default=None,
        help=(
            "CSV output path for pairwise same-target time saved AUC. "
            "Defaults to the main CSV path with _regret_auc before the suffix."
        ),
    )
    parser.add_argument(
        "--no-regret-auc",
        action="store_true",
        help="Skip writing the pairwise same-target time saved AUC CSV.",
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


def _parse_user_ids_csv(value: str, *, label: str = "--user-ids") -> list[int]:
    user_ids: list[int] = []
    for item in parse_csv(value):
        try:
            user_id = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {label} value '{item}'.") from exc
        if user_id <= 0:
            raise SystemExit(f"{label} must contain positive integers.")
        user_ids.append(user_id)
    if not user_ids:
        raise SystemExit(f"{label} must contain at least one user ID.")
    if len(set(user_ids)) != len(user_ids):
        raise SystemExit(f"{label} contains duplicate user IDs.")
    return user_ids


def _resolve_user_ids(args: argparse.Namespace) -> list[int]:
    raw_user_ids = getattr(args, "user_ids", None)
    if raw_user_ids is None or not str(raw_user_ids).strip():
        user_ids = [int(args.user_id or 1)]
    else:
        user_ids = _parse_user_ids_csv(str(raw_user_ids))
        if args.user_id is not None and (
            len(user_ids) > 1 or user_ids[0] != args.user_id
        ):
            raise SystemExit(
                "--user-id cannot be combined with multi-value --user-ids."
            )
    if len(user_ids) > 1 and args.engine == "event":
        raise SystemExit(
            "--engine event supports only one user; use --engine vectorized."
        )
    return user_ids


def _resolve_user_policy_template(template: str, *, user_id: int) -> Path:
    if "{user_id}" not in template:
        raise SystemExit("Per-user policy templates must contain {user_id}.")
    path = Path(template.format(user_id=int(user_id))).expanduser()
    return path


def _resolve_stationary_finite_distill_policy_path(
    args: argparse.Namespace,
    *,
    user_id: int,
    multiuser: bool,
) -> Path:
    template = getattr(args, "oracle_stationary_finite_distill_policy_template", None)
    if template is not None and str(template).strip():
        return _resolve_user_policy_template(str(template), user_id=user_id)
    if multiuser:
        return Path(args.oracle_stationary_finite_distill_policy)
    return Path(args.oracle_stationary_finite_distill_policy)


def _run_specs(args: argparse.Namespace) -> list[tuple[str, str, float | None]]:
    specs: list[tuple[str, str, float | None]] = []
    for raw in parse_csv(args.sched) or ["fsrs6_default"]:
        try:
            name, fixed_interval, raw_spec = parse_scheduler_spec(raw)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        custom_schedulers = {
            FSRS6_ORACLE_SCHEDULER,
            FSRS6_ORACLE_INFINITE_SCHEDULER,
            FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER,
            FSRS6_ORACLE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
            FSRS6_ORACLE_INTERVAL_SCHEDULER,
            FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER,
            FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER,
            UVFA_PPO_SCHEDULER,
            UVFA_PPO_RNN_INTERVAL_SCHEDULER,
            *FSRS6_ADR_SCHEDULERS,
        }
        if (
            name not in simulate_cli.SCHEDULER_FACTORIES
            and name not in custom_schedulers
        ):
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
            values.append(value)
        if not values:
            raise SystemExit("--target-retentions must include at least one value.")
        validate_retention_values(values, name="Target retention")
        return values
    try:
        values = dr_values(args.start_retention, args.end_retention, args.step)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_retention_values(values, name="Target retention")
    return values


def _fsrs6_adr_lambda_values(args: argparse.Namespace) -> tuple[float, ...] | None:
    raw = getattr(args, "fsrs6_adr_lambda_values", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        return tuple(_parse_float_list(raw, label="FSRS6 ADR lambda value"))
    values = tuple(float(value) for value in raw)
    if not values:
        return None
    if any(not math.isfinite(value) for value in values):
        raise SystemExit("FSRS6 ADR lambda values must be finite.")
    return values
