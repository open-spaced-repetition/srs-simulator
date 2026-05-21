from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from experiments.single_card_tradeoff.config import (
    load_single_card_fsrs6_config,
    SingleCardFSRS6Config,
    SingleCardRuntimeContext,
    single_card_runtime_context_from_args,
    SUPPORTED_SINGLE_CARD_ENVS,
)
from experiments.single_card_tradeoff.defaults import (
    DEFAULT_FIXED_INTERVALS,
    DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT,
    DEFAULT_FSRS6_ORACLE_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_INFINITE_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_INTERVAL_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_RETENTION_DISTILL_POLICY,
    DEFAULT_FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_POLICY,
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS,
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
from experiments.single_card_tradeoff.results import (
    build_regret_auc_rows as _build_regret_auc_rows,
    pareto_frontier as _pareto_frontier,
    row_user_id as _row_user_id,
    write_csv as _write_csv,
    write_regret_auc_csv as _write_regret_auc_csv,
)
from experiments.single_card_tradeoff.run_monitoring import (
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.retention_space import validate_retention_values
from experiments.single_card_tradeoff.single_card_env import FSRS6SingleCardBatch
from experiments.single_card_tradeoff.types import (
    BatchRow,
    SchedulerPoint,
    SimMetrics,
    UserContext,
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
from simulator.models.lstm import _resolve_benchmark_weights as _resolve_lstm_weights
from simulator.models.lstm_batch import PackedLSTMWeights
from simulator.batched_sweep.fsrs6_adr_policy import (
    FSRS6ADRPolicySpec,
    resolve_fsrs6_adr_policy_specs,
)
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.retention_sweep.grid import dr_values
from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps
from simulator.schedulers.fixed import FixedBatchSchedulerOps
from simulator.schedulers.fsrs import FSRS3BatchSchedulerOps, FSRS6BatchSchedulerOps
from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps
from simulator.schedulers.hlr import HLRBatchSchedulerOps
from simulator.schedulers.lstm import LSTMBatchSchedulerOps
from simulator.schedulers.memrise import MemriseBatchSchedulerOps
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.batched_engine.multiuser_types import MultiUserBehavior, MultiUserCost

__all__ = [
    "DEFAULT_FIXED_INTERVALS",
    "DEFAULT_FSRS6_ORACLE_DISTILL_POLICY",
    "DEFAULT_FSRS6_ORACLE_INFINITE_DISTILL_POLICY",
    "DEFAULT_FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_POLICY",
    "DEFAULT_FSRS6_ORACLE_INTERVAL_DISTILL_POLICY",
    "DEFAULT_FSRS6_ORACLE_RETENTION_DISTILL_POLICY",
    "DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS",
    "DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS",
    "DEFAULT_TARGET_RETENTIONS",
    "DEFAULT_UVFA_PPO_POLICY",
    "DEFAULT_UVFA_PPO_RNN_INTERVAL_POLICY",
    "FSRS6_ORACLE_DISTILL_SCHEDULER",
    "FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER",
    "FSRS6_ORACLE_INFINITE_SCHEDULER",
    "FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER",
    "FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER",
    "FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER",
    "FSRS6_ORACLE_INTERVAL_SCHEDULER",
    "FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER",
    "FSRS6_ORACLE_SCHEDULER",
    "MIN_TARGET_RETENTION",
    "UVFA_PPO_RNN_INTERVAL_SCHEDULER",
    "UVFA_PPO_SCHEDULER",
    "main",
]


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
            "Create one with experiments/single_card_tradeoff/uvfa_ppo.py."
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
            "experiments/single_card_tradeoff/oracle_distill.py."
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
            "experiments/single_card_tradeoff/uvfa_ppo_rnn_interval.py."
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


def _user_args(args: argparse.Namespace, user_id: int) -> argparse.Namespace:
    user_args = argparse.Namespace(**vars(args))
    user_args.user_id = int(user_id)
    return user_args


def _runtime_context(args: argparse.Namespace) -> SingleCardRuntimeContext:
    context = getattr(args, "single_card_runtime_context", None)
    if isinstance(context, SingleCardRuntimeContext):
        return context
    out_path = getattr(args, "out", None)
    return single_card_runtime_context_from_args(
        args,
        repo_root=REPO_ROOT,
        output_dir=out_path.parent if isinstance(out_path, Path) else None,
    )


def _load_user_contexts(
    args: argparse.Namespace,
    *,
    environment_name: str,
    user_ids: Sequence[int],
) -> list[UserContext]:
    contexts: list[UserContext] = []
    for user_id in user_ids:
        user_args = _user_args(args, user_id)
        fsrs_config = load_single_card_fsrs6_config(
            user_args,
            environment=environment_name,
        )
        behavior, cost_model = _make_behavior(user_args)
        contexts.append(
            UserContext(
                user_id=int(user_id),
                args=user_args,
                runtime_context=_runtime_context(user_args),
                fsrs_config=fsrs_config,
                behavior=behavior,
                cost_model=cost_model,
            )
        )
    return contexts


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


def _resolve_torch_device(
    args: argparse.Namespace,
    *,
    prefer_cuda: bool = False,
) -> torch.device:
    if args.torch_device:
        return torch.device(args.torch_device)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_torch_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    return None


def _fsrs_config_kwargs(
    fsrs_config: SingleCardFSRS6Config | None,
) -> dict[str, Any]:
    if fsrs_config is None:
        return {}
    return {
        "fsrs_weights": fsrs_config.fsrs_weights,
        "first_rating_prob": fsrs_config.first_rating_prob,
        "review_rating_prob": fsrs_config.review_rating_prob,
        "learning_costs": fsrs_config.learning_costs,
        "review_costs": fsrs_config.review_costs,
    }


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


def _sort_fsrs6_adr_policy_specs(
    specs: Sequence[FSRS6ADRPolicySpec],
) -> tuple[FSRS6ADRPolicySpec, ...]:
    def key(spec: FSRS6ADRPolicySpec) -> tuple[int, int, float, float, str]:
        if spec.baseline_desired_retention is not None:
            return (
                spec.user_id,
                0,
                float(spec.baseline_desired_retention),
                float("-inf")
                if spec.lambda_value is None
                else float(spec.lambda_value),
                str(spec.path),
            )
        if spec.policy_index is not None:
            return (spec.user_id, 1, float(spec.policy_index), 0.0, str(spec.path))
        return (spec.user_id, 2, 0.0, 0.0, str(spec.path))

    return tuple(sorted(specs, key=key))


def _load_single_fsrs6_adr_policy_spec(
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
) -> tuple[FSRS6ADRPolicySpec, ...]:
    policy_path = Path(args.fsrs6_adr_policy).expanduser()
    if not policy_path.exists():
        raise SystemExit(f"FSRS6 ADR policy not found: {policy_path}")
    policy = FSRS6ADRPolicy.from_json(policy_path)
    return tuple(
        FSRS6ADRPolicySpec(
            user_id=int(user_id),
            baseline_desired_retention=policy.baseline_desired_retention,
            lambda_value=None,
            policy_index=None,
            path=policy_path.resolve(),
        )
        for user_id in user_ids
    )


def _load_fsrs6_adr_policy_specs(
    args: argparse.Namespace,
    *,
    retention_values: Sequence[float],
    user_ids: Sequence[int] | None = None,
) -> tuple[FSRS6ADRPolicySpec, ...]:
    policy_path = getattr(args, "fsrs6_adr_policy", None)
    policy_root = getattr(args, "fsrs6_adr_policy_root", None)
    train_run_root = getattr(args, "fsrs6_adr_train_run_root", None)
    policy_manifest = getattr(args, "fsrs6_adr_policy_manifest", None)
    expanded_sources = [
        policy_root is not None,
        train_run_root is not None,
        policy_manifest is not None,
    ]
    if policy_path is not None and any(expanded_sources):
        raise SystemExit(
            "--fsrs6-adr-policy cannot be combined with "
            "--fsrs6-adr-policy-root, --fsrs6-adr-train-run-root, or "
            "--fsrs6-adr-policy-manifest."
        )
    if sum(expanded_sources) > 1:
        raise SystemExit(
            "Configure only one expanded FSRS6 ADR policy source: "
            "--fsrs6-adr-policy-root, --fsrs6-adr-train-run-root, or "
            "--fsrs6-adr-policy-manifest."
        )
    if policy_path is not None:
        return _load_single_fsrs6_adr_policy_spec(
            args,
            user_ids=user_ids or [int(args.user_id or 1)],
        )
    if not any(expanded_sources):
        if DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT.exists():
            train_run_root = DEFAULT_FSRS6_ADR_TRAIN_RUN_ROOT
        else:
            raise SystemExit(
                "FSRS6 ADR tradeoff requires one policy source: "
                "--fsrs6-adr-policy, --fsrs6-adr-policy-root, "
                "--fsrs6-adr-train-run-root, or --fsrs6-adr-policy-manifest."
            )
    try:
        return _sort_fsrs6_adr_policy_specs(
            resolve_fsrs6_adr_policy_specs(
                user_ids=[
                    int(user_id) for user_id in (user_ids or [int(args.user_id or 1)])
                ],
                dr_values=retention_values,
                policy_root=policy_root,
                train_run_root=train_run_root,
                policy_manifest=policy_manifest,
                lambda_values=_fsrs6_adr_lambda_values(args),
            )
        )
    except (FileNotFoundError, ValueError) as exc:
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
        review_markov_transition=(
            usage.get("long_term_transition")
            if getattr(args, "review_markov_transition", False)
            else None
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


def _row_from_stats(
    args: argparse.Namespace,
    *,
    user_id: int | None = None,
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
        "user_id": int(user_id if user_id is not None else (args.user_id or 1)),
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
        "review_markov_transition": bool(
            getattr(args, "review_markov_transition", False)
        ),
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
        stats = _run_single_batched_point(
            args,
            environment_name=environment_name,
            scheduler_name=scheduler_name,
            scheduler_spec=scheduler_spec,
            fixed_interval=fixed_interval,
            desired_retention=desired_retention,
            env=env,
            scheduler=scheduler,
            seed=seed,
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
        and scheduler_name
        in {"fsrs6", "fsrs6_default", "fsrs3", "fsrs3_default", "hlr", "lstm"}
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


def _fsrs3_weights(obj: Any, *, label: str) -> tuple[float, ...]:
    params = getattr(obj, "params", None)
    weights = getattr(params, "weights", None)
    if weights is None or len(weights) != 13:
        raise ValueError(f"{label} must expose FSRS3 params.weights.")
    return tuple(float(weight) for weight in weights)


def _hlr_weights(obj: Any, *, label: str) -> tuple[float, ...]:
    weights = getattr(obj, "w", None)
    if weights is None or len(weights) != 3:
        raise ValueError(f"{label} must expose HLR weights.")
    return tuple(float(weight) for weight in weights)


def _lstm_batch_weights(
    args: argparse.Namespace,
    *,
    rows: int | None = None,
    user_ids: Sequence[int] | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedLSTMWeights:
    benchmark_root = (
        Path(args.srs_benchmark_root)
        if args.srs_benchmark_root is not None
        else REPO_ROOT.parent / "srs-benchmark"
    )
    if user_ids is None:
        if rows is None:
            raise ValueError("rows is required when user_ids is not provided.")
        user_ids = [int(args.user_id or 1) for _ in range(rows)]
    paths: list[Path] = []
    for user_id in user_ids:
        path = _resolve_lstm_weights(
            int(user_id),
            benchmark_root,
            short_term=False,
        )
        if path is None:
            raise FileNotFoundError(
                f"LSTM weights for user {int(user_id)} not found under "
                f"{benchmark_root / 'weights'}"
            )
        paths.append(path)
    return PackedLSTMWeights.from_paths(
        paths,
        use_duration_feature=False,
        device=device,
        dtype=dtype,
    )


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
    user_contexts: Sequence[UserContext] | None = None,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    if user_contexts is None:
        behavior, cost_model = _make_behavior(args)
        behaviors = [behavior for _ in range(rows)]
        cost_models = [cost_model for _ in range(rows)]
    else:
        if len(user_contexts) != rows:
            raise ValueError("user_contexts length must match rows.")
        behaviors = [context.behavior for context in user_contexts]
        cost_models = [context.cost_model for context in user_contexts]

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
            (rows,), args.particles, device=device, dtype=torch.int64
        ),
        max_reviews_per_day=torch.full(
            (rows,), args.particles, device=device, dtype=torch.int64
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


def _point_run_args(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    point: SchedulerPoint,
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.env = environment_name
    run_args.environment = environment_name
    run_args.scheduler = scheduler_name
    run_args.sched = point.scheduler_spec
    run_args.scheduler_spec = point.scheduler_spec
    run_args.fixed_interval = point.fixed_interval
    run_args.desired_retention = point.desired_retention
    run_args.short_term_source = None
    run_args.short_term = False
    run_args.sspmmc_policy = None
    run_args.lstm_interval_mode = "integer"
    run_args.lstm_min_interval = 1.0
    return run_args


def _batch_row_chunks(
    rows: Sequence[BatchRow],
    *,
    batch_size: int,
) -> list[list[BatchRow]]:
    if not rows:
        return []
    chunk_size = len(rows) if batch_size <= 0 else batch_size
    return [
        list(rows[idx : idx + chunk_size]) for idx in range(0, len(rows), chunk_size)
    ]


def _expand_batch_rows(
    user_contexts: Sequence[UserContext],
    points: Sequence[SchedulerPoint],
) -> list[BatchRow]:
    return [
        BatchRow(user_context=context, point=point)
        for point in points
        for context in user_contexts
    ]


def _build_batch_scheduler_ops(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    batch_rows: Sequence[BatchRow],
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    def desired_values() -> list[float]:
        values: list[float] = []
        for row in batch_rows:
            if row.point.desired_retention is None:
                raise ValueError(f"{scheduler_name} requires desired_retention.")
            values.append(float(row.point.desired_retention))
        return values

    if scheduler_name == "fixed":
        intervals = []
        for row in batch_rows:
            if row.point.fixed_interval is None:
                raise ValueError("fixed scheduler requires fixed_interval.")
            intervals.append(float(row.point.fixed_interval))
        return FixedBatchSchedulerOps(
            interval=torch.tensor(intervals, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )

    if scheduler_name in {"fsrs6", "fsrs6_default"}:
        schedulers = [
            simulate_cli.SCHEDULER_FACTORIES[scheduler_name](
                _point_run_args(
                    row.user_context.args,
                    environment_name=environment_name,
                    scheduler_name=scheduler_name,
                    point=row.point,
                )
            )
            for row in batch_rows
        ]
        desired = desired_values()
        return FSRS6BatchSchedulerOps(
            weights=torch.tensor(
                [
                    _fsrs6_weights(scheduler, label=scheduler_name)
                    for scheduler in schedulers
                ],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(desired, device=device, dtype=dtype),
            bounds=Bounds(),
            priority_mode=args.scheduler_priority,
            device=device,
            dtype=dtype,
        )

    if scheduler_name in {"fsrs3", "fsrs3_default"}:
        schedulers = [
            simulate_cli.SCHEDULER_FACTORIES[scheduler_name](
                _point_run_args(
                    row.user_context.args,
                    environment_name=environment_name,
                    scheduler_name=scheduler_name,
                    point=row.point,
                )
            )
            for row in batch_rows
        ]
        desired = desired_values()
        return FSRS3BatchSchedulerOps(
            weights=torch.tensor(
                [
                    _fsrs3_weights(scheduler, label=scheduler_name)
                    for scheduler in schedulers
                ],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(desired, device=device, dtype=dtype),
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )

    if scheduler_name == "hlr":
        schedulers = [
            simulate_cli.SCHEDULER_FACTORIES[scheduler_name](
                _point_run_args(
                    row.user_context.args,
                    environment_name=environment_name,
                    scheduler_name=scheduler_name,
                    point=row.point,
                )
            )
            for row in batch_rows
        ]
        desired = desired_values()
        return HLRBatchSchedulerOps(
            weights=torch.tensor(
                [
                    _hlr_weights(scheduler, label=scheduler_name)
                    for scheduler in schedulers
                ],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(desired, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )

    if scheduler_name == "lstm":
        desired = desired_values()
        return LSTMBatchSchedulerOps(
            _lstm_batch_weights(
                args,
                user_ids=[row.user_context.user_id for row in batch_rows],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(desired, device=device, dtype=dtype),
            interval_mode="integer",
            min_interval=1.0,
            device=device,
            dtype=dtype,
        )

    if scheduler_name == "anki_sm2":
        scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](
            _point_run_args(
                batch_rows[0].user_context.args,
                environment_name=environment_name,
                scheduler_name=scheduler_name,
                point=batch_rows[0].point,
            )
        )
        return AnkiSM2BatchSchedulerOps(
            graduating_interval=scheduler.graduating_interval,
            easy_interval=scheduler.easy_interval,
            easy_bonus=scheduler.easy_bonus,
            hard_interval_factor=scheduler.hard_interval_factor,
            ease_start=scheduler.ease_start,
            ease_min=scheduler.ease_min,
            ease_max=scheduler.ease_max,
            new_interval_factor=scheduler.new_interval_factor,
            interval_multiplier=scheduler.interval_multiplier,
            device=device,
            dtype=dtype,
        )

    if scheduler_name == "memrise":
        scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](
            _point_run_args(
                batch_rows[0].user_context.args,
                environment_name=environment_name,
                scheduler_name=scheduler_name,
                point=batch_rows[0].point,
            )
        )
        return MemriseBatchSchedulerOps(scheduler, device=device, dtype=dtype)

    raise SystemExit(
        f"{scheduler_name} is not supported by single-card --engine vectorized."
    )


def _run_vectorized_batch_rows(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    points: Sequence[SchedulerPoint],
    user_contexts: Sequence[UserContext],
    seed: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    if environment_name not in {"fsrs6", "fsrs6_default"}:
        raise SystemExit(
            f"{environment_name} is not supported by single-card --engine vectorized."
        )
    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    dtype = torch.float64
    rows = _expand_batch_rows(user_contexts, points)
    output_rows: list[dict[str, Any]] = []
    for batch_index, batch_rows in enumerate(
        _batch_row_chunks(rows, batch_size=args.target_batch_size)
    ):
        row_count = len(batch_rows)
        env_ops = FSRS6BatchEnvOps(
            weights=torch.tensor(
                [row.user_context.fsrs_config.fsrs_weights for row in batch_rows],
                device=device,
                dtype=dtype,
            ),
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )
        sched_ops = _build_batch_scheduler_ops(
            args,
            environment_name=environment_name,
            scheduler_name=scheduler_name,
            batch_rows=batch_rows,
            device=device,
            dtype=dtype,
        )
        behavior, cost_model = _make_multiuser_behavior_cost(
            args,
            rows=row_count,
            device=device,
            dtype=dtype,
            user_contexts=[row.user_context for row in batch_rows],
        )
        label_suffix = "" if len(rows) == row_count else f" batch {batch_index + 1}"
        start = time.perf_counter()
        stats_by_row = simulate_multiuser(
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
            progress_label=f"{progress_label}{label_suffix}",
        )
        runtime_per_row = (time.perf_counter() - start) / max(1, row_count)
        for batch_row, stats in zip(batch_rows, stats_by_row, strict=True):
            output_rows.append(
                _row_from_stats(
                    args,
                    user_id=batch_row.user_context.user_id,
                    environment_name=environment_name,
                    scheduler_name=scheduler_name,
                    scheduler_spec=batch_row.point.scheduler_spec,
                    fixed_interval=batch_row.point.fixed_interval,
                    desired_retention=batch_row.point.desired_retention,
                    seed=seed,
                    stats=stats,
                    runtime_s=runtime_per_row,
                )
            )
    return output_rows


def _run_single_batched_point(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    fixed_interval: float | None,
    desired_retention: float | None,
    env: Any,
    scheduler: Any,
    seed: int,
):
    if environment_name not in {"fsrs6", "fsrs6_default"}:
        raise SystemExit(
            f"{environment_name} is not supported by single-card --engine vectorized."
        )
    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    dtype = torch.float64
    env_ops = FSRS6BatchEnvOps(
        weights=torch.tensor(
            [_fsrs6_weights(env, label=environment_name)],
            device=device,
            dtype=dtype,
        ),
        bounds=Bounds(),
        device=device,
        dtype=dtype,
    )
    if scheduler_name == "fixed":
        if fixed_interval is None:
            raise ValueError("fixed scheduler requires fixed_interval.")
        sched_ops = FixedBatchSchedulerOps(
            interval=torch.tensor([fixed_interval], device=device, dtype=dtype),
            device=device,
            dtype=dtype,
        )
    elif scheduler_name in {"fsrs6", "fsrs6_default"}:
        if desired_retention is None:
            raise ValueError(f"{scheduler_name} requires desired_retention.")
        sched_ops = FSRS6BatchSchedulerOps(
            weights=torch.tensor(
                [_fsrs6_weights(scheduler, label=scheduler_name)],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(
                [desired_retention], device=device, dtype=dtype
            ),
            bounds=Bounds(),
            priority_mode=args.scheduler_priority,
            device=device,
            dtype=dtype,
        )
    elif scheduler_name in {"fsrs3", "fsrs3_default"}:
        if desired_retention is None:
            raise ValueError(f"{scheduler_name} requires desired_retention.")
        sched_ops = FSRS3BatchSchedulerOps(
            weights=torch.tensor(
                [_fsrs3_weights(scheduler, label=scheduler_name)],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(
                [desired_retention], device=device, dtype=dtype
            ),
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "hlr":
        if desired_retention is None:
            raise ValueError("hlr requires desired_retention.")
        sched_ops = HLRBatchSchedulerOps(
            weights=torch.tensor(
                [_hlr_weights(scheduler, label=scheduler_name)],
                device=device,
                dtype=dtype,
            ),
            desired_retention=torch.tensor(
                [desired_retention], device=device, dtype=dtype
            ),
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "lstm":
        if desired_retention is None:
            raise ValueError("lstm requires desired_retention.")
        sched_ops = LSTMBatchSchedulerOps(
            _lstm_batch_weights(args, rows=1, device=device, dtype=dtype),
            desired_retention=torch.tensor(
                [desired_retention], device=device, dtype=dtype
            ),
            interval_mode="integer",
            min_interval=1.0,
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "anki_sm2":
        sched_ops = AnkiSM2BatchSchedulerOps(
            graduating_interval=scheduler.graduating_interval,
            easy_interval=scheduler.easy_interval,
            easy_bonus=scheduler.easy_bonus,
            hard_interval_factor=scheduler.hard_interval_factor,
            ease_start=scheduler.ease_start,
            ease_min=scheduler.ease_min,
            ease_max=scheduler.ease_max,
            new_interval_factor=scheduler.new_interval_factor,
            interval_multiplier=scheduler.interval_multiplier,
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "memrise":
        sched_ops = MemriseBatchSchedulerOps(
            scheduler,
            device=device,
            dtype=dtype,
        )
    else:
        raise SystemExit(
            f"{scheduler_name} is not supported by single-card --engine vectorized."
        )
    behavior, cost_model = _make_multiuser_behavior_cost(
        args, rows=1, device=device, dtype=dtype
    )
    stats_list = simulate_multiuser(
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
        progress_label=f"{environment_name}/{scheduler_spec}",
    )
    return stats_list[0]


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
    env_weights = torch.tensor(
        [_fsrs6_weights(env, label=environment_name) for _ in range(target_count)],
        device=device,
        dtype=dtype,
    )
    env_ops = FSRS6BatchEnvOps(
        weights=env_weights,
        bounds=Bounds(),
        device=device,
        dtype=dtype,
    )
    desired = torch.tensor(desired_values, device=device, dtype=dtype)
    if scheduler_name in {"fsrs6", "fsrs6_default"}:
        scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](run_args)
        sched_weights = torch.tensor(
            [
                _fsrs6_weights(scheduler, label=scheduler_name)
                for _ in range(target_count)
            ],
            device=device,
            dtype=dtype,
        )
        sched_ops = FSRS6BatchSchedulerOps(
            weights=sched_weights,
            desired_retention=desired,
            bounds=Bounds(),
            priority_mode=args.scheduler_priority,
            device=device,
            dtype=dtype,
        )
    elif scheduler_name in {"fsrs3", "fsrs3_default"}:
        scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](run_args)
        sched_weights = torch.tensor(
            [
                _fsrs3_weights(scheduler, label=scheduler_name)
                for _ in range(target_count)
            ],
            device=device,
            dtype=dtype,
        )
        sched_ops = FSRS3BatchSchedulerOps(
            weights=sched_weights,
            desired_retention=desired,
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "hlr":
        scheduler = simulate_cli.SCHEDULER_FACTORIES[scheduler_name](run_args)
        sched_weights = torch.tensor(
            [
                _hlr_weights(scheduler, label=scheduler_name)
                for _ in range(target_count)
            ],
            device=device,
            dtype=dtype,
        )
        sched_ops = HLRBatchSchedulerOps(
            weights=sched_weights,
            desired_retention=desired,
            device=device,
            dtype=dtype,
        )
    elif scheduler_name == "lstm":
        sched_weights = _lstm_batch_weights(
            args,
            rows=target_count,
            device=device,
            dtype=dtype,
        )
        sched_ops = LSTMBatchSchedulerOps(
            sched_weights,
            desired_retention=desired,
            interval_mode="integer",
            min_interval=1.0,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported target-batched scheduler '{scheduler_name}'.")
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


def _policy_batches(
    specs: Sequence[FSRS6ADRPolicySpec],
    *,
    batch_size: int,
) -> list[list[FSRS6ADRPolicySpec]]:
    if not specs:
        return []
    chunk_size = len(specs) if batch_size <= 0 else batch_size
    return [
        list(specs[idx : idx + chunk_size]) for idx in range(0, len(specs), chunk_size)
    ]


def _same_bounds(left: Bounds, right: Bounds) -> bool:
    return (
        math.isclose(left.s_min, right.s_min, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(left.s_max, right.s_max, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(left.d_min, right.d_min, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(left.d_max, right.d_max, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_fsrs6_adr_policy_batch(
    policies: Sequence[FSRS6ADRPolicy],
    specs: Sequence[FSRS6ADRPolicySpec],
) -> None:
    if not policies:
        raise ValueError("FSRS6 ADR policy batch must not be empty.")
    first = policies[0]
    for policy, spec in zip(policies[1:], specs[1:], strict=True):
        if policy.feature_version != first.feature_version:
            raise SystemExit(
                "Cannot batch FSRS6 ADR policies with mixed feature versions: "
                f"{first.feature_version!r} and {policy.feature_version!r} "
                f"({spec.path})."
            )
        if not math.isclose(
            policy.retention_min,
            first.retention_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            policy.retention_max,
            first.retention_max,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemExit(
                "Cannot batch FSRS6 ADR policies with mixed retention bounds: "
                f"{spec.path}."
            )
        if not _same_bounds(policy.bounds, first.bounds):
            raise SystemExit(
                "Cannot batch FSRS6 ADR policies with mixed FSRS state bounds: "
                f"{spec.path}."
            )


def _fsrs6_adr_point_label(spec: FSRS6ADRPolicySpec) -> str:
    if spec.policy_index is not None:
        return f"policy_{spec.policy_index}"
    if spec.baseline_desired_retention is not None:
        label = f"dr_{format_float(spec.baseline_desired_retention)}"
        if spec.lambda_value is not None:
            label += f"_lambda_{format_float(spec.lambda_value)}"
        return label
    if spec.path.name == "policy.json":
        return spec.path.parent.name
    return spec.path.stem


def _run_fsrs6_adr(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    seed: int,
    retention_values: Sequence[float],
    user_contexts: Sequence[UserContext] | None = None,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            f"{scheduler_name} currently supports only --env fsrs6_default or "
            "--env fsrs6 in single-card tradeoff."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            f"{scheduler_name} is supported only with --engine vectorized."
        )

    if user_contexts is None:
        user_contexts = _load_user_contexts(
            args,
            environment_name=environment_name,
            user_ids=[int(args.user_id or 1)],
        )
    context_by_user_id = {context.user_id: context for context in user_contexts}
    specs = _load_fsrs6_adr_policy_specs(
        args,
        retention_values=retention_values,
        user_ids=[context.user_id for context in user_contexts],
    )
    device = _resolve_torch_device(args, prefer_cuda=True)
    dtype = torch.float64

    rows: list[dict[str, Any]] = []
    for batch_index, batch_specs in enumerate(
        _policy_batches(specs, batch_size=args.target_batch_size)
    ):
        policies = [FSRS6ADRPolicy.from_json(spec.path) for spec in batch_specs]
        _validate_fsrs6_adr_policy_batch(policies, batch_specs)
        row_count = len(batch_specs)
        first_policy = policies[0]
        batch_contexts = []
        for spec in batch_specs:
            context = context_by_user_id.get(int(spec.user_id))
            if context is None:
                raise SystemExit(
                    f"No user context loaded for ADR policy user {spec.user_id}."
                )
            batch_contexts.append(context)
        env_weights = torch.tensor(
            [context.fsrs_config.fsrs_weights for context in batch_contexts],
            device=device,
            dtype=dtype,
        )
        env_ops = FSRS6BatchEnvOps(
            weights=env_weights,
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )
        sched_weights = torch.tensor(
            [context.fsrs_config.fsrs_weights for context in batch_contexts],
            device=device,
            dtype=dtype,
        )
        coefficients = torch.tensor(
            [policy.coefficients for policy in policies],
            device=device,
            dtype=dtype,
        )
        sched_ops = FSRS6ADRBatchSchedulerOps(
            weights=sched_weights,
            policy=first_policy,
            coefficients=coefficients,
            bounds=first_policy.bounds,
            priority_mode=args.scheduler_priority,
            device=device,
            dtype=dtype,
        )
        behavior, cost_model = _make_multiuser_behavior_cost(
            args,
            rows=row_count,
            device=device,
            dtype=dtype,
            user_contexts=batch_contexts,
        )
        label_suffix = (
            " policies" if len(specs) == row_count else f" policies {batch_index + 1}"
        )
        start = time.perf_counter()
        stats_by_policy = simulate_multiuser(
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
            progress_label=f"{environment_name}/{scheduler_spec}{label_suffix}",
        )
        runtime_per_policy = (time.perf_counter() - start) / max(1, row_count)
        for spec, policy, stats in zip(
            batch_specs, policies, stats_by_policy, strict=True
        ):
            row = _row_from_stats(
                args,
                user_id=spec.user_id,
                environment_name=environment_name,
                scheduler_name=scheduler_name,
                scheduler_spec=scheduler_spec,
                fixed_interval=None,
                desired_retention=None,
                seed=seed,
                stats=stats,
                runtime_s=runtime_per_policy,
            )
            row.update(
                {
                    "fsrs6_adr_policy": str(spec.path),
                    "fsrs6_adr_baseline_desired_retention": (
                        spec.baseline_desired_retention
                    ),
                    "fsrs6_adr_lambda_value": spec.lambda_value,
                    "fsrs6_adr_policy_index": spec.policy_index,
                    "fsrs6_adr_policy_title": policy.title,
                    "fsrs6_adr_feature_version": policy.feature_version,
                    "fsrs6_adr_point_label": _fsrs6_adr_point_label(spec),
                }
            )
            rows.append(row)
    return rows


def _load_policy_checkpoint(
    policy_path: Path,
    *,
    device: torch.device,
    policy_label: str,
    train_hint: str,
    expected_policy_type: str | None = None,
) -> tuple[Any, list[float], list[float], float, str]:
    if not policy_path.exists():
        raise SystemExit(
            f"{policy_label} policy not found: {policy_path}. {train_hint}"
        )

    from experiments.single_card_tradeoff.policy_net import PolicyValueNet

    checkpoint = torch.load(policy_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Invalid {policy_label} checkpoint: {policy_path}")
    if (
        expected_policy_type is not None
        and checkpoint.get("policy_type") != expected_policy_type
    ):
        raise SystemExit(f"{policy_label} checkpoint has unexpected policy_type.")
    raw_actions = checkpoint.get("action_retentions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise SystemExit(f"{policy_label} checkpoint is missing action_retentions.")
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit(f"{policy_label} checkpoint is missing cost_weights.")
    action_retentions = [float(value) for value in raw_actions]
    policy_cost_weights = [float(value) for value in raw_cost_weights]
    validate_retention_values(
        action_retentions,
        name=f"{policy_label} action retention",
    )
    obs_dim = int(checkpoint.get("obs_dim", 7))
    hidden_size = int(checkpoint.get("hidden_size", 96))
    network = str(checkpoint.get("network", "mlp"))
    network_depth = int(checkpoint.get("network_depth", 3))
    obs_mode = str(checkpoint.get("obs_mode", "basic"))
    model = PolicyValueNet(
        obs_dim=obs_dim,
        action_count=len(action_retentions),
        hidden_size=hidden_size,
        architecture=network,
        depth=network_depth,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"{policy_label} checkpoint is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return (
        model,
        action_retentions,
        policy_cost_weights,
        max(policy_cost_weights),
        obs_mode,
    )


def _load_uvfa_ppo_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float, str]:
    return _load_policy_checkpoint(
        args.uvfa_ppo_policy,
        device=device,
        policy_label="UVFA PPO",
        train_hint=(
            "Train one with `uv run experiments/single_card_tradeoff/uvfa_ppo.py "
            f"--model-out {args.uvfa_ppo_policy}` or pass --uvfa-ppo-policy."
        ),
    )


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


def _load_fsrs6_oracle_distill_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float, str]:
    return _load_policy_checkpoint(
        args.oracle_distill_policy,
        device=device,
        policy_label="FSRS-6 oracle-distilled",
        train_hint=(
            "Train one with `uv run experiments/single_card_tradeoff/oracle_distill.py "
            f"--model-out {args.oracle_distill_policy}` or pass "
            "--oracle-distill-policy."
        ),
    )


def _load_fsrs6_oracle_infinite_distill_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float, str]:
    return _load_policy_checkpoint(
        args.oracle_infinite_distill_policy,
        device=device,
        policy_label="FSRS-6 average-reward oracle-distilled",
        train_hint=(
            "Train one with `uv run "
            "experiments/single_card_tradeoff/oracle_infinite_distill.py "
            f"--model-out {args.oracle_infinite_distill_policy}` or pass "
            "--oracle-infinite-distill-policy."
        ),
    )


def _load_fsrs6_oracle_stationary_finite_distill_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float, str]:
    return _load_policy_checkpoint(
        args.oracle_stationary_finite_distill_policy,
        device=device,
        policy_label="FSRS-6 stationary finite-lifecycle oracle-distilled",
        expected_policy_type="fsrs6_oracle_stationary_finite_distill",
        train_hint=(
            "Train one with `uv run "
            "experiments/single_card_tradeoff/"
            "oracle_stationary_finite_distill.py "
            f"--model-out {args.oracle_stationary_finite_distill_policy}` or pass "
            "--oracle-stationary-finite-distill-policy."
        ),
    )


def _oracle_distill_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "oracle_distill_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="Oracle-distilled cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("Oracle-distilled cost weights must be >= 0.")
    return values


def _oracle_infinite_distill_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "oracle_infinite_distill_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="Average-reward oracle-distilled cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("Average-reward oracle-distilled cost weights must be >= 0.")
    return values


def _oracle_stationary_finite_distill_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "oracle_stationary_finite_distill_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(
        raw,
        label="Stationary finite oracle-distilled cost weight",
    )
    if any(value < 0.0 for value in values):
        raise SystemExit(
            "Stationary finite oracle-distilled cost weights must be >= 0."
        )
    return values


def _load_fsrs6_oracle_retention_distill_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], list[float], float, str, float, float, float]:
    if not args.oracle_retention_distill_policy.exists():
        raise SystemExit(
            "FSRS6 oracle retention distill policy not found: "
            f"{args.oracle_retention_distill_policy}. Train one with "
            "`uv run experiments/single_card_tradeoff/oracle_retention_distill.py "
            f"--model-out {args.oracle_retention_distill_policy}` or pass "
            "--oracle-retention-distill-policy."
        )

    from experiments.single_card_tradeoff.oracle_retention_distill import (
        RetentionDistillNet,
    )

    checkpoint = torch.load(args.oracle_retention_distill_policy, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(
            f"Invalid FSRS6 oracle retention distill checkpoint: "
            f"{args.oracle_retention_distill_policy}"
        )
    if checkpoint.get("policy_type") != "fsrs6_oracle_retention_distill":
        raise SystemExit(
            "FSRS6 oracle retention distill checkpoint has unexpected policy_type."
        )
    if checkpoint.get("action_mode") != "desired_retention":
        raise SystemExit(
            "FSRS6 oracle retention distill checkpoint must use "
            "action_mode=desired_retention."
        )
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit(
            "FSRS6 oracle retention distill checkpoint is missing cost_weights."
        )
    raw_action_retentions = checkpoint.get("action_retentions")
    if not isinstance(raw_action_retentions, list) or not raw_action_retentions:
        raise SystemExit(
            "FSRS6 oracle retention distill checkpoint is missing action_retentions."
        )
    policy_cost_weights = [float(value) for value in raw_cost_weights]
    action_retentions = [float(value) for value in raw_action_retentions]
    obs_dim = int(checkpoint.get("obs_dim", 4))
    obs_mode = str(checkpoint.get("obs_mode", "oracle_rho4"))
    hidden_size = int(checkpoint.get("hidden_size", 16))
    network = str(checkpoint.get("network", "residual"))
    network_depth = int(checkpoint.get("network_depth", 2))
    model = RetentionDistillNet(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        action_count=len(action_retentions),
        architecture=network,
        depth=network_depth,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(
            "FSRS6 oracle retention distill checkpoint is missing model_state_dict."
        )
    model.load_state_dict(state_dict)
    model.eval()
    validate_retention_values(
        action_retentions,
        name="FSRS6 oracle retention distill action retention",
    )
    retention_min = float(checkpoint.get("retention_min", MIN_TARGET_RETENTION))
    retention_max = float(checkpoint.get("retention_max", 0.999))
    if not MIN_TARGET_RETENTION <= retention_min < retention_max < 1.0:
        raise SystemExit(
            "FSRS6 oracle retention distill retention range must be within [0.5, 1)."
        )
    terminal_snap_ratio = float(checkpoint.get("terminal_snap_ratio", 0.0))
    return (
        model,
        action_retentions,
        policy_cost_weights,
        max(policy_cost_weights),
        obs_mode,
        retention_min,
        retention_max,
        terminal_snap_ratio,
    )


def _oracle_retention_distill_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "oracle_retention_distill_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="Oracle retention distill cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("Oracle retention distill cost weights must be >= 0.")
    return values


def _load_uvfa_ppo_rnn_interval_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], float, int | None]:
    if not args.uvfa_ppo_rnn_interval_policy.exists():
        raise SystemExit(
            "Recurrent UVFA PPO interval policy not found: "
            f"{args.uvfa_ppo_rnn_interval_policy}. Train one with "
            "`uv run experiments/single_card_tradeoff/uvfa_ppo_rnn_interval.py --model-out "
            f"{args.uvfa_ppo_rnn_interval_policy}` or pass "
            "--uvfa-ppo-rnn-interval-policy."
        )

    from experiments.single_card_tradeoff.uvfa_ppo_rnn_interval import (
        RecurrentIntervalPolicyValueNet,
    )

    checkpoint = torch.load(args.uvfa_ppo_rnn_interval_policy, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(
            f"Invalid recurrent UVFA PPO checkpoint: "
            f"{args.uvfa_ppo_rnn_interval_policy}"
        )
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit("Recurrent UVFA PPO checkpoint is missing cost_weights.")
    policy_cost_weights = [float(value) for value in raw_cost_weights]
    obs_dim = int(checkpoint.get("obs_dim", 10))
    hidden_size = int(checkpoint.get("hidden_size", 128))
    initial_mean_interval = float(checkpoint.get("initial_mean_interval", 32.0))
    initial_log_std = float(checkpoint.get("initial_log_std", 0.7))
    max_interval_days_raw = checkpoint.get("max_interval_days")
    max_interval_days = (
        int(max_interval_days_raw) if max_interval_days_raw is not None else None
    )
    model = RecurrentIntervalPolicyValueNet(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        initial_mean_interval=initial_mean_interval,
        initial_log_std=initial_log_std,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit("Recurrent UVFA PPO checkpoint is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, policy_cost_weights, max(policy_cost_weights), max_interval_days


def _uvfa_ppo_rnn_interval_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "uvfa_ppo_rnn_interval_cost_weights", None)
    if raw is None or not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="Recurrent UVFA PPO cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("Recurrent UVFA PPO cost weights must be >= 0.")
    return values


def _load_fsrs6_oracle_interval_distill_policy(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, list[float], float, float, float]:
    if not args.oracle_interval_distill_policy.exists():
        raise SystemExit(
            "FSRS6 oracle interval distill policy not found: "
            f"{args.oracle_interval_distill_policy}. Train one with "
            "`uv run experiments/single_card_tradeoff/oracle_interval_distill.py --model-out "
            f"{args.oracle_interval_distill_policy}` or pass "
            "--oracle-interval-distill-policy."
        )

    from experiments.single_card_tradeoff.oracle_interval_distill import (
        IntervalDistillNet,
    )

    checkpoint = torch.load(args.oracle_interval_distill_policy, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(
            f"Invalid FSRS6 oracle interval distill checkpoint: "
            f"{args.oracle_interval_distill_policy}"
        )
    if checkpoint.get("policy_type") != "fsrs6_oracle_interval_distill":
        raise SystemExit(
            "FSRS6 oracle interval distill checkpoint has unexpected policy_type."
        )
    if checkpoint.get("action_mode") != "log_interval":
        raise SystemExit(
            "FSRS6 oracle interval distill checkpoint must use action_mode=log_interval."
        )
    if checkpoint.get("obs_mode") != "oracle":
        raise SystemExit(
            "FSRS6 oracle interval distill checkpoint must use obs_mode=oracle."
        )
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit(
            "FSRS6 oracle interval distill checkpoint is missing cost_weights."
        )
    policy_cost_weights = [float(value) for value in raw_cost_weights]
    obs_dim = int(checkpoint.get("obs_dim", 4))
    hidden_size = int(checkpoint.get("hidden_size", 96))
    network = str(checkpoint.get("network", "residual"))
    network_depth = int(checkpoint.get("network_depth", 3))
    model = IntervalDistillNet(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        architecture=network,
        depth=network_depth,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit(
            "FSRS6 oracle interval distill checkpoint is missing model_state_dict."
        )
    model.load_state_dict(state_dict)
    model.eval()
    log_interval_bias = float(checkpoint.get("log_interval_bias", 0.0))
    terminal_snap_ratio = float(checkpoint.get("terminal_snap_ratio", 0.0))
    return (
        model,
        policy_cost_weights,
        max(policy_cost_weights),
        log_interval_bias,
        terminal_snap_ratio,
    )


def _oracle_interval_distill_cost_weights(
    args: argparse.Namespace,
    *,
    policy_cost_weights: Sequence[float],
) -> list[float]:
    raw = getattr(args, "oracle_interval_distill_cost_weights", None)
    if raw is None:
        return [float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS]
    if not raw.strip():
        return [float(value) for value in policy_cost_weights]
    values = _parse_float_list(raw, label="FSRS6 oracle interval distill cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("FSRS6 oracle interval distill cost weights must be >= 0.")
    return values


def _row_from_uvfa_metrics(
    args: argparse.Namespace,
    *,
    user_id: int | None = None,
    environment_name: str,
    scheduler_name: str,
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
        "user_id": int(user_id if user_id is not None else (args.user_id or 1)),
        "environment": environment_name,
        "scheduler": scheduler_name,
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
        "engine": scheduler_name,
        "fuzz": False,
        "review_markov_transition": False,
    }


def _oracle_cost_weights(args: argparse.Namespace) -> list[float]:
    values = _parse_float_list(args.oracle_cost_weights, label="Oracle cost weight")
    if any(value < 0.0 for value in values):
        raise SystemExit("Oracle cost weights must be >= 0.")
    return values


def _oracle_action_retentions(args: argparse.Namespace) -> list[float]:
    values = _parse_float_list(
        args.oracle_action_retentions,
        label="Oracle action retention",
    )
    validate_retention_values(values, name="Oracle action retention")
    return values


def _oracle_s_to_idx(oracle: Any, s: torch.Tensor) -> torch.Tensor:
    log_s = torch.log(torch.clamp(s, oracle.bounds.s_min, oracle.bounds.s_max))
    ratio = (log_s - oracle.log_s_min) / (oracle.log_s_max - oracle.log_s_min)
    return torch.clamp(
        torch.round(ratio * float(oracle.s_grid.numel() - 1)),
        min=0,
        max=oracle.s_grid.numel() - 1,
    ).to(torch.int64)


def _oracle_d_to_idx(oracle: Any, d: torch.Tensor) -> torch.Tensor:
    ratio = torch.clamp(d, oracle.bounds.d_min, oracle.bounds.d_max)
    ratio = (ratio - oracle.bounds.d_min) / (oracle.bounds.d_max - oracle.bounds.d_min)
    return torch.clamp(
        torch.round(ratio * float(oracle.d_grid.numel() - 1)),
        min=0,
        max=oracle.d_grid.numel() - 1,
    ).to(torch.int64)


@torch.inference_mode()
def _evaluate_fsrs6_oracle_policies(
    *,
    args: argparse.Namespace,
    device: torch.device,
    oracle: Any,
    policies: torch.Tensor,
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    weight_count = len(cost_weights)
    env_count = args.particles * weight_count

    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        **_fsrs_config_kwargs(fsrs_config),
    )
    goal_indices = torch.repeat_interleave(
        torch.arange(weight_count, device=device, dtype=torch.int64),
        args.particles,
    )
    for weight_idx, cost_weight in enumerate(cost_weights):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        env.reset_indices(idx, goal_weight=cost_weight)

    policies = policies.to(device=device)
    while not bool(env.done.all().item()):
        remaining = torch.clamp((env.days - 1) - env.day, min=0, max=oracle.horizon)
        s_idx = _oracle_s_to_idx(oracle, env.s)
        d_idx = _oracle_d_to_idx(oracle, env.d)
        action = policies[goal_indices, remaining, s_idx, d_idx]
        env.step(action)

    metrics: list[SimMetrics] = []
    day_count = float(env.days)
    for weight_idx in range(weight_count):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        particle_count = float(idx.numel())
        total_memorized = env.total_memorized.index_select(0, idx).sum().item()
        total_cost_seconds = env.total_cost_seconds.index_select(0, idx).sum().item()
        total_reviews = float(env.total_reviews.index_select(0, idx).sum().item())
        total_lapses = float(env.total_lapses.index_select(0, idx).sum().item())
        observed_retention = (
            1.0 - total_lapses / total_reviews if total_reviews > 0.0 else None
        )
        metrics.append(
            SimMetrics(
                card_expected_retrievability=total_memorized
                / day_count
                / particle_count,
                card_minutes_per_day=total_cost_seconds
                / day_count
                / 60.0
                / particle_count,
                card_reviews_per_day=total_reviews / day_count / particle_count,
                card_total_reviews=total_reviews / particle_count,
                card_total_lapses=total_lapses / particle_count,
                card_total_cost_seconds=total_cost_seconds / particle_count,
                observed_retention=observed_retention,
            )
        )
    return metrics


@torch.inference_mode()
def _evaluate_fsrs6_oracle_infinite_policies(
    *,
    args: argparse.Namespace,
    device: torch.device,
    oracle: Any,
    policies: torch.Tensor,
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    weight_count = len(cost_weights)
    env_count = args.particles * weight_count

    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        **_fsrs_config_kwargs(fsrs_config),
    )
    goal_indices = torch.repeat_interleave(
        torch.arange(weight_count, device=device, dtype=torch.int64),
        args.particles,
    )
    for weight_idx, cost_weight in enumerate(cost_weights):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        env.reset_indices(idx, goal_weight=cost_weight)

    policies = policies.to(device=device)
    while not bool(env.done.all().item()):
        s_idx = _oracle_s_to_idx(oracle, env.s)
        d_idx = _oracle_d_to_idx(oracle, env.d)
        action = policies[goal_indices, s_idx, d_idx]
        env.step(action)

    metrics: list[SimMetrics] = []
    day_count = float(env.days)
    for weight_idx in range(weight_count):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        particle_count = float(idx.numel())
        total_memorized = env.total_memorized.index_select(0, idx).sum().item()
        total_cost_seconds = env.total_cost_seconds.index_select(0, idx).sum().item()
        total_reviews = float(env.total_reviews.index_select(0, idx).sum().item())
        total_lapses = float(env.total_lapses.index_select(0, idx).sum().item())
        observed_retention = (
            1.0 - total_lapses / total_reviews if total_reviews > 0.0 else None
        )
        metrics.append(
            SimMetrics(
                card_expected_retrievability=total_memorized
                / day_count
                / particle_count,
                card_minutes_per_day=total_cost_seconds
                / day_count
                / 60.0
                / particle_count,
                card_reviews_per_day=total_reviews / day_count / particle_count,
                card_total_reviews=total_reviews / particle_count,
                card_total_lapses=total_lapses / particle_count,
                card_total_cost_seconds=total_cost_seconds / particle_count,
                observed_retention=observed_retention,
            )
        )
    return metrics


@torch.inference_mode()
def _evaluate_fsrs6_oracle_interval_policies(
    *,
    args: argparse.Namespace,
    device: torch.device,
    oracle: Any,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    weight_count = len(cost_weights)
    env_count = args.particles * weight_count

    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=[0.9],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        **_fsrs_config_kwargs(fsrs_config),
    )
    goal_indices = torch.repeat_interleave(
        torch.arange(weight_count, device=device, dtype=torch.int64),
        args.particles,
    )
    for weight_idx, cost_weight in enumerate(cost_weights):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        env.reset_indices(idx, goal_weight=cost_weight)

    policies = policies.to(device=device)
    while not bool(env.done.all().item()):
        remaining = torch.clamp((env.days - 1) - env.day, min=0, max=oracle.horizon)
        s_idx = _oracle_s_to_idx(oracle, env.s)
        d_idx = _oracle_d_to_idx(oracle, env.d)
        intervals = policies[goal_indices, remaining, s_idx, d_idx]
        env.step_intervals(intervals)

    metrics: list[SimMetrics] = []
    day_count = float(env.days)
    for weight_idx in range(weight_count):
        start = weight_idx * args.particles
        stop = start + args.particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        particle_count = float(idx.numel())
        total_memorized = env.total_memorized.index_select(0, idx).sum().item()
        total_cost_seconds = env.total_cost_seconds.index_select(0, idx).sum().item()
        total_reviews = float(env.total_reviews.index_select(0, idx).sum().item())
        total_lapses = float(env.total_lapses.index_select(0, idx).sum().item())
        observed_retention = (
            1.0 - total_lapses / total_reviews if total_reviews > 0.0 else None
        )
        metrics.append(
            SimMetrics(
                card_expected_retrievability=total_memorized
                / day_count
                / particle_count,
                card_minutes_per_day=total_cost_seconds
                / day_count
                / 60.0
                / particle_count,
                card_reviews_per_day=total_reviews / day_count / particle_count,
                card_total_reviews=total_reviews / particle_count,
                card_total_lapses=total_lapses / particle_count,
                card_total_cost_seconds=total_cost_seconds / particle_count,
                observed_retention=observed_retention,
            )
        )
    return metrics


def _reset_cost_weight_slices(
    env: Any,
    *,
    cost_weights: Sequence[float],
    particles: int,
    device: torch.device,
) -> None:
    for weight_idx, cost_weight in enumerate(cost_weights):
        start = weight_idx * particles
        stop = start + particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        env.reset_indices(idx, goal_weight=cost_weight)


def _single_card_metrics_by_weight(
    env: Any,
    *,
    weight_count: int,
    particles: int,
    device: torch.device,
) -> list[Any]:
    metrics: list[SimMetrics] = []
    day_count = float(env.days)
    for weight_idx in range(weight_count):
        start = weight_idx * particles
        stop = start + particles
        idx = torch.arange(start, stop, device=device, dtype=torch.int64)
        particle_count = float(idx.numel())
        total_memorized = env.total_memorized.index_select(0, idx).sum().item()
        total_cost_seconds = env.total_cost_seconds.index_select(0, idx).sum().item()
        total_reviews = float(env.total_reviews.index_select(0, idx).sum().item())
        total_lapses = float(env.total_lapses.index_select(0, idx).sum().item())
        observed_retention = (
            1.0 - total_lapses / total_reviews if total_reviews > 0.0 else None
        )
        metrics.append(
            SimMetrics(
                card_expected_retrievability=total_memorized
                / day_count
                / particle_count,
                card_minutes_per_day=total_cost_seconds
                / day_count
                / 60.0
                / particle_count,
                card_reviews_per_day=total_reviews / day_count / particle_count,
                card_total_reviews=total_reviews / particle_count,
                card_total_lapses=total_lapses / particle_count,
                card_total_cost_seconds=total_cost_seconds / particle_count,
                observed_retention=observed_retention,
            )
        )
    return metrics


def _progress_done(
    *,
    enabled: bool,
    total: int,
    label: str,
) -> Any:
    if not enabled:
        return None
    from tqdm import tqdm

    return tqdm(total=total, desc=label, unit="env", leave=False)


@torch.inference_mode()
def _evaluate_action_policy_weights(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: Any,
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    seed: int,
    goal_norm_max: float,
    obs_mode: str,
    progress_label: str,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    weight_count = len(cost_weights)
    env_count = args.particles * weight_count
    model_dtype = next(model.parameters()).dtype
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        **_fsrs_config_kwargs(fsrs_config),
    )
    _reset_cost_weight_slices(
        env,
        cost_weights=cost_weights,
        particles=args.particles,
        device=device,
    )
    model.eval()
    progress = _progress_done(
        enabled=not args.no_progress,
        total=env_count,
        label=progress_label,
    )
    completed = 0
    try:
        while not bool(env.done.all().item()):
            obs = env.obs().to(dtype=model_dtype)
            logits, _ = model(obs)
            action = torch.argmax(logits, dim=1)
            env.step(action)
            if progress is not None:
                next_completed = int(env.done.sum().item())
                progress.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress is not None:
            progress.close()
    return _single_card_metrics_by_weight(
        env,
        weight_count=weight_count,
        particles=args.particles,
        device=device,
    )


@torch.inference_mode()
def _evaluate_interval_distill_weights(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: Any,
    cost_weights: Sequence[float],
    seed: int,
    goal_norm_max: float,
    log_interval_bias: float,
    terminal_snap_ratio: float,
    progress_label: str,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    from experiments.single_card_tradeoff.oracle_interval_distill import (
        predicted_intervals,
    )

    weight_count = len(cost_weights)
    env_count = args.particles * weight_count
    model_dtype = next(model.parameters()).dtype
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=[0.9],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode="oracle",
        **_fsrs_config_kwargs(fsrs_config),
    )
    _reset_cost_weight_slices(
        env,
        cost_weights=cost_weights,
        particles=args.particles,
        device=device,
    )
    model.eval()
    progress = _progress_done(
        enabled=not args.no_progress,
        total=env_count,
        label=progress_label,
    )
    completed = 0
    try:
        while not bool(env.done.all().item()):
            obs = env.obs().to(dtype=model_dtype)
            pred_log_interval = model(obs)
            intervals = predicted_intervals(
                env=env,
                log_interval=pred_log_interval,
                log_interval_bias=log_interval_bias,
                terminal_snap_ratio=terminal_snap_ratio,
            )
            env.step_intervals(intervals)
            if progress is not None:
                next_completed = int(env.done.sum().item())
                progress.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress is not None:
            progress.close()
    return _single_card_metrics_by_weight(
        env,
        weight_count=weight_count,
        particles=args.particles,
        device=device,
    )


@torch.inference_mode()
def _evaluate_retention_distill_weights(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: Any,
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    seed: int,
    goal_norm_max: float,
    obs_mode: str,
    retention_min: float,
    retention_max: float,
    terminal_snap_ratio: float,
    progress_label: str,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    from experiments.single_card_tradeoff.oracle_retention_distill import (
        predicted_retentions,
        rounded_intervals_for_retentions,
    )

    weight_count = len(cost_weights)
    env_count = args.particles * weight_count
    model_dtype = next(model.parameters()).dtype
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        **_fsrs_config_kwargs(fsrs_config),
    )
    _reset_cost_weight_slices(
        env,
        cost_weights=cost_weights,
        particles=args.particles,
        device=device,
    )
    model.eval()
    progress = _progress_done(
        enabled=not args.no_progress,
        total=env_count,
        label=progress_label,
    )
    completed = 0
    try:
        while not bool(env.done.all().item()):
            obs = env.obs().to(dtype=model_dtype)
            raw_retention, _ = model(obs)
            retention = predicted_retentions(
                raw_retention.to(dtype=env.dtype),
                retention_min=retention_min,
                retention_max=retention_max,
            )
            intervals = rounded_intervals_for_retentions(
                env=env,
                retention=retention,
                terminal_snap_ratio=terminal_snap_ratio,
            )
            env.step_intervals(intervals)
            if progress is not None:
                next_completed = int(env.done.sum().item())
                progress.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress is not None:
            progress.close()
    return _single_card_metrics_by_weight(
        env,
        weight_count=weight_count,
        particles=args.particles,
        device=device,
    )


@torch.inference_mode()
def _evaluate_recurrent_interval_weights(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: Any,
    cost_weights: Sequence[float],
    seed: int,
    goal_norm_max: float,
    max_interval_days: int | None,
    progress_label: str,
    fsrs_config: SingleCardFSRS6Config | None = None,
) -> list[Any]:
    weight_count = len(cost_weights)
    env_count = args.particles * weight_count
    model_dtype = next(model.parameters()).dtype
    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=[0.9],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=True,
        goal_norm_max=goal_norm_max,
        obs_mode="belief",
        max_interval_days=max_interval_days or args.days * 4,
        **_fsrs_config_kwargs(fsrs_config),
    )
    _reset_cost_weight_slices(
        env,
        cost_weights=cost_weights,
        particles=args.particles,
        device=device,
    )
    hidden_state = model.initial_state(
        env_count,
        device=device,
        dtype=model_dtype,
    )
    model.eval()
    progress = _progress_done(
        enabled=not args.no_progress,
        total=env_count,
        label=progress_label,
    )
    completed = 0
    try:
        while not bool(env.done.all().item()):
            obs = env.obs().to(dtype=model_dtype)
            hidden = model.encode(obs, hidden_state)
            dist, _ = model.dist_value(
                hidden,
                env.goal_weight.to(dtype=model_dtype),
                max_goal_weight=goal_norm_max,
            )
            action = dist.mean.squeeze(1)
            _, _, done = env.step_log_interval(action.to(dtype=env.dtype))
            hidden_state = hidden.detach()
            if done.any():
                hidden_state[done.nonzero(as_tuple=False).squeeze(1)] = 0.0
            if progress is not None:
                next_completed = int(done.sum().item())
                progress.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress is not None:
            progress.close()
    return _single_card_metrics_by_weight(
        env,
        weight_count=weight_count,
        particles=args.particles,
        device=device,
    )


def _run_fsrs6_oracle(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle currently supports only --env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit("fsrs6_oracle is supported only with --engine vectorized.")
    if args.fuzz:
        raise SystemExit("fsrs6_oracle does not support --fuzz.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")

    from experiments.single_card_tradeoff.oracles import FSRS6GridOracle

    device = _resolve_torch_device(args, prefer_cuda=True)
    cost_weights = _oracle_cost_weights(args)
    action_retentions = _oracle_action_retentions(args)
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)
    oracle = FSRS6GridOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        cache_config=_runtime_context(args).dp_cache_config,
        **_fsrs_config_kwargs(fsrs_config),
    )

    start = time.perf_counter()
    policies = oracle.solve_policies(cost_weights, progress=not args.no_progress)
    metrics_by_weight = _evaluate_fsrs6_oracle_policies(
        args=args,
        device=device,
        oracle=oracle,
        policies=policies,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 50_000,
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / float(len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_infinite(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_infinite currently supports only --env fsrs6_default "
            "or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_infinite is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_infinite does not support --fuzz.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_infinite_max_iterations <= 0:
        raise SystemExit("--oracle-infinite-max-iterations must be > 0.")
    if args.oracle_infinite_tolerance <= 0.0:
        raise SystemExit("--oracle-infinite-tolerance must be > 0.")

    from experiments.single_card_tradeoff.oracles import (
        FSRS6AverageRewardOracle,
    )

    device = _resolve_torch_device(args, prefer_cuda=True)
    cost_weights = _oracle_cost_weights(args)
    action_retentions = _oracle_action_retentions(args)
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)
    oracle = FSRS6AverageRewardOracle(
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        cache_config=_runtime_context(args).dp_cache_config,
        **_fsrs_config_kwargs(fsrs_config),
    )

    start = time.perf_counter()
    solution = oracle.solve_average_reward_policies(
        cost_weights,
        max_iterations=args.oracle_infinite_max_iterations,
        tolerance=args.oracle_infinite_tolerance,
        progress=not args.no_progress,
    )
    if not all(solution.converged):
        failed = [
            format_float(weight)
            for weight, converged in zip(cost_weights, solution.converged, strict=True)
            if not converged
        ]
        raise SystemExit(
            "fsrs6_oracle_infinite did not converge for cost weights: "
            + ",".join(failed)
        )
    metrics_by_weight = _evaluate_fsrs6_oracle_infinite_policies(
        args=args,
        device=device,
        oracle=oracle,
        policies=solution.policy,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 55_000,
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / float(len(cost_weights))

    for cost_weight, gain, iterations, residual in zip(
        cost_weights,
        solution.gains.tolist(),
        solution.iterations,
        solution.residuals,
        strict=True,
    ):
        print(
            " ".join(
                [
                    f"infinite_oracle w={format_float(cost_weight)}",
                    f"gain={float(gain):.8f}",
                    f"iterations={iterations}",
                    f"residual={residual:.3g}",
                ]
            )
        )

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_INFINITE_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_stationary_finite(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_stationary_finite currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_stationary_finite is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_stationary_finite does not support --fuzz.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_stationary_finite_max_iterations <= 0:
        raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
    if args.oracle_stationary_finite_tolerance <= 0.0:
        raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")

    from experiments.single_card_tradeoff.oracles import (
        FSRS6StationaryFiniteOracle,
    )

    device = _resolve_torch_device(args, prefer_cuda=True)
    cost_weights = _oracle_cost_weights(args)
    action_retentions = _oracle_action_retentions(args)
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)
    oracle = FSRS6StationaryFiniteOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        cache_config=_runtime_context(args).dp_cache_config,
        **_fsrs_config_kwargs(fsrs_config),
    )

    start = time.perf_counter()
    solution = oracle.solve_stationary_finite_policies(
        cost_weights,
        max_iterations=args.oracle_stationary_finite_max_iterations,
        tolerance=args.oracle_stationary_finite_tolerance,
        progress=not args.no_progress,
    )
    if not all(solution.converged):
        failed = [
            format_float(weight)
            for weight, converged in zip(cost_weights, solution.converged, strict=True)
            if not converged
        ]
        raise SystemExit(
            "fsrs6_oracle_stationary_finite did not converge for cost weights: "
            + ",".join(failed)
        )
    metrics_by_weight = _evaluate_fsrs6_oracle_infinite_policies(
        args=args,
        device=device,
        oracle=oracle,
        policies=solution.policy,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 57_000,
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / float(len(cost_weights))

    for cost_weight, objective, iterations, residual in zip(
        cost_weights,
        solution.objectives.tolist(),
        solution.iterations,
        solution.residuals,
        strict=True,
    ):
        print(
            " ".join(
                [
                    f"stationary_finite_oracle w={format_float(cost_weight)}",
                    f"objective={float(objective):.8f}",
                    f"iterations={iterations}",
                    f"residual={residual:.3g}",
                ]
            )
        )

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_interval(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_interval currently supports only --env fsrs6_default "
            "or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_interval is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_interval does not support --fuzz.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")

    from experiments.single_card_tradeoff.oracles import FSRS6IntervalOracle

    device = _resolve_torch_device(args, prefer_cuda=True)
    cost_weights = _oracle_cost_weights(args)
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)
    oracle = FSRS6IntervalOracle(
        days=args.days,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        interval_chunk_size=args.oracle_interval_chunk_size,
        device=device,
        cache_config=_runtime_context(args).dp_cache_config,
        **_fsrs_config_kwargs(fsrs_config),
    )

    start = time.perf_counter()
    policies = oracle.solve_policies(cost_weights, progress=not args.no_progress)
    metrics_by_weight = _evaluate_fsrs6_oracle_interval_policies(
        args=args,
        device=device,
        oracle=oracle,
        policies=policies,
        cost_weights=cost_weights,
        seed=seed + 60_000,
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / float(len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_INTERVAL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_uvfa_ppo(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "uvfa_ppo currently supports only --env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit("uvfa_ppo is supported only with --engine vectorized.")
    if args.fuzz:
        raise SystemExit("uvfa_ppo does not support --fuzz.")

    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    model, action_retentions, policy_cost_weights, goal_norm_max, obs_mode = (
        _load_uvfa_ppo_policy(
            args,
            device=device,
        )
    )
    cost_weights = _uvfa_ppo_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_action_policy_weights(
        args=args,
        device=device,
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 30_000,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=UVFA_PPO_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_distill(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_distill currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_distill is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_distill does not support --fuzz.")

    device = _resolve_torch_device(args, prefer_cuda=True)
    model, action_retentions, policy_cost_weights, goal_norm_max, obs_mode = (
        _load_fsrs6_oracle_distill_policy(
            args,
            device=device,
        )
    )
    cost_weights = _oracle_distill_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_action_policy_weights(
        args=args,
        device=device,
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 35_000,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_DISTILL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_infinite_distill(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_infinite_distill currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_infinite_distill is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_infinite_distill does not support --fuzz.")

    device = _resolve_torch_device(args, prefer_cuda=True)
    model, action_retentions, policy_cost_weights, goal_norm_max, obs_mode = (
        _load_fsrs6_oracle_infinite_distill_policy(
            args,
            device=device,
        )
    )
    cost_weights = _oracle_infinite_distill_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_action_policy_weights(
        args=args,
        device=device,
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 37_000,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_stationary_finite_distill(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_stationary_finite_distill currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_stationary_finite_distill is supported only with "
            "--engine vectorized."
        )
    if args.fuzz:
        raise SystemExit(
            "fsrs6_oracle_stationary_finite_distill does not support --fuzz."
        )

    device = _resolve_torch_device(args, prefer_cuda=True)
    model, action_retentions, policy_cost_weights, goal_norm_max, obs_mode = (
        _load_fsrs6_oracle_stationary_finite_distill_policy(
            args,
            device=device,
        )
    )
    cost_weights = _oracle_stationary_finite_distill_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_action_policy_weights(
        args=args,
        device=device,
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 38_000,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_retention_distill(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_retention_distill currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_retention_distill is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_retention_distill does not support --fuzz.")

    device = _resolve_torch_device(args, prefer_cuda=True)
    (
        model,
        action_retentions,
        policy_cost_weights,
        goal_norm_max,
        obs_mode,
        retention_min,
        retention_max,
        terminal_snap_ratio,
    ) = _load_fsrs6_oracle_retention_distill_policy(
        args,
        device=device,
    )
    cost_weights = _oracle_retention_distill_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_retention_distill_weights(
        args=args,
        device=device,
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        seed=seed + 75_000,
        goal_norm_max=goal_norm_max,
        obs_mode=obs_mode,
        retention_min=retention_min,
        retention_max=retention_max,
        terminal_snap_ratio=terminal_snap_ratio,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_uvfa_ppo_rnn_interval(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "uvfa_ppo_rnn_interval currently supports only --env fsrs6_default "
            "or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "uvfa_ppo_rnn_interval is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("uvfa_ppo_rnn_interval does not support --fuzz.")

    device = _resolve_torch_device(args, prefer_cuda=True)
    model, policy_cost_weights, goal_norm_max, max_interval_days = (
        _load_uvfa_ppo_rnn_interval_policy(
            args,
            device=device,
        )
    )
    cost_weights = _uvfa_ppo_rnn_interval_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_recurrent_interval_weights(
        args=args,
        device=device,
        model=model,
        cost_weights=cost_weights,
        seed=seed + 40_000,
        goal_norm_max=goal_norm_max,
        max_interval_days=max_interval_days,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=UVFA_PPO_RNN_INTERVAL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


def _run_fsrs6_oracle_interval_distill(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_spec: str,
    seed: int,
) -> list[dict[str, Any]]:
    if environment_name not in SUPPORTED_SINGLE_CARD_ENVS:
        raise SystemExit(
            "fsrs6_oracle_interval_distill currently supports only "
            "--env fsrs6_default or --env fsrs6."
        )
    if args.engine != "vectorized":
        raise SystemExit(
            "fsrs6_oracle_interval_distill is supported only with --engine vectorized."
        )
    if args.fuzz:
        raise SystemExit("fsrs6_oracle_interval_distill does not support --fuzz.")

    device = _resolve_torch_device(args, prefer_cuda=True)
    (
        model,
        policy_cost_weights,
        goal_norm_max,
        log_interval_bias,
        terminal_snap_ratio,
    ) = _load_fsrs6_oracle_interval_distill_policy(
        args,
        device=device,
    )
    cost_weights = _oracle_interval_distill_cost_weights(
        args,
        policy_cost_weights=policy_cost_weights,
    )
    fsrs_config = load_single_card_fsrs6_config(args, environment=environment_name)

    start = time.perf_counter()
    metrics_by_weight = _evaluate_interval_distill_weights(
        args=args,
        device=device,
        model=model,
        cost_weights=cost_weights,
        seed=seed + 70_000,
        goal_norm_max=goal_norm_max,
        log_interval_bias=log_interval_bias,
        terminal_snap_ratio=terminal_snap_ratio,
        progress_label=f"{environment_name}/{scheduler_spec}",
        fsrs_config=fsrs_config,
    )
    runtime_s = (time.perf_counter() - start) / max(1, len(cost_weights))

    rows: list[dict[str, Any]] = []
    for cost_weight, metrics in zip(cost_weights, metrics_by_weight, strict=True):
        rows.append(
            _row_from_uvfa_metrics(
                args,
                environment_name=environment_name,
                scheduler_name=FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER,
                scheduler_spec=scheduler_spec,
                goal_cost_weight=cost_weight,
                seed=seed,
                metrics=metrics,
                runtime_s=runtime_s,
            )
        )
    return rows


_CUSTOM_SINGLE_USER_RUNNERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    FSRS6_ORACLE_SCHEDULER: _run_fsrs6_oracle,
    FSRS6_ORACLE_INFINITE_SCHEDULER: _run_fsrs6_oracle_infinite,
    FSRS6_ORACLE_STATIONARY_FINITE_SCHEDULER: _run_fsrs6_oracle_stationary_finite,
    FSRS6_ORACLE_INTERVAL_SCHEDULER: _run_fsrs6_oracle_interval,
    FSRS6_ORACLE_INTERVAL_DISTILL_SCHEDULER: _run_fsrs6_oracle_interval_distill,
    FSRS6_ORACLE_INFINITE_DISTILL_SCHEDULER: _run_fsrs6_oracle_infinite_distill,
    FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER: (
        _run_fsrs6_oracle_stationary_finite_distill
    ),
    FSRS6_ORACLE_RETENTION_DISTILL_SCHEDULER: _run_fsrs6_oracle_retention_distill,
    UVFA_PPO_SCHEDULER: _run_uvfa_ppo,
    FSRS6_ORACLE_DISTILL_SCHEDULER: _run_fsrs6_oracle_distill,
    UVFA_PPO_RNN_INTERVAL_SCHEDULER: _run_uvfa_ppo_rnn_interval,
}


@dataclass(frozen=True)
class TradeoffEvaluationRequest:
    args: argparse.Namespace
    environment_name: str
    scheduler_name: str
    scheduler_spec: str
    fixed_interval: float | None
    retention_values: Sequence[float]
    user_contexts: Sequence[UserContext]
    seed: int


TradeoffEvaluator = Callable[
    [TradeoffEvaluationRequest],
    list[dict[str, Any]] | None,
]

_VECTORIZED_BATCH_SCHEDULERS = frozenset(
    {
        "fsrs6",
        "fsrs6_default",
        "fsrs3",
        "fsrs3_default",
        "hlr",
        "lstm",
        "anki_sm2",
        "memrise",
    }
)


def _run_registered_custom_scheduler(
    args: argparse.Namespace,
    *,
    environment_name: str,
    scheduler_name: str,
    scheduler_spec: str,
    seed: int,
    user_contexts: Sequence[UserContext],
) -> list[dict[str, Any]] | None:
    runner = _CUSTOM_SINGLE_USER_RUNNERS.get(scheduler_name)
    if runner is None:
        return None

    output_rows: list[dict[str, Any]] = []
    for context in user_contexts:
        run_args = context.args
        if scheduler_name == FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_SCHEDULER:
            run_args = argparse.Namespace(**vars(context.args))
            run_args.oracle_stationary_finite_distill_policy = (
                _resolve_stationary_finite_distill_policy_path(
                    args,
                    user_id=context.user_id,
                    multiuser=len(user_contexts) > 1,
                )
            )
        output_rows.extend(
            runner(
                run_args,
                environment_name=environment_name,
                scheduler_spec=scheduler_spec,
                seed=seed,
            )
        )
    return output_rows


def _custom_scheduler_evaluator(
    request: TradeoffEvaluationRequest,
) -> list[dict[str, Any]] | None:
    return _run_registered_custom_scheduler(
        request.args,
        environment_name=request.environment_name,
        scheduler_name=request.scheduler_name,
        scheduler_spec=request.scheduler_spec,
        seed=request.seed,
        user_contexts=request.user_contexts,
    )


def _adr_scheduler_evaluator(
    request: TradeoffEvaluationRequest,
) -> list[dict[str, Any]] | None:
    if request.scheduler_name not in FSRS6_ADR_SCHEDULERS:
        return None
    return _run_fsrs6_adr(
        request.args,
        environment_name=request.environment_name,
        scheduler_name=request.scheduler_name,
        scheduler_spec=request.scheduler_spec,
        seed=request.seed,
        retention_values=request.retention_values,
        user_contexts=request.user_contexts,
    )


def _desired_retention_values(
    scheduler_name: str,
    retention_values: Sequence[float],
) -> Sequence[float | None]:
    if scheduler_uses_desired_retention(scheduler_name):
        return retention_values
    return [None]


def _standard_scheduler_evaluator(
    request: TradeoffEvaluationRequest,
) -> list[dict[str, Any]] | None:
    desired_values = _desired_retention_values(
        request.scheduler_name,
        request.retention_values,
    )
    if (
        request.args.engine == "vectorized"
        and request.environment_name in {"fsrs6", "fsrs6_default"}
        and request.scheduler_name in _VECTORIZED_BATCH_SCHEDULERS
    ):
        points = [
            SchedulerPoint(
                scheduler_spec=request.scheduler_spec,
                desired_retention=(
                    float(desired_retention) if desired_retention is not None else None
                ),
            )
            for desired_retention in desired_values
        ]
        return _run_vectorized_batch_rows(
            request.args,
            environment_name=request.environment_name,
            scheduler_name=request.scheduler_name,
            points=points,
            user_contexts=request.user_contexts,
            seed=request.seed,
            progress_label=f"{request.environment_name}/{request.scheduler_spec}",
        )

    rows: list[dict[str, Any]] = []
    for desired_retention in desired_values:
        for context in request.user_contexts:
            rows.append(
                _run_point(
                    context.args,
                    environment_name=request.environment_name,
                    scheduler_name=request.scheduler_name,
                    scheduler_spec=request.scheduler_spec,
                    fixed_interval=normalize_fixed_interval(request.fixed_interval)
                    if request.scheduler_name == "fixed"
                    else None,
                    desired_retention=desired_retention,
                    seed=request.seed,
                )
            )
    return rows


_SCHEDULER_EVALUATORS: tuple[TradeoffEvaluator, ...] = (
    _custom_scheduler_evaluator,
    _adr_scheduler_evaluator,
    _standard_scheduler_evaluator,
)


def _run_scheduler_spec_rows(
    request: TradeoffEvaluationRequest,
) -> list[dict[str, Any]]:
    for evaluator in _SCHEDULER_EVALUATORS:
        rows = evaluator(request)
        if rows is not None:
            return rows
    raise RuntimeError(f"No evaluator handled scheduler '{request.scheduler_name}'.")


def _point_label(row: dict[str, Any]) -> str:
    goal_cost_weight = row.get("goal_cost_weight")
    if goal_cost_weight is not None and goal_cost_weight != "":
        return f"w={format_float(float(goal_cost_weight))}"
    if row.get("scheduler") in FSRS6_ADR_SCHEDULERS:
        point_label = row.get("fsrs6_adr_point_label")
        if point_label is not None and point_label != "":
            return str(point_label)
    desired_retention = row["desired_retention"]
    if desired_retention is not None:
        return format_float(float(desired_retention))
    fixed_interval = row["fixed_interval"]
    if fixed_interval is not None:
        return f"{format_float(float(fixed_interval))}d"
    return row["scheduler_spec"]


def _regret_auc_path(args: argparse.Namespace) -> Path:
    if args.regret_auc_out is not None:
        return args.regret_auc_out
    suffix = args.out.suffix or ".csv"
    return args.out.with_name(f"{args.out.stem}_regret_auc{suffix}")


def _plot_group_key(row: dict[str, Any]) -> tuple[int, str, str]:
    scheduler = str(row["scheduler"])
    scheduler_label = "fixed" if scheduler == "fixed" else str(row["scheduler_spec"])
    return _row_user_id(row), str(row["environment"]), scheduler_label


def _plot_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    goal_cost_weight = row.get("goal_cost_weight")
    if goal_cost_weight is not None and goal_cost_weight != "":
        return 0.5, float(goal_cost_weight)
    if row.get("scheduler") in FSRS6_ADR_SCHEDULERS:
        policy_index = row.get("fsrs6_adr_policy_index")
        if policy_index is not None and policy_index != "":
            return 0.75, float(policy_index)
        baseline_dr = row.get("fsrs6_adr_baseline_desired_retention")
        if baseline_dr is not None and baseline_dr != "":
            lambda_value = row.get("fsrs6_adr_lambda_value")
            lambda_offset = (
                0.0
                if lambda_value is None or lambda_value == ""
                else float(lambda_value)
            )
            return 0.75, float(baseline_dr) + lambda_offset * 1e-6
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

    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = _plot_group_key(row)
        groups.setdefault(key, []).append(row)

    user_ids = sorted({_row_user_id(row) for row in rows})
    label_rows: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    for (user_id, environment, scheduler_label), group in groups.items():
        group = sorted(group, key=_plot_sort_key)
        x = [row["deck_expected_memorized"] for row in group]
        y = [row["deck_minutes_per_day"] for row in group]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.0,
            alpha=0.45,
            label=f"user {user_id}/{environment}/{scheduler_label}",
        )
        label_rows.extend(group)

    for user_id in user_ids:
        frontier = _pareto_frontier(
            [row for row in rows if _row_user_id(row) == user_id]
        )
        if not frontier:
            continue
        ax.plot(
            [row["deck_expected_memorized"] for row in frontier],
            [row["deck_minutes_per_day"] for row in frontier],
            color="black",
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            alpha=0.75,
            label=f"user {user_id} Pareto frontier ({len(frontier)} points)",
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
                    f"user={row.get('user_id', 1)}",
                    f"{row['environment']}/{row['scheduler_spec']}",
                    f"target={target}",
                    f"card_mem={row['card_expected_retrievability']:.4f}",
                    f"card_min/day={row['card_minutes_per_day']:.6f}",
                    f"deck_mem={row['deck_expected_memorized']:.1f}",
                    f"deck_min/day={row['deck_minutes_per_day']:.2f}",
                ]
            )
        )


def _print_regret_auc_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    preferred_baselines = {"fsrs6_default", "fsrs6"}
    filtered = [
        row
        for row in rows
        if row["baseline_scheduler"] in preferred_baselines
        and row["scheduler"] != row["baseline_scheduler"]
    ]
    if not filtered:
        filtered = [
            row for row in rows if row["scheduler"] != row["baseline_scheduler"]
        ]
    for row in filtered:
        same_target_time_saved_auc = row["same_target_time_saved_auc"]
        relative_same_target_time_saved_auc_percent = row[
            "relative_same_target_time_saved_auc_percent"
        ]
        time_text = (
            f"{same_target_time_saved_auc:.4f}"
            if isinstance(same_target_time_saved_auc, (int, float))
            else "n/a"
        )
        relative_text = (
            f"{relative_same_target_time_saved_auc_percent:.2f}%"
            if isinstance(relative_same_target_time_saved_auc_percent, (int, float))
            else "n/a"
        )
        print(
            " ".join(
                [
                    "same_target_time_saved_auc",
                    f"user={row.get('user_id', 1)}",
                    f"{row['environment']}/{row['scheduler']}",
                    f"review_markov={row.get('review_markov_transition')}",
                    f"vs={row['baseline_scheduler']}",
                    f"time_saved={time_text}",
                    f"relative_time_saved={relative_text}",
                    f"span={row['span_coverage_percent']:.1f}%",
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
    user_ids = _resolve_user_ids(args)
    args.user_id = user_ids[0]
    if (
        len(user_ids) > 1
        and getattr(args, "oracle_stationary_finite_distill_policy_template", None)
        is not None
        and "{user_id}"
        not in str(args.oracle_stationary_finite_distill_policy_template)
    ):
        raise SystemExit(
            "--oracle-stationary-finite-distill-policy-template must contain "
            "{user_id} for multi-user runs."
        )
    if args.torch_device is None:
        args.torch_device = _default_torch_device()
    setattr(
        args,
        "single_card_runtime_context",
        single_card_runtime_context_from_args(
            args,
            repo_root=REPO_ROOT,
            output_dir=args.out.parent,
        ),
    )
    register_run_monitor(
        args,
        device=args.torch_device,
        output_dir=args.out.parent,
        stage_name=Path(__file__).stem,
    )

    environments = parse_csv(args.env) or ["fsrs6_default"]
    for environment in environments:
        if environment not in simulate_cli.ENVIRONMENT_FACTORIES:
            raise SystemExit(f"Unknown environment '{environment}'.")
    scheduler_specs = _run_specs(args)
    retention_values = _retention_grid(args)

    rows: list[dict[str, Any]] = []
    for environment in environments:
        user_contexts = _load_user_contexts(
            args,
            environment_name=environment,
            user_ids=user_ids,
        )
        fixed_specs = [
            (scheduler_spec, normalize_fixed_interval(fixed_interval))
            for scheduler_name, scheduler_spec, fixed_interval in scheduler_specs
            if scheduler_name == "fixed"
        ]
        batched_fixed_specs: set[str] = set()
        if (
            args.engine == "vectorized"
            and environment in {"fsrs6", "fsrs6_default"}
            and fixed_specs
        ):
            rows.extend(
                _run_vectorized_batch_rows(
                    args,
                    environment_name=environment,
                    scheduler_name="fixed",
                    points=[
                        SchedulerPoint(
                            scheduler_spec=scheduler_spec,
                            fixed_interval=interval,
                        )
                        for scheduler_spec, interval in fixed_specs
                    ],
                    user_contexts=user_contexts,
                    seed=args.seed,
                    progress_label=f"{environment}/fixed intervals",
                )
            )
            batched_fixed_specs = {scheduler_spec for scheduler_spec, _ in fixed_specs}

        for scheduler_name, scheduler_spec, fixed_interval in scheduler_specs:
            if scheduler_name == "fixed" and scheduler_spec in batched_fixed_specs:
                continue
            rows.extend(
                _run_scheduler_spec_rows(
                    TradeoffEvaluationRequest(
                        args=args,
                        environment_name=environment,
                        scheduler_name=scheduler_name,
                        scheduler_spec=scheduler_spec,
                        fixed_interval=fixed_interval,
                        retention_values=retention_values,
                        user_contexts=user_contexts,
                        seed=args.seed,
                    )
                )
            )

    _write_csv(args.out, rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        _write_plot(plot_path, rows)
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {args.out}")
    if not args.no_regret_auc:
        regret_auc_rows = _build_regret_auc_rows(rows)
        regret_auc_path = _regret_auc_path(args)
        _write_regret_auc_csv(regret_auc_path, regret_auc_rows)
        print(f"Wrote same-target time saved AUC CSV: {regret_auc_path}")
        _print_regret_auc_summary(regret_auc_rows)
    _print_summary(rows)


if __name__ == "__main__":
    main()
