from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
from simulator.batched_sweep.logging import (
    BatchedSweepLogLane,
    simulate_and_log_lanes,
)
from simulator.batched_sweep.utils import dr_values
from simulator.batched_sweep.weights import (
    load_fsrs6_weights,
    resolve_lstm_paths,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_PRIORITY,
    DEFAULT_RETENTION_STEP,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
    DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    DEFAULT_START_RETENTION,
    DEFAULT_END_RETENTION,
)
from simulator.math.fsrs import Bounds
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.sa_fsrs6_policy import SAFSRS6Policy
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.sa_fsrs6 import SAFSRS6BatchSchedulerOps
from simulator.vectorized.mixed_scheduler import (
    MixedBatchSchedulerOps,
    MixedSchedulerGroup,
)

import simulate as simulate_cli


def _format_float_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def _policy_path(
    *,
    train_outputs_root: Path,
    user_id: int,
    lambda_token: str,
    retention: float,
) -> Path:
    return (
        train_outputs_root
        / f"user_{user_id}"
        / f"lambda_{lambda_token}"
        / f"dr_{_format_float_token(retention)}"
        / "policy.json"
    )


def _validate_policy(policy: SAFSRS6Policy, *, path: Path, retention: float) -> None:
    if not math.isclose(
        policy.baseline_desired_retention,
        retention,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Policy {path} has baseline_desired_retention="
            f"{policy.baseline_desired_retention}, expected {retention}."
        )


def _same_policy_bounds(lhs: SAFSRS6Policy, rhs: SAFSRS6Policy) -> bool:
    return (
        math.isclose(lhs.retention_min, rhs.retention_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            lhs.retention_max, rhs.retention_max, rel_tol=0.0, abs_tol=1e-9
        )
        and math.isclose(lhs.bounds.s_min, rhs.bounds.s_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.s_max, rhs.bounds.s_max, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_min, rhs.bounds.d_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_max, rhs.bounds.d_max, rel_tol=0.0, abs_tol=1e-9)
    )


def _build_lanes(
    *,
    users: Sequence[int],
    retentions: Sequence[float],
    train_outputs_root: Path,
    output_root: Path,
    lambda_value: float,
) -> list[BatchedSweepLogLane]:
    lanes: list[BatchedSweepLogLane] = []
    lambda_token = _format_float_token(lambda_value)
    sweep_root = output_root / "sweep_outputs"
    missing: list[Path] = []
    for user_id in users:
        for retention in retentions:
            dr_token = _format_float_token(retention)
            lanes.append(
                BatchedSweepLogLane(
                    user_id=user_id,
                    log_root=(
                        sweep_root
                        / f"user_{user_id}"
                        / "sched_fsrs6"
                        / f"dr_{dr_token}"
                    ),
                    environment="lstm",
                    scheduler_name="fsrs6",
                    scheduler_spec="fsrs6",
                    desired_retention=retention,
                    fixed_interval=None,
                    sa_fsrs6_policy=None,
                )
            )
            policy = _policy_path(
                train_outputs_root=train_outputs_root,
                user_id=user_id,
                lambda_token=lambda_token,
                retention=retention,
            )
            if not policy.exists():
                missing.append(policy)
                continue
            _validate_policy(
                SAFSRS6Policy.from_json(policy),
                path=policy,
                retention=retention,
            )
            lanes.append(
                BatchedSweepLogLane(
                    user_id=user_id,
                    log_root=(
                        sweep_root
                        / f"user_{user_id}"
                        / "sched_sa_fsrs6"
                        / f"dr_{dr_token}"
                        / f"lambda_{lambda_token}"
                    ),
                    environment="lstm",
                    scheduler_name="sa_fsrs6",
                    scheduler_spec="sa_fsrs6",
                    desired_retention=None,
                    fixed_interval=None,
                    sa_fsrs6_policy=policy,
                )
            )
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing SA FSRS-6 policies:\n{preview}{suffix}")
    return lanes


def _chunked_lanes(
    lanes: list[BatchedSweepLogLane],
    batch_size: int,
) -> list[list[BatchedSweepLogLane]]:
    return [
        lanes[index : index + batch_size] for index in range(0, len(lanes), batch_size)
    ]


def _build_scheduler_ops(
    *,
    lanes: Sequence[BatchedSweepLogLane],
    fsrs_weights: torch.Tensor,
    scheduler_priority: str,
    device: torch.device,
) -> MixedBatchSchedulerOps:
    groups: list[MixedSchedulerGroup] = []
    fsrs6_indices = [
        index for index, lane in enumerate(lanes) if lane.scheduler_name == "fsrs6"
    ]
    if fsrs6_indices:
        group_indices = torch.tensor(fsrs6_indices, device=device, dtype=torch.int64)
        desired_values: list[float] = []
        for index in fsrs6_indices:
            desired = lanes[index].desired_retention
            if desired is None:
                raise ValueError("FSRS-6 lanes require desired_retention.")
            desired_values.append(float(desired))
        desired_retention = torch.tensor(
            desired_values,
            device=device,
            dtype=torch.float32,
        )
        groups.append(
            MixedSchedulerGroup(
                lane_indices=group_indices,
                ops=FSRS6BatchSchedulerOps(
                    weights=fsrs_weights.index_select(0, group_indices),
                    desired_retention=desired_retention,
                    bounds=Bounds(),
                    priority_mode=scheduler_priority,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

    sa_indices = [
        index for index, lane in enumerate(lanes) if lane.scheduler_name == "sa_fsrs6"
    ]
    if sa_indices:
        policies: list[SAFSRS6Policy] = []
        policy_paths: list[Path] = []
        for index in sa_indices:
            policy_path = lanes[index].sa_fsrs6_policy
            if policy_path is None:
                raise ValueError("SA FSRS-6 lanes require a policy path.")
            policy = SAFSRS6Policy.from_json(policy_path)
            policies.append(policy)
            policy_paths.append(policy_path)
        template = policies[0]
        for path, policy in zip(policy_paths, policies, strict=True):
            if not _same_policy_bounds(policy, template):
                raise ValueError(
                    "Batched SA FSRS-6 external eval requires identical policy "
                    f"retention/bounds. Mismatch at {path}."
                )
        coefficients = torch.tensor(
            [policy.coefficients for policy in policies],
            device=device,
            dtype=torch.float32,
        )
        group_indices = torch.tensor(sa_indices, device=device, dtype=torch.int64)
        groups.append(
            MixedSchedulerGroup(
                lane_indices=group_indices,
                ops=SAFSRS6BatchSchedulerOps(
                    weights=fsrs_weights.index_select(0, group_indices),
                    policy=template,
                    coefficients=coefficients,
                    bounds=template.bounds,
                    priority_mode=scheduler_priority,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )

    return MixedBatchSchedulerOps(
        groups=groups,
        lane_count=len(lanes),
        device=device,
        dtype=torch.float32,
    )


def _run_lanes(
    *,
    args: argparse.Namespace,
    lanes: list[BatchedSweepLogLane],
    repo_root: Path,
    benchmark_root: Path,
    overrides: dict[str, str],
    device: torch.device,
    chunk_index: int,
    chunk_count: int,
) -> None:
    lane_user_ids = [lane.user_id for lane in lanes]
    lstm_paths, lstm_users = resolve_lstm_paths(
        lane_user_ids,
        benchmark_root,
        short_term=False,
    )
    if lstm_users != lane_user_ids:
        raise ValueError("Could not resolve LSTM weights for every eval lane.")
    fsrs_weights, fsrs_users = load_fsrs6_weights(
        repo_root=repo_root,
        user_ids=lane_user_ids,
        benchmark_root=benchmark_root,
        benchmark_partition=args.benchmark_partition,
        overrides=overrides,
        short_term=False,
        device=device,
    )
    if fsrs_weights is None or fsrs_users != lane_user_ids:
        raise ValueError("Could not resolve FSRS-6 scheduler weights for every lane.")

    lstm_packed = PackedLSTMWeights.from_paths(
        lstm_paths,
        use_duration_feature=False,
        device=device,
        dtype=torch.float32,
    )
    env_ops = LSTMBatchedEnvOps(
        lstm_packed,
        device=lstm_packed.process_0_weight.device,
        dtype=torch.float32,
    )
    scheduler_ops = _build_scheduler_ops(
        lanes=lanes,
        fsrs_weights=fsrs_weights.to(env_ops.device),
        scheduler_priority=args.scheduler_priority,
        device=env_ops.device,
    )

    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
        review_markov_success_weights,
    ) = load_usage(lane_user_ids, args.button_usage)
    behavior, cost_model = build_behavior_cost(
        len(lane_user_ids),
        deck_size=args.deck,
        learn_limit=args.learn_limit,
        review_limit=args.review_limit,
        cost_limit_minutes=args.cost_limit_minutes,
        learn_costs=learn_costs.to(env_ops.device),
        review_costs=review_costs.to(env_ops.device),
        first_rating_prob=first_rating_prob.to(env_ops.device),
        review_rating_prob=review_rating_prob.to(env_ops.device),
        learning_rating_prob=learning_rating_prob.to(env_ops.device),
        relearning_rating_prob=relearning_rating_prob.to(env_ops.device),
        state_rating_costs=state_rating_costs.to(env_ops.device),
        review_markov_success_weights=review_markov_success_weights.to(env_ops.device),
        short_term=False,
    )
    run_label = (
        f"lstm external eval chunk {chunk_index}/{chunk_count} lanes={len(lanes)}"
    )
    simulate_and_log_lanes(
        write_log=simulate_cli._write_log,
        args=args,
        lanes=lanes,
        env_ops=env_ops,
        sched_ops=scheduler_ops,
        behavior=behavior,
        cost_model=cost_model,
        progress=not args.no_progress,
        progress_queue=None,
        device_label=str(env_ops.device),
        run_label=run_label,
        short_term_source=None,
        learning_steps=[],
        relearning_steps=[],
        learning_steps_arg=None,
        relearning_steps_arg=None,
        batch_log_root=args.output_dir / "batch_logs",
    )
    if env_ops.device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FSRS-6-trained SA FSRS-6 policies in an external LSTM "
            "memory environment without retraining."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--train-run-root",
        type=Path,
        required=True,
        help="Run root containing train-overfit/train_outputs from FSRS-6 training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LSTM external-eval logs.",
    )
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument("--end-user", type=int, default=8, help="Last user id.")
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.5,
        help="SA objective lambda value used in the training output path.",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=DEFAULT_START_RETENTION,
        help="First trained baseline desired retention.",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=DEFAULT_END_RETENTION,
        help="Last trained baseline desired retention.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_RETENTION_STEP,
        help="Retention grid step.",
    )
    parser.add_argument(
        "--lane-batch-size",
        type=int,
        default=0,
        help="Lanes per simulation batch. Use 0 to run all lanes in one GPU batch.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for the batched simulation, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Simulation days.",
    )
    parser.add_argument(
        "--deck", type=int, default=DEFAULT_DECK_SIZE, help="Deck size."
    )
    parser.add_argument(
        "--learn-limit",
        type=int,
        default=DEFAULT_LEARN_LIMIT,
        help="Max new cards per day.",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=DEFAULT_REVIEW_LIMIT,
        help="Max reviews per day.",
    )
    parser.add_argument(
        "--cost-limit-minutes",
        type=float,
        default=DEFAULT_COST_LIMIT_MINUTES,
        help="Daily study time limit in minutes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default=DEFAULT_PRIORITY,
        help="Action priority.",
    )
    parser.add_argument(
        "--scheduler-priority",
        default=DEFAULT_SCHEDULER_PRIORITY,
        help="FSRS scheduler review priority mode.",
    )
    parser.add_argument(
        "--fuzz",
        action="store_true",
        help="Enable interval fuzzing. Default is off.",
    )
    parser.add_argument(
        "--short-term",
        choices=["off"],
        default="off",
        help="External LSTM eval for this experiment is short-term off only.",
    )
    parser.add_argument(
        "--short-term-threshold",
        type=float,
        default=0.5,
        help="Logged short-term threshold; unused when short-term is off.",
    )
    parser.add_argument(
        "--short-term-loops-limit",
        type=int,
        default=DEFAULT_SHORT_TERM_LOOPS_LIMIT,
        help="Logged short-term loop limit; unused when short-term is off.",
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL.",
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help="Path to the srs-benchmark repo.",
    )
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help="Override benchmark result base names, key=value comma-separated.",
    )
    parser.add_argument(
        "--benchmark-partition",
        default="0",
        help="Benchmark parameter partition key.",
    )
    parser.add_argument(
        "--diagnostic-csv-logs",
        action="store_true",
        help="Write diagnostic CSV logs. Default is off to avoid large artifacts.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable JSONL logs.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and write the eval plan without running simulation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    users = list(range(args.start_user, args.end_user + 1))
    if not users:
        raise SystemExit("Empty user range.")
    retentions = dr_values(args.start_retention, args.end_retention, args.step)
    train_outputs_root = args.train_run_root / "train-overfit" / "train_outputs"
    lanes = _build_lanes(
        users=users,
        retentions=retentions,
        train_outputs_root=train_outputs_root,
        output_root=args.output_dir,
        lambda_value=args.lambda_value,
    )
    device = torch.device(
        args.torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lane_batch_size = args.lane_batch_size if args.lane_batch_size > 0 else len(lanes)
    lane_chunks = _chunked_lanes(lanes, lane_batch_size)
    plan = {
        "type": "sa-fsrs6-external-lstm-eval-plan",
        "train_run_root": str(args.train_run_root),
        "train_outputs_root": str(train_outputs_root),
        "output_dir": str(args.output_dir),
        "environment": "lstm",
        "schedulers": ["fsrs6", "sa_fsrs6"],
        "users": users,
        "retentions": retentions,
        "lambda_value": args.lambda_value,
        "lane_count": len(lanes),
        "lane_batch_size": lane_batch_size,
        "chunk_count": len(lane_chunks),
        "days": args.days,
        "deck": args.deck,
        "learn_limit": args.learn_limit,
        "review_limit": args.review_limit,
        "cost_limit_minutes": args.cost_limit_minutes,
        "priority": args.priority,
        "scheduler_priority": args.scheduler_priority,
        "seed": args.seed,
        "fuzz": bool(args.fuzz),
        "short_term": False,
        "device": str(device),
        "diagnostic_csv_logs": bool(args.diagnostic_csv_logs),
    }
    plan_path = args.output_dir / "eval_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print(f"Wrote eval plan for {len(lanes)} lanes to {plan_path}")
        return 0

    for chunk_index, lane_chunk in enumerate(lane_chunks, start=1):
        _run_lanes(
            args=args,
            lanes=list(lane_chunk),
            repo_root=REPO_ROOT,
            benchmark_root=benchmark_root,
            overrides=overrides,
            device=device,
            chunk_index=chunk_index,
            chunk_count=len(lane_chunks),
        )

    done_path = args.output_dir / "eval_summary.json"
    done = dict(plan)
    done["completed"] = True
    done_path.write_text(json.dumps(done, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote LSTM external eval logs for {len(lanes)} lanes to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
