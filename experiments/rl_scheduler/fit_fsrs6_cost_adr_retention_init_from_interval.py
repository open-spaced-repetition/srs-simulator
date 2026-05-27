from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.cli.fsrs6_cost_adr_train import (  # noqa: E402
    TeacherTable,
    fit_policy_from_table,
)
from simulator.batched_sweep.fsrs6_cost_adr_policy import (  # noqa: E402
    DEFAULT_COST_WEIGHTS,
)
from simulator.benchmark_loader import (  # noqa: E402
    load_benchmark_weights,
    parse_result_overrides,
    resolve_benchmark_root,
)
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FSRS6CostConditionedADRPolicy,
    STATE_FEATURE_COUNT_COMPACT,
)
from simulator.math.fsrs import (  # noqa: E402
    Bounds,
    FSRS6Params,
    fsrs6_forgetting_curve,
)


DEFAULT_OUT_DIR = Path(
    "artifacts/rl_scheduler/fsrs6_cost_adr_retention_init_from_interval_first8_stdpre_r030_0995"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-user FSRS6 Cost-ADR initial policies from "
            "the implied retention distribution of interval-head policies."
        ),
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--interval-policy-root",
        type=Path,
        help="Root containing user_*/policy.json interval-head policies.",
    )
    source.add_argument(
        "--interval-train-run-root",
        type=Path,
        help="Run root containing train-overfit/train_outputs interval policies.",
    )
    parser.add_argument(
        "--users",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated user ids to fit.",
    )
    parser.add_argument(
        "--cost-weights",
        default=",".join(f"{value:g}" for value in DEFAULT_COST_WEIGHTS),
        help="Comma-separated Cost-ADR cost weights.",
    )
    parser.add_argument("--s-points", type=int, default=64)
    parser.add_argument("--d-points", type=int, default=32)
    parser.add_argument("--retention-min", type=float, default=0.30)
    parser.add_argument("--retention-max", type=float, default=0.995)
    parser.add_argument("--epochs", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _parse_csv_ints(raw: str, *, field: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{field} must not be empty.")
    if len(set(values)) != len(values):
        raise SystemExit(f"{field} must not contain duplicates.")
    if any(value <= 0 for value in values):
        raise SystemExit(f"{field} values must be positive.")
    return values


def _parse_csv_floats(raw: str, *, field: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise SystemExit(f"{field} must not be empty.")
    if len(set(values)) != len(values):
        raise SystemExit(f"{field} must not contain duplicates.")
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        raise SystemExit(f"{field} values must be finite and >= 0.")
    return values


def _resolve_interval_policy_root(args: argparse.Namespace) -> Path:
    if args.interval_train_run_root is not None:
        return args.interval_train_run_root / "train-overfit" / "train_outputs"
    return args.interval_policy_root


def _linspace(
    start: float, end: float, count: int, *, device: torch.device
) -> torch.Tensor:
    if count < 2:
        raise ValueError("count must be >= 2.")
    return torch.linspace(start, end, steps=count, device=device, dtype=torch.float64)


def _logspace(
    start: float, end: float, count: int, *, device: torch.device
) -> torch.Tensor:
    if start <= 0.0:
        raise ValueError("logspace start must be > 0.")
    return torch.exp(
        torch.linspace(
            math.log(start),
            math.log(end),
            steps=count,
            device=device,
            dtype=torch.float64,
        )
    )


def _load_interval_policy(
    policy_root: Path, *, user_id: int
) -> tuple[Path, FSRS6CostConditionedADRPolicy]:
    policy_path = policy_root / f"user_{int(user_id)}" / "policy.json"
    if not policy_path.exists():
        raise SystemExit(f"Interval policy not found: {policy_path}")
    policy = FSRS6CostConditionedADRPolicy.from_json(policy_path)
    if policy.action_head != ACTION_HEAD_INTERVAL:
        raise SystemExit(
            f"{policy_path} must use action_head={ACTION_HEAD_INTERVAL!r}, "
            f"got {policy.action_head!r}."
        )
    return policy_path, policy


def _load_fsrs6_params(
    *,
    user_id: int,
    benchmark_root: Path,
    benchmark_result: str | None,
    benchmark_partition: str | None,
) -> FSRS6Params:
    weights = load_benchmark_weights(
        repo_root=REPO_ROOT,
        benchmark_root=benchmark_root,
        environment="fsrs6",
        user_id=user_id,
        partition_key=benchmark_partition or "0",
        overrides=parse_result_overrides(benchmark_result),
        short_term=False,
    )
    return FSRS6Params(tuple(weights))


def _distribution_stats(
    values: torch.Tensor,
    *,
    retention_min: float,
    retention_max: float,
) -> dict[str, float]:
    flat = values.reshape(-1).to(dtype=torch.float64)
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.0, 0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999, 1.0],
            device=flat.device,
            dtype=torch.float64,
        ),
    )
    names = [
        "q000",
        "q001",
        "q005",
        "q010",
        "q050",
        "q500",
        "q950",
        "q990",
        "q995",
        "q999",
        "q100",
    ]
    stats = {
        name: float(value.item()) for name, value in zip(names, quantiles, strict=True)
    }
    stats.update(
        {
            "mean": float(torch.mean(flat).item()),
            "std": float(torch.std(flat, unbiased=False).item()),
            "below_retention_min_fraction": float(
                torch.mean((flat < retention_min).to(dtype=torch.float64)).item()
            ),
            "above_retention_max_fraction": float(
                torch.mean((flat > retention_max).to(dtype=torch.float64)).item()
            ),
            "below_0_50_fraction": float(
                torch.mean((flat < 0.5).to(dtype=torch.float64)).item()
            ),
            "above_0_98_fraction": float(
                torch.mean((flat > 0.98).to(dtype=torch.float64)).item()
            ),
        }
    )
    return stats


def _implied_retention_table(
    *,
    interval_policy: FSRS6CostConditionedADRPolicy,
    fsrs_params: FSRS6Params,
    cost_weights: Sequence[float],
    s_points: int,
    d_points: int,
    retention_min: float,
    retention_max: float,
    device: torch.device,
) -> tuple[TeacherTable, dict[str, float]]:
    bounds = Bounds()
    s_grid = _logspace(bounds.s_min, bounds.s_max, s_points, device=device)
    d_grid = _linspace(bounds.d_min, bounds.d_max, d_points, device=device)
    retentions: list[torch.Tensor] = []
    for cost_weight in cost_weights:
        rows: list[list[float]] = []
        for s_value in s_grid.detach().cpu().tolist():
            row: list[float] = []
            for d_value in d_grid.detach().cpu().tolist():
                interval_days = interval_policy.evaluate_interval(
                    float(s_value),
                    float(d_value),
                    cost_weight=float(cost_weight),
                )
                row.append(
                    fsrs6_forgetting_curve(fsrs_params, interval_days, float(s_value))
                )
            rows.append(row)
        retentions.append(torch.tensor(rows, device=device, dtype=torch.float64))

    policy = torch.stack(retentions, dim=0)
    decay = torch.tensor(
        -float(fsrs_params.weights[20]), device=device, dtype=torch.float64
    )
    factor = (
        torch.pow(
            torch.tensor(0.9, device=device, dtype=torch.float64),
            1.0 / decay,
        )
        - 1.0
    )
    table = TeacherTable(
        s_grid=s_grid,
        d_grid=d_grid,
        policy=policy,
        factor=factor,
        decay=decay,
        retention_min=retention_min,
        retention_max=retention_max,
        bounds=bounds,
        source="fsrs6_cost_adr_interval_implied_retention",
    )
    return table, _distribution_stats(
        policy,
        retention_min=retention_min,
        retention_max=retention_max,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.s_points < 2 or args.d_points < 2:
        raise SystemExit("--s-points and --d-points must be >= 2.")
    if not (0.0 < args.retention_min < args.retention_max < 1.0):
        raise SystemExit("--retention-min/max must satisfy 0 < min < max < 1.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0.")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be > 0.")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.max_grad_norm <= 0.0:
        raise SystemExit("--max-grad-norm must be > 0.")

    user_ids = _parse_csv_ints(args.users, field="--users")
    cost_weights = _parse_csv_floats(args.cost_weights, field="--cost-weights")
    policy_root = _resolve_interval_policy_root(args)
    benchmark_root = resolve_benchmark_root(REPO_ROOT, args.srs_benchmark_root)
    device = torch.device(args.torch_device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []
    for user_id in user_ids:
        policy_path, interval_policy = _load_interval_policy(
            policy_root, user_id=user_id
        )
        fsrs_params = _load_fsrs6_params(
            user_id=user_id,
            benchmark_root=benchmark_root,
            benchmark_result=args.benchmark_result,
            benchmark_partition=args.benchmark_partition,
        )
        table, distribution = _implied_retention_table(
            interval_policy=interval_policy,
            fsrs_params=fsrs_params,
            cost_weights=cost_weights,
            s_points=args.s_points,
            d_points=args.d_points,
            retention_min=args.retention_min,
            retention_max=args.retention_max,
            device=device,
        )
        policy, fit_stats = fit_policy_from_table(
            table=table,
            cost_weights=cost_weights,
            action_head=ACTION_HEAD_RETENTION,
            state_feature_count=STATE_FEATURE_COUNT_COMPACT,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
        user_dir = args.out_dir / f"user_{user_id}"
        policy_path_out = user_dir / "policy.json"
        policy.write_json(policy_path_out)
        metadata = {
            "policy_kind": "fsrs6-cost-conditioned-adr-retention-init",
            "user_id": user_id,
            "source_interval_policy_path": str(policy_path),
            "action_head": ACTION_HEAD_RETENTION,
            "state_feature_count": STATE_FEATURE_COUNT_COMPACT,
            "cost_weights": list(cost_weights),
            "retention_min": args.retention_min,
            "retention_max": args.retention_max,
            "s_points": args.s_points,
            "d_points": args.d_points,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "torch_device": str(device),
            "benchmark_root": str(benchmark_root),
            "benchmark_partition": args.benchmark_partition or "0",
            "fit_stats": fit_stats,
            "implied_retention_distribution": distribution,
            "policy_path": "policy.json",
        }
        _write_json(user_dir / "metadata.json", metadata)
        summary_rows.append(
            {
                "user_id": user_id,
                "policy_path": str(policy_path_out),
                "source_interval_policy_path": str(policy_path),
                "fit_stats": fit_stats,
                "implied_retention_distribution": distribution,
            }
        )
        print(policy_path_out)

    summary = {
        "schema_version": 1,
        "source": "fsrs6_cost_adr_interval_implied_retention",
        "interval_policy_root": str(policy_root),
        "out_dir": str(args.out_dir),
        "users": user_ids,
        "cost_weights": list(cost_weights),
        "retention_min": args.retention_min,
        "retention_max": args.retention_max,
        "s_points": args.s_points,
        "d_points": args.d_points,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "runtime_s": time.perf_counter() - started_at,
        "rows": summary_rows,
    }
    _write_json(args.out_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
