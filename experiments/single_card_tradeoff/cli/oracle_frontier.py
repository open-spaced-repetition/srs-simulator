from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.oracles import (  # noqa: E402,F401
    AverageRewardOracleSolution,
    BatchedStationaryFiniteOracleSolution,
    BatchedTransitionCache,
    FSRS6AverageRewardOracle,
    FSRS6BatchedStationaryFiniteOracle,
    FSRS6GridOracle,
    FSRS6IntervalOracle,
    FSRS6StationaryFiniteOracle,
    OracleMetrics,
    OracleSolution,
    StationaryActionKernelTables,
    StationaryFiniteOracleSolution,
    TransitionCache,
    scalar_objective,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.defaults import DEFAULT_TARGET_RETENTIONS
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float

DEFAULT_COST_WEIGHTS = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]


def parse_csv_floats(value: str, *, name: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value '{item}'.") from exc
        if not math.isfinite(parsed):
            raise SystemExit(f"{name} values must be finite.")
        values.append(parsed)
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a single-card FSRS6 oracle frontier with grid DP.",
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help="Comma-separated scalarization weights for oracle frontier points.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to the oracle.",
    )
    parser.add_argument(
        "--s-grid-size",
        type=int,
        default=64,
        help="Number of log-spaced stability grid points.",
    )
    parser.add_argument(
        "--d-grid-size",
        type=int,
        default=32,
        help="Number of linearly spaced difficulty grid points.",
    )
    parser.add_argument(
        "--baseline-particles",
        type=int,
        default=10_000,
        help="Particles for static-FSRS baseline rows. Set 0 to skip baselines.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/oracle_frontier.csv"),
    )
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def row_from_metrics(
    *,
    args: argparse.Namespace,
    scheduler: str,
    scheduler_spec: str,
    metrics: Any,
    goal_cost_weight: float | None,
    desired_retention: float | None,
    runtime_s: float,
    scalar: float | None,
    delta_vs_best_fsrs: float | None,
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    return {
        "environment": args.env,
        "scheduler": scheduler,
        "scheduler_spec": scheduler_spec,
        "goal_cost_weight": goal_cost_weight,
        "desired_retention": desired_retention,
        "fixed_interval": None,
        "seed": args.seed,
        "days": args.days,
        "particles": 0 if scheduler == "oracle_grid" else args.baseline_particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "scalar_objective": scalar,
        "delta_vs_best_fsrs": delta_vs_best_fsrs,
        "runtime_s": runtime_s,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "scheduler",
        "scheduler_spec",
        "goal_cost_weight",
        "desired_retention",
        "fixed_interval",
        "seed",
        "days",
        "particles",
        "deck_scale",
        "card_expected_retrievability",
        "card_minutes_per_day",
        "card_reviews_per_day",
        "card_total_reviews",
        "card_total_lapses",
        "card_total_cost_seconds",
        "observed_retention",
        "deck_expected_memorized",
        "deck_minutes_per_day",
        "deck_reviews_per_day",
        "scalar_objective",
        "delta_vs_best_fsrs",
        "runtime_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["scheduler"]), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in groups.items():
        if label == "fsrs6_static":
            group = sorted(group, key=lambda row: float(row["desired_retention"]))
        else:
            group = sorted(group, key=lambda row: float(row["goal_cost_weight"]))
        ax.plot(
            [row["deck_expected_memorized"] for row in group],
            [row["deck_minutes_per_day"] for row in group],
            marker="o",
            linewidth=1.4 if label == "oracle_grid" else 1.0,
            alpha=0.9 if label == "oracle_grid" else 0.5,
            label=label,
        )
    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title("FSRS6 single-card oracle frontier estimate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    from importlib import import_module

    evaluate_static_fsrs = import_module(
        "experiments.single_card_tradeoff.cli.uvfa_ppo"
    ).evaluate_static_fsrs

    args = parse_args()
    cache_config = configure_oracle_dp_cache_from_args(args)
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.baseline_particles < 0:
        raise SystemExit("--baseline-particles must be >= 0.")
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    fsrs_config = load_single_card_fsrs6_config(args)
    oracle = FSRS6GridOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
        cache_config=cache_config,
        fsrs_weights=fsrs_config.fsrs_weights,
        first_rating_prob=fsrs_config.first_rating_prob,
        review_rating_prob=fsrs_config.review_rating_prob,
        learning_costs=fsrs_config.learning_costs,
        review_costs=fsrs_config.review_costs,
    )
    rows: list[dict[str, Any]] = []
    fsrs_metrics_by_weight: dict[float, float] = {}

    if args.baseline_particles > 0:
        baseline_ns = argparse.Namespace(days=args.days)
        for retention in action_retentions:
            metrics = evaluate_static_fsrs(
                args=baseline_ns,
                device=torch.device("cpu"),
                retention=retention,
                particles=args.baseline_particles,
                seed=args.seed + 10_000 + int(round(retention * 10_000)),
                fsrs_config=fsrs_config,
            )
            for cost_weight in cost_weights:
                scalar = scalar_objective(metrics, cost_weight)
                current = fsrs_metrics_by_weight.get(cost_weight)
                if current is None or scalar > current:
                    fsrs_metrics_by_weight[cost_weight] = scalar
            rows.append(
                row_from_metrics(
                    args=args,
                    scheduler="fsrs6_static",
                    scheduler_spec=f"fsrs@{format_float(retention)}",
                    metrics=metrics,
                    goal_cost_weight=None,
                    desired_retention=retention,
                    runtime_s=0.0,
                    scalar=None,
                    delta_vs_best_fsrs=None,
                )
            )

    metrics_by_oracle_weight = oracle.estimate_many(
        cost_weights,
        progress=not args.no_progress,
    )
    for cost_weight, metrics in zip(
        cost_weights,
        metrics_by_oracle_weight,
        strict=True,
    ):
        best_fsrs = fsrs_metrics_by_weight.get(cost_weight)
        delta = metrics.scalar_objective - best_fsrs if best_fsrs is not None else None
        rows.append(
            row_from_metrics(
                args=args,
                scheduler="oracle_grid",
                scheduler_spec=f"oracle_grid_{args.s_grid_size}x{args.d_grid_size}",
                metrics=metrics,
                goal_cost_weight=cost_weight,
                desired_retention=None,
                runtime_s=metrics.runtime_s,
                scalar=metrics.scalar_objective,
                delta_vs_best_fsrs=delta,
            )
        )
        print(
            " ".join(
                [
                    f"oracle w={format_float(cost_weight)}",
                    f"card_mem={metrics.card_expected_retrievability:.4f}",
                    f"card_min/day={metrics.card_minutes_per_day:.6f}",
                    f"scalar={metrics.scalar_objective:.6f}",
                    f"delta_fsrs={delta:.6f}" if delta is not None else "delta_fsrs=NA",
                    f"runtime_s={metrics.runtime_s:.2f}",
                ]
            )
        )

    write_csv(args.out, rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        write_plot(plot_path, rows)
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {args.out}")


if __name__ == "__main__":
    main()
