from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from experiments.retention_sweep.cli_utils import add_torch_device_arg, parse_csv
from experiments.single_card_tradeoff.core.config import (
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    fsrs_config_kwargs,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.core.defaults import (
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
)
from experiments.single_card_tradeoff.core.run_monitoring import (
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.oracles import (
    FSRS6ContinuousStationaryFiniteOracle,
    FSRS6ContinuousStationaryUniformTerminationOracle,
    FSRS6ContinuousUniformTerminationOracle,
    OracleMetrics,
)
from experiments.single_card_tradeoff.oracles.dp_cache import (
    oracle_dp_cache_stats_snapshot,
    reset_oracle_dp_cache_stats,
)
from simulator.defaults import DEFAULT_DAYS
from simulator.scheduler_spec import format_float

DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/continuous_uniform_h_stationary_analysis"
)
DEFAULT_LANDMARKS = [1, 2, 7, 30, 90, 365, 730, 1125, 1600, 1824]
DEFAULT_SELECTED_WEIGHTS = [0.0, 16.0, 64.0, 256.0, 1024.0]


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


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value '{item}'.") from exc
        values.append(parsed)
    if not values:
        raise SystemExit(f"{name} must contain at least one value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the best stationary approximation for hidden Uniform-H and "
            "compare it with the fixed-H continuous stationary finite oracle."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    add_torch_device_arg(parser)
    parser.add_argument("--user-ids", default=None)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
    )
    parser.add_argument("--oracle-s-grid-size", type=int, default=64)
    parser.add_argument("--oracle-d-grid-size", type=int, default=32)
    parser.add_argument("--oracle-continuous-retention-min", type=float, default=0.5)
    parser.add_argument("--oracle-continuous-retention-max", type=float, default=0.98)
    parser.add_argument("--oracle-continuous-interval-chunk-size", type=int, default=64)
    parser.add_argument("--stationary-max-iterations", type=int, default=128)
    parser.add_argument("--stationary-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--landmark-remaining-days",
        default=",".join(str(value) for value in DEFAULT_LANDMARKS),
    )
    parser.add_argument(
        "--plot-selected-cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_SELECTED_WEIGHTS),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    add_run_monitoring_args(parser)
    return parser.parse_args()


def resolve_device(raw: str | None) -> torch.device:
    if raw:
        device = torch.device(raw)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def user_ids_from_args(args: argparse.Namespace) -> list[int]:
    raw = getattr(args, "user_ids", None)
    if raw:
        values = [int(item) for item in parse_csv(str(raw))]
        if not values:
            raise SystemExit("--user-ids must contain at least one user id.")
        return values
    return [int(getattr(args, "user_id", None) or 1)]


def namespace_for_user(args: argparse.Namespace, user_id: int) -> argparse.Namespace:
    values = vars(args).copy()
    values["user_id"] = user_id
    return argparse.Namespace(**values)


def build_oracles(
    args: argparse.Namespace,
    *,
    device: torch.device,
    user_id: int,
) -> tuple[
    FSRS6ContinuousStationaryFiniteOracle,
    FSRS6ContinuousStationaryUniformTerminationOracle,
    FSRS6ContinuousUniformTerminationOracle,
]:
    user_args = namespace_for_user(args, user_id)
    fsrs_config = load_single_card_fsrs6_config(user_args)
    kwargs = {
        "days": int(args.days),
        "s_grid_size": int(args.oracle_s_grid_size),
        "d_grid_size": int(args.oracle_d_grid_size),
        "retention_min": float(args.oracle_continuous_retention_min),
        "retention_max": float(args.oracle_continuous_retention_max),
        "interval_chunk_size": int(args.oracle_continuous_interval_chunk_size),
        "device": device,
        "cache_config": configure_oracle_dp_cache_from_args(args),
        **fsrs_config_kwargs(fsrs_config),
    }
    return (
        FSRS6ContinuousStationaryFiniteOracle(**kwargs),
        FSRS6ContinuousStationaryUniformTerminationOracle(**kwargs),
        FSRS6ContinuousUniformTerminationOracle(**kwargs),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_tensor(metrics: list[OracleMetrics], attr: str) -> torch.Tensor:
    return torch.tensor(
        [float(getattr(metric, attr)) for metric in metrics],
        dtype=torch.float64,
    )


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("oracle grid sizes must be >= 8.")
    if args.oracle_continuous_interval_chunk_size <= 0:
        raise SystemExit("--oracle-continuous-interval-chunk-size must be > 0.")
    if args.stationary_max_iterations <= 0:
        raise SystemExit("--stationary-max-iterations must be > 0.")
    if args.stationary_tolerance <= 0.0:
        raise SystemExit("--stationary-tolerance must be > 0.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    landmarks = parse_csv_ints(
        args.landmark_remaining_days,
        name="--landmark-remaining-days",
    )
    selected_weights = parse_csv_floats(
        args.plot_selected_cost_weights,
        name="--plot-selected-cost-weights",
    )
    device = resolve_device(args.torch_device)
    user_ids = user_ids_from_args(args)
    horizon = int(args.days) - 1
    landmarks = sorted({value for value in landmarks if 1 <= value <= horizon})
    if horizon not in landmarks:
        landmarks.append(horizon)
    landmarks = sorted(set(landmarks))
    if not landmarks:
        raise SystemExit("--landmark-remaining-days has no values within horizon.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=out_dir,
        stage_name="continuous_uniform_h_stationary_analysis",
    )
    reset_oracle_dp_cache_stats()
    start_time = time.perf_counter()

    weight_count = len(cost_weights)
    s_count = int(args.oracle_s_grid_size)
    d_count = int(args.oracle_d_grid_size)
    uniform_policy_sum = torch.zeros(
        (weight_count, s_count, d_count), dtype=torch.float64
    )
    fixed_policy_sum = torch.zeros_like(uniform_policy_sum)
    uniform_mean = torch.zeros(weight_count, dtype=torch.float64)
    uniform_min_share = torch.zeros_like(uniform_mean)
    uniform_max_share = torch.zeros_like(uniform_mean)

    uniform_memory = torch.zeros_like(uniform_mean)
    uniform_minutes = torch.zeros_like(uniform_mean)
    uniform_reviews = torch.zeros_like(uniform_mean)
    uniform_lapses = torch.zeros_like(uniform_mean)
    uniform_objective = torch.zeros_like(uniform_mean)
    uniform_iterations = torch.zeros_like(uniform_mean)
    uniform_converged = torch.zeros_like(uniform_mean)
    uniform_residual = torch.zeros_like(uniform_mean)

    fixed_surface_delta = torch.zeros_like(uniform_mean)
    fixed_surface_abs_delta = torch.zeros_like(uniform_mean)
    fixed_surface_share_delta = torch.zeros_like(uniform_mean)
    fixed_uniform_objective = torch.zeros_like(uniform_mean)
    fixed_fixed_objective = torch.zeros_like(uniform_mean)

    nonstationary_start_objective = torch.zeros_like(uniform_mean)
    nonstationary_delta = torch.zeros(
        (weight_count, len(landmarks)), dtype=torch.float64
    )
    nonstationary_abs_delta = torch.zeros_like(nonstationary_delta)
    nonstationary_share_delta = torch.zeros_like(nonstationary_delta)

    for index, user_id in enumerate(user_ids, start=1):
        print(f"[{index}/{len(user_ids)}] solving user {user_id} on {device}")
        fixed_oracle, stationary_uniform_oracle, uniform_oracle = build_oracles(
            args,
            device=device,
            user_id=user_id,
        )
        stationary_uniform = (
            stationary_uniform_oracle.solve_stationary_uniform_termination_policies(
                cost_weights,
                max_iterations=int(args.stationary_max_iterations),
                tolerance=float(args.stationary_tolerance),
                progress=not args.no_progress,
            )
        )
        fixed_stationary = fixed_oracle.solve_stationary_finite_policies(
            cost_weights,
            max_iterations=int(args.stationary_max_iterations),
            tolerance=float(args.stationary_tolerance),
            progress=not args.no_progress,
        )
        uniform_nonstationary_policy = uniform_oracle.solve_policies(
            cost_weights,
            progress=not args.no_progress,
        )

        min_value = float(args.oracle_continuous_retention_min)
        max_value = float(args.oracle_continuous_retention_max)
        uniform_policy = stationary_uniform.policy.detach()
        fixed_policy = fixed_stationary.policy.detach()
        nonstationary_start = uniform_nonstationary_policy[:, horizon].detach()

        uniform_policy_cpu = uniform_policy.cpu().to(dtype=torch.float64)
        fixed_policy_cpu = fixed_policy.cpu().to(dtype=torch.float64)
        uniform_policy_sum += uniform_policy_cpu
        fixed_policy_sum += fixed_policy_cpu
        uniform_mean += uniform_policy_cpu.mean(dim=(1, 2))
        uniform_min_share += (
            (uniform_policy_cpu <= (min_value + 1e-7))
            .to(dtype=torch.float64)
            .mean(dim=(1, 2))
        )
        uniform_max_share += (
            (uniform_policy_cpu >= (max_value - 1e-7))
            .to(dtype=torch.float64)
            .mean(dim=(1, 2))
        )

        uniform_memory += metric_tensor(
            stationary_uniform.metrics,
            "card_expected_retrievability",
        )
        uniform_minutes += metric_tensor(
            stationary_uniform.metrics, "card_minutes_per_day"
        )
        uniform_reviews += metric_tensor(
            stationary_uniform.metrics, "card_reviews_per_day"
        )
        uniform_lapses += metric_tensor(stationary_uniform.metrics, "card_total_lapses")
        uniform_objective += metric_tensor(
            stationary_uniform.metrics, "scalar_objective"
        )
        uniform_iterations += torch.tensor(
            stationary_uniform.iterations,
            dtype=torch.float64,
        )
        uniform_converged += torch.tensor(
            [1.0 if value else 0.0 for value in stationary_uniform.converged],
            dtype=torch.float64,
        )
        uniform_residual = torch.maximum(
            uniform_residual,
            torch.tensor(stationary_uniform.residuals, dtype=torch.float64),
        )

        fixed_delta = (uniform_policy - fixed_policy).cpu().to(dtype=torch.float64)
        fixed_surface_delta += fixed_delta.mean(dim=(1, 2))
        fixed_surface_abs_delta += torch.abs(fixed_delta).mean(dim=(1, 2))
        fixed_surface_share_delta += (
            (torch.abs(fixed_delta) > 0.02).to(dtype=torch.float64).mean(dim=(1, 2))
        )

        fixed_under_uniform = (
            stationary_uniform_oracle.evaluate_stationary_uniform_termination_policy(
                policy=fixed_policy,
                cost_weights=cost_weights,
            )
        )
        nonstationary_start_under_uniform = (
            stationary_uniform_oracle.evaluate_stationary_uniform_termination_policy(
                policy=nonstationary_start,
                cost_weights=cost_weights,
            )
        )
        fixed_uniform_objective += metric_tensor(
            fixed_under_uniform, "scalar_objective"
        )
        fixed_fixed_objective += metric_tensor(
            fixed_stationary.metrics, "scalar_objective"
        )
        nonstationary_start_objective += metric_tensor(
            nonstationary_start_under_uniform,
            "scalar_objective",
        )

        for landmark_idx, rem in enumerate(landmarks):
            reference = uniform_nonstationary_policy[:, rem].detach()
            delta = (uniform_policy - reference).cpu().to(dtype=torch.float64)
            nonstationary_delta[:, landmark_idx] += delta.mean(dim=(1, 2))
            nonstationary_abs_delta[:, landmark_idx] += torch.abs(delta).mean(
                dim=(1, 2)
            )
            nonstationary_share_delta[:, landmark_idx] += (
                (torch.abs(delta) > 0.02).to(dtype=torch.float64).mean(dim=(1, 2))
            )

        del (
            stationary_uniform,
            fixed_stationary,
            uniform_nonstationary_policy,
            uniform_policy,
            fixed_policy,
            nonstationary_start,
            fixed_delta,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    divisor = float(len(user_ids))
    for tensor in (
        uniform_policy_sum,
        fixed_policy_sum,
        uniform_mean,
        uniform_min_share,
        uniform_max_share,
        uniform_memory,
        uniform_minutes,
        uniform_reviews,
        uniform_lapses,
        uniform_objective,
        uniform_iterations,
        uniform_converged,
        fixed_surface_delta,
        fixed_surface_abs_delta,
        fixed_surface_share_delta,
        fixed_uniform_objective,
        fixed_fixed_objective,
        nonstationary_start_objective,
        nonstationary_delta,
        nonstationary_abs_delta,
        nonstationary_share_delta,
    ):
        tensor /= divisor

    summary_rows = []
    fixed_rows = []
    nonstationary_rows = []
    for weight_idx, weight in enumerate(cost_weights):
        summary_rows.append(
            {
                "cost_weight": weight,
                "mean_retention": float(uniform_mean[weight_idx].item()),
                "retention_min_share": float(uniform_min_share[weight_idx].item()),
                "retention_max_share": float(uniform_max_share[weight_idx].item()),
                "card_expected_retrievability": float(
                    uniform_memory[weight_idx].item()
                ),
                "card_minutes_per_day": float(uniform_minutes[weight_idx].item()),
                "card_reviews_per_day": float(uniform_reviews[weight_idx].item()),
                "card_total_lapses": float(uniform_lapses[weight_idx].item()),
                "uniform_h_objective": float(uniform_objective[weight_idx].item()),
                "mean_iterations": float(uniform_iterations[weight_idx].item()),
                "converged_share": float(uniform_converged[weight_idx].item()),
                "max_residual": float(uniform_residual[weight_idx].item()),
            }
        )
        fixed_rows.append(
            {
                "cost_weight": weight,
                "mean_delta_uniform_stationary_minus_fixed_stationary": float(
                    fixed_surface_delta[weight_idx].item()
                ),
                "mean_abs_delta_uniform_stationary_vs_fixed_stationary": float(
                    fixed_surface_abs_delta[weight_idx].item()
                ),
                "share_abs_delta_gt_0_02": float(
                    fixed_surface_share_delta[weight_idx].item()
                ),
                "uniform_stationary_uniform_h_objective": float(
                    uniform_objective[weight_idx].item()
                ),
                "fixed_stationary_uniform_h_objective": float(
                    fixed_uniform_objective[weight_idx].item()
                ),
                "objective_delta_under_uniform_h": float(
                    (
                        uniform_objective[weight_idx]
                        - fixed_uniform_objective[weight_idx]
                    ).item()
                ),
                "fixed_stationary_fixed_h_objective": float(
                    fixed_fixed_objective[weight_idx].item()
                ),
            }
        )
        for landmark_idx, rem in enumerate(landmarks):
            nonstationary_rows.append(
                {
                    "cost_weight": weight,
                    "uniform_nonstationary_remaining_days": rem,
                    "mean_delta_stationary_minus_nonstationary": float(
                        nonstationary_delta[weight_idx, landmark_idx].item()
                    ),
                    "mean_abs_delta_stationary_vs_nonstationary": float(
                        nonstationary_abs_delta[weight_idx, landmark_idx].item()
                    ),
                    "share_abs_delta_gt_0_02": float(
                        nonstationary_share_delta[weight_idx, landmark_idx].item()
                    ),
                    "stationary_uniform_h_objective": float(
                        uniform_objective[weight_idx].item()
                    ),
                    "nonstationary_start_as_stationary_uniform_h_objective": float(
                        nonstationary_start_objective[weight_idx].item()
                    ),
                }
            )

    write_csv(out_dir / "stationary_uniform_summary.csv", summary_rows)
    write_csv(out_dir / "stationary_uniform_vs_fixed_stationary.csv", fixed_rows)
    write_csv(
        out_dir / "stationary_uniform_vs_nonstationary_uniform.csv",
        nonstationary_rows,
    )

    metadata = {
        "users": user_ids,
        "days": int(args.days),
        "horizon": horizon,
        "cost_weights": cost_weights,
        "s_grid_size": s_count,
        "d_grid_size": d_count,
        "retention_min": float(args.oracle_continuous_retention_min),
        "retention_max": float(args.oracle_continuous_retention_max),
        "stationary_max_iterations": int(args.stationary_max_iterations),
        "stationary_tolerance": float(args.stationary_tolerance),
        "termination_distribution": (
            FSRS6ContinuousStationaryUniformTerminationOracle.TERMINATION_DISTRIBUTION_VERSION
        ),
        "policy_iteration": (
            FSRS6ContinuousStationaryUniformTerminationOracle.STATIONARY_UNIFORM_POLICY_ITERATION_VERSION
        ),
        "elapsed_s": time.perf_counter() - start_time,
        "device": str(device),
        "dp_cache": oracle_dp_cache_stats_snapshot(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if not args.no_plot:
        write_plots(
            out_dir=out_dir,
            cost_weights=cost_weights,
            selected_weights=selected_weights,
            landmarks=landmarks,
            uniform_mean=uniform_mean,
            uniform_min_share=uniform_min_share,
            fixed_surface_abs_delta=fixed_surface_abs_delta,
            nonstationary_abs_delta=nonstationary_abs_delta,
            uniform_objective=uniform_objective,
            fixed_uniform_objective=fixed_uniform_objective,
            nonstationary_start_objective=nonstationary_start_objective,
            uniform_policy=uniform_policy_sum,
            fixed_policy=fixed_policy_sum,
            retention_min=float(args.oracle_continuous_retention_min),
            retention_max=float(args.oracle_continuous_retention_max),
        )
    write_readme(out_dir=out_dir, metadata=metadata, landmarks=landmarks)


def selected_weight_indices(
    cost_weights: list[float],
    selected_weights: list[float],
) -> list[int]:
    indices: list[int] = []
    for selected in selected_weights:
        best_idx = min(
            range(len(cost_weights)),
            key=lambda idx: abs(float(cost_weights[idx]) - float(selected)),
        )
        if best_idx not in indices:
            indices.append(best_idx)
    return indices


def write_plots(
    *,
    out_dir: Path,
    cost_weights: list[float],
    selected_weights: list[float],
    landmarks: list[int],
    uniform_mean: torch.Tensor,
    uniform_min_share: torch.Tensor,
    fixed_surface_abs_delta: torch.Tensor,
    nonstationary_abs_delta: torch.Tensor,
    uniform_objective: torch.Tensor,
    fixed_uniform_objective: torch.Tensor,
    nonstationary_start_objective: torch.Tensor,
    uniform_policy: torch.Tensor,
    fixed_policy: torch.Tensor,
    retention_min: float,
    retention_max: float,
) -> None:
    import matplotlib.pyplot as plt

    indices = selected_weight_indices(cost_weights, selected_weights)
    labels = [f"w={format_float(cost_weights[idx])}" for idx in indices]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(cost_weights, uniform_mean.numpy(), marker="o")
    axes[1].plot(cost_weights, uniform_min_share.numpy(), marker="o")
    axes[0].set_ylabel("Mean retention")
    axes[1].set_ylabel("Retention-min share")
    axes[1].set_xlabel("Cost weight")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "stationary_uniform_mean_retention_and_min_share.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cost_weights, fixed_surface_abs_delta.numpy(), marker="o")
    ax.set_xlabel("Cost weight")
    ax.set_ylabel("Mean abs retention delta")
    ax.set_title("Stationary Uniform-H vs fixed-H stationary")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "stationary_uniform_vs_fixed_stationary_delta.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, label in zip(indices, labels, strict=True):
        ax.plot(
            landmarks,
            nonstationary_abs_delta[idx].numpy(),
            marker="o",
            label=label,
        )
    ax.set_xlabel("Uniform-H nonstationary remaining days")
    ax.set_ylabel("Mean abs retention delta")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stationary_uniform_vs_nonstationary_uniform_delta.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        cost_weights,
        uniform_objective.numpy(),
        marker="o",
        label="stationary Uniform-H",
    )
    ax.plot(
        cost_weights,
        fixed_uniform_objective.numpy(),
        marker="o",
        label="fixed-H stationary evaluated under Uniform-H",
    )
    ax.plot(
        cost_weights,
        nonstationary_start_objective.numpy(),
        marker="o",
        label="Uniform-H start slice used stationary",
    )
    ax.set_xlabel("Cost weight")
    ax.set_ylabel("Uniform-H scalar objective")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "uniform_h_objective_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    image: Any | None = None
    for ax, idx, label in zip(axes, indices, labels, strict=True):
        image = ax.imshow(
            uniform_policy[idx].numpy(),
            aspect="auto",
            origin="lower",
            vmin=retention_min,
            vmax=retention_max,
        )
        ax.set_title(label)
        ax.set_xlabel("Difficulty grid index")
        ax.set_ylabel("Stability grid index")
    fig.suptitle("Stationary Uniform-H retention policy")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(out_dir / "stationary_uniform_policy_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(12.5, max(4.5, 2.8 * len(indices))),
        squeeze=False,
        constrained_layout=True,
    )
    delta = uniform_policy - fixed_policy
    delta_limit = max(0.02, float(torch.max(torch.abs(delta[indices])).item()))
    policy_image: Any | None = None
    delta_image: Any | None = None
    column_titles = [
        "Uniform-H stationary",
        "Fixed-H stationary",
        "Uniform-H - fixed-H",
    ]
    for row_idx, (idx, label) in enumerate(zip(indices, labels, strict=True)):
        policy_image = axes[row_idx, 0].imshow(
            uniform_policy[idx].numpy(),
            aspect="auto",
            origin="lower",
            vmin=retention_min,
            vmax=retention_max,
        )
        axes[row_idx, 0].set_ylabel(f"{label}\nStability index", fontsize=9)

        axes[row_idx, 1].imshow(
            fixed_policy[idx].numpy(),
            aspect="auto",
            origin="lower",
            vmin=retention_min,
            vmax=retention_max,
        )

        delta_image = axes[row_idx, 2].imshow(
            delta[idx].numpy(),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-delta_limit,
            vmax=delta_limit,
        )

        if row_idx == 0:
            for col_idx, title in enumerate(column_titles):
                axes[row_idx, col_idx].set_title(title, fontsize=10, pad=8)
        if row_idx < len(indices) - 1:
            for ax in axes[row_idx]:
                ax.tick_params(labelbottom=False)

    for ax in axes[-1]:
        ax.set_xlabel("Difficulty index")
    if policy_image is not None:
        fig.colorbar(
            policy_image,
            ax=axes[:, :2].ravel().tolist(),
            shrink=0.82,
            label="Retention",
        )
    if delta_image is not None:
        fig.colorbar(
            delta_image,
            ax=axes[:, 2].ravel().tolist(),
            shrink=0.82,
            label="Retention delta",
        )
    fig.suptitle("Stationary Uniform-H vs fixed-H Stationary Policy", fontsize=12)
    fig.savefig(
        out_dir / "stationary_uniform_fixed_policy_comparison_heatmaps.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_readme(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    landmarks: list[int],
) -> None:
    lines = [
        "# Continuous Uniform-H Stationary Analysis Artifacts",
        "",
        "Source policies: hidden Uniform-H stationary approximation, fixed-H "
        "`FSRS6ContinuousStationaryFiniteOracle`, and nonstationary hidden "
        "Uniform-H reference slices.",
        "",
        f"Users: {', '.join(str(user_id) for user_id in metadata['users'])}.",
        f"Lifecycle upper bound: {metadata['days']} days.",
        (
            f"Grid: {metadata['s_grid_size']} log-spaced stability values x "
            f"{metadata['d_grid_size']} difficulty values."
        ),
        f"Cost weights: {', '.join(format_float(value) for value in metadata['cost_weights'])}.",
        (
            "The fixed-H stationary objective is reported separately because its "
            "native scalar objective uses a different terminal model."
        ),
        "",
        "Files:",
        "",
        "- `stationary_uniform_summary.csv`: Uniform-H stationary policy surface "
        "and Uniform-H metrics by cost weight.",
        "- `stationary_uniform_vs_fixed_stationary.csv`: fixed-H stationary policy "
        "surface deltas and Uniform-H objective deltas.",
        "- `stationary_uniform_vs_nonstationary_uniform.csv`: deltas to selected "
        f"nonstationary Uniform-H remaining slices {landmarks}.",
        "- `stationary_uniform_fixed_policy_comparison_heatmaps.png`: direct "
        "Uniform-H stationary, fixed-H stationary, and delta heatmaps.",
        "- `metadata.json`: run configuration, elapsed time, and DP cache counters.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
