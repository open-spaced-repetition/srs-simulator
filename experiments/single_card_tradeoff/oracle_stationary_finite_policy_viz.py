from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.oracle_frontier import (  # noqa: E402
    FSRS6StationaryFiniteOracle,
    StationaryFiniteOracleSolution,
    parse_csv_floats,
)
from experiments.single_card_tradeoff.tradeoff import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.uvfa_ppo import fsrs_config_kwargs  # noqa: E402
from simulator.defaults import DEFAULT_DAYS  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402

DEFAULT_OUT_DIR = Path("artifacts/single_card_tradeoff/stationary_finite_policy_viz")
DEFAULT_SELECTED_WEIGHTS = [0.0, 16.0, 64.0, 256.0, 1024.0]
DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS = 128
DEFAULT_STATIONARY_FINITE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class BinSpec:
    indices: torch.Tensor
    min_values: tuple[float, ...]
    max_values: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.min_values)


@dataclass(frozen=True)
class OutputPaths:
    summary: Path
    grid: Path
    binned: Path
    report: Path
    distribution_plot: Path
    heatmap_plot: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and visualize FSRS-6 stationary finite-lifecycle oracle "
            "actions over the (stability, difficulty, cost weight) policy grid."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help="Comma-separated scalarization weights to solve and visualize.",
    )
    parser.add_argument(
        "--selected-weights",
        default=",".join(format_float(value) for value in DEFAULT_SELECTED_WEIGHTS),
        help=(
            "Comma-separated weights to show in the policy heatmap panel. "
            "The nearest solved weight is used for each requested value."
        ),
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
        "--s-bins",
        type=int,
        default=8,
        help="Number of stability bins for the binned action CSV.",
    )
    parser.add_argument(
        "--d-bins",
        type=int,
        default=8,
        help="Number of difficulty bins for the binned action CSV.",
    )
    parser.add_argument(
        "--oracle-stationary-finite-max-iterations",
        type=int,
        default=DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
        help="Policy-iteration limit for the stationary finite oracle.",
    )
    parser.add_argument(
        "--oracle-stationary-finite-tolerance",
        type=float,
        default=DEFAULT_STATIONARY_FINITE_TOLERANCE,
        help="Convergence tolerance for the stationary finite oracle.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for oracle solving, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV, report, and plot outputs.",
    )
    parser.add_argument("--no-grid-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _resolve_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _output_paths(out_dir: Path) -> OutputPaths:
    return OutputPaths(
        summary=out_dir / "action_summary.csv",
        grid=out_dir / "grid_actions.csv",
        binned=out_dir / "binned_actions.csv",
        report=out_dir / "findings.md",
        distribution_plot=out_dir / "action_distribution.png",
        heatmap_plot=out_dir / "policy_heatmaps.png",
    )


def _build_bin_spec(values: torch.Tensor, requested_bins: int) -> BinSpec:
    if requested_bins <= 0:
        raise ValueError("bin counts must be > 0.")
    flat_values = values.reshape(-1)
    if flat_values.numel() <= 0:
        raise ValueError("cannot build bins for an empty tensor.")

    count = min(int(requested_bins), int(flat_values.numel()))
    positions = torch.arange(
        int(flat_values.numel()),
        device=flat_values.device,
        dtype=torch.int64,
    )
    indices = torch.div(
        positions * count,
        int(flat_values.numel()),
        rounding_mode="floor",
    ).clamp(max=count - 1)

    cpu_values = flat_values.detach().to(device="cpu")
    cpu_indices = indices.detach().to(device="cpu")
    min_values: list[float] = []
    max_values: list[float] = []
    for bin_idx in range(count):
        members = cpu_values[cpu_indices == bin_idx]
        min_values.append(float(members.min().item()))
        max_values.append(float(members.max().item()))
    return BinSpec(
        indices=indices,
        min_values=tuple(min_values),
        max_values=tuple(max_values),
    )


def _entropy(shares: torch.Tensor) -> float:
    positive = shares[shares > 0.0]
    if positive.numel() <= 0:
        return 0.0
    return float((-(positive * torch.log2(positive))).sum().item())


def _summary_rows(
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    solution: StationaryFiniteOracleSolution,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> list[dict[str, Any]]:
    action_tensor = torch.tensor(
        list(action_retentions),
        device=solution.policy.device,
        dtype=torch.float64,
    )
    policy = solution.policy
    rows: list[dict[str, Any]] = []
    for weight_idx, cost_weight in enumerate(cost_weights):
        flat_actions = policy[weight_idx].reshape(-1)
        counts = torch.bincount(
            flat_actions,
            minlength=len(action_retentions),
        )[: len(action_retentions)].to(dtype=torch.float64)
        total = float(counts.sum().item())
        shares = counts / max(total, 1.0)
        mean_retention = float((shares * action_tensor).sum().item())
        modal_idx = int(torch.argmax(counts).item())
        modal_share = float(shares[modal_idx].item())
        entropy_bits = _entropy(shares)
        normalized_entropy = entropy_bits / math.log2(len(action_retentions))
        metric = solution.metrics[weight_idx]
        for action_idx, retention in enumerate(action_retentions):
            count = int(counts[action_idx].item())
            rows.append(
                {
                    "environment": fsrs_config.environment,
                    "days": args.days,
                    "s_grid_size": args.s_grid_size,
                    "d_grid_size": args.d_grid_size,
                    "goal_cost_weight": cost_weight,
                    "action_index": action_idx,
                    "action_retention": retention,
                    "cell_count": count,
                    "cell_share": float(shares[action_idx].item()),
                    "total_cells": int(total),
                    "mean_action_retention": mean_retention,
                    "modal_action_index": modal_idx,
                    "modal_action_retention": action_retentions[modal_idx],
                    "modal_cell_share": modal_share,
                    "action_entropy_bits": entropy_bits,
                    "normalized_action_entropy": normalized_entropy,
                    "oracle_objective": metric.scalar_objective,
                    "oracle_card_expected_retrievability": (
                        metric.card_expected_retrievability
                    ),
                    "oracle_card_minutes_per_day": metric.card_minutes_per_day,
                    "policy_iterations": solution.iterations[weight_idx],
                    "policy_residual": solution.residuals[weight_idx],
                    "policy_converged": solution.converged[weight_idx],
                }
            )
    return rows


def _grid_rows(
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    oracle: FSRS6StationaryFiniteOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> list[dict[str, Any]]:
    cpu_policy = policies.to(device="cpu")
    s_values = oracle.s_grid.detach().to(device="cpu")
    d_values = oracle.d_grid.detach().to(device="cpu")
    rows: list[dict[str, Any]] = []
    for weight_idx, cost_weight in enumerate(cost_weights):
        for s_idx, stability in enumerate(s_values.tolist()):
            for d_idx, difficulty in enumerate(d_values.tolist()):
                action_idx = int(cpu_policy[weight_idx, s_idx, d_idx].item())
                rows.append(
                    {
                        "environment": fsrs_config.environment,
                        "days": args.days,
                        "s_grid_size": args.s_grid_size,
                        "d_grid_size": args.d_grid_size,
                        "goal_cost_weight": cost_weight,
                        "s_idx": s_idx,
                        "stability": stability,
                        "d_idx": d_idx,
                        "difficulty": difficulty,
                        "action_index": action_idx,
                        "action_retention": action_retentions[action_idx],
                    }
                )
    return rows


def _binned_rows(
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    oracle: FSRS6StationaryFiniteOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> list[dict[str, Any]]:
    action_count = len(action_retentions)
    s_spec = _build_bin_spec(oracle.s_grid, args.s_bins)
    d_spec = _build_bin_spec(oracle.d_grid, args.d_bins)
    s_code = s_spec.indices[:, None]
    d_code = d_spec.indices[None, :]
    bin_code = (s_code * d_spec.count + d_code).reshape(-1)
    bin_count = s_spec.count * d_spec.count
    rows: list[dict[str, Any]] = []
    for weight_idx, cost_weight in enumerate(cost_weights):
        flat = bin_code * action_count + policies[weight_idx].reshape(-1)
        counts = torch.bincount(
            flat,
            minlength=bin_count * action_count,
        ).reshape(s_spec.count, d_spec.count, action_count)
        bin_totals = counts.sum(dim=2)
        for s_bin in range(s_spec.count):
            for d_bin in range(d_spec.count):
                total = int(bin_totals[s_bin, d_bin].item())
                if total <= 0:
                    continue
                modal_idx = int(torch.argmax(counts[s_bin, d_bin]).item())
                modal_share = float(counts[s_bin, d_bin, modal_idx].item() / total)
                for action_idx, retention in enumerate(action_retentions):
                    count = int(counts[s_bin, d_bin, action_idx].item())
                    rows.append(
                        {
                            "environment": fsrs_config.environment,
                            "days": args.days,
                            "s_grid_size": args.s_grid_size,
                            "d_grid_size": args.d_grid_size,
                            "goal_cost_weight": cost_weight,
                            "stability_bin": s_bin,
                            "stability_min": s_spec.min_values[s_bin],
                            "stability_max": s_spec.max_values[s_bin],
                            "difficulty_bin": d_bin,
                            "difficulty_min": d_spec.min_values[d_bin],
                            "difficulty_max": d_spec.max_values[d_bin],
                            "action_index": action_idx,
                            "action_retention": retention,
                            "cell_count": count,
                            "bin_cell_share": count / total,
                            "bin_total_cells": total,
                            "modal_action_index": modal_idx,
                            "modal_action_retention": action_retentions[modal_idx],
                            "modal_cell_share": modal_share,
                        }
                    )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_matrix(
    *,
    rows: Sequence[dict[str, Any]],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> torch.Tensor:
    matrix = torch.zeros(
        (len(cost_weights), len(action_retentions)),
        dtype=torch.float64,
    )
    weight_to_idx = {float(value): idx for idx, value in enumerate(cost_weights)}
    action_to_idx = {float(value): idx for idx, value in enumerate(action_retentions)}
    for row in rows:
        weight_idx = weight_to_idx[float(row["goal_cost_weight"])]
        action_idx = action_to_idx[float(row["action_retention"])]
        matrix[weight_idx, action_idx] = float(row["cell_share"])
    return matrix


def _plot_distribution(
    path: Path,
    *,
    summary_rows: Sequence[dict[str, Any]],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    matrix = _summary_matrix(
        rows=summary_rows,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    width = max(8.0, 0.45 * len(action_retentions) + 3.0)
    height = max(4.5, 0.34 * len(cost_weights) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(action_retentions)))
    ax.set_xticklabels(
        [format_float(value) for value in action_retentions],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(cost_weights)))
    ax.set_yticklabels([format_float(value) for value in cost_weights])
    ax.set_xlabel("Desired-retention action")
    ax.set_ylabel("Goal cost weight")
    ax.set_title("Stationary finite oracle action distribution over (s, d) grid")
    if len(action_retentions) <= 18 and len(cost_weights) <= 20:
        for weight_idx in range(len(cost_weights)):
            for action_idx in range(len(action_retentions)):
                value = float(matrix[weight_idx, action_idx].item())
                if value <= 0.0:
                    continue
                color = "white" if value >= 0.45 else "black"
                ax.text(
                    action_idx,
                    weight_idx,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Grid-cell share")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grid_edges(values: torch.Tensor, *, log_space: bool) -> list[float]:
    cpu = values.detach().to(device="cpu", dtype=torch.float64)
    if int(cpu.numel()) == 1:
        value = float(cpu[0].item())
        delta = 0.5 * value if log_space else 0.5
        return [max(value - delta, 1e-12), value + delta]
    working = torch.log(cpu) if log_space else cpu
    mids = (working[:-1] + working[1:]) * 0.5
    first = working[0] - (mids[0] - working[0])
    last = working[-1] + (working[-1] - mids[-1])
    edges = torch.cat([first.reshape(1), mids, last.reshape(1)])
    if log_space:
        edges = torch.exp(edges)
    return [float(value) for value in edges.tolist()]


def _nearest_weight_indices(
    *,
    cost_weights: Sequence[float],
    selected_weights: Sequence[float],
) -> list[int]:
    indices: list[int] = []
    for requested in selected_weights:
        idx = min(
            range(len(cost_weights)),
            key=lambda candidate: abs(float(cost_weights[candidate]) - requested),
        )
        if idx not in indices:
            indices.append(idx)
    return indices


def _plot_policy_heatmaps(
    path: Path,
    *,
    oracle: FSRS6StationaryFiniteOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    selected_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    weight_indices = _nearest_weight_indices(
        cost_weights=cost_weights,
        selected_weights=selected_weights,
    )
    if not weight_indices:
        return

    action_tensor = torch.tensor(
        list(action_retentions),
        device=policies.device,
        dtype=torch.float64,
    )
    d_edges = _grid_edges(oracle.d_grid, log_space=False)
    s_edges = _grid_edges(oracle.s_grid, log_space=True)
    col_count = min(3, len(weight_indices))
    row_count = math.ceil(len(weight_indices) / col_count)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(4.8 * col_count, 3.8 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for panel_idx, weight_idx in enumerate(weight_indices):
        row_idx = panel_idx // col_count
        col_idx = panel_idx % col_count
        ax = axes[row_idx][col_idx]
        retention_grid = action_tensor[policies[weight_idx]].detach().to(device="cpu")
        mesh = ax.pcolormesh(
            d_edges,
            s_edges,
            retention_grid.numpy(),
            shading="auto",
            cmap="viridis",
            vmin=min(action_retentions),
            vmax=max(action_retentions),
        )
        ax.set_yscale("log")
        ax.set_title(f"w={format_float(cost_weights[weight_idx])}")
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Stability (days)")
    for panel_idx in range(len(weight_indices), row_count * col_count):
        axes[panel_idx // col_count][panel_idx % col_count].axis("off")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist())
        cbar.set_label("Desired-retention action")
    fig.suptitle("Stationary finite oracle policy over (stability, difficulty)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rows_by_weight(
    rows: Sequence[dict[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["goal_cost_weight"]), []).append(row)
    return grouped


def _dominant_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    dominant: list[dict[str, Any]] = []
    for _, rows in sorted(_rows_by_weight(summary_rows).items()):
        dominant.append(max(rows, key=lambda row: float(row["cell_share"])))
    return dominant


def _axis_bin_mean_retention(
    *,
    binned_rows: Sequence[dict[str, Any]],
    cost_weight: float,
    axis: str,
    bin_idx: int,
) -> float:
    key = "stability_bin" if axis == "stability" else "difficulty_bin"
    total = 0.0
    weighted = 0.0
    for row in binned_rows:
        if float(row["goal_cost_weight"]) != float(cost_weight):
            continue
        if int(row[key]) != bin_idx:
            continue
        count = float(row["cell_count"])
        total += count
        weighted += count * float(row["action_retention"])
    return weighted / total if total > 0.0 else 0.0


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    solution: StationaryFiniteOracleSolution,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    summary_rows: Sequence[dict[str, Any]],
    binned_rows: Sequence[dict[str, Any]],
    selected_weights: Sequence[float],
    runtime_s: float,
) -> None:
    dominant = _dominant_rows(summary_rows)
    first_weight = float(cost_weights[0])
    last_weight = float(cost_weights[-1])
    first_mean = float(dominant[0]["mean_action_retention"])
    last_mean = float(dominant[-1]["mean_action_retention"])
    weight_indices = _nearest_weight_indices(
        cost_weights=cost_weights,
        selected_weights=selected_weights,
    )
    max_stability_bin = max(
        (int(row["stability_bin"]) for row in binned_rows),
        default=0,
    )
    max_difficulty_bin = max(
        (int(row["difficulty_bin"]) for row in binned_rows),
        default=0,
    )
    lines = [
        "# Stationary Finite Oracle Policy Action Analysis",
        "",
        f"- Environment: `{fsrs_config.environment}`",
        f"- Days: `{args.days}`",
        f"- Grid: `{args.s_grid_size} x {args.d_grid_size}`",
        f"- Action count: `{len(action_retentions)}`",
        f"- Cost weights: `{','.join(format_float(value) for value in cost_weights)}`",
        f"- Solve runtime: `{runtime_s:.2f}s`",
        f"- Policy iterations: `{list(solution.iterations)}`",
        "",
        "## Weight-Level Pattern",
        "",
        (
            "Mean desired-retention action over the equal-weighted `(s, d)` grid "
            f"moves from `{first_mean:.4f}` at `w={format_float(first_weight)}` "
            f"to `{last_mean:.4f}` at `w={format_float(last_weight)}`."
        ),
        "",
        "| weight | modal retention | modal share | mean retention | entropy | objective |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dominant:
        lines.append(
            " | ".join(
                [
                    f"| {format_float(float(row['goal_cost_weight']))}",
                    format_float(float(row["modal_action_retention"])),
                    f"{float(row['modal_cell_share']):.1%}",
                    f"{float(row['mean_action_retention']):.4f}",
                    f"{float(row['normalized_action_entropy']):.3f}",
                    f"{float(row['oracle_objective']):.6f} |",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Axis Extremes",
            "",
            (
                "Each row compares the mean selected retention in the lowest and "
                "highest stability/difficulty bins. These are table-weighted policy "
                "summaries, not rollout occupancies."
            ),
            "",
            "| weight | low stability | high stability | low difficulty | high difficulty |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for weight_idx in weight_indices:
        cost_weight = float(cost_weights[weight_idx])
        low_s = _axis_bin_mean_retention(
            binned_rows=binned_rows,
            cost_weight=cost_weight,
            axis="stability",
            bin_idx=0,
        )
        high_s = _axis_bin_mean_retention(
            binned_rows=binned_rows,
            cost_weight=cost_weight,
            axis="stability",
            bin_idx=max_stability_bin,
        )
        low_d = _axis_bin_mean_retention(
            binned_rows=binned_rows,
            cost_weight=cost_weight,
            axis="difficulty",
            bin_idx=0,
        )
        high_d = _axis_bin_mean_retention(
            binned_rows=binned_rows,
            cost_weight=cost_weight,
            axis="difficulty",
            bin_idx=max_difficulty_bin,
        )
        lines.append(
            " | ".join(
                [
                    f"| {format_float(cost_weight)}",
                    f"{low_s:.4f}",
                    f"{high_s:.4f}",
                    f"{low_d:.4f}",
                    f"{high_d:.4f} |",
                ]
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary_rows: Sequence[dict[str, Any]]) -> None:
    for row in _dominant_rows(summary_rows):
        print(
            " ".join(
                [
                    f"w={format_float(float(row['goal_cost_weight']))}",
                    f"modal_retention={format_float(float(row['modal_action_retention']))}",
                    f"modal_share={float(row['modal_cell_share']):.3f}",
                    f"mean_retention={float(row['mean_action_retention']):.4f}",
                    f"entropy={float(row['normalized_action_entropy']):.3f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.s_grid_size < 8 or args.d_grid_size < 8:
        raise SystemExit("--grid sizes must be >= 8.")
    if args.s_bins <= 0 or args.d_bins <= 0:
        raise SystemExit("--s-bins and --d-bins must be > 0.")
    if args.oracle_stationary_finite_max_iterations <= 0:
        raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
    if args.oracle_stationary_finite_tolerance <= 0.0:
        raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    selected_weights = parse_csv_floats(
        args.selected_weights,
        name="--selected-weights",
    )
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    if any(value <= 0.0 or value >= 1.0 for value in action_retentions):
        raise SystemExit("--action-retentions values must be within (0, 1).")

    device = _resolve_device(args.torch_device)
    fsrs_config = load_single_card_fsrs6_config(args)
    oracle = FSRS6StationaryFiniteOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
        device=device,
        **fsrs_config_kwargs(fsrs_config),
    )
    start = time.perf_counter()
    solution = oracle.solve_stationary_finite_policies(
        cost_weights,
        max_iterations=args.oracle_stationary_finite_max_iterations,
        tolerance=args.oracle_stationary_finite_tolerance,
        progress=not args.no_progress,
    )
    runtime_s = time.perf_counter() - start
    if not all(solution.converged):
        failed = [
            format_float(weight)
            for weight, converged in zip(cost_weights, solution.converged, strict=True)
            if not converged
        ]
        raise RuntimeError(
            "Stationary finite oracle did not converge for cost weights: "
            + ",".join(failed)
        )

    paths = _output_paths(args.out_dir)
    summary_rows = _summary_rows(
        args=args,
        fsrs_config=fsrs_config,
        solution=solution,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    binned_rows = _binned_rows(
        args=args,
        fsrs_config=fsrs_config,
        oracle=oracle,
        policies=solution.policy,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    _write_csv(paths.summary, summary_rows)
    _write_csv(paths.binned, binned_rows)
    if not args.no_grid_csv:
        _write_csv(
            paths.grid,
            _grid_rows(
                args=args,
                fsrs_config=fsrs_config,
                oracle=oracle,
                policies=solution.policy,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
            ),
        )
    _write_report(
        paths.report,
        args=args,
        fsrs_config=fsrs_config,
        solution=solution,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        summary_rows=summary_rows,
        binned_rows=binned_rows,
        selected_weights=selected_weights,
        runtime_s=runtime_s,
    )
    if not args.no_plot:
        _plot_distribution(
            paths.distribution_plot,
            summary_rows=summary_rows,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
        )
        _plot_policy_heatmaps(
            paths.heatmap_plot,
            oracle=oracle,
            policies=solution.policy,
            cost_weights=cost_weights,
            selected_weights=selected_weights,
            action_retentions=action_retentions,
        )
        print(f"Wrote plot: {paths.distribution_plot}")
        print(f"Wrote plot: {paths.heatmap_plot}")
    print(f"Wrote CSV: {paths.summary}")
    print(f"Wrote CSV: {paths.binned}")
    if not args.no_grid_csv:
        print(f"Wrote CSV: {paths.grid}")
    print(f"Wrote report: {paths.report}")
    _print_summary(summary_rows)


if __name__ == "__main__":
    main()
