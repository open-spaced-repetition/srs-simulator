from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.oracle_frontier import (  # noqa: E402
    DEFAULT_COST_WEIGHTS,
    FSRS6GridOracle,
    parse_csv_floats,
)
from experiments.single_card_tradeoff.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.tradeoff import DEFAULT_TARGET_RETENTIONS  # noqa: E402
from experiments.single_card_tradeoff.uvfa_ppo import FSRS6SingleCardBatch  # noqa: E402
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FSRS6 oracle policy output distributions.",
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--particles",
        type=int,
        default=10_000,
        help="Monte Carlo particles per cost weight for --source rollout.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
        help="Comma-separated scalarization weights to solve and plot.",
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
        "--source",
        choices=["rollout", "table"],
        default="rollout",
        help=(
            "rollout counts actions on Monte Carlo states visited by the solved "
            "policy; table counts every nonterminal grid cell equally."
        ),
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for oracle solving and rollout, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/fsrs6_oracle_policy_outputs.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=None,
        help=(
            "Detailed binned CSV output path. Defaults to the summary CSV stem "
            "with _detail appended."
        ),
    )
    parser.add_argument(
        "--remaining-bins",
        type=int,
        default=8,
        help="Number of remaining-horizon bins for the detailed CSV.",
    )
    parser.add_argument(
        "--s-bins",
        type=int,
        default=8,
        help="Number of stability bins for the detailed CSV.",
    )
    parser.add_argument(
        "--d-bins",
        type=int,
        default=8,
        help="Number of difficulty bins for the detailed CSV.",
    )
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Plot output path. Defaults to the CSV path with .png suffix.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class BinSpec:
    indices: torch.Tensor
    min_values: tuple[float, ...]
    max_values: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.min_values)


@dataclass(frozen=True)
class DetailBinSpecs:
    remaining: BinSpec
    stability: BinSpec
    difficulty: BinSpec


@dataclass(frozen=True)
class RolloutCounts:
    summary: torch.Tensor
    detail: torch.Tensor | None


def _default_detail_path(path: Path) -> Path:
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}_detail{suffix}")


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.torch_device:
        return torch.device(args.torch_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fsrs_config_kwargs(config: SingleCardFSRS6Config) -> dict[str, Any]:
    return {
        "fsrs_weights": config.fsrs_weights,
        "first_rating_prob": config.first_rating_prob,
        "review_rating_prob": config.review_rating_prob,
        "learning_costs": config.learning_costs,
        "review_costs": config.review_costs,
    }


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


def _build_table_detail_specs(
    *,
    args: argparse.Namespace,
    oracle: FSRS6GridOracle,
) -> DetailBinSpecs:
    remaining_values = torch.arange(
        1,
        oracle.horizon + 1,
        device=oracle.device,
        dtype=oracle.dtype,
    )
    return DetailBinSpecs(
        remaining=_build_bin_spec(remaining_values, args.remaining_bins),
        stability=_build_bin_spec(oracle.s_grid, args.s_bins),
        difficulty=_build_bin_spec(oracle.d_grid, args.d_bins),
    )


def _build_rollout_detail_specs(
    *,
    args: argparse.Namespace,
    oracle: FSRS6GridOracle,
) -> DetailBinSpecs:
    remaining_values = torch.arange(
        0,
        oracle.horizon + 1,
        device=oracle.device,
        dtype=oracle.dtype,
    )
    return DetailBinSpecs(
        remaining=_build_bin_spec(remaining_values, args.remaining_bins),
        stability=_build_bin_spec(oracle.s_grid, args.s_bins),
        difficulty=_build_bin_spec(oracle.d_grid, args.d_bins),
    )


def _rows_from_counts(
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    source: str,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    counts: torch.Tensor,
) -> list[dict[str, Any]]:
    counts = counts.to(device="cpu")
    rows: list[dict[str, Any]] = []
    for weight_idx, cost_weight in enumerate(cost_weights):
        total = int(counts[weight_idx].sum().item())
        for action_idx, retention in enumerate(action_retentions):
            count = int(counts[weight_idx, action_idx].item())
            rows.append(
                {
                    "environment": fsrs_config.environment,
                    "source": source,
                    "days": args.days,
                    "particles": args.particles if source == "rollout" else "",
                    "seed": args.seed if source == "rollout" else "",
                    "s_grid_size": args.s_grid_size,
                    "d_grid_size": args.d_grid_size,
                    "goal_cost_weight": cost_weight,
                    "action_index": action_idx,
                    "action_retention": retention,
                    "decision_count": count,
                    "decision_share": count / total if total > 0 else 0.0,
                    "total_decisions": total,
                }
            )
    return rows


def _detail_rows_from_counts(
    *,
    args: argparse.Namespace,
    fsrs_config: SingleCardFSRS6Config,
    source: str,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    counts: torch.Tensor,
    specs: DetailBinSpecs,
) -> list[dict[str, Any]]:
    counts = counts.to(device="cpu")
    rows: list[dict[str, Any]] = []
    weight_totals = counts.sum(dim=(1, 2, 3, 4))
    bin_totals = counts.sum(dim=4)
    for weight_idx, cost_weight in enumerate(cost_weights):
        weight_total = int(weight_totals[weight_idx].item())
        for remaining_bin in range(specs.remaining.count):
            for stability_bin in range(specs.stability.count):
                for difficulty_bin in range(specs.difficulty.count):
                    bin_total = int(
                        bin_totals[
                            weight_idx,
                            remaining_bin,
                            stability_bin,
                            difficulty_bin,
                        ].item()
                    )
                    if bin_total <= 0:
                        continue
                    for action_idx, retention in enumerate(action_retentions):
                        count = int(
                            counts[
                                weight_idx,
                                remaining_bin,
                                stability_bin,
                                difficulty_bin,
                                action_idx,
                            ].item()
                        )
                        rows.append(
                            {
                                "environment": fsrs_config.environment,
                                "source": source,
                                "days": args.days,
                                "particles": (
                                    args.particles if source == "rollout" else ""
                                ),
                                "seed": args.seed if source == "rollout" else "",
                                "s_grid_size": args.s_grid_size,
                                "d_grid_size": args.d_grid_size,
                                "goal_cost_weight": cost_weight,
                                "remaining_bin": remaining_bin,
                                "remaining_days_min": specs.remaining.min_values[
                                    remaining_bin
                                ],
                                "remaining_days_max": specs.remaining.max_values[
                                    remaining_bin
                                ],
                                "stability_bin": stability_bin,
                                "stability_min": specs.stability.min_values[
                                    stability_bin
                                ],
                                "stability_max": specs.stability.max_values[
                                    stability_bin
                                ],
                                "difficulty_bin": difficulty_bin,
                                "difficulty_min": specs.difficulty.min_values[
                                    difficulty_bin
                                ],
                                "difficulty_max": specs.difficulty.max_values[
                                    difficulty_bin
                                ],
                                "action_index": action_idx,
                                "action_retention": retention,
                                "decision_count": count,
                                "bin_decision_share": count / bin_total,
                                "weight_decision_share": (
                                    count / weight_total if weight_total > 0 else 0.0
                                ),
                                "bin_total_decisions": bin_total,
                                "weight_total_decisions": weight_total,
                            }
                        )
    return rows


def _count_table_outputs(
    *,
    policies: torch.Tensor,
    action_count: int,
) -> torch.Tensor:
    nonterminal = policies[:, 1:, :, :]
    counts = torch.zeros(
        (int(policies.shape[0]), action_count),
        device=policies.device,
        dtype=torch.int64,
    )
    for weight_idx in range(int(policies.shape[0])):
        counts[weight_idx] = torch.bincount(
            nonterminal[weight_idx].reshape(-1),
            minlength=action_count,
        )[:action_count]
    return counts


def _count_table_detail_outputs(
    *,
    policies: torch.Tensor,
    action_count: int,
    specs: DetailBinSpecs,
) -> torch.Tensor:
    nonterminal = policies[:, 1:, :, :]
    weight_count = int(nonterminal.shape[0])
    counts = torch.zeros(
        (
            weight_count,
            specs.remaining.count,
            specs.stability.count,
            specs.difficulty.count,
            action_count,
        ),
        device=policies.device,
        dtype=torch.int64,
    )
    remaining_code = specs.remaining.indices[:, None, None]
    stability_code = specs.stability.indices[None, :, None]
    difficulty_code = specs.difficulty.indices[None, None, :]
    bin_code = (
        (remaining_code * specs.stability.count + stability_code)
        * specs.difficulty.count
        + difficulty_code
    ).reshape(-1)
    bin_count = specs.remaining.count * specs.stability.count * specs.difficulty.count
    for weight_idx in range(weight_count):
        flat = bin_code * action_count + nonterminal[weight_idx].reshape(-1)
        counts[weight_idx] = torch.bincount(
            flat,
            minlength=bin_count * action_count,
        ).reshape(
            specs.remaining.count,
            specs.stability.count,
            specs.difficulty.count,
            action_count,
        )
    return counts


@torch.inference_mode()
def _count_rollout_outputs(
    *,
    args: argparse.Namespace,
    device: torch.device,
    oracle: FSRS6GridOracle,
    policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config,
    detail_specs: DetailBinSpecs | None,
) -> RolloutCounts:
    from tqdm import tqdm

    weight_count = len(cost_weights)
    action_count = len(action_retentions)
    env_count = args.particles * weight_count

    env = FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float64,
        seed=args.seed,
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
    counts = torch.zeros(
        (weight_count, action_count),
        device=device,
        dtype=torch.int64,
    )
    detail_counts = (
        torch.zeros(
            (
                weight_count,
                detail_specs.remaining.count,
                detail_specs.stability.count,
                detail_specs.difficulty.count,
                action_count,
            ),
            device=device,
            dtype=torch.int64,
        )
        if detail_specs is not None
        else None
    )
    progress_bar = None
    completed = 0
    if not args.no_progress:
        progress_bar = tqdm(
            total=env_count,
            desc="Oracle policy rollout",
            unit="card",
            leave=False,
        )
    try:
        while not bool(env.done.all().item()):
            active = ~env.done
            remaining = torch.clamp((env.days - 1) - env.day, min=0, max=oracle.horizon)
            s_idx = oracle._s_to_idx(env.s)
            d_idx = oracle._d_to_idx(env.d)
            action = policies[goal_indices, remaining, s_idx, d_idx]

            flat = goal_indices[active] * action_count + action[active]
            counts += torch.bincount(
                flat,
                minlength=weight_count * action_count,
            ).reshape(weight_count, action_count)

            if detail_specs is not None and detail_counts is not None:
                active_remaining = remaining[active].to(torch.int64)
                remaining_bin = detail_specs.remaining.indices.index_select(
                    0,
                    active_remaining,
                )
                stability_bin = detail_specs.stability.indices.index_select(
                    0,
                    s_idx[active],
                )
                difficulty_bin = detail_specs.difficulty.indices.index_select(
                    0,
                    d_idx[active],
                )
                detail_flat = (
                    (
                        (
                            goal_indices[active] * detail_specs.remaining.count
                            + remaining_bin
                        )
                        * detail_specs.stability.count
                        + stability_bin
                    )
                    * detail_specs.difficulty.count
                    + difficulty_bin
                ) * action_count + action[active]
                detail_counts += torch.bincount(
                    detail_flat,
                    minlength=(
                        weight_count
                        * detail_specs.remaining.count
                        * detail_specs.stability.count
                        * detail_specs.difficulty.count
                        * action_count
                    ),
                ).reshape(
                    weight_count,
                    detail_specs.remaining.count,
                    detail_specs.stability.count,
                    detail_specs.difficulty.count,
                    action_count,
                )

            env.step(action)
            if progress_bar is not None:
                next_completed = int(env.done.sum().item())
                progress_bar.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return RolloutCounts(summary=counts, detail=detail_counts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "source",
        "days",
        "particles",
        "seed",
        "s_grid_size",
        "d_grid_size",
        "goal_cost_weight",
        "action_index",
        "action_retention",
        "decision_count",
        "decision_share",
        "total_decisions",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "source",
        "days",
        "particles",
        "seed",
        "s_grid_size",
        "d_grid_size",
        "goal_cost_weight",
        "remaining_bin",
        "remaining_days_min",
        "remaining_days_max",
        "stability_bin",
        "stability_min",
        "stability_max",
        "difficulty_bin",
        "difficulty_min",
        "difficulty_max",
        "action_index",
        "action_retention",
        "decision_count",
        "bin_decision_share",
        "weight_decision_share",
        "bin_total_decisions",
        "weight_total_decisions",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _distribution_matrix(
    *,
    rows: Sequence[dict[str, Any]],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> torch.Tensor:
    matrix = torch.zeros(
        (len(cost_weights), len(action_retentions)), dtype=torch.float64
    )
    weight_to_idx = {float(value): idx for idx, value in enumerate(cost_weights)}
    action_to_idx = {float(value): idx for idx, value in enumerate(action_retentions)}
    for row in rows:
        weight_idx = weight_to_idx[float(row["goal_cost_weight"])]
        action_idx = action_to_idx[float(row["action_retention"])]
        matrix[weight_idx, action_idx] = float(row["decision_share"])
    return matrix


def write_plot(
    path: Path,
    *,
    rows: Sequence[dict[str, Any]],
    source: str,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    matrix = _distribution_matrix(
        rows=rows,
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
    ax.set_xlabel("Oracle output desired retention")
    ax.set_ylabel("Goal cost weight")
    ax.set_title(f"FSRS6 oracle policy output distribution ({source})")

    if len(action_retentions) <= 16 and len(cost_weights) <= 18:
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
    cbar.set_label("Decision share")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_summary(rows: Sequence[dict[str, Any]]) -> None:
    by_weight: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_weight.setdefault(float(row["goal_cost_weight"]), []).append(row)
    for cost_weight in sorted(by_weight):
        best = max(by_weight[cost_weight], key=lambda row: float(row["decision_share"]))
        print(
            " ".join(
                [
                    f"w={format_float(cost_weight)}",
                    f"top_retention={format_float(float(best['action_retention']))}",
                    f"share={float(best['decision_share']):.3f}",
                    f"decisions={int(best['total_decisions'])}",
                ]
            )
        )


def _print_detail_summary(counts: torch.Tensor) -> None:
    counts = counts.to(device="cpu")
    bin_totals = counts.sum(dim=4)
    nonempty = bin_totals > 0
    if not bool(nonempty.any().item()):
        print("detail_bins=0")
        return
    modal = counts.max(dim=4).values
    modal_share = modal[nonempty].to(dtype=torch.float64) / bin_totals[nonempty].to(
        dtype=torch.float64
    )
    print(
        " ".join(
            [
                f"detail_bins={int(nonempty.sum().item())}",
                f"median_modal_share={float(torch.median(modal_share).item()):.3f}",
                f"p10_modal_share={float(torch.quantile(modal_share, 0.10).item()):.3f}",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.particles <= 0:
        raise SystemExit("--particles must be > 0.")
    if args.s_grid_size < 8 or args.d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.remaining_bins <= 0 or args.s_bins <= 0 or args.d_bins <= 0:
        raise SystemExit("--remaining-bins, --s-bins, and --d-bins must be > 0.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")

    device = _resolve_device(args)
    fsrs_config = load_single_card_fsrs6_config(args)
    oracle = FSRS6GridOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
        device=device,
        **_fsrs_config_kwargs(fsrs_config),
    )
    policies = oracle.solve_policies(cost_weights, progress=not args.no_progress)
    detail_specs = None
    if not args.no_detail:
        detail_specs = (
            _build_table_detail_specs(args=args, oracle=oracle)
            if args.source == "table"
            else _build_rollout_detail_specs(args=args, oracle=oracle)
        )

    if args.source == "table":
        counts = _count_table_outputs(
            policies=policies,
            action_count=len(action_retentions),
        )
        detail_counts = (
            _count_table_detail_outputs(
                policies=policies,
                action_count=len(action_retentions),
                specs=detail_specs,
            )
            if detail_specs is not None
            else None
        )
    else:
        rollout_counts = _count_rollout_outputs(
            args=args,
            device=device,
            oracle=oracle,
            policies=policies,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            fsrs_config=fsrs_config,
            detail_specs=detail_specs,
        )
        counts = rollout_counts.summary
        detail_counts = rollout_counts.detail

    rows = _rows_from_counts(
        args=args,
        fsrs_config=fsrs_config,
        source=args.source,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        counts=counts,
    )
    write_csv(args.out, rows)
    detail_out: Path | None = None
    if detail_specs is not None and detail_counts is not None:
        resolved_detail_out = _default_detail_path(args.out)
        if args.detail_out is not None:
            resolved_detail_out = Path(args.detail_out)
        detail_out = resolved_detail_out
        detail_rows = _detail_rows_from_counts(
            args=args,
            fsrs_config=fsrs_config,
            source=args.source,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            counts=detail_counts,
            specs=detail_specs,
        )
        write_detail_csv(resolved_detail_out, detail_rows)
    if not args.no_plot:
        plot_path = args.plot_path or args.out.with_suffix(".png")
        write_plot(
            plot_path,
            rows=rows,
            source=args.source,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
        )
        print(f"Wrote plot: {plot_path}")
    print(f"Wrote CSV: {args.out}")
    if (
        detail_specs is not None
        and detail_counts is not None
        and detail_out is not None
    ):
        print(f"Wrote detail CSV: {detail_out}")
        _print_detail_summary(detail_counts)
    _print_summary(rows)


if __name__ == "__main__":
    main()
