from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections.abc import Sequence
import csv
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
        "--plot-path",
        type=Path,
        default=None,
        help="Plot output path. Defaults to the CSV path with .png suffix.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


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
) -> torch.Tensor:
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

            env.step(action)
            if progress_bar is not None:
                next_completed = int(env.done.sum().item())
                progress_bar.update(next_completed - completed)
                completed = next_completed
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return counts


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


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.particles <= 0:
        raise SystemExit("--particles must be > 0.")
    if args.s_grid_size < 8 or args.d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    if any(value <= 0.0 or value >= 1.0 for value in action_retentions):
        raise SystemExit("--action-retentions values must be within (0, 1).")

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

    if args.source == "table":
        counts = _count_table_outputs(
            policies=policies,
            action_count=len(action_retentions),
        )
    else:
        counts = _count_rollout_outputs(
            args=args,
            device=device,
            oracle=oracle,
            policies=policies,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            fsrs_config=fsrs_config,
        )

    rows = _rows_from_counts(
        args=args,
        fsrs_config=fsrs_config,
        source=args.source,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        counts=counts,
    )
    write_csv(args.out, rows)
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
    _print_summary(rows)


if __name__ == "__main__":
    main()
