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
    FSRS6ContinuousRetentionOracle,
    FSRS6ContinuousUniformTerminationOracle,
)
from experiments.single_card_tradeoff.oracles.dp_cache import (
    oracle_dp_cache_stats_snapshot,
    reset_oracle_dp_cache_stats,
)
from simulator.defaults import DEFAULT_DAYS
from simulator.scheduler_spec import format_float

DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/continuous_uniform_h_policy_analysis"
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
            "Compare fixed-H and hidden Uniform-H continuous-retention oracle "
            "policy surfaces."
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
) -> tuple[FSRS6ContinuousRetentionOracle, FSRS6ContinuousUniformTerminationOracle]:
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
        FSRS6ContinuousRetentionOracle(**kwargs),
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


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("oracle grid sizes must be >= 8.")
    if args.oracle_continuous_interval_chunk_size <= 0:
        raise SystemExit("--oracle-continuous-interval-chunk-size must be > 0.")

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
    if not landmarks:
        raise SystemExit("--landmark-remaining-days has no values within horizon.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=out_dir,
        stage_name="continuous_uniform_h_policy_analysis",
    )
    reset_oracle_dp_cache_stats()
    start_time = time.perf_counter()

    weight_count = len(cost_weights)
    rem_count = horizon + 1
    uniform_mean = torch.zeros((weight_count, rem_count), dtype=torch.float64)
    uniform_min_share = torch.zeros_like(uniform_mean)
    uniform_max_share = torch.zeros_like(uniform_mean)
    uniform_self_delta = torch.zeros_like(uniform_mean)
    compare_mean_delta = torch.zeros_like(uniform_mean)
    compare_abs_delta = torch.zeros_like(uniform_mean)
    compare_share_delta = torch.zeros_like(uniform_mean)
    start_to_fixed_abs_delta = torch.zeros_like(uniform_mean)
    stability_surface: torch.Tensor | None = None
    difficulty_surface: torch.Tensor | None = None

    for index, user_id in enumerate(user_ids, start=1):
        print(f"[{index}/{len(user_ids)}] solving user {user_id} on {device}")
        fixed_oracle, uniform_oracle = build_oracles(
            args,
            device=device,
            user_id=user_id,
        )
        fixed_policy = fixed_oracle.solve_policies(
            cost_weights,
            progress=not args.no_progress,
        )
        uniform_policy = uniform_oracle.solve_policies(
            cost_weights,
            progress=not args.no_progress,
        )
        min_value = float(args.oracle_continuous_retention_min)
        max_value = float(args.oracle_continuous_retention_max)
        uniform_active = uniform_policy[:, 1:]
        fixed_active = fixed_policy[:, 1:]
        uniform_start = uniform_policy[:, horizon]

        uniform_mean[:, 1:] += uniform_active.mean(dim=(2, 3)).detach().cpu()
        uniform_min_share[:, 1:] += (
            (uniform_active <= (min_value + 1e-7))
            .to(dtype=torch.float64)
            .mean(dim=(2, 3))
            .detach()
            .cpu()
        )
        uniform_max_share[:, 1:] += (
            (uniform_active >= (max_value - 1e-7))
            .to(dtype=torch.float64)
            .mean(dim=(2, 3))
            .detach()
            .cpu()
        )
        uniform_self_delta[:, 1:] += (
            torch.abs(uniform_active - uniform_start[:, None])
            .mean(dim=(2, 3))
            .detach()
            .cpu()
        )

        delta = uniform_active - fixed_active
        compare_mean_delta[:, 1:] += delta.mean(dim=(2, 3)).detach().cpu()
        compare_abs_delta[:, 1:] += torch.abs(delta).mean(dim=(2, 3)).detach().cpu()
        compare_share_delta[:, 1:] += (
            (torch.abs(delta) > 0.02)
            .to(dtype=torch.float64)
            .mean(dim=(2, 3))
            .detach()
            .cpu()
        )
        start_to_fixed_abs_delta[:, 1:] += (
            torch.abs(fixed_active - uniform_start[:, None])
            .mean(dim=(2, 3))
            .detach()
            .cpu()
        )

        stability_user = uniform_active.mean(dim=3).detach().cpu()
        difficulty_user = uniform_active.mean(dim=2).detach().cpu()
        if stability_surface is None or difficulty_surface is None:
            stability_surface = torch.zeros_like(stability_user, dtype=torch.float64)
            difficulty_surface = torch.zeros_like(difficulty_user, dtype=torch.float64)
        assert stability_surface is not None and difficulty_surface is not None
        stability_surface += stability_user.to(dtype=torch.float64)
        difficulty_surface += difficulty_user.to(dtype=torch.float64)

        del fixed_policy, uniform_policy, fixed_active, uniform_active, delta
        if device.type == "cuda":
            torch.cuda.empty_cache()

    divisor = float(len(user_ids))
    for tensor in (
        uniform_mean,
        uniform_min_share,
        uniform_max_share,
        uniform_self_delta,
        compare_mean_delta,
        compare_abs_delta,
        compare_share_delta,
        start_to_fixed_abs_delta,
    ):
        tensor /= divisor
    assert stability_surface is not None and difficulty_surface is not None
    stability_surface /= divisor
    difficulty_surface /= divisor

    summary_rows = []
    compare_rows = []
    for weight_idx, weight in enumerate(cost_weights):
        for rem in range(1, horizon + 1):
            summary_rows.append(
                {
                    "cost_weight": weight,
                    "remaining_days": rem,
                    "mean_retention": float(uniform_mean[weight_idx, rem].item()),
                    "retention_min_share": float(
                        uniform_min_share[weight_idx, rem].item()
                    ),
                    "retention_max_share": float(
                        uniform_max_share[weight_idx, rem].item()
                    ),
                    "mean_abs_delta_to_uniform_1824d": float(
                        uniform_self_delta[weight_idx, rem].item()
                    ),
                }
            )
            compare_rows.append(
                {
                    "cost_weight": weight,
                    "remaining_days": rem,
                    "mean_delta_uniform_minus_fixed": float(
                        compare_mean_delta[weight_idx, rem].item()
                    ),
                    "mean_abs_delta_uniform_vs_fixed": float(
                        compare_abs_delta[weight_idx, rem].item()
                    ),
                    "share_abs_delta_gt_0_02": float(
                        compare_share_delta[weight_idx, rem].item()
                    ),
                }
            )

    landmark_rows = [
        row for row in summary_rows if int(row["remaining_days"]) in set(landmarks)
    ]
    start_rows = []
    for weight_idx, weight in enumerate(cost_weights):
        closest_delta, closest_idx = start_to_fixed_abs_delta[weight_idx, 1:].min(dim=0)
        row: dict[str, Any] = {
            "cost_weight": weight,
            "closest_fixed_remaining_days": int(closest_idx.item()) + 1,
            "closest_mean_abs_delta": float(closest_delta.item()),
        }
        for rem in landmarks:
            row[f"fixed_{rem}d_mean_abs_delta"] = float(
                start_to_fixed_abs_delta[weight_idx, rem].item()
            )
        start_rows.append(row)

    write_csv(out_dir / "uniform_remaining_summary.csv", summary_rows)
    write_csv(out_dir / "fixed_vs_uniform_remaining_comparison.csv", compare_rows)
    write_csv(out_dir / "remaining_landmarks.csv", landmark_rows)
    write_csv(out_dir / "uniform_start_vs_fixed_landmarks.csv", start_rows)

    metadata = {
        "users": user_ids,
        "days": int(args.days),
        "horizon": horizon,
        "cost_weights": cost_weights,
        "s_grid_size": int(args.oracle_s_grid_size),
        "d_grid_size": int(args.oracle_d_grid_size),
        "retention_min": float(args.oracle_continuous_retention_min),
        "retention_max": float(args.oracle_continuous_retention_max),
        "termination_distribution": (
            FSRS6ContinuousUniformTerminationOracle.TERMINATION_DISTRIBUTION_VERSION
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
            uniform_mean=uniform_mean,
            uniform_min_share=uniform_min_share,
            compare_abs_delta=compare_abs_delta,
            start_to_fixed_abs_delta=start_to_fixed_abs_delta,
            stability_surface=stability_surface,
            difficulty_surface=difficulty_surface,
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
    uniform_mean: torch.Tensor,
    uniform_min_share: torch.Tensor,
    compare_abs_delta: torch.Tensor,
    start_to_fixed_abs_delta: torch.Tensor,
    stability_surface: torch.Tensor,
    difficulty_surface: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    remaining = list(range(1, uniform_mean.shape[1]))
    indices = selected_weight_indices(cost_weights, selected_weights)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for idx in indices:
        label = f"w={format_float(cost_weights[idx])}"
        axes[0].plot(remaining, uniform_mean[idx, 1:].numpy(), label=label)
        axes[1].plot(remaining, uniform_min_share[idx, 1:].numpy(), label=label)
    axes[0].set_ylabel("Mean retention")
    axes[1].set_ylabel("Retention-min share")
    axes[1].set_xlabel("Remaining days")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "uniform_mean_retention_and_min_share_by_remaining.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in indices:
        ax.plot(
            remaining,
            compare_abs_delta[idx, 1:].numpy(),
            label=f"w={format_float(cost_weights[idx])}",
        )
    ax.set_xlabel("Remaining days")
    ax.set_ylabel("Mean abs retention delta")
    ax.set_title("Uniform-H vs fixed-H policy surface")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fixed_vs_uniform_delta_by_remaining.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in indices:
        ax.plot(
            remaining,
            start_to_fixed_abs_delta[idx, 1:].numpy(),
            label=f"w={format_float(cost_weights[idx])}",
        )
    ax.set_xlabel("Fixed-H remaining days")
    ax.set_ylabel("Mean abs delta to Uniform-H start")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "uniform_start_closest_fixed_remaining.png")
    plt.close(fig)

    write_heatmap_grid(
        out_dir / "stability_remaining_heatmaps.png",
        title="Uniform-H retention by stability and remaining time",
        cost_weights=cost_weights,
        indices=indices,
        surface=stability_surface,
        x_label="Stability grid index",
    )
    write_heatmap_grid(
        out_dir / "difficulty_remaining_heatmaps.png",
        title="Uniform-H retention by difficulty and remaining time",
        cost_weights=cost_weights,
        indices=indices,
        surface=difficulty_surface,
        x_label="Difficulty grid index",
    )


def write_heatmap_grid(
    path: Path,
    *,
    title: str,
    cost_weights: list[float],
    indices: list[int],
    surface: torch.Tensor,
    x_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    image: Any | None = None
    for ax, idx in zip(axes, indices, strict=True):
        image = ax.imshow(
            surface[idx].numpy(),
            aspect="auto",
            origin="lower",
            vmin=0.5,
            vmax=0.98,
        )
        ax.set_title(f"w={format_float(cost_weights[idx])}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Remaining days")
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    *,
    out_dir: Path,
    metadata: dict[str, Any],
    landmarks: list[int],
) -> None:
    lines = [
        "# Continuous Uniform-H Policy Analysis Artifacts",
        "",
        "Source policies: fixed-H `fsrs6_oracle_continuous_retention` and hidden "
        "Uniform-H continuous-retention oracle.",
        "",
        f"Users: {', '.join(str(user_id) for user_id in metadata['users'])}.",
        f"Lifecycle upper bound: {metadata['days']} days.",
        (
            f"Grid: {metadata['s_grid_size']} log-spaced stability values x "
            f"{metadata['d_grid_size']} difficulty values."
        ),
        f"Cost weights: {', '.join(format_float(value) for value in metadata['cost_weights'])}.",
        (
            "Retention-min share is a retention-surface proxy for terminal/no-review "
            "and ordinary low-retention actions, not an exact skip rate."
        ),
        "",
        "Files:",
        "",
        "- `uniform_remaining_summary.csv`: Uniform-H retention metrics by cost "
        "weight and remaining day.",
        "- `fixed_vs_uniform_remaining_comparison.csv`: fixed-H versus Uniform-H "
        "surface deltas.",
        "- `uniform_start_vs_fixed_landmarks.csv`: fixed-H remaining slices closest "
        "to the Uniform-H start policy.",
        f"- `remaining_landmarks.csv`: selected remaining days {landmarks}.",
        "- `metadata.json`: run configuration, elapsed time, and DP cache counters.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
