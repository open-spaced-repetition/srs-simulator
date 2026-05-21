from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
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

from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    configure_oracle_dp_cache_from_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.cli.oracle_frontier import parse_csv_floats  # noqa: E402
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    FSRS6StationaryFiniteOracle,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_policy_viz import (  # noqa: E402
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    _distill_policy_grid,
    _grid_edges,
    _load_distill_policy,
    _retention_lists_match,
)
from experiments.single_card_tradeoff.core.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.core.defaults import DEFAULT_TARGET_RETENTIONS  # noqa: E402
from experiments.single_card_tradeoff.cli.uvfa_ppo import fsrs_config_kwargs  # noqa: E402
from simulator.defaults import DEFAULT_DAYS  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402


DEFAULT_POLICY_DIR = Path(
    "artifacts/single_card_tradeoff/stationary_finite_model_size_ablation"
)
DEFAULT_OUT_DIR = DEFAULT_POLICY_DIR / "policy_viz"
DEFAULT_COST_WEIGHTS = [0.0, 16.0, 64.0, 256.0, 1024.0]
DEFAULT_CANDIDATES = ["r16d2", "r12d2", "r10d2", "r8d2", "r8d1", "r6d1"]
DEFAULT_SUB216_CANDIDATES = [
    "r5d1",
    "r4d1",
    "r3d1",
    "mlp8",
    "mlp6",
    "mlp4",
    "linear",
    "quadratic",
]
DEFAULT_CANDIDATE_SETS = {
    "default": DEFAULT_CANDIDATES,
    "sub216": DEFAULT_SUB216_CANDIDATES,
    "all": [*DEFAULT_CANDIDATES, *DEFAULT_SUB216_CANDIDATES],
}


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    filename: str


CANDIDATES: dict[str, Candidate] = {
    "r16d2": Candidate("r16d2", "residual:16:2", "r16d2_e128_policy.pt"),
    "r12d2": Candidate("r12d2", "residual:12:2", "r12d2_e128_policy.pt"),
    "r10d2": Candidate("r10d2", "residual:10:2", "r10d2_e128_policy.pt"),
    "r8d2": Candidate("r8d2", "residual:8:2", "r8d2_e128_policy.pt"),
    "r8d1": Candidate("r8d1", "residual:8:1", "r8d1_e128_policy.pt"),
    "r6d1": Candidate("r6d1", "residual:6:1", "r6d1_e128_policy.pt"),
    "r5d1": Candidate("r5d1", "residual:5:1", "r5d1_e128_policy.pt"),
    "r4d1": Candidate("r4d1", "residual:4:1", "r4d1_e128_policy.pt"),
    "r3d1": Candidate("r3d1", "residual:3:1", "r3d1_e128_policy.pt"),
    "mlp8": Candidate("mlp8", "mlp:8", "mlp8_e128_policy.pt"),
    "mlp6": Candidate("mlp6", "mlp:6", "mlp6_e128_policy.pt"),
    "mlp4": Candidate("mlp4", "mlp:4", "mlp4_e128_policy.pt"),
    "linear": Candidate("linear", "linear", "linear_e128_policy.pt"),
    "quadratic": Candidate("quadratic", "quadratic", "quadratic_e128_policy.pt"),
}


def _csv_floats(values: Sequence[float]) -> str:
    return ",".join(format_float(value) for value in values)


def _parse_candidate_csv(raw: str) -> list[Candidate]:
    selected: list[Candidate] = []
    for item in raw.split(","):
        key = item.strip()
        if not key:
            continue
        try:
            selected.append(CANDIDATES[key])
        except KeyError as exc:
            valid = ",".join(CANDIDATES)
            raise SystemExit(
                f"Unknown candidate '{key}'. Valid values: {valid}"
            ) from exc
    if not selected:
        raise SystemExit("--candidates must include at least one value.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize stationary finite distill policies from the model-size "
            "ablation side-by-side against the exact stationary finite oracle."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--cost-weights",
        default=_csv_floats(DEFAULT_COST_WEIGHTS),
        help="Comma-separated scalarization weights to solve and visualize.",
    )
    parser.add_argument(
        "--action-retentions",
        default=_csv_floats(DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention actions available to the oracle.",
    )
    parser.add_argument("--s-grid-size", type=int, default=64)
    parser.add_argument("--d-grid-size", type=int, default=32)
    parser.add_argument(
        "--oracle-stationary-finite-max-iterations",
        type=int,
        default=DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--oracle-stationary-finite-tolerance",
        type=float,
        default=DEFAULT_STATIONARY_FINITE_TOLERANCE,
    )
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--candidate-set",
        choices=sorted(DEFAULT_CANDIDATE_SETS),
        default="default",
        help="Named candidate set to visualize when --candidates is omitted.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help=f"Comma-separated candidate keys. Valid values: {','.join(CANDIDATES)}",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _resolve_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _policy_stats(
    *,
    policy_name: str,
    policy_label: str,
    policies: torch.Tensor,
    exact_policies: torch.Tensor,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> list[dict[str, Any]]:
    action_tensor = torch.tensor(
        list(action_retentions),
        device=policies.device,
        dtype=torch.float64,
    )
    rows: list[dict[str, Any]] = []
    for weight_idx, cost_weight in enumerate(cost_weights):
        flat = policies[weight_idx].reshape(-1)
        exact_flat = exact_policies[weight_idx].reshape(-1)
        counts = torch.bincount(flat, minlength=len(action_retentions)).to(
            dtype=torch.float64
        )[: len(action_retentions)]
        total = float(counts.sum().item())
        shares = counts / max(total, 1.0)
        modal_idx = int(torch.argmax(counts).item())
        retention = action_tensor[flat]
        exact_retention = action_tensor[exact_flat]
        diff = retention - exact_retention
        exact_match = float((flat == exact_flat).to(dtype=torch.float64).mean().item())
        lower_share = float((diff < 0.0).to(dtype=torch.float64).mean().item())
        higher_share = float((diff > 0.0).to(dtype=torch.float64).mean().item())
        rows.append(
            {
                "policy": policy_name,
                "label": policy_label,
                "goal_cost_weight": cost_weight,
                "mean_action_retention": float(retention.mean().item()),
                "modal_action_retention": action_retentions[modal_idx],
                "modal_cell_share": float(shares[modal_idx].item()),
                "exact_match": exact_match,
                "distill_lower_share": lower_share,
                "distill_higher_share": higher_share,
                "mean_abs_retention_diff": float(torch.abs(diff).mean().item()),
                "mean_retention_diff": float(diff.mean().item()),
            }
        )
    return rows


def _plot_policy_grid(
    path: Path,
    *,
    oracle: FSRS6StationaryFiniteOracle,
    policies_by_name: Mapping[str, torch.Tensor],
    labels_by_name: Mapping[str, str],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    action_tensor = torch.tensor(
        list(action_retentions),
        device=next(iter(policies_by_name.values())).device,
        dtype=torch.float64,
    )
    names = list(policies_by_name)
    d_edges = _grid_edges(oracle.d_grid, log_space=False)
    s_edges = _grid_edges(oracle.s_grid, log_space=True)
    row_count = len(names)
    col_count = len(cost_weights)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(3.4 * col_count, 2.65 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for row_idx, name in enumerate(names):
        policies = policies_by_name[name]
        for col_idx, cost_weight in enumerate(cost_weights):
            ax = axes[row_idx][col_idx]
            retention_grid = action_tensor[policies[col_idx]].detach().to(device="cpu")
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
            if row_idx == 0:
                ax.set_title(f"w={format_float(cost_weight)}")
            if col_idx == 0:
                ax.set_ylabel(f"{labels_by_name[name]}\nStability")
            if row_idx == row_count - 1:
                ax.set_xlabel("Difficulty")
    fig.suptitle("Stationary finite policies by architecture")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("Desired-retention action")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_difference_grid(
    path: Path,
    *,
    oracle: FSRS6StationaryFiniteOracle,
    student_policies_by_name: Mapping[str, torch.Tensor],
    exact_policies: torch.Tensor,
    labels_by_name: Mapping[str, str],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    action_tensor = torch.tensor(
        list(action_retentions),
        device=exact_policies.device,
        dtype=torch.float64,
    )
    names = list(student_policies_by_name)
    d_edges = _grid_edges(oracle.d_grid, log_space=False)
    s_edges = _grid_edges(oracle.s_grid, log_space=True)
    diffs: dict[tuple[str, int], torch.Tensor] = {}
    max_abs = 0.0
    for name, policies in student_policies_by_name.items():
        for weight_idx in range(len(cost_weights)):
            diff = (
                action_tensor[policies[weight_idx]]
                - action_tensor[exact_policies[weight_idx]]
            ).detach()
            diffs[(name, weight_idx)] = diff
            max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))
    vmax = max(max_abs, 0.01)
    row_count = len(names)
    col_count = len(cost_weights)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(3.4 * col_count, 2.65 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mesh = None
    for row_idx, name in enumerate(names):
        for col_idx, cost_weight in enumerate(cost_weights):
            ax = axes[row_idx][col_idx]
            mesh = ax.pcolormesh(
                d_edges,
                s_edges,
                diffs[(name, col_idx)].to(device="cpu").numpy(),
                shading="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_yscale("log")
            if row_idx == 0:
                ax.set_title(f"w={format_float(cost_weight)}")
            if col_idx == 0:
                ax.set_ylabel(f"{labels_by_name[name]}\nStability")
            if row_idx == row_count - 1:
                ax.set_xlabel("Difficulty")
    fig.suptitle("Student policy action difference from exact oracle")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("Student retention - exact retention")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mean_action_lines(
    path: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    cost_weights: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    by_policy: dict[str, list[Mapping[str, Any]]] = {}
    for row in summary_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for policy, rows in by_policy.items():
        rows = sorted(rows, key=lambda row: float(row["goal_cost_weight"]))
        label = str(rows[0]["label"])
        ax.plot(
            [float(row["goal_cost_weight"]) for row in rows],
            [float(row["mean_action_retention"]) for row in rows],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks(list(cost_weights))
    ax.set_xticklabels([format_float(value) for value in cost_weights])
    ax.set_xlabel("Goal cost weight")
    ax.set_ylabel("Mean desired-retention action")
    ax.set_title("Mean policy action by architecture")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_findings(
    path: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    runtime_s: float,
    device: torch.device,
) -> None:
    student_rows = [row for row in summary_rows if row["policy"] != "exact"]
    by_policy: dict[str, list[Mapping[str, Any]]] = {}
    for row in student_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    lines = [
        "# Stationary Finite Architecture Policy Visualization",
        "",
        f"Runtime: {runtime_s:.2f}s on `{device}`.",
        "",
        "| policy | mean exact match | mean abs retention diff | max abs retention diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy, rows in by_policy.items():
        label = str(rows[0]["label"])
        exact_match = sum(float(row["exact_match"]) for row in rows) / len(rows)
        mean_abs = sum(float(row["mean_abs_retention_diff"]) for row in rows) / len(
            rows
        )
        max_abs = max(float(row["mean_abs_retention_diff"]) for row in rows)
        lines.append(
            f"| {label} | {exact_match:.2%} | {mean_abs:.4f} | {max_abs:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_config = configure_oracle_dp_cache_from_args(args)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    candidate_keys = (
        args.candidates
        if args.candidates is not None
        else ",".join(DEFAULT_CANDIDATE_SETS[args.candidate_set])
    )
    candidates = _parse_candidate_csv(candidate_keys)
    device = _resolve_device(args.torch_device)
    fsrs_config = load_single_card_fsrs6_config(args)

    oracle = FSRS6StationaryFiniteOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.s_grid_size,
        d_grid_size=args.d_grid_size,
        device=device,
        cache_config=cache_config,
        **fsrs_config_kwargs(fsrs_config),
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
        raise RuntimeError(
            "Stationary finite oracle did not converge for cost weights: "
            + ",".join(failed)
        )

    policies_by_name: dict[str, torch.Tensor] = {"exact": solution.policy}
    labels_by_name: dict[str, str] = {"exact": "exact oracle"}
    student_policies_by_name: dict[str, torch.Tensor] = {}
    for candidate in candidates:
        policy_path = args.policy_dir / candidate.filename
        model, distill_actions, _, goal_norm_max, _ = _load_distill_policy(
            policy_path,
            device=device,
        )
        if not _retention_lists_match(distill_actions, action_retentions):
            raise SystemExit(
                f"{policy_path} action_retentions must match --action-retentions."
            )
        policies = _distill_policy_grid(
            oracle=oracle,
            model=model,
            cost_weights=cost_weights,
            goal_norm_max=goal_norm_max,
        )
        policies_by_name[candidate.key] = policies
        student_policies_by_name[candidate.key] = policies
        labels_by_name[candidate.key] = candidate.label

    summary_rows: list[dict[str, Any]] = []
    for name, policies in policies_by_name.items():
        summary_rows.extend(
            _policy_stats(
                policy_name=name,
                policy_label=labels_by_name[name],
                policies=policies,
                exact_policies=solution.policy,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
            )
        )

    runtime_s = time.perf_counter() - start
    summary_path = args.out_dir / "arch_policy_summary.csv"
    heatmap_path = args.out_dir / "arch_policy_heatmaps.png"
    diff_path = args.out_dir / "arch_exact_difference_heatmaps.png"
    mean_path = args.out_dir / "arch_mean_action_retention.png"
    findings_path = args.out_dir / "findings.md"
    _write_csv(summary_path, summary_rows)
    _plot_policy_grid(
        heatmap_path,
        oracle=oracle,
        policies_by_name=policies_by_name,
        labels_by_name=labels_by_name,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    _plot_difference_grid(
        diff_path,
        oracle=oracle,
        student_policies_by_name=student_policies_by_name,
        exact_policies=solution.policy,
        labels_by_name=labels_by_name,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    _plot_mean_action_lines(
        mean_path,
        summary_rows=summary_rows,
        cost_weights=cost_weights,
    )
    _write_findings(
        findings_path,
        summary_rows=summary_rows,
        runtime_s=runtime_s,
        device=device,
    )
    print(f"Wrote CSV: {summary_path}")
    print(f"Wrote plot: {heatmap_path}")
    print(f"Wrote plot: {diff_path}")
    print(f"Wrote plot: {mean_path}")
    print(f"Wrote report: {findings_path}")


if __name__ == "__main__":
    main()
