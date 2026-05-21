from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateImportUsage=false

import argparse
from collections.abc import Mapping, Sequence
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

from experiments.single_card_tradeoff.config import SingleCardFSRS6Config  # noqa: E402
from experiments.single_card_tradeoff.oracle_frontier import parse_csv_floats  # noqa: E402
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    FSRS6BatchedStationaryFiniteOracle,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_policy_viz import (  # noqa: E402
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    _grid_edges,
    _retention_lists_match,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.defaults import DEFAULT_TARGET_RETENTIONS  # noqa: E402
from experiments.single_card_tradeoff.policy_net import PolicyValueNet  # noqa: E402
from simulator.defaults import DEFAULT_DAYS  # noqa: E402
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402


DEFAULT_BASE_DIR = Path(
    "artifacts/single_card_tradeoff/"
    "stationary_finite_distill_first8_users_per_user_uniform_table_supervision_"
    "fsrs6_baseline_gpu"
)
DEFAULT_R4_DIR = Path(
    "artifacts/single_card_tradeoff/"
    "stationary_finite_distill_first8_users_per_user_r4d1_e512_"
    "uniform_table_supervision_fsrs6_baseline_gpu"
)
DEFAULT_OUT_DIR = Path("artifacts/single_card_tradeoff/first8_r4d1_vs_476_policy_viz")
DEFAULT_USER_IDS = tuple(range(1, 9))
DEFAULT_COST_WEIGHTS = [0.0, 16.0, 64.0, 256.0, 1024.0]


@dataclass(frozen=True)
class LoadedPolicy:
    path: Path
    label: str
    params_per_user: int
    model: PolicyValueNet
    action_retentions: list[float]
    cost_weights: list[float]
    goal_norm_max: float
    config: SingleCardFSRS6Config


@dataclass(frozen=True)
class PolicyComparison:
    name: str
    label: str
    reference_name: str
    reference_label: str
    values: torch.Tensor


@dataclass(frozen=True)
class UserPolicyBundle:
    user_id: int
    exact: torch.Tensor
    base: torch.Tensor
    r4: torch.Tensor
    base_policy: LoadedPolicy
    r4_policy: LoadedPolicy


def _csv_floats(values: Sequence[float]) -> str:
    return ",".join(format_float(value) for value in values)


def _csv_ints(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _parse_user_ids(raw: str) -> list[int]:
    user_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid --user-ids value '{item}'.") from exc
        if value <= 0:
            raise SystemExit("--user-ids must contain positive integers.")
        user_ids.append(value)
    if not user_ids:
        raise SystemExit("--user-ids must contain at least one value.")
    if len(set(user_ids)) != len(user_ids):
        raise SystemExit("--user-ids contains duplicates.")
    return user_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize first-eight per-user exact, 476-parameter, and "
            "residual:4:1 stationary-finite distill policies on the same grid."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--user-ids",
        default=_csv_ints(DEFAULT_USER_IDS),
        help="Comma-separated user IDs to visualize.",
    )
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
    parser.add_argument(
        "--exact-user-batch-size",
        type=int,
        default=0,
        help=(
            "Number of users to solve together for the exact table. "
            "Use 0 to solve all selected users in one batch."
        ),
    )
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--base-policy-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--r4-policy-dir", type=Path, default=DEFAULT_R4_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-label", default="476")
    parser.add_argument("--r4-label", default="residual:4:1")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _resolve_device(raw: str | None) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _float_tuple(
    payload: Mapping[str, object],
    key: str,
    *,
    expected_len: int,
) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise SystemExit(f"user config is missing {key}.")
    values = tuple(float(value) for value in raw)
    if len(values) != expected_len:
        raise SystemExit(f"user config {key} must contain {expected_len} values.")
    return values


def _config_from_payload(
    payload: Mapping[str, object], *, path: Path
) -> SingleCardFSRS6Config:
    raw_user_id = payload.get("user_id")
    if raw_user_id is None:
        raise SystemExit(f"{path} user config is missing user_id.")
    raw_partition = payload.get("benchmark_partition", "0")
    raw_result = payload.get("benchmark_result")
    raw_root = payload.get("srs_benchmark_root")
    raw_button_usage = payload.get("button_usage")
    user_id = int(str(raw_user_id))
    return SingleCardFSRS6Config(
        environment=str(payload.get("environment", "fsrs6")),
        fsrs_weights=_float_tuple(payload, "fsrs_weights", expected_len=21),
        first_rating_prob=_float_tuple(
            payload,
            "first_rating_prob",
            expected_len=4,
        ),
        review_rating_prob=_float_tuple(
            payload,
            "review_rating_prob",
            expected_len=3,
        ),
        learning_costs=_float_tuple(payload, "learning_costs", expected_len=4),
        review_costs=_float_tuple(payload, "review_costs", expected_len=4),
        user_id=user_id,
        benchmark_result=str(raw_result) if raw_result is not None else None,
        benchmark_partition=str(raw_partition),
        srs_benchmark_root=str(raw_root) if raw_root is not None else None,
        button_usage=str(raw_button_usage) if raw_button_usage is not None else None,
    )


def _checkpoint_user_config(
    checkpoint: Mapping[str, object],
    *,
    path: Path,
) -> SingleCardFSRS6Config:
    raw_configs = checkpoint.get("user_configs")
    if isinstance(raw_configs, Sequence) and not isinstance(raw_configs, str):
        if len(raw_configs) != 1:
            raise SystemExit(f"{path} must contain exactly one user config.")
        raw_config = raw_configs[0]
        if not isinstance(raw_config, Mapping):
            raise SystemExit(f"{path} user_configs entry must be a mapping.")
        return _config_from_payload(raw_config, path=path)
    return _config_from_payload(checkpoint, path=path)


def _load_policy(
    path: Path,
    *,
    label: str,
    device: torch.device,
) -> LoadedPolicy:
    if not path.exists():
        raise SystemExit(f"Policy checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise SystemExit(f"Invalid policy checkpoint: {path}")
    if checkpoint.get("policy_type") != "fsrs6_oracle_stationary_finite_distill":
        raise SystemExit(
            f"{path} must have policy_type=fsrs6_oracle_stationary_finite_distill."
        )
    raw_actions = checkpoint.get("action_retentions")
    raw_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise SystemExit(f"{path} is missing action_retentions.")
    if not isinstance(raw_cost_weights, list) or not raw_cost_weights:
        raise SystemExit(f"{path} is missing cost_weights.")
    action_retentions = [float(value) for value in raw_actions]
    validate_retention_values(action_retentions, name=f"{path} action_retentions")
    cost_weights = [float(value) for value in raw_cost_weights]
    obs_mode = str(checkpoint.get("obs_mode", "oracle_stationary"))
    if obs_mode != "oracle_stationary":
        raise SystemExit(f"{path} must have obs_mode=oracle_stationary.")

    obs_dim = int(checkpoint.get("obs_dim", 3))
    hidden_size = int(checkpoint.get("hidden_size", 16))
    network = str(checkpoint.get("network", "residual"))
    network_depth = int(checkpoint.get("network_depth", 2))
    model = PolicyValueNet(
        obs_dim=obs_dim,
        action_count=len(action_retentions),
        hidden_size=hidden_size,
        architecture=network,
        depth=network_depth,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise SystemExit(f"{path} is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedPolicy(
        path=path,
        label=label,
        params_per_user=int(checkpoint.get("params_per_user", 0)),
        model=model,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        goal_norm_max=max(cost_weights),
        config=_checkpoint_user_config(checkpoint, path=path),
    )


def _policy_path(policy_dir: Path, user_id: int) -> Path:
    return policy_dir / f"user_{user_id}_policy.pt"


def _config_identity(config: SingleCardFSRS6Config) -> tuple[object, ...]:
    return (
        config.environment,
        config.fsrs_weights,
        config.first_rating_prob,
        config.review_rating_prob,
        config.learning_costs,
        config.review_costs,
        config.user_id,
    )


def _validate_policy_pair(
    *,
    base: LoadedPolicy,
    r4: LoadedPolicy,
    action_retentions: Sequence[float],
) -> None:
    if not _retention_lists_match(base.action_retentions, action_retentions):
        raise SystemExit(
            f"{base.path} action_retentions differ from --action-retentions."
        )
    if not _retention_lists_match(r4.action_retentions, action_retentions):
        raise SystemExit(
            f"{r4.path} action_retentions differ from --action-retentions."
        )
    if _config_identity(base.config) != _config_identity(r4.config):
        raise SystemExit(f"User config mismatch between {base.path} and {r4.path}.")


@torch.inference_mode()
def _distill_policy_grid(
    *,
    oracle: Any,
    model: PolicyValueNet,
    cost_weights: Sequence[float],
    goal_norm_max: float,
) -> torch.Tensor:
    s_count = int(oracle.s_grid.numel())
    d_count = int(oracle.d_grid.numel())
    model_dtype = next(model.parameters()).dtype
    s_norm = (
        (
            torch.log(torch.clamp(oracle.s_grid, min=oracle.bounds.s_min))
            - oracle.log_s_min
        )
        / (oracle.log_s_max - oracle.log_s_min)
    )[:, None].expand(s_count, d_count)
    d_norm = (
        (oracle.d_grid - oracle.bounds.d_min)
        / (oracle.bounds.d_max - oracle.bounds.d_min)
    )[None, :].expand(s_count, d_count)
    max_goal = max(1.0, float(goal_norm_max))
    policies: list[torch.Tensor] = []
    for cost_weight in cost_weights:
        goal_norm = math.log1p(float(cost_weight)) / math.log1p(max_goal)
        goal_grid = torch.full_like(s_norm, goal_norm)
        obs = torch.stack(
            [
                s_norm.reshape(-1),
                d_norm.reshape(-1),
                goal_grid.reshape(-1),
            ],
            dim=1,
        ).to(dtype=model_dtype)
        logits, _ = model(obs)
        policies.append(torch.argmax(logits, dim=1).reshape(s_count, d_count))
    return torch.stack(policies, dim=0).to(dtype=torch.int64)


def _solve_exact_policies(
    *,
    args: argparse.Namespace,
    configs: Sequence[SingleCardFSRS6Config],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    device: torch.device,
) -> tuple[Any, torch.Tensor, float, list[list[int]], list[list[float]]]:
    chunk_size = (
        len(configs)
        if int(args.exact_user_batch_size) <= 0
        else int(args.exact_user_batch_size)
    )
    policy_chunks: list[torch.Tensor] = []
    iterations: list[list[int]] = []
    residuals: list[list[float]] = []
    oracle_for_grid: Any | None = None
    start = time.perf_counter()
    for offset in range(0, len(configs), chunk_size):
        chunk = configs[offset : offset + chunk_size]
        oracle = FSRS6BatchedStationaryFiniteOracle(
            days=args.days,
            action_retentions=action_retentions,
            s_grid_size=args.s_grid_size,
            d_grid_size=args.d_grid_size,
            device=device,
            fsrs_weights=[
                tuple(config.fsrs_weights)
                if config.fsrs_weights
                else DEFAULT_FSRS6_WEIGHTS
                for config in chunk
            ],
            first_rating_prob=[tuple(config.first_rating_prob) for config in chunk],
            review_rating_prob=[tuple(config.review_rating_prob) for config in chunk],
            learning_costs=[tuple(config.learning_costs) for config in chunk],
            review_costs=[tuple(config.review_costs) for config in chunk],
        )
        solution = oracle.solve_stationary_finite_policies(
            cost_weights,
            max_iterations=args.oracle_stationary_finite_max_iterations,
            tolerance=args.oracle_stationary_finite_tolerance,
            progress=not args.no_progress,
        )
        failed = [
            f"user={config.user_id}:w={format_float(weight)}"
            for config, row in zip(chunk, solution.converged, strict=True)
            for weight, did_converge in zip(cost_weights, row, strict=True)
            if not did_converge
        ]
        if failed:
            raise RuntimeError(
                "Stationary finite oracle did not converge for " + ",".join(failed)
            )
        policy_chunks.append(solution.policy.to(device=device, dtype=torch.int64))
        iterations.extend(solution.iterations)
        residuals.extend(solution.residuals)
        oracle_for_grid = oracle
    if oracle_for_grid is None:
        raise RuntimeError("no exact policies were solved.")
    runtime_s = time.perf_counter() - start
    return (
        oracle_for_grid,
        torch.cat(policy_chunks, dim=0),
        runtime_s,
        iterations,
        residuals,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_and_modal(
    policy: torch.Tensor,
    action_retentions: Sequence[float],
) -> tuple[float, float]:
    action_tensor = torch.tensor(
        list(action_retentions),
        device=policy.device,
        dtype=torch.float64,
    )
    flat = policy.reshape(-1)
    counts = torch.bincount(flat, minlength=len(action_retentions))[
        : len(action_retentions)
    ]
    modal_idx = int(torch.argmax(counts).item())
    retention = action_tensor[flat]
    return float(retention.mean().item()), float(action_retentions[modal_idx])


def _comparison_values(
    *,
    policy: torch.Tensor,
    reference: torch.Tensor,
    action_retentions: Sequence[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    action_tensor = torch.tensor(
        list(action_retentions),
        device=policy.device,
        dtype=torch.float64,
    )
    policy_flat = policy.reshape(-1)
    reference_flat = reference.reshape(-1)
    diff = action_tensor[policy_flat] - action_tensor[reference_flat]
    total = float(diff.numel())
    return (
        diff.reshape_as(policy),
        {
            "action_match_share": float(
                (policy_flat == reference_flat).sum().item() / total
            ),
            "policy_lower_share": float((diff < 0.0).sum().item() / total),
            "policy_higher_share": float((diff > 0.0).sum().item() / total),
            "mean_retention_diff": float(diff.mean().item()),
            "mean_abs_retention_diff": float(torch.abs(diff).mean().item()),
            "max_abs_retention_diff": float(torch.max(torch.abs(diff)).item()),
        },
    )


def _build_rows(
    *,
    bundles: Sequence[UserPolicyBundle],
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    by_user_rows: list[dict[str, Any]] = []
    comparisons = [
        ("476_vs_exact", "476", "exact"),
        ("r4d1_vs_exact", "residual:4:1", "exact"),
        ("r4d1_vs_476", "residual:4:1", "476"),
    ]
    for bundle in bundles:
        policies = {
            "exact": bundle.exact,
            "476": bundle.base,
            "residual:4:1": bundle.r4,
        }
        params = {
            "exact": 0,
            "476": bundle.base_policy.params_per_user,
            "residual:4:1": bundle.r4_policy.params_per_user,
        }
        for weight_idx, cost_weight in enumerate(cost_weights):
            for policy_name, policy in policies.items():
                mean_retention, modal_retention = _mean_and_modal(
                    policy[weight_idx],
                    action_retentions,
                )
                action_rows.append(
                    {
                        "user_id": bundle.user_id,
                        "goal_cost_weight": cost_weight,
                        "policy": policy_name,
                        "params_per_user": params[policy_name],
                        "mean_action_retention": mean_retention,
                        "modal_action_retention": modal_retention,
                    }
                )
            for comparison_name, policy_name, reference_name in comparisons:
                _, metrics = _comparison_values(
                    policy=policies[policy_name][weight_idx],
                    reference=policies[reference_name][weight_idx],
                    action_retentions=action_retentions,
                )
                policy_mean, policy_modal = _mean_and_modal(
                    policies[policy_name][weight_idx],
                    action_retentions,
                )
                reference_mean, reference_modal = _mean_and_modal(
                    policies[reference_name][weight_idx],
                    action_retentions,
                )
                comparison_rows.append(
                    {
                        "user_id": bundle.user_id,
                        "goal_cost_weight": cost_weight,
                        "comparison": comparison_name,
                        "policy": policy_name,
                        "reference_policy": reference_name,
                        "policy_params_per_user": params[policy_name],
                        "reference_params_per_user": params[reference_name],
                        "policy_mean_action_retention": policy_mean,
                        "reference_mean_action_retention": reference_mean,
                        "policy_modal_retention": policy_modal,
                        "reference_modal_retention": reference_modal,
                        **metrics,
                    }
                )

        for comparison_name, policy_name, reference_name in comparisons:
            weighted_rows = [
                row
                for row in comparison_rows
                if int(row["user_id"]) == bundle.user_id
                and str(row["comparison"]) == comparison_name
            ]
            by_user_rows.append(
                {
                    "user_id": bundle.user_id,
                    "comparison": comparison_name,
                    "policy": policy_name,
                    "reference_policy": reference_name,
                    "mean_abs_retention_diff_all_weights": sum(
                        float(row["mean_abs_retention_diff"]) for row in weighted_rows
                    )
                    / len(weighted_rows),
                    "action_match_share_all_weights": sum(
                        float(row["action_match_share"]) for row in weighted_rows
                    )
                    / len(weighted_rows),
                    "policy_lower_share_all_weights": sum(
                        float(row["policy_lower_share"]) for row in weighted_rows
                    )
                    / len(weighted_rows),
                    "policy_higher_share_all_weights": sum(
                        float(row["policy_higher_share"]) for row in weighted_rows
                    )
                    / len(weighted_rows),
                    "largest_weight_mean_abs_diff": max(
                        weighted_rows,
                        key=lambda row: float(row["mean_abs_retention_diff"]),
                    )["goal_cost_weight"],
                }
            )
    return action_rows, comparison_rows, by_user_rows


def _comparison_grids(
    *,
    bundle: UserPolicyBundle,
    action_retentions: Sequence[float],
) -> list[PolicyComparison]:
    base_exact, _ = _comparison_values(
        policy=bundle.base,
        reference=bundle.exact,
        action_retentions=action_retentions,
    )
    r4_exact, _ = _comparison_values(
        policy=bundle.r4,
        reference=bundle.exact,
        action_retentions=action_retentions,
    )
    r4_base, _ = _comparison_values(
        policy=bundle.r4,
        reference=bundle.base,
        action_retentions=action_retentions,
    )
    return [
        PolicyComparison("476_vs_exact", "476 - exact", "exact", "exact", base_exact),
        PolicyComparison(
            "r4d1_vs_exact",
            "residual:4:1 - exact",
            "exact",
            "exact",
            r4_exact,
        ),
        PolicyComparison(
            "r4d1_vs_476",
            "residual:4:1 - 476",
            "476",
            "476",
            r4_base,
        ),
    ]


def _plot_user_bundle(
    path: Path,
    *,
    oracle: Any,
    bundle: UserPolicyBundle,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
) -> Any:
    import matplotlib.pyplot as plt

    policies = [
        ("exact", bundle.exact),
        ("476", bundle.base),
        ("residual:4:1", bundle.r4),
    ]
    comparisons = _comparison_grids(bundle=bundle, action_retentions=action_retentions)
    action_tensor = torch.tensor(list(action_retentions), dtype=torch.float64)
    d_edges = _grid_edges(oracle.d_grid, log_space=False)
    s_edges = _grid_edges(oracle.s_grid, log_space=True)
    row_count = len(policies) + len(comparisons)
    col_count = len(cost_weights)
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(3.25 * col_count, 2.1 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    policy_mesh = None
    diff_mesh = None
    max_abs = max(
        0.01,
        max(
            float(torch.max(torch.abs(comparison.values)).item())
            for comparison in comparisons
        ),
    )

    for row_idx, (label, policy) in enumerate(policies):
        for col_idx, cost_weight in enumerate(cost_weights):
            ax = axes[row_idx][col_idx]
            retention_grid = action_tensor[
                policy[col_idx].detach().to(device="cpu", dtype=torch.int64)
            ]
            policy_mesh = ax.pcolormesh(
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
                ax.set_ylabel(f"{label}\nStability")

    for offset, comparison in enumerate(comparisons):
        row_idx = len(policies) + offset
        for col_idx, cost_weight in enumerate(cost_weights):
            ax = axes[row_idx][col_idx]
            diff_mesh = ax.pcolormesh(
                d_edges,
                s_edges,
                comparison.values[col_idx].detach().to(device="cpu").numpy(),
                shading="auto",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            ax.set_yscale("log")
            if col_idx == 0:
                ax.set_ylabel(f"{comparison.label}\nStability")
            if row_idx == row_count - 1:
                ax.set_xlabel("Difficulty")

    fig.suptitle(f"User {bundle.user_id} stationary-finite policy grids")
    if policy_mesh is not None:
        cbar = fig.colorbar(policy_mesh, ax=axes[: len(policies), :].ravel().tolist())
        cbar.set_label("Desired-retention action")
    if diff_mesh is not None:
        cbar = fig.colorbar(diff_mesh, ax=axes[len(policies) :, :].ravel().tolist())
        cbar.set_label("Retention difference")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    return fig


def _plot_overview(
    path: Path,
    *,
    comparison_rows: Sequence[Mapping[str, Any]],
    user_ids: Sequence[int],
    cost_weights: Sequence[float],
) -> None:
    import matplotlib.pyplot as plt

    comparisons = ["476_vs_exact", "r4d1_vs_exact", "r4d1_vs_476"]
    metrics = [
        ("mean_abs_retention_diff", "Mean abs retention diff", "magma", "{:.3f}"),
        ("action_match_share", "Action match share", "viridis", "{:.0%}"),
    ]
    row_lookup = {
        (
            int(row["user_id"]),
            float(row["goal_cost_weight"]),
            str(row["comparison"]),
        ): row
        for row in comparison_rows
    }
    fig, axes = plt.subplots(
        len(metrics),
        len(comparisons),
        figsize=(4.2 * len(comparisons), 3.8 * len(metrics)),
        squeeze=False,
        constrained_layout=True,
    )
    for metric_idx, (metric, title, cmap, annotation_format) in enumerate(metrics):
        for comparison_idx, comparison in enumerate(comparisons):
            ax = axes[metric_idx][comparison_idx]
            matrix = torch.tensor(
                [
                    [
                        float(row_lookup[(user_id, float(weight), comparison)][metric])
                        for weight in cost_weights
                    ]
                    for user_id in user_ids
                ],
                dtype=torch.float64,
            )
            im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
            ax.set_title(f"{comparison}\n{title}")
            ax.set_xticks(range(len(cost_weights)))
            ax.set_xticklabels([format_float(weight) for weight in cost_weights])
            ax.set_yticks(range(len(user_ids)))
            ax.set_yticklabels([str(user_id) for user_id in user_ids])
            ax.set_xlabel("Goal cost weight")
            if comparison_idx == 0:
                ax.set_ylabel("User")
            for user_idx in range(len(user_ids)):
                for weight_idx in range(len(cost_weights)):
                    value = float(matrix[user_idx, weight_idx].item())
                    ax.text(
                        weight_idx,
                        user_idx,
                        annotation_format.format(value),
                        ha="center",
                        va="center",
                        color="white"
                        if metric == "mean_abs_retention_diff"
                        else "black",
                        fontsize=7,
                    )
            fig.colorbar(im, ax=ax, shrink=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_findings(
    path: Path,
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    exact_runtime_s: float,
    device: torch.device,
    action_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    max_diff = max(
        comparison_rows,
        key=lambda row: float(row["mean_abs_retention_diff"]),
    )
    min_match = min(
        comparison_rows,
        key=lambda row: float(row["action_match_share"]),
    )
    exact_means = [
        float(row["mean_action_retention"])
        for row in action_rows
        if str(row["policy"]) == "exact"
    ]
    lines = [
        "# residual:4:1 vs 476 Policy Visualization",
        "",
        "Inputs:",
        f"- Users: `{','.join(str(user_id) for user_id in user_ids)}`",
        f"- 476 checkpoints: `{args.base_policy_dir}`",
        f"- residual:4:1 checkpoints: `{args.r4_policy_dir}`",
        f"- Grid: `{args.s_grid_size}` log-spaced stability values x `{args.d_grid_size}` difficulty values",
        f"- Goal cost weights: `{','.join(format_float(value) for value in parse_csv_floats(args.cost_weights, name='--cost-weights'))}`",
        f"- Exact solve runtime: `{exact_runtime_s:.2f}s` on `{device}`",
        "",
        "Artifacts:",
        "- `user_<id>_policy_compare.png`: exact, 476, residual:4:1, and three difference panels.",
        "- `all_users_policy_compare.pdf`: all per-user panels in one PDF.",
        "- `policy_difference_overview.png`: user x weight heatmaps for match and mean absolute differences.",
        "- `policy_action_summary.csv`: policy-level mean/modal actions for exact, 476, and residual:4:1.",
        "- `policy_comparison_summary.csv`: pairwise comparisons for 476-vs-exact, residual:4:1-vs-exact, and residual:4:1-vs-476.",
        "- `policy_comparison_by_user.csv`: pairwise metrics aggregated across all weights.",
        "",
        "Quick scan:",
        (
            "- Exact mean action retention spans "
            f"`{min(exact_means):.4f}` to `{max(exact_means):.4f}` across visualized user-weight grids."
        ),
        (
            "- Largest mean absolute retention difference: "
            f"{max_diff['comparison']}, user {max_diff['user_id']}, "
            f"w={format_float(float(max_diff['goal_cost_weight']))}, "
            f"MAD={float(max_diff['mean_abs_retention_diff']):.8f}."
        ),
        (
            "- Lowest action match: "
            f"{min_match['comparison']}, user {min_match['user_id']}, "
            f"w={format_float(float(min_match['goal_cost_weight']))}, "
            f"match={float(min_match['action_match_share']):.2%}."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.s_grid_size < 8 or args.d_grid_size < 8:
        raise SystemExit("--grid sizes must be >= 8.")
    if args.oracle_stationary_finite_max_iterations <= 0:
        raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
    if args.oracle_stationary_finite_tolerance <= 0.0:
        raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")
    if args.exact_user_batch_size < 0:
        raise SystemExit("--exact-user-batch-size must be >= 0.")

    user_ids = _parse_user_ids(args.user_ids)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    device = _resolve_device(args.torch_device)

    base_policies: list[LoadedPolicy] = []
    r4_policies: list[LoadedPolicy] = []
    configs: list[SingleCardFSRS6Config] = []
    for user_id in user_ids:
        base = _load_policy(
            _policy_path(args.base_policy_dir, user_id),
            label=args.base_label,
            device=device,
        )
        r4 = _load_policy(
            _policy_path(args.r4_policy_dir, user_id),
            label=args.r4_label,
            device=device,
        )
        if base.config.user_id != user_id:
            raise SystemExit(f"{base.path} stores user_id={base.config.user_id}.")
        if r4.config.user_id != user_id:
            raise SystemExit(f"{r4.path} stores user_id={r4.config.user_id}.")
        _validate_policy_pair(
            base=base,
            r4=r4,
            action_retentions=action_retentions,
        )
        base_policies.append(base)
        r4_policies.append(r4)
        configs.append(base.config)

    oracle, exact_policy, exact_runtime_s, _, _ = _solve_exact_policies(
        args=args,
        configs=configs,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
    )

    bundles: list[UserPolicyBundle] = []
    for user_idx, user_id in enumerate(user_ids):
        base_grid = _distill_policy_grid(
            oracle=oracle,
            model=base_policies[user_idx].model,
            cost_weights=cost_weights,
            goal_norm_max=base_policies[user_idx].goal_norm_max,
        )
        r4_grid = _distill_policy_grid(
            oracle=oracle,
            model=r4_policies[user_idx].model,
            cost_weights=cost_weights,
            goal_norm_max=r4_policies[user_idx].goal_norm_max,
        )
        bundles.append(
            UserPolicyBundle(
                user_id=user_id,
                exact=exact_policy[user_idx],
                base=base_grid,
                r4=r4_grid,
                base_policy=base_policies[user_idx],
                r4_policy=r4_policies[user_idx],
            )
        )

    action_rows, comparison_rows, by_user_rows = _build_rows(
        bundles=bundles,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )
    _write_csv(args.out_dir / "policy_action_summary.csv", action_rows)
    _write_csv(args.out_dir / "policy_comparison_summary.csv", comparison_rows)
    _write_csv(args.out_dir / "policy_comparison_by_user.csv", by_user_rows)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(args.out_dir / "all_users_policy_compare.pdf") as pdf:
        for bundle in bundles:
            fig = _plot_user_bundle(
                args.out_dir / f"user_{bundle.user_id}_policy_compare.png",
                oracle=oracle,
                bundle=bundle,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
            )
            pdf.savefig(fig)
            plt.close(fig)

    _plot_overview(
        args.out_dir / "policy_difference_overview.png",
        comparison_rows=comparison_rows,
        user_ids=user_ids,
        cost_weights=cost_weights,
    )
    _write_findings(
        args.out_dir / "findings.md",
        args=args,
        user_ids=user_ids,
        exact_runtime_s=exact_runtime_s,
        device=device,
        action_rows=action_rows,
        comparison_rows=comparison_rows,
    )
    print(f"Wrote CSV: {args.out_dir / 'policy_action_summary.csv'}")
    print(f"Wrote CSV: {args.out_dir / 'policy_comparison_summary.csv'}")
    print(f"Wrote CSV: {args.out_dir / 'policy_comparison_by_user.csv'}")
    print(f"Wrote plot: {args.out_dir / 'policy_difference_overview.png'}")
    print(f"Wrote PDF: {args.out_dir / 'all_users_policy_compare.pdf'}")
    print(f"Wrote report: {args.out_dir / 'findings.md'}")


if __name__ == "__main__":
    main()
