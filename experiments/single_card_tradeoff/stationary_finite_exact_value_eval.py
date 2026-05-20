from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.auc_outputs import (  # noqa: E402
    DETAILED_AUC_FIELDS,
    write_auc_summary,
    write_mean_auc_summary,
)
from experiments.single_card_tradeoff.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
)
from experiments.single_card_tradeoff.low_param_direct_policy_search_multiuser import (  # noqa: E402
    DIRECT_POLICY_SCHEDULER,
    direct_policy_retention,
)
from experiments.single_card_tradeoff.oracle_frontier import (  # noqa: E402
    FSRS6BatchedStationaryFiniteOracle,
    OracleMetrics,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    BASELINE_SCHEDULER,
    DEFAULT_USER_IDS,
    PER_USER_SCHEDULER,
    batched_ensemble_forward,
    load_per_user_distill_ensemble,
    load_user_configs,
    parse_user_ids,
)
from experiments.single_card_tradeoff.retention_space import (  # noqa: E402
    validate_retention_values,
)
from experiments.single_card_tradeoff.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.tradeoff import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
    _build_regret_auc_rows,
    _write_csv,
    _write_regret_auc_csv,
)
from experiments.single_card_tradeoff.uvfa_ppo import (  # noqa: E402
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    parse_csv_floats,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED  # noqa: E402
from simulator.batched_sweep.fsrs6_adr_policy import (  # noqa: E402
    FSRS6ADRPolicySpec,
    resolve_fsrs6_adr_policy_specs,
)
from simulator.fsrs6_adr_policy import (  # noqa: E402
    FEATURE_VERSION_LOG_LINEAR,
    FEATURE_VERSION_LOG_POLY,
    FEATURE_VERSION_LOG_POLY_TIME,
    FSRS6ADRPolicy,
)
from simulator.scheduler_spec import format_float  # noqa: E402


EXACT_SCHEDULER = "fsrs6_oracle_stationary_finite"
ADR_SCHEDULER = "fsrs6_adr"
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/stationary_finite_exact_value_first8_users"
)
DEFAULT_DISTILL_POLICIES = (
    "distill_476=artifacts/single_card_tradeoff/"
    "stationary_finite_distill_first8_users_per_user_uniform_table_supervision_"
    "fsrs6_baseline_gpu",
    "distill_r4d1_e512=artifacts/single_card_tradeoff/"
    "stationary_finite_distill_first8_users_per_user_r4d1_e512_"
    "uniform_table_supervision_fsrs6_baseline_gpu",
)
DEFAULT_DIRECT_POLICIES = (
    "direct_7_sparse=artifacts/single_card_tradeoff/"
    "low_param_direct_policy_search_first8_users/policy.pt",
    "direct_7_dense=artifacts/single_card_tradeoff/"
    "low_param_direct_policy_search_first8_users_dense_weights/policy.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stationary finite policy tables deterministically with the "
            "FSRS-6 finite-lifecycle DP/occupancy model."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.set_defaults(env="fsrs6")
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
        help="Comma-separated benchmark user IDs to evaluate.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help="Cost weights to evaluate for stationary policies.",
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
        help="Discrete desired-retention action grid used by the DP evaluator.",
    )
    parser.add_argument(
        "--oracle-s-grid-size",
        type=int,
        default=DEFAULT_ORACLE_S_GRID_SIZE,
    )
    parser.add_argument(
        "--oracle-d-grid-size",
        type=int,
        default=DEFAULT_ORACLE_D_GRID_SIZE,
    )
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
        "--distill-policy",
        action="append",
        default=None,
        metavar="LABEL=DIR",
        help=(
            "Per-user distill checkpoint directory to discretize and evaluate. "
            "May be passed multiple times; defaults to the first-eight 476-param "
            "and residual:4:1 e512 artifacts."
        ),
    )
    parser.add_argument(
        "--direct-policy",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "Direct-search policy.pt to discretize and evaluate. May be passed "
            "multiple times; defaults to the existing 7-param sparse/dense runs."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-policy-root",
        type=Path,
        default=None,
        help=(
            "Root containing expanded FSRS6 ADR policy JSONs. When set, ADR "
            "policies are evaluated as fixed continuous interval policies with "
            "deterministic occupancy."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-train-run-root",
        type=Path,
        default=None,
        help=(
            "ADR train run root containing train-overfit/train_outputs. Mutually "
            "exclusive with --fsrs6-adr-policy-root and --fsrs6-adr-policy-manifest."
        ),
    )
    parser.add_argument(
        "--fsrs6-adr-policy-manifest",
        type=Path,
        default=None,
        help="TOML manifest with explicit FSRS6 ADR policy entries.",
    )
    parser.add_argument(
        "--fsrs6-adr-lambda-values",
        default=None,
        help="Optional comma-separated lambda filter for ADR policy discovery.",
    )
    parser.add_argument(
        "--skip-default-policies",
        action="store_true",
        help="Evaluate only explicitly supplied --distill-policy/--direct-policy specs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_run_monitoring_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Policy spec must be LABEL=PATH, got '{raw}'.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit(f"Policy spec must be LABEL=PATH, got '{raw}'.")
    return label, Path(path)


def _policy_specs(
    explicit: Sequence[str] | None,
    defaults: Sequence[str],
    *,
    skip_defaults: bool,
) -> list[tuple[str, Path]]:
    raw_specs = list(explicit or [])
    if not skip_defaults:
        raw_specs = [*defaults, *raw_specs]
    return [_label_path(raw) for raw in raw_specs]


def _validate_args(args: argparse.Namespace) -> None:
    if args.env != "fsrs6":
        raise SystemExit("stationary finite exact-value eval requires --env fsrs6.")
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle-*-grid-size values must be >= 8.")
    if args.oracle_stationary_finite_max_iterations <= 0:
        raise SystemExit("--oracle-stationary-finite-max-iterations must be > 0.")
    if args.oracle_stationary_finite_tolerance <= 0.0:
        raise SystemExit("--oracle-stationary-finite-tolerance must be > 0.")
    adr_sources = [
        args.fsrs6_adr_policy_root is not None,
        args.fsrs6_adr_train_run_root is not None,
        args.fsrs6_adr_policy_manifest is not None,
    ]
    if sum(adr_sources) > 1:
        raise SystemExit(
            "Configure at most one ADR policy source: "
            "--fsrs6-adr-policy-root, --fsrs6-adr-train-run-root, or "
            "--fsrs6-adr-policy-manifest."
        )


def _build_oracle(
    args: argparse.Namespace,
    *,
    configs: Sequence[SingleCardFSRS6Config],
    action_retentions: Sequence[float],
    device: torch.device,
) -> FSRS6BatchedStationaryFiniteOracle:
    return FSRS6BatchedStationaryFiniteOracle(
        days=args.days,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        fsrs_weights=[config.fsrs_weights for config in configs],
        first_rating_prob=[config.first_rating_prob for config in configs],
        review_rating_prob=[config.review_rating_prob for config in configs],
        learning_costs=[config.learning_costs for config in configs],
        review_costs=[config.review_costs for config in configs],
        dtype=torch.float64,
        device=device,
    )


def _metric_row(
    args: argparse.Namespace,
    *,
    user_id: int,
    scheduler: str,
    scheduler_spec: str,
    desired_retention: float | None,
    goal_cost_weight: float | None,
    metrics: OracleMetrics,
    runtime_s: float,
    engine: str,
) -> dict[str, Any]:
    deck_scale = float(args.deck_scale)
    particles = 1
    return {
        "user_id": user_id,
        "environment": f"fsrs6_user_{user_id}",
        "scheduler": scheduler,
        "scheduler_spec": scheduler_spec,
        "desired_retention": desired_retention,
        "fixed_interval": None,
        "goal_cost_weight": goal_cost_weight,
        "seed": args.seed,
        "days": args.days,
        "particles": particles,
        "deck_scale": args.deck_scale,
        "card_expected_retrievability": metrics.card_expected_retrievability,
        "card_minutes_per_day": metrics.card_minutes_per_day,
        "card_reviews_per_day": metrics.card_reviews_per_day,
        "card_total_reviews": metrics.card_total_reviews,
        "card_total_lapses": metrics.card_total_lapses,
        "card_total_cost_seconds": metrics.card_total_cost_seconds,
        "card_final_projected_retrievability": None,
        "observed_retention": metrics.observed_retention,
        "deck_expected_memorized": metrics.card_expected_retrievability * deck_scale,
        "deck_minutes_per_day": metrics.card_minutes_per_day * deck_scale,
        "deck_reviews_per_day": metrics.card_reviews_per_day * deck_scale,
        "total_reviews": metrics.card_total_reviews * particles,
        "total_lapses": metrics.card_total_lapses * particles,
        "total_cost_seconds": metrics.card_total_cost_seconds * particles,
        "runtime_s": runtime_s,
        "review_markov_transition": False,
        "engine": engine,
        "fuzz": False,
    }


def _append_policy_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
    scheduler: str,
    scheduler_spec: str,
    goal_cost_weights: Sequence[float],
    metrics_by_user: Sequence[Sequence[OracleMetrics]],
    runtime_s: float,
    engine: str,
) -> None:
    runtime_per_group = runtime_s / float(
        max(1, len(user_ids) * len(goal_cost_weights))
    )
    for weight_idx, cost_weight in enumerate(goal_cost_weights):
        for user_idx, user_id in enumerate(user_ids):
            rows.append(
                _metric_row(
                    args,
                    user_id=user_id,
                    scheduler=scheduler,
                    scheduler_spec=scheduler_spec,
                    desired_retention=None,
                    goal_cost_weight=cost_weight,
                    metrics=metrics_by_user[user_idx][weight_idx],
                    runtime_s=runtime_per_group,
                    engine=engine,
                )
            )


def _append_static_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
    action_retentions: Sequence[float],
    metrics_by_user: Sequence[Sequence[OracleMetrics]],
    runtime_s: float,
) -> None:
    runtime_per_group = runtime_s / float(
        max(1, len(user_ids) * len(action_retentions))
    )
    for action_idx, retention in enumerate(action_retentions):
        for user_idx, user_id in enumerate(user_ids):
            rows.append(
                _metric_row(
                    args,
                    user_id=user_id,
                    scheduler=BASELINE_SCHEDULER,
                    scheduler_spec=BASELINE_SCHEDULER,
                    desired_retention=retention,
                    goal_cost_weight=None,
                    metrics=metrics_by_user[user_idx][action_idx],
                    runtime_s=runtime_per_group,
                    engine="stationary_finite_exact_value_static",
                )
            )


@torch.inference_mode()
def _evaluate_static_retention_grid(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    action_retentions: Sequence[float],
) -> tuple[list[list[OracleMetrics]], float]:
    user_count = oracle.user_count
    action_count = len(action_retentions)
    policy = (
        torch.arange(action_count, device=oracle.device, dtype=torch.uint8)
        .view(1, action_count, 1, 1)
        .expand(user_count, action_count, oracle.s_count, oracle.d_count)
        .contiguous()
    )
    cost_weights = torch.zeros(action_count, device=oracle.device, dtype=oracle.dtype)
    start = time.perf_counter()
    metrics = oracle._metrics_from_occupancy_batch(
        policy=policy,
        cost_weights=cost_weights,
    )
    if oracle.device.type == "cuda":
        torch.cuda.synchronize()
    return metrics, time.perf_counter() - start


@torch.inference_mode()
def _evaluate_policy_table(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    policy: torch.Tensor,
    cost_weights: Sequence[float],
) -> tuple[list[list[OracleMetrics]], float]:
    cost_tensor = torch.tensor(cost_weights, device=oracle.device, dtype=oracle.dtype)
    start = time.perf_counter()
    metrics = oracle._metrics_from_occupancy_batch(
        policy=policy.to(device=oracle.device, dtype=torch.uint8),
        cost_weights=cost_tensor,
    )
    if oracle.device.type == "cuda":
        torch.cuda.synchronize()
    return metrics, time.perf_counter() - start


def _parse_optional_csv_floats(
    value: str | None, *, name: str
) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(parse_csv_floats(value, name=name))


def _adr_policy_key(
    spec: FSRS6ADRPolicySpec,
) -> tuple[int, str, str, str]:
    if spec.policy_index is not None:
        return (0, "", "", f"{int(spec.policy_index):012d}")
    if spec.baseline_desired_retention is not None:
        lambda_token = (
            "" if spec.lambda_value is None else format_float(float(spec.lambda_value))
        )
        return (
            1,
            format_float(float(spec.baseline_desired_retention)),
            lambda_token,
            "",
        )
    lambda_token = "" if spec.lambda_value is None else format_float(spec.lambda_value)
    return (2, "", lambda_token, str(spec.path))


def _adr_point_label(spec: FSRS6ADRPolicySpec) -> str:
    if spec.policy_index is not None:
        return f"policy_{spec.policy_index}"
    if spec.baseline_desired_retention is not None:
        label = f"dr_{format_float(spec.baseline_desired_retention)}"
        if spec.lambda_value is not None:
            label += f"_lambda_{format_float(spec.lambda_value)}"
        return label
    return spec.path.parent.name if spec.path.name == "policy.json" else spec.path.stem


def _load_adr_policy_grid(
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
    action_retentions: Sequence[float],
) -> tuple[list[list[FSRS6ADRPolicySpec]], list[list[FSRS6ADRPolicy]]]:
    if (
        args.fsrs6_adr_policy_root is None
        and args.fsrs6_adr_train_run_root is None
        and args.fsrs6_adr_policy_manifest is None
    ):
        return [], []
    lambda_values = _parse_optional_csv_floats(
        args.fsrs6_adr_lambda_values,
        name="--fsrs6-adr-lambda-values",
    )
    try:
        specs = resolve_fsrs6_adr_policy_specs(
            user_ids=user_ids,
            dr_values=action_retentions,
            policy_root=args.fsrs6_adr_policy_root,
            train_run_root=args.fsrs6_adr_train_run_root,
            policy_manifest=args.fsrs6_adr_policy_manifest,
            lambda_values=lambda_values,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    specs_by_user: dict[int, list[FSRS6ADRPolicySpec]] = {
        int(user_id): [] for user_id in user_ids
    }
    for spec in specs:
        specs_by_user.setdefault(int(spec.user_id), []).append(spec)
    missing_users = [user_id for user_id in user_ids if not specs_by_user[user_id]]
    if missing_users:
        raise SystemExit(f"ADR policy source is missing users: {missing_users}")

    ordered_specs: list[list[FSRS6ADRPolicySpec]] = []
    reference_keys: list[tuple[int, str, str, str]] | None = None
    for user_id in user_ids:
        user_specs = sorted(specs_by_user[int(user_id)], key=_adr_policy_key)
        keys = [_adr_policy_key(spec) for spec in user_specs]
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise SystemExit(
                "ADR policy source does not expose the same policy points for "
                f"all users; first mismatch is user {user_id}."
            )
        ordered_specs.append(user_specs)

    policies = [
        [FSRS6ADRPolicy.from_json(spec.path) for spec in user_specs]
        for user_specs in ordered_specs
    ]
    _validate_adr_fixed_policy_grid(policies)
    return ordered_specs, policies


def _validate_adr_fixed_policy_grid(
    policies: Sequence[Sequence[FSRS6ADRPolicy]],
) -> None:
    if not policies or not policies[0]:
        return
    point_count = len(policies[0])
    for user_policies in policies:
        if len(user_policies) != point_count:
            raise ValueError("ADR policy grid must have the same width for all users.")
        for policy in user_policies:
            if policy.feature_version == FEATURE_VERSION_LOG_POLY_TIME:
                raise ValueError(
                    "Exact ADR fixed-policy evaluation currently supports "
                    "stationary ADR policies only; time-dependent ADR needs "
                    "finite-horizon transition tables."
                )
            if policy.feature_version not in {
                FEATURE_VERSION_LOG_POLY,
                FEATURE_VERSION_LOG_LINEAR,
            }:
                raise ValueError(
                    f"Unsupported ADR feature_version: {policy.feature_version}"
                )


def _round_half_up_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.floor(values + 0.5)


def _adr_retention_grid(
    policy: FSRS6ADRPolicy,
    oracle: FSRS6BatchedStationaryFiniteOracle,
) -> torch.Tensor:
    bounds = policy.bounds
    s = torch.clamp(oracle.s_mesh, bounds.s_min, bounds.s_max)
    d = torch.clamp(oracle.d_mesh, bounds.d_min, bounds.d_max)
    log_s_min = math.log(bounds.s_min)
    log_s_span = math.log(bounds.s_max) - log_s_min
    x_s = torch.clamp((torch.log(s) - log_s_min) / log_s_span, 0.0, 1.0)
    x_d = torch.clamp((d - bounds.d_min) / (bounds.d_max - bounds.d_min), 0.0, 1.0)
    if policy.feature_version == FEATURE_VERSION_LOG_POLY:
        features = torch.stack(
            [
                torch.ones_like(x_s),
                x_s,
                x_d,
                x_s * x_d,
                x_s.square(),
                x_d.square(),
            ],
            dim=0,
        )
    elif policy.feature_version == FEATURE_VERSION_LOG_LINEAR:
        features = torch.stack([torch.ones_like(x_s), x_s, x_d], dim=0)
    else:
        raise ValueError(f"Unsupported ADR feature_version: {policy.feature_version}")

    coeffs = torch.tensor(
        policy.coefficients,
        device=oracle.device,
        dtype=oracle.dtype,
    )
    logit = (coeffs.view(-1, 1, 1) * features).sum(dim=0)
    return policy.retention_min + (
        policy.retention_max - policy.retention_min
    ) * torch.sigmoid(logit)


def _adr_transition_tables(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    policies: Sequence[Sequence[FSRS6ADRPolicy]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    user_count = len(policies)
    if user_count != oracle.user_count:
        raise ValueError("ADR policy user count does not match oracle user count.")
    if user_count == 0 or not policies[0]:
        raise ValueError("ADR policy grid must not be empty.")
    point_count = len(policies[0])
    interval = torch.empty(
        (oracle.user_count, point_count, oracle.state_count),
        device=oracle.device,
        dtype=torch.int64,
    )
    for user_idx, user_policies in enumerate(policies):
        for point_idx, policy in enumerate(user_policies):
            retention = _adr_retention_grid(policy, oracle)
            interval_float = (
                oracle.s_mesh
                / oracle.factor[user_idx]
                * (torch.pow(retention, 1.0 / oracle.decay[user_idx]) - 1.0)
            )
            interval[user_idx, point_idx] = (
                torch.clamp(
                    _round_half_up_tensor(interval_float),
                    min=1.0,
                )
                .to(torch.int64)
                .reshape(-1)
            )

    return interval, *_transition_tables_for_interval(oracle, interval)


def _transition_tables_for_interval(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    interval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    user_count, point_count, state_count = interval.shape
    if user_count != oracle.user_count or state_count != oracle.state_count:
        raise ValueError("Interval table shape does not match oracle state shape.")
    elapsed = interval.to(dtype=oracle.dtype)
    s_flat = oracle.s_mesh.reshape(1, 1, oracle.state_count)
    d_flat = oracle.d_mesh.reshape(1, 1, oracle.state_count)
    retrievability = oracle._forgetting_curve(elapsed, s_flat)
    prob = torch.stack(
        [
            1.0 - retrievability,
            retrievability * oracle.review_rating_prob[:, 0].view(user_count, 1, 1),
            retrievability * oracle.review_rating_prob[:, 1].view(user_count, 1, 1),
            retrievability * oracle.review_rating_prob[:, 2].view(user_count, 1, 1),
        ],
        dim=2,
    )
    rating = torch.empty(
        (user_count, point_count, state_count),
        device=oracle.device,
        dtype=torch.int64,
    )
    next_idx: list[torch.Tensor] = []
    next_weight: list[torch.Tensor] = []
    for rating_value in range(1, 5):
        rating.fill_(rating_value)
        if rating_value > 1:
            new_s = oracle._stability_after_success(
                s_flat,
                retrievability,
                d_flat,
                rating,
            )
        else:
            new_s = oracle._stability_after_failure(
                s_flat,
                retrievability,
                d_flat,
            )
        new_d = oracle._next_d(d_flat, rating)
        kernel_idx, kernel_weight = oracle._state_kernel(new_s, new_d)
        next_idx.append(kernel_idx)
        next_weight.append(kernel_weight)
    return (
        prob,
        torch.stack(next_idx, dim=2)
        .permute(1, 3, 2, 0, 4)
        .reshape(
            user_count,
            point_count,
            4,
            4,
            state_count,
        ),
        torch.stack(next_weight, dim=2)
        .permute(1, 3, 2, 0, 4)
        .reshape(
            user_count,
            point_count,
            4,
            4,
            state_count,
        ),
    )


def _rollout_fixed_transition_occupancy(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    interval: torch.Tensor,
    prob: torch.Tensor,
    next_idx: torch.Tensor,
    next_weight: torch.Tensor,
) -> torch.Tensor:
    point_count = int(interval.shape[1])
    occupancy = torch.zeros(
        (
            oracle.user_count,
            point_count,
            oracle.horizon + 1,
            oracle.state_count,
        ),
        device=oracle.device,
        dtype=oracle.dtype,
    )
    batch_offset = (
        torch.arange(
            oracle.user_count * point_count,
            device=oracle.device,
            dtype=torch.int64,
        )
        .view(oracle.user_count, point_count, 1)
        .mul((oracle.horizon + 1) * oracle.state_count)
    )

    for rating_value in range(1, 5):
        initial_prob = oracle.first_rating_prob[:, rating_value - 1]
        s0, d0 = oracle._init_state_scalar(rating_value)
        state_idx, state_weight = oracle._state_kernel(s0, d0)
        for corner_idx in range(4):
            target = (
                batch_offset
                + oracle.horizon * oracle.state_count
                + state_idx[corner_idx].view(oracle.user_count, 1, 1)
            )
            amount = (initial_prob[:, None] * state_weight[corner_idx][:, None]).expand(
                oracle.user_count, point_count
            )
            occupancy.reshape(-1).scatter_add_(
                0,
                target.reshape(-1),
                amount.reshape(-1),
            )

    flat_occupancy = occupancy.reshape(-1)
    for rem in range(oracle.horizon, 0, -1):
        current = occupancy[:, :, rem, :]
        if not bool(current.sum().item()):
            continue
        cont_mask = interval <= rem
        if not bool(cont_mask.any().item()):
            continue
        source = current * cont_mask.to(dtype=oracle.dtype)
        if not bool(source.sum().item()):
            continue
        future_rem = torch.clamp(rem - interval, min=0).to(torch.int64)
        base = batch_offset + future_rem * oracle.state_count
        for rating_idx in range(4):
            for corner_idx in range(4):
                amount = (
                    source
                    * prob[:, :, rating_idx, :]
                    * next_weight[:, :, rating_idx, corner_idx, :]
                )
                target = base + next_idx[:, :, rating_idx, corner_idx, :]
                flat_occupancy.scatter_add_(0, target.reshape(-1), amount.reshape(-1))

    return occupancy


def _metrics_from_fixed_transition_tables(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    interval: torch.Tensor,
    prob: torch.Tensor,
    next_idx: torch.Tensor,
    next_weight: torch.Tensor,
    cost_weights: torch.Tensor,
) -> list[list[OracleMetrics]]:
    occupancy = _rollout_fixed_transition_occupancy(
        oracle,
        interval=interval,
        prob=prob,
        next_idx=next_idx,
        next_weight=next_weight,
    )
    point_count = int(interval.shape[1])
    total_mem = torch.zeros(
        (oracle.user_count, point_count),
        device=oracle.device,
        dtype=oracle.dtype,
    )
    total_minutes = (
        (oracle.first_rating_prob * oracle.learning_cost_minutes)
        .sum(dim=1)[:, None]
        .expand_as(total_mem)
        .clone()
    )
    total_reviews = torch.zeros_like(total_mem)
    total_lapses = torch.zeros_like(total_mem)
    for rem in range(1, oracle.horizon + 1):
        current = occupancy[:, :, rem, :]
        if not bool(current.sum().item()):
            continue
        active_days = torch.minimum(interval, torch.full_like(interval, rem))
        total_mem += (current * oracle._memorized_sum_batch(active_days)).sum(dim=2)
        cont_mask = interval <= rem
        source = current * cont_mask.to(dtype=oracle.dtype)
        if not bool(source.sum().item()):
            continue
        expected_review_minutes = (
            prob * oracle.review_cost_minutes[:, None, :, None]
        ).sum(dim=2)
        total_minutes += (source * expected_review_minutes).sum(dim=2)
        total_reviews += source.sum(dim=2)
        total_lapses += (source * prob[:, :, 0, :]).sum(dim=2)

    day_count = float(oracle.days)
    objectives = total_mem / day_count - cost_weights.view(1, -1) * (
        total_minutes / day_count
    )
    metrics: list[list[OracleMetrics]] = []
    for user_idx in range(oracle.user_count):
        row: list[OracleMetrics] = []
        for point_idx in range(point_count):
            reviews_float = float(total_reviews[user_idx, point_idx].item())
            lapses_float = float(total_lapses[user_idx, point_idx].item())
            observed_retention = (
                1.0 - lapses_float / reviews_float if reviews_float > 0.0 else None
            )
            row.append(
                OracleMetrics(
                    card_expected_retrievability=float(
                        total_mem[user_idx, point_idx].item() / day_count
                    ),
                    card_minutes_per_day=float(
                        total_minutes[user_idx, point_idx].item() / day_count
                    ),
                    card_reviews_per_day=float(
                        total_reviews[user_idx, point_idx].item() / day_count
                    ),
                    card_total_reviews=reviews_float,
                    card_total_lapses=lapses_float,
                    card_total_cost_seconds=float(
                        total_minutes[user_idx, point_idx].item() * 60.0
                    ),
                    observed_retention=observed_retention,
                    scalar_objective=float(objectives[user_idx, point_idx].item()),
                    runtime_s=0.0,
                )
            )
        metrics.append(row)
    return metrics


@torch.inference_mode()
def _evaluate_adr_fixed_policies(
    oracle: FSRS6BatchedStationaryFiniteOracle,
    *,
    policies: Sequence[Sequence[FSRS6ADRPolicy]],
) -> tuple[list[list[OracleMetrics]], float]:
    start = time.perf_counter()
    interval, prob, next_idx, next_weight = _adr_transition_tables(
        oracle,
        policies=policies,
    )
    cost_weights = torch.zeros(
        int(interval.shape[1]),
        device=oracle.device,
        dtype=oracle.dtype,
    )
    metrics = _metrics_from_fixed_transition_tables(
        oracle,
        interval=interval,
        prob=prob,
        next_idx=next_idx,
        next_weight=next_weight,
        cost_weights=cost_weights,
    )
    if oracle.device.type == "cuda":
        torch.cuda.synchronize()
    return metrics, time.perf_counter() - start


def _append_adr_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    user_ids: Sequence[int],
    specs: Sequence[Sequence[FSRS6ADRPolicySpec]],
    policies: Sequence[Sequence[FSRS6ADRPolicy]],
    metrics_by_user: Sequence[Sequence[OracleMetrics]],
    runtime_s: float,
) -> None:
    point_count = len(specs[0])
    runtime_per_group = runtime_s / float(max(1, len(user_ids) * point_count))
    for point_idx in range(point_count):
        for user_idx, user_id in enumerate(user_ids):
            spec = specs[user_idx][point_idx]
            policy = policies[user_idx][point_idx]
            row = _metric_row(
                args,
                user_id=user_id,
                scheduler=ADR_SCHEDULER,
                scheduler_spec=ADR_SCHEDULER,
                desired_retention=None,
                goal_cost_weight=None,
                metrics=metrics_by_user[user_idx][point_idx],
                runtime_s=runtime_per_group,
                engine="stationary_finite_exact_value_adr_fixed_policy",
            )
            row.update(
                {
                    "review_markov_transition": False,
                    "fsrs6_adr_policy": str(spec.path),
                    "fsrs6_adr_baseline_desired_retention": (
                        spec.baseline_desired_retention
                    ),
                    "fsrs6_adr_lambda_value": spec.lambda_value,
                    "fsrs6_adr_policy_index": spec.policy_index,
                    "fsrs6_adr_policy_title": policy.title,
                    "fsrs6_adr_feature_version": policy.feature_version,
                    "fsrs6_adr_point_label": _adr_point_label(spec),
                }
            )
            rows.append(row)


@torch.inference_mode()
def _distill_policy_table(
    *,
    args: argparse.Namespace,
    label: str,
    distill_dir: Path,
    user_ids: Sequence[int],
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    oracle: FSRS6BatchedStationaryFiniteOracle,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    ensemble, checkpoint_actions, checkpoint_cost_weights = (
        load_per_user_distill_ensemble(
            distill_dir=distill_dir,
            user_ids=user_ids,
            device=device,
        )
    )
    if [float(value) for value in checkpoint_actions] != [
        float(value) for value in action_retentions
    ]:
        raise ValueError(f"Action grid mismatch for {label}: {distill_dir}")

    model_dtype = next(iter(ensemble.params.values())).dtype
    s_norm = torch.linspace(0.0, 1.0, oracle.s_count, device=device, dtype=model_dtype)
    d_norm = torch.linspace(0.0, 1.0, oracle.d_count, device=device, dtype=model_dtype)
    s_grid, d_grid = torch.meshgrid(s_norm, d_norm, indexing="ij")
    state_features = torch.stack([s_grid.reshape(-1), d_grid.reshape(-1)], dim=1)
    goal_norm_max = max(1.0, max(checkpoint_cost_weights))
    goal_norm = torch.log1p(
        torch.tensor(cost_weights, device=device, dtype=model_dtype)
    ) / math.log1p(goal_norm_max)
    obs_by_weight = []
    for goal_value in goal_norm:
        obs_by_weight.append(
            torch.column_stack(
                [
                    state_features,
                    torch.full(
                        (state_features.shape[0],),
                        float(goal_value.item()),
                        device=device,
                        dtype=model_dtype,
                    ),
                ]
            )
        )
    obs = torch.cat(obs_by_weight, dim=0)
    obs = obs[None, :, :].expand(len(user_ids), -1, -1).contiguous()
    logits, _ = batched_ensemble_forward(ensemble, obs)
    action = torch.argmax(logits, dim=2)
    policy = action.reshape(
        len(user_ids),
        len(cost_weights),
        oracle.s_count,
        oracle.d_count,
    ).to(dtype=torch.uint8)
    return policy, {
        "label": label,
        "path": str(distill_dir),
        "type": "distill",
        "scheduler_spec": _distill_scheduler_spec(label),
        "action_retentions": list(checkpoint_actions),
        "train_cost_weights": list(checkpoint_cost_weights),
        "params_per_user": ensemble.params_per_user,
    }


@torch.inference_mode()
def _direct_policy_table(
    *,
    label: str,
    policy_path: Path,
    user_ids: Sequence[int],
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    oracle: FSRS6BatchedStationaryFiniteOracle,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != DIRECT_POLICY_SCHEDULER:
        raise ValueError(f"Unexpected policy_type in {policy_path}.")
    checkpoint_user_ids = [int(value) for value in checkpoint["user_ids"]]
    theta_source = checkpoint["theta_by_user"].to(device=device, dtype=torch.float64)
    theta_rows = []
    for user_id in user_ids:
        try:
            source_idx = checkpoint_user_ids.index(int(user_id))
        except ValueError as exc:
            raise ValueError(f"Missing user {user_id} in {policy_path}.") from exc
        theta_rows.append(theta_source[source_idx])
    theta = torch.stack(theta_rows, dim=0)
    policy_family = str(checkpoint.get("policy_family", "bilinear_monotone"))
    min_retention = float(checkpoint.get("min_retention", min(action_retentions)))
    max_retention = float(checkpoint.get("max_retention", max(action_retentions)))
    goal_norm_max = max(
        1.0,
        max(
            float(value) for value in checkpoint.get("eval_cost_weights", cost_weights)
        ),
    )

    s_norm = torch.linspace(
        0.0, 1.0, oracle.s_count, device=device, dtype=torch.float64
    )
    d_norm = torch.linspace(
        0.0, 1.0, oracle.d_count, device=device, dtype=torch.float64
    )
    s_grid, d_grid = torch.meshgrid(s_norm, d_norm, indexing="ij")
    state_features = torch.stack([s_grid.reshape(-1), d_grid.reshape(-1)], dim=1)
    action_grid = torch.tensor(action_retentions, device=device, dtype=torch.float64)
    policy_by_weight = []
    for cost_weight in cost_weights:
        goal_norm = math.log1p(float(cost_weight)) / math.log1p(goal_norm_max)
        obs = torch.column_stack(
            [
                state_features,
                torch.full(
                    (state_features.shape[0],),
                    goal_norm,
                    device=device,
                    dtype=torch.float64,
                ),
            ]
        )
        obs = obs[None, :, :].expand(len(user_ids), -1, -1).reshape(-1, 3)
        selected_theta = (
            theta[:, None, :]
            .expand(len(user_ids), state_features.shape[0], theta.shape[1])
            .reshape(-1, theta.shape[1])
        )
        retention = direct_policy_retention(
            selected_theta,
            obs,
            policy_family=policy_family,
            min_retention=min_retention,
            max_retention=max_retention,
        )
        action = torch.argmin(
            torch.abs(retention[:, None] - action_grid[None, :]), dim=1
        )
        policy_by_weight.append(
            action.reshape(len(user_ids), oracle.s_count, oracle.d_count)
        )
    policy = torch.stack(policy_by_weight, dim=1).to(dtype=torch.uint8)
    return policy, {
        "label": label,
        "path": str(policy_path),
        "type": "direct",
        "scheduler_spec": _direct_scheduler_spec(label),
        "policy_family": policy_family,
        "params_per_user": int(checkpoint["params_per_user"]),
    }


def _distill_scheduler_spec(label: str) -> str:
    if label == "distill_476":
        return PER_USER_SCHEDULER
    suffix = label.removeprefix("distill_")
    return f"{PER_USER_SCHEDULER}_{suffix}"


def _direct_scheduler_spec(label: str) -> str:
    return f"{DIRECT_POLICY_SCHEDULER}_{label}"


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    user_ids: Sequence[int],
    device: torch.device,
    action_retentions: Sequence[float],
    cost_weights: Sequence[float],
    policy_metadata: Sequence[Mapping[str, Any]],
    mean_rows: Sequence[Mapping[str, Any]],
    total_runtime_s: float,
) -> None:
    metadata = {
        "experiment": "stationary_finite_exact_value_eval",
        "device": str(device),
        "user_ids": list(user_ids),
        "days": args.days,
        "deck_scale": args.deck_scale,
        "oracle_s_grid_size": args.oracle_s_grid_size,
        "oracle_d_grid_size": args.oracle_d_grid_size,
        "action_retentions": list(action_retentions),
        "cost_weights": list(cost_weights),
        "policies": list(policy_metadata),
        "mean_summary": list(mean_rows),
        "total_runtime_s": total_runtime_s,
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    user_ids = parse_user_ids(args.user_ids)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    validate_retention_values(action_retentions, name="--action-retentions")
    device = resolve_torch_device(args.torch_device)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    configs = load_user_configs(args, user_ids)
    distill_specs = _policy_specs(
        args.distill_policy,
        DEFAULT_DISTILL_POLICIES,
        skip_defaults=args.skip_default_policies,
    )
    direct_specs = _policy_specs(
        args.direct_policy,
        DEFAULT_DIRECT_POLICIES,
        skip_defaults=args.skip_default_policies,
    )
    adr_specs, adr_policies = _load_adr_policy_grid(
        args,
        user_ids=user_ids,
        action_retentions=action_retentions,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    policy_metadata: list[dict[str, Any]] = []
    start = time.perf_counter()

    oracle = _build_oracle(
        args,
        configs=configs,
        action_retentions=action_retentions,
        device=device,
    )
    static_metrics, static_runtime_s = _evaluate_static_retention_grid(
        oracle,
        action_retentions=action_retentions,
    )
    _append_static_rows(
        rows,
        args,
        user_ids=user_ids,
        action_retentions=action_retentions,
        metrics_by_user=static_metrics,
        runtime_s=static_runtime_s,
    )

    exact_solution = oracle.solve_stationary_finite_policies(
        cost_weights,
        max_iterations=args.oracle_stationary_finite_max_iterations,
        tolerance=args.oracle_stationary_finite_tolerance,
        progress=not args.no_progress,
    )
    _append_policy_rows(
        rows,
        args,
        user_ids=user_ids,
        scheduler=EXACT_SCHEDULER,
        scheduler_spec=EXACT_SCHEDULER,
        goal_cost_weights=cost_weights,
        metrics_by_user=exact_solution.metrics,
        runtime_s=exact_solution.runtime_s,
        engine="stationary_finite_exact_value_teacher",
    )
    policy_metadata.append(
        {
            "label": "exact",
            "type": "exact",
            "scheduler_spec": EXACT_SCHEDULER,
            "runtime_s": exact_solution.runtime_s,
            "converged": exact_solution.converged,
            "iterations": exact_solution.iterations,
            "residuals": exact_solution.residuals,
        }
    )

    for label, distill_dir in distill_specs:
        policy, metadata = _distill_policy_table(
            args=args,
            label=label,
            distill_dir=distill_dir,
            user_ids=user_ids,
            action_retentions=action_retentions,
            cost_weights=cost_weights,
            oracle=oracle,
            device=device,
        )
        metrics, runtime_s = _evaluate_policy_table(
            oracle,
            policy=policy,
            cost_weights=cost_weights,
        )
        _append_policy_rows(
            rows,
            args,
            user_ids=user_ids,
            scheduler=metadata["scheduler_spec"],
            scheduler_spec=metadata["scheduler_spec"],
            goal_cost_weights=cost_weights,
            metrics_by_user=metrics,
            runtime_s=runtime_s,
            engine="stationary_finite_exact_value_distill_table",
        )
        metadata["runtime_s"] = runtime_s
        policy_metadata.append(metadata)

    if adr_specs:
        adr_metrics, adr_runtime_s = _evaluate_adr_fixed_policies(
            oracle,
            policies=adr_policies,
        )
        _append_adr_rows(
            rows,
            args,
            user_ids=user_ids,
            specs=adr_specs,
            policies=adr_policies,
            metrics_by_user=adr_metrics,
            runtime_s=adr_runtime_s,
        )
        policy_metadata.append(
            {
                "label": "fsrs6_adr",
                "type": "fsrs6_adr",
                "scheduler_spec": ADR_SCHEDULER,
                "engine": "stationary_finite_exact_value_adr_fixed_policy",
                "runtime_s": adr_runtime_s,
                "policy_count_per_user": len(adr_specs[0]),
                "policy_paths_by_user": [
                    [str(spec.path) for spec in user_specs] for user_specs in adr_specs
                ],
            }
        )

    for label, policy_path in direct_specs:
        policy, metadata = _direct_policy_table(
            label=label,
            policy_path=policy_path,
            user_ids=user_ids,
            action_retentions=action_retentions,
            cost_weights=cost_weights,
            oracle=oracle,
            device=device,
        )
        metrics, runtime_s = _evaluate_policy_table(
            oracle,
            policy=policy,
            cost_weights=cost_weights,
        )
        _append_policy_rows(
            rows,
            args,
            user_ids=user_ids,
            scheduler=metadata["scheduler_spec"],
            scheduler_spec=metadata["scheduler_spec"],
            goal_cost_weights=cost_weights,
            metrics_by_user=metrics,
            runtime_s=runtime_s,
            engine="stationary_finite_exact_value_direct_table",
        )
        metadata["runtime_s"] = runtime_s
        policy_metadata.append(metadata)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_runtime_s = time.perf_counter() - start

    results_path = args.out_dir / "results.csv"
    regret_path = args.out_dir / "regret_auc.csv"
    summary_path = args.out_dir / "summary.csv"
    mean_summary_path = args.out_dir / "mean_summary.csv"
    metadata_path = args.out_dir / "metadata.json"

    _write_csv(results_path, rows)
    auc_rows = _build_regret_auc_rows(rows)
    _write_regret_auc_csv(regret_path, auc_rows)
    scheduler_specs = [
        EXACT_SCHEDULER,
        *(
            metadata["scheduler_spec"]
            for metadata in policy_metadata
            if metadata["type"] != "exact"
        ),
    ]
    write_auc_summary(
        summary_path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER, EXACT_SCHEDULER],
        schedulers=scheduler_specs,
        fieldnames=DETAILED_AUC_FIELDS,
    )
    mean_rows = write_mean_auc_summary(
        mean_summary_path,
        auc_rows,
        baselines=[BASELINE_SCHEDULER, EXACT_SCHEDULER],
        schedulers=scheduler_specs,
        include_baseline_scheduler=True,
    )
    _write_metadata(
        metadata_path,
        args=args,
        user_ids=user_ids,
        device=device,
        action_retentions=action_retentions,
        cost_weights=cost_weights,
        policy_metadata=policy_metadata,
        mean_rows=mean_rows,
        total_runtime_s=total_runtime_s,
    )

    print(f"Wrote exact-value results: {results_path}")
    print(f"Wrote exact-value same-target time saved AUC: {regret_path}")
    print(f"Wrote exact-value summary: {summary_path}")
    print(f"Wrote exact-value mean summary: {mean_summary_path}")
    print(f"Wrote exact-value metadata: {metadata_path}")
    for row in mean_rows:
        if row["baseline_scheduler"] != BASELINE_SCHEDULER:
            continue
        rel = row["mean_relative_same_target_time_saved_auc_percent"]
        coverage = row["mean_span_coverage_percent"]
        print(
            f"{row['scheduler']} vs fsrs6: "
            f"relative_time_saved={float(rel):.2f}% coverage={float(coverage):.2f}%"
        )


if __name__ == "__main__":
    main()
