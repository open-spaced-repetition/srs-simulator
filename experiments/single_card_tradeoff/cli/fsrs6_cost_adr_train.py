from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (  # noqa: E402
    resolve_torch_device,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    DEFAULT_USER_IDS,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
    single_card_runtime_context_from_args,
)
from experiments.single_card_tradeoff.core.defaults import (  # noqa: E402
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.models.policy_runtime import (  # noqa: E402
    RetentionDistillNet,
    predicted_retentions,
)
from experiments.single_card_tradeoff.oracles import (  # noqa: E402
    FSRS6ContinuousStationaryFiniteOracle,
    retention_interval_float,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402
from simulator.fsrs_defaults import resolve_fsrs6_weights  # noqa: E402
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FEATURE_VERSION_INTERVAL_MONO,
    FEATURE_VERSION_RETENTION_MONO,
    FSRS6CostConditionedADRPolicy,
    STATE_FEATURE_COUNT_COMPACT,
    STATE_FEATURE_COUNT_HINGE,
)
from simulator.math.fsrs import Bounds  # noqa: E402
from simulator.scheduler_spec import format_float  # noqa: E402


DEFAULT_TRAIN_COST_WEIGHTS = [0, 4, 16, 64, 256, 1024]
DEFAULT_OUT_DIR = Path("artifacts/single_card_tradeoff/fsrs6_cost_adr_train")
CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY_TYPE = (
    "fsrs6_oracle_continuous_stationary_finite_distill"
)


@dataclass(frozen=True)
class TeacherTable:
    s_grid: torch.Tensor
    d_grid: torch.Tensor
    policy: torch.Tensor
    factor: torch.Tensor
    decay: torch.Tensor
    retention_min: float
    retention_max: float
    bounds: Bounds
    source: str
    source_path: Path | None = None


def parse_csv_floats(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid {name} value {item!r}.") from exc
        if not math.isfinite(value):
            raise SystemExit(f"{name} values must be finite.")
        values.append(value)
    if not values:
        raise SystemExit(f"{name} must include at least one value.")
    return values


def parse_user_ids(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        raw = "1"
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("--user-ids must contain at least one user.")
    if any(value <= 0 for value in values):
        raise SystemExit("--user-ids must be positive.")
    if len(set(values)) != len(values):
        raise SystemExit("--user-ids must not contain duplicates.")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit compact cost-weight-conditioned FSRS6 ADR policies from the "
            "continuous stationary finite single-card teacher."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_TRAIN_COST_WEIGHTS),
        help="Teacher cost weights used for fitting.",
    )
    parser.add_argument(
        "--eval-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
        help="Recorded in metadata for downstream tradeoff evaluation.",
    )
    parser.add_argument(
        "--action-head",
        choices=(ACTION_HEAD_INTERVAL, ACTION_HEAD_RETENTION),
        default=ACTION_HEAD_RETENTION,
    )
    parser.add_argument(
        "--state-feature-count",
        choices=(STATE_FEATURE_COUNT_COMPACT, STATE_FEATURE_COUNT_HINGE),
        type=int,
        default=STATE_FEATURE_COUNT_HINGE,
    )
    parser.add_argument("--epochs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--oracle-s-grid-size", type=int, default=64)
    parser.add_argument("--oracle-d-grid-size", type=int, default=32)
    parser.add_argument("--oracle-interval-chunk-size", type=int, default=64)
    parser.add_argument("--retention-min", type=float, default=0.5)
    parser.add_argument("--retention-max", type=float, default=0.98)
    parser.add_argument("--oracle-max-iterations", type=int, default=128)
    parser.add_argument("--oracle-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--teacher-continuous-stationary-finite-distill-policy",
        type=Path,
        default=None,
        help=(
            "Optional per-user FSRS6 continuous stationary finite distill "
            "checkpoint to use as the supervised teacher instead of solving "
            "the exact continuous stationary finite oracle."
        ),
    )
    parser.add_argument(
        "--teacher-continuous-stationary-finite-distill-policy-template",
        default=None,
        help=(
            "Optional per-user teacher checkpoint template containing {user_id}. "
            "When provided, it takes precedence over the single teacher policy "
            "path."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_run_monitoring_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.env not in {"fsrs6", "fsrs6_default"}:
        raise SystemExit("fsrs6_cost_adr_train requires --env fsrs6 or fsrs6_default.")
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0.")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be > 0.")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be >= 0.")
    if args.max_grad_norm <= 0.0:
        raise SystemExit("--max-grad-norm must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")
    if not (0.0 < args.retention_min < args.retention_max < 1.0):
        raise SystemExit("--retention-min/max must satisfy 0 < min < max < 1.")
    if args.oracle_max_iterations <= 0 or args.oracle_tolerance <= 0.0:
        raise SystemExit("--oracle-max-iterations and --oracle-tolerance must be > 0.")
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(value < 0.0 for value in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    eval_weights = parse_csv_floats(
        args.eval_cost_weights,
        name="--eval-cost-weights",
    )
    if any(value < 0.0 for value in eval_weights):
        raise SystemExit("--eval-cost-weights must be >= 0.")


def _tensor_state_features(
    *,
    s_grid: torch.Tensor,
    d_grid: torch.Tensor,
    state_feature_count: int,
) -> torch.Tensor:
    s_norm = torch.linspace(
        0.0,
        1.0,
        steps=int(s_grid.numel()),
        device=s_grid.device,
        dtype=torch.float64,
    )
    d_norm = torch.linspace(
        0.0,
        1.0,
        steps=int(d_grid.numel()),
        device=d_grid.device,
        dtype=torch.float64,
    )
    s = s_norm[:, None].expand(s_norm.numel(), d_norm.numel())
    d = d_norm[None, :].expand(s_norm.numel(), d_norm.numel())
    features = [
        torch.ones_like(s),
        s,
        d,
        s * d,
        s * s,
        d * d,
    ]
    if state_feature_count == STATE_FEATURE_COUNT_HINGE:
        features.extend([torch.clamp(s - 0.5, min=0.0), torch.clamp(d - 0.5, min=0.0)])
    return torch.stack(features, dim=2).reshape(-1, state_feature_count)


def _cost_features(
    cost_weights: Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.tensor(list(cost_weights), device=device, dtype=dtype)
    max_weight = max(1.0, max(float(value) for value in cost_weights))
    z = torch.log1p(weights) / math.log1p(max_weight)
    z = torch.clamp(z, 0.0, 1.0)
    return torch.sqrt(z), z, z * z


def _has_continuous_distill_teacher(args: argparse.Namespace) -> bool:
    template = getattr(
        args,
        "teacher_continuous_stationary_finite_distill_policy_template",
        None,
    )
    policy = getattr(args, "teacher_continuous_stationary_finite_distill_policy", None)
    return bool(template is not None and str(template).strip()) or policy is not None


def _resolve_continuous_distill_teacher_policy_path(
    args: argparse.Namespace,
    *,
    user_id: int,
) -> Path:
    template = getattr(
        args,
        "teacher_continuous_stationary_finite_distill_policy_template",
        None,
    )
    if template is not None and str(template).strip():
        if "{user_id}" not in str(template):
            raise SystemExit(
                "--teacher-continuous-stationary-finite-distill-policy-template "
                "must contain {user_id}."
            )
        return Path(str(template).format(user_id=int(user_id))).expanduser()
    policy = getattr(args, "teacher_continuous_stationary_finite_distill_policy", None)
    if policy is None:
        raise SystemExit(
            "Missing --teacher-continuous-stationary-finite-distill-policy."
        )
    return Path(policy)


def _load_continuous_distill_checkpoint(
    policy_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], RetentionDistillNet]:
    if not policy_path.exists():
        raise SystemExit(f"Teacher distill policy not found: {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Invalid teacher distill checkpoint: {policy_path}")
    if (
        checkpoint.get("policy_type")
        != CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY_TYPE
    ):
        raise SystemExit(
            "Teacher checkpoint must have policy_type="
            f"{CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY_TYPE!r}."
        )
    if checkpoint.get("action_mode") != ACTION_HEAD_RETENTION:
        raise SystemExit("Teacher checkpoint must use action_mode='desired_retention'.")
    if str(checkpoint.get("obs_mode", "oracle_stationary")) != "oracle_stationary":
        raise SystemExit("Teacher checkpoint must use obs_mode='oracle_stationary'.")
    raw_action_retentions = checkpoint.get("action_retentions", [0.5, 0.98])
    if not isinstance(raw_action_retentions, list) or not raw_action_retentions:
        raise SystemExit("Teacher checkpoint has invalid action_retentions.")
    obs_dim = int(checkpoint.get("obs_dim", 3))
    if obs_dim != 3:
        raise SystemExit("Teacher checkpoint must have obs_dim=3.")
    model = RetentionDistillNet(
        obs_dim=obs_dim,
        hidden_size=int(checkpoint.get("hidden_size", 8)),
        action_count=len(raw_action_retentions),
        architecture=str(checkpoint.get("network", "residual")),
        depth=int(checkpoint.get("network_depth", 2)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit("Teacher checkpoint is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return checkpoint, model


@torch.inference_mode()
def _continuous_distill_teacher_table(
    *,
    policy_path: Path,
    cost_weights: Sequence[float],
    s_grid_size: int,
    d_grid_size: int,
    device: torch.device,
) -> TeacherTable:
    checkpoint, model = _load_continuous_distill_checkpoint(
        policy_path,
        device=device,
    )
    bounds = Bounds()
    dtype = torch.float64
    s_grid = torch.exp(
        torch.linspace(
            math.log(bounds.s_min),
            math.log(bounds.s_max),
            steps=s_grid_size,
            device=device,
            dtype=dtype,
        )
    )
    d_grid = torch.linspace(
        bounds.d_min,
        bounds.d_max,
        steps=d_grid_size,
        device=device,
        dtype=dtype,
    )
    s_norm = torch.linspace(0.0, 1.0, steps=s_grid_size, device=device, dtype=dtype)
    d_norm = torch.linspace(0.0, 1.0, steps=d_grid_size, device=device, dtype=dtype)
    s_feature = s_norm[:, None].expand(s_grid_size, d_grid_size).reshape(-1)
    d_feature = d_norm[None, :].expand(s_grid_size, d_grid_size).reshape(-1)
    raw_policy_cost_weights = checkpoint.get("cost_weights")
    if not isinstance(raw_policy_cost_weights, list) or not raw_policy_cost_weights:
        raise SystemExit("Teacher checkpoint is missing cost_weights.")
    goal_norm_max = max(1.0, max(float(value) for value in raw_policy_cost_weights))
    model_dtype = next(model.parameters()).dtype
    retentions: list[torch.Tensor] = []
    for cost_weight in cost_weights:
        goal_norm = math.log1p(float(cost_weight)) / math.log1p(goal_norm_max)
        obs = torch.stack(
            [
                s_feature,
                d_feature,
                torch.full_like(s_feature, goal_norm),
            ],
            dim=1,
        ).to(dtype=model_dtype)
        raw_retention, _ = model(obs)
        retentions.append(
            predicted_retentions(
                raw_retention.to(dtype=dtype),
                retention_min=float(checkpoint.get("retention_min", 0.5)),
                retention_max=float(checkpoint.get("retention_max", 0.98)),
            ).reshape(s_grid_size, d_grid_size)
        )
    fsrs_weights = resolve_fsrs6_weights(checkpoint.get("fsrs_weights"))
    decay = torch.tensor(-float(fsrs_weights[20]), device=device, dtype=dtype)
    factor = (
        torch.pow(
            torch.tensor(0.9, device=device, dtype=dtype),
            1.0 / decay,
        )
        - 1.0
    )
    return TeacherTable(
        s_grid=s_grid,
        d_grid=d_grid,
        policy=torch.stack(retentions, dim=0),
        factor=factor,
        decay=decay,
        retention_min=float(checkpoint.get("retention_min", 0.5)),
        retention_max=float(checkpoint.get("retention_max", 0.98)),
        bounds=bounds,
        source=CONTINUOUS_STATIONARY_FINITE_DISTILL_POLICY_TYPE,
        source_path=policy_path,
    )


def _predict(
    theta: torch.Tensor,
    *,
    features: torch.Tensor,
    sqrt_z: torch.Tensor,
    z: torch.Tensor,
    z2: torch.Tensor,
    action_head: str,
) -> torch.Tensor:
    feature_count = int(features.shape[1])
    groups = theta.reshape(4, feature_count)
    base = features @ groups[0]
    slope_1 = torch.nn.functional.softplus(features @ groups[1])
    slope_2 = torch.nn.functional.softplus(features @ groups[2])
    slope_3 = torch.nn.functional.softplus(features @ groups[3])
    cost_effect = (
        sqrt_z[:, None] * slope_1[None, :]
        + z[:, None] * slope_2[None, :]
        + z2[:, None] * slope_3[None, :]
    )
    if action_head == ACTION_HEAD_RETENTION:
        return base[None, :] - cost_effect
    return base[None, :] + cost_effect


def _initial_theta(
    *,
    target: torch.Tensor,
    state_feature_count: int,
    action_head: str,
) -> torch.Tensor:
    theta = torch.zeros(
        4, state_feature_count, dtype=torch.float64, device=target.device
    )
    theta[0, 0] = torch.mean(target[0]).item()
    theta[1:, 0] = -6.0
    if action_head == ACTION_HEAD_RETENTION:
        theta[0, 0] = torch.median(target[0]).item()
    return theta.reshape(-1).clone().detach().requires_grad_(True)


def _retention_logits(
    retention: torch.Tensor,
    *,
    retention_min: float,
    retention_max: float,
) -> torch.Tensor:
    ratio = (retention - retention_min) / (retention_max - retention_min)
    ratio = torch.clamp(ratio, 1e-6, 1.0 - 1e-6)
    return torch.logit(ratio)


def fit_policy_from_table(
    *,
    table: TeacherTable,
    cost_weights: Sequence[float],
    action_head: str,
    state_feature_count: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> tuple[FSRS6CostConditionedADRPolicy, dict[str, float]]:
    device = table.policy.device
    features = _tensor_state_features(
        s_grid=table.s_grid,
        d_grid=table.d_grid,
        state_feature_count=state_feature_count,
    ).to(device=device, dtype=torch.float64)
    sqrt_z, z, z2 = _cost_features(
        cost_weights,
        device=device,
        dtype=torch.float64,
    )
    if action_head == ACTION_HEAD_INTERVAL:
        s = table.s_grid[:, None].expand(table.s_grid.numel(), table.d_grid.numel())
        interval = retention_interval_float(
            s=s.reshape(-1),
            retention=table.policy.reshape(len(cost_weights), -1),
            factor=table.factor,
            decay=table.decay,
        )
        target = torch.log(torch.clamp(interval, min=1.0))
    else:
        target = _retention_logits(
            table.policy.reshape(len(cost_weights), -1),
            retention_min=table.retention_min,
            retention_max=table.retention_max,
        )
    theta = _initial_theta(
        target=target,
        state_feature_count=state_feature_count,
        action_head=action_head,
    )
    optimizer = torch.optim.AdamW([theta], lr=learning_rate, weight_decay=weight_decay)
    start = time.perf_counter()
    loss = torch.tensor(float("nan"), device=device, dtype=torch.float64)
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = _predict(
            theta,
            features=features,
            sqrt_z=sqrt_z,
            z=z,
            z2=z2,
            action_head=action_head,
        )
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=max_grad_norm)
        optimizer.step()
    runtime_s = time.perf_counter() - start
    with torch.inference_mode():
        pred = _predict(
            theta,
            features=features,
            sqrt_z=sqrt_z,
            z=z,
            z2=z2,
            action_head=action_head,
        )
        abs_error = torch.abs(pred - target)
        final_loss = float(torch.nn.functional.smooth_l1_loss(pred, target).item())
        mean_abs_error = float(torch.mean(abs_error).item())
        p95_abs_error = float(torch.quantile(abs_error.reshape(-1), 0.95).item())
    coefficients = tuple(float(value) for value in theta.detach().cpu().tolist())
    if action_head == ACTION_HEAD_INTERVAL:
        feature_version = FEATURE_VERSION_INTERVAL_MONO
        policy_action_head = ACTION_HEAD_INTERVAL
    else:
        feature_version = FEATURE_VERSION_RETENTION_MONO
        policy_action_head = ACTION_HEAD_RETENTION
    policy = FSRS6CostConditionedADRPolicy(
        coefficients=coefficients,
        action_head=policy_action_head,
        feature_version=feature_version,
        cost_weight_min=0.0,
        cost_weight_max=max(1.0, max(float(value) for value in cost_weights)),
        retention_min=table.retention_min,
        retention_max=table.retention_max,
        bounds=table.bounds,
        title=f"FSRS6 cost-conditioned ADR fit from {table.source}",
    )
    return policy, {
        "final_loss": final_loss,
        "mean_abs_error": mean_abs_error,
        "p95_abs_error": p95_abs_error,
        "fit_runtime_s": runtime_s,
    }


def fit_policy(
    *,
    oracle: FSRS6ContinuousStationaryFiniteOracle,
    teacher_policy: torch.Tensor,
    cost_weights: Sequence[float],
    action_head: str,
    state_feature_count: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
) -> tuple[FSRS6CostConditionedADRPolicy, dict[str, float]]:
    return fit_policy_from_table(
        table=TeacherTable(
            s_grid=oracle.s_grid,
            d_grid=oracle.d_grid,
            policy=teacher_policy,
            factor=oracle.factor,
            decay=oracle.decay,
            retention_min=oracle.retention_min,
            retention_max=oracle.retention_max,
            bounds=oracle.bounds,
            source="fsrs6_oracle_continuous_stationary_finite",
        ),
        cost_weights=cost_weights,
        action_head=action_head,
        state_feature_count=state_feature_count,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_from_args(args: argparse.Namespace) -> None:
    validate_args(args)
    user_ids = parse_user_ids(args.user_ids)
    use_continuous_distill_teacher = _has_continuous_distill_teacher(args)
    if use_continuous_distill_teacher and len(user_ids) > 1:
        template = getattr(
            args,
            "teacher_continuous_stationary_finite_distill_policy_template",
            None,
        )
        if template is None or not str(template).strip():
            raise SystemExit(
                "Multi-user distill-teacher training requires "
                "--teacher-continuous-stationary-finite-distill-policy-template."
            )
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    eval_cost_weights = parse_csv_floats(
        args.eval_cost_weights,
        name="--eval-cost-weights",
    )
    device = resolve_torch_device(args.torch_device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name="fsrs6_cost_adr_train",
    )
    summary_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for user_id in user_ids:
        if use_continuous_distill_teacher:
            teacher_path = _resolve_continuous_distill_teacher_policy_path(
                args,
                user_id=user_id,
            )
            solve_start = time.perf_counter()
            table = _continuous_distill_teacher_table(
                policy_path=teacher_path,
                cost_weights=cost_weights,
                s_grid_size=args.oracle_s_grid_size,
                d_grid_size=args.oracle_d_grid_size,
                device=device,
            )
            solve_runtime_s = time.perf_counter() - solve_start
            oracle_iterations: list[int] = []
            oracle_residuals: list[float] = []
        else:
            user_args = argparse.Namespace(**vars(args))
            user_args.user_id = user_id
            fsrs_config = load_single_card_fsrs6_config(
                user_args,
                environment=args.env,
            )
            oracle = FSRS6ContinuousStationaryFiniteOracle(
                days=args.days,
                s_grid_size=args.oracle_s_grid_size,
                d_grid_size=args.oracle_d_grid_size,
                retention_min=args.retention_min,
                retention_max=args.retention_max,
                interval_chunk_size=args.oracle_interval_chunk_size,
                device=device,
                cache_config=single_card_runtime_context_from_args(
                    args,
                    repo_root=REPO_ROOT,
                    output_dir=args.out_dir,
                ).dp_cache_config,
                fsrs_weights=fsrs_config.fsrs_weights,
                first_rating_prob=fsrs_config.first_rating_prob,
                review_rating_prob=fsrs_config.review_rating_prob,
                learning_costs=fsrs_config.learning_costs,
                review_costs=fsrs_config.review_costs,
            )
            solve_start = time.perf_counter()
            solution = oracle.solve_stationary_finite_policies(
                cost_weights,
                max_iterations=args.oracle_max_iterations,
                tolerance=args.oracle_tolerance,
                progress=not args.no_progress,
            )
            solve_runtime_s = time.perf_counter() - solve_start
            if not all(solution.converged):
                failed = [
                    format_float(weight)
                    for weight, did_converge in zip(
                        cost_weights,
                        solution.converged,
                        strict=True,
                    )
                    if not did_converge
                ]
                raise SystemExit(
                    "Continuous stationary finite teacher did not converge for "
                    f"user {user_id}: " + ",".join(failed)
                )
            table = TeacherTable(
                s_grid=oracle.s_grid,
                d_grid=oracle.d_grid,
                policy=solution.policy.to(device=device, dtype=torch.float64),
                factor=oracle.factor,
                decay=oracle.decay,
                retention_min=oracle.retention_min,
                retention_max=oracle.retention_max,
                bounds=oracle.bounds,
                source="fsrs6_oracle_continuous_stationary_finite",
            )
            oracle_iterations = [int(value) for value in solution.iterations]
            oracle_residuals = [float(value) for value in solution.residuals]
        policy, fit_stats = fit_policy_from_table(
            table=table,
            cost_weights=cost_weights,
            action_head=args.action_head,
            state_feature_count=args.state_feature_count,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
        user_dir = args.out_dir / f"user_{user_id}"
        policy_path = user_dir / "policy.json"
        policy.write_json(policy_path)
        metadata = {
            "policy_kind": "fsrs6-cost-conditioned-adr",
            "user_id": user_id,
            "environment": args.env,
            "teacher": table.source,
            "teacher_path": str(table.source_path) if table.source_path else None,
            "train_cost_weights": cost_weights,
            "eval_cost_weights": eval_cost_weights,
            "action_head": args.action_head,
            "state_feature_count": args.state_feature_count,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_iterations": oracle_iterations,
            "oracle_residuals": oracle_residuals,
            "policy_path": str(policy_path),
        }
        (user_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        row = {
            "user_id": user_id,
            "environment": args.env,
            "policy_path": str(policy_path),
            "action_head": args.action_head,
            "state_feature_count": args.state_feature_count,
            "parameter_count": policy.parameter_count,
            "cost_weight_count": len(cost_weights),
            "epochs": args.epochs,
            "teacher_solve_runtime_s": solve_runtime_s,
            **fit_stats,
        }
        summary_rows.append(row)
        manifest_entries.append(
            {
                "user_id": user_id,
                "policy_path": str(policy_path),
                "action_head": args.action_head,
                "parameter_count": policy.parameter_count,
            }
        )
        print(
            " ".join(
                [
                    f"user={user_id}",
                    f"policy={policy_path}",
                    f"loss={fit_stats['final_loss']:.6g}",
                    f"mae={fit_stats['mean_abs_error']:.6g}",
                ]
            )
        )
    write_csv(args.out_dir / "summary.csv", summary_rows)
    (args.out_dir / "policy_manifest.json").write_text(
        json.dumps(
            {
                "policy_kind": "fsrs6-cost-conditioned-adr",
                "entries": manifest_entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    main_from_args(parse_args())


if __name__ == "__main__":
    main()
