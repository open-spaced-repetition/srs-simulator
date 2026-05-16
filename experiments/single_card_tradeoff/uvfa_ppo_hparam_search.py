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
import re
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.uvfa_ppo import (
    DEFAULT_ADVANTAGE_NORMALIZATION,
    DEFAULT_CLIP_COEF,
    DEFAULT_COST_WEIGHTS,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_GUIDE_POLICY,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MINIBATCH_SIZE,
    DEFAULT_NETWORK,
    DEFAULT_NETWORK_DEPTH,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_PPO_EPOCHS,
    DEFAULT_PRIOR_COEF,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_TRAIN_ENVS,
    DEFAULT_TARGET_RETENTIONS,
    DEFAULT_UPDATES,
    DEFAULT_VALUE_COEF,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_STEPS,
    build_policy_guide,
    evaluate_policy,
    parse_csv_floats,
    scalar_objective,
    save_model,
    train_policy,
)
from experiments.single_card_tradeoff.retention_space import validate_retention_values
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED
from simulator.scheduler_spec import format_float

DEFAULT_CANDIDATES = (
    f"default={DEFAULT_NETWORK}:{DEFAULT_HIDDEN_SIZE}:{DEFAULT_NETWORK_DEPTH},"
    "res96d3=residual:96:3,"
    "res64d2=residual:64:2,"
    "res48d2=residual:48:2,"
    "res32d2=residual:32:2,"
    "mlp64=mlp:64:1"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    network: str
    hidden_size: int
    network_depth: int


def parse_candidates(raw: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    names: set[str] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            name, spec = item.split("=", 1)
            network, hidden_raw, depth_raw = spec.split(":", 2)
        except ValueError as exc:
            raise SystemExit(
                "Candidate specs must look like name=network:hidden:depth."
            ) from exc
        name = sanitize_name(name)
        if name in names:
            raise SystemExit(f"Duplicate candidate name '{name}'.")
        names.add(name)
        if network not in {"residual", "mlp"}:
            raise SystemExit("Candidate network must be 'residual' or 'mlp'.")
        try:
            hidden_size = int(hidden_raw)
            network_depth = int(depth_raw)
        except ValueError as exc:
            raise SystemExit("Candidate hidden/depth values must be integers.") from exc
        if hidden_size <= 0 or network_depth <= 0:
            raise SystemExit("Candidate hidden/depth values must be > 0.")
        candidates.append(
            Candidate(
                name=name,
                network=network,
                hidden_size=hidden_size,
                network_depth=network_depth,
            )
        )
    if not candidates:
        raise SystemExit("--candidates must include at least one candidate.")
    return candidates


def sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not name:
        raise SystemExit("Candidate names must not be empty.")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search UVFA PPO model-scale hyperparameters.",
        allow_abbrev=False,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
    )
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument(
        "--advantage-normalization",
        choices=["global", "goal"],
        default=DEFAULT_ADVANTAGE_NORMALIZATION,
    )
    parser.add_argument(
        "--guide-policy",
        choices=["oracle", "static", "none"],
        default=DEFAULT_GUIDE_POLICY,
    )
    parser.add_argument(
        "--oracle-s-grid-size", type=int, default=DEFAULT_ORACLE_S_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-d-grid-size", type=int, default=DEFAULT_ORACLE_D_GRID_SIZE
    )
    parser.add_argument("--clip-coef", type=float, default=DEFAULT_CLIP_COEF)
    parser.add_argument("--prior-coef", type=float, default=DEFAULT_PRIOR_COEF)
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    parser.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/uvfa_ppo_hparam_search_summary.csv"
        ),
    )
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=Path(
            "artifacts/single_card_tradeoff/uvfa_ppo_hparam_search_detail.csv"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/single_card_tradeoff/uvfa_ppo_hparam_search_models"),
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def make_train_args(
    args: argparse.Namespace,
    *,
    candidate: Candidate,
    model_out: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        days=args.days,
        seed=args.seed,
        torch_device=args.torch_device,
        train_envs=args.train_envs,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        advantage_normalization=args.advantage_normalization,
        obs_mode="rich",
        network=candidate.network,
        network_depth=candidate.network_depth,
        guide_policy=args.guide_policy,
        oracle_s_grid_size=args.oracle_s_grid_size,
        oracle_d_grid_size=args.oracle_d_grid_size,
        clip_coef=args.clip_coef,
        prior_coef=args.prior_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_size=candidate.hidden_size,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
        eval_particles=args.eval_particles,
        baseline_particles=args.eval_particles,
        baseline="fixed",
        deck_scale=1,
        out=args.detail_out,
        model_out=model_out,
        plot_path=None,
        no_plot=True,
        no_progress=args.no_progress,
    )


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.train_envs <= 0 or args.eval_particles <= 0:
        raise SystemExit("--train-envs and --eval-particles must be > 0.")
    if args.updates < 0:
        raise SystemExit("--updates must be >= 0.")
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    validate_retention_values(action_retentions, name="--action-retentions")

    candidates = parse_candidates(args.candidates)
    device = (
        torch.device(args.torch_device) if args.torch_device else torch.device("cpu")
    )
    guide_args = make_train_args(
        args,
        candidate=candidates[0],
        model_out=args.model_dir / f"{candidates[0].name}.pt",
    )
    shared_guide = build_policy_guide(
        args=guide_args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate_idx, candidate in enumerate(candidates):
        model_out = args.model_dir / f"{candidate.name}.pt"
        train_args = make_train_args(args, candidate=candidate, model_out=model_out)
        if not args.no_progress:
            print(
                f"candidate={candidate.name} "
                f"network={candidate.network} "
                f"hidden={candidate.hidden_size} depth={candidate.network_depth}",
                flush=True,
            )
        model, train_stats = train_policy(
            train_args,
            device=device,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            policy_guide=shared_guide,
            build_guide_if_missing=False,
        )
        if args.save_models:
            save_model(
                model_out,
                model=model,
                args=train_args,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
                train_stats=train_stats,
            )
        param_count = sum(parameter.numel() for parameter in model.parameters())
        scalar_values: list[float] = []
        for cost_weight in cost_weights:
            metrics = evaluate_policy(
                model,
                args=train_args,
                device=device,
                cost_weight=cost_weight,
                action_retentions=action_retentions,
                particles=args.eval_particles,
                seed=args.seed + 30_000 + int(round(cost_weight * 10.0)),
                goal_norm_max=max(cost_weights),
                obs_mode=train_args.obs_mode,
            )
            scalar = scalar_objective(metrics, cost_weight)
            scalar_values.append(scalar)
            detail_rows.append(
                {
                    "candidate": candidate.name,
                    "candidate_index": candidate_idx,
                    "network": candidate.network,
                    "hidden_size": candidate.hidden_size,
                    "network_depth": candidate.network_depth,
                    "param_count": param_count,
                    "cost_weight": cost_weight,
                    "card_expected_retrievability": metrics.card_expected_retrievability,
                    "card_minutes_per_day": metrics.card_minutes_per_day,
                    "card_reviews_per_day": metrics.card_reviews_per_day,
                    "card_total_reviews": metrics.card_total_reviews,
                    "card_total_lapses": metrics.card_total_lapses,
                    "card_total_cost_seconds": metrics.card_total_cost_seconds,
                    "observed_retention": metrics.observed_retention,
                    "scalar_objective": scalar,
                    "delta_vs_best_candidate": None,
                    "train_runtime_s": train_stats.runtime_s,
                    "train_transitions": train_stats.transitions,
                }
            )
        mean_scalar = sum(scalar_values) / float(len(scalar_values))
        summary_rows.append(
            {
                "candidate": candidate.name,
                "candidate_index": candidate_idx,
                "network": candidate.network,
                "hidden_size": candidate.hidden_size,
                "network_depth": candidate.network_depth,
                "param_count": param_count,
                "mean_scalar_objective": mean_scalar,
                "min_scalar_objective": min(scalar_values),
                "max_scalar_objective": max(scalar_values),
                "mean_delta_vs_best_candidate": None,
                "min_delta_vs_best_candidate": None,
                "train_runtime_s": train_stats.runtime_s,
                "train_transitions": train_stats.transitions,
                "updates": train_stats.updates,
            }
        )

    best_by_weight = {
        weight: max(
            float(row["scalar_objective"])
            for row in detail_rows
            if math.isclose(float(row["cost_weight"]), weight)
        )
        for weight in cost_weights
    }
    deltas_by_candidate: dict[str, list[float]] = {
        row["candidate"]: [] for row in summary_rows
    }
    for row in detail_rows:
        weight = float(row["cost_weight"])
        delta = float(row["scalar_objective"]) - best_by_weight[weight]
        row["delta_vs_best_candidate"] = delta
        deltas_by_candidate[str(row["candidate"])].append(delta)
    for row in summary_rows:
        deltas = deltas_by_candidate[str(row["candidate"])]
        row["mean_delta_vs_best_candidate"] = sum(deltas) / float(len(deltas))
        row["min_delta_vs_best_candidate"] = min(deltas)

    summary_rows.sort(
        key=lambda row: (
            -float(row["mean_scalar_objective"]),
            int(row["param_count"]),
        )
    )
    summary_fieldnames = [
        "candidate",
        "candidate_index",
        "network",
        "hidden_size",
        "network_depth",
        "param_count",
        "mean_scalar_objective",
        "min_scalar_objective",
        "max_scalar_objective",
        "mean_delta_vs_best_candidate",
        "min_delta_vs_best_candidate",
        "train_runtime_s",
        "train_transitions",
        "updates",
    ]
    detail_fieldnames = [
        "candidate",
        "candidate_index",
        "network",
        "hidden_size",
        "network_depth",
        "param_count",
        "cost_weight",
        "card_expected_retrievability",
        "card_minutes_per_day",
        "card_reviews_per_day",
        "card_total_reviews",
        "card_total_lapses",
        "card_total_cost_seconds",
        "observed_retention",
        "scalar_objective",
        "delta_vs_best_candidate",
        "train_runtime_s",
        "train_transitions",
    ]
    write_csv(args.summary_out, summary_rows, summary_fieldnames)
    write_csv(args.detail_out, detail_rows, detail_fieldnames)
    print(f"Wrote summary CSV: {args.summary_out}")
    print(f"Wrote detail CSV: {args.detail_out}")
    for row in summary_rows:
        print(
            " ".join(
                [
                    f"candidate={row['candidate']}",
                    f"params={row['param_count']}",
                    f"mean_scalar={float(row['mean_scalar_objective']):.6f}",
                    f"mean_delta_best={float(row['mean_delta_vs_best_candidate']):.6f}",
                    f"min_delta_best={float(row['min_delta_vs_best_candidate']):.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
