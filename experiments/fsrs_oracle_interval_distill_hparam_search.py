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
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.fsrs_oracle_frontier import FSRS6IntervalOracle
from experiments.fsrs_oracle_interval_distill import (
    DEFAULT_LOG_INTERVAL_BIAS,
    DEFAULT_STUDENT_ROLLOUT_PROB,
    DEFAULT_STUDENT_ROLLOUT_WARMUP_EPOCHS,
    DEFAULT_TERMINAL_SNAP_RATIO,
    DEFAULT_TERMINAL_UNDERPREDICTION_LOSS_WEIGHT,
    DEFAULT_UNDERPREDICTION_LOSS_WEIGHT,
    IntervalDistillNet,
    evaluate_interval_agreement,
    evaluate_policy,
    resolve_torch_device,
    save_model,
    train_model,
)
from experiments.single_card_config import (
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.uvfa_ppo_single_card import (
    DEFAULT_COST_WEIGHTS,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    fsrs_config_kwargs,
    parse_csv_floats,
    scalar_objective,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float

DEFAULT_CANDIDATES = (
    "res96d3=residual:96:3,"
    "res80d2=residual:80:2,"
    "res64d2=residual:64:2,"
    "res64d1=residual:64:1,"
    "res48d2=residual:48:2,"
    "res48d1=residual:48:1,"
    "res32d2=residual:32:2,"
    "mlp96=mlp:96:1,"
    "mlp64=mlp:64:1"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    network: str
    hidden_size: int
    network_depth: int


def sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not name:
        raise SystemExit("Candidate names must not be empty.")
    return name


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


def param_count(candidate: Candidate) -> int:
    model = IntervalDistillNet(
        obs_dim=4,
        hidden_size=candidate.hidden_size,
        architecture=candidate.network,
        depth=candidate.network_depth,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search FSRS6 interval oracle distillation model sizes.",
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
    )
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--train-envs", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--underprediction-loss-weight",
        type=float,
        default=DEFAULT_UNDERPREDICTION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--terminal-underprediction-loss-weight",
        type=float,
        default=DEFAULT_TERMINAL_UNDERPREDICTION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--student-rollout-prob",
        type=float,
        default=DEFAULT_STUDENT_ROLLOUT_PROB,
    )
    parser.add_argument(
        "--student-rollout-warmup-epochs",
        type=int,
        default=DEFAULT_STUDENT_ROLLOUT_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--terminal-snap-ratio", type=float, default=DEFAULT_TERMINAL_SNAP_RATIO
    )
    parser.add_argument(
        "--log-interval-bias", type=float, default=DEFAULT_LOG_INTERVAL_BIAS
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
    parser.add_argument("--oracle-interval-chunk-size", type=int, default=64)
    parser.add_argument("--eval-particles", type=int, default=3000)
    parser.add_argument(
        "--agreement-particles",
        type=int,
        default=0,
        help="Particles per cost weight for agreement metrics. 0 skips agreement eval.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_interval_distill_hparam_summary.csv"
        ),
    )
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_interval_distill_hparam_detail.csv"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_interval_distill_hparam_models"
        ),
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def make_candidate_args(
    args: argparse.Namespace,
    *,
    candidate: Candidate,
    model_out: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        env=args.env,
        user_id=args.user_id,
        benchmark_result=args.benchmark_result,
        benchmark_partition=args.benchmark_partition,
        srs_benchmark_root=args.srs_benchmark_root,
        button_usage=args.button_usage,
        days=args.days,
        deck_scale=args.deck_scale,
        seed=args.seed,
        torch_device=args.torch_device,
        cost_weights=args.cost_weights,
        train_envs=args.train_envs,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        hidden_size=candidate.hidden_size,
        network=candidate.network,
        network_depth=candidate.network_depth,
        underprediction_loss_weight=args.underprediction_loss_weight,
        terminal_underprediction_loss_weight=args.terminal_underprediction_loss_weight,
        student_rollout_prob=args.student_rollout_prob,
        student_rollout_warmup_epochs=args.student_rollout_warmup_epochs,
        terminal_snap_ratio=args.terminal_snap_ratio,
        log_interval_bias=args.log_interval_bias,
        oracle_s_grid_size=args.oracle_s_grid_size,
        oracle_d_grid_size=args.oracle_d_grid_size,
        oracle_interval_chunk_size=args.oracle_interval_chunk_size,
        eval_particles=args.eval_particles,
        out=args.detail_out,
        model_out=model_out,
        no_progress=args.no_progress,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: Sequence[str],
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
    if args.deck_scale <= 0:
        raise SystemExit("--deck-scale must be > 0.")
    if args.train_envs <= 0 or args.eval_particles <= 0:
        raise SystemExit("--train-envs and --eval-particles must be > 0.")
    if args.epochs < 0:
        raise SystemExit("--epochs must be >= 0.")
    if args.steps_per_epoch <= 0:
        raise SystemExit("--steps-per-epoch must be > 0.")
    if args.oracle_s_grid_size < 8 or args.oracle_d_grid_size < 8:
        raise SystemExit("--oracle grid sizes must be >= 8.")
    if args.oracle_interval_chunk_size <= 0:
        raise SystemExit("--oracle-interval-chunk-size must be > 0.")
    if args.underprediction_loss_weight < 0.0:
        raise SystemExit("--underprediction-loss-weight must be >= 0.")
    if args.terminal_underprediction_loss_weight < 0.0:
        raise SystemExit("--terminal-underprediction-loss-weight must be >= 0.")
    if not 0.0 <= args.student_rollout_prob <= 1.0:
        raise SystemExit("--student-rollout-prob must be within [0, 1].")
    if args.student_rollout_warmup_epochs < 0:
        raise SystemExit("--student-rollout-warmup-epochs must be >= 0.")
    if args.terminal_snap_ratio < 0.0:
        raise SystemExit("--terminal-snap-ratio must be >= 0.")
    if args.agreement_particles < 0:
        raise SystemExit("--agreement-particles must be >= 0.")

    candidates = parse_candidates(args.candidates)
    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    device = resolve_torch_device(args.torch_device)
    fsrs_config = load_single_card_fsrs6_config(args)

    oracle = FSRS6IntervalOracle(
        days=args.days,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        interval_chunk_size=args.oracle_interval_chunk_size,
        device=device,
        **fsrs_config_kwargs(fsrs_config),
    )
    oracle_start = time.perf_counter()
    policies = oracle.solve_policies(cost_weights, progress=not args.no_progress)
    oracle_solve_runtime_s = time.perf_counter() - oracle_start

    baseline_params = param_count(
        Candidate(
            name="res96d3",
            network="residual",
            hidden_size=96,
            network_depth=3,
        )
    )
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate_idx, candidate in enumerate(candidates):
        model_out = args.model_dir / f"{candidate.name}.pt"
        candidate_args = make_candidate_args(
            args,
            candidate=candidate,
            model_out=model_out,
        )
        if not args.no_progress:
            print(
                " ".join(
                    [
                        f"candidate={candidate.name}",
                        f"network={candidate.network}",
                        f"hidden={candidate.hidden_size}",
                        f"depth={candidate.network_depth}",
                    ]
                ),
                flush=True,
            )
        model, train_stats = train_model(
            candidate_args,
            device=device,
            oracle=oracle,
            policies=policies,
            cost_weights=cost_weights,
            fsrs_config=fsrs_config,
        )
        eval_stats: dict[str, float] = {}
        if args.agreement_particles > 0:
            agreement_args = argparse.Namespace(**vars(candidate_args))
            agreement_args.eval_particles = args.agreement_particles
            eval_stats = evaluate_interval_agreement(
                args=agreement_args,
                model=model,
                device=device,
                oracle=oracle,
                policies=policies,
                cost_weights=cost_weights,
                seed=args.seed + 70_000,
                fsrs_config=fsrs_config,
            )
        if args.save_models:
            save_model(
                model_out,
                model=model,
                args=candidate_args,
                cost_weights=cost_weights,
                train_stats=train_stats,
                eval_stats=eval_stats,
                oracle_solve_runtime_s=oracle_solve_runtime_s,
                fsrs_config=fsrs_config,
            )

        params = sum(parameter.numel() for parameter in model.parameters())
        compression_ratio = params / float(baseline_params)
        scalar_values: list[float] = []
        eval_runtime_s = 0.0
        for cost_weight in cost_weights:
            start = time.perf_counter()
            metrics = evaluate_policy(
                model,
                args=candidate_args,
                device=device,
                cost_weight=cost_weight,
                particles=args.eval_particles,
                seed=args.seed + 80_000 + int(round(cost_weight * 10.0)),
                goal_norm_max=max(cost_weights),
                fsrs_config=fsrs_config,
            )
            runtime_s = time.perf_counter() - start
            eval_runtime_s += runtime_s
            scalar = scalar_objective(metrics, cost_weight)
            scalar_values.append(scalar)
            detail_rows.append(
                {
                    "candidate": candidate.name,
                    "candidate_index": candidate_idx,
                    "network": candidate.network,
                    "hidden_size": candidate.hidden_size,
                    "network_depth": candidate.network_depth,
                    "param_count": params,
                    "compression_ratio": compression_ratio,
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
                    "runtime_s": runtime_s,
                    "train_runtime_s": train_stats.runtime_s,
                    "train_transitions": train_stats.transitions,
                }
            )

        summary_rows.append(
            {
                "candidate": candidate.name,
                "candidate_index": candidate_idx,
                "network": candidate.network,
                "hidden_size": candidate.hidden_size,
                "network_depth": candidate.network_depth,
                "param_count": params,
                "compression_ratio": compression_ratio,
                "mean_scalar_objective": sum(scalar_values) / float(len(scalar_values)),
                "min_scalar_objective": min(scalar_values),
                "max_scalar_objective": max(scalar_values),
                "mean_delta_vs_best_candidate": None,
                "min_delta_vs_best_candidate": None,
                "train_runtime_s": train_stats.runtime_s,
                "policy_eval_runtime_s": eval_runtime_s,
                "train_transitions": train_stats.transitions,
                "oracle_solve_runtime_s": oracle_solve_runtime_s,
                **eval_stats,
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
        str(row["candidate"]): [] for row in summary_rows
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
        "compression_ratio",
        "mean_scalar_objective",
        "min_scalar_objective",
        "max_scalar_objective",
        "mean_delta_vs_best_candidate",
        "min_delta_vs_best_candidate",
        "train_runtime_s",
        "policy_eval_runtime_s",
        "train_transitions",
        "oracle_solve_runtime_s",
        "eval_smooth_l1_loss",
        "eval_log_interval_mae",
        "eval_log_interval_rmse",
        "eval_interval_mae_days",
        "eval_rounded_interval_agreement",
        "eval_runtime_s",
    ]
    detail_fieldnames = [
        "candidate",
        "candidate_index",
        "network",
        "hidden_size",
        "network_depth",
        "param_count",
        "compression_ratio",
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
        "runtime_s",
        "train_runtime_s",
        "train_transitions",
    ]
    write_csv(args.summary_out, summary_rows, summary_fieldnames)
    write_csv(args.detail_out, detail_rows, detail_fieldnames)
    print(f"Wrote summary CSV: {args.summary_out}")
    print(f"Wrote detail CSV: {args.detail_out}")
    print(
        f"Interval oracle solve runtime_s={oracle_solve_runtime_s:.2f} device={device}"
    )
    for row in summary_rows:
        print(
            " ".join(
                [
                    f"candidate={row['candidate']}",
                    f"params={row['param_count']}",
                    f"compression={float(row['compression_ratio']):.3f}",
                    f"mean_scalar={float(row['mean_scalar_objective']):.6f}",
                    f"mean_delta_best={float(row['mean_delta_vs_best_candidate']):.6f}",
                    f"min_delta_best={float(row['min_delta_vs_best_candidate']):.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
