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

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.fsrs_oracle_distill import (
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_OBS_MODE,
    DEFAULT_EVAL_PARTICLES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_STEPS_PER_EPOCH,
    DistillStats,
    OracleGridGuide,
    PolicyValueNet,
    estimate_teacher_action_agreement,
    resolve_torch_device,
    save_model,
)
from experiments.single_card_config import (
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
    SingleCardFSRS6Config,
)
from experiments.single_card_tradeoff import (
    DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.uvfa_ppo_single_card import (
    DEFAULT_TRAIN_ENVS,
    FSRS6SingleCardBatch,
    fsrs_config_kwargs,
    evaluate_policy,
    parse_csv_floats,
    scalar_objective,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float

DEFAULT_CANDIDATES = (
    "res96d3=residual:96:3,"
    "res80d3=residual:80:3,"
    "res64d3=residual:64:3,"
    "res64d2=residual:64:2,"
    "res48d3=residual:48:3,"
    "res48d2=residual:48:2,"
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


@dataclass(frozen=True)
class CandidateStats:
    final_loss: float
    final_action_agreement: float
    runtime_s: float
    transitions: int


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


def param_count(candidate: Candidate, *, action_count: int) -> int:
    model = PolicyValueNet(
        4,
        action_count,
        candidate.hidden_size,
        architecture=candidate.network,
        depth=candidate.network_depth,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search FSRS6 oracle distillation model sizes.",
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_TRAIN_COST_WEIGHTS
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
    )
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--obs-mode",
        choices=["basic", "rich", "belief", "oracle"],
        default=DEFAULT_DISTILL_OBS_MODE,
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
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--agreement-envs",
        type=int,
        default=0,
        help="Particles for teacher-agreement evaluation. 0 skips it.",
    )
    parser.add_argument(
        "--agreement-steps",
        type=int,
        default=0,
        help="Teacher-agreement steps. 0 skips it.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_distill_hparam_summary.csv"
        ),
    )
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=Path(
            "logs/single_card_tradeoff/fsrs6_oracle_distill_hparam_detail.csv"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("logs/single_card_tradeoff/fsrs6_oracle_distill_hparam_models"),
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
        action_retentions=args.action_retentions,
        train_envs=args.train_envs,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        obs_mode=args.obs_mode,
        network=candidate.network,
        network_depth=candidate.network_depth,
        hidden_size=candidate.hidden_size,
        oracle_s_grid_size=args.oracle_s_grid_size,
        oracle_d_grid_size=args.oracle_d_grid_size,
        max_grad_norm=args.max_grad_norm,
        eval_particles=args.eval_particles,
        agreement_envs=args.agreement_envs,
        agreement_steps=args.agreement_steps,
        out=args.detail_out,
        model_out=model_out,
        no_progress=args.no_progress,
    )


def build_env(
    args: argparse.Namespace,
    *,
    device: torch.device,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    env_count: int,
    seed: int,
    fsrs_config: SingleCardFSRS6Config | None,
) -> FSRS6SingleCardBatch:
    return FSRS6SingleCardBatch(
        days=args.days,
        env_count=env_count,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        device=device,
        dtype=torch.float32,
        seed=seed,
        exact_memory=False,
        goal_norm_max=max(cost_weights),
        obs_mode=args.obs_mode,
        **fsrs_config_kwargs(fsrs_config),
    )


def train_candidate(
    args: argparse.Namespace,
    *,
    candidate: Candidate,
    device: torch.device,
    guide: OracleGridGuide,
    cost_weights: Sequence[float],
    action_retentions: Sequence[float],
    fsrs_config: SingleCardFSRS6Config | None,
) -> tuple[PolicyValueNet, CandidateStats]:
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    env = build_env(
        args,
        device=device,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        env_count=args.train_envs,
        seed=args.seed,
        fsrs_config=fsrs_config,
    )
    model = PolicyValueNet(
        env.obs_dim,
        env.action_count,
        candidate.hidden_size,
        architecture=candidate.network,
        depth=candidate.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = env.obs()
    final_loss = 0.0
    final_action_agreement = 0.0
    for epoch in range(args.epochs):
        loss_sum = 0.0
        correct = 0
        total = 0
        for _ in range(args.steps_per_epoch):
            label = guide.labels(env)
            logits, _ = model(obs)
            loss = nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                batch_total = int(label.numel())
                correct += int((pred == label).sum().item())
                total += batch_total
                loss_sum += float(loss.item()) * batch_total
                next_obs, _, done = env.step(label)
                if done.any():
                    env.reset_indices(done.nonzero(as_tuple=False).squeeze(1))
                    next_obs = env.obs()
                obs = next_obs

        final_loss = loss_sum / float(max(1, total))
        final_action_agreement = correct / float(max(1, total))
        if not args.no_progress:
            print(
                f"candidate={candidate.name} epoch={epoch + 1}/{args.epochs} "
                f"ce={final_loss:.5f} "
                f"teacher_action_agreement={final_action_agreement:.4f}",
                flush=True,
            )

    runtime_s = time.perf_counter() - start
    return (
        model,
        CandidateStats(
            final_loss=final_loss,
            final_action_agreement=final_action_agreement,
            runtime_s=runtime_s,
            transitions=args.epochs * args.steps_per_epoch * args.train_envs,
        ),
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
    if args.agreement_envs < 0:
        raise SystemExit("--agreement-envs must be >= 0.")
    if args.agreement_steps < 0:
        raise SystemExit("--agreement-steps must be >= 0.")

    cost_weights = parse_csv_floats(args.cost_weights, name="--cost-weights")
    action_retentions = parse_csv_floats(
        args.action_retentions,
        name="--action-retentions",
    )
    if any(weight < 0.0 for weight in cost_weights):
        raise SystemExit("--cost-weights must be >= 0.")
    if any(retention <= 0.0 or retention >= 1.0 for retention in action_retentions):
        raise SystemExit("--action-retentions must be within (0, 1).")

    candidates = parse_candidates(args.candidates)
    device = resolve_torch_device(args.torch_device)
    fsrs_config = load_single_card_fsrs6_config(args)

    oracle_solve_start = time.perf_counter()
    guide = OracleGridGuide(
        days=args.days,
        cost_weights=cost_weights,
        action_retentions=action_retentions,
        s_grid_size=args.oracle_s_grid_size,
        d_grid_size=args.oracle_d_grid_size,
        device=device,
        progress=not args.no_progress,
        fsrs_config=fsrs_config,
    )
    oracle_solve_runtime_s = time.perf_counter() - oracle_solve_start

    baseline_params = param_count(
        Candidate(
            name="res96d3",
            network="residual",
            hidden_size=96,
            network_depth=3,
        ),
        action_count=len(action_retentions),
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

        model, stats = train_candidate(
            candidate_args,
            candidate=candidate,
            device=device,
            guide=guide,
            cost_weights=cost_weights,
            action_retentions=action_retentions,
            fsrs_config=fsrs_config,
        )

        eval_teacher_action_agreement = 0.0
        if args.agreement_envs > 0 and args.agreement_steps > 0:
            eval_teacher_action_agreement = estimate_teacher_action_agreement(
                model,
                guide,
                args=candidate_args,
                device=device,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
                fsrs_config=fsrs_config,
            )

        if args.save_models:
            save_model(
                model_out,
                model=model,
                args=candidate_args,
                cost_weights=cost_weights,
                action_retentions=action_retentions,
                stats=DistillStats(
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    transitions=stats.transitions,
                    runtime_s=stats.runtime_s,
                    final_loss=stats.final_loss,
                    final_action_agreement=stats.final_action_agreement,
                ),
                eval_teacher_action_agreement=eval_teacher_action_agreement,
                fsrs_config=fsrs_config,
            )

        params = param_count(candidate, action_count=len(action_retentions))
        compression_ratio = params / float(baseline_params)
        scalar_values: list[float] = []
        policy_eval_runtime_s = 0.0
        for cost_weight in cost_weights:
            start = time.perf_counter()
            metrics = evaluate_policy(
                model,
                args=candidate_args,
                device=device,
                cost_weight=cost_weight,
                action_retentions=action_retentions,
                particles=args.eval_particles,
                seed=args.seed + 30_000 + int(round(cost_weight * 10.0)),
                goal_norm_max=max(cost_weights),
                obs_mode=args.obs_mode,
                fsrs_config=fsrs_config,
            )
            runtime_s = time.perf_counter() - start
            policy_eval_runtime_s += runtime_s
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
                    "train_runtime_s": stats.runtime_s,
                    "train_transitions": stats.transitions,
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
                "train_final_loss": stats.final_loss,
                "train_final_action_agreement": stats.final_action_agreement,
                "eval_teacher_action_agreement": eval_teacher_action_agreement,
                "train_runtime_s": stats.runtime_s,
                "policy_eval_runtime_s": policy_eval_runtime_s,
                "train_transitions": stats.transitions,
                "oracle_solve_runtime_s": oracle_solve_runtime_s,
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
        "train_final_loss",
        "train_final_action_agreement",
        "eval_teacher_action_agreement",
        "train_runtime_s",
        "policy_eval_runtime_s",
        "train_transitions",
        "oracle_solve_runtime_s",
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
    print(f"Oracle guide solve runtime_s={oracle_solve_runtime_s:.2f} device={device}")
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
