from __future__ import annotations

# ruff: noqa: E402
# pyright: reportPrivateUsage=false

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.cli.low_param_direct_policy_search_multiuser import (  # noqa: E402
    POLICY_FAMILY_CHOICES,
    direct_policy_retention,
    initial_theta,
    parameter_count_for_family,
    parameter_names_for_family,
    sample_population,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (  # noqa: E402
    resolve_torch_device,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    DEFAULT_USER_IDS,
    MultiUserFSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.core.config import (  # noqa: E402
    SingleCardFSRS6Config,
    add_single_card_fsrs6_config_args,
    load_single_card_fsrs6_config,
)
from experiments.single_card_tradeoff.core.run_monitoring import (  # noqa: E402
    add_run_monitoring_args,
    register_run_monitor,
)
from experiments.single_card_tradeoff.core.target_search.comparison import (  # noqa: E402
    compare_target_answers_to_oracle,
    oracle_gap_row,
    read_target_answer_records,
    resolve_target_answers_path,
    summarize_oracle_gaps,
    target_answer_records_from_rows,
)
from experiments.single_card_tradeoff.core.target_search.direct_training import (  # noqa: E402
    DirectTargetJob,
    constrained_rank_candidates,
    direct_target_jobs,
)
from experiments.single_card_tradeoff.core.target_search.frontier import (  # noqa: E402
    empirical_frontier,
    frontier_segments,
)
from experiments.single_card_tradeoff.core.target_search.io import (  # noqa: E402
    answer_row,
    point_row,
    segment_row,
)
from experiments.single_card_tradeoff.core.target_search.types import (  # noqa: E402
    EvaluatedPoint,
    TargetAnswer,
)
from experiments.single_card_tradeoff.core.types import SimMetrics  # noqa: E402
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402

DIRECT_CONSTRAINED_FAMILY = "fsrs6_low_param_direct_constrained"
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/target_constrained_direct_policy_search"
)


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
            "Train target-constrained low-parameter FSRS-6 policies with "
            "feasible-first CEM ranking."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
    )
    parser.add_argument("--target-memories", default="")
    parser.add_argument("--target-times", default="")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--min-retention", type=float, default=0.5)
    parser.add_argument("--max-retention", type=float, default=0.98)
    parser.add_argument(
        "--policy-family",
        choices=POLICY_FAMILY_CHOICES,
        default="bilinear_monotone",
    )
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=64)
    parser.add_argument("--train-particles", type=int, default=64)
    parser.add_argument("--eval-particles", type=int, default=10_000)
    parser.add_argument("--initial-std", type=float, default=1.5)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--theta-clip", type=float, default=8.0)
    parser.add_argument("--train-exact-memory", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--oracle-target-answers",
        type=Path,
        default=None,
        help=(
            "Optional oracle target_answers.csv, or directory containing it, "
            "used to write target_oracle_gaps.csv."
        ),
    )
    parser.add_argument("--oracle-gap-target-tolerance", type=float, default=1e-9)
    add_run_monitoring_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.env not in {"fsrs6", "fsrs6_default"}:
        raise SystemExit("target constrained direct search requires an FSRS6 env.")
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if not (0.0 < args.min_retention < args.max_retention < 1.0):
        raise SystemExit(
            "--min-retention/--max-retention must satisfy 0 < min < max < 1."
        )
    if args.population_size < 2:
        raise SystemExit("--population-size must be >= 2.")
    if args.elite_count <= 0 or args.elite_count > args.population_size:
        raise SystemExit("--elite-count must be in [1, population-size].")
    if args.generations <= 0:
        raise SystemExit("--generations must be > 0.")
    if args.train_particles <= 0 or args.eval_particles <= 0:
        raise SystemExit("--train-particles and --eval-particles must be > 0.")
    if args.initial_std <= 0.0 or args.min_std <= 0.0:
        raise SystemExit("--initial-std and --min-std must be > 0.")
    if args.max_std < args.min_std:
        raise SystemExit("--max-std must be >= --min-std.")
    if not (0.0 < args.cem_alpha <= 1.0):
        raise SystemExit("--cem-alpha must be in (0, 1].")
    if args.theta_clip <= 0.0:
        raise SystemExit("--theta-clip must be > 0.")
    if args.oracle_gap_target_tolerance < 0.0 or not math.isfinite(
        args.oracle_gap_target_tolerance
    ):
        raise SystemExit("--oracle-gap-target-tolerance must be finite and >= 0.")
    if args.oracle_target_answers is not None:
        oracle_path = resolve_target_answers_path(args.oracle_target_answers)
        if not oracle_path.exists():
            raise SystemExit(f"--oracle-target-answers does not exist: {oracle_path}")
    target_memories = parse_csv_floats(args.target_memories, name="--target-memories")
    target_times = parse_csv_floats(args.target_times, name="--target-times")
    if not target_memories and not target_times:
        raise SystemExit("Provide at least one --target-memories or --target-times.")
    if any(value <= 0.0 or value >= 1.0 for value in target_memories):
        raise SystemExit("--target-memories values must satisfy 0 < M0 < 1.")
    if any(value < 0.0 for value in target_times):
        raise SystemExit("--target-times values must be >= 0.")


def load_user_configs(
    args: argparse.Namespace,
    user_ids: Sequence[int],
) -> list[SingleCardFSRS6Config]:
    configs: list[SingleCardFSRS6Config] = []
    for user_id in user_ids:
        user_args = argparse.Namespace(**vars(args))
        user_args.user_id = user_id
        configs.append(load_single_card_fsrs6_config(user_args, environment=args.env))
    return configs


def _candidate_layout(
    *,
    jobs: Sequence[DirectTargetJob],
    candidate_count: int,
    particles_per_group: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    user_indices: list[int] = []
    job_indices: list[int] = []
    candidate_indices: list[int] = []
    group_indices: list[int] = []
    for job_idx, job in enumerate(jobs):
        for candidate_idx in range(candidate_count):
            group_idx = job_idx * candidate_count + candidate_idx
            user_indices.extend([job.user_idx] * particles_per_group)
            job_indices.extend([job_idx] * particles_per_group)
            candidate_indices.extend([candidate_idx] * particles_per_group)
            group_indices.extend([group_idx] * particles_per_group)
    return (
        user_indices,
        torch.tensor(job_indices, device=device, dtype=torch.int64),
        torch.tensor(candidate_indices, device=device, dtype=torch.int64),
        torch.tensor(group_indices, device=device, dtype=torch.int64),
    )


@torch.inference_mode()
def evaluate_candidate_metrics(
    args: argparse.Namespace,
    *,
    theta: torch.Tensor,
    jobs: Sequence[DirectTargetJob],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
    particles: int,
    seed: int,
    exact_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[SimMetrics]]:
    job_count, candidate_count, _ = theta.shape
    user_indices, job_idx, candidate_idx, group_index = _candidate_layout(
        jobs=jobs,
        candidate_count=candidate_count,
        particles_per_group=particles,
        device=device,
    )
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=[0.0],
        action_retentions=[args.min_retention, args.max_retention],
        device=device,
        dtype=torch.float64,
        seed=seed,
        exact_memory=exact_memory,
        goal_norm_max=1.0,
        reset_on_init=False,
    )
    env.reset_all(goal_values=0.0)
    while not bool(env.done.all().item()):
        selected_theta = theta[job_idx, candidate_idx].to(dtype=torch.float64)
        retention = direct_policy_retention(
            selected_theta,
            env.obs(),
            policy_family=args.policy_family,
            min_retention=args.min_retention,
            max_retention=args.max_retention,
        )
        env.step_retention(retention)
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics = env.metrics_by_group(
        group_index=group_index,
        group_count=job_count * candidate_count,
        particles_per_group=particles,
    )
    memory = torch.tensor(
        [metric.card_expected_retrievability for metric in metrics],
        device=device,
        dtype=torch.float64,
    ).reshape(job_count, candidate_count)
    minutes = torch.tensor(
        [metric.card_minutes_per_day for metric in metrics],
        device=device,
        dtype=torch.float64,
    ).reshape(job_count, candidate_count)
    return memory, minutes, metrics


def _better_candidate(
    *,
    new_feasible: torch.Tensor,
    new_metric: torch.Tensor,
    old_feasible: torch.Tensor,
    old_metric: torch.Tensor,
) -> torch.Tensor:
    return (new_feasible & ~old_feasible) | (
        (new_feasible == old_feasible) & (new_metric > old_metric)
    )


def optimize_constrained_policies(
    args: argparse.Namespace,
    *,
    jobs: Sequence[DirectTargetJob],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    mean = initial_theta(
        policy_family=args.policy_family,
        user_count=len(jobs),
        device=device,
        dtype=torch.float64,
    )
    std = torch.full_like(mean, float(args.initial_std))
    best_theta = mean.clone()
    best_feasible = torch.zeros(len(jobs), device=device, dtype=torch.bool)
    best_metric = torch.full(
        (len(jobs),), -math.inf, device=device, dtype=torch.float64
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 210_000)
    history: list[dict[str, Any]] = []
    start_s = time.perf_counter()

    for generation in range(1, args.generations + 1):
        candidates = sample_population(
            mean=mean,
            std=std,
            population_size=args.population_size,
            theta_clip=args.theta_clip,
            generator=generator,
        )
        memory, minutes, _metrics = evaluate_candidate_metrics(
            args,
            theta=candidates,
            jobs=jobs,
            configs=configs,
            device=device,
            particles=args.train_particles,
            seed=args.seed + 220_000 + generation,
            exact_memory=args.train_exact_memory,
        )
        rank = constrained_rank_candidates(memory=memory, minutes=minutes, jobs=jobs)
        elite_score, elite_idx = torch.topk(rank.score, k=args.elite_count, dim=1)
        gather_idx = elite_idx[:, :, None].expand(-1, -1, mean.shape[1])
        elite_theta = candidates.gather(1, gather_idx)
        generation_best_score, generation_best_idx = torch.max(rank.score, dim=1)
        generation_best_feasible = rank.feasible[
            torch.arange(len(jobs), device=device),
            generation_best_idx,
        ]
        generation_best_metric = rank.rank_metric[
            torch.arange(len(jobs), device=device),
            generation_best_idx,
        ]
        improved = _better_candidate(
            new_feasible=generation_best_feasible,
            new_metric=generation_best_metric,
            old_feasible=best_feasible,
            old_metric=best_metric,
        )
        if bool(improved.any().item()):
            best_feasible = torch.where(
                improved, generation_best_feasible, best_feasible
            )
            best_metric = torch.where(improved, generation_best_metric, best_metric)
            best_theta[improved] = candidates[improved, generation_best_idx[improved]]

        elite_mean = elite_theta.mean(dim=1)
        elite_std = torch.clamp(
            elite_theta.std(dim=1, unbiased=False),
            min=args.min_std,
            max=args.max_std,
        )
        mean = (1.0 - args.cem_alpha) * mean + args.cem_alpha * elite_mean
        std = (1.0 - args.cem_alpha) * std + args.cem_alpha * elite_std
        std = torch.clamp(std, min=args.min_std, max=args.max_std)

        for job_idx, job in enumerate(jobs):
            history.append(
                {
                    "generation": generation,
                    "user_id": job.user_id,
                    "target_type": job.target_type,
                    "target_value": job.target_value,
                    "best_feasible": bool(best_feasible[job_idx].item()),
                    "generation_best_feasible": bool(
                        generation_best_feasible[job_idx].item()
                    ),
                    "generation_best_M": float(
                        memory[job_idx, generation_best_idx[job_idx]].item()
                    ),
                    "generation_best_T": float(
                        minutes[job_idx, generation_best_idx[job_idx]].item()
                    ),
                    "elite_mean_rank_score": float(elite_score[job_idx].mean().item()),
                    "mean_std": float(std[job_idx].mean().item()),
                }
            )
        if not args.no_progress:
            feasible_count = int(best_feasible.sum().item())
            print(
                f"generation={generation}/{args.generations} "
                f"feasible_jobs={feasible_count}/{len(jobs)} "
                f"mean_std={std.mean().item():.4f}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    return best_theta, history, time.perf_counter() - start_s


def evaluate_final_policies(
    args: argparse.Namespace,
    *,
    theta: torch.Tensor,
    jobs: Sequence[DirectTargetJob],
    configs: Sequence[SingleCardFSRS6Config],
    device: torch.device,
) -> tuple[list[SimMetrics], float]:
    start_s = time.perf_counter()
    _memory, _minutes, metrics = evaluate_candidate_metrics(
        args,
        theta=theta[:, None, :],
        jobs=jobs,
        configs=configs,
        device=device,
        particles=args.eval_particles,
        seed=args.seed + 290_000,
        exact_memory=True,
    )
    return metrics, time.perf_counter() - start_s


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def point_for_job(
    *,
    job: DirectTargetJob,
    metric: SimMetrics,
    policy_path: Path,
    particles: int,
    runtime_s: float,
) -> EvaluatedPoint:
    return EvaluatedPoint(
        user_id=job.user_id,
        family=DIRECT_CONSTRAINED_FAMILY,
        theta_name=f"target_{job.target_type}",
        theta_value=job.target_value,
        memory=metric.card_expected_retrievability,
        minutes=metric.card_minutes_per_day,
        policy_ref=f"{policy_path}#user={job.user_id}:{job.target_type}={job.target_value:.12g}",
        exact=False,
        eval_stage="confirmed",
        particles=particles,
        runtime_s=runtime_s,
    )


def answer_for_job(job: DirectTargetJob, point: EvaluatedPoint) -> TargetAnswer:
    if job.target_type == "memory":
        feasible = point.memory >= job.target_value
        memory_slack = point.memory - job.target_value
        time_slack = None
    else:
        feasible = point.minutes <= job.target_value
        memory_slack = None
        time_slack = job.target_value - point.minutes
    return TargetAnswer(
        target=job.target,
        family=DIRECT_CONSTRAINED_FAMILY,
        feasible=feasible,
        point=point,
        achieved_memory=point.memory,
        achieved_minutes=point.minutes,
        memory_slack=memory_slack,
        time_slack=time_slack,
        certified=False,
        neighbor_low=None,
        neighbor_high=None,
        mixed_available=False,
        mixed_probability_high=None,
        mixed_memory=None,
        mixed_minutes=None,
    )


def save_policy(
    path: Path,
    *,
    args: argparse.Namespace,
    jobs: Sequence[DirectTargetJob],
    configs: Sequence[SingleCardFSRS6Config],
    theta: torch.Tensor,
    train_runtime_s: float,
    eval_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": DIRECT_CONSTRAINED_FAMILY,
            "policy_family": args.policy_family,
            "parameter_names": list(parameter_names_for_family(args.policy_family)),
            "params_per_policy": parameter_count_for_family(args.policy_family),
            "policy_count": len(jobs),
            "theta_by_job": theta.detach().cpu(),
            "jobs": [
                {
                    "user_idx": job.user_idx,
                    "user_id": job.user_id,
                    "target_type": job.target_type,
                    "target_value": job.target_value,
                }
                for job in jobs
            ],
            "user_configs": [config.checkpoint_payload() for config in configs],
            "days": args.days,
            "min_retention": args.min_retention,
            "max_retention": args.max_retention,
            "generations": args.generations,
            "population_size": args.population_size,
            "elite_count": args.elite_count,
            "train_particles": args.train_particles,
            "eval_particles": args.eval_particles,
            "train_exact_memory": args.train_exact_memory,
            "train_runtime_s": train_runtime_s,
            "eval_runtime_s": eval_runtime_s,
        },
        path,
    )


def write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_csv(path, rows)


def oracle_gap_outputs(
    *,
    args: argparse.Namespace,
    answer_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if args.oracle_target_answers is None:
        return {}, None
    oracle_path = resolve_target_answers_path(args.oracle_target_answers)
    candidate_records = target_answer_records_from_rows(answer_rows)
    oracle_records = read_target_answer_records(oracle_path)
    gaps = compare_target_answers_to_oracle(
        candidate_records,
        oracle_records,
        target_tolerance=args.oracle_gap_target_tolerance,
    )
    gap_rows = [oracle_gap_row(gap) for gap in gaps]
    gap_path = args.out_dir / "target_oracle_gaps.csv"
    gap_metadata_path = args.out_dir / "target_oracle_gaps_metadata.json"
    write_csv(gap_path, gap_rows)
    metadata = {
        "oracle_target_answers": str(oracle_path),
        "target_tolerance": args.oracle_gap_target_tolerance,
        "summary": summarize_oracle_gaps(gaps),
        "outputs": {
            "target_oracle_gaps": str(gap_path),
            "metadata": str(gap_metadata_path),
        },
    }
    gap_metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "target_oracle_gaps": str(gap_path),
        "target_oracle_gaps_metadata": str(gap_metadata_path),
    }, metadata


def main_from_args(args: argparse.Namespace) -> None:
    validate_args(args)
    user_ids = parse_user_ids(args.user_ids)
    target_memories = parse_csv_floats(args.target_memories, name="--target-memories")
    target_times = parse_csv_floats(args.target_times, name="--target-times")
    device = resolve_torch_device(args.torch_device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    configs = load_user_configs(args, user_ids)
    jobs = direct_target_jobs(
        user_ids=user_ids,
        target_memories=target_memories,
        target_times=target_times,
    )

    theta, history, train_runtime_s = optimize_constrained_policies(
        args,
        jobs=jobs,
        configs=configs,
        device=device,
    )
    metrics, eval_runtime_s = evaluate_final_policies(
        args,
        theta=theta,
        jobs=jobs,
        configs=configs,
        device=device,
    )

    policy_path = args.out_dir / "policy.pt"
    save_policy(
        policy_path,
        args=args,
        jobs=jobs,
        configs=configs,
        theta=theta,
        train_runtime_s=train_runtime_s,
        eval_runtime_s=eval_runtime_s,
    )
    runtime_per_job = eval_runtime_s / float(max(1, len(jobs)))
    points = [
        point_for_job(
            job=job,
            metric=metric,
            policy_path=policy_path,
            particles=args.eval_particles,
            runtime_s=runtime_per_job,
        )
        for job, metric in zip(jobs, metrics, strict=True)
    ]
    answers = [
        answer_for_job(job, point) for job, point in zip(jobs, points, strict=True)
    ]
    frontier = empirical_frontier(points)
    segments = frontier_segments(frontier)

    points_path = args.out_dir / "points.csv"
    frontier_path = args.out_dir / "frontier.csv"
    answers_path = args.out_dir / "target_answers.csv"
    segments_path = args.out_dir / "segments.csv"
    history_path = args.out_dir / "train_history.csv"
    metadata_path = args.out_dir / "metadata.json"
    answer_rows = [answer_row(answer) for answer in answers]
    write_csv(points_path, [point_row(point) for point in points])
    write_csv(frontier_path, [point_row(point) for point in frontier])
    write_csv(answers_path, answer_rows)
    write_csv(segments_path, [segment_row(segment) for segment in segments])
    write_history(history_path, history)
    gap_output_paths, gap_metadata = oracle_gap_outputs(
        args=args,
        answer_rows=answer_rows,
    )
    metadata: dict[str, Any] = {
        "family": DIRECT_CONSTRAINED_FAMILY,
        "policy_family": args.policy_family,
        "params_per_policy": parameter_count_for_family(args.policy_family),
        "policy_count": len(jobs),
        "environment": args.env,
        "user_ids": list(user_ids),
        "target_memories": target_memories,
        "target_times": target_times,
        "days": args.days,
        "seed": args.seed,
        "device": str(device),
        "certification_scope": "family_constrained_direct_training",
        "globally_certified": False,
        "family_constrained": True,
        "population_size": args.population_size,
        "elite_count": args.elite_count,
        "generations": args.generations,
        "train_particles": args.train_particles,
        "eval_particles": args.eval_particles,
        "train_runtime_s": train_runtime_s,
        "eval_runtime_s": eval_runtime_s,
        "feasible_targets": sum(1 for answer in answers if answer.feasible),
        "target_count": len(answers),
        "outputs": {
            "policy": str(policy_path),
            "points": str(points_path),
            "frontier": str(frontier_path),
            "target_answers": str(answers_path),
            "segments": str(segments_path),
            "train_history": str(history_path),
            **gap_output_paths,
        },
    }
    if gap_metadata is not None:
        metadata["oracle_gap_report"] = gap_metadata
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote policy: {policy_path}")
    print(f"Wrote points: {points_path}")
    print(f"Wrote frontier: {frontier_path}")
    print(f"Wrote target answers: {answers_path}")
    print(f"Wrote segments: {segments_path}")
    print(f"Wrote train history: {history_path}")
    for output_path in gap_output_paths.values():
        print(f"Wrote oracle gap artifact: {output_path}")
    print(f"Wrote metadata: {metadata_path}")


def main() -> None:
    main_from_args(parse_args())


if __name__ == "__main__":
    main()
