from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.cli.low_param_direct_policy_search_multiuser import (  # noqa: E402
    direct_policy_retention,
    parameter_count_for_family,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (  # noqa: E402
    DEFAULT_DISTILL_HIDDEN_SIZE,
    DEFAULT_DISTILL_NETWORK_DEPTH,
    resolve_torch_device,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (  # noqa: E402
    MultiUserFSRS6SingleCardBatch,
)
from experiments.single_card_tradeoff.cli.target_constrained_direct_policy_search import (  # noqa: E402
    DIRECT_CONSTRAINED_FAMILY,
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
    ConstrainedTarget,
    EvaluatedPoint,
    TargetAnswer,
)
from experiments.single_card_tradeoff.models.policy_runtime import (  # noqa: E402
    RetentionDistillNet,
    predicted_retentions,
    retention_logits_for_retentions,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_SEED  # noqa: E402

POLICY_TYPE = "fsrs6_target_conditioned_retention_distill"
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/target_conditioned_retention_distill"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill per-target constrained direct policies into one target-"
            "conditioned desired-retention network."
        ),
        allow_abbrev=False,
    )
    add_single_card_fsrs6_config_args(parser)
    parser.add_argument("--teacher-policy", type=Path, required=True)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--samples-per-job", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_DISTILL_HIDDEN_SIZE)
    parser.add_argument("--network", choices=["mlp", "residual"], default="residual")
    parser.add_argument(
        "--network-depth", type=int, default=DEFAULT_DISTILL_NETWORK_DEPTH
    )
    parser.add_argument("--eval-particles", type=int, default=10_000)
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
    if not args.teacher_policy.exists():
        raise SystemExit(f"--teacher-policy does not exist: {args.teacher_policy}")
    if args.days <= 1:
        raise SystemExit("--days must be > 1.")
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise SystemExit("--epochs and --steps-per-epoch must be > 0.")
    if args.samples_per_job <= 0:
        raise SystemExit("--samples-per-job must be > 0.")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be > 0.")
    if args.max_grad_norm <= 0.0:
        raise SystemExit("--max-grad-norm must be > 0.")
    if args.eval_particles <= 0:
        raise SystemExit("--eval-particles must be > 0.")
    if args.oracle_gap_target_tolerance < 0.0 or not math.isfinite(
        args.oracle_gap_target_tolerance
    ):
        raise SystemExit("--oracle-gap-target-tolerance must be finite and >= 0.")
    if args.oracle_target_answers is not None:
        oracle_path = resolve_target_answers_path(args.oracle_target_answers)
        if not oracle_path.exists():
            raise SystemExit(f"--oracle-target-answers does not exist: {oracle_path}")


def _target_norm_scale(jobs: Sequence[DirectTargetJob]) -> float:
    return max(1.0, max(float(job.target_value) for job in jobs))


def _target_features(
    jobs: Sequence[DirectTargetJob],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _target_norm_scale(jobs)
    target_type = torch.tensor(
        [1.0 if job.target_type == "memory" else 0.0 for job in jobs],
        device=device,
        dtype=dtype,
    )
    target_norm = torch.tensor(
        [float(job.target_value) / scale for job in jobs],
        device=device,
        dtype=dtype,
    )
    return target_type, target_norm


def _student_obs(
    *,
    s_norm: torch.Tensor,
    d_norm: torch.Tensor,
    job_idx: torch.Tensor,
    target_type: torch.Tensor,
    target_norm: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            s_norm,
            d_norm,
            target_type.index_select(0, job_idx),
            target_norm.index_select(0, job_idx),
        ],
        dim=1,
    )


def load_teacher(
    path: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[DirectTargetJob], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != DIRECT_CONSTRAINED_FAMILY:
        raise SystemExit(f"{path} is not a constrained direct target policy.")
    jobs: list[DirectTargetJob] = []
    for raw in checkpoint["jobs"]:
        target_type = str(raw["target_type"])
        if target_type not in {"memory", "time"}:
            raise SystemExit(f"Invalid target_type in {path}: {target_type}")
        jobs.append(
            DirectTargetJob(
                user_idx=int(raw["user_idx"]),
                user_id=int(raw["user_id"]),
                target=ConstrainedTarget(
                    "memory" if target_type == "memory" else "time",
                    float(raw["target_value"]),
                    user_id=int(raw["user_id"]),
                ),
            )
        )
    theta = checkpoint["theta_by_job"].to(device=device, dtype=torch.float64)
    return theta, jobs, checkpoint


def load_configs(
    args: argparse.Namespace,
    jobs: Sequence[DirectTargetJob],
) -> tuple[list[int], list[SingleCardFSRS6Config], list[DirectTargetJob]]:
    user_ids_by_idx = {job.user_idx: job.user_id for job in jobs}
    user_ids = [user_ids_by_idx[idx] for idx in sorted(user_ids_by_idx)]
    configs: list[SingleCardFSRS6Config] = []
    for user_id in user_ids:
        user_args = argparse.Namespace(**vars(args))
        user_args.user_id = user_id
        configs.append(load_single_card_fsrs6_config(user_args, environment=args.env))
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    remapped_jobs = [replace(job, user_idx=user_to_idx[job.user_id]) for job in jobs]
    return user_ids, configs, remapped_jobs


def train_model(
    args: argparse.Namespace,
    *,
    teacher_theta: torch.Tensor,
    jobs: Sequence[DirectTargetJob],
    teacher_policy_family: str,
    retention_min: float,
    retention_max: float,
    device: torch.device,
) -> tuple[RetentionDistillNet, list[dict[str, Any]], float]:
    model = RetentionDistillNet(
        obs_dim=4,
        hidden_size=args.hidden_size,
        action_count=2,
        architecture=args.network,
        depth=args.network_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 310_000)
    target_type, target_norm = _target_features(
        jobs, device=device, dtype=torch.float64
    )
    job_count = len(jobs)
    history: list[dict[str, Any]] = []
    start_s = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_retention_mae = 0.0
        for _step in range(args.steps_per_epoch):
            job_idx = torch.arange(
                job_count, device=device, dtype=torch.int64
            ).repeat_interleave(args.samples_per_job)
            s_norm = torch.rand(
                job_count * args.samples_per_job,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )
            d_norm = torch.rand(
                job_count * args.samples_per_job,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )
            teacher_obs = torch.stack(
                [s_norm, d_norm, torch.zeros_like(s_norm)],
                dim=1,
            )
            teacher_retention = direct_policy_retention(
                teacher_theta.index_select(0, job_idx),
                teacher_obs,
                policy_family=teacher_policy_family,
                min_retention=retention_min,
                max_retention=retention_max,
            )
            obs = _student_obs(
                s_norm=s_norm,
                d_norm=d_norm,
                job_idx=job_idx,
                target_type=target_type,
                target_norm=target_norm,
            ).to(dtype=torch.float32)
            pred_logit, _aux = model(obs)
            target_logit = retention_logits_for_retentions(
                teacher_retention.to(device=device, dtype=pred_logit.dtype),
                retention_min=retention_min,
                retention_max=retention_max,
            )
            loss = nn.functional.smooth_l1_loss(pred_logit, target_logit)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                pred_retention = predicted_retentions(
                    pred_logit,
                    retention_min=retention_min,
                    retention_max=retention_max,
                )
                epoch_loss += float(loss.item())
                epoch_retention_mae += float(
                    torch.mean(torch.abs(pred_retention - teacher_retention)).item()
                )
        denom = float(args.steps_per_epoch)
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / denom,
                "retention_mae": epoch_retention_mae / denom,
            }
        )
        if not args.no_progress:
            print(
                f"epoch={epoch}/{args.epochs} loss={epoch_loss / denom:.6f} "
                f"retention_mae={epoch_retention_mae / denom:.6f}",
                flush=True,
            )
    return model, history, time.perf_counter() - start_s


def _job_eval_layout(
    jobs: Sequence[DirectTargetJob],
    *,
    particles: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    user_indices: list[int] = []
    job_indices: list[int] = []
    group_indices: list[int] = []
    for job_idx, job in enumerate(jobs):
        user_indices.extend([job.user_idx] * particles)
        job_indices.extend([job_idx] * particles)
        group_indices.extend([job_idx] * particles)
    return (
        user_indices,
        torch.tensor(job_indices, device=device, dtype=torch.int64),
        torch.tensor(group_indices, device=device, dtype=torch.int64),
    )


@torch.inference_mode()
def evaluate_model(
    args: argparse.Namespace,
    *,
    model: RetentionDistillNet,
    jobs: Sequence[DirectTargetJob],
    configs: Sequence[SingleCardFSRS6Config],
    retention_min: float,
    retention_max: float,
    device: torch.device,
) -> tuple[list[Any], float]:
    user_indices, job_idx, group_index = _job_eval_layout(
        jobs,
        particles=args.eval_particles,
        device=device,
    )
    env = MultiUserFSRS6SingleCardBatch(
        days=args.days,
        user_indices=user_indices,
        configs=configs,
        cost_weights=[0.0],
        action_retentions=[retention_min, retention_max],
        device=device,
        dtype=torch.float64,
        seed=args.seed + 390_000,
        exact_memory=True,
        goal_norm_max=1.0,
        reset_on_init=False,
    )
    env.reset_all(goal_values=0.0)
    target_type, target_norm = _target_features(
        jobs, device=device, dtype=torch.float64
    )
    start_s = time.perf_counter()
    while not bool(env.done.all().item()):
        base_obs = env.obs()
        obs = _student_obs(
            s_norm=base_obs[:, 0],
            d_norm=base_obs[:, 1],
            job_idx=job_idx,
            target_type=target_type,
            target_norm=target_norm,
        ).to(dtype=torch.float32)
        pred_logit, _aux = model(obs)
        retention = predicted_retentions(
            pred_logit.to(dtype=torch.float64),
            retention_min=retention_min,
            retention_max=retention_max,
        )
        env.step_retention(retention)
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics = env.metrics_by_group(
        group_index=group_index,
        group_count=len(jobs),
        particles_per_group=args.eval_particles,
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


def save_policy(
    path: Path,
    *,
    args: argparse.Namespace,
    model: RetentionDistillNet,
    jobs: Sequence[DirectTargetJob],
    teacher_policy: Path,
    teacher_policy_family: str,
    retention_min: float,
    retention_max: float,
    train_runtime_s: float,
    eval_runtime_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_type": POLICY_TYPE,
            "state_dict": model.state_dict(),
            "obs_dim": 4,
            "hidden_size": args.hidden_size,
            "network": args.network,
            "network_depth": args.network_depth,
            "teacher_policy": str(teacher_policy),
            "teacher_policy_family": teacher_policy_family,
            "retention_min": retention_min,
            "retention_max": retention_max,
            "jobs": [
                {
                    "user_idx": job.user_idx,
                    "user_id": job.user_id,
                    "target_type": job.target_type,
                    "target_value": job.target_value,
                }
                for job in jobs
            ],
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "samples_per_job": args.samples_per_job,
            "eval_particles": args.eval_particles,
            "train_runtime_s": train_runtime_s,
            "eval_runtime_s": eval_runtime_s,
        },
        path,
    )


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


def answer_for_distilled_job(
    job: DirectTargetJob, point: EvaluatedPoint
) -> TargetAnswer:
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
        family=POLICY_TYPE,
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


def main_from_args(args: argparse.Namespace) -> None:
    validate_args(args)
    device = resolve_torch_device(args.torch_device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    register_run_monitor(
        args,
        device=device,
        output_dir=args.out_dir,
        stage_name=Path(__file__).stem,
    )
    teacher_theta, jobs, teacher = load_teacher(args.teacher_policy, device=device)
    user_ids, configs, jobs = load_configs(args, jobs)
    retention_min = float(teacher.get("min_retention", 0.5))
    retention_max = float(teacher.get("max_retention", 0.98))
    teacher_policy_family = str(teacher.get("policy_family", "bilinear_monotone"))
    model, history, train_runtime_s = train_model(
        args,
        teacher_theta=teacher_theta,
        jobs=jobs,
        teacher_policy_family=teacher_policy_family,
        retention_min=retention_min,
        retention_max=retention_max,
        device=device,
    )
    metrics, eval_runtime_s = evaluate_model(
        args,
        model=model,
        jobs=jobs,
        configs=configs,
        retention_min=retention_min,
        retention_max=retention_max,
        device=device,
    )

    policy_path = args.out_dir / "policy.pt"
    save_policy(
        policy_path,
        args=args,
        model=model,
        jobs=jobs,
        teacher_policy=args.teacher_policy,
        teacher_policy_family=teacher_policy_family,
        retention_min=retention_min,
        retention_max=retention_max,
        train_runtime_s=train_runtime_s,
        eval_runtime_s=eval_runtime_s,
    )
    runtime_per_job = eval_runtime_s / float(max(1, len(jobs)))
    points = [
        EvaluatedPoint(
            user_id=job.user_id,
            family=POLICY_TYPE,
            theta_name=f"target_{job.target_type}",
            theta_value=job.target_value,
            memory=metric.card_expected_retrievability,
            minutes=metric.card_minutes_per_day,
            policy_ref=f"{policy_path}#user={job.user_id}:{job.target_type}={job.target_value:.12g}",
            eval_stage="confirmed",
            particles=args.eval_particles,
            runtime_s=runtime_per_job,
        )
        for job, metric in zip(jobs, metrics, strict=True)
    ]
    answers = [
        answer_row(answer_for_distilled_job(job, point))
        for job, point in zip(jobs, points, strict=True)
    ]
    frontier = empirical_frontier(points)
    segments = frontier_segments(frontier)
    points_path = args.out_dir / "points.csv"
    frontier_path = args.out_dir / "frontier.csv"
    answers_path = args.out_dir / "target_answers.csv"
    segments_path = args.out_dir / "segments.csv"
    history_path = args.out_dir / "train_history.csv"
    metadata_path = args.out_dir / "metadata.json"
    write_csv(points_path, [point_row(point) for point in points])
    write_csv(frontier_path, [point_row(point) for point in frontier])
    write_csv(answers_path, answers)
    write_csv(segments_path, [segment_row(segment) for segment in segments])
    write_csv(history_path, history)
    gap_output_paths, gap_metadata = oracle_gap_outputs(
        args=args,
        answer_rows=answers,
    )
    metadata: dict[str, Any] = {
        "family": POLICY_TYPE,
        "teacher_policy": str(args.teacher_policy),
        "teacher_policy_family": teacher_policy_family,
        "teacher_params_per_policy": parameter_count_for_family(teacher_policy_family),
        "user_ids": list(user_ids),
        "target_count": len(jobs),
        "days": args.days,
        "seed": args.seed,
        "device": str(device),
        "certification_scope": "family_constrained_target_distill",
        "globally_certified": False,
        "family_constrained": True,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "samples_per_job": args.samples_per_job,
        "eval_particles": args.eval_particles,
        "train_runtime_s": train_runtime_s,
        "eval_runtime_s": eval_runtime_s,
        "feasible_targets": sum(1 for row in answers if row["feasible"]),
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
    print(f"Wrote target answers: {answers_path}")
    for output_path in gap_output_paths.values():
        print(f"Wrote oracle gap artifact: {output_path}")
    print(f"Wrote metadata: {metadata_path}")


def main() -> None:
    main_from_args(parse_args())


if __name__ == "__main__":
    main()
