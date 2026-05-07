from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser


SUPPORTED_TRAINERS = {
    "fsrs6_adr_direct",
    "fsrs6_adr_direct_cmaes",
    "fsrs6_adr_direct_dr_grid",
    "fsrs6_adr_delta",
    "fsrs6_adr_delta_cmaes",
}


@dataclass(frozen=True, slots=True)
class InProcessTrainJob:
    user_id: int
    baseline_desired_retention: float
    baseline_desired_retention_token: str
    lambda_value: float
    lambda_token: str
    output_dir: Path
    command_record_path: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True, slots=True)
class InProcessTrainOutcome:
    job: InProcessTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _CommonContext:
    raw_training_sa: dict[str, Any]
    short_term_source: str | None
    learning_steps: list[float]
    relearning_steps: list[float]
    device: torch.device
    benchmark_root: Path
    overrides: dict[str, str]


def resolve_in_process_trainer(
    *, configured_trainer: str, command_template: tuple[str, ...]
) -> str:
    if configured_trainer != "auto":
        if configured_trainer not in SUPPORTED_TRAINERS:
            raise ValueError(f"Unsupported in-process trainer: {configured_trainer}")
        return configured_trainer
    script_names = {Path(item).name for item in command_template}
    if "train_cmaes_fsrs6_adr_direct.py" in script_names:
        return "fsrs6_adr_direct_cmaes"
    if "train_cmaes_fsrs6_adr_delta.py" in script_names:
        return "fsrs6_adr_delta_cmaes"
    if "train_fsrs6_adr_direct_dr_grid.py" in script_names:
        return "fsrs6_adr_direct_dr_grid"
    if "train_fsrs6_adr_delta.py" in script_names:
        return "fsrs6_adr_delta"
    if "train_fsrs6_adr_direct.py" in script_names:
        return "fsrs6_adr_direct"
    raise ValueError(
        "training.batch.trainer = 'auto' requires an in-tree RL trainer script "
        "in training.command_template."
    )


def estimate_lanes_per_job(*, trainer: str, config: ExperimentConfig) -> int:
    from experiments.rl_scheduler.train_cmaes_fsrs6_adr_direct import (
        optimizer_settings_from_mapping,
    )
    from experiments.rl_scheduler.train_cmaes_fsrs6_adr_delta import CMAESSettings
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _read_training_sa,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        _policy_feature_version as _sa_policy_feature_version,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_delta import (
        _baseline_dr_values,
        _dr_batch_size,
        _policy_feature_version,
    )
    from simulator.fsrs6_adr_delta_policy import feature_count

    settings = SASettings.from_mapping(config.training_sa)
    if trainer == "fsrs6_adr_direct":
        return max(1, settings.chains)

    raw_training_sa: dict[str, Any]
    if config.config_path is not None and config.config_path.exists():
        raw_training_sa = dict(_read_training_sa(config.config_path))
    else:
        raw_training_sa = dict(config.training_sa)
    if trainer == "fsrs6_adr_direct_cmaes":
        feature_version = _sa_policy_feature_version(raw_training_sa)
        optimizer = optimizer_settings_from_mapping(
            config.training_optimizer,
            settings=settings,
            feature_version=feature_version,
        )
        return max(1, optimizer.population_size)
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    dr_lanes = min(len(baseline_dr_values), dr_batch_size)
    if trainer in {"fsrs6_adr_delta", "fsrs6_adr_direct_dr_grid"}:
        return max(len(baseline_dr_values), dr_lanes * settings.chains)
    if trainer == "fsrs6_adr_delta_cmaes":
        feature_version = _policy_feature_version(raw_training_sa)
        optimizer = CMAESSettings.from_mapping(
            config.training_optimizer,
            coefficient_count=feature_count(feature_version),
            coefficient_min=settings.coefficient_min,
            coefficient_max=settings.coefficient_max,
        )
        return max(len(baseline_dr_values), dr_lanes * optimizer.population_size)
    raise ValueError(f"Unsupported in-process trainer: {trainer}")


def run_in_process_train_batch(
    *,
    trainer: str,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    if not jobs:
        return []
    if trainer == "fsrs6_adr_direct":
        return _run_fsrs6_adr_direct_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_adr_direct_cmaes":
        return _run_fsrs6_adr_direct_cmaes_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_adr_direct_dr_grid":
        return _run_fsrs6_adr_direct_dr_grid_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_adr_delta":
        return _run_fsrs6_adr_delta_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_adr_delta_cmaes":
        return _run_fsrs6_adr_delta_cmaes_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    raise ValueError(f"Unsupported in-process trainer: {trainer}")


def _common_context(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    settings: Any,
) -> _CommonContext:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import _read_training_sa

    raw_training_sa = dict(_read_training_sa(config_path))
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_sa.get("learning_steps"),
        relearning_steps=raw_training_sa.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    return _CommonContext(
        raw_training_sa=raw_training_sa,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        device=torch.device(settings.torch_device),
        benchmark_root=resolve_benchmark_root(repo_root, None).resolve(),
        overrides=parse_result_overrides(None),
    )


def _progress_for_jobs(
    *, jobs: list[InProcessTrainJob], config_path: Path
) -> list[Any]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import TrainingProgress

    progresses = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(job.output_dir / "training_progress.jsonl")
        progress.write(
            "started",
            config_path=str(config_path),
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            execution_mode="in_process_batch",
        )
        progresses.append(progress)
    return progresses


def _build_bundle_for_lanes(
    *,
    config: ExperimentConfig,
    settings: Any,
    lane_user_ids: list[int],
    ctx: _CommonContext,
) -> Any:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import _build_bundle

    return _build_bundle(
        config=config,
        settings=settings,
        lane_user_ids=lane_user_ids,
        benchmark_root=ctx.benchmark_root,
        overrides=ctx.overrides,
        benchmark_partition=None,
        button_usage=DEFAULT_BUTTON_USAGE_PATH,
        device=ctx.device,
        short_term_source=ctx.short_term_source,
        learning_steps=ctx.learning_steps,
        relearning_steps=ctx.relearning_steps,
    )


def _evaluate_fsrs6_baselines_for_lanes(
    *,
    config: ExperimentConfig,
    settings: Any,
    bundle: Any,
    desired_retention: torch.Tensor,
    seed: int,
) -> list[Any]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import _metrics_from_stats

    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=desired_retention,
        bounds=Bounds(),
        priority_mode=config.simulation.scheduler_priority,
        device=bundle.device,
        dtype=torch.float32,
    )
    stats = simulate_multiuser(
        days=config.simulation.days,
        deck_size=config.simulation.deck,
        env_ops=bundle.env_ops,
        sched_ops=sched_ops,
        behavior=bundle.behavior,
        cost_model=bundle.cost_model,
        seed=seed,
        device=bundle.device,
        dtype=torch.float32,
        fuzz=config.simulation.fuzz,
        priority_mode=config.simulation.priority,
        progress=False,
        short_term_source=bundle.short_term_source,
        learning_steps=bundle.learning_steps,
        relearning_steps=bundle.relearning_steps,
        short_term_threshold=settings.short_term_threshold,
        short_term_loops_limit=settings.short_term_loops_limit,
    )
    return [_metrics_from_stats(item) for item in stats]


def _run_fsrs6_adr_direct_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _clamp_coefficients,
        _metrics_from_stats,
        _policy_feature_version,
        _relative_gain,
        _score,
        _temperature,
    )
    from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
    from simulator.schedulers.fsrs6_adr_direct import FSRS6ADRDirectBatchSchedulerOps

    settings = SASettings.from_mapping(config.training_sa)
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    feature_version = _policy_feature_version(ctx.raw_training_sa)
    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    for job, progress in zip(jobs, progresses, strict=True):
        effective_settings = replace(
            settings, baseline_desired_retention=job.baseline_desired_retention
        )
        progress.write(
            "config_loaded",
            settings=asdict(effective_settings),
            feature_version=feature_version,
            simulation=config.simulation.to_dict(),
            seed=config.seed,
        )
        progress.write(
            "device_resolved", device=ctx.device, torch_device=str(ctx.device)
        )

    baseline_lane_users = [job.user_id for job in jobs]
    baseline_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=baseline_lane_users,
        ctx=ctx,
    )
    desired = torch.tensor(
        [job.baseline_desired_retention for job in jobs],
        device=baseline_bundle.device,
        dtype=torch.float32,
    )
    baselines = _evaluate_fsrs6_baselines_for_lanes(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        desired_retention=desired,
        seed=config.seed,
    )
    for baseline, progress in zip(baselines, progresses, strict=True):
        progress.write(
            "baseline_evaluated",
            device=baseline_bundle.device,
            effective_lanes=1,
            metrics=asdict(baseline),
        )

    train_lane_users = [
        job.user_id for job in jobs for _chain in range(settings.chains)
    ]
    train_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=train_lane_users,
        ctx=ctx,
    )
    for progress in progresses:
        progress.write(
            "train_bundle_built",
            device=train_bundle.device,
            effective_lanes=settings.chains,
            batch_effective_lanes=len(train_lane_users),
        )

    generators = []
    current_by_job = []
    for job in jobs:
        generator = torch.Generator(device=train_bundle.device)
        generator.manual_seed(config.seed)
        generators.append(generator)
        base_policy = FSRS6ADRDirectPolicy.baseline(
            desired_retention=job.baseline_desired_retention,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            feature_version=feature_version,
        )
        current = torch.tensor(
            base_policy.coefficients,
            dtype=torch.float32,
            device=train_bundle.device,
        ).repeat(settings.chains, 1)
        if settings.chains > 1:
            current[1:] = _clamp_coefficients(
                current[1:]
                + torch.randn(
                    current[1:].shape,
                    device=train_bundle.device,
                    generator=generator,
                )
                * settings.proposal_scale,
                settings,
            )
        current_by_job.append(current)
    current = torch.stack(current_by_job, dim=0)

    def evaluate(coefficients_by_job: torch.Tensor) -> list[list[Any]]:
        flat_coefficients = coefficients_by_job.reshape(
            len(jobs) * settings.chains, coefficients_by_job.shape[-1]
        )
        template = FSRS6ADRDirectPolicy.baseline(
            desired_retention=settings.baseline_desired_retention,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            feature_version=feature_version,
        )
        sched_ops = FSRS6ADRDirectBatchSchedulerOps(
            weights=train_bundle.scheduler_weights,
            policy=template,
            coefficients=flat_coefficients,
            bounds=Bounds(),
            priority_mode=config.simulation.scheduler_priority,
            device=train_bundle.device,
            dtype=torch.float32,
        )
        stats = simulate_multiuser(
            days=config.simulation.days,
            deck_size=config.simulation.deck,
            env_ops=train_bundle.env_ops,
            sched_ops=sched_ops,
            behavior=train_bundle.behavior,
            cost_model=train_bundle.cost_model,
            seed=config.seed,
            device=train_bundle.device,
            dtype=torch.float32,
            fuzz=config.simulation.fuzz,
            priority_mode=config.simulation.priority,
            progress=False,
            short_term_source=train_bundle.short_term_source,
            learning_steps=train_bundle.learning_steps,
            relearning_steps=train_bundle.relearning_steps,
            short_term_threshold=settings.short_term_threshold,
            short_term_loops_limit=settings.short_term_loops_limit,
        )
        metrics = [_metrics_from_stats(item) for item in stats]
        return [
            metrics[index * settings.chains : (index + 1) * settings.chains]
            for index in range(len(jobs))
        ]

    current_metrics = evaluate(current)
    current_scores = torch.tensor(
        [
            [
                _score(metric, baselines[job_index], jobs[job_index].lambda_value)
                for metric in current_metrics[job_index]
            ]
            for job_index in range(len(jobs))
        ],
        device=train_bundle.device,
        dtype=torch.float32,
    )
    best_coefficients = []
    best_metrics = []
    best_scores = []
    histories: list[list[dict[str, float]]] = [[] for _ in jobs]
    for job_index, progress in enumerate(progresses):
        best_idx = int(torch.argmax(current_scores[job_index]).item())
        best_coefficients.append(current[job_index, best_idx].detach().clone())
        best_metrics.append(current_metrics[job_index][best_idx])
        best_score = float(current_scores[job_index, best_idx].item())
        best_scores.append(best_score)
        progress.write(
            "initial_candidates_evaluated",
            device=train_bundle.device,
            effective_lanes=settings.chains,
            batch_effective_lanes=len(train_lane_users),
            best_score=best_score,
            best=asdict(best_metrics[job_index]),
            best_relative_memorized_gain=_relative_gain(
                best_metrics[job_index].memorized_average,
                baselines[job_index].memorized_average,
            ),
            best_relative_efficiency_gain=_relative_gain(
                best_metrics[job_index].memorized_per_minute,
                baselines[job_index].memorized_per_minute,
            ),
        )

    for iteration in range(settings.iterations):
        temp = _temperature(settings, iteration)
        proposals = []
        for job_index, generator in enumerate(generators):
            proposals.append(
                _clamp_coefficients(
                    current[job_index]
                    + torch.randn(
                        current[job_index].shape,
                        device=train_bundle.device,
                        generator=generator,
                    )
                    * settings.proposal_scale,
                    settings,
                )
            )
        proposal = torch.stack(proposals, dim=0)
        proposal_metrics = evaluate(proposal)
        proposal_scores = torch.tensor(
            [
                [
                    _score(
                        metric,
                        baselines[job_index],
                        jobs[job_index].lambda_value,
                    )
                    for metric in proposal_metrics[job_index]
                ]
                for job_index in range(len(jobs))
            ],
            device=train_bundle.device,
            dtype=torch.float32,
        )
        delta = proposal_scores - current_scores
        accept_prob = torch.exp(delta / max(temp, 1e-9))
        accept_rows = []
        for job_index, generator in enumerate(generators):
            random_values = torch.rand(
                (settings.chains,), device=train_bundle.device, generator=generator
            )
            accept_rows.append(
                (delta[job_index] >= 0) | (random_values < accept_prob[job_index])
            )
        accept = torch.stack(accept_rows, dim=0)
        for job_index in range(len(jobs)):
            if bool(torch.any(accept[job_index]).item()):
                current[job_index, accept[job_index]] = proposal[
                    job_index, accept[job_index]
                ]
                current_scores[job_index, accept[job_index]] = proposal_scores[
                    job_index, accept[job_index]
                ]
                for idx in (
                    torch.nonzero(accept[job_index], as_tuple=False).flatten().tolist()
                ):
                    current_metrics[job_index][int(idx)] = proposal_metrics[job_index][
                        int(idx)
                    ]

            iteration_best_idx = int(torch.argmax(current_scores[job_index]).item())
            iteration_best_score = float(
                current_scores[job_index, iteration_best_idx].item()
            )
            if iteration_best_score > best_scores[job_index]:
                best_scores[job_index] = iteration_best_score
                best_coefficients[job_index] = (
                    current[job_index, iteration_best_idx].detach().clone()
                )
                best_metrics[job_index] = current_metrics[job_index][iteration_best_idx]
            history_entry = {
                "iteration": float(iteration),
                "temperature": float(temp),
                "best_score": best_scores[job_index],
                "best_relative_memorized_gain": _relative_gain(
                    best_metrics[job_index].memorized_average,
                    baselines[job_index].memorized_average,
                ),
                "best_relative_efficiency_gain": _relative_gain(
                    best_metrics[job_index].memorized_per_minute,
                    baselines[job_index].memorized_per_minute,
                ),
            }
            histories[job_index].append(history_entry)
            progresses[job_index].write(
                "annealing_iteration",
                device=train_bundle.device,
                iteration=iteration,
                temperature=float(temp),
                accepted_count=int(accept[job_index].sum().item()),
                effective_lanes=settings.chains,
                batch_effective_lanes=len(train_lane_users),
                best_score=best_scores[job_index],
                best_relative_memorized_gain=history_entry[
                    "best_relative_memorized_gain"
                ],
                best_relative_efficiency_gain=history_entry[
                    "best_relative_efficiency_gain"
                ],
            )

    outcomes = []
    for job_index, job in enumerate(jobs):
        progresses[job_index].write(
            "annealing_completed",
            device=train_bundle.device,
            best=asdict(best_metrics[job_index]),
            iterations=len(histories[job_index]),
        )
        metadata_path = _write_fsrs6_adr_direct_artifact(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=replace(
                settings, baseline_desired_retention=job.baseline_desired_retention
            ),
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            baseline=baselines[job_index],
            best=best_metrics[job_index],
            best_coefficients=best_coefficients[job_index].detach().cpu(),
            feature_version=feature_version,
            history=histories[job_index],
        )
        rel_mem = _relative_gain(
            best_metrics[job_index].memorized_average,
            baselines[job_index].memorized_average,
        )
        rel_eff = _relative_gain(
            best_metrics[job_index].memorized_per_minute,
            baselines[job_index].memorized_per_minute,
        )
        passed = rel_mem > 0.0 and rel_eff > 0.0
        progresses[job_index].write(
            "artifacts_written",
            device=train_bundle.device,
            passed=passed,
            metadata_path=str(metadata_path),
            metrics_path=str(job.output_dir / "metrics.json"),
            policy_path=str(job.output_dir / "policy.json"),
        )
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=passed,
                artifact_paths=(metadata_path,),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes


def _write_fsrs6_adr_direct_artifact(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: Any,
    user_id: int,
    lambda_value: float,
    training_command_path: Path | None,
    baseline: Any,
    best: Any,
    best_coefficients: torch.Tensor,
    feature_version: str,
    history: list[dict[str, float]],
) -> Path:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        _artifact_id,
        _git_commit,
        _relative_gain,
        _write_json,
    )
    from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy

    output_dir.mkdir(parents=True, exist_ok=True)
    rel_mem = _relative_gain(best.memorized_average, baseline.memorized_average)
    rel_eff = _relative_gain(best.memorized_per_minute, baseline.memorized_per_minute)
    passed = rel_mem > 0.0 and rel_eff > 0.0
    policy = FSRS6ADRDirectPolicy(
        coefficients=tuple(float(v) for v in best_coefficients.tolist()),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        baseline_desired_retention=settings.baseline_desired_retention,
        feature_version=feature_version,
        title=(
            f"fsrs6_adr_direct_u{user_id}_dr_"
            f"{settings.baseline_desired_retention:.2f}_lambda_{lambda_value:g}"
        ),
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)
    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "passed_overfit_gate": passed,
            "gate": {
                "memorized_average_gt_baseline": rel_mem > 0.0,
                "memorized_per_minute_gt_baseline": rel_eff > 0.0,
                "relative_memorized_gain": rel_mem,
                "relative_efficiency_gain": rel_eff,
            },
            "baseline": asdict(baseline),
            "best": asdict(best),
            "settings": asdict(settings),
            "feature_version": feature_version,
            "history": history,
        },
    )
    metadata_path = output_dir / "metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "scheduler-policy",
            "artifact_id": _artifact_id(
                user_id,
                lambda_value,
                settings.baseline_desired_retention,
                config.seed,
            ),
            "family": config.family,
            "scheduler_name": "fsrs6_adr_direct",
            "environment": config.simulation.environment,
            "engine": config.simulation.engine,
            "training_user_ids": [user_id],
            "validation_user_ids": list(config.users.validation),
            "seed": config.seed,
            "policy_path": "policy.json",
            "feature_version": feature_version,
            "action_space": "sd_retention_function",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "code_commit": _git_commit(),
            "lambda_value": lambda_value,
            "baseline_desired_retention": settings.baseline_desired_retention,
            "config_snapshot_path": str(config_path.resolve()),
            "training_command_path": str(training_command_path)
            if training_command_path
            else None,
            "metrics_path": "metrics.json",
            "capabilities": ["event", "vectorized", "batched"],
        },
    )
    return metadata_path


def _run_fsrs6_adr_direct_cmaes_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    import cma

    from experiments.rl_scheduler.train_cmaes_fsrs6_adr_direct import (
        CMAESFSRS6TrainingResult,
        optimizer_settings_from_mapping,
        write_artifact,
    )
    from experiments.rl_scheduler.train_cmaes_fsrs6_adr_delta import _optimizer_seed
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _metrics_from_stats,
        _policy_feature_version,
        _relative_gain,
        _score,
    )
    from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
    from simulator.schedulers.fsrs6_adr_direct import FSRS6ADRDirectBatchSchedulerOps

    settings = SASettings.from_mapping(config.training_sa)
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    feature_version = _policy_feature_version(ctx.raw_training_sa)
    effective_settings_by_job = [
        replace(settings, baseline_desired_retention=job.baseline_desired_retention)
        for job in jobs
    ]
    optimizer_settings_by_job = [
        optimizer_settings_from_mapping(
            config.training_optimizer,
            settings=effective_settings,
            feature_version=feature_version,
        )
        for effective_settings in effective_settings_by_job
    ]
    optimizer_settings = optimizer_settings_by_job[0]
    optimizer_seeds = [
        _optimizer_seed(
            config=config,
            settings=optimizer_settings_by_job[job_index],
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        for job_index, job in enumerate(jobs)
    ]

    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    for progress, effective_settings, optimizer_settings_for_job, seed in zip(
        progresses,
        effective_settings_by_job,
        optimizer_settings_by_job,
        optimizer_seeds,
        strict=True,
    ):
        progress.write(
            "config_loaded",
            settings=asdict(effective_settings),
            optimizer=optimizer_settings_for_job.to_dict(),
            optimizer_seed=seed,
            feature_version=feature_version,
            simulation=config.simulation.to_dict(),
            seed=config.seed,
        )
        progress.write(
            "device_resolved", device=ctx.device, torch_device=str(ctx.device)
        )

    baseline_lane_users = [job.user_id for job in jobs]
    baseline_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=baseline_lane_users,
        ctx=ctx,
    )
    desired = torch.tensor(
        [job.baseline_desired_retention for job in jobs],
        device=baseline_bundle.device,
        dtype=torch.float32,
    )
    baselines = _evaluate_fsrs6_baselines_for_lanes(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        desired_retention=desired,
        seed=config.seed,
    )
    for baseline, progress in zip(baselines, progresses, strict=True):
        progress.write(
            "baseline_evaluated",
            device=baseline_bundle.device,
            effective_lanes=1,
            metrics=asdict(baseline),
        )

    train_lane_users = [
        job.user_id
        for job in jobs
        for _candidate in range(optimizer_settings.population_size)
    ]
    train_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=train_lane_users,
        ctx=ctx,
    )
    for progress in progresses:
        progress.write(
            "train_bundle_built",
            device=train_bundle.device,
            effective_lanes=optimizer_settings.population_size,
            batch_effective_lanes=len(train_lane_users),
        )

    strategies = []
    for settings_for_job, seed in zip(
        optimizer_settings_by_job, optimizer_seeds, strict=True
    ):
        strategies.append(
            cma.CMAEvolutionStrategy(
                list(settings_for_job.initial_mean),
                settings_for_job.sigma0,
                {
                    "bounds": [
                        list(settings_for_job.bounds[0]),
                        list(settings_for_job.bounds[1]),
                    ],
                    "popsize": settings_for_job.population_size,
                    "seed": seed,
                    "verb_disp": 0,
                    "verb_log": 0,
                    "verbose": -9,
                },
            )
        )

    def evaluate(coefficients_by_job: torch.Tensor) -> list[list[Any]]:
        flat_coefficients = coefficients_by_job.reshape(
            len(jobs) * optimizer_settings.population_size,
            coefficients_by_job.shape[-1],
        )
        template = FSRS6ADRDirectPolicy.baseline(
            desired_retention=settings.baseline_desired_retention,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            feature_version=feature_version,
        )
        sched_ops = FSRS6ADRDirectBatchSchedulerOps(
            weights=train_bundle.scheduler_weights,
            policy=template,
            coefficients=flat_coefficients,
            bounds=Bounds(),
            priority_mode=config.simulation.scheduler_priority,
            device=train_bundle.device,
            dtype=torch.float32,
        )
        stats = simulate_multiuser(
            days=config.simulation.days,
            deck_size=config.simulation.deck,
            env_ops=train_bundle.env_ops,
            sched_ops=sched_ops,
            behavior=train_bundle.behavior,
            cost_model=train_bundle.cost_model,
            seed=config.seed,
            device=train_bundle.device,
            dtype=torch.float32,
            fuzz=config.simulation.fuzz,
            priority_mode=config.simulation.priority,
            progress=False,
            short_term_source=train_bundle.short_term_source,
            learning_steps=train_bundle.learning_steps,
            relearning_steps=train_bundle.relearning_steps,
            short_term_threshold=settings.short_term_threshold,
            short_term_loops_limit=settings.short_term_loops_limit,
        )
        metrics = [_metrics_from_stats(item) for item in stats]
        return [
            metrics[
                index * optimizer_settings.population_size : (index + 1)
                * optimizer_settings.population_size
            ]
            for index in range(len(jobs))
        ]

    best_coefficients: list[torch.Tensor | None] = [None for _job in jobs]
    best_metrics: list[Any | None] = [None for _job in jobs]
    best_scores = [float("-inf") for _job in jobs]
    histories: list[list[dict[str, float]]] = [[] for _job in jobs]

    for generation in range(optimizer_settings.generations):
        solutions_by_job = []
        for job_index, strategy in enumerate(strategies):
            optimizer_settings_for_job = optimizer_settings_by_job[job_index]
            solutions = [list(map(float, item)) for item in strategy.ask()]
            if len(solutions) != optimizer_settings_for_job.population_size:
                raise RuntimeError(
                    "CMA-ES returned an unexpected population size: "
                    f"{len(solutions)} != "
                    f"{optimizer_settings_for_job.population_size}."
                )
            if generation == 0:
                solutions[0] = list(optimizer_settings_for_job.initial_mean)
            solutions_by_job.append(solutions)
        coefficients = torch.tensor(
            solutions_by_job,
            device=train_bundle.device,
            dtype=torch.float32,
        )
        metrics_by_job = evaluate(coefficients)
        for job_index, strategy in enumerate(strategies):
            scores = [
                _score(metric, baselines[job_index], jobs[job_index].lambda_value)
                for metric in metrics_by_job[job_index]
            ]
            strategy.tell(solutions_by_job[job_index], [-score for score in scores])
            generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
            generation_best = metrics_by_job[job_index][generation_best_idx]
            generation_best_score = float(scores[generation_best_idx])
            if generation_best_score > best_scores[job_index]:
                best_scores[job_index] = generation_best_score
                best_coefficients[job_index] = (
                    coefficients[job_index, generation_best_idx].detach().clone()
                )
                best_metrics[job_index] = generation_best
            history_entry = {
                "generation": float(generation),
                "sigma": float(strategy.sigma),
                "best_score": float(best_scores[job_index]),
                "generation_best_score": generation_best_score,
                "mean_score": float(sum(scores) / max(len(scores), 1)),
                "generation_best_relative_memorized_gain": _relative_gain(
                    generation_best.memorized_average,
                    baselines[job_index].memorized_average,
                ),
                "generation_best_relative_efficiency_gain": _relative_gain(
                    generation_best.memorized_per_minute,
                    baselines[job_index].memorized_per_minute,
                ),
            }
            histories[job_index].append(history_entry)
            progresses[job_index].write(
                "cmaes_generation",
                device=train_bundle.device,
                effective_lanes=optimizer_settings.population_size,
                batch_effective_lanes=len(train_lane_users),
                **history_entry,
            )

    outcomes = []
    for job_index, job in enumerate(jobs):
        best_coefficients_for_job = best_coefficients[job_index]
        best = best_metrics[job_index]
        if best_coefficients_for_job is None or best is None:
            raise RuntimeError("CMA-ES did not evaluate any candidates.")
        rel_mem = _relative_gain(
            best.memorized_average,
            baselines[job_index].memorized_average,
        )
        rel_eff = _relative_gain(
            best.memorized_per_minute,
            baselines[job_index].memorized_per_minute,
        )
        result = CMAESFSRS6TrainingResult(
            baseline=baselines[job_index],
            best=best,
            best_coefficients=best_coefficients_for_job.detach().cpu(),
            best_score=best_scores[job_index],
            history=histories[job_index],
            passed=rel_mem > 0.0 and rel_eff > 0.0,
        )
        progresses[job_index].write(
            "cmaes_completed",
            device=train_bundle.device,
            best_score=result.best_score,
            best=asdict(result.best),
            relative_memorized_gain=rel_mem,
            relative_efficiency_gain=rel_eff,
            generations=len(result.history),
        )
        policy_path, metrics_path, metadata_path = write_artifact(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=effective_settings_by_job[job_index],
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            feature_version=feature_version,
            result=result,
            optimizer_settings=optimizer_settings_by_job[job_index],
            optimizer_seed=optimizer_seeds[job_index],
        )
        progresses[job_index].write(
            "artifacts_written",
            device=train_bundle.device,
            passed=result.passed,
            policy_path=str(policy_path),
            metrics_path=str(metrics_path),
            metadata_path=str(metadata_path),
        )
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=result.passed,
                artifact_paths=(metadata_path,),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes


def _evaluate_dr_conditioned_batch(
    *,
    config: ExperimentConfig,
    settings: Any,
    bundle: Any,
    jobs: list[InProcessTrainJob],
    baseline_dr_values: tuple[float, ...],
    dr_batch_size: int,
    baselines_by_job: list[list[Any]],
    coefficients: torch.Tensor,
    feature_version: str,
    seed: int,
) -> list[list[Any]]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        _metrics_from_stats,
        _relative_gain,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_delta import (
        ChainEvaluation,
        _iter_dr_chunks,
        _pad_tuple,
        _score_from_relative_gains,
    )
    from simulator.fsrs6_adr_delta_policy import FSRS6ADRDeltaPolicy
    from simulator.schedulers.fsrs6_adr_delta import FSRS6ADRDeltaBatchSchedulerOps

    job_count = len(jobs)
    candidate_count = int(coefficients.shape[1])
    coefficient_count = int(coefficients.shape[2])
    dr_count = len(baseline_dr_values)
    template = FSRS6ADRDeltaPolicy.baseline(
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    if coefficient_count != template.feature_count:
        raise ValueError(
            "DR-conditioned coefficients have an unexpected feature count."
        )

    metrics_by_job_candidate: list[list[list[Any]]] = [
        [[] for _candidate in range(candidate_count)] for _job in jobs
    ]
    rel_mem_sums = [[0.0 for _candidate in range(candidate_count)] for _job in jobs]
    rel_eff_sums = [[0.0 for _candidate in range(candidate_count)] for _job in jobs]

    for chunk_dr_values, _chunk_baselines in _iter_dr_chunks(
        baseline_dr_values, baselines_by_job[0], dr_batch_size
    ):
        actual_count = len(chunk_dr_values)
        padded_dr_values = _pad_tuple(chunk_dr_values, dr_batch_size)
        desired_retention = torch.tensor(
            [
                dr
                for _job in jobs
                for _candidate in range(candidate_count)
                for dr in padded_dr_values
            ],
            device=bundle.device,
            dtype=torch.float32,
        )
        lane_coefficients = (
            coefficients[:, :, None, :]
            .expand(job_count, candidate_count, dr_batch_size, coefficient_count)
            .reshape(job_count * candidate_count * dr_batch_size, coefficient_count)
        )
        sched_ops = FSRS6ADRDeltaBatchSchedulerOps(
            weights=bundle.scheduler_weights,
            desired_retention=desired_retention,
            policy=template,
            coefficients=lane_coefficients,
            bounds=Bounds(),
            priority_mode=config.simulation.scheduler_priority,
            device=bundle.device,
            dtype=torch.float32,
        )
        stats = simulate_multiuser(
            days=config.simulation.days,
            deck_size=config.simulation.deck,
            env_ops=bundle.env_ops,
            sched_ops=sched_ops,
            behavior=bundle.behavior,
            cost_model=bundle.cost_model,
            seed=seed,
            device=bundle.device,
            dtype=torch.float32,
            fuzz=config.simulation.fuzz,
            priority_mode=config.simulation.priority,
            progress=False,
            short_term_source=bundle.short_term_source,
            learning_steps=bundle.learning_steps,
            relearning_steps=bundle.relearning_steps,
            short_term_threshold=settings.short_term_threshold,
            short_term_loops_limit=settings.short_term_loops_limit,
        )
        metrics = [_metrics_from_stats(item) for item in stats]
        for job_index in range(job_count):
            for candidate_index in range(candidate_count):
                start = (job_index * candidate_count + candidate_index) * dr_batch_size
                chunk_metrics = metrics[start : start + actual_count]
                metrics_by_job_candidate[job_index][candidate_index].extend(
                    chunk_metrics
                )
                dr_offset = baseline_dr_values.index(chunk_dr_values[0])
                chunk_baselines = baselines_by_job[job_index][
                    dr_offset : dr_offset + actual_count
                ]
                for metric, baseline in zip(
                    chunk_metrics, chunk_baselines, strict=True
                ):
                    rel_mem_sums[job_index][candidate_index] += _relative_gain(
                        metric.memorized_average,
                        baseline.memorized_average,
                    )
                    rel_eff_sums[job_index][candidate_index] += _relative_gain(
                        metric.memorized_per_minute,
                        baseline.memorized_per_minute,
                    )

    evaluations_by_job: list[list[Any]] = []
    for job_index, job in enumerate(jobs):
        evaluations = []
        for candidate_index in range(candidate_count):
            candidate_metrics = metrics_by_job_candidate[job_index][candidate_index]
            if len(candidate_metrics) != dr_count:
                raise AssertionError("DR-conditioned batch evaluation missed metrics.")
            mean_rel_mem = rel_mem_sums[job_index][candidate_index] / max(dr_count, 1)
            mean_rel_eff = rel_eff_sums[job_index][candidate_index] / max(dr_count, 1)
            evaluations.append(
                ChainEvaluation(
                    metrics_by_dr=candidate_metrics,
                    mean_relative_memorized_gain=mean_rel_mem,
                    mean_relative_efficiency_gain=mean_rel_eff,
                    score=_score_from_relative_gains(
                        mean_rel_mem,
                        mean_rel_eff,
                        job.lambda_value,
                    ),
                )
            )
        evaluations_by_job.append(evaluations)
    return evaluations_by_job


def _evaluate_baseline_grid(
    *,
    config: ExperimentConfig,
    settings: Any,
    jobs: list[InProcessTrainJob],
    baseline_dr_values: tuple[float, ...],
    ctx: _CommonContext,
) -> tuple[Any, list[list[Any]]]:
    lane_user_ids = [job.user_id for job in jobs for _dr in baseline_dr_values]
    bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=lane_user_ids,
        ctx=ctx,
    )
    desired = torch.tensor(
        [dr for _job in jobs for dr in baseline_dr_values],
        device=bundle.device,
        dtype=torch.float32,
    )
    metrics = _evaluate_fsrs6_baselines_for_lanes(
        config=config,
        settings=settings,
        bundle=bundle,
        desired_retention=desired,
        seed=config.seed,
    )
    by_job = [
        metrics[index * len(baseline_dr_values) : (index + 1) * len(baseline_dr_values)]
        for index in range(len(jobs))
    ]
    return bundle, by_job


def _run_fsrs6_adr_delta_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _clamp_coefficients,
        _read_training_sa,
        _temperature,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_delta import (
        DRConditionedTrainingResult,
        _baseline_dr_values,
        _dr_batch_size,
        _initial_coefficients,
        _policy_feature_version,
        _write_artifact,
    )
    from simulator.fsrs6_adr_delta_policy import feature_count

    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = dict(_read_training_sa(config_path))
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    feature_version = _policy_feature_version(raw_training_sa)
    coefficient_count = feature_count(feature_version)
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    for progress in progresses:
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            feature_version=feature_version,
            baseline_desired_retention_values=list(baseline_dr_values),
            dr_batch_size=dr_batch_size,
        )
        progress.write(
            "device_resolved", device=ctx.device, torch_device=str(ctx.device)
        )

    baseline_bundle, baselines_by_job = _evaluate_baseline_grid(
        config=config,
        settings=settings,
        jobs=jobs,
        baseline_dr_values=baseline_dr_values,
        ctx=ctx,
    )
    for baselines, progress in zip(baselines_by_job, progresses, strict=True):
        progress.write(
            "baselines_evaluated",
            device=baseline_bundle.device,
            effective_lanes=len(baseline_dr_values),
            batch_effective_lanes=len(jobs) * len(baseline_dr_values),
            metrics=[
                {"baseline_desired_retention": dr, **asdict(metric)}
                for dr, metric in zip(baseline_dr_values, baselines, strict=True)
            ],
        )

    lane_user_ids = [
        job.user_id
        for job in jobs
        for _chain in range(settings.chains)
        for _dr in range(dr_batch_size)
    ]
    train_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=lane_user_ids,
        ctx=ctx,
    )
    for progress in progresses:
        progress.write(
            "train_bundle_built",
            device=train_bundle.device,
            effective_lanes=dr_batch_size * settings.chains,
            grid_lanes=len(baseline_dr_values) * settings.chains,
            batch_effective_lanes=len(lane_user_ids),
        )

    generators = []
    currents = []
    for _job in jobs:
        generator = torch.Generator(device=train_bundle.device)
        generator.manual_seed(config.seed)
        generators.append(generator)
        currents.append(
            _initial_coefficients(
                settings=settings,
                coefficient_count=coefficient_count,
                device=train_bundle.device,
                generator=generator,
            )
        )
    current = torch.stack(currents, dim=0)
    current_evaluations = _evaluate_dr_conditioned_batch(
        config=config,
        settings=settings,
        bundle=train_bundle,
        jobs=jobs,
        baseline_dr_values=baseline_dr_values,
        dr_batch_size=dr_batch_size,
        baselines_by_job=baselines_by_job,
        coefficients=current,
        feature_version=feature_version,
        seed=config.seed,
    )
    current_scores = torch.tensor(
        [
            [evaluation.score for evaluation in evaluations]
            for evaluations in current_evaluations
        ],
        device=train_bundle.device,
        dtype=torch.float32,
    )
    best_coefficients = []
    best_evaluations = []
    best_scores = []
    histories: list[list[dict[str, float]]] = [[] for _job in jobs]
    for job_index, progress in enumerate(progresses):
        best_idx = int(torch.argmax(current_scores[job_index]).item())
        best_coefficients.append(current[job_index, best_idx].detach().clone())
        best_evaluations.append(current_evaluations[job_index][best_idx])
        best_score = float(current_scores[job_index, best_idx].item())
        best_scores.append(best_score)
        progress.write(
            "initial_candidates_evaluated",
            device=train_bundle.device,
            effective_lanes=len(baseline_dr_values) * settings.chains,
            max_batch_lanes=dr_batch_size * settings.chains,
            batch_effective_lanes=len(lane_user_ids),
            best_score=best_score,
            best_mean_relative_memorized_gain=best_evaluations[
                job_index
            ].mean_relative_memorized_gain,
            best_mean_relative_efficiency_gain=best_evaluations[
                job_index
            ].mean_relative_efficiency_gain,
        )

    for iteration in range(settings.iterations):
        temp = _temperature(settings, iteration)
        proposals = []
        for job_index, generator in enumerate(generators):
            proposals.append(
                _clamp_coefficients(
                    current[job_index]
                    + torch.randn(
                        current[job_index].shape,
                        device=train_bundle.device,
                        generator=generator,
                    )
                    * settings.proposal_scale,
                    settings,
                )
            )
        proposal = torch.stack(proposals, dim=0)
        proposal_evaluations = _evaluate_dr_conditioned_batch(
            config=config,
            settings=settings,
            bundle=train_bundle,
            jobs=jobs,
            baseline_dr_values=baseline_dr_values,
            dr_batch_size=dr_batch_size,
            baselines_by_job=baselines_by_job,
            coefficients=proposal,
            feature_version=feature_version,
            seed=config.seed,
        )
        proposal_scores = torch.tensor(
            [
                [evaluation.score for evaluation in evaluations]
                for evaluations in proposal_evaluations
            ],
            device=train_bundle.device,
            dtype=torch.float32,
        )
        delta = proposal_scores - current_scores
        accept_prob = torch.exp(delta / max(temp, 1e-9))
        accept_rows = []
        for job_index, generator in enumerate(generators):
            random_values = torch.rand(
                (settings.chains,), device=train_bundle.device, generator=generator
            )
            accept_rows.append(
                (delta[job_index] >= 0) | (random_values < accept_prob[job_index])
            )
        accept = torch.stack(accept_rows, dim=0)
        for job_index, progress in enumerate(progresses):
            if bool(torch.any(accept[job_index]).item()):
                current[job_index, accept[job_index]] = proposal[
                    job_index, accept[job_index]
                ]
                current_scores[job_index, accept[job_index]] = proposal_scores[
                    job_index, accept[job_index]
                ]
                for idx in (
                    torch.nonzero(accept[job_index], as_tuple=False).flatten().tolist()
                ):
                    current_evaluations[job_index][int(idx)] = proposal_evaluations[
                        job_index
                    ][int(idx)]
            best_idx = int(torch.argmax(current_scores[job_index]).item())
            iteration_best_score = float(current_scores[job_index, best_idx].item())
            if iteration_best_score > best_scores[job_index]:
                best_scores[job_index] = iteration_best_score
                best_coefficients[job_index] = (
                    current[job_index, best_idx].detach().clone()
                )
                best_evaluations[job_index] = current_evaluations[job_index][best_idx]
            history_entry = {
                "iteration": float(iteration),
                "temperature": float(temp),
                "best_score": best_scores[job_index],
                "best_mean_relative_memorized_gain": best_evaluations[
                    job_index
                ].mean_relative_memorized_gain,
                "best_mean_relative_efficiency_gain": best_evaluations[
                    job_index
                ].mean_relative_efficiency_gain,
            }
            histories[job_index].append(history_entry)
            progress.write(
                "annealing_iteration",
                device=train_bundle.device,
                accepted_count=int(accept[job_index].sum().item()),
                effective_lanes=len(baseline_dr_values) * settings.chains,
                max_batch_lanes=dr_batch_size * settings.chains,
                batch_effective_lanes=len(lane_user_ids),
                **history_entry,
            )

    outcomes = []
    for job_index, job in enumerate(jobs):
        best = best_evaluations[job_index]
        passed = (
            best.mean_relative_memorized_gain > 0.0
            and best.mean_relative_efficiency_gain > 0.0
        )
        result = DRConditionedTrainingResult(
            baseline_desired_retention_values=baseline_dr_values,
            baselines=baselines_by_job[job_index],
            best_coefficients=best_coefficients[job_index].detach().cpu(),
            best=best,
            history=histories[job_index],
            passed=passed,
        )
        policy_path, metrics_path, metadata_path = _write_artifact(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            feature_version=feature_version,
            result=result,
        )
        progresses[job_index].write(
            "artifacts_written",
            device=train_bundle.device,
            passed=passed,
            policy_path=str(policy_path),
            metrics_path=str(metrics_path),
            metadata_path=str(metadata_path),
        )
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=passed,
                artifact_paths=(metadata_path,),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes


def _run_fsrs6_adr_delta_cmaes_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    import cma

    from experiments.rl_scheduler.train_cmaes_fsrs6_adr_delta import (
        CMAESSettings,
        _augment_artifact,
        _optimizer_seed,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _read_training_sa,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_delta import (
        DRConditionedTrainingResult,
        _baseline_dr_values,
        _dr_batch_size,
        _policy_feature_version,
        _write_artifact,
    )
    from simulator.fsrs6_adr_delta_policy import feature_count

    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = dict(_read_training_sa(config_path))
    feature_version = _policy_feature_version(raw_training_sa)
    coefficient_count = feature_count(feature_version)
    optimizer_settings = CMAESSettings.from_mapping(
        config.training_optimizer,
        coefficient_count=coefficient_count,
        coefficient_min=settings.coefficient_min,
        coefficient_max=settings.coefficient_max,
    )
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    optimizer_seeds = [
        _optimizer_seed(
            config=config,
            settings=optimizer_settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        for job in jobs
    ]
    for progress, seed in zip(progresses, optimizer_seeds, strict=True):
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            optimizer=optimizer_settings.to_dict(),
            optimizer_seed=seed,
            feature_version=feature_version,
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            baseline_desired_retention_values=list(baseline_dr_values),
            dr_batch_size=dr_batch_size,
        )
        progress.write(
            "device_resolved", device=ctx.device, torch_device=str(ctx.device)
        )

    baseline_bundle, baselines_by_job = _evaluate_baseline_grid(
        config=config,
        settings=settings,
        jobs=jobs,
        baseline_dr_values=baseline_dr_values,
        ctx=ctx,
    )
    for baselines, progress in zip(baselines_by_job, progresses, strict=True):
        progress.write(
            "baselines_evaluated",
            device=baseline_bundle.device,
            effective_lanes=len(baseline_dr_values),
            batch_effective_lanes=len(jobs) * len(baseline_dr_values),
            metrics=[
                {"baseline_desired_retention": dr, **asdict(metric)}
                for dr, metric in zip(baseline_dr_values, baselines, strict=True)
            ],
        )

    lane_user_ids = [
        job.user_id
        for job in jobs
        for _candidate in range(optimizer_settings.population_size)
        for _dr in range(dr_batch_size)
    ]
    train_bundle = _build_bundle_for_lanes(
        config=config,
        settings=settings,
        lane_user_ids=lane_user_ids,
        ctx=ctx,
    )
    for progress in progresses:
        progress.write(
            "train_bundle_built",
            device=train_bundle.device,
            effective_lanes=dr_batch_size * optimizer_settings.population_size,
            grid_lanes=len(baseline_dr_values) * optimizer_settings.population_size,
            batch_effective_lanes=len(lane_user_ids),
        )

    strategies = []
    for seed in optimizer_seeds:
        strategies.append(
            cma.CMAEvolutionStrategy(
                list(optimizer_settings.initial_mean),
                optimizer_settings.sigma0,
                {
                    "bounds": [
                        list(optimizer_settings.bounds[0]),
                        list(optimizer_settings.bounds[1]),
                    ],
                    "popsize": optimizer_settings.population_size,
                    "seed": seed,
                    "verb_disp": 0,
                    "verb_log": 0,
                    "verbose": -9,
                },
            )
        )

    best_coefficients: list[torch.Tensor | None] = [None for _job in jobs]
    best_evaluations: list[Any | None] = [None for _job in jobs]
    best_scores = [float("-inf") for _job in jobs]
    histories: list[list[dict[str, float]]] = [[] for _job in jobs]

    for generation in range(optimizer_settings.generations):
        solutions_by_job = []
        for strategy in strategies:
            solutions = [list(map(float, item)) for item in strategy.ask()]
            if len(solutions) != optimizer_settings.population_size:
                raise RuntimeError(
                    "CMA-ES returned an unexpected population size: "
                    f"{len(solutions)} != {optimizer_settings.population_size}."
                )
            if generation == 0:
                solutions[0] = list(optimizer_settings.initial_mean)
            solutions_by_job.append(solutions)
        coefficients = torch.tensor(
            solutions_by_job,
            device=train_bundle.device,
            dtype=torch.float32,
        )
        evaluations_by_job = _evaluate_dr_conditioned_batch(
            config=config,
            settings=settings,
            bundle=train_bundle,
            jobs=jobs,
            baseline_dr_values=baseline_dr_values,
            dr_batch_size=dr_batch_size,
            baselines_by_job=baselines_by_job,
            coefficients=coefficients,
            feature_version=feature_version,
            seed=config.seed,
        )
        for job_index, strategy in enumerate(strategies):
            scores = [evaluation.score for evaluation in evaluations_by_job[job_index]]
            strategy.tell(solutions_by_job[job_index], [-score for score in scores])
            generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
            generation_best = evaluations_by_job[job_index][generation_best_idx]
            generation_best_score = float(scores[generation_best_idx])
            if generation_best_score > best_scores[job_index]:
                best_scores[job_index] = generation_best_score
                best_coefficients[job_index] = (
                    coefficients[job_index, generation_best_idx].detach().clone()
                )
                best_evaluations[job_index] = generation_best
            history_entry = {
                "generation": float(generation),
                "sigma": float(strategy.sigma),
                "best_score": float(best_scores[job_index]),
                "generation_best_score": generation_best_score,
                "mean_score": float(sum(scores) / max(len(scores), 1)),
                "generation_best_mean_relative_memorized_gain": (
                    generation_best.mean_relative_memorized_gain
                ),
                "generation_best_mean_relative_efficiency_gain": (
                    generation_best.mean_relative_efficiency_gain
                ),
            }
            histories[job_index].append(history_entry)
            progresses[job_index].write(
                "cmaes_generation",
                device=train_bundle.device,
                effective_lanes=len(baseline_dr_values)
                * optimizer_settings.population_size,
                max_batch_lanes=dr_batch_size * optimizer_settings.population_size,
                batch_effective_lanes=len(lane_user_ids),
                **history_entry,
            )

    outcomes = []
    for job_index, job in enumerate(jobs):
        best_coefficients_for_job = best_coefficients[job_index]
        best = best_evaluations[job_index]
        if best_coefficients_for_job is None or best is None:
            raise RuntimeError("CMA-ES did not evaluate any candidates.")
        passed = (
            best.mean_relative_memorized_gain > 0.0
            and best.mean_relative_efficiency_gain > 0.0
        )
        result = DRConditionedTrainingResult(
            baseline_desired_retention_values=baseline_dr_values,
            baselines=baselines_by_job[job_index],
            best_coefficients=best_coefficients_for_job.detach().cpu(),
            best=best,
            history=histories[job_index],
            passed=passed,
        )
        progresses[job_index].write(
            "cmaes_completed",
            device=train_bundle.device,
            best_score=best.score,
            mean_relative_memorized_gain=best.mean_relative_memorized_gain,
            mean_relative_efficiency_gain=best.mean_relative_efficiency_gain,
            generations=len(histories[job_index]),
        )
        policy_path, metrics_path, metadata_path = _write_artifact(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            feature_version=feature_version,
            result=result,
        )
        _augment_artifact(
            metrics_path=metrics_path,
            metadata_path=metadata_path,
            optimizer_settings=optimizer_settings,
            optimizer_seed=optimizer_seeds[job_index],
        )
        progresses[job_index].write(
            "artifacts_written",
            device=train_bundle.device,
            passed=passed,
            policy_path=str(policy_path),
            metrics_path=str(metrics_path),
            metadata_path=str(metadata_path),
        )
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=passed,
                artifact_paths=(metadata_path,),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes


def _run_fsrs6_adr_direct_dr_grid_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_fsrs6_adr_direct import (
        SASettings,
        _clamp_coefficients,
        _read_training_sa,
        _temperature,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_direct_dr_grid import (
        DRTrainingResult,
        _baseline_dr_values,
        _best_by_dr,
        _candidate_scores,
        _dr_batch_size,
        _initial_coefficients,
        _iter_chunks,
        _progress_best_by_dr,
        _write_grid_artifacts,
    )
    from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
    from simulator.schedulers.fsrs6_adr_direct import FSRS6ADRDirectBatchSchedulerOps
    from experiments.rl_scheduler.train_fsrs6_adr_direct import _metrics_from_stats

    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = dict(_read_training_sa(config_path))
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    for progress in progresses:
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            baseline_desired_retention_values=list(baseline_dr_values),
            dr_batch_size=dr_batch_size,
        )
        progress.write(
            "device_resolved", device=ctx.device, torch_device=str(ctx.device)
        )

    baseline_bundle, baselines_by_job = _evaluate_baseline_grid(
        config=config,
        settings=settings,
        jobs=jobs,
        baseline_dr_values=baseline_dr_values,
        ctx=ctx,
    )
    for baselines, progress in zip(baselines_by_job, progresses, strict=True):
        progress.write(
            "baselines_evaluated",
            device=baseline_bundle.device,
            effective_lanes=len(baseline_dr_values),
            batch_effective_lanes=len(jobs) * len(baseline_dr_values),
            baseline_desired_retention_values=list(baseline_dr_values),
            metrics=[
                {"baseline_desired_retention": dr, **asdict(metric)}
                for dr, metric in zip(baseline_dr_values, baselines, strict=True)
            ],
        )

    results_by_job: list[list[Any]] = [[] for _job in jobs]
    baselines_by_job_dr = [
        dict(zip(baseline_dr_values, baselines, strict=True))
        for baselines in baselines_by_job
    ]
    for chunk_start, chunk_dr_values in _iter_chunks(baseline_dr_values, dr_batch_size):
        lane_user_ids = [
            job.user_id
            for job in jobs
            for _dr in chunk_dr_values
            for _chain in range(settings.chains)
        ]
        train_bundle = _build_bundle_for_lanes(
            config=config,
            settings=settings,
            lane_user_ids=lane_user_ids,
            ctx=ctx,
        )
        for progress in progresses:
            progress.write(
                "train_bundle_built",
                device=train_bundle.device,
                chunk_start=chunk_start,
                chunk_size=len(chunk_dr_values),
                baseline_desired_retention_values=list(chunk_dr_values),
                effective_lanes=len(chunk_dr_values) * settings.chains,
                batch_effective_lanes=len(lane_user_ids),
            )

        generators = []
        currents = []
        for _job in jobs:
            generator = torch.Generator(device=train_bundle.device)
            generator.manual_seed(config.seed + chunk_start)
            generators.append(generator)
            currents.append(
                _initial_coefficients(
                    baseline_dr_values=chunk_dr_values,
                    settings=settings,
                    device=train_bundle.device,
                    generator=generator,
                ).reshape(len(chunk_dr_values), settings.chains, 6)
            )
        current = torch.stack(currents, dim=0)

        def evaluate(coefficients_by_job: torch.Tensor) -> list[list[Any]]:
            flat = coefficients_by_job.reshape(
                len(jobs) * len(chunk_dr_values) * settings.chains,
                6,
            )
            template = FSRS6ADRDirectPolicy.baseline(
                desired_retention=settings.baseline_desired_retention,
                retention_min=settings.retention_min,
                retention_max=settings.retention_max,
            )
            sched_ops = FSRS6ADRDirectBatchSchedulerOps(
                weights=train_bundle.scheduler_weights,
                policy=template,
                coefficients=flat,
                bounds=Bounds(),
                priority_mode=config.simulation.scheduler_priority,
                device=train_bundle.device,
                dtype=torch.float32,
            )
            stats = simulate_multiuser(
                days=config.simulation.days,
                deck_size=config.simulation.deck,
                env_ops=train_bundle.env_ops,
                sched_ops=sched_ops,
                behavior=train_bundle.behavior,
                cost_model=train_bundle.cost_model,
                seed=config.seed + chunk_start,
                device=train_bundle.device,
                dtype=torch.float32,
                fuzz=config.simulation.fuzz,
                priority_mode=config.simulation.priority,
                progress=False,
                short_term_source=train_bundle.short_term_source,
                learning_steps=train_bundle.learning_steps,
                relearning_steps=train_bundle.relearning_steps,
                short_term_threshold=settings.short_term_threshold,
                short_term_loops_limit=settings.short_term_loops_limit,
            )
            metrics = [_metrics_from_stats(item) for item in stats]
            lanes_per_job = len(chunk_dr_values) * settings.chains
            return [
                metrics[index * lanes_per_job : (index + 1) * lanes_per_job]
                for index in range(len(jobs))
            ]

        current_metrics = evaluate(current)
        current_scores = torch.stack(
            [
                _candidate_scores(
                    metrics=current_metrics[job_index],
                    baselines=[
                        baselines_by_job_dr[job_index][dr] for dr in chunk_dr_values
                    ],
                    lane_dr_indices=[
                        dr_index
                        for dr_index in range(len(chunk_dr_values))
                        for _chain in range(settings.chains)
                    ],
                    lambda_value=jobs[job_index].lambda_value,
                    device=train_bundle.device,
                )
                for job_index in range(len(jobs))
            ],
            dim=0,
        )
        best_coefficients = []
        best_metrics = []
        best_scores = []
        histories: list[list[list[dict[str, float]]]] = [
            [[] for _dr in chunk_dr_values] for _job in jobs
        ]
        for job_index, progress in enumerate(progresses):
            coeffs, metrics, scores = _best_by_dr(
                coefficients=current[job_index].reshape(
                    len(chunk_dr_values) * settings.chains, 6
                ),
                metrics=current_metrics[job_index],
                scores=current_scores[job_index],
                chains=settings.chains,
                dr_count=len(chunk_dr_values),
            )
            best_coefficients.append(coeffs)
            best_metrics.append(metrics)
            best_scores.append(scores)
            progress.write(
                "initial_candidates_evaluated",
                device=train_bundle.device,
                baseline_desired_retention_values=list(chunk_dr_values),
                effective_lanes=len(chunk_dr_values) * settings.chains,
                batch_effective_lanes=len(lane_user_ids),
                best_by_dr=_progress_best_by_dr(
                    baseline_dr_values=chunk_dr_values,
                    baselines=[
                        baselines_by_job_dr[job_index][dr] for dr in chunk_dr_values
                    ],
                    best_metrics=metrics,
                    best_scores=scores,
                ),
            )

        for iteration in range(settings.iterations):
            temp = _temperature(settings, iteration)
            proposals = []
            for job_index, generator in enumerate(generators):
                proposals.append(
                    _clamp_coefficients(
                        current[job_index]
                        + torch.randn(
                            current[job_index].shape,
                            device=train_bundle.device,
                            generator=generator,
                        )
                        * settings.proposal_scale,
                        settings,
                    )
                )
            proposal = torch.stack(proposals, dim=0)
            proposal_metrics = evaluate(proposal)
            proposal_scores = torch.stack(
                [
                    _candidate_scores(
                        metrics=proposal_metrics[job_index],
                        baselines=[
                            baselines_by_job_dr[job_index][dr] for dr in chunk_dr_values
                        ],
                        lane_dr_indices=[
                            dr_index
                            for dr_index in range(len(chunk_dr_values))
                            for _chain in range(settings.chains)
                        ],
                        lambda_value=jobs[job_index].lambda_value,
                        device=train_bundle.device,
                    )
                    for job_index in range(len(jobs))
                ],
                dim=0,
            )
            delta = proposal_scores - current_scores
            accept_prob = torch.exp(delta / max(temp, 1e-9))
            accept_rows = []
            for job_index, generator in enumerate(generators):
                random_values = torch.rand(
                    (len(chunk_dr_values) * settings.chains,),
                    device=train_bundle.device,
                    generator=generator,
                )
                accept_rows.append(
                    (delta[job_index] >= 0) | (random_values < accept_prob[job_index])
                )
            accept = torch.stack(accept_rows, dim=0)
            for job_index, progress in enumerate(progresses):
                flat_current = current[job_index].reshape(
                    len(chunk_dr_values) * settings.chains, 6
                )
                flat_proposal = proposal[job_index].reshape(
                    len(chunk_dr_values) * settings.chains, 6
                )
                if bool(torch.any(accept[job_index]).item()):
                    flat_current[accept[job_index]] = flat_proposal[accept[job_index]]
                    current_scores[job_index, accept[job_index]] = proposal_scores[
                        job_index, accept[job_index]
                    ]
                    for idx in (
                        torch.nonzero(accept[job_index], as_tuple=False)
                        .flatten()
                        .tolist()
                    ):
                        current_metrics[job_index][int(idx)] = proposal_metrics[
                            job_index
                        ][int(idx)]
                current[job_index] = flat_current.reshape(
                    len(chunk_dr_values), settings.chains, 6
                )
                for dr_index, baseline_dr in enumerate(chunk_dr_values):
                    start = dr_index * settings.chains
                    end = start + settings.chains
                    local_idx = int(
                        torch.argmax(current_scores[job_index, start:end]).item()
                    )
                    flat_idx = start + local_idx
                    score = float(current_scores[job_index, flat_idx].item())
                    if score > best_scores[job_index][dr_index]:
                        best_scores[job_index][dr_index] = score
                        best_coefficients[job_index][dr_index] = (
                            flat_current[flat_idx].detach().clone()
                        )
                        best_metrics[job_index][dr_index] = current_metrics[job_index][
                            flat_idx
                        ]
                    baseline = baselines_by_job_dr[job_index][baseline_dr]
                    rel_mem = (
                        best_metrics[job_index][dr_index].memorized_average
                        / baseline.memorized_average
                        - 1.0
                        if baseline.memorized_average
                        else 0.0
                    )
                    rel_eff = (
                        best_metrics[job_index][dr_index].memorized_per_minute
                        / baseline.memorized_per_minute
                        - 1.0
                        if baseline.memorized_per_minute
                        else 0.0
                    )
                    histories[job_index][dr_index].append(
                        {
                            "iteration": float(iteration),
                            "temperature": float(temp),
                            "best_score": best_scores[job_index][dr_index],
                            "best_relative_memorized_gain": rel_mem,
                            "best_relative_efficiency_gain": rel_eff,
                            "baseline_desired_retention": float(baseline_dr),
                        }
                    )
                progress.write(
                    "annealing_iteration",
                    device=train_bundle.device,
                    iteration=iteration,
                    temperature=float(temp),
                    accepted_count=int(accept[job_index].sum().item()),
                    baseline_desired_retention_values=list(chunk_dr_values),
                    effective_lanes=len(chunk_dr_values) * settings.chains,
                    batch_effective_lanes=len(lane_user_ids),
                    best_by_dr=_progress_best_by_dr(
                        baseline_dr_values=chunk_dr_values,
                        baselines=[
                            baselines_by_job_dr[job_index][dr] for dr in chunk_dr_values
                        ],
                        best_metrics=best_metrics[job_index],
                        best_scores=best_scores[job_index],
                    ),
                )

        for job_index in range(len(jobs)):
            for dr_index, baseline_dr in enumerate(chunk_dr_values):
                baseline = baselines_by_job_dr[job_index][baseline_dr]
                best = best_metrics[job_index][dr_index]
                rel_mem = (
                    best.memorized_average / baseline.memorized_average - 1.0
                    if baseline.memorized_average
                    else 0.0
                )
                rel_eff = (
                    best.memorized_per_minute / baseline.memorized_per_minute - 1.0
                    if baseline.memorized_per_minute
                    else 0.0
                )
                results_by_job[job_index].append(
                    DRTrainingResult(
                        baseline_desired_retention=baseline_dr,
                        baseline=baseline,
                        best_coefficients=best_coefficients[job_index][dr_index]
                        .detach()
                        .cpu(),
                        best=best,
                        best_score=best_scores[job_index][dr_index],
                        history=histories[job_index][dr_index],
                        passed=rel_mem > 0.0 and rel_eff > 0.0,
                    )
                )

    outcomes = []
    for job_index, job in enumerate(jobs):
        artifact_paths = _write_grid_artifacts(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            results=results_by_job[job_index],
        )
        passed_count = sum(1 for result in results_by_job[job_index] if result.passed)
        progresses[job_index].write(
            "artifacts_written",
            device=ctx.device,
            policies=len(results_by_job[job_index]),
            passed_policies=passed_count,
            artifact_paths=[str(path) for path in artifact_paths],
        )
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=passed_count > 0,
                artifact_paths=tuple(artifact_paths),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes
