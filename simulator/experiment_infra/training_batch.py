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
from simulator.batched_engine.multiuser_engine import simulate_multiuser


SUPPORTED_TRAINERS = {
    "fsrs6_adr_portfolio",
    "fsrs6_adr_cmaes",
    "fsrs6_ap_cmaes",
    "fsrs6_ap_portfolio",
}


@dataclass(frozen=True, slots=True)
class InProcessTrainJob:
    user_id: int
    baseline_desired_retention: float
    baseline_desired_retention_token: str
    lambda_value: float | None
    lambda_token: str | None
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
    raw_training_policy_search: dict[str, Any]
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
    if "train_cmaes_fsrs6_adr.py" in script_names:
        return "fsrs6_adr_cmaes"
    if "train_cmaes_fsrs6_ap.py" in script_names:
        return "fsrs6_ap_cmaes"
    if "train_fsrs6_ap_portfolio.py" in script_names:
        return "fsrs6_ap_portfolio"
    if "train_fsrs6_adr_portfolio.py" in script_names:
        return "fsrs6_adr_portfolio"
    raise ValueError(
        "training.batch.trainer = 'auto' requires an in-tree RL trainer script "
        "in training.command_template."
    )


def estimate_lanes_per_job(*, trainer: str, config: ExperimentConfig) -> int:
    from experiments.rl_scheduler.train_cmaes_fsrs6_adr import (
        optimizer_settings_from_mapping,
    )
    from experiments.rl_scheduler.policy_search_common import (
        _baseline_dr_values,
        PolicySearchSettings,
        _read_training_policy_search,
    )
    from experiments.rl_scheduler.policy_search_common import (
        _policy_feature_version as _policy_search_feature_version,
    )
    from experiments.rl_scheduler.train_fsrs6_adr_portfolio import (
        PortfolioSettings,
    )
    from experiments.rl_scheduler.train_fsrs6_ap_portfolio import (
        APPortfolioSettings,
    )
    from experiments.rl_scheduler.train_cmaes_fsrs6_ap import (
        APSettings,
        optimizer_settings_from_mapping as ap_optimizer_settings_from_mapping,
    )

    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search: dict[str, Any]
    if config.config_path is not None and config.config_path.exists():
        raw_training_policy_search = dict(
            _read_training_policy_search(config.config_path)
        )
    else:
        raw_training_policy_search = dict(config.training_policy_search)
    if trainer == "fsrs6_adr_cmaes":
        feature_version = _policy_search_feature_version(raw_training_policy_search)
        optimizer = optimizer_settings_from_mapping(
            config.training_optimizer,
            settings=settings,
            feature_version=feature_version,
        )
        return max(1, optimizer.population_size)
    if trainer == "fsrs6_adr_portfolio":
        portfolio = PortfolioSettings.from_mapping(
            config.training_portfolio,
            settings=settings,
            default_seed_retention_values=_baseline_dr_values(
                raw_training_policy_search, settings
            ),
        )
        return max(
            len(portfolio.seed_retention_values or ()),
            portfolio.population_size,
            portfolio.offspring_size,
        )
    baseline_dr_values = _baseline_dr_values(raw_training_policy_search, settings)
    if trainer == "fsrs6_ap_cmaes":
        optimizer = ap_optimizer_settings_from_mapping(config.training_optimizer)
        ap_settings = APSettings.from_config(
            config,
            raw_training_policy_search=raw_training_policy_search,
            dr_count=len(baseline_dr_values),
        )
        return max(
            len(baseline_dr_values),
            ap_settings.dr_batch_size * optimizer.population_size,
        )
    if trainer == "fsrs6_ap_portfolio":
        portfolio = APPortfolioSettings.from_mapping(
            config.training_portfolio,
            settings=settings,
            default_seed_retention_values=baseline_dr_values,
        )
        return max(
            len(portfolio.seed_retention_values or ()),
            portfolio.population_size,
            portfolio.offspring_size,
        )
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
    if trainer == "fsrs6_adr_cmaes":
        return _run_fsrs6_adr_cmaes_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_adr_portfolio":
        return _run_fsrs6_adr_portfolio_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_ap_cmaes":
        return _run_fsrs6_ap_cmaes_jobs(
            jobs=jobs, config=config, config_path=config_path, repo_root=repo_root
        )
    if trainer == "fsrs6_ap_portfolio":
        return _run_fsrs6_ap_portfolio_jobs(
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
    from experiments.rl_scheduler.policy_search_common import (
        _read_training_policy_search,
    )

    raw_training_policy_search = dict(_read_training_policy_search(config_path))
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_policy_search.get("learning_steps"),
        relearning_steps=raw_training_policy_search.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    return _CommonContext(
        raw_training_policy_search=raw_training_policy_search,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        device=torch.device(settings.torch_device),
        benchmark_root=resolve_benchmark_root(repo_root, None).resolve(),
        overrides=parse_result_overrides(None),
    )


def _require_lambda_value(job: InProcessTrainJob) -> float:
    if job.lambda_value is None:
        raise ValueError("lambda_value is required for CMA-ES trainers.")
    return float(job.lambda_value)


def _progress_for_jobs(
    *, jobs: list[InProcessTrainJob], config_path: Path
) -> list[Any]:
    from experiments.rl_scheduler.policy_search_common import (
        TrainingProgress,
        _relative_path_string,
    )

    progresses = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(job.output_dir / "training_progress.jsonl")
        progress.write(
            "started",
            config_path=_relative_path_string(config_path, base=job.output_dir),
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
    from experiments.rl_scheduler.policy_search_common import _build_bundle

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
    from experiments.rl_scheduler.policy_search_common import _metrics_from_stats

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


def _run_fsrs6_adr_portfolio_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_fsrs6_adr_portfolio import (
        PortfolioTrainJob,
        run_portfolio_train_jobs,
    )

    outcomes = run_portfolio_train_jobs(
        jobs=[
            PortfolioTrainJob(
                user_id=job.user_id,
                output_dir=job.output_dir,
                command_record_path=job.command_record_path,
            )
            for job in jobs
        ],
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        execution_mode="in_process_batch",
    )
    outcome_by_key = {
        (outcome.job.user_id, outcome.job.output_dir): outcome for outcome in outcomes
    }
    results: list[InProcessTrainOutcome] = []
    for job in jobs:
        outcome = outcome_by_key.get((job.user_id, job.output_dir))
        if outcome is None:
            results.append(
                InProcessTrainOutcome(
                    job=job,
                    passed=False,
                    artifact_paths=(),
                    progress_path=None,
                    error="ADR portfolio trainer did not return an outcome for this job.",
                )
            )
            continue
        results.append(
            InProcessTrainOutcome(
                job=job,
                passed=outcome.passed,
                artifact_paths=outcome.artifact_paths,
                progress_path=outcome.progress_path,
                error=outcome.error,
            )
        )
    return results


def _run_fsrs6_adr_cmaes_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    import cma

    from experiments.rl_scheduler.train_cmaes_fsrs6_adr import (
        CMAESFSRS6TrainingResult,
        optimizer_settings_from_mapping,
        write_artifact,
    )
    from experiments.rl_scheduler.policy_search_common import (
        PolicySearchSettings,
        _metrics_from_stats,
        _optimizer_seed,
        _passes_overfit_gate,
        _policy_feature_version,
        _relative_gain,
        _relative_path_string,
        _score,
    )
    from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
    from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps

    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    ctx = _common_context(
        config=config, config_path=config_path, repo_root=repo_root, settings=settings
    )
    feature_version = _policy_feature_version(ctx.raw_training_policy_search)
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
            lambda_value=_require_lambda_value(job),
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
        template = FSRS6ADRPolicy.baseline(
            desired_retention=settings.baseline_desired_retention,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            feature_version=feature_version,
        )
        sched_ops = FSRS6ADRBatchSchedulerOps(
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
                _score(
                    metric,
                    baselines[job_index],
                    _require_lambda_value(jobs[job_index]),
                )
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
            passed=_passes_overfit_gate(rel_mem, rel_eff),
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
            lambda_value=_require_lambda_value(job),
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
            policy_path=_relative_path_string(policy_path, base=job.output_dir),
            metrics_path=_relative_path_string(metrics_path, base=job.output_dir),
            metadata_path=_relative_path_string(metadata_path, base=job.output_dir),
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


def _run_fsrs6_ap_cmaes_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_cmaes_fsrs6_ap import (
        APTrainJob,
        run_training_batch_jobs,
    )

    results = run_training_batch_jobs(
        jobs=[
            APTrainJob(
                user_id=job.user_id,
                lambda_value=_require_lambda_value(job),
                output_dir=job.output_dir,
                command_record_path=job.command_record_path,
            )
            for job in jobs
        ],
        config=config,
        config_path=config_path,
        repo_root=repo_root,
    )
    result_by_key = {
        (result.job.user_id, result.job.lambda_value, result.job.output_dir): result
        for result in results
    }
    outcomes: list[InProcessTrainOutcome] = []
    for job in jobs:
        result = result_by_key.get(
            (job.user_id, _require_lambda_value(job), job.output_dir)
        )
        if result is None:
            outcomes.append(
                InProcessTrainOutcome(
                    job=job,
                    passed=False,
                    artifact_paths=(),
                    progress_path=None,
                    error="AP trainer did not return an outcome for this job.",
                )
            )
            continue
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=result.passed,
                artifact_paths=result.artifact_paths,
                progress_path=result.progress_path,
            )
        )
    return outcomes


def _run_fsrs6_ap_portfolio_jobs(
    *,
    jobs: list[InProcessTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[InProcessTrainOutcome]:
    from experiments.rl_scheduler.train_fsrs6_ap_portfolio import (
        APPortfolioTrainJob,
        run_portfolio_train_jobs,
    )

    results = run_portfolio_train_jobs(
        jobs=[
            APPortfolioTrainJob(
                user_id=job.user_id,
                output_dir=job.output_dir,
                command_record_path=job.command_record_path,
            )
            for job in jobs
        ],
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        execution_mode="in_process_batch",
    )
    result_by_key = {
        (result.job.user_id, result.job.output_dir): result for result in results
    }
    outcomes: list[InProcessTrainOutcome] = []
    for job in jobs:
        result = result_by_key.get((job.user_id, job.output_dir))
        if result is None:
            outcomes.append(
                InProcessTrainOutcome(
                    job=job,
                    passed=False,
                    artifact_paths=(),
                    progress_path=None,
                    error="AP portfolio trainer did not return an outcome for this job.",
                )
            )
            continue
        outcomes.append(
            InProcessTrainOutcome(
                job=job,
                passed=result.passed,
                artifact_paths=result.artifact_paths,
                progress_path=result.progress_path,
                error=result.error,
            )
        )
    return outcomes
