from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any

import cma
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    CMAESSettings,
    CandidateMetrics,
    PolicySearchSettings,
    TrainingProgress,
    _artifact_id,
    _build_bundle,
    _evaluate_fsrs6_baseline,
    _evaluate_adr_candidates,
    _git_commit,
    _passes_overfit_gate,
    _policy_feature_version,
    _read_training_policy_search,
    _relative_gain,
    _relative_gain_gate_metrics,
    _score,
    _optimizer_seed,
    _write_json,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy, feature_count
from simulator.short_term_config import resolve_short_term_config


@dataclass(frozen=True, slots=True)
class CMAESFSRS6TrainingResult:
    baseline: CandidateMetrics
    best: CandidateMetrics
    best_coefficients: torch.Tensor
    best_score: float
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FSRS6 ADR scheduler policy with CMA-ES.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument(
        "--baseline-desired-retention",
        type=float,
        default=None,
        help="Override training.policy_search.baseline_desired_retention for DR-grid runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL.",
    )
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument("--training-command-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress(output_dir / "training_progress.jsonl")
    progress.write(
        "started",
        config_path=str(args.config),
        user_id=args.user_id,
        lambda_value=args.lambda_value,
    )

    config = ExperimentConfig.from_toml(args.config)
    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    if args.baseline_desired_retention is not None:
        settings = replace(
            settings,
            baseline_desired_retention=args.baseline_desired_retention,
        )
        settings.__post_init__()
    raw_training_policy_search = _read_training_policy_search(args.config)
    feature_version = _policy_feature_version(raw_training_policy_search)
    optimizer_settings = optimizer_settings_from_mapping(
        config.training_optimizer,
        settings=settings,
        feature_version=feature_version,
    )
    optimizer_seed = _optimizer_seed(
        config=config,
        settings=optimizer_settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
    )
    progress.write(
        "config_loaded",
        settings=asdict(settings),
        optimizer=optimizer_settings.to_dict(),
        optimizer_seed=optimizer_seed,
        feature_version=feature_version,
        simulation=config.simulation.to_dict(),
        seed=config.seed,
    )

    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_policy_search.get("learning_steps"),
        relearning_steps=raw_training_policy_search.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    device = torch.device(settings.torch_device)
    progress.write("device_resolved", device=device, torch_device=str(device))

    baseline_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=1,
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baseline = _evaluate_fsrs6_baseline(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        seed=config.seed,
    )
    progress.write(
        "baseline_evaluated",
        device=baseline_bundle.device,
        effective_lanes=1,
        metrics=asdict(baseline),
    )
    del baseline_bundle
    _clear_cuda_cache(device)

    train_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=optimizer_settings.population_size,
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    progress.write(
        "train_bundle_built",
        device=train_bundle.device,
        effective_lanes=optimizer_settings.population_size,
    )
    result = _run_cmaes(
        config=config,
        settings=settings,
        optimizer_settings=optimizer_settings,
        optimizer_seed=optimizer_seed,
        bundle=train_bundle,
        baseline=baseline,
        lambda_value=args.lambda_value,
        feature_version=feature_version,
        progress=progress,
    )
    progress.write(
        "cmaes_completed",
        device=train_bundle.device,
        best_score=result.best_score,
        best=asdict(result.best),
        relative_memorized_gain=_relative_gain(
            result.best.memorized_average,
            result.baseline.memorized_average,
        ),
        relative_efficiency_gain=_relative_gain(
            result.best.memorized_per_minute,
            result.baseline.memorized_per_minute,
        ),
        generations=len(result.history),
    )

    policy_path, metrics_path, metadata_path = write_artifact(
        output_dir=output_dir,
        config=config,
        config_path=args.config,
        settings=settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
        training_command_path=args.training_command_path,
        feature_version=feature_version,
        result=result,
        optimizer_settings=optimizer_settings,
        optimizer_seed=optimizer_seed,
    )
    progress.write(
        "artifacts_written",
        device=train_bundle.device,
        passed=result.passed,
        policy_path=str(policy_path),
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
    )
    return 0 if result.passed else 1


def optimizer_settings_from_mapping(
    raw: Mapping[str, Any],
    *,
    settings: PolicySearchSettings,
    feature_version: str,
) -> CMAESSettings:
    optimizer_raw = dict(raw)
    if "initial_mean" not in optimizer_raw:
        optimizer_raw["initial_mean"] = [
            min(settings.coefficient_max, max(settings.coefficient_min, value))
            for value in baseline_coefficients(
                settings=settings, feature_version=feature_version
            )
        ]
    return CMAESSettings.from_mapping(
        optimizer_raw,
        coefficient_count=feature_count(feature_version),
        coefficient_min=settings.coefficient_min,
        coefficient_max=settings.coefficient_max,
    )


def baseline_coefficients(
    *,
    settings: PolicySearchSettings,
    feature_version: str,
) -> tuple[float, ...]:
    return FSRS6ADRPolicy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    ).coefficients


def _run_cmaes(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
    bundle: Any,
    baseline: CandidateMetrics,
    lambda_value: float,
    feature_version: str,
    progress: TrainingProgress,
) -> CMAESFSRS6TrainingResult:
    opts: dict[str, Any] = {
        "bounds": [
            list(optimizer_settings.bounds[0]),
            list(optimizer_settings.bounds[1]),
        ],
        "popsize": optimizer_settings.population_size,
        "seed": optimizer_seed,
        "verb_disp": 0,
        "verb_log": 0,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(
        list(optimizer_settings.initial_mean),
        optimizer_settings.sigma0,
        opts,
    )
    best_coefficients: torch.Tensor | None = None
    best_metrics: CandidateMetrics | None = None
    best_score = float("-inf")
    history: list[dict[str, float]] = []

    for generation in range(optimizer_settings.generations):
        solutions = [list(map(float, item)) for item in es.ask()]
        if len(solutions) != optimizer_settings.population_size:
            raise RuntimeError(
                "CMA-ES returned an unexpected population size: "
                f"{len(solutions)} != {optimizer_settings.population_size}."
            )
        if generation == 0:
            solutions[0] = list(optimizer_settings.initial_mean)
        coefficients = torch.tensor(
            solutions,
            device=bundle.device,
            dtype=torch.float32,
        )
        metrics = _evaluate_adr_candidates(
            config=config,
            settings=settings,
            bundle=bundle,
            coefficients=coefficients,
            feature_version=feature_version,
            seed=config.seed,
        )
        scores = [_score(metric, baseline, lambda_value) for metric in metrics]
        es.tell(solutions, [-score for score in scores])

        generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
        generation_best = metrics[generation_best_idx]
        generation_best_score = float(scores[generation_best_idx])
        if generation_best_score > best_score:
            best_score = generation_best_score
            best_coefficients = coefficients[generation_best_idx].detach().clone()
            best_metrics = generation_best

        history_entry = {
            "generation": float(generation),
            "sigma": float(es.sigma),
            "best_score": float(best_score),
            "generation_best_score": generation_best_score,
            "mean_score": float(sum(scores) / max(len(scores), 1)),
            "generation_best_relative_memorized_gain": _relative_gain(
                generation_best.memorized_average,
                baseline.memorized_average,
            ),
            "generation_best_relative_efficiency_gain": _relative_gain(
                generation_best.memorized_per_minute,
                baseline.memorized_per_minute,
            ),
        }
        history.append(history_entry)
        progress.write(
            "cmaes_generation",
            device=bundle.device,
            effective_lanes=optimizer_settings.population_size,
            **history_entry,
        )

    if best_coefficients is None or best_metrics is None:
        raise RuntimeError("CMA-ES did not evaluate any candidates.")

    rel_mem = _relative_gain(best_metrics.memorized_average, baseline.memorized_average)
    rel_eff = _relative_gain(
        best_metrics.memorized_per_minute,
        baseline.memorized_per_minute,
    )
    return CMAESFSRS6TrainingResult(
        baseline=baseline,
        best=best_metrics,
        best_coefficients=best_coefficients.detach().cpu(),
        best_score=best_score,
        history=history,
        passed=_passes_overfit_gate(rel_mem, rel_eff),
    )


def write_artifact(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    user_id: int,
    lambda_value: float,
    training_command_path: Path | None,
    feature_version: str,
    result: CMAESFSRS6TrainingResult,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rel_mem = _relative_gain(
        result.best.memorized_average,
        result.baseline.memorized_average,
    )
    rel_eff = _relative_gain(
        result.best.memorized_per_minute,
        result.baseline.memorized_per_minute,
    )
    policy = FSRS6ADRPolicy(
        coefficients=tuple(float(v) for v in result.best_coefficients.tolist()),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        baseline_desired_retention=settings.baseline_desired_retention,
        feature_version=feature_version,
        title=(
            f"fsrs6_adr_cmaes_u{user_id}_dr_"
            f"{settings.baseline_desired_retention:.2f}_lambda_{lambda_value:g}"
        ),
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)

    optimizer = {
        **optimizer_settings.to_dict(),
        "seed_resolved": optimizer_seed,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "passed_overfit_gate": result.passed,
            "gate": _relative_gain_gate_metrics(rel_mem, rel_eff),
            "baseline": asdict(result.baseline),
            "best": asdict(result.best),
            "best_score": result.best_score,
            "feature_version": feature_version,
            "optimizer": optimizer,
            "settings": asdict(settings),
            "history": result.history,
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
            "scheduler_name": "fsrs6_adr",
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
            "optimizer": "cma_es",
            "optimizer_settings": optimizer,
            "capabilities": ["event", "batched"],
        },
    )
    return policy_path, metrics_path, metadata_path


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
