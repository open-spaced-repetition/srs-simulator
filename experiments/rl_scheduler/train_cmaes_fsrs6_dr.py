from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cma
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_sa_fsrs6 import (
    SASettings,
    TrainingProgress,
    _build_bundle,
    _read_training_sa,
)
from experiments.rl_scheduler.train_sa_fsrs6_dr import (
    DRConditionedTrainingResult,
    _baseline_dr_values,
    _clear_cuda_cache,
    _dr_batch_size,
    _evaluate_fsrs6_baselines,
    _evaluate_sa_dr_chains,
    _policy_feature_version,
    _write_artifact,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.sa_fsrs6_dr_policy import feature_count
from simulator.short_term_config import resolve_short_term_config


@dataclass(frozen=True, slots=True)
class CMAESSettings:
    name: str = "cma_es"
    population_size: int = 32
    generations: int = 10
    sigma0: float = 0.8
    initial_mean: tuple[float, ...] = ()
    bounds: tuple[tuple[float, ...], tuple[float, ...]] = ((), ())
    seed: int | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        coefficient_count: int,
        coefficient_min: float,
        coefficient_max: float,
    ) -> CMAESSettings:
        defaults = cls()
        name = _str(raw.get("name", defaults.name), "training.optimizer.name")
        if name != "cma_es":
            raise ValueError("training.optimizer.name must be 'cma_es'.")
        return cls(
            name=name,
            population_size=_int(
                raw.get("population_size", defaults.population_size),
                "training.optimizer.population_size",
                2,
            ),
            generations=_int(
                raw.get("generations", defaults.generations),
                "training.optimizer.generations",
                1,
            ),
            sigma0=_float(
                raw.get("sigma0", defaults.sigma0),
                "training.optimizer.sigma0",
                0.0,
            ),
            initial_mean=_float_tuple(
                raw.get("initial_mean", [0.0] * coefficient_count),
                "training.optimizer.initial_mean",
                coefficient_count,
            ),
            bounds=_bounds(
                raw.get(
                    "bounds",
                    [
                        [coefficient_min] * coefficient_count,
                        [coefficient_max] * coefficient_count,
                    ],
                ),
                coefficient_count,
            ),
            seed=_optional_int(raw.get("seed"), "training.optimizer.seed", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "population_size": self.population_size,
            "generations": self.generations,
            "sigma0": self.sigma0,
            "initial_mean": list(self.initial_mean),
            "bounds": [list(self.bounds[0]), list(self.bounds[1])],
            "seed": self.seed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one DR-conditioned FSRS-6 scheduler policy with CMA-ES.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
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
    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = _read_training_sa(args.config)
    policy_feature_version = _policy_feature_version(raw_training_sa)
    coefficient_count = feature_count(policy_feature_version)
    optimizer_settings = CMAESSettings.from_mapping(
        config.training_optimizer,
        coefficient_count=coefficient_count,
        coefficient_min=settings.coefficient_min,
        coefficient_max=settings.coefficient_max,
    )
    optimizer_seed = _optimizer_seed(
        config=config,
        settings=optimizer_settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
    )
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    progress.write(
        "config_loaded",
        settings=asdict(settings),
        optimizer=optimizer_settings.to_dict(),
        optimizer_seed=optimizer_seed,
        feature_version=policy_feature_version,
        simulation=config.simulation.to_dict(),
        seed=config.seed,
        baseline_desired_retention_values=list(baseline_dr_values),
        dr_batch_size=dr_batch_size,
    )

    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_sa.get("learning_steps"),
        relearning_steps=raw_training_sa.get("relearning_steps"),
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
        lanes=len(baseline_dr_values),
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baselines = _evaluate_fsrs6_baselines(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        baseline_dr_values=baseline_dr_values,
        seed=config.seed,
    )
    progress.write(
        "baselines_evaluated",
        device=baseline_bundle.device,
        effective_lanes=len(baseline_dr_values),
        metrics=[
            {
                "baseline_desired_retention": dr,
                **asdict(metrics),
            }
            for dr, metrics in zip(baseline_dr_values, baselines, strict=True)
        ],
    )
    del baseline_bundle
    _clear_cuda_cache(device)

    train_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=dr_batch_size * optimizer_settings.population_size,
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
        effective_lanes=dr_batch_size * optimizer_settings.population_size,
        grid_lanes=len(baseline_dr_values) * optimizer_settings.population_size,
    )
    result = _run_cmaes_dr_conditioned(
        config=config,
        settings=settings,
        optimizer_settings=optimizer_settings,
        optimizer_seed=optimizer_seed,
        bundle=train_bundle,
        baseline_dr_values=baseline_dr_values,
        dr_batch_size=dr_batch_size,
        baselines=baselines,
        lambda_value=args.lambda_value,
        feature_version=policy_feature_version,
        progress=progress,
    )
    progress.write(
        "cmaes_completed",
        device=train_bundle.device,
        best_score=result.best.score,
        mean_relative_memorized_gain=result.best.mean_relative_memorized_gain,
        mean_relative_efficiency_gain=result.best.mean_relative_efficiency_gain,
        generations=len(result.history),
    )

    policy_path, metrics_path, metadata_path = _write_artifact(
        output_dir=output_dir,
        config=config,
        config_path=args.config,
        settings=settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
        training_command_path=args.training_command_path,
        feature_version=policy_feature_version,
        result=result,
    )
    _augment_artifact(
        metrics_path=metrics_path,
        metadata_path=metadata_path,
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


def _run_cmaes_dr_conditioned(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
    bundle: Any,
    baseline_dr_values: tuple[float, ...],
    dr_batch_size: int,
    baselines: list[Any],
    lambda_value: float,
    feature_version: str,
    progress: TrainingProgress,
) -> DRConditionedTrainingResult:
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
    best_evaluation = None
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
        evaluations = _evaluate_sa_dr_chains(
            config=config,
            settings=settings,
            bundle=bundle,
            baseline_dr_values=baseline_dr_values,
            dr_batch_size=dr_batch_size,
            baselines=baselines,
            coefficients=coefficients,
            lambda_value=lambda_value,
            feature_version=feature_version,
            seed=config.seed,
        )
        scores = [evaluation.score for evaluation in evaluations]
        fitnesses = [-score for score in scores]
        es.tell(solutions, fitnesses)

        generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
        generation_best = evaluations[generation_best_idx]
        generation_best_score = float(scores[generation_best_idx])
        if generation_best_score > best_score:
            best_score = generation_best_score
            best_coefficients = coefficients[generation_best_idx].detach().clone()
            best_evaluation = generation_best

        history_entry = {
            "generation": float(generation),
            "sigma": float(es.sigma),
            "best_score": float(best_score),
            "generation_best_score": generation_best_score,
            "mean_score": float(sum(scores) / max(len(scores), 1)),
            "generation_best_mean_relative_memorized_gain": (
                generation_best.mean_relative_memorized_gain
            ),
            "generation_best_mean_relative_efficiency_gain": (
                generation_best.mean_relative_efficiency_gain
            ),
        }
        history.append(history_entry)
        progress.write(
            "cmaes_generation",
            device=bundle.device,
            effective_lanes=len(baseline_dr_values)
            * optimizer_settings.population_size,
            max_batch_lanes=dr_batch_size * optimizer_settings.population_size,
            **history_entry,
        )

    if best_coefficients is None or best_evaluation is None:
        raise RuntimeError("CMA-ES did not evaluate any candidates.")

    passed = (
        best_evaluation.mean_relative_memorized_gain > 0.0
        and best_evaluation.mean_relative_efficiency_gain > 0.0
    )
    return DRConditionedTrainingResult(
        baseline_desired_retention_values=baseline_dr_values,
        baselines=baselines,
        best_coefficients=best_coefficients.detach().cpu(),
        best=best_evaluation,
        history=history,
        passed=passed,
    )


def _augment_artifact(
    *,
    metrics_path: Path,
    metadata_path: Path,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
) -> None:
    optimizer = {
        **optimizer_settings.to_dict(),
        "seed_resolved": optimizer_seed,
    }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics JSON must be an object: {metrics_path}")
    metrics["optimizer"] = optimizer
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata JSON must be an object: {metadata_path}")
    metadata["optimizer"] = "cma_es"
    metadata["optimizer_settings"] = optimizer
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _optimizer_seed(
    *,
    config: ExperimentConfig,
    settings: CMAESSettings,
    user_id: int,
    lambda_value: float,
) -> int:
    if settings.seed is not None:
        return settings.seed
    return int(config.seed + 1009 * user_id + round(lambda_value * 1000))


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _float(value: Any, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if minimum is not None and result <= minimum:
        raise ValueError(f"{field_name} must be > {minimum}.")
    return result


def _int(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _optional_int(value: Any, field_name: str, minimum: int) -> int | None:
    if value is None:
        return None
    return _int(value, field_name, minimum)


def _float_tuple(value: Any, field_name: str, expected_count: int) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    values = tuple(
        _float(item, f"{field_name}[{idx}]") for idx, item in enumerate(value)
    )
    if len(values) != expected_count:
        raise ValueError(f"{field_name} must contain {expected_count} values.")
    return values


def _bounds(
    value: Any, expected_count: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("training.optimizer.bounds must be [lower, upper].")
    lower = _float_tuple(value[0], "training.optimizer.bounds[0]", expected_count)
    upper = _float_tuple(value[1], "training.optimizer.bounds[1]", expected_count)
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError(
            "training.optimizer.bounds lower values must be < upper values."
        )
    return lower, upper


if __name__ == "__main__":
    raise SystemExit(main())
