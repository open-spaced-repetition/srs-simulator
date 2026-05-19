from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import cma
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    _build_bundle,
    _evaluate_fsrs6_baseline_grid,
    _git_commit,
    _read_training_policy_search,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    objective_hypervolume_2d,
    point_from_metrics,
    reference_point,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.short_term_config import resolve_short_term_config


DEFAULT_MAX_LANES_PER_BATCH = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-user FSRS6 baseline DR selection manifests.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help=(
            "Manifest path. Defaults to [baseline_dr_selection].manifest from "
            "the config."
        ),
    )
    parser.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument(
        "--max-lanes-per-batch",
        type=int,
        default=DEFAULT_MAX_LANES_PER_BATCH,
        help=(
            "Maximum FSRS6 lanes per selector evaluation batch. The selector "
            "batches multiple users together up to this cap."
        ),
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help=("JSONL progress log path. Defaults to <output-manifest>.progress.jsonl."),
    )
    parser.add_argument(
        "--no-progress-log",
        action="store_true",
        help="Disable the JSONL progress log.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)
    output_manifest = args.output_manifest or config.baseline_dr_selection.manifest
    if output_manifest is None:
        raise SystemExit(
            "--output-manifest is required when baseline_dr_selection.manifest is absent."
        )
    if args.max_lanes_per_batch < 1:
        raise SystemExit("--max-lanes-per-batch must be >= 1.")
    if not output_manifest.is_absolute():
        output_manifest = (REPO_ROOT / output_manifest).resolve()
    progress_log_path = _resolve_progress_log_path(
        output_manifest=output_manifest,
        progress_log=args.progress_log,
        no_progress_log=args.no_progress_log,
    )

    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search = dict(_read_training_policy_search(args.config))
    benchmark_root = resolve_benchmark_root(
        REPO_ROOT,
        args.srs_benchmark_root,
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
    selection_config = config.baseline_dr_selection
    selection_simulation = replace(
        config.simulation,
        environment=selection_config.selection_environment,
    )
    selection_experiment = replace(config, simulation=selection_simulation)
    device = torch.device(settings.torch_device)

    with _SelectionProgressLogger.open(progress_log_path) as progress_logger:
        progress_logger.write_run_start(
            config_path=args.config,
            output_manifest=output_manifest,
            user_ids=config.users.train,
            selection_environment=selection_config.selection_environment,
            target_count=selection_config.target_count,
            population_size=selection_config.population_size,
            generations=selection_config.generations,
            max_lanes_per_batch=args.max_lanes_per_batch,
        )
        users = _select_users_drs_batched(
            config=selection_experiment,
            config_path=args.config,
            settings=settings,
            user_ids=config.users.train,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=args.benchmark_partition,
            button_usage=args.button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            max_lanes_per_batch=args.max_lanes_per_batch,
            progress_logger=progress_logger,
        )

        _write_json(
            output_manifest,
            {
                "schema_version": 1,
                "artifact_kind": "baseline-dr-selection-manifest",
                "created_at": _utc_timestamp(),
                "config": {
                    "path": str(args.config),
                    "sha256": _file_sha256(args.config),
                    "name": config.name,
                    "family": config.family,
                    "seed": config.seed,
                },
                "code_commit": _git_commit(),
                "selection_environment": selection_config.selection_environment,
                "simulation": selection_experiment.simulation.to_dict(),
                "button_usage": str(args.button_usage)
                if args.button_usage is not None
                else None,
                "target_count": selection_config.target_count,
                "reference": selection_config.reference,
                "objective": {
                    "name": "hypervolume_2d",
                    "dimensions": ["memorized_average", "negative_time_average"],
                },
                "optimizer": {
                    "name": "cma_es",
                    "population_size": selection_config.population_size,
                    "generations": selection_config.generations,
                    "max_lanes_per_batch": args.max_lanes_per_batch,
                },
                "progress_log": str(progress_log_path)
                if progress_log_path is not None
                else None,
                "users": users,
            },
        )
        progress_logger.write_run_complete(
            output_manifest=output_manifest,
            completed_users=len(users),
        )
    return 0


@dataclass(slots=True)
class _EvaluationJob:
    user_id: int
    candidate_index: int
    desired_retention_values: tuple[float, ...]


@dataclass(slots=True)
class _UserSelectionState:
    user_id: int
    strategy: Any
    sigma0: float
    reference: Any
    best_values: tuple[float, ...]
    best_metrics: list[CandidateMetrics]
    best_hv: float
    history: list[dict[str, Any]]


def _select_users_drs_batched(
    *,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    user_ids: Sequence[int],
    benchmark_root: Path,
    overrides: dict[str, str],
    benchmark_partition: str | None,
    button_usage: Path | None,
    device: torch.device,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
    max_lanes_per_batch: int,
    progress_logger: _SelectionProgressLogger,
) -> list[dict[str, Any]]:
    selection = config.baseline_dr_selection
    target_count = selection.target_count
    anchor = _uniform_dr_values(
        settings.retention_min,
        settings.retention_max,
        target_count,
    )
    anchor_metrics_by_user = _evaluate_multi_user_candidate_sets(
        config=config,
        settings=settings,
        user_candidate_sets={int(user_id): (anchor,) for user_id in user_ids},
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        max_lanes_per_batch=max_lanes_per_batch,
    )
    sigma0 = max((settings.retention_max - settings.retention_min) / 6.0, 1e-3)
    states: dict[int, _UserSelectionState] = {}
    for user_id in user_ids:
        user_id = int(user_id)
        anchor_metrics = anchor_metrics_by_user[user_id][0]
        anchor_points = [point_from_metrics(metrics) for metrics in anchor_metrics]
        reference = reference_point(anchor_points)
        best_hv = objective_hypervolume_2d(anchor_points, reference=reference)
        history = [
            {
                "generation": -1,
                "best_hypervolume": best_hv,
                "mean_hypervolume": best_hv,
            }
        ]
        progress_logger.write_generation(
            user_id=user_id,
            generation=-1,
            candidate_count=1,
            generation_best_hypervolume=best_hv,
            generation_mean_hypervolume=best_hv,
            generation_min_hypervolume=best_hv,
            generation_max_hypervolume=best_hv,
            incumbent_best_hypervolume=best_hv,
            improved=True,
            generation_best_desired_retention_values=anchor,
            incumbent_desired_retention_values=anchor,
        )
        states[user_id] = _UserSelectionState(
            user_id=user_id,
            strategy=cma.CMAEvolutionStrategy(
                list(anchor),
                sigma0,
                {
                    "bounds": [settings.retention_min, settings.retention_max],
                    "popsize": selection.population_size,
                    "seed": config.seed + 7919 * user_id,
                    "verbose": -9,
                },
            ),
            sigma0=sigma0,
            reference=reference,
            best_values=anchor,
            best_metrics=anchor_metrics,
            best_hv=best_hv,
            history=history,
        )
    _clear_cuda_cache(device)

    for generation in range(selection.generations):
        asked_by_user: dict[int, Sequence[Any]] = {}
        candidate_sets_by_user: dict[int, tuple[tuple[float, ...], ...]] = {}
        for user_id in user_ids:
            user_id = int(user_id)
            state = states[user_id]
            asked = state.strategy.ask()
            asked_by_user[user_id] = asked
            candidate_sets_by_user[user_id] = tuple(
                _canonical_dr_values(
                    values,
                    retention_min=settings.retention_min,
                    retention_max=settings.retention_max,
                    target_count=target_count,
                    tolerance=selection.tolerance,
                )
                for values in asked
            )
        metrics_by_user = _evaluate_multi_user_candidate_sets(
            config=config,
            settings=settings,
            user_candidate_sets=candidate_sets_by_user,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            max_lanes_per_batch=max_lanes_per_batch,
        )
        _clear_cuda_cache(device)
        for user_id in user_ids:
            user_id = int(user_id)
            state = states[user_id]
            candidate_sets = candidate_sets_by_user[user_id]
            metrics_by_candidate = metrics_by_user[user_id]
            hypervolumes = [
                objective_hypervolume_2d(
                    [point_from_metrics(metrics) for metrics in candidate_metrics],
                    reference=state.reference,
                )
                for candidate_metrics in metrics_by_candidate
            ]
            state.strategy.tell(
                asked_by_user[user_id], [-value for value in hypervolumes]
            )
            generation_best_index = max(
                range(len(hypervolumes)),
                key=lambda index: hypervolumes[index],
            )
            generation_best_hv = hypervolumes[generation_best_index]
            mean_hv = float(sum(hypervolumes) / len(hypervolumes))
            improved = generation_best_hv > state.best_hv
            if improved:
                state.best_hv = generation_best_hv
                state.best_values = candidate_sets[generation_best_index]
                state.best_metrics = metrics_by_candidate[generation_best_index]
            state.history.append(
                {
                    "generation": generation,
                    "best_hypervolume": state.best_hv,
                    "mean_hypervolume": mean_hv,
                    "generation_best_hypervolume": generation_best_hv,
                }
            )
            progress_logger.write_generation(
                user_id=user_id,
                generation=generation,
                candidate_count=len(candidate_sets),
                generation_best_hypervolume=generation_best_hv,
                generation_mean_hypervolume=mean_hv,
                generation_min_hypervolume=float(min(hypervolumes)),
                generation_max_hypervolume=float(max(hypervolumes)),
                incumbent_best_hypervolume=state.best_hv,
                improved=improved,
                generation_best_desired_retention_values=candidate_sets[
                    generation_best_index
                ],
                incumbent_desired_retention_values=state.best_values,
            )

    return [
        _manifest_entry_from_state(
            state=states[int(user_id)],
            config=config,
            config_path=config_path,
        )
        for user_id in user_ids
    ]


def _manifest_entry_from_state(
    *,
    state: _UserSelectionState,
    config: ExperimentConfig,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "user_id": state.user_id,
        "desired_retention_values": list(state.best_values),
        "objective": {
            "name": "hypervolume_2d",
            "hypervolume": state.best_hv,
            "anchor_hypervolume": state.history[0]["best_hypervolume"],
            "history": state.history,
        },
        "reference_point": asdict(state.reference),
        "selected_metrics": [
            {"desired_retention": dr, **asdict(metrics)}
            for dr, metrics in zip(state.best_values, state.best_metrics, strict=True)
        ],
        "optimizer": {
            "name": "cma_es",
            "population_size": config.baseline_dr_selection.population_size,
            "generations": config.baseline_dr_selection.generations,
            "sigma0": state.sigma0,
        },
        "config_snapshot": {
            "config_path": str(config_path),
            "selection_environment": config.baseline_dr_selection.selection_environment,
        },
    }


class _SelectionProgressLogger:
    def __init__(self, path: Path | None, handle: TextIO | None) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def open(cls, path: Path | None) -> _SelectionProgressLogger:
        if path is None:
            return cls(path=None, handle=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=path, handle=path.open("w", encoding="utf-8"))

    def __enter__(self) -> _SelectionProgressLogger:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_run_start(
        self,
        *,
        config_path: Path,
        output_manifest: Path,
        user_ids: Sequence[int],
        selection_environment: str,
        target_count: int,
        population_size: int,
        generations: int,
        max_lanes_per_batch: int,
    ) -> None:
        self._write(
            "run_started",
            config_path=str(config_path),
            output_manifest=str(output_manifest),
            user_ids=list(user_ids),
            selection_environment=selection_environment,
            target_count=target_count,
            population_size=population_size,
            generations=generations,
            max_lanes_per_batch=max_lanes_per_batch,
        )

    def write_generation(
        self,
        *,
        user_id: int,
        generation: int,
        candidate_count: int,
        generation_best_hypervolume: float,
        generation_mean_hypervolume: float,
        generation_min_hypervolume: float,
        generation_max_hypervolume: float,
        incumbent_best_hypervolume: float,
        improved: bool,
        generation_best_desired_retention_values: Sequence[float],
        incumbent_desired_retention_values: Sequence[float],
    ) -> None:
        self._write(
            "generation_evaluated",
            user_id=user_id,
            generation=generation,
            candidate_count=candidate_count,
            generation_best_hypervolume=generation_best_hypervolume,
            generation_mean_hypervolume=generation_mean_hypervolume,
            generation_min_hypervolume=generation_min_hypervolume,
            generation_max_hypervolume=generation_max_hypervolume,
            incumbent_best_hypervolume=incumbent_best_hypervolume,
            improved=improved,
            generation_best_desired_retention_values=list(
                generation_best_desired_retention_values
            ),
            incumbent_desired_retention_values=list(incumbent_desired_retention_values),
        )

    def write_run_complete(
        self, *, output_manifest: Path, completed_users: int
    ) -> None:
        self._write(
            "run_completed",
            output_manifest=str(output_manifest),
            completed_users=completed_users,
        )

    def _write(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            return
        payload = {
            "type": "baseline_dr_selection_progress",
            "event": event,
            "created_at": _utc_timestamp(),
            **fields,
        }
        json.dump(payload, self._handle, sort_keys=True, allow_nan=False)
        self._handle.write("\n")
        self._handle.flush()


def _evaluate_multi_user_candidate_sets(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    user_candidate_sets: Mapping[int, Sequence[tuple[float, ...]]],
    benchmark_root: Path,
    overrides: dict[str, str],
    benchmark_partition: str | None,
    button_usage: Path | None,
    device: torch.device,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
    max_lanes_per_batch: int,
) -> dict[int, list[list[CandidateMetrics]]]:
    jobs: list[_EvaluationJob] = []
    raw_results: dict[int, list[list[CandidateMetrics] | None]] = {}
    for user_id, candidate_sets in user_candidate_sets.items():
        user_id = int(user_id)
        raw_results[user_id] = [None for _candidate in candidate_sets]
        for candidate_index, desired_retention_values in enumerate(candidate_sets):
            jobs.append(
                _EvaluationJob(
                    user_id=user_id,
                    candidate_index=candidate_index,
                    desired_retention_values=desired_retention_values,
                )
            )
    if not jobs:
        return {}

    for chunk in _chunk_evaluation_jobs(
        jobs,
        max_lanes_per_batch=max_lanes_per_batch,
    ):
        lane_user_ids = [
            job.user_id for job in chunk for _dr in job.desired_retention_values
        ]
        baseline_dr_values_by_job = [job.desired_retention_values for job in chunk]
        bundle = _build_bundle(
            config=config,
            settings=settings,
            lane_user_ids=lane_user_ids,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        try:
            metrics_by_job = _evaluate_fsrs6_baseline_grid(
                config=config,
                settings=settings,
                bundle=bundle,
                baseline_dr_values=baseline_dr_values_by_job[0],
                baseline_dr_values_by_job=baseline_dr_values_by_job,
                job_count=len(chunk),
                seed=config.seed,
            )
        finally:
            del bundle
        for job, metrics in zip(chunk, metrics_by_job, strict=True):
            raw_results[job.user_id][job.candidate_index] = metrics

    results: dict[int, list[list[CandidateMetrics]]] = {}
    for user_id, user_results in raw_results.items():
        completed: list[list[CandidateMetrics]] = []
        for item in user_results:
            if item is None:
                raise RuntimeError(
                    f"Missing baseline DR selection metrics for user {user_id}."
                )
            completed.append(item)
        results[user_id] = completed
    return results


def _chunk_evaluation_jobs(
    jobs: Sequence[_EvaluationJob],
    *,
    max_lanes_per_batch: int,
) -> list[list[_EvaluationJob]]:
    if max_lanes_per_batch < 1:
        raise ValueError("max_lanes_per_batch must be >= 1.")
    chunks: list[list[_EvaluationJob]] = []
    current: list[_EvaluationJob] = []
    current_lanes = 0
    for job in jobs:
        job_lanes = len(job.desired_retention_values)
        if job_lanes < 1:
            raise ValueError("candidate desired_retention_values must not be empty.")
        if current and current_lanes + job_lanes > max_lanes_per_batch:
            chunks.append(current)
            current = []
            current_lanes = 0
        current.append(job)
        current_lanes += job_lanes
    if current:
        chunks.append(current)
    return chunks


def _uniform_dr_values(
    retention_min: float,
    retention_max: float,
    count: int,
) -> tuple[float, ...]:
    if count == 1:
        return (float((retention_min + retention_max) / 2.0),)
    step = (retention_max - retention_min) / (count - 1)
    return tuple(float(retention_min + step * index) for index in range(count))


def _canonical_dr_values(
    values: Sequence[float],
    *,
    retention_min: float,
    retention_max: float,
    target_count: int,
    tolerance: float,
) -> tuple[float, ...]:
    clipped = sorted(
        min(retention_max, max(retention_min, float(value))) for value in values
    )
    result: list[float] = []
    for value in clipped:
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    for value in _uniform_dr_values(retention_min, retention_max, target_count):
        if len(result) >= target_count:
            break
        if all(abs(value - existing) > tolerance for existing in result):
            result.append(value)
    result = sorted(result[:target_count])
    if len(result) != target_count:
        raise ValueError("Could not build a de-duplicated DR vector.")
    return tuple(result)


def _resolve_progress_log_path(
    *,
    output_manifest: Path,
    progress_log: Path | None,
    no_progress_log: bool,
) -> Path | None:
    if no_progress_log:
        return None
    path = progress_log or output_manifest.with_suffix(".progress.jsonl")
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
