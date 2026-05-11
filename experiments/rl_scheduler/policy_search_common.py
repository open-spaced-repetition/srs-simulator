from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Sequence
import tomllib
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
from simulator.batched_sweep.weights import (
    build_default_fsrs6_weights,
    load_fsrs6_weights,
    resolve_lstm_paths,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.defaults import DEFAULT_LEARN_LIMIT, DEFAULT_SHORT_TERM_LOOPS_LIMIT
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.fsrs6_adr_policy import (
    FEATURE_VERSION,
    FSRS6ADRPolicy,
    feature_count,
)
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.batched_engine.multiuser_types import MultiUserBehavior, MultiUserCost


RELATIVE_GAIN_GATE_FLOOR = 0.0
RELATIVE_GAIN_GATE_PASS_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class PolicySearchSettings:
    coefficient_min: float = -8.0
    coefficient_max: float = 8.0
    retention_min: float = 0.70
    retention_max: float = 0.98
    baseline_desired_retention: float = 0.90
    torch_device: str = "cpu"
    short_term_threshold: float = 0.5
    short_term_loops_limit: int = DEFAULT_SHORT_TERM_LOOPS_LIMIT

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PolicySearchSettings:
        defaults = cls()
        return cls(
            coefficient_min=_float(
                raw.get("coefficient_min", defaults.coefficient_min),
                "training.policy_search.coefficient_min",
            ),
            coefficient_max=_float(
                raw.get("coefficient_max", defaults.coefficient_max),
                "training.policy_search.coefficient_max",
            ),
            retention_min=_float(
                raw.get("retention_min", defaults.retention_min),
                "training.policy_search.retention_min",
            ),
            retention_max=_float(
                raw.get("retention_max", defaults.retention_max),
                "training.policy_search.retention_max",
            ),
            baseline_desired_retention=_float(
                raw.get(
                    "baseline_desired_retention",
                    defaults.baseline_desired_retention,
                ),
                "training.policy_search.baseline_desired_retention",
            ),
            torch_device=_str(
                raw.get("torch_device", defaults.torch_device),
                "training.policy_search.torch_device",
            ),
            short_term_threshold=_float(
                raw.get("short_term_threshold", defaults.short_term_threshold),
                "training.policy_search.short_term_threshold",
                0.0,
            ),
            short_term_loops_limit=_int(
                raw.get("short_term_loops_limit", defaults.short_term_loops_limit),
                "training.policy_search.short_term_loops_limit",
                0,
            ),
        )

    def __post_init__(self) -> None:
        if self.coefficient_min >= self.coefficient_max:
            raise ValueError("coefficient_min must be less than coefficient_max.")
        if not (0.0 < self.retention_min < self.retention_max < 1.0):
            raise ValueError("retention bounds must satisfy 0 < min < max < 1.")
        if not (
            self.retention_min <= self.baseline_desired_retention <= self.retention_max
        ):
            raise ValueError(
                "baseline_desired_retention must be inside the retention bounds."
            )


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    memorized_average: float
    time_average: float
    memorized_per_minute: float
    total_reviews: int
    total_lapses: int
    total_cost: float


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
            sigma0=_float_gt(
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


@dataclass(frozen=True, slots=True)
class SimulationBundle:
    env_ops: Any
    scheduler_weights: torch.Tensor
    behavior: MultiUserBehavior
    cost_model: MultiUserCost
    device: torch.device
    short_term_source: str | None
    learning_steps: list[float]
    relearning_steps: list[float]


class TrainingProgress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        device: torch.device | None = None,
        **payload: Any,
    ) -> None:
        record: dict[str, Any] = {
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "elapsed_seconds": time.monotonic() - self._started,
            "event": event,
            **payload,
        }
        gpu = _progress_gpu_snapshot(device)
        if gpu is not None:
            record["gpu"] = gpu
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _build_bundle(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    user_id: int | None = None,
    lanes: int | None = None,
    lane_user_ids: Sequence[int] | None = None,
    benchmark_root: Path,
    overrides: dict[str, str],
    benchmark_partition: str | None,
    button_usage: Path | None,
    device: torch.device,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
) -> SimulationBundle:
    if lane_user_ids is None:
        if user_id is None or lanes is None:
            raise ValueError("user_id and lanes are required without lane_user_ids.")
        user_ids = [user_id for _ in range(lanes)]
    else:
        user_ids = [int(item) for item in lane_user_ids]
        lanes = len(user_ids)
        if lanes < 1:
            raise ValueError("lane_user_ids must not be empty.")
    short_term = bool(short_term_source)
    scheduler_weights, kept_users = load_fsrs6_weights(
        repo_root=REPO_ROOT,
        user_ids=user_ids,
        benchmark_root=benchmark_root,
        benchmark_partition=benchmark_partition,
        overrides=overrides,
        short_term=short_term,
        device=device,
    )
    if scheduler_weights is None or len(kept_users) != lanes:
        raise SystemExit(f"Missing FSRS-6 scheduler weights for user {user_id}.")

    environment = config.simulation.environment
    if environment == "lstm":
        lstm_paths, kept_lstm = resolve_lstm_paths(
            user_ids, benchmark_root, short_term=short_term
        )
        if len(kept_lstm) != lanes:
            raise SystemExit(f"Missing LSTM environment weights for user {user_id}.")
        packed = PackedLSTMWeights.from_paths(
            lstm_paths,
            use_duration_feature=False,
            device=device,
            dtype=torch.float32,
        )
        env_ops = LSTMBatchedEnvOps(
            packed,
            device=packed.process_0_weight.device,
            dtype=torch.float32,
        )
    elif environment == "fsrs6":
        env_weights, kept_env = load_fsrs6_weights(
            repo_root=REPO_ROOT,
            user_ids=user_ids,
            benchmark_root=benchmark_root,
            benchmark_partition=benchmark_partition,
            overrides=overrides,
            short_term=short_term,
            device=device,
        )
        if env_weights is None or len(kept_env) != lanes:
            raise SystemExit(f"Missing FSRS-6 environment weights for user {user_id}.")
        env_ops = FSRS6BatchEnvOps(
            weights=env_weights.to(device),
            bounds=Bounds(),
            device=device,
            dtype=torch.float32,
        )
    elif environment == "fsrs6_default":
        env_weights = build_default_fsrs6_weights(user_ids=user_ids, device=device)
        env_ops = FSRS6BatchEnvOps(
            weights=env_weights,
            bounds=Bounds(),
            device=device,
            dtype=torch.float32,
        )
    else:
        raise SystemExit(
            "FSRS6 ADR trainer supports lstm, fsrs6, and fsrs6_default environments."
        )

    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
        review_markov_success_weights,
    ) = load_usage(user_ids, button_usage)
    behavior, cost_model = build_behavior_cost(
        lanes,
        deck_size=config.simulation.deck,
        learn_limit=config.simulation.learn_limit or DEFAULT_LEARN_LIMIT,
        review_limit=config.simulation.review_limit,
        cost_limit_minutes=config.simulation.cost_limit_minutes,
        learn_costs=learn_costs.to(env_ops.device),
        review_costs=review_costs.to(env_ops.device),
        first_rating_prob=first_rating_prob.to(env_ops.device),
        review_rating_prob=review_rating_prob.to(env_ops.device),
        learning_rating_prob=learning_rating_prob.to(env_ops.device),
        relearning_rating_prob=relearning_rating_prob.to(env_ops.device),
        state_rating_costs=state_rating_costs.to(env_ops.device),
        review_markov_success_weights=review_markov_success_weights.to(env_ops.device),
        short_term=short_term,
    )
    return SimulationBundle(
        env_ops=env_ops,
        scheduler_weights=scheduler_weights.to(env_ops.device),
        behavior=behavior,
        cost_model=cost_model,
        device=env_ops.device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )


def _evaluate_fsrs6_baseline(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: SimulationBundle,
    seed: int,
) -> CandidateMetrics:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=settings.baseline_desired_retention,
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
    return _metrics_from_stats(stats[0])


def _evaluate_fsrs6_baseline_grid(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    job_count: int,
    seed: int,
    baseline_dr_values_by_job: Sequence[tuple[float, ...]] | None = None,
) -> list[list[CandidateMetrics]]:
    if baseline_dr_values_by_job is None:
        baseline_dr_values_by_job = [baseline_dr_values for _job in range(job_count)]
    if len(baseline_dr_values_by_job) != job_count:
        raise ValueError("baseline_dr_values_by_job length must match job_count.")
    flat_dr_values = [
        dr for job_dr_values in baseline_dr_values_by_job for dr in job_dr_values
    ]
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            flat_dr_values,
            device=bundle.device,
            dtype=torch.float32,
        ),
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
    split_metrics: list[list[CandidateMetrics]] = []
    offset = 0
    for job_dr_values in baseline_dr_values_by_job:
        next_offset = offset + len(job_dr_values)
        split_metrics.append(metrics[offset:next_offset])
        offset = next_offset
    return split_metrics


def _evaluate_adr_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: SimulationBundle,
    coefficients: torch.Tensor,
    feature_version: str = FEATURE_VERSION,
    seed: int,
) -> list[CandidateMetrics]:
    template = FSRS6ADRPolicy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    sched_ops = FSRS6ADRBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=template,
        coefficients=coefficients,
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


def _metrics_from_stats(stats: Any) -> CandidateMetrics:
    time_average = (
        sum(stats.daily_cost) / len(stats.daily_cost) / 60.0
        if stats.daily_cost
        else 0.0
    )
    memorized_average = (
        sum(stats.daily_memorized) / len(stats.daily_memorized)
        if stats.daily_memorized
        else 0.0
    )
    return CandidateMetrics(
        memorized_average=float(memorized_average),
        time_average=float(time_average),
        memorized_per_minute=float(
            memorized_average / time_average if time_average > 0.0 else 0.0
        ),
        total_reviews=int(stats.total_reviews),
        total_lapses=int(stats.total_lapses),
        total_cost=float(stats.total_cost),
    )


def _score(
    metrics: CandidateMetrics,
    baseline: CandidateMetrics,
    lambda_value: float,
) -> float:
    rel_mem = _relative_gain(metrics.memorized_average, baseline.memorized_average)
    rel_eff = _relative_gain(
        metrics.memorized_per_minute,
        baseline.memorized_per_minute,
    )
    return _score_from_relative_gains(rel_mem, rel_eff, lambda_value)


def _score_from_relative_gains(
    relative_memorized_gain: float,
    relative_efficiency_gain: float,
    lambda_value: float,
) -> float:
    if _passes_overfit_gate(relative_memorized_gain, relative_efficiency_gain):
        return float(
            (1.0 - lambda_value) * relative_memorized_gain
            + lambda_value * relative_efficiency_gain
            - RELATIVE_GAIN_GATE_FLOOR
        )
    violation = max(0.0, RELATIVE_GAIN_GATE_FLOOR - relative_memorized_gain) + max(
        0.0,
        RELATIVE_GAIN_GATE_FLOOR - relative_efficiency_gain,
    )
    return -float(violation)


def _relative_gain_point_passes(
    relative_memorized_gain: float,
    relative_efficiency_gain: float,
) -> bool:
    return (
        relative_memorized_gain > RELATIVE_GAIN_GATE_FLOOR
        and relative_efficiency_gain > RELATIVE_GAIN_GATE_FLOOR
    )


def _passes_overfit_gate(
    relative_memorized_gain: float,
    relative_efficiency_gain: float,
) -> bool:
    return _relative_gain_point_passes(
        relative_memorized_gain,
        relative_efficiency_gain,
    )


def _required_relative_gain_pass_count(
    point_count: int,
    *,
    pass_fraction: float = RELATIVE_GAIN_GATE_PASS_FRACTION,
) -> int:
    if point_count < 1:
        raise ValueError("point_count must be positive.")
    if not (0.0 < pass_fraction <= 1.0):
        raise ValueError("pass_fraction must satisfy 0 < pass_fraction <= 1.")
    return max(1, math.ceil(point_count * pass_fraction - 1e-12))


def _relative_gain_pass_count(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
) -> int:
    if len(relative_memorized_gains) != len(relative_efficiency_gains):
        raise ValueError("Relative gain grids must have the same length.")
    if not relative_memorized_gains:
        raise ValueError("Relative gain grids must not be empty.")
    return sum(
        1
        for rel_mem, rel_eff in zip(
            relative_memorized_gains,
            relative_efficiency_gains,
            strict=True,
        )
        if _relative_gain_point_passes(rel_mem, rel_eff)
    )


def _passes_relative_gain_fraction_gate(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
    *,
    pass_fraction: float = RELATIVE_GAIN_GATE_PASS_FRACTION,
) -> bool:
    passed_count = _relative_gain_pass_count(
        relative_memorized_gains,
        relative_efficiency_gains,
    )
    return passed_count >= _required_relative_gain_pass_count(
        len(relative_memorized_gains),
        pass_fraction=pass_fraction,
    )


def _relative_gain_gate_metrics(
    relative_memorized_gain: float,
    relative_efficiency_gain: float,
) -> dict[str, float | bool]:
    return {
        "relative_gain_floor": RELATIVE_GAIN_GATE_FLOOR,
        "memorized_average_gt_baseline": relative_memorized_gain > 0.0,
        "memorized_per_minute_gt_baseline": relative_efficiency_gain > 0.0,
        "memorized_average_gt_relative_gain_floor": (
            relative_memorized_gain > RELATIVE_GAIN_GATE_FLOOR
        ),
        "memorized_per_minute_gt_relative_gain_floor": (
            relative_efficiency_gain > RELATIVE_GAIN_GATE_FLOOR
        ),
        "relative_memorized_gain": relative_memorized_gain,
        "relative_efficiency_gain": relative_efficiency_gain,
    }


def _relative_gain_fraction_gate_metrics(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
    *,
    pass_fraction: float = RELATIVE_GAIN_GATE_PASS_FRACTION,
) -> dict[str, float | int | bool]:
    passed_count = _relative_gain_pass_count(
        relative_memorized_gains,
        relative_efficiency_gains,
    )
    total_count = len(relative_memorized_gains)
    required_count = _required_relative_gain_pass_count(
        total_count,
        pass_fraction=pass_fraction,
    )
    return {
        "relative_gain_floor": RELATIVE_GAIN_GATE_FLOOR,
        "relative_gain_pass_fraction_required": pass_fraction,
        "desired_retention_points": total_count,
        "passed_desired_retention_points": passed_count,
        "required_passed_desired_retention_points": required_count,
        "relative_gain_pass_fraction": passed_count / total_count,
        "passed_relative_gain_fraction_gate": passed_count >= required_count,
    }


def _relative_gain(value: float, baseline: float) -> float:
    denom = max(abs(baseline), 1e-9)
    return (float(value) - float(baseline)) / denom


def _baseline_dr_values(
    raw_training_policy: Mapping[str, Any],
    settings: PolicySearchSettings,
) -> tuple[float, ...]:
    raw_values = raw_training_policy.get("baseline_desired_retention_values")
    if raw_values is None:
        values = (settings.baseline_desired_retention,)
    else:
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise ValueError(
                "training.policy_search.baseline_desired_retention_values must be an array."
            )
        values = tuple(
            _float(item, "training.policy_search.baseline_desired_retention_values")
            for item in raw_values
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "training.policy_search.baseline_desired_retention_values must not contain duplicates."
        )
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "training.policy_search.baseline_desired_retention_values must be inside "
                "the retention bounds."
            )
    return values


def _dr_batch_size(raw_training_policy: Mapping[str, Any], value_count: int) -> int:
    default = min(max(value_count, 1), 4)
    return min(
        _int(
            raw_training_policy.get("dr_batch_size", default),
            "training.policy_search.dr_batch_size",
            1,
        ),
        value_count,
    )


def _progress_gpu_snapshot(device: torch.device | None) -> dict[str, int] | None:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "current_allocated_memory_bytes": int(torch.cuda.memory_allocated(device)),
        "current_reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _read_training_policy_search(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw.get("training", {})
    if not isinstance(training, Mapping):
        return {}
    policy_search = training.get("policy_search", {})
    return policy_search if isinstance(policy_search, Mapping) else {}


def _policy_feature_version(raw_training_policy: Mapping[str, Any]) -> str:
    value = raw_training_policy.get("feature_version", FEATURE_VERSION)
    if not isinstance(value, str):
        raise ValueError("training.policy_search.feature_version must be a string.")
    feature_count(value)
    return value


def _artifact_id(
    user_id: int,
    lambda_value: float,
    baseline_desired_retention: float,
    seed: int,
) -> str:
    lambda_token = _float_token(lambda_value)
    dr_token = _float_token(baseline_desired_retention)
    return f"fsrs6-adr-user-{user_id}-dr-{dr_token}-lambda-{lambda_token}-seed-{seed}"


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


def _float_token(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _iter_chunks(
    values: tuple[float, ...],
    chunk_size: int,
) -> Sequence[tuple[int, tuple[float, ...]]]:
    return [
        (start, values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.SubprocessError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _relative_path_string(path: Path, *, base: Path) -> str:
    relative = Path(os.path.relpath(path.resolve(), base.resolve()))
    return relative.as_posix()


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


def _float(value: Any, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return result


def _float_gt(value: Any, field_name: str, minimum: float) -> float:
    result = _float(value, field_name)
    if result <= minimum:
        raise ValueError(f"{field_name} must be > {minimum}.")
    return result


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


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()
