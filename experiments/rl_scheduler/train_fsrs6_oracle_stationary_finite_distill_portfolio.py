from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    _float,
    _git_commit,
    _int,
    _metrics_from_stats,
    _relative_path_string,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    LightweightSelectionPool,
    ObjectivePoint,
    SelectionTask,
)
from experiments.rl_scheduler.portfolio_training_common import (
    PortfolioFamilyAdapter,
    clear_cuda_cache as _common_clear_cuda_cache,
    float_tuple as _common_float_tuple,
    generator_for_job,
    optional_float_tuple as _common_optional_float_tuple,
    progress_for_jobs,
    run_portfolio_train_jobs as _run_common_portfolio_train_jobs,
    selection_executor as _common_selection_executor,
    select_portfolio_children as _common_select_portfolio_children,
    select_survivors_for_generation as _common_select_survivors_for_generation,
    zero_metrics,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_HIDDEN_SIZE,
    DEFAULT_DISTILL_NETWORK_DEPTH,
    DEFAULT_DISTILL_SUPERVISION,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (
    DEFAULT_AGREEMENT_ENVS_PER_USER,
    DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
    DEFAULT_TRAIN_ENVS_PER_USER,
    SingleUserTrainStats,
    build_guide,
    estimate_batched_per_user_table_agreement,
    load_user_configs,
    materialize_ensemble_model,
    save_single_user_checkpoint,
    train_batched_per_user_models,
)
from experiments.single_card_tradeoff.core.defaults import DEFAULT_TARGET_RETENTIONS
from experiments.single_card_tradeoff.cli.uvfa_ppo import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
)
from experiments.single_card_tradeoff.models.policy_net import PolicyValueNet
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_oracle_stationary_finite_distill_policy import (
    FEATURE_VERSION,
    POLICY_TYPE,
    PORTFOLIO_CHILD_ACTION_SPACE,
    FSRS6OracleStationaryFiniteDistillPolicy,
)
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs6_oracle_stationary_finite_distill import (
    FSRS6OracleStationaryFiniteDistillBatchSchedulerOps,
)


_SELECTION_PROCESS_POOL_ENV = (
    "FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_PORTFOLIO_SELECTION_PROCESS_POOL"
)
_SELECTION_PROCESS_POOL_WORKERS_ENV = (
    "FSRS6_ORACLE_STATIONARY_FINITE_DISTILL_PORTFOLIO_SELECTION_WORKERS"
)
_SELECTION_ENV_VARS = (
    _SELECTION_PROCESS_POOL_ENV,
    "FSRS6_PORTFOLIO_SELECTION_PROCESS_POOL",
)
_SELECTION_WORKER_ENV_VARS = (
    _SELECTION_PROCESS_POOL_WORKERS_ENV,
    "FSRS6_PORTFOLIO_SELECTION_WORKERS",
)


@dataclass(frozen=True, slots=True)
class OracleDistillPortfolioSettings:
    algorithm: str = "sms_emoa"
    population_size: int = 16
    generations: int = 20
    offspring_size: int = 16
    portfolio_size: int = 16
    mutation_scale: float = 0.20
    reference_margin_fraction: float = 0.05
    hv_epsilon: float = 0.0
    min_goal_cost_weight: float = 0.0
    max_goal_cost_weight: float = 1024.0
    seed_cost_weights: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        settings: PolicySearchSettings,
        default_seed_retention_values: Sequence[float],
    ) -> OracleDistillPortfolioSettings:
        del settings, default_seed_retention_values
        defaults = cls()
        seed_cost_weights = _optional_float_tuple(
            raw.get("seed_cost_weights"),
            "training.portfolio.seed_cost_weights",
        )
        if seed_cost_weights is None:
            seed_cost_weights = tuple(
                float(value) for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
            )
        algorithm = raw.get("algorithm", defaults.algorithm)
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("training.portfolio.algorithm must be a non-empty string.")
        min_goal = _float(
            raw.get("min_goal_cost_weight", defaults.min_goal_cost_weight),
            "training.portfolio.min_goal_cost_weight",
            0.0,
        )
        max_goal = _float(
            raw.get("max_goal_cost_weight", defaults.max_goal_cost_weight),
            "training.portfolio.max_goal_cost_weight",
            0.0,
        )
        if max_goal <= min_goal:
            raise ValueError(
                "training.portfolio.max_goal_cost_weight must be greater than "
                "min_goal_cost_weight."
            )
        for value in seed_cost_weights:
            if value < min_goal or value > max_goal:
                raise ValueError(
                    "training.portfolio.seed_cost_weights must be inside the "
                    "configured goal cost weight bounds."
                )
        return cls(
            algorithm=algorithm.strip(),
            population_size=_int(
                raw.get("population_size", defaults.population_size),
                "training.portfolio.population_size",
                1,
            ),
            generations=_int(
                raw.get("generations", defaults.generations),
                "training.portfolio.generations",
                0,
            ),
            offspring_size=_int(
                raw.get("offspring_size", defaults.offspring_size),
                "training.portfolio.offspring_size",
                1,
            ),
            portfolio_size=_int(
                raw.get("portfolio_size", defaults.portfolio_size),
                "training.portfolio.portfolio_size",
                1,
            ),
            mutation_scale=_float(
                raw.get("mutation_scale", defaults.mutation_scale),
                "training.portfolio.mutation_scale",
                0.0,
            ),
            reference_margin_fraction=_float(
                raw.get(
                    "reference_margin_fraction",
                    defaults.reference_margin_fraction,
                ),
                "training.portfolio.reference_margin_fraction",
                0.0,
            ),
            hv_epsilon=_float(
                raw.get("hv_epsilon", defaults.hv_epsilon),
                "training.portfolio.hv_epsilon",
                0.0,
            ),
            min_goal_cost_weight=min_goal,
            max_goal_cost_weight=max_goal,
            seed_cost_weights=seed_cost_weights,
        )

    def __post_init__(self) -> None:
        if self.algorithm != "sms_emoa":
            raise ValueError("training.portfolio.algorithm must be 'sms_emoa'.")
        if self.reference_margin_fraction < 0.0:
            raise ValueError("reference_margin_fraction must be >= 0.")
        if self.hv_epsilon < 0.0:
            raise ValueError("hv_epsilon must be >= 0.")
        if self.min_goal_cost_weight < 0.0:
            raise ValueError("min_goal_cost_weight must be >= 0.")
        if self.max_goal_cost_weight <= self.min_goal_cost_weight:
            raise ValueError("max_goal_cost_weight must be greater than min.")
        if self.seed_cost_weights is not None and not self.seed_cost_weights:
            raise ValueError("seed_cost_weights must not be empty.")


@dataclass(frozen=True, slots=True)
class OracleDistillPortfolioTrainJob:
    user_id: int
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class OracleDistillPortfolioTrainOutcome:
    job: OracleDistillPortfolioTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path
    error: str | None = None


@dataclass(frozen=True, slots=True)
class OracleDistillPortfolioCandidate:
    candidate_id: int
    goal_cost_weight: float
    metrics: CandidateMetrics

    @property
    def point(self) -> ObjectivePoint:
        from experiments.rl_scheduler.portfolio_selection import point_from_metrics

        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedOracleDistillPortfolioChild:
    portfolio_index: int
    candidate: OracleDistillPortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


@dataclass(frozen=True, slots=True)
class OracleDistillFamilyContext:
    config: ExperimentConfig
    distill_args: argparse.Namespace
    distill_cost_weights: tuple[float, ...]
    action_retentions: tuple[float, ...]
    goal_norm_max: float


@dataclass(frozen=True, slots=True)
class OracleDistillFamilyState:
    checkpoint_by_user_id: dict[int, Path]
    user_ids_by_job: tuple[int, ...]
    distill_summary_path: Path


@dataclass(frozen=True, slots=True)
class UserOracleDistillPortfolioResult:
    job: OracleDistillPortfolioTrainJob
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    portfolio_hypervolume: float
    hypervolume_improvement: float
    final_population_hypervolume: float
    final_population_hypervolume_improvement: float
    reference_point: ObjectivePoint
    selected_children: list[SelectedOracleDistillPortfolioChild]
    final_population: list[OracleDistillPortfolioCandidate]
    family_state: OracleDistillFamilyState
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train per-user FSRS6 oracle stationary finite distill checkpoints and "
            "search a goal-cost-weight policy portfolio with SMS-EMOA."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
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
    config = ExperimentConfig.from_toml(args.config)
    outcomes = run_portfolio_train_jobs(
        jobs=[
            OracleDistillPortfolioTrainJob(
                user_id=args.user_id,
                output_dir=args.output_dir,
                command_record_path=args.training_command_path,
            )
        ],
        config=config,
        config_path=args.config,
        repo_root=REPO_ROOT,
        button_usage=args.button_usage,
        srs_benchmark_root=args.srs_benchmark_root,
        benchmark_result=args.benchmark_result,
        benchmark_partition=args.benchmark_partition,
        execution_mode="subprocess",
    )
    return 0 if outcomes and outcomes[0].passed else 1


def run_portfolio_train_jobs(
    *,
    jobs: Sequence[OracleDistillPortfolioTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[OracleDistillPortfolioTrainOutcome]:
    return _run_common_portfolio_train_jobs(
        jobs=jobs,
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        adapter=_ADAPTER,
        button_usage=button_usage,
        srs_benchmark_root=srs_benchmark_root,
        benchmark_result=benchmark_result,
        benchmark_partition=benchmark_partition,
        execution_mode=execution_mode,
    )


def _build_family_context(
    *,
    config: ExperimentConfig,
    raw_training_policy_search: Mapping[str, Any],
    baseline_dr_values: Sequence[float],
) -> OracleDistillFamilyContext:
    del baseline_dr_values
    settings = PolicySearchSettings.from_mapping(raw_training_policy_search)
    distill_cost_weights = _optional_float_tuple(
        raw_training_policy_search.get("distill_cost_weights"),
        "training.policy_search.distill_cost_weights",
    )
    if distill_cost_weights is None:
        distill_cost_weights = tuple(
            float(value) for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
        )
    action_retentions = _optional_float_tuple(
        raw_training_policy_search.get("distill_action_retentions"),
        "training.policy_search.distill_action_retentions",
    )
    if action_retentions is None:
        action_retentions = tuple(float(value) for value in DEFAULT_TARGET_RETENTIONS)
    goal_norm_max = _float(
        raw_training_policy_search.get(
            "distill_goal_norm_max", max(distill_cost_weights)
        ),
        "training.policy_search.distill_goal_norm_max",
        1.0,
    )
    args = argparse.Namespace(
        env="fsrs6",
        user_id=None,
        days=config.simulation.days,
        deck_scale=config.simulation.deck,
        seed=config.seed,
        torch_device=settings.torch_device,
        benchmark_result=raw_training_policy_search.get("benchmark_result"),
        benchmark_partition=str(
            raw_training_policy_search.get("benchmark_partition", "0")
        ),
        srs_benchmark_root=None,
        button_usage=DEFAULT_BUTTON_USAGE_PATH,
        dp_cache_enabled=bool(
            raw_training_policy_search.get("distill_dp_cache_enabled", True)
        ),
        dp_cache_dir=Path(
            str(
                raw_training_policy_search.get(
                    "distill_dp_cache_dir",
                    "artifacts/single_card_tradeoff/dp_cache",
                )
            )
        ),
        refresh_dp_cache=bool(
            raw_training_policy_search.get("distill_refresh_dp_cache", False)
        ),
        train_envs_per_user=_int(
            raw_training_policy_search.get(
                "distill_train_envs_per_user", DEFAULT_TRAIN_ENVS_PER_USER
            ),
            "training.policy_search.distill_train_envs_per_user",
            1,
        ),
        epochs=_int(
            raw_training_policy_search.get("distill_epochs", DEFAULT_DISTILL_EPOCHS),
            "training.policy_search.distill_epochs",
            1,
        ),
        steps_per_epoch=_int(
            raw_training_policy_search.get("distill_steps_per_epoch", 64),
            "training.policy_search.distill_steps_per_epoch",
            1,
        ),
        learning_rate=_float(
            raw_training_policy_search.get(
                "distill_learning_rate", DEFAULT_LEARNING_RATE
            ),
            "training.policy_search.distill_learning_rate",
            0.0,
        ),
        per_user_supervision=str(
            raw_training_policy_search.get(
                "distill_supervision", DEFAULT_DISTILL_SUPERVISION
            )
        ),
        table_samples_per_weight=_int(
            raw_training_policy_search.get(
                "distill_table_samples_per_weight", DEFAULT_TABLE_SAMPLES_PER_WEIGHT
            ),
            "training.policy_search.distill_table_samples_per_weight",
            1,
        ),
        network=str(raw_training_policy_search.get("distill_network", DEFAULT_NETWORK)),
        network_depth=_int(
            raw_training_policy_search.get(
                "distill_network_depth", DEFAULT_DISTILL_NETWORK_DEPTH
            ),
            "training.policy_search.distill_network_depth",
            1,
        ),
        hidden_size=_int(
            raw_training_policy_search.get(
                "distill_hidden_size", DEFAULT_DISTILL_HIDDEN_SIZE
            ),
            "training.policy_search.distill_hidden_size",
            1,
        ),
        oracle_s_grid_size=_int(
            raw_training_policy_search.get(
                "distill_oracle_s_grid_size", DEFAULT_ORACLE_S_GRID_SIZE
            ),
            "training.policy_search.distill_oracle_s_grid_size",
            2,
        ),
        oracle_d_grid_size=_int(
            raw_training_policy_search.get(
                "distill_oracle_d_grid_size", DEFAULT_ORACLE_D_GRID_SIZE
            ),
            "training.policy_search.distill_oracle_d_grid_size",
            2,
        ),
        oracle_stationary_finite_max_iterations=_int(
            raw_training_policy_search.get(
                "distill_oracle_stationary_finite_max_iterations",
                DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
            ),
            "training.policy_search.distill_oracle_stationary_finite_max_iterations",
            1,
        ),
        oracle_stationary_finite_tolerance=_float(
            raw_training_policy_search.get(
                "distill_oracle_stationary_finite_tolerance",
                DEFAULT_STATIONARY_FINITE_TOLERANCE,
            ),
            "training.policy_search.distill_oracle_stationary_finite_tolerance",
            0.0,
        ),
        oracle_teacher_user_batch_size=_int(
            raw_training_policy_search.get(
                "distill_oracle_teacher_user_batch_size",
                DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
            ),
            "training.policy_search.distill_oracle_teacher_user_batch_size",
            0,
        ),
        max_grad_norm=_float(
            raw_training_policy_search.get(
                "distill_max_grad_norm", DEFAULT_MAX_GRAD_NORM
            ),
            "training.policy_search.distill_max_grad_norm",
            0.0,
        ),
        agreement_envs_per_user=_int(
            raw_training_policy_search.get(
                "distill_agreement_envs_per_user", DEFAULT_AGREEMENT_ENVS_PER_USER
            ),
            "training.policy_search.distill_agreement_envs_per_user",
            1,
        ),
        agreement_steps=_int(
            raw_training_policy_search.get("distill_agreement_steps", 1),
            "training.policy_search.distill_agreement_steps",
            0,
        ),
        no_progress=bool(raw_training_policy_search.get("distill_no_progress", True)),
    )
    if args.per_user_supervision not in {"uniform_table", "rollout"}:
        raise ValueError(
            "training.policy_search.distill_supervision must be uniform_table or rollout."
        )
    return OracleDistillFamilyContext(
        config=config,
        distill_args=args,
        distill_cost_weights=distill_cost_weights,
        action_retentions=action_retentions,
        goal_norm_max=goal_norm_max,
    )


def _progress_payload(
    *, family_context: OracleDistillFamilyContext
) -> Mapping[str, Any]:
    return {
        "feature_version": FEATURE_VERSION,
        "distill_cost_weights": list(family_context.distill_cost_weights),
        "action_retentions": list(family_context.action_retentions),
        "goal_norm_max": family_context.goal_norm_max,
    }


def _prepare_family_state(
    *,
    bundle: Any,
    jobs: Sequence[OracleDistillPortfolioTrainJob],
    portfolio: OracleDistillPortfolioSettings,
    family_context: OracleDistillFamilyContext,
) -> OracleDistillFamilyState:
    del bundle, portfolio
    return _train_distill_checkpoints(jobs=jobs, family_context=family_context)


def _train_distill_checkpoints(
    *,
    jobs: Sequence[OracleDistillPortfolioTrainJob],
    family_context: OracleDistillFamilyContext,
) -> OracleDistillFamilyState:
    args = family_context.distill_args
    user_ids = [job.user_id for job in jobs]
    configs = load_user_configs(args, user_ids)
    device = torch.device(args.torch_device)
    setup_start = time.perf_counter()
    params_probe = PolicyValueNet(
        3,
        len(family_context.action_retentions),
        args.hidden_size,
        architecture=args.network,
        depth=args.network_depth,
    )
    params = sum(param.numel() for param in params_probe.parameters())
    setup_runtime_s = time.perf_counter() - setup_start
    guide, teacher_runtime_s = build_guide(
        args,
        device=device,
        configs=configs,
        cost_weights=family_context.distill_cost_weights,
        action_retentions=family_context.action_retentions,
    )
    ensemble, train_runtime_s, final_loss_by_user, final_agreement_by_user = (
        train_batched_per_user_models(
            args,
            device=device,
            configs=configs,
            guide=guide,
            cost_weights=family_context.distill_cost_weights,
            action_retentions=family_context.action_retentions,
            params_per_user=params,
        )
    )
    eval_agreement, eval_agreement_by_user, agreement_runtime_s = (
        estimate_batched_per_user_table_agreement(
            ensemble=ensemble,
            guide=guide,
            device=device,
            cost_weights=family_context.distill_cost_weights,
        )
    )
    del eval_agreement
    ensemble_trainable_params = params * len(user_ids)
    checkpoint_by_user_id: dict[int, Path] = {}
    stats_rows: list[dict[str, Any]] = []
    for user_idx, (job, user_id) in enumerate(zip(jobs, user_ids, strict=True)):
        train_samples_per_user = (
            args.table_samples_per_weight * len(family_context.distill_cost_weights)
            if args.per_user_supervision == "uniform_table"
            else args.train_envs_per_user
        )
        user_stats = SingleUserTrainStats(
            user_id=user_id,
            user_index=user_idx,
            params_per_user=params,
            ensemble_trainable_params=ensemble_trainable_params,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            train_envs=train_samples_per_user,
            train_transitions=args.epochs
            * args.steps_per_epoch
            * train_samples_per_user,
            train_runtime_s=train_runtime_s,
            agreement_runtime_s=agreement_runtime_s,
            final_ce_loss=final_loss_by_user[user_idx],
            final_teacher_action_agreement=final_agreement_by_user[user_idx],
            eval_teacher_action_agreement=eval_agreement_by_user[user_idx],
            supervision=args.per_user_supervision,
            table_samples_per_weight=args.table_samples_per_weight,
        )
        checkpoint_path = job.output_dir / "distill" / f"user_{user_id}_policy.pt"
        model = materialize_ensemble_model(
            args,
            ensemble=ensemble,
            user_idx=user_idx,
            obs_dim=3,
            action_count=len(family_context.action_retentions),
        )
        save_single_user_checkpoint(
            checkpoint_path,
            model=model,
            args=args,
            user_id=user_id,
            user_idx=user_idx,
            config=configs[user_idx],
            cost_weights=family_context.distill_cost_weights,
            action_retentions=family_context.action_retentions,
            guide=guide,
            stats=user_stats,
            teacher_runtime_s=teacher_runtime_s,
            training_scope="rl_scheduler_train_overfit_per_user_batched",
        )
        checkpoint_by_user_id[user_id] = checkpoint_path
        stats_rows.append(asdict(user_stats))
    summary_path = jobs[0].output_dir.parent / "distill_training_summary.json"
    _write_json(
        summary_path,
        {
            "policy_type": POLICY_TYPE,
            "feature_version": FEATURE_VERSION,
            "user_ids": user_ids,
            "checkpoint_paths": {
                str(user_id): str(path)
                for user_id, path in checkpoint_by_user_id.items()
            },
            "distill_cost_weights": list(family_context.distill_cost_weights),
            "action_retentions": list(family_context.action_retentions),
            "setup_runtime_s": setup_runtime_s,
            "teacher_runtime_s": teacher_runtime_s,
            "train_runtime_s": train_runtime_s,
            "agreement_runtime_s": agreement_runtime_s,
            "stats_by_user": stats_rows,
        },
    )
    clear_cuda_cache(device)
    return OracleDistillFamilyState(
        checkpoint_by_user_id=checkpoint_by_user_id,
        user_ids_by_job=tuple(user_ids),
        distill_summary_path=summary_path,
    )


def _evaluate_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    portfolio: OracleDistillPortfolioSettings,
    family_context: OracleDistillFamilyContext,
    family_state: OracleDistillFamilyState,
    bundle: Any,
    candidates_by_job: Sequence[Sequence[OracleDistillPortfolioCandidate]],
    seed: int,
) -> list[list[OracleDistillPortfolioCandidate]]:
    del portfolio
    candidate_count = len(candidates_by_job[0])
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")
    flat_policies: list[FSRS6OracleStationaryFiniteDistillPolicy] = []
    for job_index, candidates in enumerate(candidates_by_job):
        if len(candidates) != candidate_count:
            raise ValueError("All jobs must evaluate the same number of candidates.")
        user_id = family_state.user_ids_by_job[job_index]
        checkpoint_path = family_state.checkpoint_by_user_id[int(user_id)]
        for candidate in candidates:
            flat_policies.append(
                FSRS6OracleStationaryFiniteDistillPolicy(
                    checkpoint_path=checkpoint_path,
                    goal_cost_weight=candidate.goal_cost_weight,
                    goal_norm_max=family_context.goal_norm_max,
                    action_retentions=family_context.action_retentions,
                    user_id=int(user_id),
                    portfolio_index=None,
                )
            )
    sched_ops = FSRS6OracleStationaryFiniteDistillBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policies=flat_policies,
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
    return [
        [
            OracleDistillPortfolioCandidate(
                candidate_id=candidates[index].candidate_id,
                goal_cost_weight=candidates[index].goal_cost_weight,
                metrics=metrics[job_index * candidate_count + index],
            )
            for index in range(len(candidates))
        ]
        for job_index, candidates in enumerate(candidates_by_job)
    ]


def _initial_populations(
    *,
    jobs: Sequence[OracleDistillPortfolioTrainJob],
    settings: PolicySearchSettings,
    portfolio: OracleDistillPortfolioSettings,
    family_context: OracleDistillFamilyContext,
    seed_retention_values_by_job: Sequence[Sequence[float]],
    device: torch.device,
    seed: int,
) -> tuple[list[list[OracleDistillPortfolioCandidate]], list[int]]:
    del settings, family_context, seed_retention_values_by_job
    populations: list[list[OracleDistillPortfolioCandidate]] = []
    next_ids: list[int] = []
    seed_weights = tuple(portfolio.seed_cost_weights or ())
    for job in jobs:
        generator = generator_for_job(device=device, seed=seed, user_id=job.user_id)
        candidates: list[OracleDistillPortfolioCandidate] = []
        for index in range(portfolio.population_size):
            if index < len(seed_weights):
                goal_weight = float(seed_weights[index])
            else:
                base = seed_weights[index % len(seed_weights)]
                goal_weight = _mutate_goal_weight(
                    base,
                    portfolio=portfolio,
                    device=device,
                    generator=generator,
                )
            candidates.append(
                OracleDistillPortfolioCandidate(
                    candidate_id=index,
                    goal_cost_weight=goal_weight,
                    metrics=zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[OracleDistillPortfolioCandidate],
    next_candidate_id: int,
    settings: PolicySearchSettings,
    portfolio: OracleDistillPortfolioSettings,
    family_context: OracleDistillFamilyContext,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[OracleDistillPortfolioCandidate], int]:
    del settings, family_context
    candidates: list[OracleDistillPortfolioCandidate] = []
    for _index in range(portfolio.offspring_size):
        parent_index = int(
            torch.randint(
                len(population),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        goal_weight = _mutate_goal_weight(
            population[parent_index].goal_cost_weight,
            portfolio=portfolio,
            device=device,
            generator=generator,
        )
        candidates.append(
            OracleDistillPortfolioCandidate(
                candidate_id=next_candidate_id,
                goal_cost_weight=goal_weight,
                metrics=zero_metrics(),
            )
        )
        next_candidate_id += 1
    return candidates, next_candidate_id


def _mutate_goal_weight(
    value: float,
    *,
    portfolio: OracleDistillPortfolioSettings,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    lo = portfolio.min_goal_cost_weight
    hi = portfolio.max_goal_cost_weight
    span = math.log1p(hi - lo)
    base_z = math.log1p(max(0.0, float(value) - lo)) / span
    noise = float(
        torch.randn((), device=device, generator=generator, dtype=torch.float32).item()
    )
    z = min(1.0, max(0.0, base_z + noise * portfolio.mutation_scale))
    return lo + math.expm1(z * span)


def _selected_child_from_candidate(
    *,
    portfolio_index: int,
    candidate: OracleDistillPortfolioCandidate,
    hypervolume_contribution: float,
    pareto_rank: int,
) -> SelectedOracleDistillPortfolioChild:
    return SelectedOracleDistillPortfolioChild(
        portfolio_index=portfolio_index,
        candidate=candidate,
        hypervolume_contribution=hypervolume_contribution,
        pareto_rank=pareto_rank,
    )


def _build_result(
    *,
    job: OracleDistillPortfolioTrainJob,
    job_index: int,
    baseline_desired_retention_values: tuple[float, ...],
    baseline_metrics: list[CandidateMetrics],
    baseline_hypervolume: float,
    portfolio_hypervolume: float,
    hypervolume_improvement: float,
    final_population_hypervolume: float,
    final_population_hypervolume_improvement: float,
    reference_point: ObjectivePoint,
    selected_children: list[SelectedOracleDistillPortfolioChild],
    final_population: list[OracleDistillPortfolioCandidate],
    family_state: OracleDistillFamilyState,
    history: list[dict[str, float]],
    passed: bool,
) -> UserOracleDistillPortfolioResult:
    del job_index
    return UserOracleDistillPortfolioResult(
        job=job,
        baseline_desired_retention_values=baseline_desired_retention_values,
        baseline_metrics=baseline_metrics,
        baseline_hypervolume=baseline_hypervolume,
        portfolio_hypervolume=portfolio_hypervolume,
        hypervolume_improvement=hypervolume_improvement,
        final_population_hypervolume=final_population_hypervolume,
        final_population_hypervolume_improvement=(
            final_population_hypervolume_improvement
        ),
        reference_point=reference_point,
        selected_children=selected_children,
        final_population=final_population,
        family_state=family_state,
        history=history,
        passed=passed,
    )


def _write_portfolio_artifacts(
    *,
    result: UserOracleDistillPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    portfolio: OracleDistillPortfolioSettings,
    family_context: OracleDistillFamilyContext,
) -> list[Path]:
    del settings
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_name = "fsrs6_oracle_stationary_finite_distill"
    portfolio_id = _portfolio_id(user_id=result.job.user_id, seed=config.seed)
    checkpoint_path = result.family_state.checkpoint_by_user_id[result.job.user_id]
    artifact_paths: list[Path] = []
    child_summaries: list[dict[str, Any]] = []
    for child in result.selected_children:
        child_dir = output_dir / "policies" / f"policy_{child.portfolio_index}"
        child_dir.mkdir(parents=True, exist_ok=True)
        policy_path = child_dir / "policy.json"
        _write_json(
            policy_path,
            {
                "policy_type": POLICY_TYPE,
                "feature_version": FEATURE_VERSION,
                "title": (
                    f"{scheduler_name}_u{result.job.user_id}_"
                    f"w_{child.candidate.goal_cost_weight:.6g}"
                ),
                "checkpoint_path": _relative_path_string(
                    checkpoint_path,
                    base=child_dir,
                ),
                "goal_cost_weight": child.candidate.goal_cost_weight,
                "goal_norm_max": family_context.goal_norm_max,
                "action_retentions": list(family_context.action_retentions),
                "obs_mode": "oracle_stationary",
                "user_id": result.job.user_id,
                "portfolio_index": child.portfolio_index,
            },
        )
        # Validate the wrapper while the artifact is still local to the trainer.
        FSRS6OracleStationaryFiniteDistillPolicy.from_json(policy_path)
        metrics_path = child_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "candidate_id": child.candidate.candidate_id,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "pareto_rank": child.pareto_rank,
                "hypervolume_contribution": child.hypervolume_contribution,
                "metrics": asdict(child.candidate.metrics),
                "goal_cost_weight": child.candidate.goal_cost_weight,
            },
        )
        metadata_path = child_dir / "metadata.json"
        metadata_dir = metadata_path.parent
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "scheduler-policy",
                "artifact_id": f"{portfolio_id}-policy-{child.portfolio_index}",
                "family": config.family,
                "scheduler_name": scheduler_name,
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "review_markov_transition": (
                    config.simulation.review_markov_transition
                ),
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": FEATURE_VERSION,
                "action_space": PORTFOLIO_CHILD_ACTION_SPACE,
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "baseline_desired_retention": None,
                "goal_cost_weight": child.candidate.goal_cost_weight,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "hypervolume_contribution": child.hypervolume_contribution,
                "training_objective": "hypervolume",
                "config_snapshot_path": _relative_path_string(
                    config_path,
                    base=metadata_dir,
                ),
                "training_command_path": _relative_path_string(
                    result.job.command_record_path,
                    base=metadata_dir,
                )
                if result.job.command_record_path
                else None,
                "metrics_path": "metrics.json",
                "capabilities": ["event", "batched"],
            },
        )
        artifact_paths.append(metadata_path)
        child_summaries.append(
            {
                "portfolio_index": child.portfolio_index,
                "candidate_id": child.candidate.candidate_id,
                "policy_path": _relative_path_string(policy_path, base=output_dir),
                "metadata_path": _relative_path_string(metadata_path, base=output_dir),
                "metrics_path": _relative_path_string(metrics_path, base=output_dir),
                "goal_cost_weight": child.candidate.goal_cost_weight,
                "hypervolume_contribution": child.hypervolume_contribution,
                "pareto_rank": child.pareto_rank,
                "metrics": asdict(child.candidate.metrics),
            }
        )

    _write_json(
        output_dir / "portfolio.json",
        {
            "schema_version": SCHEMA_VERSION,
            "portfolio_id": portfolio_id,
            "scheduler_name": scheduler_name,
            "training_user_ids": [result.job.user_id],
            "baseline_desired_retention": None,
            "algorithm": portfolio.algorithm,
            "training_objective": "hypervolume",
            "passed": result.passed,
            "child_count": len(child_summaries),
            "children": child_summaries,
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
    )
    _write_json(
        output_dir / "portfolio_metrics.json",
        {
            "passed": result.passed,
            "hv_epsilon": portfolio.hv_epsilon,
            "baseline_hypervolume": result.baseline_hypervolume,
            "portfolio_hypervolume": result.portfolio_hypervolume,
            "hypervolume_improvement": result.hypervolume_improvement,
            "final_population_hypervolume": result.final_population_hypervolume,
            "final_population_hypervolume_improvement": (
                result.final_population_hypervolume_improvement
            ),
            "reference_point": asdict(result.reference_point),
            "baseline_desired_retention_values": list(
                result.baseline_desired_retention_values
            ),
            "baseline_metrics": [
                {
                    "baseline_desired_retention": dr,
                    **asdict(metrics),
                }
                for dr, metrics in zip(
                    result.baseline_desired_retention_values,
                    result.baseline_metrics,
                    strict=True,
                )
            ],
            "selected_child_count": len(result.selected_children),
            "final_population_size": len(result.final_population),
            "selection_algorithm": "greedy_subset_hypervolume",
            "settings": {
                "feature_version": FEATURE_VERSION,
                "distill_cost_weights": list(family_context.distill_cost_weights),
                "action_retentions": list(family_context.action_retentions),
                "goal_norm_max": family_context.goal_norm_max,
            },
            "portfolio_settings": asdict(portfolio),
            "distill_summary_path": _relative_path_string(
                result.family_state.distill_summary_path,
                base=output_dir,
            ),
            "history": result.history,
        },
    )
    return artifact_paths


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[OracleDistillPortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedOracleDistillPortfolioChild]:
    return _common_select_portfolio_children(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
        selected_child_from_candidate=_selected_child_from_candidate,
    )


def _portfolio_id(*, user_id: int, seed: int) -> str:
    return (
        f"fsrs6-oracle-stationary-finite-distill-portfolio-user-{user_id}-seed-{seed}"
    )


def _clear_cuda_cache(device: torch.device) -> None:
    _common_clear_cuda_cache(device)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    return _common_float_tuple(value, field_name)


def _optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    return _common_optional_float_tuple(value, field_name)


def _selection_executor(job_count: int) -> LightweightSelectionPool | None:
    return _common_selection_executor(
        job_count,
        enabled_env_vars=_SELECTION_ENV_VARS,
        worker_env_vars=_SELECTION_WORKER_ENV_VARS,
        default_min_jobs=DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
        default_workers=DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    )


def _select_survivors_for_generation(
    *,
    tasks: Sequence[SelectionTask[OracleDistillPortfolioCandidate]],
    executor: LightweightSelectionPool | None,
) -> tuple[list[list[OracleDistillPortfolioCandidate]], list[float]]:
    return _common_select_survivors_for_generation(tasks=tasks, executor=executor)


def _build_outcome(
    *,
    job: OracleDistillPortfolioTrainJob,
    passed: bool,
    artifact_paths: tuple[Path, ...],
    progress_path: Path,
    error: str | None,
) -> OracleDistillPortfolioTrainOutcome:
    return OracleDistillPortfolioTrainOutcome(
        job=job,
        passed=passed,
        artifact_paths=artifact_paths,
        progress_path=progress_path,
        error=error,
    )


clear_cuda_cache = _clear_cuda_cache
_progress_for_jobs = progress_for_jobs


_ADAPTER = PortfolioFamilyAdapter(
    settings_from_mapping=OracleDistillPortfolioSettings.from_mapping,
    build_family_context=_build_family_context,
    progress_payload=_progress_payload,
    initial_populations=_initial_populations,
    prepare_family_state=_prepare_family_state,
    evaluate_candidates=_evaluate_candidates,
    make_offspring=_make_offspring,
    selected_child_from_candidate=_selected_child_from_candidate,
    build_result=_build_result,
    write_artifacts=_write_portfolio_artifacts,
    build_outcome=_build_outcome,
    selection_enabled_env_vars=_SELECTION_ENV_VARS,
    selection_worker_env_vars=_SELECTION_WORKER_ENV_VARS,
    selection_default_min_jobs=DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    selection_default_workers=DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
)


if __name__ == "__main__":
    raise SystemExit(main())
