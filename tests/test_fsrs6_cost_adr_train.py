from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.cli.fsrs6_cost_adr_train import (  # noqa: E402
    _continuous_distill_teacher_table,
    fit_policy_from_table,
)
from experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr import (  # noqa: E402
    COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1,
    CoefficientPreconditioningSettings,
    CostADRTrainJob,
    CoverageObjectiveSettings,
    FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS,
    FIRST8_DISTILL24_SAMPLE_STD_V1_COEFFICIENTS,
    HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
    HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
    INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1,
    INITIAL_MEAN_SOURCE_RETENTION_BASELINE_COST_DECAY_V1,
    CostADRActionSettings,
    InitialPolicySettings,
    hypervolume_delta_mode_from_mapping,
    optimizer_settings_from_mapping,
    run_training_jobs,
    _coefficient_search_transform,
    _optimizer_settings_for_search_transform,
    _score_candidate,
)
from experiments.rl_scheduler.policy_search_common import (  # noqa: E402
    CandidateMetrics,
    PolicySearchSettings,
)
from experiments.rl_scheduler.portfolio_selection import (  # noqa: E402
    objective_hypervolume_2d,
    point_from_metrics,
    reference_point,
)
from experiments.single_card_tradeoff.models.policy_runtime import (  # noqa: E402
    RetentionDistillNet,
)
from simulator.experiment_infra.schemas import ExperimentConfig  # noqa: E402
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS, resolve_fsrs6_weights  # noqa: E402
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FEATURE_VERSION_INTERVAL_MONO,
    FSRS6CostConditionedADRPolicy,
    STATE_FEATURE_COUNT_COMPACT,
)


class FSRS6CostADRTrainTests(unittest.TestCase):
    def test_action_head_setting_accepts_retention_alias(self) -> None:
        default = CostADRActionSettings.from_mapping({})
        interval = CostADRActionSettings.from_mapping({"action_head": "interval"})
        retention = CostADRActionSettings.from_mapping({"action_head": "retention"})

        self.assertEqual(default.action_head, ACTION_HEAD_RETENTION)
        self.assertEqual(
            default.feature_version,
            "fsrs6_cost_adr_retention_mono_v1",
        )
        self.assertEqual(interval.action_head, ACTION_HEAD_INTERVAL)
        self.assertEqual(interval.feature_version, "fsrs6_cost_adr_interval_mono_v1")
        self.assertEqual(retention.action_head, ACTION_HEAD_RETENTION)
        self.assertEqual(
            retention.feature_version,
            "fsrs6_cost_adr_retention_mono_v1",
        )

    def test_builtin_mean_source_is_exclusive_with_policy_sources(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only one Cost-ADR initial policy"):
            InitialPolicySettings.from_mapping(
                {
                    "initial_mean_source": INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1,
                    "initial_policy": "policy.json",
                }
            )

    def test_optimizer_bounds_use_policy_search_coefficient_bounds(self) -> None:
        settings = PolicySearchSettings.from_mapping(
            {
                "coefficient_min": -64.0,
                "coefficient_max": 64.0,
                "retention_min": 0.5,
                "retention_max": 0.98,
                "baseline_desired_retention": 0.9,
            }
        )

        optimizer = optimizer_settings_from_mapping(
            {"name": "cma_es", "population_size": 2, "generations": 1},
            settings=settings,
        )

        self.assertEqual(optimizer.bounds[0], (-64.0,) * 24)
        self.assertEqual(optimizer.bounds[1], (64.0,) * 24)

    def test_first8_std_preconditioning_maps_z_space_to_coefficients(self) -> None:
        settings = PolicySearchSettings.from_mapping(
            {
                "coefficient_min": -64.0,
                "coefficient_max": 64.0,
                "retention_min": 0.5,
                "retention_max": 0.98,
                "baseline_desired_retention": 0.9,
            }
        )
        optimizer = optimizer_settings_from_mapping(
            {
                "name": "cma_es",
                "population_size": 2,
                "generations": 1,
                "sigma0": 1.0,
                "initial_mean": list(FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS),
            },
            settings=settings,
        )
        preconditioning = CoefficientPreconditioningSettings.from_mapping(
            {
                "coefficient_preconditioning": (
                    COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1
                ),
                "coefficient_scale_floor": 0.5,
                "coefficient_scale_multiplier": 1.0,
            }
        )

        transform = _coefficient_search_transform(
            optimizer_settings=optimizer,
            preconditioning_settings=preconditioning,
        )
        search_optimizer = _optimizer_settings_for_search_transform(
            optimizer_settings=optimizer,
            transform=transform,
        )

        expected_scale = tuple(
            max(0.5, value) for value in FIRST8_DISTILL24_SAMPLE_STD_V1_COEFFICIENTS
        )
        self.assertEqual(
            transform.mode,
            COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1,
        )
        self.assertEqual(transform.origin, FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS)
        self.assertEqual(transform.scale, expected_scale)
        self.assertEqual(transform.search_initial_mean, (0.0,) * 24)
        self.assertEqual(search_optimizer.initial_mean, (0.0,) * 24)
        self.assertNotEqual(search_optimizer.bounds, optimizer.bounds)
        self.assertAlmostEqual(
            search_optimizer.bounds[0][0],
            (-64.0 - FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS[0]) / expected_scale[0],
        )
        self.assertAlmostEqual(
            search_optimizer.bounds[1][13],
            (64.0 - FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS[13]) / expected_scale[13],
        )
        mapped = transform.coefficients_from_search([1.0] * 24)
        for index, (mean, scale) in enumerate(
            zip(
                FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS,
                expected_scale,
                strict=True,
            )
        ):
            self.assertAlmostEqual(mapped[index], mean + scale)
        self.assertEqual(transform.coefficients_from_search([999.0] * 24), (64.0,) * 24)

    def test_hypervolume_delta_mode_defaults_to_union_contribution(self) -> None:
        self.assertEqual(
            hypervolume_delta_mode_from_mapping({}),
            HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
        )
        self.assertEqual(
            hypervolume_delta_mode_from_mapping(
                {"hypervolume_delta_mode": HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE}
            ),
            HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
        )
        with self.assertRaisesRegex(ValueError, "hypervolume_delta_mode"):
            hypervolume_delta_mode_from_mapping({"hypervolume_delta_mode": "bad"})

    def test_default_union_contribution_scores_external_hypervolume_gain(self) -> None:
        baseline = [
            CandidateMetrics(
                memorized_average=100.0,
                time_average=10.0,
                memorized_per_minute=10.0,
                total_reviews=1,
                total_lapses=0,
                total_cost=600.0,
            ),
            CandidateMetrics(
                memorized_average=200.0,
                time_average=20.0,
                memorized_per_minute=10.0,
                total_reviews=2,
                total_lapses=0,
                total_cost=1200.0,
            ),
        ]
        candidate = [
            CandidateMetrics(
                memorized_average=210.0,
                time_average=19.0,
                memorized_per_minute=11.0,
                total_reviews=2,
                total_lapses=0,
                total_cost=1140.0,
            )
        ]
        baseline_points = [point_from_metrics(metric) for metric in baseline]
        reference = reference_point(baseline_points)
        baseline_hypervolume = objective_hypervolume_2d(
            baseline_points,
            reference=reference,
        )

        score = _score_candidate(
            baseline_metrics=baseline,
            baseline_points=baseline_points,
            baseline_hypervolume=baseline_hypervolume,
            reference=reference,
            candidate_metrics=candidate,
            coverage_settings=CoverageObjectiveSettings(),
        )

        self.assertAlmostEqual(
            score.hypervolume_delta,
            objective_hypervolume_2d(
                [
                    *baseline_points,
                    *[point_from_metrics(metric) for metric in candidate],
                ],
                reference=reference,
            )
            - baseline_hypervolume,
        )
        self.assertGreater(score.hypervolume_delta, 0.0)

    def test_coverage_objective_penalizes_narrow_frontier_overlap(self) -> None:
        baseline = [
            CandidateMetrics(
                memorized_average=100.0,
                time_average=10.0,
                memorized_per_minute=10.0,
                total_reviews=1,
                total_lapses=0,
                total_cost=600.0,
            ),
            CandidateMetrics(
                memorized_average=200.0,
                time_average=20.0,
                memorized_per_minute=10.0,
                total_reviews=2,
                total_lapses=0,
                total_cost=1200.0,
            ),
        ]
        candidate = [
            CandidateMetrics(
                memorized_average=190.0,
                time_average=19.0,
                memorized_per_minute=10.0,
                total_reviews=2,
                total_lapses=0,
                total_cost=1140.0,
            )
        ]
        baseline_objective_points = [point_from_metrics(metric) for metric in baseline]
        reference = reference_point(baseline_objective_points)

        score = _score_candidate(
            baseline_metrics=baseline,
            baseline_points=baseline_objective_points,
            baseline_hypervolume=1000.0,
            reference=reference,
            candidate_metrics=candidate,
            coverage_settings=CoverageObjectiveSettings(
                enabled=True,
                min_budget_span_coverage=0.9,
                min_target_span_coverage=0.9,
                penalty_weight=0.1,
            ),
            hypervolume_delta_mode=HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
        )

        self.assertGreater(score.coverage_penalty, 0.0)
        self.assertAlmostEqual(
            score.hypervolume_delta,
            objective_hypervolume_2d(
                [point_from_metrics(metric) for metric in candidate],
                reference=reference,
            )
            - 1000.0,
        )
        self.assertLess(score.objective_score, score.hypervolume_delta)
        self.assertEqual(score.coverage_diagnostics.covered_budget_count, 0)

    def test_quality_aware_coverage_ignores_baseline_dominated_candidates(
        self,
    ) -> None:
        baseline = [
            CandidateMetrics(
                memorized_average=100.0,
                time_average=10.0,
                memorized_per_minute=10.0,
                total_reviews=1,
                total_lapses=0,
                total_cost=600.0,
            ),
            CandidateMetrics(
                memorized_average=200.0,
                time_average=20.0,
                memorized_per_minute=10.0,
                total_reviews=2,
                total_lapses=0,
                total_cost=1200.0,
            ),
        ]
        candidate = [
            CandidateMetrics(
                memorized_average=90.0,
                time_average=10.0,
                memorized_per_minute=9.0,
                total_reviews=1,
                total_lapses=0,
                total_cost=600.0,
            ),
            CandidateMetrics(
                memorized_average=190.0,
                time_average=20.0,
                memorized_per_minute=9.5,
                total_reviews=2,
                total_lapses=0,
                total_cost=1200.0,
            ),
        ]
        baseline_objective_points = [point_from_metrics(metric) for metric in baseline]
        reference = reference_point(baseline_objective_points)

        unfiltered = _score_candidate(
            baseline_metrics=baseline,
            baseline_points=baseline_objective_points,
            baseline_hypervolume=1000.0,
            reference=reference,
            candidate_metrics=candidate,
            coverage_settings=CoverageObjectiveSettings(
                enabled=True,
                min_budget_span_coverage=0.9,
                min_target_span_coverage=0.9,
                penalty_weight=0.1,
            ),
        )
        filtered = _score_candidate(
            baseline_metrics=baseline,
            baseline_points=baseline_objective_points,
            baseline_hypervolume=1000.0,
            reference=reference,
            candidate_metrics=candidate,
            coverage_settings=CoverageObjectiveSettings(
                enabled=True,
                min_budget_span_coverage=0.9,
                min_target_span_coverage=0.9,
                penalty_weight=0.1,
                filter_baseline_dominated=True,
            ),
        )

        self.assertEqual(unfiltered.coverage_penalty, 0.0)
        self.assertGreater(filtered.coverage_penalty, 0.0)
        self.assertEqual(filtered.coverage_diagnostics.candidate_count, 2)
        self.assertEqual(
            filtered.coverage_diagnostics.baseline_dominated_candidate_count,
            2,
        )
        self.assertEqual(filtered.coverage_diagnostics.coverage_candidate_count, 0)

    def test_fits_from_continuous_distill_teacher_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "teacher.pt"
            model = RetentionDistillNet(
                obs_dim=3,
                hidden_size=4,
                action_count=2,
                architecture="residual",
                depth=1,
            )
            torch.save(
                {
                    "policy_type": "fsrs6_oracle_continuous_stationary_finite_distill",
                    "action_mode": "desired_retention",
                    "obs_mode": "oracle_stationary",
                    "obs_dim": 3,
                    "hidden_size": 4,
                    "network": "residual",
                    "network_depth": 1,
                    "action_retentions": [0.5, 0.98],
                    "cost_weights": [0.0, 4.0],
                    "retention_min": 0.5,
                    "retention_max": 0.98,
                    "fsrs_weights": list(resolve_fsrs6_weights(None)),
                    "model_state_dict": model.state_dict(),
                },
                policy_path,
            )

            table = _continuous_distill_teacher_table(
                policy_path=policy_path,
                cost_weights=[0.0, 4.0],
                s_grid_size=8,
                d_grid_size=8,
                device=torch.device("cpu"),
            )
            policy, stats = fit_policy_from_table(
                table=table,
                cost_weights=[0.0, 4.0],
                action_head=ACTION_HEAD_INTERVAL,
                state_feature_count=STATE_FEATURE_COUNT_COMPACT,
                epochs=2,
                learning_rate=0.01,
                weight_decay=0.0,
                max_grad_norm=10.0,
            )

        self.assertEqual(policy.parameter_count, 24)
        self.assertEqual(policy.action_head, ACTION_HEAD_INTERVAL)
        self.assertTrue(stats["final_loss"] >= 0.0)

    def test_builtin_mean_initializer_uses_64_bounds_and_generation_zero(self) -> None:
        bundle_lane_user_ids = []
        training_coefficients = []

        def fake_build_bundle(**kwargs):
            lane_user_ids = kwargs.get("lane_user_ids")
            if not isinstance(lane_user_ids, list):
                raise AssertionError("trainer smoke test expects lane_user_ids.")
            bundle_lane_user_ids.append(list(lane_user_ids))
            lanes = len(lane_user_ids)
            device = kwargs["device"]
            return SimpleNamespace(
                env_ops=SimpleNamespace(device=device),
                scheduler_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS for _ in range(lanes)],
                    device=device,
                    dtype=torch.float32,
                ),
                behavior=object(),
                cost_model=object(),
                device=device,
                short_term_source=None,
                learning_steps=[],
                relearning_steps=[],
            )

        def fake_simulate_multiuser(**kwargs):
            sched_ops = kwargs["sched_ops"]
            lane_count = int(sched_ops._weights.shape[0])
            is_cost_adr = hasattr(sched_ops, "_goal_cost_weight")
            if is_cost_adr:
                training_coefficients.append(sched_ops._coefficients.detach().cpu())
            stats = []
            for index in range(lane_count):
                if is_cost_adr:
                    memorized = 8.0 + index * 0.01
                    minutes = 1.0
                else:
                    memorized = 5.0 + index * 0.01
                    minutes = 2.0
                stats.append(
                    SimpleNamespace(
                        daily_cost=[minutes * 60.0],
                        daily_memorized=[memorized],
                        total_reviews=1,
                        total_lapses=0,
                        total_cost=minutes * 60.0,
                    )
                )
            return stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cost_adr_meaninit.toml"
            output_dir = root / "user_1"
            config_path.write_text(
                f"""
schema_version = 1
name = "cost-adr-meaninit-smoke"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/cost-adr-meaninit-smoke"
stages = ["train-overfit"]

[users]
train = [1]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "logs/retention_sweep"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.90]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[training]
artifact_metadata_glob = "metadata.json"
command_template = [
  "uv",
  "run",
  "python",
  "experiments/rl_scheduler/train_cmaes_fsrs6_cost_adr.py",
]

[training.policy_search]
action_head = "interval"
coefficient_min = -64.0
coefficient_max = 64.0
retention_min = 0.50
retention_max = 0.98
baseline_desired_retention = 0.90
torch_device = "cpu"
short_term_threshold = 0.5
short_term_loops_limit = 1
initial_mean_source = "{INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1}"
initial_policy_expand_bounds = false
initial_policy_evaluate_in_generation_zero = true

[training.optimizer]
name = "cma_es"
population_size = 2
generations = 1
sigma0 = 1.0
seed = 7
""".lstrip(),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_toml(config_path)

            with (
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr._build_bundle",
                    side_effect=fake_build_bundle,
                ),
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr.simulate_multiuser",
                    side_effect=fake_simulate_multiuser,
                ),
            ):
                results = run_training_jobs(
                    jobs=[CostADRTrainJob(user_id=1, output_dir=output_dir)],
                    config=config,
                    config_path=config_path,
                    repo_root=REPO_ROOT,
                    button_usage=None,
                )

            progress_records = [
                json.loads(line)
                for line in (output_dir / "training_progress.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual([result.passed for result in results], [True])
        self.assertEqual(bundle_lane_user_ids[0], [1])
        self.assertEqual(bundle_lane_user_ids[1], [1] * 32)
        self.assertEqual(len(training_coefficients), 1)
        expected = torch.tensor(
            FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(training_coefficients[0][0:16], expected.expand(16, -1))
        )
        config_loaded = next(
            record for record in progress_records if record["event"] == "config_loaded"
        )
        self.assertEqual(
            config_loaded["initial_policy"]["source"],
            INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1,
        )
        self.assertEqual(
            config_loaded["hypervolume_delta_mode"],
            HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
        )
        self.assertIsNone(config_loaded["initial_policy"]["path"])
        self.assertEqual(config_loaded["optimizer"]["bounds"][0], [-64.0] * 24)
        self.assertEqual(config_loaded["optimizer"]["bounds"][1], [64.0] * 24)
        generation = next(
            record
            for record in progress_records
            if record["event"] == "cmaes_generation"
        )
        self.assertEqual(generation["effective_lanes"], 32)
        self.assertEqual(
            metrics["initial_policy"]["source"],
            INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1,
        )
        self.assertEqual(metrics["optimizer"]["bounds"][0], [-64.0] * 24)
        self.assertEqual(metrics["optimizer"]["bounds"][1], [64.0] * 24)
        self.assertEqual(len(metrics["selected_cost_weight_rollout_points"]), 16)

    def test_builtin_mean_std_preconditioning_uses_scheduler_hv_gate(self) -> None:
        bundle_lane_user_ids = []
        training_coefficients = []

        def fake_build_bundle(**kwargs):
            lane_user_ids = kwargs.get("lane_user_ids")
            if not isinstance(lane_user_ids, list):
                raise AssertionError("trainer smoke test expects lane_user_ids.")
            bundle_lane_user_ids.append(list(lane_user_ids))
            lanes = len(lane_user_ids)
            device = kwargs["device"]
            return SimpleNamespace(
                env_ops=SimpleNamespace(device=device),
                scheduler_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS for _ in range(lanes)],
                    device=device,
                    dtype=torch.float32,
                ),
                behavior=object(),
                cost_model=object(),
                device=device,
                short_term_source=None,
                learning_steps=[],
                relearning_steps=[],
            )

        def fake_simulate_multiuser(**kwargs):
            sched_ops = kwargs["sched_ops"]
            lane_count = int(sched_ops._weights.shape[0])
            is_cost_adr = hasattr(sched_ops, "_goal_cost_weight")
            if is_cost_adr:
                training_coefficients.append(sched_ops._coefficients.detach().cpu())
            stats = []
            for index in range(lane_count):
                if is_cost_adr:
                    memorized = 8.0 + index * 0.01
                    minutes = 1.0
                else:
                    memorized = 5.0 + index * 0.01
                    minutes = 2.0
                stats.append(
                    SimpleNamespace(
                        daily_cost=[minutes * 60.0],
                        daily_memorized=[memorized],
                        total_reviews=1,
                        total_lapses=0,
                        total_cost=minutes * 60.0,
                    )
                )
            return stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cost_adr_stdpre.toml"
            output_dir = root / "user_1"
            config_path.write_text(
                f"""
schema_version = 1
name = "cost-adr-stdpre-smoke"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/cost-adr-stdpre-smoke"
stages = ["train-overfit"]

[users]
train = [1]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "logs/retention_sweep"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.90]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[training]
artifact_metadata_glob = "metadata.json"
command_template = [
  "uv",
  "run",
  "python",
  "experiments/rl_scheduler/train_cmaes_fsrs6_cost_adr.py",
]

[training.policy_search]
action_head = "interval"
coefficient_min = -64.0
coefficient_max = 64.0
retention_min = 0.50
retention_max = 0.98
baseline_desired_retention = 0.90
torch_device = "cpu"
short_term_threshold = 0.5
short_term_loops_limit = 1
initial_mean_source = "{INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1}"
initial_policy_expand_bounds = false
initial_policy_evaluate_in_generation_zero = true
hypervolume_delta_mode = "{HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE}"
coefficient_preconditioning = "{COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1}"
coefficient_scale_floor = 0.5
coefficient_scale_multiplier = 1.0

[training.optimizer]
name = "cma_es"
population_size = 2
generations = 1
sigma0 = 1.0
seed = 7
""".lstrip(),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_toml(config_path)

            with (
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr._build_bundle",
                    side_effect=fake_build_bundle,
                ),
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr.simulate_multiuser",
                    side_effect=fake_simulate_multiuser,
                ),
            ):
                results = run_training_jobs(
                    jobs=[CostADRTrainJob(user_id=1, output_dir=output_dir)],
                    config=config,
                    config_path=config_path,
                    repo_root=REPO_ROOT,
                    button_usage=None,
                )

            progress_records = [
                json.loads(line)
                for line in (output_dir / "training_progress.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual([result.passed for result in results], [True])
        self.assertEqual(bundle_lane_user_ids[0], [1])
        self.assertEqual(bundle_lane_user_ids[1], [1] * 32)
        expected = torch.tensor(
            FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS,
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(training_coefficients[0][0:16], expected.expand(16, -1))
        )
        config_loaded = next(
            record for record in progress_records if record["event"] == "config_loaded"
        )
        self.assertEqual(
            config_loaded["hypervolume_delta_mode"],
            HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
        )
        self.assertEqual(config_loaded["optimizer"]["bounds"][0], [-64.0] * 24)
        self.assertEqual(config_loaded["optimizer"]["bounds"][1], [64.0] * 24)
        self.assertEqual(config_loaded["search_optimizer"]["initial_mean"], [0.0] * 24)
        self.assertNotEqual(
            config_loaded["search_optimizer"]["bounds"],
            config_loaded["optimizer"]["bounds"],
        )
        self.assertEqual(
            config_loaded["coefficient_preconditioning"]["mode"],
            COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1,
        )
        self.assertTrue(config_loaded["coefficient_preconditioning"]["enabled"])
        self.assertEqual(
            metrics["hypervolume_delta_mode"],
            HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
        )
        self.assertEqual(
            metrics["coefficient_preconditioning"]["mode"],
            COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1,
        )
        self.assertEqual(metrics["search_optimizer"]["initial_mean"], [0.0] * 24)
        self.assertEqual(
            metadata["coefficient_preconditioning"]["mode"],
            COEFFICIENT_PRECONDITIONING_FIRST8_DISTILL24_STD_V1,
        )
        self.assertEqual(
            metadata["search_optimizer_settings"]["initial_mean"],
            [0.0] * 24,
        )

    def test_retention_head_trainer_writes_retention_artifacts(self) -> None:
        bundle_lane_user_ids = []
        training_action_heads = []

        def fake_build_bundle(**kwargs):
            lane_user_ids = kwargs.get("lane_user_ids")
            if not isinstance(lane_user_ids, list):
                raise AssertionError("trainer smoke test expects lane_user_ids.")
            bundle_lane_user_ids.append(list(lane_user_ids))
            lanes = len(lane_user_ids)
            device = kwargs["device"]
            return SimpleNamespace(
                env_ops=SimpleNamespace(device=device),
                scheduler_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS for _ in range(lanes)],
                    device=device,
                    dtype=torch.float32,
                ),
                behavior=object(),
                cost_model=object(),
                device=device,
                short_term_source=None,
                learning_steps=[],
                relearning_steps=[],
            )

        def fake_simulate_multiuser(**kwargs):
            sched_ops = kwargs["sched_ops"]
            lane_count = int(sched_ops._weights.shape[0])
            is_cost_adr = hasattr(sched_ops, "_goal_cost_weight")
            if is_cost_adr:
                training_action_heads.append(sched_ops._action_head)
            stats = []
            for index in range(lane_count):
                if is_cost_adr:
                    memorized = 8.0 + index * 0.01
                    minutes = 1.0
                else:
                    memorized = 5.0 + index * 0.01
                    minutes = 2.0
                stats.append(
                    SimpleNamespace(
                        daily_cost=[minutes * 60.0],
                        daily_memorized=[memorized],
                        total_reviews=1,
                        total_lapses=0,
                        total_cost=minutes * 60.0,
                    )
                )
            return stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cost_adr_retention.toml"
            output_dir = root / "user_1"
            config_path.write_text(
                f"""
schema_version = 1
name = "cost-adr-retention-smoke"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/cost-adr-retention-smoke"
stages = ["train-overfit"]

[users]
train = [1]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "logs/retention_sweep"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.90]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[training]
artifact_metadata_glob = "metadata.json"
command_template = [
  "uv",
  "run",
  "python",
  "experiments/rl_scheduler/train_cmaes_fsrs6_cost_adr.py",
]

[training.policy_search]
action_head = "desired_retention"
coefficient_min = -64.0
coefficient_max = 64.0
retention_min = 0.50
retention_max = 0.98
baseline_desired_retention = 0.90
torch_device = "cpu"
short_term_threshold = 0.5
short_term_loops_limit = 1
initial_mean_source = "{INITIAL_MEAN_SOURCE_RETENTION_BASELINE_COST_DECAY_V1}"
initial_policy_expand_bounds = false
initial_policy_evaluate_in_generation_zero = true

[training.optimizer]
name = "cma_es"
population_size = 2
generations = 1
sigma0 = 1.0
seed = 7
""".lstrip(),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_toml(config_path)

            with (
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr._build_bundle",
                    side_effect=fake_build_bundle,
                ),
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr.simulate_multiuser",
                    side_effect=fake_simulate_multiuser,
                ),
            ):
                results = run_training_jobs(
                    jobs=[CostADRTrainJob(user_id=1, output_dir=output_dir)],
                    config=config,
                    config_path=config_path,
                    repo_root=REPO_ROOT,
                    button_usage=None,
                )

            policy = FSRS6CostConditionedADRPolicy.from_json(output_dir / "policy.json")
            metadata = json.loads(
                (output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            progress_records = [
                json.loads(line)
                for line in (output_dir / "training_progress.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual([result.passed for result in results], [True])
        self.assertEqual(bundle_lane_user_ids[0], [1])
        self.assertEqual(bundle_lane_user_ids[1], [1] * 32)
        self.assertEqual(training_action_heads, [ACTION_HEAD_RETENTION])
        self.assertEqual(policy.action_head, ACTION_HEAD_RETENTION)
        self.assertEqual(policy.feature_version, "fsrs6_cost_adr_retention_mono_v1")
        self.assertIsNone(policy.max_interval_days)
        self.assertEqual(metadata["action_space"], "sd_cost_retention_function")
        self.assertEqual(metadata["action_head"], ACTION_HEAD_RETENTION)
        self.assertEqual(metrics["action_head"], ACTION_HEAD_RETENTION)
        self.assertEqual(
            metrics["initial_policy"]["source"],
            INITIAL_MEAN_SOURCE_RETENTION_BASELINE_COST_DECAY_V1,
        )
        config_loaded = next(
            record for record in progress_records if record["event"] == "config_loaded"
        )
        self.assertEqual(config_loaded["action_head"], ACTION_HEAD_RETENTION)
        self.assertEqual(
            config_loaded["feature_version"],
            "fsrs6_cost_adr_retention_mono_v1",
        )

    def test_cmaes_trainer_batches_users_and_writes_artifacts(self) -> None:
        bundle_lane_user_ids = []
        simulate_calls = []
        training_coefficients = []

        def fake_build_bundle(**kwargs):
            lane_user_ids = kwargs.get("lane_user_ids")
            if not isinstance(lane_user_ids, list):
                raise AssertionError("trainer smoke test expects lane_user_ids.")
            bundle_lane_user_ids.append(list(lane_user_ids))
            lanes = len(lane_user_ids)
            device = kwargs["device"]
            return SimpleNamespace(
                env_ops=SimpleNamespace(device=device),
                scheduler_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS for _ in range(lanes)],
                    device=device,
                    dtype=torch.float32,
                ),
                behavior=object(),
                cost_model=object(),
                device=device,
                short_term_source=None,
                learning_steps=[],
                relearning_steps=[],
            )

        def fake_simulate_multiuser(**kwargs):
            sched_ops = kwargs["sched_ops"]
            lane_count = int(sched_ops._weights.shape[0])
            is_cost_adr = hasattr(sched_ops, "_goal_cost_weight")
            if is_cost_adr:
                training_coefficients.append(sched_ops._coefficients.detach().cpu())
            simulate_calls.append(
                {
                    "is_cost_adr": is_cost_adr,
                    "lane_count": lane_count,
                }
            )
            stats = []
            for index in range(lane_count):
                if is_cost_adr:
                    memorized = 8.0 + index * 0.01
                    minutes = 1.0
                else:
                    memorized = 5.0 + index * 0.01
                    minutes = 2.0
                stats.append(
                    SimpleNamespace(
                        daily_cost=[minutes * 60.0],
                        daily_memorized=[memorized],
                        total_reviews=1,
                        total_lapses=0,
                        total_cost=minutes * 60.0,
                    )
                )
            return stats

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "cost_adr.toml"
            output_dirs = [root / "user_1", root / "user_2"]
            initial_policy_root = root / "initial_policies"
            for user_id, coefficient in ((1, 20.0), (2, -20.0)):
                FSRS6CostConditionedADRPolicy(
                    coefficients=(coefficient,) + (0.0,) * 23,
                    action_head=ACTION_HEAD_INTERVAL,
                    feature_version=FEATURE_VERSION_INTERVAL_MONO,
                    cost_weight_min=0.0,
                    cost_weight_max=1024.0,
                    retention_min=0.50,
                    retention_max=0.98,
                    title=f"initial u{user_id}",
                ).write_json(initial_policy_root / f"user_{user_id}" / "policy.json")
            config_path.write_text(
                f"""
schema_version = 1
name = "cost-adr-smoke"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/cost-adr-smoke"
stages = ["train-overfit"]

[users]
train = [1, 2]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "logs/retention_sweep"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.90]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[training]
artifact_metadata_glob = "metadata.json"
command_template = [
  "uv",
  "run",
  "python",
  "experiments/rl_scheduler/train_cmaes_fsrs6_cost_adr.py",
]

[training.policy_search]
action_head = "interval"
coefficient_min = -12.0
coefficient_max = 12.0
retention_min = 0.50
retention_max = 0.98
baseline_desired_retention = 0.90
torch_device = "cpu"
short_term_threshold = 0.5
short_term_loops_limit = 1
initial_policy_root = "{initial_policy_root.as_posix()}"
initial_policy_bounds_padding = 2.0

[training.optimizer]
name = "cma_es"
population_size = 2
generations = 1
sigma0 = 1.0
seed = 7
""".lstrip(),
                encoding="utf-8",
            )
            config = ExperimentConfig.from_toml(config_path)

            with (
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr._build_bundle",
                    side_effect=fake_build_bundle,
                ),
                patch(
                    "experiments.rl_scheduler.train_cmaes_fsrs6_cost_adr.simulate_multiuser",
                    side_effect=fake_simulate_multiuser,
                ),
            ):
                results = run_training_jobs(
                    jobs=[
                        CostADRTrainJob(user_id=1, output_dir=output_dirs[0]),
                        CostADRTrainJob(user_id=2, output_dir=output_dirs[1]),
                    ],
                    config=config,
                    config_path=config_path,
                    repo_root=REPO_ROOT,
                    button_usage=None,
                )

            artifacts = []
            progress_records_by_user = []
            for output_dir in output_dirs:
                policy_path = output_dir / "policy.json"
                metadata_path = output_dir / "metadata.json"
                metrics_path = output_dir / "metrics.json"
                progress_path = output_dir / "training_progress.jsonl"
                artifacts.append(
                    (
                        FSRS6CostConditionedADRPolicy.from_json(policy_path),
                        metadata_path,
                        json.loads(metadata_path.read_text(encoding="utf-8")),
                        json.loads(metrics_path.read_text(encoding="utf-8")),
                    )
                )
                progress_records_by_user.append(
                    [
                        json.loads(line)
                        for line in progress_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                )

        self.assertEqual([result.passed for result in results], [True, True])
        self.assertEqual(
            [result.artifact_paths for result in results],
            [
                (output_dirs[0] / "metadata.json",),
                (output_dirs[1] / "metadata.json",),
            ],
        )
        self.assertEqual(bundle_lane_user_ids[0], [1, 2])
        self.assertEqual(bundle_lane_user_ids[1], [1] * 32 + [2] * 32)
        self.assertEqual(len(training_coefficients), 1)
        coefficients = training_coefficients[0]
        self.assertTrue(torch.all(coefficients[0:16, 0] == 20.0))
        self.assertTrue(torch.all(coefficients[32:48, 0] == -20.0))
        self.assertEqual(
            simulate_calls,
            [
                {"is_cost_adr": False, "lane_count": 2},
                {"is_cost_adr": True, "lane_count": 64},
            ],
        )
        for policy, _metadata_path, metadata, metrics in artifacts:
            self.assertEqual(policy.parameter_count, 24)
            self.assertEqual(policy.action_head, ACTION_HEAD_INTERVAL)
            self.assertEqual(metadata["scheduler_name"], "fsrs6_cost_adr")
            self.assertEqual(metadata["action_space"], "sd_cost_interval_function")
            self.assertIsNone(metadata["lambda_value"])
            self.assertIsNone(metadata["baseline_desired_retention"])
            self.assertGreater(metrics["best_hypervolume_delta"], 0.0)
            self.assertEqual(metrics["training_objective"], "hypervolume_delta")
            self.assertEqual(
                metrics["hypervolume_delta_mode"],
                HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
            )
            self.assertFalse(metrics["coverage_objective"]["enabled"])
            self.assertIn("best_coverage", metrics)
            self.assertIsNotNone(metrics["initial_policy"])
            self.assertEqual(len(metrics["selected_cost_weight_rollout_points"]), 16)
            self.assertEqual(metrics["optimizer"]["population_size"], 2)
            if metrics["initial_policy"]["coefficient_max"] > 0.0:
                self.assertGreaterEqual(metrics["optimizer"]["bounds"][1][0], 22.0)
            else:
                self.assertLessEqual(metrics["optimizer"]["bounds"][0][0], -22.0)

        for progress_records in progress_records_by_user:
            progress_events = [record["event"] for record in progress_records]
            generation = next(
                record
                for record in progress_records
                if record["event"] == "cmaes_generation"
            )
            self.assertIn("cmaes_generation", progress_events)
            self.assertEqual(generation["batched_user_count"], 2)
            self.assertEqual(generation["batched_user_ids"], [1, 2])
            self.assertEqual(generation["effective_lanes"], 64)


if __name__ == "__main__":
    unittest.main()
