from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_cmaes_fsrs6_ap import APSettings
from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
)
from experiments.rl_scheduler.train_fsrs6_ap_portfolio import (
    APPortfolioCandidate,
    APPortfolioSettings,
    APPortfolioTrainJob,
    ObjectivePoint,
    SelectedAPPortfolioChild,
    UserAPPortfolioResult,
    _mutate_genome,
    _selection_payload,
    _select_portfolio_children,
    _write_portfolio_artifacts,
)
from experiments.rl_scheduler.portfolio_selection import (
    SelectionTask,
    select_survivors_for_generation,
)
from simulator.experiment_infra import ExperimentConfig, validate_scheduler_artifact
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.fsrs6_ap_policy import clip_fsrs6_ap_weights


def _metrics(memorized: float, time_average: float) -> CandidateMetrics:
    return CandidateMetrics(
        memorized_average=memorized,
        time_average=time_average,
        memorized_per_minute=memorized / time_average,
        total_reviews=0,
        total_lapses=0,
        total_cost=0.0,
    )


def _ap_candidate(
    candidate_id: int,
    memorized: float,
    time_average: float,
) -> APPortfolioCandidate:
    base_weights = clip_fsrs6_ap_weights(DEFAULT_FSRS6_WEIGHTS)
    return APPortfolioCandidate(
        candidate_id=candidate_id,
        desired_retention=0.83,
        search_vector=(0.0,) * 21,
        weights=base_weights,
        metrics=_metrics(memorized, time_average),
    )


def _config(output_root: Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "ap-portfolio-test",
            "family": "rl_scheduler",
            "seed": 42,
            "output_root": str(output_root),
            "stages": ["train-overfit"],
            "users": {"train": [1], "validation": [], "reserved_test": []},
            "baseline": {
                "scheduler": "fsrs6",
                "log_root": str(output_root / "logs"),
                "expected_engine": "batched",
                "stage_mode": "copy",
            },
            "simulation": {
                "engine": "batched",
                "environment": "fsrs6",
                "days": 2,
                "deck": 10,
                "learn_limit": 1,
                "review_limit": 10,
                "cost_limit_minutes": 60.0,
                "priority": "new-first",
                "scheduler_priority": "low_retrievability",
                "fuzz": False,
            },
            "gpu_guard": {"required": False, "device": "cpu", "smoke": False},
            "performance": {"device": "cpu", "write_performance_summary": True},
            "training": {
                "policy_search": {
                    "retention_min": 0.5,
                    "retention_max": 0.98,
                    "baseline_desired_retention": 0.9,
                    "torch_device": "cpu",
                },
                "portfolio": {},
                "ap": {"weight_delta_scale": 0.5},
            },
        },
        config_path=output_root / "config.toml",
    )


class FSRS6APPortfolioTests(unittest.TestCase):
    def test_mutation_clips_desired_retention_and_keeps_21d_search_vector(
        self,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1)

        desired_retention, search_vector = _mutate_genome(
            desired_retention=0.9,
            search_vector=[0.0] * 21,
            mutation_scale=0.35,
            retention_mutation_scale=100.0,
            retention_min=0.6,
            retention_max=0.6,
            device=torch.device("cpu"),
            generator=generator,
        )

        self.assertEqual(desired_retention, 0.6)
        self.assertEqual(len(search_vector), 21)
        self.assertTrue(any(value != 0.0 for value in search_vector))

    def test_selection_helper_returns_timed_ap_survivors(self) -> None:
        base_weights = clip_fsrs6_ap_weights(DEFAULT_FSRS6_WEIGHTS)
        baseline_points = [ObjectivePoint(0.0, -10.0)]
        reference = ObjectivePoint(-1.0, -11.0)
        candidates = [
            APPortfolioCandidate(
                candidate_id=1,
                desired_retention=0.70,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(4.0, 4.0),
            ),
            APPortfolioCandidate(
                candidate_id=2,
                desired_retention=0.80,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(8.0, 8.0),
            ),
            APPortfolioCandidate(
                candidate_id=3,
                desired_retention=0.90,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(5.0, 2.0),
            ),
        ]

        survivors, worker_seconds = select_survivors_for_generation(
            tasks=[
                SelectionTask(
                    candidates=tuple(candidates),
                    payload=_selection_payload(
                        baseline_points=baseline_points,
                        candidates=candidates,
                        population_size=2,
                        reference=reference,
                    ),
                )
            ],
            executor=None,
        )

        self.assertEqual(len(survivors), 1)
        self.assertEqual(len(survivors[0]), 2)
        self.assertEqual(len(worker_seconds), 1)
        self.assertGreaterEqual(worker_seconds[0], 0.0)

    def test_portfolio_child_indexes_sort_by_study_time(self) -> None:
        baseline_points = [ObjectivePoint(0.1, -9.9)]
        reference = ObjectivePoint(0.0, -10.0)
        candidates = [
            _ap_candidate(1, 9.0, 4.0),
            _ap_candidate(2, 6.0, 2.0),
            _ap_candidate(3, 7.0, 2.0),
        ]

        children = _select_portfolio_children(
            baseline_points=baseline_points,
            candidates=candidates,
            portfolio_size=3,
            reference=reference,
        )

        self.assertEqual(
            [child.candidate.candidate_id for child in children], [3, 2, 1]
        )
        self.assertEqual([child.portfolio_index for child in children], [0, 1, 2])

    def test_artifacts_record_exported_subset_hypervolume_for_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            config_path = root / "config.toml"
            config_path.write_text("", encoding="utf-8")
            command_record_path = root / "commands" / "train.json"
            command_record_path.parent.mkdir()
            command_record_path.write_text("{}", encoding="utf-8")
            settings = PolicySearchSettings.from_mapping(config.training_policy_search)
            portfolio = APPortfolioSettings(portfolio_size=1)
            ap_settings = APSettings(dr_batch_size=1, weight_delta_scale=0.5)
            base_weights = clip_fsrs6_ap_weights(DEFAULT_FSRS6_WEIGHTS)
            candidate = APPortfolioCandidate(
                candidate_id=7,
                desired_retention=0.83,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(20.0, 2.0),
            )
            result = UserAPPortfolioResult(
                job=APPortfolioTrainJob(
                    user_id=1,
                    output_dir=root / "out",
                    command_record_path=command_record_path,
                ),
                baseline_desired_retention_values=(0.52, 0.54),
                baseline_metrics=[_metrics(10.0, 4.0), _metrics(11.0, 5.0)],
                baseline_hypervolume=1.0,
                portfolio_hypervolume=3.0,
                hypervolume_improvement=2.0,
                final_population_hypervolume=10.0,
                final_population_hypervolume_improvement=9.0,
                reference_point=ObjectivePoint(0.0, -10.0),
                selected_children=[
                    SelectedAPPortfolioChild(
                        portfolio_index=0,
                        candidate=candidate,
                        hypervolume_contribution=2.0,
                        pareto_rank=0,
                    )
                ],
                final_population=[candidate],
                base_weights=base_weights,
                history=[],
                passed=True,
            )

            artifact_paths = _write_portfolio_artifacts(
                result=result,
                config=config,
                config_path=config_path,
                settings=settings,
                ap_settings=ap_settings,
                portfolio=portfolio,
            )
            metrics = json.loads(
                (root / "out" / "policies" / "policy_0" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
            portfolio_payload = json.loads(
                (root / "out" / "portfolio.json").read_text(encoding="utf-8")
            )
            validated = validate_scheduler_artifact(
                artifact_paths[0],
                require_files=True,
            )
            metadata_config_resolves = (
                validated.config_snapshot_path == config_path.resolve()
            )
            metadata_command_resolves = (
                validated.training_command_path == command_record_path.resolve()
            )
            policy = json.loads(
                (root / "out" / "policies" / "policy_0" / "policy.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(metrics["portfolio_hypervolume"], 3.0)
        self.assertEqual(metrics["final_population_hypervolume"], 10.0)
        self.assertIsNone(metadata["baseline_desired_retention"])
        self.assertNotIn("lambda_value", metadata)
        self.assertFalse(Path(metadata["config_snapshot_path"]).is_absolute())
        self.assertFalse(Path(metadata["training_command_path"]).is_absolute())
        self.assertTrue(metadata_config_resolves)
        self.assertTrue(metadata_command_resolves)
        self.assertEqual(metadata["scheduler_desired_retention"], 0.83)
        self.assertEqual(
            metadata["action_space"], "fsrs6_ap_weight_delta_portfolio_child"
        )
        self.assertEqual(
            metadata["policy_path"],
            "policy.json",
        )
        self.assertEqual(
            metrics["portfolio_id"],
            metadata["portfolio_id"],
        )
        self.assertEqual(
            portfolio_payload["children"][0]["policy_path"],
            "policies/policy_0/policy.json",
        )
        self.assertEqual(
            portfolio_payload["children"][0]["metadata_path"],
            "policies/policy_0/metadata.json",
        )
        self.assertEqual(
            portfolio_payload["children"][0]["metrics_path"],
            "policies/policy_0/metrics.json",
        )
        self.assertEqual(policy["baseline_desired_retention"], 0.83)


if __name__ == "__main__":
    unittest.main()
