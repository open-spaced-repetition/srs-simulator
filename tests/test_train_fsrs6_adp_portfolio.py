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

from experiments.rl_scheduler.train_cmaes_fsrs6_adp import ADPSettings
from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
)
from experiments.rl_scheduler.train_fsrs6_adp_portfolio import (
    ADPPortfolioCandidate,
    ADPPortfolioSettings,
    ADPPortfolioTrainJob,
    ObjectivePoint,
    SelectedADPPortfolioChild,
    UserADPPortfolioResult,
    _mutate_genome,
    _selection_payload,
    _write_portfolio_artifacts,
)
from experiments.rl_scheduler.portfolio_selection import (
    SelectionTask,
    select_survivors_for_generation,
)
from simulator.experiment_infra import ExperimentConfig
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.fsrs6_adp_policy import clip_fsrs6_adp_weights


def _metrics(memorized: float, time_average: float) -> CandidateMetrics:
    return CandidateMetrics(
        memorized_average=memorized,
        time_average=time_average,
        memorized_per_minute=memorized / time_average,
        total_reviews=0,
        total_lapses=0,
        total_cost=0.0,
    )


def _config(output_root: Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "adp-portfolio-test",
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
                "adp": {"weight_delta_scale": 0.5},
            },
        },
        config_path=output_root / "config.toml",
    )


class FSRS6ADPPortfolioTests(unittest.TestCase):
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

    def test_selection_helper_returns_timed_adp_survivors(self) -> None:
        base_weights = clip_fsrs6_adp_weights(DEFAULT_FSRS6_WEIGHTS)
        baseline_points = [ObjectivePoint(0.0, -10.0)]
        reference = ObjectivePoint(-1.0, -11.0)
        candidates = [
            ADPPortfolioCandidate(
                candidate_id=1,
                desired_retention=0.70,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(4.0, 4.0),
            ),
            ADPPortfolioCandidate(
                candidate_id=2,
                desired_retention=0.80,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(8.0, 8.0),
            ),
            ADPPortfolioCandidate(
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

    def test_artifacts_record_exported_subset_hypervolume_for_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            config_path = root / "config.toml"
            config_path.write_text("", encoding="utf-8")
            settings = PolicySearchSettings.from_mapping(config.training_policy_search)
            portfolio = ADPPortfolioSettings(portfolio_size=1)
            adp_settings = ADPSettings(dr_batch_size=1, weight_delta_scale=0.5)
            base_weights = clip_fsrs6_adp_weights(DEFAULT_FSRS6_WEIGHTS)
            candidate = ADPPortfolioCandidate(
                candidate_id=7,
                desired_retention=0.83,
                search_vector=(0.0,) * 21,
                weights=base_weights,
                metrics=_metrics(20.0, 2.0),
            )
            result = UserADPPortfolioResult(
                job=ADPPortfolioTrainJob(
                    user_id=1,
                    output_dir=root / "out",
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
                    SelectedADPPortfolioChild(
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
                adp_settings=adp_settings,
                portfolio=portfolio,
            )
            metrics = json.loads(
                (root / "out" / "policies" / "policy_0" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
            policy = json.loads(
                (root / "out" / "policies" / "policy_0" / "policy.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(metrics["portfolio_hypervolume"], 3.0)
        self.assertEqual(metrics["final_population_hypervolume"], 10.0)
        self.assertIsNone(metadata["baseline_desired_retention"])
        self.assertNotIn("lambda_value", metadata)
        self.assertEqual(metadata["scheduler_desired_retention"], 0.83)
        self.assertEqual(
            metadata["action_space"], "fsrs6_adp_weight_delta_portfolio_child"
        )
        self.assertEqual(policy["baseline_desired_retention"], 0.83)


if __name__ == "__main__":
    unittest.main()
