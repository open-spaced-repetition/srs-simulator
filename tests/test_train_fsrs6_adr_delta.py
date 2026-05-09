from __future__ import annotations

from pathlib import Path
import inspect
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.adr_delta_common import (
    _candidate_evaluation,
    _dr_grid_passed_relative_gains,
    _score_dr_grid_relative_gains,
)
from experiments.rl_scheduler.adr_delta_common import _policy_feature_version
from experiments.rl_scheduler.policy_search_common import CandidateMetrics


class TrainFSRS6ADRDeltaConfigTests(unittest.TestCase):
    def test_policy_feature_version_defaults_to_log_poly(self) -> None:
        self.assertEqual(
            _policy_feature_version({}),
            "fsrs6_adr_delta_log_poly_v1",
        )

    def test_policy_feature_version_accepts_linear_variant(self) -> None:
        self.assertEqual(
            _policy_feature_version(
                {"feature_version": "fsrs6_adr_delta_log_linear_v1"}
            ),
            "fsrs6_adr_delta_log_linear_v1",
        )

    def test_policy_feature_version_rejects_unknown_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _policy_feature_version({"feature_version": "unknown"})

    def test_delta_score_rejects_mean_positive_candidate_with_failed_dr(self) -> None:
        relative_memorized_gains = [-0.01, 0.07]
        relative_efficiency_gains = [0.20, 0.20]

        self.assertGreater(sum(relative_memorized_gains) / 2, 0.0)
        self.assertGreater(sum(relative_efficiency_gains) / 2, 0.0)
        self.assertFalse(
            _dr_grid_passed_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
            )
        )
        self.assertLessEqual(
            _score_dr_grid_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
                0.5,
            ),
            0.0,
        )

    def test_delta_score_accepts_eighty_percent_positive_dr_points(self) -> None:
        relative_memorized_gains = [0.01, 0.02, 0.03, 0.04, -0.01]
        relative_efficiency_gains = [0.02, 0.03, 0.04, 0.05, 0.20]

        self.assertTrue(
            _dr_grid_passed_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
            )
        )
        self.assertGreater(
            _score_dr_grid_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
                0.5,
            ),
            0.0,
        )

    def test_delta_score_accepts_all_dr_positive_candidate(self) -> None:
        relative_memorized_gains = [1e-6, 0.02]
        relative_efficiency_gains = [1e-6, 0.04]

        self.assertTrue(
            _dr_grid_passed_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
            )
        )
        self.assertGreater(
            _score_dr_grid_relative_gains(
                relative_memorized_gains,
                relative_efficiency_gains,
                0.5,
            ),
            0.0,
        )

    def test_in_process_batch_uses_delta_grid_scoring_helper(self) -> None:
        from simulator.experiment_infra.training_batch import (
            _evaluate_dr_conditioned_batch,
        )

        source = inspect.getsource(_evaluate_dr_conditioned_batch)

        self.assertIn("_candidate_evaluation", source)
        self.assertNotIn("_score_from_relative_gains", source)

    def test_candidate_evaluation_reports_all_dr_gate_fields(self) -> None:
        baselines = [
            CandidateMetrics(100.0, 10.0, 10.0, 10, 0, 10.0),
            CandidateMetrics(100.0, 10.0, 10.0, 10, 0, 10.0),
        ]
        metrics = [
            CandidateMetrics(101.0, 10.0, 10.1, 10, 0, 10.0),
            CandidateMetrics(99.0, 10.0, 12.0, 10, 0, 10.0),
        ]

        evaluation = _candidate_evaluation(metrics, baselines, 0.5)

        self.assertFalse(evaluation.passed_overfit_gate)
        self.assertLess(evaluation.min_relative_memorized_gain, 0.0)
        self.assertGreater(evaluation.mean_relative_efficiency_gain, 0.0)
        self.assertEqual(len(evaluation.relative_memorized_gains), 2)
        self.assertEqual(len(evaluation.relative_efficiency_gains), 2)


if __name__ == "__main__":
    unittest.main()
