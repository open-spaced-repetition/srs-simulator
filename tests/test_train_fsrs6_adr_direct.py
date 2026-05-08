from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_cmaes_fsrs6_adr_direct import (
    baseline_coefficients,
    optimizer_settings_from_mapping,
)
from experiments.rl_scheduler.train_fsrs6_adr_direct import (
    SASettings,
    _policy_feature_version,
    _relative_gain_fraction_gate_metrics,
    _required_relative_gain_pass_count,
    _score_from_relative_gains,
)
from simulator.fsrs6_adr_direct_policy import (
    FEATURE_VERSION_LOG_LINEAR,
    FSRS6ADRDirectPolicy,
)


class TrainFSRS6ADRDirectConfigTests(unittest.TestCase):
    def test_policy_feature_version_defaults_to_log_poly(self) -> None:
        self.assertEqual(_policy_feature_version({}), "fsrs6_adr_direct_log_poly_v1")

    def test_policy_feature_version_accepts_linear_variant(self) -> None:
        self.assertEqual(
            _policy_feature_version({"feature_version": FEATURE_VERSION_LOG_LINEAR}),
            FEATURE_VERSION_LOG_LINEAR,
        )

    def test_policy_feature_version_rejects_unknown_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _policy_feature_version({"feature_version": "unknown"})

    def test_cmaes_default_initial_mean_uses_baseline_policy(self) -> None:
        settings = SASettings(
            baseline_desired_retention=0.9,
            retention_min=0.5,
            retention_max=0.98,
        )

        optimizer = optimizer_settings_from_mapping(
            {"name": "cma_es", "population_size": 4},
            settings=settings,
            feature_version=FEATURE_VERSION_LOG_LINEAR,
        )

        self.assertEqual(
            optimizer.initial_mean,
            FSRS6ADRDirectPolicy.baseline(
                desired_retention=settings.baseline_desired_retention,
                retention_min=settings.retention_min,
                retention_max=settings.retention_max,
                feature_version=FEATURE_VERSION_LOG_LINEAR,
            ).coefficients,
        )
        self.assertEqual(
            optimizer.initial_mean,
            baseline_coefficients(
                settings=settings,
                feature_version=FEATURE_VERSION_LOG_LINEAR,
            ),
        )

    def test_cmaes_default_initial_mean_is_clamped_to_bounds(self) -> None:
        settings = SASettings(
            baseline_desired_retention=0.5,
            retention_min=0.5,
            retention_max=0.98,
            coefficient_min=-8.0,
            coefficient_max=8.0,
        )

        optimizer = optimizer_settings_from_mapping(
            {"name": "cma_es", "population_size": 4},
            settings=settings,
            feature_version=FEATURE_VERSION_LOG_LINEAR,
        )

        self.assertEqual(optimizer.initial_mean, (-8.0, 0.0, 0.0))

    def test_score_makes_non_positive_candidates_strictly_worse(self) -> None:
        efficiency_trap = _score_from_relative_gains(
            -0.3347,
            37.9,
            0.5,
        )
        barely_feasible = _score_from_relative_gains(
            1e-9,
            1e-9,
            0.5,
        )
        below_baseline_failure = _score_from_relative_gains(
            -0.000386,
            0.023,
            0.5,
        )
        zero_boundary_failure = _score_from_relative_gains(
            0.0,
            37.9,
            0.5,
        )

        self.assertLess(efficiency_trap, barely_feasible)
        self.assertLess(below_baseline_failure, barely_feasible)
        self.assertLess(zero_boundary_failure, barely_feasible)
        self.assertLess(efficiency_trap, 0.0)
        self.assertLess(below_baseline_failure, 0.0)
        self.assertLessEqual(zero_boundary_failure, 0.0)

    def test_fraction_gate_requires_eighty_percent_points(self) -> None:
        self.assertEqual(_required_relative_gain_pass_count(5), 4)
        metrics = _relative_gain_fraction_gate_metrics(
            [0.01, 0.02, 0.03, 0.04, -0.001],
            [0.01, 0.02, 0.03, 0.04, 0.30],
        )

        self.assertTrue(metrics["passed_relative_gain_fraction_gate"])
        self.assertEqual(metrics["passed_desired_retention_points"], 4)
        self.assertEqual(metrics["required_passed_desired_retention_points"], 4)


if __name__ == "__main__":
    unittest.main()
