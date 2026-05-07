from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_cmaes_fsrs6 import (
    baseline_coefficients,
    optimizer_settings_from_mapping,
)
from experiments.rl_scheduler.train_sa_fsrs6 import (
    SASettings,
    _policy_feature_version,
    _score_from_relative_gains,
)
from simulator.sa_fsrs6_policy import FEATURE_VERSION_LOG_LINEAR, SAFSRS6Policy


class TrainSAFSRS6ConfigTests(unittest.TestCase):
    def test_policy_feature_version_defaults_to_log_poly(self) -> None:
        self.assertEqual(_policy_feature_version({}), "sa_fsrs6_log_poly_v1")

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
            SAFSRS6Policy.baseline(
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

    def test_score_makes_infeasible_candidates_strictly_worse(self) -> None:
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
        boundary_failure = _score_from_relative_gains(
            -0.000386,
            0.023,
            0.5,
        )

        self.assertLess(efficiency_trap, barely_feasible)
        self.assertLess(boundary_failure, barely_feasible)
        self.assertLess(efficiency_trap, 0.0)
        self.assertLess(boundary_failure, 0.0)


if __name__ == "__main__":
    unittest.main()
