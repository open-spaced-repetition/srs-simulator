from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_sa_fsrs6_dr import _policy_feature_version


class TrainSAFSRS6DRConfigTests(unittest.TestCase):
    def test_policy_feature_version_defaults_to_log_poly(self) -> None:
        self.assertEqual(
            _policy_feature_version({}),
            "sa_fsrs6_dr_log_poly_v1",
        )

    def test_policy_feature_version_accepts_linear_variant(self) -> None:
        self.assertEqual(
            _policy_feature_version({"feature_version": "sa_fsrs6_dr_log_linear_v1"}),
            "sa_fsrs6_dr_log_linear_v1",
        )

    def test_policy_feature_version_rejects_unknown_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _policy_feature_version({"feature_version": "unknown"})


if __name__ == "__main__":
    unittest.main()
