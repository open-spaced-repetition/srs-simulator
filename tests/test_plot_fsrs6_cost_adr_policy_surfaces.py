from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler import plot_fsrs6_cost_adr_policy_surfaces as plot  # noqa: E402
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FEATURE_VERSION_INTERVAL_MONO,
    FEATURE_VERSION_RETENTION_MONO,
    FSRS6CostConditionedADRPolicy,
)
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS  # noqa: E402
from simulator.math.fsrs import FSRS6Params  # noqa: E402


COEFFICIENTS = (0.0,) * 24


def _write_policy(
    root: Path,
    *,
    user_id: int,
    metadata_cost_weights: list[float] | None = None,
) -> Path:
    policy_dir = root / f"user_{user_id}"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "policy.json"
    FSRS6CostConditionedADRPolicy(
        coefficients=COEFFICIENTS,
        action_head=ACTION_HEAD_INTERVAL,
        feature_version=FEATURE_VERSION_INTERVAL_MONO,
    ).write_json(policy_path)
    metadata: dict[str, object] = {"training_user_ids": [user_id]}
    if metadata_cost_weights is not None:
        metadata["cost_weights"] = metadata_cost_weights
    (policy_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return policy_path


class PlotFSRS6CostADRPolicySurfacesTests(unittest.TestCase):
    def test_discovers_cost_adr_policy_with_metadata_cost_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = _write_policy(
                root,
                user_id=1,
                metadata_cost_weights=[0.0, 4.0, 16.0],
            )

            entries = plot._discover_policies(
                root,
                users=(1,),
                start_user=None,
                end_user=None,
                cost_weights=None,
            )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].user_id, 1)
        self.assertEqual(entries[0].path, policy_path.resolve())
        self.assertEqual(entries[0].cost_weights, (0.0, 4.0, 16.0))

    def test_cli_cost_weights_override_metadata_cost_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_policy(root, user_id=1, metadata_cost_weights=[0.0, 4.0])

            entries = plot._discover_policies(
                root,
                users=(1,),
                start_user=None,
                end_user=None,
                cost_weights=(1.0, 2.0),
            )

        self.assertEqual(entries[0].cost_weights, (1.0, 2.0))

    def test_builds_log_interval_surface_arrays(self) -> None:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=COEFFICIENTS,
            action_head=ACTION_HEAD_INTERVAL,
            feature_version=FEATURE_VERSION_INTERVAL_MONO,
        )

        z_values, customdata = plot._build_surface_arrays(
            policy=policy,
            cost_weight=0.0,
            s_grid=(1.0, 10.0),
            d_grid=(1.0, 5.0),
            z_mode="log_interval",
        )

        self.assertEqual(len(z_values), 2)
        self.assertEqual(len(z_values[0]), 2)
        self.assertEqual(len(customdata), 2)
        self.assertEqual(len(customdata[0]), 2)
        self.assertEqual(customdata[0][0][0], 0.0)
        self.assertGreaterEqual(customdata[0][0][2], 1.0)
        self.assertAlmostEqual(z_values[0][0], 0.0)

    def test_builds_retention_surface_arrays_from_interval_policy(self) -> None:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=COEFFICIENTS,
            action_head=ACTION_HEAD_INTERVAL,
            feature_version=FEATURE_VERSION_INTERVAL_MONO,
        )
        params = FSRS6Params(DEFAULT_FSRS6_WEIGHTS)

        z_values, customdata = plot._build_surface_arrays(
            policy=policy,
            fsrs6_params=params,
            cost_weight=0.0,
            s_grid=(1.0, 10.0, 100.0),
            d_grid=(1.0, 5.0),
            z_mode="retention",
        )

        self.assertEqual(len(z_values), 2)
        self.assertEqual(len(z_values[0]), 3)
        self.assertEqual(len(customdata), 3)
        self.assertEqual(len(customdata[0]), 2)
        self.assertAlmostEqual(z_values[0][0], 0.9)
        self.assertAlmostEqual(customdata[0][0][2], 1.0)
        self.assertAlmostEqual(customdata[0][0][3], z_values[0][0])
        self.assertAlmostEqual(customdata[2][1][3], z_values[1][2])

    def test_retention_policy_retention_surface_includes_interval_customdata(
        self,
    ) -> None:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=COEFFICIENTS,
            action_head=ACTION_HEAD_RETENTION,
            feature_version=FEATURE_VERSION_RETENTION_MONO,
        )
        params = FSRS6Params(DEFAULT_FSRS6_WEIGHTS)

        z_values, customdata = plot._build_surface_arrays(
            policy=policy,
            fsrs6_params=params,
            cost_weight=0.0,
            s_grid=(1.0, 10.0),
            d_grid=(1.0,),
            z_mode="retention",
        )

        self.assertTrue(
            plot._entry_needs_fsrs6_params(
                plot.PolicyEntry(1, Path("policy.json"), policy, (0.0,)), "retention"
            )
        )
        self.assertAlmostEqual(z_values[0][0], customdata[0][0][3])
        self.assertGreater(customdata[0][0][2], 0.0)


if __name__ == "__main__":
    unittest.main()
