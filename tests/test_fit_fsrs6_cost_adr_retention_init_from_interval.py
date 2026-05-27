from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler import (  # noqa: E402
    fit_fsrs6_cost_adr_retention_init_from_interval as fit_init,
)
from experiments.single_card_tradeoff.cli.fsrs6_cost_adr_train import (  # noqa: E402
    fit_policy_from_table,
)
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS  # noqa: E402
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_RETENTION,
    FSRS6CostConditionedADRPolicy,
)
from simulator.math.fsrs import FSRS6Params  # noqa: E402


class FitFSRS6CostADRRetentionInitFromIntervalTests(unittest.TestCase):
    def test_distribution_stats_reports_clipped_fractions(self) -> None:
        values = torch.tensor([0.2, 0.4, 0.9, 0.999], dtype=torch.float64)

        stats = fit_init._distribution_stats(
            values,
            retention_min=0.3,
            retention_max=0.995,
        )

        self.assertAlmostEqual(stats["q000"], 0.2)
        self.assertAlmostEqual(stats["q100"], 0.999)
        self.assertAlmostEqual(stats["below_retention_min_fraction"], 0.25)
        self.assertAlmostEqual(stats["above_retention_max_fraction"], 0.25)
        self.assertAlmostEqual(stats["below_0_50_fraction"], 0.5)
        self.assertAlmostEqual(stats["above_0_98_fraction"], 0.25)

    def test_implied_retention_table_can_seed_retention_policy_fit(self) -> None:
        interval_policy = FSRS6CostConditionedADRPolicy(coefficients=(0.0,) * 24)
        table, stats = fit_init._implied_retention_table(
            interval_policy=interval_policy,
            fsrs_params=FSRS6Params(DEFAULT_FSRS6_WEIGHTS),
            cost_weights=(0.0, 4.0),
            s_points=4,
            d_points=3,
            retention_min=0.3,
            retention_max=0.995,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(table.policy.shape), (2, 4, 3))
        self.assertGreaterEqual(stats["q000"], 0.0)
        self.assertLessEqual(stats["q100"], 1.0)

        policy, fit_stats = fit_policy_from_table(
            table=table,
            cost_weights=(0.0, 4.0),
            action_head=ACTION_HEAD_RETENTION,
            state_feature_count=6,
            epochs=2,
            learning_rate=0.01,
            weight_decay=0.0,
            max_grad_norm=10.0,
        )

        self.assertEqual(policy.action_head, ACTION_HEAD_RETENTION)
        self.assertEqual(policy.parameter_count, 24)
        self.assertTrue(fit_stats["final_loss"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
