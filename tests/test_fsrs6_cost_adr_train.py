from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.cli.fsrs6_cost_adr_train import (  # noqa: E402
    _continuous_distill_teacher_table,
    fit_policy_from_table,
)
from experiments.single_card_tradeoff.models.policy_runtime import (  # noqa: E402
    RetentionDistillNet,
)
from simulator.fsrs_defaults import resolve_fsrs6_weights  # noqa: E402
from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    ACTION_HEAD_INTERVAL,
    STATE_FEATURE_COUNT_COMPACT,
)


class FSRS6CostADRTrainTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
