# ruff: noqa: E402
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

from experiments.single_card_tradeoff.uvfa_ppo import PolicyValueNet
from simulator.core import CardView
from simulator.fsrs6_oracle_stationary_finite_distill_policy import (
    FSRS6OracleStationaryFiniteDistillBatchPolicy,
    FSRS6OracleStationaryFiniteDistillPolicy,
    POLICY_TYPE,
)
from simulator.schedulers.fsrs6_oracle_stationary_finite_distill import (
    FSRS6OracleStationaryFiniteDistillScheduler,
)


def _write_checkpoint(path: Path) -> None:
    model = PolicyValueNet(3, 3, 4, architecture="linear", depth=1)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.policy.bias[1] = 1.0
    torch.save(
        {
            "policy_type": POLICY_TYPE,
            "model_state_dict": model.state_dict(),
            "cost_weights": [0.0, 16.0, 64.0],
            "action_retentions": [0.5, 0.8, 0.95],
            "days": 10,
            "obs_dim": 3,
            "obs_mode": "oracle_stationary",
            "hidden_size": 4,
            "network": "linear",
            "network_depth": 1,
        },
        path,
    )


def _write_policy(path: Path, checkpoint_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "policy_type": POLICY_TYPE,
                "checkpoint_path": checkpoint_path.name,
                "goal_cost_weight": 16.0,
                "goal_norm_max": 64.0,
                "action_retentions": [0.5, 0.8, 0.95],
                "obs_mode": "oracle_stationary",
                "user_id": 1,
                "portfolio_index": 0,
            }
        ),
        encoding="utf-8",
    )


def _view(state: object | None) -> CardView:
    return CardView(
        id=1,
        due=0.0,
        last_review=0.0,
        interval=1.0,
        reps=0,
        lapses=0,
        history=[],
        scheduler_state=state,
    )


class FSRS6OracleStationaryFiniteDistillSchedulerTests(unittest.TestCase):
    def test_policy_json_loads_checkpoint_and_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "policy.pt"
            policy_path = root / "policy.json"
            _write_checkpoint(checkpoint_path)
            _write_policy(policy_path, checkpoint_path)

            policy = FSRS6OracleStationaryFiniteDistillPolicy.from_json(policy_path)

            self.assertEqual(policy.checkpoint_path, checkpoint_path)
            self.assertEqual(policy.user_id, 1)
            self.assertAlmostEqual(policy.evaluate(2.0, 5.0), 0.8)

    def test_event_scheduler_uses_distilled_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "policy.pt"
            policy_path = root / "policy.json"
            _write_checkpoint(checkpoint_path)
            _write_policy(policy_path, checkpoint_path)

            scheduler = FSRS6OracleStationaryFiniteDistillScheduler(
                policy_json=policy_path,
                fsrs_weights=None,
            )
            interval, state = scheduler.init_card(_view(None), 3, 0.0)

            self.assertGreaterEqual(interval, 1.0)
            self.assertEqual(set(state), {"s", "d"})

    def test_batch_policy_evaluates_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "policy.pt"
            policy_path = root / "policy.json"
            _write_checkpoint(checkpoint_path)
            _write_policy(policy_path, checkpoint_path)
            policy = FSRS6OracleStationaryFiniteDistillPolicy.from_json(policy_path)
            batch_policy = FSRS6OracleStationaryFiniteDistillBatchPolicy(
                [policy, policy],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

            retentions = batch_policy.evaluate(
                torch.tensor([2.0, 4.0]),
                torch.tensor([5.0, 6.0]),
            )

            self.assertEqual(retentions.shape, (2,))
            self.assertAlmostEqual(float(retentions[0]), 0.8)
            self.assertAlmostEqual(float(retentions[1]), 0.8)


if __name__ == "__main__":
    unittest.main()
