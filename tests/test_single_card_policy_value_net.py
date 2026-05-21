from __future__ import annotations

import argparse
import unittest

import torch

from experiments.single_card_tradeoff.cli.uvfa_ppo import (
    model_checkpoint_payload,
    TrainStats,
)
from experiments.single_card_tradeoff.models.policy_net import PolicyValueNet


class PolicyValueNetArchitectureTests(unittest.TestCase):
    def test_small_architectures_have_expected_parameter_counts(self) -> None:
        obs = torch.zeros((2, 3), dtype=torch.float32)
        cases = [
            ("linear", 1, 44),
            ("quadratic", 1, 110),
            ("mlp", 8, 212),
            ("residual", 5, 172),
        ]
        for architecture, hidden_size, expected_params in cases:
            with self.subTest(architecture=architecture):
                model = PolicyValueNet(
                    obs_dim=3,
                    action_count=11,
                    hidden_size=hidden_size,
                    architecture=architecture,
                    depth=1,
                )
                param_count = sum(
                    value.numel() for value in model.state_dict().values()
                )
                logits, values = model(obs)

                self.assertEqual(param_count, expected_params)
                self.assertEqual(logits.shape, (2, 11))
                self.assertEqual(values.shape, (2,))

    def test_uvfa_checkpoint_payload_keeps_existing_schema(self) -> None:
        model = PolicyValueNet(
            obs_dim=3,
            action_count=2,
            hidden_size=4,
            architecture="linear",
            depth=1,
        )
        args = argparse.Namespace(
            days=30,
            obs_mode="rich",
            hidden_size=4,
            network="linear",
            network_depth=1,
            guide_policy="oracle",
            oracle_s_grid_size=8,
            oracle_d_grid_size=8,
        )

        payload = model_checkpoint_payload(
            model=model,
            args=args,
            cost_weights=[1.0, 2.0],
            action_retentions=[0.5, 0.9],
            train_stats=TrainStats(updates=3, transitions=4, runtime_s=5.0),
        )

        self.assertEqual(
            set(payload),
            {
                "model_state_dict",
                "cost_weights",
                "action_retentions",
                "days",
                "obs_dim",
                "obs_mode",
                "hidden_size",
                "network",
                "network_depth",
                "guide_policy",
                "oracle_s_grid_size",
                "oracle_d_grid_size",
                "train_updates",
                "train_transitions",
                "train_runtime_s",
            },
        )
        self.assertEqual(payload["cost_weights"], [1.0, 2.0])
        self.assertEqual(payload["action_retentions"], [0.5, 0.9])


if __name__ == "__main__":
    unittest.main()
