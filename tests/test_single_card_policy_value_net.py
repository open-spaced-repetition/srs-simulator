from __future__ import annotations

import unittest

import torch

from experiments.single_card_tradeoff.uvfa_ppo import PolicyValueNet


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


if __name__ == "__main__":
    unittest.main()
