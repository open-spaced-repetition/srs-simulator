from __future__ import annotations

import unittest

import torch

from experiments.single_card_tradeoff.cli.low_param_direct_policy_search_multiuser import (
    direct_policy_retention,
    initial_theta,
    parameter_count_for_family,
    parameter_names_for_family,
)


class LowParamDirectPolicySearchTest(unittest.TestCase):
    def test_parameter_counts_match_family_names(self) -> None:
        self.assertEqual(parameter_count_for_family("bilinear_monotone"), 7)
        self.assertEqual(parameter_count_for_family("interaction15_monotone"), 15)
        self.assertEqual(parameter_count_for_family("basis32_monotone"), 32)
        self.assertEqual(parameter_count_for_family("basis64_monotone"), 64)
        for family in (
            "bilinear_monotone",
            "interaction15_monotone",
            "basis32_monotone",
            "basis64_monotone",
        ):
            self.assertEqual(
                len(parameter_names_for_family(family)),
                parameter_count_for_family(family),
            )

    def test_default_initial_policy_is_monotone_in_cost(self) -> None:
        for family in (
            "bilinear_monotone",
            "interaction15_monotone",
            "basis32_monotone",
            "basis64_monotone",
        ):
            theta = initial_theta(
                policy_family=family,
                user_count=1,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            obs = torch.tensor(
                [
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.25],
                    [0.5, 0.5, 0.75],
                    [0.5, 0.5, 1.0],
                ],
                dtype=torch.float64,
            )
            retention = direct_policy_retention(
                theta.expand(obs.shape[0], -1),
                obs,
                policy_family=family,
                min_retention=0.5,
                max_retention=0.98,
            )
            self.assertTrue(bool(torch.all(retention[:-1] >= retention[1:])))


if __name__ == "__main__":
    unittest.main()
