from __future__ import annotations

import unittest
from pathlib import Path

import torch

from experiments.single_card_tradeoff.oracle_frontier import (
    FSRS6BatchedStationaryFiniteOracle,
)
from experiments.single_card_tradeoff.stationary_finite_exact_value_eval import (
    _adr_transition_tables,
    _direct_scheduler_spec,
    _distill_scheduler_spec,
    _label_path,
    _round_half_up_tensor,
)
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS


class StationaryFiniteExactValueEvalTest(unittest.TestCase):
    def test_label_path_requires_label_and_path(self) -> None:
        self.assertEqual(_label_path("foo=bar"), ("foo", Path("bar")))
        with self.assertRaises(SystemExit):
            _label_path("missing_separator")
        with self.assertRaises(SystemExit):
            _label_path("=bar")
        with self.assertRaises(SystemExit):
            _label_path("foo=")

    def test_scheduler_specs_are_stable(self) -> None:
        self.assertEqual(
            _distill_scheduler_spec("distill_476"),
            "fsrs6_oracle_stationary_finite_distill_per_user",
        )
        self.assertEqual(
            _distill_scheduler_spec("distill_r4d1_e512"),
            "fsrs6_oracle_stationary_finite_distill_per_user_r4d1_e512",
        )
        self.assertEqual(
            _direct_scheduler_spec("direct_7_sparse"),
            "fsrs6_low_param_direct_policy_search_direct_7_sparse",
        )

    def test_round_half_up_matches_simulator_interval_rounding(self) -> None:
        values = torch.tensor([1.49, 1.5, 2.5, 3.51], dtype=torch.float64)
        rounded = _round_half_up_tensor(values)
        self.assertEqual(rounded.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_baseline_adr_transition_matches_same_retention_action(self) -> None:
        oracle = FSRS6BatchedStationaryFiniteOracle(
            days=10,
            action_retentions=[0.9],
            s_grid_size=8,
            d_grid_size=8,
            fsrs_weights=[DEFAULT_FSRS6_WEIGHTS],
            first_rating_prob=[DEFAULT_FIRST_RATING_PROB],
            review_rating_prob=[DEFAULT_REVIEW_RATING_PROB],
            learning_costs=[DEFAULT_STATE_RATING_COSTS.learning],
            review_costs=[DEFAULT_STATE_RATING_COSTS.review],
            dtype=torch.float64,
            device="cpu",
        )
        policy = FSRS6ADRPolicy.baseline(desired_retention=0.9)
        interval, prob, next_idx, next_weight = _adr_transition_tables(
            oracle,
            policies=[[policy]],
        )
        expected = oracle._action_tables
        self.assertTrue(torch.equal(interval, expected.interval))
        self.assertTrue(torch.allclose(prob, expected.prob))
        self.assertTrue(torch.equal(next_idx, expected.next_idx))
        self.assertTrue(torch.allclose(next_weight, expected.next_weight))


if __name__ == "__main__":
    unittest.main()
