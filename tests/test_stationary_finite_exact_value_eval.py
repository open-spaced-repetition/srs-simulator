from __future__ import annotations

import unittest
from pathlib import Path

from experiments.single_card_tradeoff.stationary_finite_exact_value_eval import (
    _direct_scheduler_spec,
    _distill_scheduler_spec,
    _label_path,
)


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


if __name__ == "__main__":
    unittest.main()
