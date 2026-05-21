from __future__ import annotations

import unittest

from experiments.single_card_tradeoff.cli.train_weight_control_analysis import (
    PER_USER_DISTILL_SCHEDULER,
    _distill_scheduler_spec,
    _segment_auc,
)


class TrainWeightControlAnalysisTests(unittest.TestCase):
    def test_distill_scheduler_spec_matches_exact_value_labels(self) -> None:
        self.assertEqual(
            _distill_scheduler_spec("distill_476"),
            PER_USER_DISTILL_SCHEDULER,
        )
        self.assertEqual(
            _distill_scheduler_spec("add_1_4"),
            f"{PER_USER_DISTILL_SCHEDULER}_add_1_4",
        )
        self.assertEqual(
            _distill_scheduler_spec("add_4_only"),
            f"{PER_USER_DISTILL_SCHEDULER}_add_4_only",
        )

    def test_segment_auc_integrates_same_target_time_saved(self) -> None:
        rows = [
            {
                "user_id": 2,
                "scheduler_spec": "baseline",
                "deck_expected_memorized": 9400,
                "deck_minutes_per_day": 20,
            },
            {
                "user_id": 2,
                "scheduler_spec": "baseline",
                "deck_expected_memorized": 9750,
                "deck_minutes_per_day": 30,
            },
            {
                "user_id": 2,
                "scheduler_spec": "candidate",
                "deck_expected_memorized": 9400,
                "deck_minutes_per_day": 18,
            },
            {
                "user_id": 2,
                "scheduler_spec": "candidate",
                "deck_expected_memorized": 9750,
                "deck_minutes_per_day": 28,
            },
        ]

        actual = _segment_auc(
            rows,
            user_id=2,
            baseline_scheduler="baseline",
            scheduler="candidate",
            segment_start=9400,
            segment_end=9750,
        )

        self.assertAlmostEqual(actual["same_target_time_saved_auc"], 2.0)
        self.assertAlmostEqual(actual["segment_coverage_percent"], 100.0)


if __name__ == "__main__":
    unittest.main()
