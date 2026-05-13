from __future__ import annotations

import unittest

import torch

from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps, AnkiSM2Scheduler


class AnkiSM2SchedulerTests(unittest.TestCase):
    def test_default_review_lapse_preserves_existing_one_day_interval(self) -> None:
        scheduler = AnkiSM2Scheduler()

        interval, ease = scheduler._next_interval(
            prev_interval=10.0,
            elapsed=10.0,
            rating=1,
            ease=2.5,
        )

        self.assertEqual(interval, 1.0)
        self.assertAlmostEqual(ease, 2.3)

    def test_new_interval_factor_affects_lapses(self) -> None:
        scheduler = AnkiSM2Scheduler(
            new_interval_factor=0.5,
            interval_multiplier=2.0,
        )

        interval, _ease = scheduler._next_interval(
            prev_interval=10.0,
            elapsed=10.0,
            rating=1,
            ease=2.5,
        )

        self.assertEqual(interval, 5.0)

    def test_interval_multiplier_affects_passing_reviews(self) -> None:
        scheduler = AnkiSM2Scheduler(interval_multiplier=2.0)

        interval, _ease = scheduler._next_interval(
            prev_interval=10.0,
            elapsed=10.0,
            rating=3,
            ease=2.5,
        )

        self.assertEqual(interval, 50.0)

    def test_batch_ops_accept_per_user_parameter_vectors(self) -> None:
        ops = AnkiSM2BatchSchedulerOps(
            graduating_interval=1.0,
            easy_interval=4.0,
            ease_start=2.5,
            easy_bonus=1.3,
            hard_interval_factor=1.2,
            ease_min=1.3,
            ease_max=5.5,
            new_interval_factor=torch.tensor([0.0, 0.5]),
            interval_multiplier=torch.tensor([1.0, 2.0]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state = ops.init_state(user_count=2, deck_size=1)

        intervals = ops.update_review(
            state,
            user_idx=torch.tensor([0, 1]),
            card_idx=torch.tensor([0, 0]),
            elapsed=torch.tensor([10.0, 10.0]),
            rating=torch.tensor([1, 1]),
            prev_interval=torch.tensor([10.0, 10.0]),
        )

        self.assertEqual([float(value) for value in intervals], [1.0, 5.0])


if __name__ == "__main__":
    unittest.main()
