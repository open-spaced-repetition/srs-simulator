import argparse
import unittest
from unittest.mock import MagicMock

from simulator.core import CardView, ReviewLog
from simulator.schedulers.lstm import LSTMScheduler


class TestLSTMScheduler(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()
        self.args.user_id = 1
        self.args.srs_benchmark_root = None
        self.args.torch_device = "cpu"
        self.args.desired_retention = 0.9
        self.scheduler = LSTMScheduler(self.args)
        self.scheduler.model.get_interval_from_history = MagicMock(return_value=10.0)

    def test_init_card(self):
        # Arrange
        card_view = CardView(
            id=1,
            due=0,
            last_review=0,
            interval=0,
            reps=0,
            lapses=0,
            history=[],
            scheduler_state=None,
            metadata={},
        )

        # Act
        interval, state = self.scheduler.init_card(card_view, rating=4, day=0)

        # Assert
        self.assertIsInstance(interval, float)
        self.assertEqual(state, [(0.0, 4)])
        self.scheduler.model.get_interval_from_history.assert_called_once_with(
            [(0.0, 4)], self.args.desired_retention
        )

    def test_schedule(self):
        # Arrange
        initial_history = [(0.0, 4)]
        card_view = CardView(
            id=1,
            due=1,
            last_review=0,
            interval=1,
            reps=1,
            lapses=0,
            history=[ReviewLog(rating=4, elapsed=0, day=0)],
            scheduler_state=initial_history,
            metadata={},
        )

        # Act
        interval, state = self.scheduler.schedule(
            card_view, rating=3, elapsed=1, day=1
        )

        # Assert
        self.assertIsInstance(interval, float)
        expected_history = [(0.0, 4), (1, 3)]
        self.assertEqual(state, expected_history)
        self.scheduler.model.get_interval_from_history.assert_called_once_with(
            expected_history, self.args.desired_retention
        )


if __name__ == "__main__":
    unittest.main()
