from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.core import CardView
from simulator.fsrs_defaults import DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds
from simulator.sa_fsrs6_policy import SAFSRS6Policy
from simulator.schedulers.fsrs import (
    FSRS3BatchSchedulerOps,
    FSRS6BatchSchedulerOps,
    FSRS6Scheduler,
)
from simulator.schedulers.sa_fsrs6 import SAFSRS6BatchSchedulerOps, SAFSRS6Scheduler


def _view(state, *, last_review: float = 0.0, interval: float = 1.0) -> CardView:
    return CardView(
        id=1,
        due=last_review + interval,
        last_review=last_review,
        interval=interval,
        reps=1,
        lapses=0,
        history=[],
        scheduler_state=state,
    )


class SAFSRS6SchedulerTests(unittest.TestCase):
    def test_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            SAFSRS6Policy.baseline(desired_retention=0.9).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            sa = SAFSRS6Scheduler(policy_json=policy_path, fsrs_weights=None)

            fsrs_interval, fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            sa_interval, sa_state = sa.init_card(_view(None), 3, 0.0)
            self.assertAlmostEqual(sa_interval, fsrs_interval, places=9)
            self.assertEqual(set(sa_state), {"s", "d"})

            elapsed_values = [1.0, 4.0, 12.0, 2.0]
            ratings = [3, 4, 1, 2]
            for elapsed, rating in zip(elapsed_values, ratings):
                fsrs_interval, fsrs_state = fsrs.schedule(
                    _view(fsrs_state), rating, elapsed, elapsed
                )
                sa_interval, sa_state = sa.schedule(
                    _view(sa_state), rating, elapsed, elapsed
                )
                self.assertAlmostEqual(sa_interval, fsrs_interval, places=9)
                self.assertAlmostEqual(sa_state["s"], fsrs_state["s"], places=9)
                self.assertAlmostEqual(sa_state["d"], fsrs_state["d"], places=9)

    def test_scheduler_does_not_reference_env_memory_state(self) -> None:
        source = inspect.getsource(SAFSRS6Scheduler)
        self.assertNotIn("memory_state", source)

    def test_policy_retention_is_clipped_to_configured_range(self) -> None:
        policy = SAFSRS6Policy(
            coefficients=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            retention_min=0.7,
            retention_max=0.98,
        )

        self.assertGreaterEqual(policy.evaluate(0.1, 1.0), 0.7)
        self.assertLessEqual(policy.evaluate(100.0, 10.0), 0.98)

    def test_batch_ops_accept_per_user_coefficients(self) -> None:
        policy = SAFSRS6Policy.baseline(desired_retention=0.9)
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [policy.coefficients[0] - 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = SAFSRS6BatchSchedulerOps(
            weights=weights,
            policy=policy,
            coefficients=coefficients,
            bounds=Bounds(),
            priority_mode="low_retrievability",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state = ops.init_state(user_count=2, deck_size=1)
        intervals = ops.update_learn(
            state,
            user_idx=torch.tensor([0, 1]),
            card_idx=torch.tensor([0, 0]),
            rating=torch.tensor([3, 3]),
        )

        self.assertGreater(float(intervals[1]), float(intervals[0]))

    def test_fsrs6_batch_ops_accept_per_user_desired_retention(self) -> None:
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        ops = FSRS6BatchSchedulerOps(
            weights=weights,
            desired_retention=torch.tensor([0.8, 0.9], dtype=torch.float32),
            bounds=Bounds(),
            priority_mode="low_retrievability",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state = ops.init_state(user_count=2, deck_size=1)
        intervals = ops.update_learn(
            state,
            user_idx=torch.tensor([0, 1]),
            card_idx=torch.tensor([0, 0]),
            rating=torch.tensor([3, 3]),
        )

        self.assertGreater(float(intervals[0]), float(intervals[1]))

    def test_fsrs3_batch_ops_accept_per_user_desired_retention(self) -> None:
        weights = torch.tensor([DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS3_WEIGHTS])
        ops = FSRS3BatchSchedulerOps(
            weights=weights,
            desired_retention=torch.tensor([0.8, 0.9], dtype=torch.float32),
            bounds=Bounds(),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state = ops.init_state(user_count=2, deck_size=1)
        intervals = ops.update_learn(
            state,
            user_idx=torch.tensor([0, 1]),
            card_idx=torch.tensor([0, 0]),
            rating=torch.tensor([3, 3]),
        )

        self.assertGreater(float(intervals[0]), float(intervals[1]))

    def test_fsrs3_batch_ops_reject_invalid_desired_retention_tensor(self) -> None:
        weights = torch.tensor([DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS3_WEIGHTS])
        with self.assertRaisesRegex(ValueError, "shape"):
            FSRS3BatchSchedulerOps(
                weights=weights,
                desired_retention=torch.tensor([0.8, 0.9, 0.95]),
                bounds=Bounds(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            FSRS3BatchSchedulerOps(
                weights=weights,
                desired_retention=torch.tensor([0.8, 1.0]),
                bounds=Bounds(),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
