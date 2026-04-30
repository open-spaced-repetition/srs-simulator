from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.core import CardView
from simulator.sa_fsrs6_policy import SAFSRS6Policy
from simulator.schedulers.fsrs import FSRS6Scheduler
from simulator.schedulers.sa_fsrs6 import SAFSRS6Scheduler


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


if __name__ == "__main__":
    unittest.main()
