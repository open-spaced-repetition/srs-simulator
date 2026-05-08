# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.core import CardView
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.fsrs6_adp_policy import (
    FSRS6_ADP_WEIGHT_BOUNDS,
    FSRS6ADPPolicy,
    clip_fsrs6_adp_weights,
    decode_weight_delta,
)
from simulator.schedulers.fsrs import FSRS6Scheduler
from simulator.schedulers.fsrs6_adp import FSRS6ADPScheduler


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


class FSRS6ADPPolicyTests(unittest.TestCase):
    def test_weight_clipper_uses_expected_bounds(self) -> None:
        low = [-100.0] * 21
        high = [1000.0] * 21

        self.assertEqual(clip_fsrs6_adp_weights(low)[0], 0.001)
        self.assertEqual(clip_fsrs6_adp_weights(low)[19], 0.01)
        self.assertEqual(clip_fsrs6_adp_weights(low)[20], 0.1)
        self.assertEqual(clip_fsrs6_adp_weights(high)[0], 100.0)
        self.assertEqual(clip_fsrs6_adp_weights(high)[16], 6.0)
        self.assertEqual(clip_fsrs6_adp_weights(high)[20], 0.8)
        self.assertEqual(FSRS6_ADP_WEIGHT_BOUNDS[19], (0.01, 0.8))

    def test_zero_delta_decodes_to_clipped_base_weights(self) -> None:
        decoded = decode_weight_delta(DEFAULT_FSRS6_WEIGHTS, [0.0] * 21)

        self.assertEqual(decoded, clip_fsrs6_adp_weights(DEFAULT_FSRS6_WEIGHTS))

    def test_policy_round_trips_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            policy = FSRS6ADPPolicy.from_search_vector(
                base_weights=DEFAULT_FSRS6_WEIGHTS,
                search_vector=[0.0] * 21,
                baseline_desired_retention=0.9,
            )
            policy.write_json(path)

            loaded = FSRS6ADPPolicy.from_json(path)

        self.assertEqual(loaded.weights, policy.weights)
        self.assertEqual(loaded.baseline_desired_retention, 0.9)
        self.assertEqual(len(loaded.search_vector), 21)

    def test_zero_delta_scheduler_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            FSRS6ADPPolicy.from_search_vector(
                base_weights=DEFAULT_FSRS6_WEIGHTS,
                search_vector=[0.0] * 21,
                baseline_desired_retention=0.9,
            ).write_json(path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            adp = FSRS6ADPScheduler(policy_json=path)

            fsrs_interval, fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            adp_interval, adp_state = adp.init_card(_view(None), 3, 0.0)
            self.assertAlmostEqual(adp_interval, fsrs_interval, places=9)

            for elapsed, rating in zip([1.0, 4.0, 12.0], [3, 4, 1]):
                fsrs_interval, fsrs_state = fsrs.schedule(
                    _view(fsrs_state), rating, elapsed, elapsed
                )
                adp_interval, adp_state = adp.schedule(
                    _view(adp_state), rating, elapsed, elapsed
                )
                self.assertAlmostEqual(adp_interval, fsrs_interval, places=9)
                self.assertAlmostEqual(adp_state["s"], fsrs_state["s"], places=9)
                self.assertAlmostEqual(adp_state["d"], fsrs_state["d"], places=9)


if __name__ == "__main__":
    unittest.main()
