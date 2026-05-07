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
from simulator.fsrs6_adr_delta_policy import (
    FEATURE_VERSION_LOG_LINEAR as DR_FEATURE_VERSION_LOG_LINEAR,
)
from simulator.fsrs6_adr_delta_policy import FSRS6ADRDeltaPolicy
from simulator.fsrs6_adr_direct_policy import (
    FEATURE_VERSION_LOG_LINEAR as SA_FEATURE_VERSION_LOG_LINEAR,
)
from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
from simulator.schedulers.fsrs import (
    FSRS3BatchSchedulerOps,
    FSRS6BatchSchedulerOps,
    FSRS6Scheduler,
    FSRS6VectorizedSchedulerOps,
)
from simulator.schedulers.fsrs6_adr_delta import (
    FSRS6ADRDeltaBatchSchedulerOps,
    FSRS6ADRDeltaScheduler,
    FSRS6ADRDeltaVectorizedSchedulerOps,
)
from simulator.schedulers.fsrs6_adr_direct import (
    FSRS6ADRDirectBatchSchedulerOps,
    FSRS6ADRDirectScheduler,
    FSRS6ADRDirectVectorizedSchedulerOps,
)


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


class FSRS6ADRDirectSchedulerTests(unittest.TestCase):
    def test_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDirectPolicy.baseline(desired_retention=0.9).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            sa = FSRS6ADRDirectScheduler(policy_json=policy_path, fsrs_weights=None)

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
        source = inspect.getsource(FSRS6ADRDirectScheduler)
        self.assertNotIn("memory_state", source)

    def test_policy_retention_is_clipped_to_configured_range(self) -> None:
        policy = FSRS6ADRDirectPolicy(
            coefficients=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            retention_min=0.7,
            retention_max=0.98,
        )

        self.assertGreaterEqual(policy.evaluate(0.1, 1.0), 0.7)
        self.assertLessEqual(policy.evaluate(100.0, 10.0), 0.98)

    def test_linear_policy_round_trips_feature_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDirectPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            loaded = FSRS6ADRDirectPolicy.from_json(policy_path)

        self.assertEqual(loaded.feature_version, SA_FEATURE_VERSION_LOG_LINEAR)
        self.assertEqual(loaded.feature_count, 3)
        self.assertEqual(len(loaded.coefficients), 3)

    def test_linear_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDirectPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            sa = FSRS6ADRDirectScheduler(policy_json=policy_path, fsrs_weights=None)

            fsrs_interval, _fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            sa_interval, _sa_state = sa.init_card(_view(None), 3, 0.0)

        self.assertAlmostEqual(sa_interval, fsrs_interval, places=9)

    def test_linear_vectorized_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDirectPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            fsrs = FSRS6VectorizedSchedulerOps(
                FSRS6Scheduler(weights=None, desired_retention=0.9),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            sa = FSRS6ADRDirectVectorizedSchedulerOps(
                FSRS6ADRDirectScheduler(policy_json=policy_path, fsrs_weights=None),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

            idx = torch.tensor([0])
            rating = torch.tensor([3])
            fsrs_state = fsrs.init_state(deck_size=1)
            sa_state = sa.init_state(deck_size=1)
            fsrs_interval = fsrs.update_learn(fsrs_state, idx, rating)
            sa_interval = sa.update_learn(sa_state, idx, rating)

        self.assertTrue(torch.allclose(sa_interval, fsrs_interval))

    def test_batch_ops_accept_per_user_coefficients(self) -> None:
        policy = FSRS6ADRDirectPolicy.baseline(desired_retention=0.9)
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [policy.coefficients[0] - 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = FSRS6ADRDirectBatchSchedulerOps(
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

    def test_batch_ops_accept_linear_per_user_coefficients(self) -> None:
        policy = FSRS6ADRDirectPolicy.baseline(
            desired_retention=0.9,
            retention_min=0.5,
            retention_max=0.98,
            feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [policy.coefficients[0] - 4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = FSRS6ADRDirectBatchSchedulerOps(
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

    def test_batch_ops_reject_invalid_linear_coefficients_shape(self) -> None:
        policy = FSRS6ADRDirectPolicy.baseline(
            desired_retention=0.9,
            feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])

        with self.assertRaisesRegex(ValueError, r"\(users, 3\)"):
            FSRS6ADRDirectBatchSchedulerOps(
                weights=weights,
                policy=policy,
                coefficients=torch.zeros((2, 6), dtype=torch.float32),
                bounds=Bounds(),
                priority_mode="low_retrievability",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

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


class FSRS6ADRDeltaSchedulerTests(unittest.TestCase):
    def test_zero_policy_reproduces_input_dr_across_states(self) -> None:
        policy = FSRS6ADRDeltaPolicy.baseline()

        for stability in (0.1, 2.5, 100.0):
            for difficulty in (1.0, 5.5, 10.0):
                for desired_retention in (0.50, 0.73, 0.98):
                    self.assertEqual(
                        policy.evaluate(stability, difficulty, desired_retention),
                        desired_retention,
                    )

    def test_linear_zero_policy_reproduces_input_dr_across_states(self) -> None:
        policy = FSRS6ADRDeltaPolicy.baseline(
            feature_version=DR_FEATURE_VERSION_LOG_LINEAR
        )

        self.assertEqual(policy.feature_count, 4)
        for stability in (0.1, 2.5, 100.0):
            for difficulty in (1.0, 5.5, 10.0):
                for desired_retention in (0.50, 0.73, 0.98):
                    self.assertEqual(
                        policy.evaluate(stability, difficulty, desired_retention),
                        desired_retention,
                    )

    def test_linear_policy_round_trips_feature_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDeltaPolicy.baseline(
                feature_version=DR_FEATURE_VERSION_LOG_LINEAR
            ).write_json(policy_path)

            loaded = FSRS6ADRDeltaPolicy.from_json(policy_path)

        self.assertEqual(loaded.feature_version, DR_FEATURE_VERSION_LOG_LINEAR)
        self.assertEqual(len(loaded.coefficients), 4)

    def test_zero_policy_matches_fsrs6_scheduler_for_same_dr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDeltaPolicy.baseline().write_json(policy_path)

            for desired_retention in (0.50, 0.90, 0.98):
                fsrs = FSRS6Scheduler(
                    weights=None,
                    desired_retention=desired_retention,
                )
                sa = FSRS6ADRDeltaScheduler(
                    policy_json=policy_path,
                    desired_retention=desired_retention,
                    fsrs_weights=None,
                )

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

    def test_linear_zero_policy_matches_fsrs6_scheduler_for_same_dr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDeltaPolicy.baseline(
                feature_version=DR_FEATURE_VERSION_LOG_LINEAR
            ).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.90)
            sa = FSRS6ADRDeltaScheduler(
                policy_json=policy_path,
                desired_retention=0.90,
                fsrs_weights=None,
            )

            fsrs_interval, _ = fsrs.init_card(_view(None), 3, 0.0)
            sa_interval, sa_state = sa.init_card(_view(None), 3, 0.0)

        self.assertAlmostEqual(sa_interval, fsrs_interval, places=9)
        self.assertEqual(set(sa_state), {"s", "d"})

    def test_linear_vectorized_zero_policy_matches_fsrs6_for_same_dr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRDeltaPolicy.baseline(
                feature_version=DR_FEATURE_VERSION_LOG_LINEAR
            ).write_json(policy_path)

            fsrs = FSRS6VectorizedSchedulerOps(
                FSRS6Scheduler(weights=None, desired_retention=0.90),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            sa = FSRS6ADRDeltaVectorizedSchedulerOps(
                FSRS6ADRDeltaScheduler(
                    policy_json=policy_path,
                    desired_retention=0.90,
                    fsrs_weights=None,
                ),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            fsrs_state = fsrs.init_state(deck_size=2)
            sa_state = sa.init_state(deck_size=2)
            idx = torch.tensor([0, 1])
            rating = torch.tensor([3, 4])

            fsrs_intervals = fsrs.update_learn(fsrs_state, idx, rating)
            sa_intervals = sa.update_learn(sa_state, idx, rating)

        self.assertTrue(torch.allclose(sa_intervals, fsrs_intervals))
        self.assertTrue(torch.allclose(sa_state.s, fsrs_state.s))
        self.assertTrue(torch.allclose(sa_state.d, fsrs_state.d))

    def test_scheduler_does_not_reference_env_memory_state(self) -> None:
        source = inspect.getsource(FSRS6ADRDeltaScheduler)
        self.assertNotIn("memory_state", source)

    def test_batch_ops_accept_per_lane_dr_and_coefficients(self) -> None:
        policy = FSRS6ADRDeltaPolicy.baseline()
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = FSRS6ADRDeltaBatchSchedulerOps(
            weights=weights,
            desired_retention=torch.tensor([0.80, 0.90], dtype=torch.float32),
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

        self.assertEqual(tuple(intervals.shape), (2,))
        self.assertGreater(float(intervals[0]), float(intervals[1]))

    def test_batch_ops_accept_linear_per_lane_dr_and_coefficients(self) -> None:
        policy = FSRS6ADRDeltaPolicy.baseline(
            feature_version=DR_FEATURE_VERSION_LOG_LINEAR
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = FSRS6ADRDeltaBatchSchedulerOps(
            weights=weights,
            desired_retention=torch.tensor([0.80, 0.90], dtype=torch.float32),
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

        self.assertEqual(tuple(intervals.shape), (2,))
        self.assertGreater(float(intervals[0]), float(intervals[1]))

    def test_batch_ops_reject_wrong_linear_coefficient_count(self) -> None:
        policy = FSRS6ADRDeltaPolicy.baseline(
            feature_version=DR_FEATURE_VERSION_LOG_LINEAR
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        with self.assertRaisesRegex(ValueError, r"users, 4"):
            FSRS6ADRDeltaBatchSchedulerOps(
                weights=weights,
                desired_retention=torch.tensor([0.80, 0.90], dtype=torch.float32),
                policy=policy,
                coefficients=torch.zeros((2, 10), dtype=torch.float32),
                bounds=Bounds(),
                priority_mode="low_retrievability",
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
