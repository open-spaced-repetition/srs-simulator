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
from simulator.fsrs6_adr_policy import (
    FEATURE_VERSION_LOG_LINEAR as SA_FEATURE_VERSION_LOG_LINEAR,
)
from simulator.fsrs6_adr_policy import (
    FEATURE_VERSION_LOG_POLY_TIME as SA_FEATURE_VERSION_LOG_POLY_TIME,
)
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.fsrs6_cost_conditioned_adr_policy import (
    ACTION_HEAD_INTERVAL,
    FSRS6CostConditionedADRPolicy,
)
from simulator.batched_engine.mixed_scheduler import (
    MixedBatchSchedulerOps,
    MixedSchedulerGroup,
)
from simulator.schedulers.fsrs import (
    FSRS3BatchSchedulerOps,
    FSRS6BatchSchedulerOps,
    FSRS6Scheduler,
    FSRS6BatchedSchedulerOps,
)
from simulator.schedulers.fsrs6_adr import (
    FSRS6ADRBatchSchedulerOps,
    FSRS6ADRScheduler,
    FSRS6ADRBatchedSchedulerOps,
)
from simulator.schedulers.fsrs6_cost_conditioned_adr import (
    FSRS6CostConditionedADRBatchSchedulerOps,
    FSRS6CostConditionedADRScheduler,
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


class FSRS6ADRSchedulerTests(unittest.TestCase):
    def test_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(desired_retention=0.9).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            adr = FSRS6ADRScheduler(policy_json=policy_path, fsrs_weights=None)

            fsrs_interval, fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            adr_interval, adr_state = adr.init_card(_view(None), 3, 0.0)
            self.assertAlmostEqual(adr_interval, fsrs_interval, places=9)
            self.assertEqual(set(adr_state), {"s", "d"})

            elapsed_values = [1.0, 4.0, 12.0, 2.0]
            ratings = [3, 4, 1, 2]
            for elapsed, rating in zip(elapsed_values, ratings):
                fsrs_interval, fsrs_state = fsrs.schedule(
                    _view(fsrs_state), rating, elapsed, elapsed
                )
                adr_interval, adr_state = adr.schedule(
                    _view(adr_state), rating, elapsed, elapsed
                )
                self.assertAlmostEqual(adr_interval, fsrs_interval, places=9)
                self.assertAlmostEqual(adr_state["s"], fsrs_state["s"], places=9)
                self.assertAlmostEqual(adr_state["d"], fsrs_state["d"], places=9)

    def test_scheduler_does_not_reference_env_memory_state(self) -> None:
        source = inspect.getsource(FSRS6ADRScheduler)
        self.assertNotIn("memory_state", source)

    def test_policy_retention_is_clipped_to_configured_range(self) -> None:
        policy = FSRS6ADRPolicy(
            coefficients=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            retention_min=0.7,
            retention_max=0.98,
        )

        self.assertGreaterEqual(policy.evaluate(0.1, 1.0), 0.7)
        self.assertLessEqual(policy.evaluate(100.0, 10.0), 0.98)

    def test_linear_policy_round_trips_feature_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            loaded = FSRS6ADRPolicy.from_json(policy_path)

        self.assertEqual(loaded.feature_version, SA_FEATURE_VERSION_LOG_LINEAR)
        self.assertEqual(loaded.feature_count, 3)
        self.assertEqual(len(loaded.coefficients), 3)

    def test_time_policy_round_trips_feature_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_POLY_TIME,
            ).write_json(policy_path)

            loaded = FSRS6ADRPolicy.from_json(policy_path)

        self.assertEqual(loaded.feature_version, SA_FEATURE_VERSION_LOG_POLY_TIME)
        self.assertEqual(loaded.feature_count, 10)
        self.assertEqual(len(loaded.coefficients), 10)

    def test_time_policy_uses_normalized_remaining_time_feature(self) -> None:
        policy = FSRS6ADRPolicy(
            coefficients=(0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            retention_min=0.5,
            retention_max=0.98,
            feature_version=SA_FEATURE_VERSION_LOG_POLY_TIME,
        )

        early = policy.evaluate(1.0, 5.0, remaining_time_norm=1.0)
        late = policy.evaluate(1.0, 5.0, remaining_time_norm=0.0)

        self.assertGreater(early, late)

    def test_policy_round_trips_null_baseline_desired_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            policy = FSRS6ADRPolicy(
                coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                baseline_desired_retention=None,
            )
            policy.write_json(policy_path)

            loaded = FSRS6ADRPolicy.from_json(policy_path)

        self.assertIsNone(loaded.baseline_desired_retention)

    def test_linear_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            adr = FSRS6ADRScheduler(policy_json=policy_path, fsrs_weights=None)

            fsrs_interval, _fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            adr_interval, _adr_state = adr.init_card(_view(None), 3, 0.0)

        self.assertAlmostEqual(adr_interval, fsrs_interval, places=9)

    def test_linear_batched_baseline_policy_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(
                desired_retention=0.9,
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
            ).write_json(policy_path)

            fsrs = FSRS6BatchedSchedulerOps(
                FSRS6Scheduler(weights=None, desired_retention=0.9),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            adr = FSRS6ADRBatchedSchedulerOps(
                FSRS6ADRScheduler(policy_json=policy_path, fsrs_weights=None),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

            idx = torch.tensor([0])
            rating = torch.tensor([3])
            fsrs_state = fsrs.init_state(deck_size=1)
            adr_state = adr.init_state(deck_size=1)
            fsrs_interval = fsrs.update_learn(fsrs_state, idx, rating)
            adr_interval = adr.update_learn(adr_state, idx, rating)

        self.assertTrue(torch.allclose(adr_interval, fsrs_interval))

    def test_event_time_policy_uses_simulation_day_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy(
                coefficients=(
                    0.0,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                retention_min=0.5,
                retention_max=0.98,
                feature_version=SA_FEATURE_VERSION_LOG_POLY_TIME,
            ).write_json(policy_path)
            scheduler = FSRS6ADRScheduler(
                policy_json=policy_path,
                fsrs_weights=None,
                simulation_days=10.0,
            )

            early_interval, _ = scheduler.init_card(_view(None), 3, 0.0)
            late_interval, _ = scheduler.init_card(_view(None), 3, 10.0)

        self.assertLess(early_interval, late_interval)

    def test_batch_ops_accept_per_user_coefficients(self) -> None:
        policy = FSRS6ADRPolicy.baseline(desired_retention=0.9)
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [
                list(policy.coefficients),
                [policy.coefficients[0] - 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ops = FSRS6ADRBatchSchedulerOps(
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


class FSRS6CostConditionedADRTests(unittest.TestCase):
    def test_retention_baseline_matches_fsrs6_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            FSRS6CostConditionedADRPolicy.baseline_retention(
                desired_retention=0.9,
            ).write_json(policy_path)

            fsrs = FSRS6Scheduler(weights=None, desired_retention=0.9)
            cost_adr = FSRS6CostConditionedADRScheduler(
                policy_json=policy_path,
                goal_cost_weight=1024.0,
                fsrs_weights=None,
            )

            fsrs_interval, _fsrs_state = fsrs.init_card(_view(None), 3, 0.0)
            adr_interval, _adr_state = cost_adr.init_card(_view(None), 3, 0.0)

        self.assertAlmostEqual(adr_interval, fsrs_interval, places=6)

    def test_retention_policy_is_anti_monotone_in_cost_weight(self) -> None:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=(
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -40.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -40.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            action_head="desired_retention",
            feature_version="fsrs6_cost_adr_retention_mono_v1",
        )

        low_cost = policy.evaluate_retention(10.0, 5.0, cost_weight=0.0)
        high_cost = policy.evaluate_retention(10.0, 5.0, cost_weight=1024.0)

        self.assertGreater(low_cost, high_cost)

    def test_interval_batch_ops_are_monotone_in_cost_weight(self) -> None:
        policy = FSRS6CostConditionedADRPolicy(
            coefficients=(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -40.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -40.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            action_head=ACTION_HEAD_INTERVAL,
            feature_version="fsrs6_cost_adr_interval_mono_v1",
        )
        ops = FSRS6CostConditionedADRBatchSchedulerOps(
            weights=torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS]),
            policy=policy,
            goal_cost_weight=torch.tensor([0.0, 1024.0]),
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

    def test_batch_ops_accept_time_per_user_coefficients(self) -> None:
        policy = FSRS6ADRPolicy(
            coefficients=(
                0.0,
                0.0,
                0.0,
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            retention_min=0.5,
            retention_max=0.98,
            feature_version=SA_FEATURE_VERSION_LOG_POLY_TIME,
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])
        coefficients = torch.tensor(
            [list(policy.coefficients), list(policy.coefficients)],
            dtype=torch.float32,
        )
        ops = FSRS6ADRBatchSchedulerOps(
            weights=weights,
            policy=policy,
            coefficients=coefficients,
            bounds=Bounds(),
            priority_mode="low_retrievability",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state = ops.init_state(user_count=2, deck_size=1)
        ops.set_time_context(current_day=0.0, simulation_days=10.0)
        early_interval = ops.update_learn(
            state,
            user_idx=torch.tensor([0]),
            card_idx=torch.tensor([0]),
            rating=torch.tensor([3]),
        )
        ops.set_time_context(current_day=10.0, simulation_days=10.0)
        late_interval = ops.update_learn(
            state,
            user_idx=torch.tensor([1]),
            card_idx=torch.tensor([0]),
            rating=torch.tensor([3]),
        )

        self.assertLess(float(early_interval[0]), float(late_interval[0]))

    def test_mixed_batch_scheduler_forwards_time_context(self) -> None:
        class DummyOps:
            def __init__(self) -> None:
                self.context: tuple[float, float] | None = None

            def init_state(self, user_count: int, deck_size: int):
                return (user_count, deck_size)

            def set_time_context(
                self, *, current_day: float, simulation_days: float
            ) -> None:
                self.context = (current_day, simulation_days)

        dummy = DummyOps()
        mixed = MixedBatchSchedulerOps(
            groups=[
                MixedSchedulerGroup(
                    lane_indices=torch.tensor([0], dtype=torch.int64),
                    ops=dummy,
                )
            ],
            lane_count=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        mixed.set_time_context(current_day=3.0, simulation_days=10.0)

        self.assertEqual(dummy.context, (3.0, 10.0))

    def test_batch_ops_accept_linear_per_user_coefficients(self) -> None:
        policy = FSRS6ADRPolicy.baseline(
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
        ops = FSRS6ADRBatchSchedulerOps(
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
        policy = FSRS6ADRPolicy.baseline(
            desired_retention=0.9,
            feature_version=SA_FEATURE_VERSION_LOG_LINEAR,
        )
        weights = torch.tensor([DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS])

        with self.assertRaisesRegex(ValueError, r"\(users, 3\)"):
            FSRS6ADRBatchSchedulerOps(
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


if __name__ == "__main__":
    unittest.main()
