from __future__ import annotations

import math
import unittest

import torch

from simulator import simulate
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.batched_engine.multiuser_types import MultiUserBehavior, MultiUserCost
from simulator.behavior import StochasticBehavior
from simulator.cost import StatefulCostModel, StateRatingCosts
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps, FSRS6Model
from simulator.retention_sweep.no_review import (
    build_batched_no_review_retention_kernels,
    build_no_review_retention_kernel,
    calculate_no_review_memory_series,
)
from simulator.schedulers.fixed import FixedBatchSchedulerOps, FixedIntervalScheduler


class NoReviewBaselineTests(unittest.TestCase):
    def test_batched_fsrs6_kernel_matches_scalar_model(self) -> None:
        probabilities = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS],
            dtype=torch.float32,
        )
        env_ops = FSRS6BatchEnvOps(
            weights=weights,
            bounds=Bounds(),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        batched = build_batched_no_review_retention_kernels(
            env_ops,
            probabilities,
            days=30,
        )

        for user_index in range(2):
            scalar = build_no_review_retention_kernel(
                FSRS6Model(DEFAULT_FSRS6_WEIGHTS),
                probabilities[user_index].tolist(),
                days=30,
            )
            for actual, expected in zip(
                batched[user_index],
                scalar,
                strict=True,
            ):
                self.assertAlmostEqual(actual, expected, places=6)

    def test_no_review_policy_has_zero_review_memory_gain(self) -> None:
        first_rating_prob = [0.0, 0.0, 1.0, 0.0]
        environment = FSRS6Model(DEFAULT_FSRS6_WEIGHTS)
        stats = simulate(
            days=10,
            deck_size=1,
            environment=environment,
            scheduler=FixedIntervalScheduler(100.0),
            behavior=StochasticBehavior(
                max_new_per_day=1,
                max_reviews_per_day=10,
                first_rating_prob=first_rating_prob,
            ),
            cost_model=StatefulCostModel(),
            seed_fn=lambda: 0.5,
            progress=False,
        )
        kernel = build_no_review_retention_kernel(
            environment,
            first_rating_prob,
            days=10,
        )

        series = calculate_no_review_memory_series(
            daily_memorized=stats.daily_memorized,
            daily_new=stats.daily_new,
            deck_size=1,
            first_rating_prob=first_rating_prob,
            no_review_retention_kernel=kernel,
        )

        for gain in series.review_memory_gain:
            self.assertAlmostEqual(gain, 0.0, places=10)

    def test_event_engine_splits_learning_and_review_costs(self) -> None:
        stats = simulate(
            days=3,
            deck_size=1,
            environment=FSRS6Model(DEFAULT_FSRS6_WEIGHTS),
            scheduler=FixedIntervalScheduler(1.0),
            behavior=StochasticBehavior(
                max_new_per_day=1,
                max_reviews_per_day=10,
                first_rating_prob=[0.0, 0.0, 1.0, 0.0],
            ),
            cost_model=StatefulCostModel(
                state_costs=StateRatingCosts(
                    learning=[10.0] * 4,
                    review=[5.0] * 4,
                    relearning=[5.0] * 4,
                )
            ),
            seed_fn=lambda: 0.5,
            progress=False,
        )

        assert stats.daily_learning_cost is not None
        assert stats.daily_review_cost is not None
        self.assertEqual(stats.daily_learning_cost, [10.0, 0.0, 0.0])
        self.assertEqual(stats.daily_review_cost, [0.0, 5.0, 5.0])
        self.assertEqual(
            stats.daily_cost,
            [
                learning + review
                for learning, review in zip(
                    stats.daily_learning_cost,
                    stats.daily_review_cost,
                    strict=True,
                )
            ],
        )

    def test_batched_engine_splits_learning_and_review_costs(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        env_ops = FSRS6BatchEnvOps(
            weights=torch.tensor([DEFAULT_FSRS6_WEIGHTS], dtype=dtype),
            bounds=Bounds(),
            device=device,
            dtype=dtype,
        )
        sched_ops = FixedBatchSchedulerOps(
            interval=1.0,
            device=device,
            dtype=dtype,
        )
        behavior = MultiUserBehavior(
            attendance_prob=torch.ones(1),
            lazy_good_bias=torch.zeros(1),
            max_new_per_day=torch.ones(1, dtype=torch.int64),
            max_reviews_per_day=torch.full((1,), 10, dtype=torch.int64),
            max_cost_per_day=torch.full((1,), math.inf),
            success_weights=torch.tensor([[0.0, 0.0, 1.0]]),
            learning_success_weights=torch.tensor([[0.0, 0.0, 1.0]]),
            relearning_success_weights=torch.tensor([[0.0, 0.0, 1.0]]),
            first_rating_prob=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        )
        cost_model = MultiUserCost(
            base=torch.zeros(1),
            penalty=torch.zeros(1),
            learn_costs=torch.full((1, 4), 10.0),
            review_costs=torch.full((1, 4), 5.0),
            learning_review_costs=torch.full((1, 4), 5.0),
            relearning_review_costs=torch.full((1, 4), 5.0),
        )

        stats = simulate_multiuser(
            days=3,
            deck_size=1,
            env_ops=env_ops,
            sched_ops=sched_ops,
            behavior=behavior,
            cost_model=cost_model,
            seed=42,
            device=device,
            dtype=dtype,
            progress=False,
        )[0]

        assert stats.daily_learning_cost is not None
        assert stats.daily_review_cost is not None
        self.assertEqual(stats.daily_learning_cost, [10.0, 0.0, 0.0])
        self.assertEqual(stats.daily_review_cost, [0.0, 5.0, 5.0])
        self.assertEqual(
            stats.daily_cost,
            [
                learning + review
                for learning, review in zip(
                    stats.daily_learning_cost,
                    stats.daily_review_cost,
                    strict=True,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
