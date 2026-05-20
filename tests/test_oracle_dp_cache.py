from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from experiments.single_card_tradeoff.oracle_dp_cache import (
    OracleDPCacheConfig,
    oracle_dp_cache_stats_snapshot,
    reset_oracle_dp_cache_stats,
)
from experiments.single_card_tradeoff.oracle_frontier import (
    FSRS6AverageRewardOracle,
    FSRS6BatchedStationaryFiniteOracle,
    FSRS6GridOracle,
    FSRS6IntervalOracle,
    FSRS6StationaryFiniteOracle,
)
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS


def _cache_config(root: Path, *, refresh: bool = False) -> OracleDPCacheConfig:
    return OracleDPCacheConfig(cache_dir=root, refresh=refresh)


def _batched_oracle(
    *,
    cache_config: OracleDPCacheConfig,
    user_count: int = 2,
) -> FSRS6BatchedStationaryFiniteOracle:
    weights = [DEFAULT_FSRS6_WEIGHTS for _ in range(user_count)]
    first_prob = [DEFAULT_FIRST_RATING_PROB for _ in range(user_count)]
    review_prob = [DEFAULT_REVIEW_RATING_PROB for _ in range(user_count)]
    learning_costs = [DEFAULT_STATE_RATING_COSTS.learning for _ in range(user_count)]
    review_costs = [DEFAULT_STATE_RATING_COSTS.review for _ in range(user_count)]
    return FSRS6BatchedStationaryFiniteOracle(
        days=8,
        action_retentions=[0.5, 0.8],
        s_grid_size=8,
        d_grid_size=8,
        fsrs_weights=weights,
        first_rating_prob=first_prob,
        review_rating_prob=review_prob,
        learning_costs=learning_costs,
        review_costs=review_costs,
        device="cpu",
        cache_config=cache_config,
    )


class OracleDpCacheTest(unittest.TestCase):
    def test_grid_cache_round_trips_per_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            oracle = FSRS6GridOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )
            reset_oracle_dp_cache_stats()
            first = oracle.solve_policies([0.0, 1.0])
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 0,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )

            second = oracle.solve_policies([1.0, 0.0])
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 2,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )
            self.assertTrue(torch.equal(first[0], second[1]))
            self.assertTrue(torch.equal(first[1], second[0]))

    def test_stationary_cache_refreshes_and_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            oracle = FSRS6StationaryFiniteOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )
            reset_oracle_dp_cache_stats()
            first = oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            self.assertEqual(first.policy.shape, (1, 8, 8))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 0,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )

            second = oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            self.assertTrue(torch.equal(first.policy, second.policy))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 1,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )

            refresh_oracle = FSRS6StationaryFiniteOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=_cache_config(Path(tmp), refresh=True),
            )
            reset_oracle_dp_cache_stats()
            refreshed = refresh_oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            self.assertTrue(torch.equal(first.policy, refreshed.policy))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 0,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 2,
                },
            )

    def test_average_reward_and_interval_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            avg_oracle = FSRS6AverageRewardOracle(
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )
            interval_oracle = FSRS6IntervalOracle(
                days=8,
                s_grid_size=8,
                d_grid_size=8,
                interval_chunk_size=4,
                device="cpu",
                cache_config=config,
            )

            reset_oracle_dp_cache_stats()
            avg_first = avg_oracle.solve_average_reward_policies(
                [1.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            avg_second = avg_oracle.solve_average_reward_policies(
                [1.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            self.assertTrue(torch.equal(avg_first.policy, avg_second.policy))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 1,
                    "misses": 1,
                    "writes": 1,
                    "refreshes": 0,
                },
            )

            reset_oracle_dp_cache_stats()
            interval_first = interval_oracle.solve_policies([1.0, 2.0])
            interval_second = interval_oracle.solve_policies([2.0, 1.0])
            self.assertTrue(torch.equal(interval_first[0], interval_second[1]))
            self.assertTrue(torch.equal(interval_first[1], interval_second[0]))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 2,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )

    def test_batched_cache_reuses_user_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            full_oracle = _batched_oracle(cache_config=config, user_count=2)
            reset_oracle_dp_cache_stats()
            full = full_oracle.solve_policies([0.0, 1.0])
            self.assertEqual(full.shape, (2, 2, 8, 64))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 0,
                    "misses": 4,
                    "writes": 4,
                    "refreshes": 0,
                },
            )

            subset_oracle = _batched_oracle(cache_config=config, user_count=1)
            reset_oracle_dp_cache_stats()
            subset = subset_oracle.solve_policies([1.0])
            self.assertEqual(subset.shape, (1, 1, 8, 64))
            self.assertTrue(torch.equal(full[0, 1], subset[0, 0]))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 1,
                    "misses": 0,
                    "writes": 0,
                    "refreshes": 0,
                },
            )

    def test_batched_stationary_cache_reuses_single_user_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            single_oracle = FSRS6StationaryFiniteOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )
            single = single_oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )

            batched_oracle = _batched_oracle(cache_config=config, user_count=1)
            reset_oracle_dp_cache_stats()
            batched = batched_oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )
            self.assertEqual(batched.policy.shape, (1, 1, 8, 8))
            self.assertTrue(torch.equal(single.policy[0], batched.policy[0, 0]))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 1,
                    "misses": 0,
                    "writes": 0,
                    "refreshes": 0,
                },
            )


if __name__ == "__main__":
    unittest.main()
