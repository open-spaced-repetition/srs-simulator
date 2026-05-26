from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from experiments.single_card_tradeoff.oracles.dp_cache import (
    OracleDPCacheConfig,
    oracle_dp_cache_stats_snapshot,
    reset_oracle_dp_cache_stats,
)
from experiments.single_card_tradeoff.oracles import (
    FSRS6AverageRewardOracle,
    FSRS6BatchedContinuousStationaryFiniteOracle,
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


def _continuous_batched_oracle(
    *,
    cache_config: OracleDPCacheConfig,
) -> FSRS6BatchedContinuousStationaryFiniteOracle:
    return FSRS6BatchedContinuousStationaryFiniteOracle(
        days=8,
        s_grid_size=8,
        d_grid_size=8,
        retention_min=0.5,
        retention_max=0.98,
        interval_chunk_size=4,
        fsrs_weights=[DEFAULT_FSRS6_WEIGHTS],
        first_rating_prob=[DEFAULT_FIRST_RATING_PROB],
        review_rating_prob=[DEFAULT_REVIEW_RATING_PROB],
        learning_costs=[DEFAULT_STATE_RATING_COSTS.learning],
        review_costs=[DEFAULT_STATE_RATING_COSTS.review],
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

    def test_grid_cache_key_names_transition_kernel(self) -> None:
        oracle = FSRS6GridOracle(
            days=8,
            action_retentions=[0.5, 0.8],
            s_grid_size=8,
            d_grid_size=8,
            device="cpu",
            cache_config=OracleDPCacheConfig(enabled=False),
        )

        key_parts = oracle._cache_key_parts(
            oracle_kind="grid",
            method="solve_policies",
            cost_weight=1.0,
            extra=oracle._grid_cache_extra(),
        )

        self.assertEqual(
            key_parts["extra"]["transition_kernel"],
            FSRS6GridOracle.TRANSITION_KERNEL_VERSION,
        )

    def test_stationary_seed_reuses_grid_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            grid_oracle = FSRS6GridOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )
            stationary_oracle = FSRS6StationaryFiniteOracle(
                days=8,
                action_retentions=[0.5, 0.8],
                s_grid_size=8,
                d_grid_size=8,
                device="cpu",
                cache_config=config,
            )

            grid_oracle.solve_policies([0.0])
            reset_oracle_dp_cache_stats()
            stationary_oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=4,
                tolerance=1e-8,
            )

            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 1,
                    "misses": 1,
                    "writes": 1,
                    "refreshes": 0,
                },
            )

    def test_grid_exact_value_bounds_stationary_finite_value(self) -> None:
        config = OracleDPCacheConfig(enabled=False)
        grid_oracle = FSRS6GridOracle(
            days=8,
            action_retentions=[0.5, 0.8],
            s_grid_size=8,
            d_grid_size=8,
            device="cpu",
            cache_config=config,
        )
        stationary_oracle = FSRS6StationaryFiniteOracle(
            days=8,
            action_retentions=[0.5, 0.8],
            s_grid_size=8,
            d_grid_size=8,
            device="cpu",
            cache_config=config,
        )

        grid_metric = grid_oracle.estimate_many([1.0])[0]
        stationary_solution = stationary_oracle.solve_stationary_finite_policies(
            [1.0],
            max_iterations=8,
            tolerance=1e-8,
        )

        self.assertGreaterEqual(
            grid_metric.scalar_objective + 1e-10,
            float(stationary_solution.objectives[0].item()),
        )

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

    def test_batched_user_weight_policy_matches_shared_diagonal(self) -> None:
        oracle = _batched_oracle(
            cache_config=OracleDPCacheConfig(enabled=False),
            user_count=2,
        )

        shared = oracle.solve_policies([0.0, 1.0])
        jagged = oracle.solve_policies_for_user_weights([0.0, 1.0])

        self.assertEqual(jagged.shape, (2, 1, 8, 64))
        self.assertTrue(torch.equal(jagged[0, 0], shared[0, 0]))
        self.assertTrue(torch.equal(jagged[1, 0], shared[1, 1]))

    def test_batched_user_weight_stationary_matches_shared_diagonal(self) -> None:
        oracle = _batched_oracle(
            cache_config=OracleDPCacheConfig(enabled=False),
            user_count=2,
        )

        shared = oracle.solve_stationary_finite_policies(
            [0.0, 1.0],
            max_iterations=2,
            tolerance=1e9,
        )
        jagged = oracle.solve_stationary_finite_policies_for_user_weights(
            [0.0, 1.0],
            max_iterations=2,
            tolerance=1e9,
        )

        self.assertEqual(jagged.policy.shape, (2, 1, 8, 8))
        self.assertTrue(torch.equal(jagged.policy[0, 0], shared.policy[0, 0]))
        self.assertTrue(torch.equal(jagged.policy[1, 0], shared.policy[1, 1]))
        self.assertAlmostEqual(
            jagged.metrics[0][0].card_expected_retrievability,
            shared.metrics[0][0].card_expected_retrievability,
        )
        self.assertAlmostEqual(
            jagged.metrics[1][0].card_minutes_per_day,
            shared.metrics[1][1].card_minutes_per_day,
        )

    def test_batched_user_weight_policy_cache_reuses_shared_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            oracle = _batched_oracle(cache_config=config, user_count=2)
            shared = oracle.solve_policies([0.0, 1.0])

            reset_oracle_dp_cache_stats()
            jagged = oracle.solve_policies_for_user_weights([0.0, 1.0])

            self.assertTrue(torch.equal(jagged[0, 0], shared[0, 0]))
            self.assertTrue(torch.equal(jagged[1, 0], shared[1, 1]))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 2,
                    "misses": 0,
                    "writes": 0,
                    "refreshes": 0,
                },
            )

    def test_batched_user_weight_policy_validates_inputs(self) -> None:
        oracle = _batched_oracle(
            cache_config=OracleDPCacheConfig(enabled=False),
            user_count=2,
        )

        for weights in ([], [0.0], [0.0, -1.0], [0.0, float("nan")]):
            with self.assertRaises(ValueError):
                oracle.solve_policies_for_user_weights(weights)

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

    def test_continuous_batched_stationary_writes_finite_seed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _cache_config(Path(tmp))
            oracle = _continuous_batched_oracle(cache_config=config)

            reset_oracle_dp_cache_stats()
            solution = oracle.solve_stationary_finite_policies(
                [0.0],
                max_iterations=2,
                tolerance=1e9,
            )
            self.assertEqual(solution.policy.shape, (1, 1, 8, 8))
            self.assertEqual(
                oracle_dp_cache_stats_snapshot(),
                {
                    "hits": 0,
                    "misses": 2,
                    "writes": 2,
                    "refreshes": 0,
                },
            )

            reset_oracle_dp_cache_stats()
            finite = oracle.solve_policies([0.0])
            self.assertEqual(finite.shape, (1, 1, 8, 8, 8))
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
