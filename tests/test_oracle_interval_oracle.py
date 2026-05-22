from __future__ import annotations

import unittest

import torch

from experiments.single_card_tradeoff.core.tradeoff_runner import (
    _lookup_interval_policy_bilinear_action,
)
from experiments.single_card_tradeoff.oracles import (
    FSRS6BatchedContinuousStationaryFiniteOracle,
    FSRS6ContinuousRetentionOracle,
    FSRS6ContinuousStationaryFiniteOracle,
    FSRS6GridOracle,
    FSRS6IntervalOracle,
    attainable_interval_mask_for_retention_bounds,
    bilinear_retention_policy_lookup,
    retention_interval_float,
)
from experiments.single_card_tradeoff.oracles.dp_cache import OracleDPCacheConfig
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS


def _batched_continuous_oracle(
    *,
    user_count: int = 2,
    days: int = 5,
    interval_chunk_size: int = 2,
) -> FSRS6BatchedContinuousStationaryFiniteOracle:
    return FSRS6BatchedContinuousStationaryFiniteOracle(
        days=days,
        s_grid_size=8,
        d_grid_size=8,
        retention_min=0.5,
        retention_max=0.98,
        interval_chunk_size=interval_chunk_size,
        fsrs_weights=[DEFAULT_FSRS6_WEIGHTS for _ in range(user_count)],
        first_rating_prob=[DEFAULT_FIRST_RATING_PROB for _ in range(user_count)],
        review_rating_prob=[DEFAULT_REVIEW_RATING_PROB for _ in range(user_count)],
        learning_costs=[DEFAULT_STATE_RATING_COSTS.learning for _ in range(user_count)],
        review_costs=[DEFAULT_STATE_RATING_COSTS.review for _ in range(user_count)],
        device="cpu",
        cache_config=OracleDPCacheConfig(enabled=False),
    )


class FSRS6IntervalOracleTests(unittest.TestCase):
    def test_value_lookup_interpolates_log_s_and_linear_d(self) -> None:
        oracle = FSRS6IntervalOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            interval_chunk_size=2,
            device="cpu",
        )
        s_count = int(oracle.s_grid.numel())
        d_count = int(oracle.d_grid.numel())
        value = torch.empty(
            (oracle.horizon + 1, s_count, d_count, 2),
            device=oracle.device,
            dtype=oracle.dtype,
        )
        log_s = torch.log(oracle.s_grid)[:, None]
        d = oracle.d_grid[None, :]
        for rem in range(oracle.horizon + 1):
            value[rem, :, :, 0] = 10.0 * rem + 2.0 * log_s + 3.0 * d
            value[rem, :, :, 1] = -5.0 * rem - log_s + 0.25 * d

        query_s = torch.sqrt(oracle.s_grid[2] * oracle.s_grid[3]).reshape(1)
        query_d = (0.25 * oracle.d_grid[4] + 0.75 * oracle.d_grid[5]).reshape(1)
        rem_idx = torch.tensor([2], device=oracle.device, dtype=torch.int64)

        actual = oracle._interpolate_interval_value(
            value=value,
            rem_idx=rem_idx,
            s=query_s,
            d=query_d,
        )
        expected = torch.stack(
            (
                10.0 * rem_idx.to(dtype=oracle.dtype)
                + 2.0 * torch.log(query_s)
                + 3.0 * query_d,
                -5.0 * rem_idx.to(dtype=oracle.dtype)
                - torch.log(query_s)
                + 0.25 * query_d,
            ),
            dim=1,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=0.0))

    def test_interval_policy_cache_key_names_value_lookup_version(self) -> None:
        oracle = FSRS6IntervalOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            interval_chunk_size=2,
            device="cpu",
        )

        self.assertEqual(
            oracle._cache_extra()["transition_value_lookup"],
            FSRS6IntervalOracle.TRANSITION_VALUE_LOOKUP_VERSION,
        )

    def test_action_lookup_interpolates_interval_actions(self) -> None:
        oracle = FSRS6IntervalOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            interval_chunk_size=2,
            device="cpu",
        )
        policies = torch.ones(
            (1, oracle.horizon + 1, oracle.s_grid.numel(), oracle.d_grid.numel()),
            device=oracle.device,
            dtype=torch.int64,
        )
        policies[0, 3, 2, 4] = 1
        policies[0, 3, 3, 4] = 2
        policies[0, 3, 2, 5] = 4
        policies[0, 3, 3, 5] = 5

        interval = _lookup_interval_policy_bilinear_action(
            oracle=oracle,
            policies=policies,
            goal_indices=torch.tensor([0], device=oracle.device),
            remaining=torch.tensor([3], device=oracle.device),
            s=torch.sqrt(oracle.s_grid[2] * oracle.s_grid[3]).reshape(1),
            d=((oracle.d_grid[4] + oracle.d_grid[5]) * 0.5).reshape(1),
        )

        self.assertEqual(interval.item(), 3)

    def test_action_lookup_uses_exact_grid_value_and_clamps_to_remaining(self) -> None:
        oracle = FSRS6IntervalOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            interval_chunk_size=2,
            device="cpu",
        )
        policies = torch.ones(
            (1, oracle.horizon + 1, oracle.s_grid.numel(), oracle.d_grid.numel()),
            device=oracle.device,
            dtype=torch.int64,
        )
        policies[0, 3, 2, 4] = 4
        policies[0, 1, :, :] = 10

        exact = _lookup_interval_policy_bilinear_action(
            oracle=oracle,
            policies=policies,
            goal_indices=torch.tensor([0], device=oracle.device),
            remaining=torch.tensor([3], device=oracle.device),
            s=oracle.s_grid[2].reshape(1),
            d=oracle.d_grid[4].reshape(1),
        )
        clamped = _lookup_interval_policy_bilinear_action(
            oracle=oracle,
            policies=policies,
            goal_indices=torch.tensor([0], device=oracle.device),
            remaining=torch.tensor([1], device=oracle.device),
            s=torch.sqrt(oracle.s_grid[2] * oracle.s_grid[3]).reshape(1),
            d=((oracle.d_grid[4] + oracle.d_grid[5]) * 0.5).reshape(1),
        )

        self.assertEqual(exact.item(), 4)
        self.assertEqual(clamped.item(), 2)

    def test_continuous_bounds_attainable_mask_includes_terminal_surrogate(
        self,
    ) -> None:
        oracle = FSRS6ContinuousRetentionOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            retention_min=0.5,
            retention_max=0.5,
            interval_chunk_size=2,
            device="cpu",
            cache_config=OracleDPCacheConfig(enabled=False),
        )
        intervals = torch.arange(1, 5, device=oracle.device, dtype=torch.int64)
        mask = attainable_interval_mask_for_retention_bounds(
            intervals=intervals,
            s_grid=oracle.s_grid,
            retention_min=0.5,
            retention_max=0.5,
            factor=oracle.factor,
            decay=oracle.decay,
            terminal_interval=4,
        )
        rounded = torch.clamp(
            torch.round(
                retention_interval_float(
                    s=oracle.s_grid,
                    retention=0.5,
                    factor=oracle.factor,
                    decay=oracle.decay,
                )
            ),
            min=1.0,
        ).to(torch.int64)

        for s_idx, interval in enumerate(rounded.tolist()):
            expected = torch.zeros(4, dtype=torch.bool)
            if interval >= 4:
                expected[3] = True
            else:
                expected[int(interval) - 1] = True
            self.assertTrue(torch.equal(mask[:, s_idx].cpu(), expected))

    def test_single_retention_continuous_policy_matches_discrete_interval(
        self,
    ) -> None:
        cache_config = OracleDPCacheConfig(enabled=False)
        discrete = FSRS6GridOracle(
            days=6,
            action_retentions=[0.8],
            s_grid_size=8,
            d_grid_size=8,
            device="cpu",
            cache_config=cache_config,
        )
        continuous = FSRS6ContinuousRetentionOracle(
            days=6,
            s_grid_size=8,
            d_grid_size=8,
            retention_min=0.8,
            retention_max=0.8,
            interval_chunk_size=2,
            device="cpu",
            cache_config=cache_config,
        )
        policy = continuous.solve_policies([1.0], progress=False)[0]
        interval_from_retention = torch.clamp(
            torch.round(
                retention_interval_float(
                    s=continuous.s_mesh,
                    retention=policy[continuous.horizon],
                    factor=continuous.factor,
                    decay=continuous.decay,
                )
            ),
            min=1.0,
        ).to(torch.int64)

        self.assertTrue(
            torch.equal(
                interval_from_retention,
                discrete.transitions[0]
                .interval[:, None]
                .expand_as(interval_from_retention),
            )
        )

    def test_continuous_stationary_finite_outputs_retention_table(self) -> None:
        oracle = FSRS6ContinuousStationaryFiniteOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            retention_min=0.5,
            retention_max=0.98,
            interval_chunk_size=2,
            device="cpu",
            cache_config=OracleDPCacheConfig(enabled=False),
        )
        solution = oracle.solve_stationary_finite_policies(
            [0.0, 16.0],
            max_iterations=2,
            tolerance=1e-8,
            progress=False,
        )

        self.assertEqual(solution.policy.shape, (2, 8, 8))
        self.assertTrue(torch.all(solution.policy >= 0.5))
        self.assertTrue(torch.all(solution.policy <= 0.98))
        self.assertEqual(len(solution.converged), 2)

    def test_batched_continuous_stationary_finite_outputs_user_retention_table(
        self,
    ) -> None:
        oracle = _batched_continuous_oracle(days=4)
        solution = oracle.solve_stationary_finite_policies(
            [0.0, 16.0],
            max_iterations=2,
            tolerance=1e-8,
            progress=False,
        )

        self.assertEqual(solution.policy.shape, (2, 2, 8, 8))
        self.assertTrue(torch.all(solution.policy >= 0.5))
        self.assertTrue(torch.all(solution.policy <= 0.98))
        self.assertTrue(torch.allclose(solution.policy[0], solution.policy[1]))
        self.assertEqual(len(solution.converged), 2)
        self.assertEqual(len(solution.converged[0]), 2)

    def test_active_cell_attainable_mask_matches_grid_mask(self) -> None:
        oracle = _batched_continuous_oracle(days=5)
        intervals = torch.arange(1, oracle.horizon + 2, device=oracle.device)
        user_idx = torch.tensor([0, 1, 1], device=oracle.device, dtype=torch.int64)
        state_idx = torch.tensor([0, 9, 63], device=oracle.device, dtype=torch.int64)
        actual = oracle._active_cell_attainable_interval_mask(
            intervals=intervals,
            user_idx=user_idx,
            state_idx=state_idx,
        )
        full = oracle._attainable_interval_mask(intervals, oracle.horizon + 1)
        s_idx = torch.div(state_idx, oracle.d_count, rounding_mode="floor")
        expected = torch.stack(
            [full[user, :, s] for user, s in zip(user_idx, s_idx, strict=True)],
            dim=0,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_active_cell_immediate_prefix_scores_match_rem_loop(self) -> None:
        oracle = _batched_continuous_oracle(days=6)
        user_idx = torch.tensor([0, 1], device=oracle.device, dtype=torch.int64)
        state_idx = torch.tensor([3, 20], device=oracle.device, dtype=torch.int64)
        block_occupancy = torch.zeros(
            (2, 2, oracle.horizon + 1),
            device=oracle.device,
            dtype=oracle.dtype,
        )
        block_occupancy[0, 0, 1:] = torch.tensor(
            [0.2, 0.3, 0.0, 0.4, 0.1],
            device=oracle.device,
            dtype=oracle.dtype,
        )
        block_occupancy[0, 1, 2:5] = torch.tensor(
            [0.5, 0.25, 0.25],
            device=oracle.device,
            dtype=oracle.dtype,
        )
        block_occupancy[1, :, 1:] = torch.tensor(
            [
                [0.1, 0.0, 0.2, 0.0, 0.3],
                [0.0, 0.6, 0.0, 0.2, 0.1],
            ],
            device=oracle.device,
            dtype=oracle.dtype,
        )
        intervals = torch.tensor(
            [1, 2, 4, oracle.horizon + 1],
            device=oracle.device,
            dtype=torch.int64,
        )
        tables = oracle._active_cell_immediate_prefix_tables(
            block_occupancy=block_occupancy,
            user_idx=user_idx,
            state_idx=state_idx,
        )
        actual = oracle._active_cell_immediate_scores_for_intervals(
            intervals=intervals,
            memorized=tables[0],
            prefix_weighted=tables[1],
            prefix_occupancy=tables[2],
            total_weighted=tables[3],
            total_occupancy=tables[4],
        )
        expected = torch.zeros_like(actual)
        s_idx = torch.div(state_idx, oracle.d_count, rounding_mode="floor")
        for block_idx in range(int(user_idx.numel())):
            for weight_idx in range(2):
                for interval_pos, interval in enumerate(intervals.tolist()):
                    total = torch.tensor(0.0, device=oracle.device, dtype=oracle.dtype)
                    for rem in range(1, oracle.horizon + 1):
                        day = (
                            rem
                            if interval == oracle.horizon + 1
                            else min(rem, interval)
                        )
                        total += (
                            block_occupancy[block_idx, weight_idx, rem]
                            * oracle.memorized_by_day[
                                user_idx[block_idx],
                                day,
                                s_idx[block_idx],
                            ]
                        )
                    expected[block_idx, weight_idx, interval_pos] = total

        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0.0))

    def test_optimized_batched_improvement_matches_dense_reference(self) -> None:
        oracle = _batched_continuous_oracle(days=5)
        cost_weights = torch.tensor(
            [0.0, 4.0], device=oracle.device, dtype=oracle.dtype
        )
        policy = torch.full(
            (2, 2, oracle.s_count, oracle.d_count),
            0.8,
            device=oracle.device,
            dtype=oracle.dtype,
        )
        value = oracle._evaluate_stationary_policy_value_batch(
            policy=policy,
            cost_weights=cost_weights,
        )
        occupancy = oracle._rollout_occupancy_batch(policy=policy)

        optimized = oracle._improve_stationary_policy_batch(
            policy=policy,
            occupancy=occupancy,
            value=value,
            cost_weights=cost_weights,
        )
        reference = oracle._improve_stationary_policy_batch_dense_reference(
            policy=policy,
            occupancy=occupancy,
            value=value,
            cost_weights=cost_weights,
        )

        self.assertTrue(
            torch.allclose(optimized[0], reference[0], atol=1e-12, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(optimized[1], reference[1], atol=1e-10, rtol=0.0)
        )
        self.assertTrue(torch.equal(optimized[2], reference[2]))

        active = torch.tensor(
            [[True, False], [False, True]],
            device=oracle.device,
            dtype=torch.bool,
        )
        optimized_active = oracle._improve_stationary_policy_batch(
            policy=policy,
            occupancy=occupancy,
            value=value,
            cost_weights=cost_weights,
            active=active,
        )
        reference_active = oracle._improve_stationary_policy_batch_dense_reference(
            policy=policy,
            occupancy=occupancy,
            value=value,
            cost_weights=cost_weights,
            active=active,
        )
        self.assertTrue(
            torch.allclose(
                optimized_active[0],
                reference_active[0],
                atol=1e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                optimized_active[1],
                reference_active[1],
                atol=1e-10,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.equal(optimized_active[2], reference_active[2]))

    def test_optimized_batched_improvement_preserves_unvisited_states(self) -> None:
        oracle = _batched_continuous_oracle(user_count=1, days=5)
        cost_weights = torch.tensor([0.0], device=oracle.device, dtype=oracle.dtype)
        policy = torch.full(
            (1, 1, oracle.s_count, oracle.d_count),
            0.72,
            device=oracle.device,
            dtype=oracle.dtype,
        )
        value = torch.zeros(
            (1, 1, oracle.horizon + 1, oracle.state_count),
            device=oracle.device,
            dtype=oracle.dtype,
        )
        occupancy = torch.zeros_like(value)
        occupancy[0, 0, oracle.horizon, 0] = 1.0

        new_policy, _, visited = oracle._improve_stationary_policy_batch(
            policy=policy,
            occupancy=occupancy,
            value=value,
            cost_weights=cost_weights,
        )

        self.assertTrue(torch.equal(new_policy[~visited], policy[~visited]))

    def test_bilinear_retention_lookup_interpolates_and_clamps(self) -> None:
        oracle = FSRS6ContinuousRetentionOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            retention_min=0.5,
            retention_max=0.98,
            interval_chunk_size=2,
            device="cpu",
            cache_config=OracleDPCacheConfig(enabled=False),
        )
        policies = torch.full(
            (1, oracle.horizon + 1, oracle.s_grid.numel(), oracle.d_grid.numel()),
            0.9,
            device=oracle.device,
            dtype=oracle.dtype,
        )
        policies[0, 3, 2, 4] = 0.4
        policies[0, 3, 3, 4] = 0.6
        policies[0, 3, 2, 5] = 1.1
        policies[0, 3, 3, 5] = 0.8

        retention = bilinear_retention_policy_lookup(
            oracle=oracle,
            policies=policies,
            goal_indices=torch.tensor([0], device=oracle.device),
            remaining=torch.tensor([3], device=oracle.device),
            s=torch.sqrt(oracle.s_grid[2] * oracle.s_grid[3]).reshape(1),
            d=((oracle.d_grid[4] + oracle.d_grid[5]) * 0.5).reshape(1),
            retention_min=0.5,
            retention_max=0.98,
        )

        self.assertAlmostEqual(float(retention.item()), 0.725, places=12)

    def test_continuous_cache_key_names_bounds_and_lookup_versions(self) -> None:
        oracle = FSRS6ContinuousStationaryFiniteOracle(
            days=4,
            s_grid_size=8,
            d_grid_size=8,
            retention_min=0.5,
            retention_max=0.98,
            interval_chunk_size=2,
            device="cpu",
            cache_config=OracleDPCacheConfig(enabled=False),
        )
        extra = oracle._stationary_cache_extra(max_iterations=3, tolerance=1e-8)

        self.assertEqual(extra["retention_min"], 0.5)
        self.assertEqual(extra["retention_max"], 0.98)
        self.assertEqual(extra["interval_chunk_size"], 2)
        self.assertEqual(
            extra["transition_value_lookup"],
            FSRS6ContinuousRetentionOracle.TRANSITION_VALUE_LOOKUP_VERSION,
        )
        self.assertEqual(
            extra["action_policy_lookup"],
            FSRS6ContinuousRetentionOracle.ACTION_POLICY_LOOKUP_VERSION,
        )
        self.assertEqual(
            extra["policy_iteration"],
            FSRS6ContinuousStationaryFiniteOracle.STATIONARY_POLICY_ITERATION_VERSION,
        )
        self.assertEqual(extra["policy_iteration"], "continuous_interval_greedy_v2")

        batched = _batched_continuous_oracle(days=4)
        self.assertNotIn("policy_iteration", batched._continuous_cache_extra())
        self.assertEqual(
            batched._stationary_cache_extra(
                max_iterations=3,
                tolerance=1e-8,
            )["policy_iteration"],
            "continuous_interval_greedy_v2",
        )


if __name__ == "__main__":
    unittest.main()
