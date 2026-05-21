from __future__ import annotations

import unittest

import torch

from experiments.single_card_tradeoff.oracles import FSRS6IntervalOracle


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


if __name__ == "__main__":
    unittest.main()
