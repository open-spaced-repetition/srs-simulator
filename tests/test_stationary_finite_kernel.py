from __future__ import annotations

import unittest

import torch

from experiments.single_card_tradeoff.oracles import FSRS6StationaryFiniteOracle


class StationaryFiniteKernelTests(unittest.TestCase):
    def test_state_kernel_interpolates_log_s_and_linear_d(self) -> None:
        oracle = FSRS6StationaryFiniteOracle(
            days=4,
            action_retentions=[0.9],
            s_grid_size=8,
            d_grid_size=8,
            device="cpu",
        )
        log_s = torch.log(oracle.s_grid)[:, None]
        d = oracle.d_grid[None, :]
        value = (2.0 * log_s + 3.0 * d).reshape(-1)

        query_s = torch.sqrt(oracle.s_grid[2] * oracle.s_grid[3])
        query_d = 0.25 * oracle.d_grid[4] + 0.75 * oracle.d_grid[5]
        state_idx, state_weight = oracle._state_kernel(query_s, query_d)

        actual = (state_weight * value[state_idx]).sum()
        expected = 2.0 * torch.log(query_s) + 3.0 * query_d

        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                state_weight.sum(),
                torch.tensor(1.0, dtype=oracle.dtype),
                atol=1e-12,
                rtol=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
