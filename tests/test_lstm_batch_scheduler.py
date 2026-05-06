from __future__ import annotations

import unittest

import torch

from simulator.schedulers.lstm import LSTMBatchSchedulerOps
from tests.lstm_batch_helpers import dummy_lstm_weights


class LSTMBatchSchedulerOpsTests(unittest.TestCase):
    def _ops(
        self,
        *,
        desired_retention: float | torch.Tensor,
        n_users: int,
        interval_mode: str = "integer",
    ) -> LSTMBatchSchedulerOps:
        return LSTMBatchSchedulerOps(
            dummy_lstm_weights(n_users),
            desired_retention=desired_retention,
            max_interval=100.0,
            search_steps=24,
            interval_mode=interval_mode,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def _state_with_simple_curve(self, ops: LSTMBatchSchedulerOps, user_count: int):
        state = ops.init_state(user_count, deck_size=1)
        state.mem_w[:, 0, 0] = 1.0
        state.mem_s[:, 0, 0] = 10.0
        state.mem_d[:, 0, 0] = 1.0
        state.has_curves[:, 0] = True
        return state

    def test_tensor_desired_retention_indexes_selected_lanes(self) -> None:
        ops = self._ops(
            desired_retention=torch.tensor([0.90, 0.50, 0.80]),
            n_users=3,
        )
        state = self._state_with_simple_curve(ops, 3)
        user_idx = torch.tensor([2, 0], dtype=torch.int64)
        card_idx = torch.tensor([0, 0], dtype=torch.int64)

        intervals = ops._target_interval(state, user_idx, card_idx, None)

        self.assertEqual(intervals.tolist(), [3.0, 2.0])

    def test_higher_tensor_retention_target_produces_shorter_float_interval(
        self,
    ) -> None:
        ops = self._ops(
            desired_retention=torch.tensor([0.95, 0.70]),
            n_users=2,
            interval_mode="float",
        )
        state = self._state_with_simple_curve(ops, 2)
        user_idx = torch.tensor([0, 1], dtype=torch.int64)
        card_idx = torch.tensor([0, 0], dtype=torch.int64)

        intervals = ops._target_interval(state, user_idx, card_idx, None)

        self.assertLess(float(intervals[0].item()), float(intervals[1].item()))

    def test_rejects_invalid_tensor_desired_retention_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            self._ops(
                desired_retention=torch.tensor([0.80, 0.90]),
                n_users=3,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            self._ops(
                desired_retention=torch.tensor([[0.80, 0.90]]),
                n_users=2,
            )

    def test_rejects_out_of_range_desired_retention_values(self) -> None:
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\)"):
            self._ops(desired_retention=1.0, n_users=1)
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\)"):
            self._ops(
                desired_retention=torch.tensor([0.90, 0.0]),
                n_users=2,
            )


if __name__ == "__main__":
    unittest.main()
