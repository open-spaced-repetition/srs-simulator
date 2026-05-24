from __future__ import annotations

from typing import Any, cast
import unittest

import torch

from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill_multiuser import (
    sample_batched_table_batch,
    sample_batched_teacher_occupancy_table_batch,
)


class _ToyGuide:
    policy = torch.tensor(
        [[[[0, 1], [2, 3]]]],
        dtype=torch.uint8,
    )

    def __init__(self, state_occupancy: torch.Tensor | None) -> None:
        self.state_occupancy = state_occupancy


class StationaryFiniteDistillSamplingTests(unittest.TestCase):
    def test_teacher_occupancy_sampling_uses_visited_state_distribution(self) -> None:
        guide = cast(
            Any,
            _ToyGuide(torch.tensor([[[0.0, 0.0, 5.0, 0.0]]], dtype=torch.float32)),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)

        obs, labels = sample_batched_table_batch(
            guide,
            cost_weights=[1024.0],
            samples_per_weight=6,
            supervision="teacher_occupancy",
            device=torch.device("cpu"),
            generator=generator,
        )

        self.assertTrue(torch.allclose(obs[0, :, 0], torch.ones(6)))
        self.assertTrue(torch.allclose(obs[0, :, 1], torch.zeros(6)))
        self.assertTrue(torch.equal(labels, torch.full((1, 6), 2, dtype=torch.int64)))

    def test_teacher_occupancy_sampling_requires_state_occupancy(self) -> None:
        guide = cast(Any, _ToyGuide(None))
        generator = torch.Generator(device="cpu")

        with self.assertRaisesRegex(ValueError, "state_occupancy"):
            sample_batched_teacher_occupancy_table_batch(
                guide,
                cost_weights=[1.0],
                samples_per_weight=1,
                device=torch.device("cpu"),
                generator=generator,
            )

    def test_uniform_sampling_still_uses_full_table(self) -> None:
        guide = cast(
            Any,
            _ToyGuide(torch.tensor([[[0.0, 0.0, 5.0, 0.0]]], dtype=torch.float32)),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)

        obs, labels = sample_batched_table_batch(
            guide,
            cost_weights=[1024.0],
            samples_per_weight=64,
            supervision="uniform_table",
            device=torch.device("cpu"),
            generator=generator,
        )

        self.assertEqual(tuple(obs.shape), (1, 64, 3))
        self.assertEqual(tuple(labels.shape), (1, 64))
        self.assertGreaterEqual(len(set(int(value) for value in labels[0].tolist())), 3)


if __name__ == "__main__":
    unittest.main()
