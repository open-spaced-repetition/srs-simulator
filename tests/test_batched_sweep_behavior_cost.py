from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage


def _button_usage_path(root: Path) -> Path:
    path = root / "button_usage.jsonl"
    entry = {
        "user": 1,
        "learn_costs": [1.0, 2.0, 3.0, 4.0],
        "review_costs": [5.0, 6.0, 7.0, 8.0],
        "first_rating_prob": [0.1, 0.2, 0.3, 0.4],
        "review_rating_prob": [0.2, 0.3, 0.5],
        "state_rating_costs": [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        "first_rating_offset": [0.0, 0.0, 0.0, 0.0],
        "first_session_len": [1.0, 1.0, 1.0, 1.0],
        "forget_rating_offset": 0.0,
        "forget_session_len": 1.0,
        "long_term_transition": [
            [0.0, 0.2, 0.3, 0.5],
            [0.0, 0.1, 0.2, 0.7],
            [0.0, 0.3, 0.3, 0.4],
            [0.0, 0.4, 0.4, 0.2],
        ],
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return path


class BatchedSweepBehaviorCostTests(unittest.TestCase):
    def test_load_usage_defaults_to_markov_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage_path = _button_usage_path(Path(tmp))

            *_, markov = load_usage([1], usage_path)

        self.assertIsNone(markov)

    def test_load_usage_can_enable_review_markov_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage_path = _button_usage_path(Path(tmp))

            *_, markov = load_usage(
                [1],
                usage_path,
                review_markov_transition=True,
            )

        self.assertIsInstance(markov, torch.Tensor)
        assert markov is not None
        self.assertEqual(tuple(markov.shape), (1, 4, 3))
        self.assertTrue(torch.allclose(markov[0, 0], torch.tensor([0.2, 0.3, 0.5])))

    def test_build_behavior_cost_accepts_no_markov_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage_path = _button_usage_path(Path(tmp))
            (
                learn_costs,
                review_costs,
                first_rating_prob,
                review_rating_prob,
                learning_rating_prob,
                relearning_rating_prob,
                state_rating_costs,
                markov,
            ) = load_usage([1], usage_path)

        behavior, _cost = build_behavior_cost(
            1,
            deck_size=10,
            learn_limit=1,
            review_limit=10,
            cost_limit_minutes=60.0,
            learn_costs=learn_costs,
            review_costs=review_costs,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            learning_rating_prob=learning_rating_prob,
            relearning_rating_prob=relearning_rating_prob,
            state_rating_costs=state_rating_costs,
            review_markov_success_weights=markov,
            short_term=False,
        )

        self.assertIsNone(behavior.review_markov_success_weights)


if __name__ == "__main__":
    unittest.main()
