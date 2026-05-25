from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.single_card_tradeoff.cli import target_search
from experiments.single_card_tradeoff.core.target_search.family_search import (
    adaptive_theta_candidates,
)
from experiments.single_card_tradeoff.core.target_search.frontier import (
    empirical_frontier,
    target_answers,
)
from experiments.single_card_tradeoff.core.target_search.types import (
    ConstrainedTarget,
    EvaluatedPoint,
)


def _point(
    theta: float,
    memory: float,
    minutes: float,
    *,
    user_id: int = 1,
    family: str = "fsrs6",
) -> EvaluatedPoint:
    return EvaluatedPoint(
        user_id=user_id,
        family=family,
        theta_name="desired_retention",
        theta_value=theta,
        memory=memory,
        minutes=minutes,
        particles=10,
    )


class SingleCardTargetSearchTests(unittest.TestCase):
    def test_frontier_removes_dominated_points(self) -> None:
        points = [
            _point(0.7, 0.80, 2.0),
            _point(0.8, 0.82, 1.9),
            _point(0.9, 0.90, 4.0),
        ]

        frontier = empirical_frontier(points)

        self.assertEqual([point.theta_value for point in frontier], [0.8, 0.9])

    def test_frontier_dominance_is_scoped_by_user_and_family(self) -> None:
        points = [
            _point(0.8, 0.90, 1.0, user_id=1),
            _point(0.8, 0.80, 2.0, user_id=2),
            _point(0.8, 0.70, 3.0, family="fixed"),
        ]

        frontier = empirical_frontier(points)

        self.assertEqual(len(frontier), 3)

    def test_memory_target_selects_lowest_time_feasible_point(self) -> None:
        points = [
            _point(0.7, 0.80, 1.0),
            _point(0.8, 0.86, 2.0),
            _point(0.9, 0.90, 4.0),
        ]
        answers = target_answers(
            points,
            [ConstrainedTarget("memory", 0.85, user_id=1)],
            family="fsrs6",
        )

        answer = answers[0]

        self.assertTrue(answer.feasible)
        self.assertIsNotNone(answer.point)
        point = answer.point
        assert point is not None
        self.assertEqual(point.theta_value, 0.8)
        self.assertAlmostEqual(answer.memory_slack or 0.0, 0.01)
        self.assertTrue(answer.mixed_available)
        self.assertAlmostEqual(answer.mixed_probability_high or 0.0, 5.0 / 6.0)

    def test_time_target_selects_highest_memory_feasible_point(self) -> None:
        points = [
            _point(0.7, 0.80, 1.0),
            _point(0.8, 0.86, 2.0),
            _point(0.9, 0.90, 4.0),
        ]
        answers = target_answers(
            points,
            [ConstrainedTarget("time", 2.5, user_id=1)],
            family="fsrs6",
        )

        answer = answers[0]

        self.assertTrue(answer.feasible)
        self.assertIsNotNone(answer.point)
        point = answer.point
        assert point is not None
        self.assertEqual(point.theta_value, 0.8)
        self.assertAlmostEqual(answer.time_slack or 0.0, 0.5)
        self.assertTrue(answer.mixed_available)
        self.assertAlmostEqual(answer.mixed_probability_high or 0.0, 0.25)

    def test_adaptive_candidates_sample_crossing_brackets(self) -> None:
        candidates = adaptive_theta_candidates(
            [_point(0.5, 0.70, 1.0), _point(0.9, 0.90, 4.0)],
            [ConstrainedTarget("memory", 0.80, user_id=1)],
            family="fsrs6",
            theta_kind="continuous",
            theta_min=0.5,
            theta_max=0.98,
            candidates_per_bracket=3,
        )

        self.assertEqual(candidates, [0.6, 0.7, 0.8])

    def test_cli_smoke_writes_target_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = target_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--family",
                    "fsrs6",
                    "--target-memories",
                    "0.7",
                    "--theta-grid",
                    "0.5,0.9",
                    "--days",
                    "10",
                    "--explore-particles",
                    "4",
                    "--confirm-particles",
                    "4",
                    "--max-refinement-rounds",
                    "0",
                    "--torch-device",
                    "cpu",
                    "--no-plot",
                    "--no-progress",
                    "--out-dir",
                    temp_dir,
                ]
            )

            target_search.run_search(args)

            self.assertTrue((Path(temp_dir) / "points.csv").exists())
            self.assertTrue((Path(temp_dir) / "frontier.csv").exists())
            answers_path = Path(temp_dir) / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            self.assertIn("target_type", answers_path.read_text(encoding="utf-8"))

    def test_fixed_cli_smoke_writes_target_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = target_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--family",
                    "fixed",
                    "--target-times",
                    "0.001",
                    "--theta-grid",
                    "8,32",
                    "--days",
                    "10",
                    "--explore-particles",
                    "4",
                    "--confirm-particles",
                    "4",
                    "--max-refinement-rounds",
                    "0",
                    "--torch-device",
                    "cpu",
                    "--no-plot",
                    "--no-progress",
                    "--out-dir",
                    temp_dir,
                ]
            )

            target_search.run_search(args)

            answers_path = Path(temp_dir) / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            self.assertIn("fixed", answers_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
