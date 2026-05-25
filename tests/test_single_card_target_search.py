from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.single_card_tradeoff.cli import target_search
from experiments.single_card_tradeoff.cli import target_conditioned_retention_distill
from experiments.single_card_tradeoff.cli import target_constrained_direct_policy_search
from experiments.single_card_tradeoff.core.target_search.direct_training import (
    constrained_rank_candidates,
    direct_target_jobs,
)
from experiments.single_card_tradeoff.core.target_search.family_search import (
    adaptive_theta_candidates,
)
from experiments.single_card_tradeoff.core.target_search.frontier import (
    empirical_frontier,
    frontier_segments,
    target_answers,
)
from experiments.single_card_tradeoff.core.target_search.oracle_refinement import (
    apply_target_certifications,
    certify_oracle_segments,
    oracle_refinement_candidates,
    segment_scalar_gap,
    target_certification_map,
)
from experiments.single_card_tradeoff.core.target_search.io import (
    point_from_row,
    point_row,
    read_points_csv,
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
    exact: bool = False,
) -> EvaluatedPoint:
    return EvaluatedPoint(
        user_id=user_id,
        family=family,
        theta_name="desired_retention",
        theta_value=theta,
        memory=memory,
        minutes=minutes,
        particles=10,
        exact=exact,
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

    def test_oracle_refinement_candidates_use_lambda_ab(self) -> None:
        points = [
            _point(
                0.0,
                0.90,
                4.0,
                family="fsrs6_oracle_stationary_finite",
                exact=True,
            ),
            _point(
                1024.0,
                0.70,
                1.0,
                family="fsrs6_oracle_stationary_finite",
                exact=True,
            ),
        ]

        candidates = oracle_refinement_candidates(
            points,
            [ConstrainedTarget("memory", 0.80, user_id=1)],
            family="fsrs6_oracle_stationary_finite",
            theta_min=0.0,
            theta_max=1024.0,
            existing_theta_values=[0.0, 1024.0],
            max_candidates=4,
        )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0], (0.90 - 0.70) / (4.0 - 1.0))

    def test_oracle_segment_certificate_gap_and_target_certification(self) -> None:
        family = "fsrs6_oracle_stationary_finite"
        low = _point(1024.0, 0.70, 1.0, family=family, exact=True)
        high = _point(0.0, 0.90, 4.0, family=family, exact=True)
        lambda_ab = (high.memory - low.memory) / (high.minutes - low.minutes)
        certificate = _point(lambda_ab, 0.80, 2.5, family=family, exact=True)
        segments = frontier_segments([low, high])

        gap = segment_scalar_gap(segments[0], certificate)
        certified = certify_oracle_segments(
            segments,
            [low, high, certificate],
            family=family,
            tolerance=1e-12,
        )
        answers = target_answers(
            [low, high],
            [ConstrainedTarget("memory", 0.80, user_id=1)],
            family=family,
        )
        answers = apply_target_certifications(
            answers,
            target_certification_map(
                certified,
                [ConstrainedTarget("memory", 0.80, user_id=1)],
                family=family,
            ),
        )

        self.assertAlmostEqual(gap or 0.0, 0.0)
        self.assertTrue(certified[0].certified)
        self.assertTrue(answers[0].certified)

    def test_point_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.csv"
            point = _point(0.8, 0.86, 2.0, exact=True)
            row = point_row(point)
            path.write_text(
                ",".join(row.keys())
                + "\n"
                + ",".join(str(value) for value in row.values())
                + "\n",
                encoding="utf-8",
            )

            loaded = read_points_csv(path)
            parsed = point_from_row({key: str(value) for key, value in row.items()})

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0], parsed)
            self.assertTrue(loaded[0].exact)

    def test_constrained_rank_prefers_feasible_memory_lower_time(self) -> None:
        jobs = direct_target_jobs(
            user_ids=[1],
            target_memories=[0.8],
            target_times=[],
        )
        rank = constrained_rank_candidates(
            memory=torch.tensor([[0.79, 0.81, 0.85]], dtype=torch.float64),
            minutes=torch.tensor([[0.01, 0.03, 0.05]], dtype=torch.float64),
            jobs=jobs,
        )

        self.assertEqual(int(rank.score.argmax(dim=1).item()), 1)

    def test_constrained_rank_prefers_feasible_time_higher_memory(self) -> None:
        jobs = direct_target_jobs(
            user_ids=[1],
            target_memories=[],
            target_times=[0.03],
        )
        rank = constrained_rank_candidates(
            memory=torch.tensor([[0.70, 0.80, 0.90]], dtype=torch.float64),
            minutes=torch.tensor([[0.02, 0.03, 0.04]], dtype=torch.float64),
            jobs=jobs,
        )

        self.assertEqual(int(rank.score.argmax(dim=1).item()), 1)

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

    def test_oracle_cli_branch_writes_certified_answers(self) -> None:
        family = "fsrs6_oracle_stationary_finite"

        def fake_evaluate_oracle_points(**kwargs: object) -> list[EvaluatedPoint]:
            theta_values = kwargs["theta_values"]
            user_ids = kwargs["user_ids"]
            assert isinstance(theta_values, list)
            assert isinstance(user_ids, list)
            points: list[EvaluatedPoint] = []
            for user_id in user_ids:
                for theta in theta_values:
                    if abs(theta) < 1e-12:
                        memory, minutes = 0.90, 4.0
                    elif abs(theta - 1024.0) < 1e-12:
                        memory, minutes = 0.70, 1.0
                    else:
                        memory, minutes = 0.80, 2.5
                    points.append(
                        EvaluatedPoint(
                            user_id=user_id,
                            family=family,
                            theta_name="goal_cost_weight",
                            theta_value=theta,
                            memory=memory,
                            minutes=minutes,
                            exact=True,
                            eval_stage="exact",
                        )
                    )
            return points

        with tempfile.TemporaryDirectory() as temp_dir:
            args = target_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--family",
                    family,
                    "--target-memories",
                    "0.8",
                    "--theta-grid",
                    "0,1024",
                    "--days",
                    "10",
                    "--max-refinement-rounds",
                    "1",
                    "--candidates-per-round",
                    "1",
                    "--torch-device",
                    "cpu",
                    "--no-plot",
                    "--no-progress",
                    "--out-dir",
                    temp_dir,
                ]
            )

            with mock.patch.object(
                target_search,
                "evaluate_oracle_points",
                side_effect=fake_evaluate_oracle_points,
            ):
                target_search.run_search(args)

            answers_path = Path(temp_dir) / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            text = answers_path.read_text(encoding="utf-8")
            self.assertIn("goal_cost_weight", text)
            self.assertIn("True", text)

    def test_oracle_cli_uses_init_points_without_resolving_existing_theta(
        self,
    ) -> None:
        family = "fsrs6_oracle_stationary_finite"

        with tempfile.TemporaryDirectory() as temp_dir:
            init_path = Path(temp_dir) / "points.csv"
            rows = [
                point_row(_point(0.0, 0.90, 4.0, family=family, exact=True)),
                point_row(_point(16.0, 0.70, 1.0, family=family, exact=True)),
            ]
            init_path.write_text(
                ",".join(rows[0].keys())
                + "\n"
                + "\n".join(
                    ",".join(str(value) for value in row.values()) for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            args = target_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--family",
                    family,
                    "--target-memories",
                    "0.8",
                    "--theta-grid",
                    "0,16",
                    "--days",
                    "10",
                    "--max-refinement-rounds",
                    "0",
                    "--init-points",
                    str(init_path),
                    "--torch-device",
                    "cpu",
                    "--no-plot",
                    "--no-progress",
                    "--out-dir",
                    str(Path(temp_dir) / "out"),
                ]
            )

            with mock.patch.object(
                target_search,
                "evaluate_oracle_points",
                side_effect=AssertionError("oracle should use warm-start points"),
            ):
                target_search.run_search(args)

            points_text = (Path(temp_dir) / "out" / "points.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("fsrs6_oracle_stationary_finite", points_text)

    def test_constrained_direct_cli_smoke_writes_policy_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = target_constrained_direct_policy_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--user-ids",
                    "1",
                    "--target-memories",
                    "0.7",
                    "--days",
                    "5",
                    "--population-size",
                    "2",
                    "--elite-count",
                    "1",
                    "--generations",
                    "1",
                    "--train-particles",
                    "2",
                    "--eval-particles",
                    "2",
                    "--torch-device",
                    "cpu",
                    "--no-progress",
                    "--out-dir",
                    temp_dir,
                ]
            )

            target_constrained_direct_policy_search.main_from_args(args)

            self.assertTrue((Path(temp_dir) / "policy.pt").exists())
            answers_path = Path(temp_dir) / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            self.assertIn(
                "fsrs6_low_param_direct_constrained",
                answers_path.read_text(encoding="utf-8"),
            )

    def test_target_conditioned_distill_smoke_writes_policy_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            teacher_dir = Path(temp_dir) / "teacher"
            teacher_args = target_constrained_direct_policy_search.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--user-ids",
                    "1",
                    "--target-memories",
                    "0.7",
                    "--days",
                    "5",
                    "--population-size",
                    "2",
                    "--elite-count",
                    "1",
                    "--generations",
                    "1",
                    "--train-particles",
                    "2",
                    "--eval-particles",
                    "2",
                    "--torch-device",
                    "cpu",
                    "--no-progress",
                    "--out-dir",
                    str(teacher_dir),
                ]
            )
            target_constrained_direct_policy_search.main_from_args(teacher_args)

            distill_dir = Path(temp_dir) / "distill"
            distill_args = target_conditioned_retention_distill.parse_args(
                [
                    "--env",
                    "fsrs6_default",
                    "--teacher-policy",
                    str(teacher_dir / "policy.pt"),
                    "--days",
                    "5",
                    "--epochs",
                    "1",
                    "--steps-per-epoch",
                    "1",
                    "--samples-per-job",
                    "4",
                    "--eval-particles",
                    "2",
                    "--torch-device",
                    "cpu",
                    "--no-progress",
                    "--out-dir",
                    str(distill_dir),
                ]
            )

            target_conditioned_retention_distill.main_from_args(distill_args)

            self.assertTrue((distill_dir / "policy.pt").exists())
            answers_path = distill_dir / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            self.assertIn(
                "fsrs6_target_conditioned_retention_distill",
                answers_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
