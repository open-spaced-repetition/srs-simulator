from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.single_card_tradeoff.cli import target_search
from experiments.single_card_tradeoff.cli import target_memory_scheduler_compare
from experiments.single_card_tradeoff.cli import target_oracle_gap_report
from experiments.single_card_tradeoff.cli import target_conditioned_retention_distill
from experiments.single_card_tradeoff.cli import target_constrained_direct_policy_search
from experiments.single_card_tradeoff.core.target_search.comparison import (
    compare_target_answers_to_oracle,
    oracle_gap_row,
    target_answer_records_from_rows,
)
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


def _target_answer_row(
    *,
    user_id: int = 1,
    target_type: str = "memory",
    target_value: float = 0.8,
    family: str = "candidate",
    feasible: bool = True,
    theta_name: str = "theta",
    theta_value: float = 1.0,
    achieved_m: float = 0.82,
    achieved_t: float = 2.0,
    certified: bool = False,
    mixed_available: bool = False,
    mixed_probability_high: float | None = None,
    mixed_m: float | None = None,
    mixed_t: float | None = None,
) -> dict[str, object]:
    memory_slack = achieved_m - target_value if target_type == "memory" else ""
    time_slack = target_value - achieved_t if target_type == "time" else ""
    return {
        "user_id": user_id,
        "target_type": target_type,
        "target_value": target_value,
        "family": family,
        "feasible": feasible,
        "theta_name": theta_name,
        "theta_value": theta_value,
        "achieved_M": achieved_m,
        "achieved_T": achieved_t,
        "memory_slack": memory_slack,
        "time_slack": time_slack,
        "certified": certified,
        "policy_ref": "",
        "cache_key": "",
        "neighbor_low_theta": "",
        "neighbor_high_theta": "",
        "mixed_available": mixed_available,
        "mixed_probability_high": ""
        if mixed_probability_high is None
        else mixed_probability_high,
        "mixed_M": "" if mixed_m is None else mixed_m,
        "mixed_T": "" if mixed_t is None else mixed_t,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    def test_target_oracle_gap_rows_report_deterministic_and_mixed_gaps(
        self,
    ) -> None:
        candidate_rows = [
            _target_answer_row(
                target_type="memory",
                target_value=0.8,
                family="direct",
                achieved_m=0.81,
                achieved_t=2.4,
            ),
            _target_answer_row(
                target_type="time",
                target_value=1.5,
                family="direct",
                achieved_m=0.78,
                achieved_t=1.4,
            ),
        ]
        oracle_rows = [
            _target_answer_row(
                target_type="memory",
                target_value=0.8,
                family="oracle",
                achieved_m=0.83,
                achieved_t=1.8,
                certified=True,
                mixed_available=True,
                mixed_probability_high=0.25,
                mixed_m=0.8,
                mixed_t=1.6,
            ),
            _target_answer_row(
                target_type="time",
                target_value=1.5,
                family="oracle",
                achieved_m=0.82,
                achieved_t=1.5,
                certified=True,
                mixed_available=True,
                mixed_probability_high=0.5,
                mixed_m=0.84,
                mixed_t=1.5,
            ),
        ]

        gaps = compare_target_answers_to_oracle(
            target_answer_records_from_rows(candidate_rows),
            target_answer_records_from_rows(oracle_rows),
        )
        rows = [oracle_gap_row(gap) for gap in gaps]

        self.assertAlmostEqual(float(rows[0]["deterministic_objective_gap"]), 0.6)
        self.assertAlmostEqual(float(rows[0]["mixed_objective_gap"]), 0.8)
        self.assertAlmostEqual(float(rows[1]["deterministic_objective_gap"]), 0.04)
        self.assertAlmostEqual(float(rows[1]["mixed_objective_gap"]), 0.06)
        self.assertEqual(rows[0]["oracle_certified"], True)

    def test_target_oracle_gap_report_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_dir = root / "candidate"
            oracle_dir = root / "oracle"
            out_dir = root / "out"
            _write_csv(
                candidate_dir / "target_answers.csv",
                [
                    _target_answer_row(
                        family="direct",
                        achieved_m=0.81,
                        achieved_t=2.4,
                    )
                ],
            )
            _write_csv(
                oracle_dir / "target_answers.csv",
                [
                    _target_answer_row(
                        family="oracle",
                        achieved_m=0.83,
                        achieved_t=1.8,
                        certified=True,
                    )
                ],
            )

            args = target_oracle_gap_report.parse_args(
                [
                    "--candidate-target-answers",
                    str(candidate_dir),
                    "--oracle-target-answers",
                    str(oracle_dir),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            target_oracle_gap_report.main_from_args(args)

            gaps_path = out_dir / "target_oracle_gaps.csv"
            metadata_path = out_dir / "target_oracle_gaps_metadata.json"
            self.assertTrue(gaps_path.exists())
            self.assertIn("deterministic_objective_gap", gaps_path.read_text())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["summary"]["oracle_matched_count"], 1)

    def test_target_memory_scheduler_compare_converts_and_aggregates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oracle_path = root / "oracle_target_answers.csv"
            direct_path = root / "direct_target_answers.csv"
            tradeoff_path = root / "tradeoff.csv"
            out_dir = root / "comparison"
            _write_csv(
                oracle_path,
                [
                    _target_answer_row(
                        user_id=1,
                        target_value=0.8,
                        family="oracle",
                        theta_name="goal_cost_weight",
                        theta_value=4.0,
                        achieved_m=0.82,
                        achieved_t=1.0,
                        certified=True,
                    )
                ],
            )
            _write_csv(
                direct_path,
                [
                    _target_answer_row(
                        user_id=1,
                        target_value=0.8,
                        family="direct_raw",
                        theta_name="target_memory",
                        theta_value=0.8,
                        achieved_m=0.83,
                        achieved_t=1.2,
                    )
                ],
            )
            _write_csv(
                tradeoff_path,
                [
                    {
                        "user_id": 1,
                        "scheduler": "fsrs6",
                        "scheduler_spec": "fsrs6",
                        "desired_retention": 0.7,
                        "card_expected_retrievability": 0.79,
                        "card_minutes_per_day": 0.5,
                    },
                    {
                        "user_id": 1,
                        "scheduler": "fsrs6",
                        "scheduler_spec": "fsrs6",
                        "desired_retention": 0.8,
                        "card_expected_retrievability": 0.81,
                        "card_minutes_per_day": 1.5,
                    },
                    {
                        "user_id": 1,
                        "scheduler": "fsrs6",
                        "scheduler_spec": "fsrs6",
                        "desired_retention": 0.9,
                        "card_expected_retrievability": 0.84,
                        "card_minutes_per_day": 2.0,
                    },
                ],
            )

            args = target_memory_scheduler_compare.parse_args(
                [
                    "--oracle",
                    f"oracle={oracle_path}",
                    "--target-answer",
                    f"direct={direct_path}",
                    "--tradeoff-result",
                    f"fsrs6={tradeoff_path}",
                    "--target-memories",
                    "0.8",
                    "--out-dir",
                    str(out_dir),
                    "--no-plots",
                    "--require-complete-target-grid",
                ]
            )
            target_memory_scheduler_compare.main_from_args(args)

            matrix_path = out_dir / "scheduler_target_matrix.csv"
            summary_path = out_dir / "scheduler_summary.csv"
            gaps_path = out_dir / "scheduler_oracle_gaps.csv"
            metadata_path = out_dir / "metadata.json"
            converted_path = root / "converted" / "fsrs6_target_answers.csv"
            converted_gaps_path = (
                root / "converted" / "fsrs6_gap" / "target_oracle_gaps.csv"
            )
            converted_metadata_path = (
                root / "converted" / "fsrs6_gap" / "target_oracle_gaps_metadata.json"
            )
            self.assertTrue(matrix_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(gaps_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(converted_path.exists())
            self.assertTrue(converted_gaps_path.exists())
            self.assertTrue(converted_metadata_path.exists())
            matrix_text = matrix_path.read_text(encoding="utf-8")
            self.assertIn("fsrs6_T", matrix_text)
            self.assertIn("1.5", matrix_text)
            gaps_text = gaps_path.read_text(encoding="utf-8")
            self.assertIn("deterministic_objective_gap", gaps_text)
            self.assertIn("0.5", gaps_text)
            converted_text = converted_path.read_text(encoding="utf-8")
            self.assertIn("fsrs6", converted_text)
            self.assertIn("False", converted_text)
            converted_gaps_text = converted_gaps_path.read_text(encoding="utf-8")
            self.assertIn("candidate_family", converted_gaps_text)
            self.assertIn("fsrs6", converted_gaps_text)
            metadata = json.loads(converted_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["summary"]["candidate_count"], 1)
            comparison_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                comparison_metadata["scheduler_categories"]["fsrs6"],
                "rollout baseline",
            )
            self.assertIn(
                "--tradeoff-result", comparison_metadata["inputs"]["cli_argv"]
            )
            self.assertEqual(
                comparison_metadata["inputs"]["tradeoff_result"],
                [f"fsrs6={tradeoff_path}"],
            )

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
            oracle_answers = Path(temp_dir) / "oracle_target_answers.csv"
            _write_csv(
                oracle_answers,
                [
                    _target_answer_row(
                        target_value=0.7,
                        family="fsrs6_oracle_stationary_finite",
                        achieved_m=0.72,
                        achieved_t=0.01,
                        certified=True,
                    )
                ],
            )
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
                    "--oracle-target-answers",
                    str(oracle_answers),
                    "--out-dir",
                    temp_dir,
                ]
            )

            target_constrained_direct_policy_search.main_from_args(args)

            self.assertTrue((Path(temp_dir) / "policy.pt").exists())
            answers_path = Path(temp_dir) / "target_answers.csv"
            self.assertTrue(answers_path.exists())
            self.assertTrue((Path(temp_dir) / "target_oracle_gaps.csv").exists())
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
