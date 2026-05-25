from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from experiments.single_card_tradeoff.cli.fsrs6_implied_cost_weight import (
    PolicyPoint,
    infer_implied_lambdas,
    load_scheduler_points,
    point_result_rows,
    retention_summary_rows,
)


def _point(retention: float, memory: float, minutes: float) -> PolicyPoint:
    return PolicyPoint(
        group_key=("1", "fsrs6", "False", "42", "1825", "10000", "fsrs6"),
        user_id="1",
        environment="fsrs6",
        review_markov_transition="False",
        seed="42",
        days="1825",
        particles="10000",
        scheduler="fsrs6",
        desired_retention=retention,
        memory=memory,
        minutes=minutes,
        source_row_index=2,
    )


class FSRS6ImpliedCostWeightTests(unittest.TestCase):
    def test_supported_points_get_lambda_intervals(self) -> None:
        results = infer_implied_lambdas(
            [
                _point(0.8, 0.8, 1.0),
                _point(0.9, 0.9, 2.0),
                _point(0.95, 0.95, 4.0),
            ]
        )
        by_retention = {result.point.desired_retention: result for result in results}

        low_time = by_retention[0.8]
        middle = by_retention[0.9]
        high_memory = by_retention[0.95]

        self.assertTrue(low_time.supported)
        self.assertAlmostEqual(low_time.lambda_min, 0.1)
        self.assertEqual(low_time.lambda_max, float("inf"))

        self.assertTrue(middle.supported)
        self.assertAlmostEqual(middle.lambda_min, 0.025)
        self.assertAlmostEqual(middle.lambda_max, 0.1)

        self.assertTrue(high_memory.supported)
        self.assertAlmostEqual(high_memory.lambda_min, 0.0)
        self.assertAlmostEqual(high_memory.lambda_max, 0.025)

    def test_dominated_point_gets_best_fit_lambda_and_regret(self) -> None:
        results = infer_implied_lambdas(
            [
                _point(0.8, 0.8, 1.0),
                _point(0.85, 0.85, 3.0),
                _point(0.9, 0.9, 2.0),
            ]
        )
        dominated = next(
            result for result in results if result.point.desired_retention == 0.85
        )

        self.assertFalse(dominated.supported)
        self.assertTrue(dominated.dominated)
        self.assertAlmostEqual(dominated.best_fit_lambda, 0.0)
        self.assertAlmostEqual(dominated.best_fit_regret, 0.05)

    def test_same_time_lower_memory_is_unsupported(self) -> None:
        results = infer_implied_lambdas(
            [
                _point(0.8, 0.8, 1.0),
                _point(0.9, 0.9, 1.0),
            ]
        )
        lower = next(
            result for result in results if result.point.desired_retention == 0.8
        )

        self.assertFalse(lower.supported)
        self.assertTrue(lower.dominated)
        self.assertAlmostEqual(lower.best_fit_regret, 0.1)

    def test_duplicate_retention_within_group_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate desired_retention"):
            infer_implied_lambdas(
                [
                    _point(0.8, 0.8, 1.0),
                    _point(0.8, 0.82, 1.2),
                ]
            )

    def test_retention_summary_aggregates_supported_counts(self) -> None:
        results = infer_implied_lambdas(
            [
                _point(0.8, 0.8, 1.0),
                _point(0.9, 0.9, 2.0),
                _point(0.95, 0.95, 4.0),
            ]
        )
        summary = retention_summary_rows(results)
        first = summary[0]

        self.assertEqual(first["desired_retention"], "0.8")
        self.assertEqual(first["point_count"], 1)
        self.assertEqual(first["supported_count"], 1)
        self.assertEqual(first["dominated_count"], 0)

    def test_load_scheduler_points_filters_fsrs6_desired_retention_rows(self) -> None:
        rows = [
            {
                "user_id": "1",
                "environment": "fsrs6",
                "review_markov_transition": "False",
                "seed": "42",
                "days": "30",
                "particles": "10",
                "scheduler": "fsrs6",
                "desired_retention": "0.8",
                "card_expected_retrievability": "0.75",
                "card_minutes_per_day": "0.02",
            },
            {
                "user_id": "1",
                "environment": "fsrs6",
                "scheduler": "fsrs6_oracle",
                "desired_retention": "",
                "card_expected_retrievability": "0.8",
                "card_minutes_per_day": "0.03",
            },
        ]

        points = load_scheduler_points(rows, scheduler="fsrs6")

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].desired_retention, 0.8)
        self.assertEqual(points[0].memory, 0.75)
        self.assertEqual(points[0].minutes, 0.02)

    def test_point_rows_are_csv_writable(self) -> None:
        results = infer_implied_lambdas(
            [
                _point(0.8, 0.8, 1.0),
                _point(0.9, 0.9, 2.0),
            ]
        )
        rows = point_result_rows(results)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["scheduler"], "fsrs6")


if __name__ == "__main__":
    unittest.main()
