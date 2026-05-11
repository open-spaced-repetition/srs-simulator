from __future__ import annotations

import os
import unittest
from argparse import ArgumentTypeError
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from experiments.rl_scheduler.benchmark_sampling import (
    CSV_FIELDS,
    LSTM_MAX_BATCH_OFF,
    _aggregate_eval_records,
    _apply_lstm_max_batch_override,
    _candidate_coefficients_by_job,
    _metric_summary,
    _parse_ints,
    _parse_lane_shapes,
    _parse_lstm_max_batch,
    _resolve_fixed_desired_retention,
    _throughput,
    _workload_throughput,
)
from experiments.rl_scheduler.benchmark_sampling_matrix import generate_matrix_cells
from experiments.rl_scheduler.policy_search_common import PolicySearchSettings
from simulator.lstm_utils import DEFAULT_LSTM_MAX_BATCH_SIZE


class BenchmarkSamplingTests(unittest.TestCase):
    def test_parse_ints_ignores_whitespace(self) -> None:
        self.assertEqual(_parse_ints("1, 2,4"), [1, 2, 4])

    def test_parse_lane_shapes_reads_user_candidate_pairs(self) -> None:
        self.assertEqual(_parse_lane_shapes("8x128, 16X64"), [(8, 128), (16, 64)])

    def test_throughput_reports_lane_days(self) -> None:
        actual = _throughput(seconds=2.0, total_lanes=8, days=365)
        self.assertEqual(actual["lanes_per_second"], 4.0)
        self.assertEqual(actual["lane_days_per_second"], 1460.0)
        self.assertEqual(actual["seconds_per_lane"], 0.25)

    def test_csv_fields_include_memory_utilization(self) -> None:
        self.assertIn("nvidia_smi_peak_utilization_memory_percent", CSV_FIELDS)
        self.assertIn("metric_total_reviews", CSV_FIELDS)
        self.assertIn("reviews_per_second", CSV_FIELDS)
        self.assertIn("effective_lstm_max_batch", CSV_FIELDS)

    def test_metric_summary_reports_workload(self) -> None:
        metrics = [
            [
                SimpleNamespace(
                    memorized_average=10.0,
                    memorized_per_minute=2.0,
                    total_reviews=3,
                    total_lapses=1,
                    total_cost=4.5,
                ),
                SimpleNamespace(
                    memorized_average=20.0,
                    memorized_per_minute=4.0,
                    total_reviews=5,
                    total_lapses=2,
                    total_cost=6.5,
                ),
            ]
        ]

        actual = _metric_summary(metrics)

        self.assertEqual(actual["metric_checksum"], 36.0)
        self.assertEqual(actual["metric_total_reviews"], 8)
        self.assertEqual(actual["metric_total_lapses"], 3)
        self.assertEqual(actual["metric_total_cost"], 11.0)

    def test_workload_throughput_reports_review_rates(self) -> None:
        actual = _workload_throughput(
            total_reviews=20,
            seconds=4.0,
            total_lanes=10,
        )

        self.assertEqual(actual["reviews_per_second"], 5.0)
        self.assertEqual(actual["reviews_per_lane"], 2.0)

    def test_fixed_dr_candidate_mode_repeats_one_policy(self) -> None:
        settings = PolicySearchSettings(
            retention_min=0.5,
            retention_max=0.99,
            baseline_desired_retention=0.9,
            torch_device="cpu",
        )

        actual = _candidate_coefficients_by_job(
            user_ids=[1, 2],
            candidate_count=3,
            settings=settings,
            portfolio=cast(Any, SimpleNamespace()),
            feature_version="fsrs6_adr_log_linear_v1",
            device=torch.device("cpu"),
            seed=42,
            candidate_mode="fixed-dr",
            fixed_desired_retention=0.98,
        )

        self.assertEqual(len(actual), 2)
        self.assertEqual([len(item) for item in actual], [3, 3])
        first = actual[0][0]
        self.assertTrue(
            all(coefficients == first for row in actual for coefficients in row)
        )

    def test_resolve_fixed_desired_retention_defaults_to_max_seed(self) -> None:
        portfolio = SimpleNamespace(seed_retention_values=(0.52, 0.96))
        settings = SimpleNamespace(retention_min=0.5, retention_max=0.98)

        actual = _resolve_fixed_desired_retention(
            None,
            portfolio=cast(Any, portfolio),
            settings=cast(Any, settings),
        )

        self.assertEqual(actual, 0.96)

    def test_parse_lstm_max_batch_accepts_integer_and_off(self) -> None:
        self.assertEqual(_parse_lstm_max_batch("1024"), 1024)
        self.assertEqual(_parse_lstm_max_batch("off"), LSTM_MAX_BATCH_OFF)

    def test_parse_lstm_max_batch_rejects_invalid_values(self) -> None:
        with self.assertRaises(ArgumentTypeError):
            _parse_lstm_max_batch("0")
        with self.assertRaises(ArgumentTypeError):
            _parse_lstm_max_batch("-1")
        with self.assertRaises(ArgumentTypeError):
            _parse_lstm_max_batch("many")

    def test_effective_lstm_max_batch_records_by_environment(self) -> None:
        environ: dict[str, str] = {}

        fsrs6_value = _apply_lstm_max_batch_override(
            environment="fsrs6",
            override=None,
            environ=environ,
        )
        with patch.dict(os.environ, {}, clear=True):
            lstm_default = _apply_lstm_max_batch_override(
                environment="lstm",
                override=None,
                environ=environ,
            )
        lstm_value = _apply_lstm_max_batch_override(
            environment="lstm",
            override=1024,
            environ=environ,
        )
        lstm_off = _apply_lstm_max_batch_override(
            environment="lstm",
            override=LSTM_MAX_BATCH_OFF,
            environ=environ,
        )

        self.assertIsNone(fsrs6_value)
        self.assertEqual(lstm_default, DEFAULT_LSTM_MAX_BATCH_SIZE)
        self.assertEqual(lstm_value, 1024)
        self.assertIsNone(lstm_off)
        self.assertEqual(environ["SRS_LSTM_MAX_BATCH"], "off")

    def test_matrix_cell_generation_counts(self) -> None:
        cells = generate_matrix_cells()
        fsrs6_cells = [cell for cell in cells if cell.environment == "fsrs6"]
        lstm_cells = [cell for cell in cells if cell.environment == "lstm"]
        fsrs6_common = [cell for cell in fsrs6_cells if cell.matrix_group == "common"]
        fsrs6_extended = [
            cell for cell in fsrs6_cells if cell.matrix_group == "fsrs6_extended"
        ]

        self.assertEqual(len(cells), 120)
        self.assertEqual(len(lstm_cells), 90)
        self.assertEqual(len(fsrs6_common), 15)
        self.assertEqual(len(fsrs6_extended), 15)
        self.assertTrue(
            all(cell.effective_lstm_max_batch is None for cell in fsrs6_cells)
        )
        self.assertIn(1024, {cell.effective_lstm_max_batch for cell in lstm_cells})
        self.assertIn(None, {cell.effective_lstm_max_batch for cell in lstm_cells})

    def test_aggregate_eval_records_groups_shapes(self) -> None:
        records = [
            {
                "user_count": 2,
                "candidate_count": 4,
                "effective_lstm_max_batch": None,
                "seconds": 2.0,
                "lanes_per_second": 4.0,
                "lane_days_per_second": 100.0,
                "reviews_per_second": 20.0,
                "reviews_per_lane": 5.0,
                "torch_peak_reserved_memory_bytes": 10,
                "nvidia_smi_peak_memory_used_mib": 100,
                "nvidia_smi_peak_utilization_gpu_percent": 40,
                "nvidia_smi_peak_utilization_memory_percent": 50,
            },
            {
                "user_count": 2,
                "candidate_count": 4,
                "effective_lstm_max_batch": None,
                "seconds": 4.0,
                "lanes_per_second": 2.0,
                "lane_days_per_second": 50.0,
                "reviews_per_second": 10.0,
                "reviews_per_lane": 5.0,
                "torch_peak_reserved_memory_bytes": 12,
                "nvidia_smi_peak_memory_used_mib": 90,
                "nvidia_smi_peak_utilization_gpu_percent": 60,
                "nvidia_smi_peak_utilization_memory_percent": 45,
            },
        ]

        actual = _aggregate_eval_records(records)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["total_lanes"], 8)
        self.assertEqual(actual[0]["mean_seconds"], 3.0)
        self.assertEqual(actual[0]["mean_lanes_per_second"], 3.0)
        self.assertEqual(actual[0]["mean_reviews_per_second"], 15.0)
        self.assertEqual(actual[0]["mean_reviews_per_lane"], 5.0)
        self.assertEqual(actual[0]["max_torch_peak_reserved_memory_bytes"], 12)
        self.assertEqual(actual[0]["max_nvidia_smi_peak_memory_used_mib"], 100)
        self.assertEqual(actual[0]["max_nvidia_smi_peak_utilization_gpu_percent"], 60)
        self.assertEqual(
            actual[0]["max_nvidia_smi_peak_utilization_memory_percent"], 50
        )


if __name__ == "__main__":
    unittest.main()
