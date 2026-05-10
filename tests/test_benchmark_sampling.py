from __future__ import annotations

import unittest

from experiments.rl_scheduler.benchmark_sampling import (
    _aggregate_eval_records,
    _parse_ints,
    _throughput,
)


class BenchmarkSamplingTests(unittest.TestCase):
    def test_parse_ints_ignores_whitespace(self) -> None:
        self.assertEqual(_parse_ints("1, 2,4"), [1, 2, 4])

    def test_throughput_reports_lane_days(self) -> None:
        actual = _throughput(seconds=2.0, total_lanes=8, days=365)
        self.assertEqual(actual["lanes_per_second"], 4.0)
        self.assertEqual(actual["lane_days_per_second"], 1460.0)
        self.assertEqual(actual["seconds_per_lane"], 0.25)

    def test_aggregate_eval_records_groups_shapes(self) -> None:
        records = [
            {
                "user_count": 2,
                "candidate_count": 4,
                "seconds": 2.0,
                "lanes_per_second": 4.0,
                "lane_days_per_second": 100.0,
                "torch_peak_reserved_memory_bytes": 10,
                "nvidia_smi_peak_memory_used_mib": 100,
                "nvidia_smi_peak_utilization_gpu_percent": 40,
                "nvidia_smi_peak_utilization_memory_percent": 50,
            },
            {
                "user_count": 2,
                "candidate_count": 4,
                "seconds": 4.0,
                "lanes_per_second": 2.0,
                "lane_days_per_second": 50.0,
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
        self.assertEqual(actual[0]["max_torch_peak_reserved_memory_bytes"], 12)
        self.assertEqual(actual[0]["max_nvidia_smi_peak_memory_used_mib"], 100)
        self.assertEqual(actual[0]["max_nvidia_smi_peak_utilization_gpu_percent"], 60)
        self.assertEqual(
            actual[0]["max_nvidia_smi_peak_utilization_memory_percent"], 50
        )


if __name__ == "__main__":
    unittest.main()
