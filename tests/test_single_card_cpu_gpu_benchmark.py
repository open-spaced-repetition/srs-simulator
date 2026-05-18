from __future__ import annotations

import unittest

from experiments.single_card_tradeoff.oracle_stationary_finite_cpu_gpu_benchmark import (
    aggregate_summary_rows,
    device_label,
    parse_devices,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_multiuser_cpu_gpu_benchmark import (
    aggregate_summary_rows as aggregate_multiuser_summary_rows,
)


class SingleCardCpuGpuBenchmarkTests(unittest.TestCase):
    def test_parse_devices_and_labels(self) -> None:
        self.assertEqual(parse_devices("cpu, cuda:0 "), ["cpu", "cuda:0"])
        self.assertEqual(device_label("cuda:0"), "cuda_0")

    def test_aggregate_summary_rows_reports_cpu_relative_speedup(self) -> None:
        rows = [
            {
                "device": "cpu",
                "device_label": "cpu",
                "wall_runtime_s": 20.0,
                "train_runtime_s": 12.0,
                "eval_runtime_s": 5.0,
                "eval_runtime_mean_s": 1.0,
                "parameter_count": 476,
                "final_ce_loss": 0.7,
                "train_teacher_action_agreement": 0.75,
                "eval_teacher_action_agreement": 0.74,
            },
            {
                "device": "cuda",
                "device_label": "cuda",
                "wall_runtime_s": 5.0,
                "train_runtime_s": 2.0,
                "eval_runtime_s": 1.0,
                "eval_runtime_mean_s": 0.2,
                "parameter_count": 476,
                "final_ce_loss": 0.69,
                "train_teacher_action_agreement": 0.76,
                "eval_teacher_action_agreement": 0.75,
                "gpu_monitor_sample_count": 3,
                "gpu_monitor_shared_memory_peak_single_adapter_bytes": 1024,
                "gpu_monitor_shared_memory_spill_detected": False,
                "gpu_monitor_nvidia_smi_peak_memory_used_mib": 1234.0,
            },
        ]

        actual = aggregate_summary_rows(rows)

        self.assertEqual([row["device"] for row in actual], ["cpu", "cuda"])
        self.assertEqual(actual[0]["cpu_relative_speedup"], 1.0)
        self.assertEqual(actual[1]["cpu_relative_speedup"], 4.0)
        self.assertEqual(actual[1]["gpu_monitor_sample_count_max"], 3)
        self.assertFalse(actual[1]["gpu_monitor_shared_memory_spill_detected"])

    def test_multiuser_aggregate_summary_rows_reports_stage_timings(self) -> None:
        rows = [
            {
                "device": "cpu",
                "device_label": "cpu",
                "user_count": 2,
                "user_ids": "1,2",
                "wall_runtime_s": 40.0,
                "setup_runtime_s": 1.0,
                "teacher_runtime_s": 10.0,
                "train_runtime_s": 20.0,
                "agreement_runtime_s": 3.0,
                "eval_runtime_s": 6.0,
                "total_runtime_s": 40.0,
                "params_per_user": 476,
                "ensemble_trainable_params": 952,
                "mean_final_ce_loss": 0.8,
                "mean_train_teacher_action_agreement": 0.7,
                "mean_eval_teacher_action_agreement": 0.69,
            },
            {
                "device": "cuda",
                "device_label": "cuda",
                "user_count": 2,
                "user_ids": "1,2",
                "wall_runtime_s": 10.0,
                "setup_runtime_s": 1.5,
                "teacher_runtime_s": 2.0,
                "train_runtime_s": 4.0,
                "agreement_runtime_s": 1.0,
                "eval_runtime_s": 1.5,
                "total_runtime_s": 10.0,
                "params_per_user": 476,
                "ensemble_trainable_params": 952,
                "mean_final_ce_loss": 0.78,
                "mean_train_teacher_action_agreement": 0.72,
                "mean_eval_teacher_action_agreement": 0.71,
                "gpu_monitor_sample_count": 4,
                "gpu_monitor_shared_memory_peak_single_adapter_bytes": 2048,
                "gpu_monitor_shared_memory_spill_detected": False,
                "gpu_monitor_nvidia_smi_peak_memory_used_mib": 1500.0,
            },
        ]

        actual = aggregate_multiuser_summary_rows(rows)

        self.assertEqual([row["device"] for row in actual], ["cpu", "cuda"])
        self.assertEqual(actual[1]["cpu_relative_speedup"], 4.0)
        self.assertEqual(actual[1]["teacher_runtime_s_mean"], 2.0)
        self.assertEqual(actual[1]["ensemble_trainable_params"], 952)
        self.assertEqual(actual[1]["gpu_monitor_sample_count_max"], 4)


if __name__ == "__main__":
    unittest.main()
