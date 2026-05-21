from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from experiments.single_card_tradeoff.cli.oracle_stationary_finite_cpu_gpu_benchmark import (
    aggregate_summary_rows,
    device_label,
    parse_devices,
)
from experiments.single_card_tradeoff.cli.oracle_stationary_finite_multiuser_cpu_gpu_benchmark import (
    aggregate_summary_rows as aggregate_multiuser_summary_rows,
    BenchmarkCell,
    _child_command as multiuser_child_command,
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

    def test_multiuser_child_command_uses_module_entrypoint(self) -> None:
        args = argparse.Namespace(
            env="fsrs6",
            user_ids="1,2",
            button_usage=Path("button_usage.jsonl"),
            per_user_supervision="uniform_table",
            days=30,
            deck_scale=10000,
            cost_weights="0,16",
            eval_cost_weights="0,16",
            action_retentions="0.8,0.9",
            train_envs_per_user=16,
            epochs=1,
            steps_per_epoch=1,
            learning_rate=0.001,
            table_samples_per_weight=8,
            network="residual",
            network_depth=2,
            hidden_size=8,
            oracle_s_grid_size=8,
            oracle_d_grid_size=8,
            oracle_teacher_user_batch_size=0,
            oracle_stationary_finite_max_iterations=100,
            oracle_stationary_finite_tolerance=1e-8,
            max_grad_norm=0.5,
            eval_particles=32,
            gpu_monitor_interval_seconds=2.0,
            no_gpu_monitor_enabled=True,
            no_progress=True,
        )
        cell = BenchmarkCell(
            device="cpu",
            repeat=1,
            seed=42,
            run_dir=Path("run"),
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            performance_summary_path=Path("performance_summary.json"),
        )

        command = multiuser_child_command(args, cell)

        self.assertIn("-m", command)
        self.assertIn(
            "experiments.single_card_tradeoff.cli."
            "oracle_stationary_finite_distill_multiuser",
            command,
        )
        self.assertFalse(any(part.endswith(".py") for part in command))


if __name__ == "__main__":
    unittest.main()
