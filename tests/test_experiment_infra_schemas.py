# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import ExperimentConfig, StageName


VALID_CONFIG = """
schema_version = 1
name = "rl-overfit-smoke"
family = "rl_scheduler"
seed = 42
output_root = "artifacts/rl_scheduler/rl-overfit-smoke"
stages = ["preflight", "train-overfit", "sweep", "pareto"]

[users]
train = [1]
validation = [2, 3]
reserved_test = [4]

[baseline]
scheduler = "fsrs6"
log_root = "logs/baseline/fsrs6"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.9]

[simulation]
engine = "batched"
environment = "lstm"
days = 365
deck = 10000
learn_limit = 10
review_limit = 9999
cost_limit_minutes = 720.0
priority = "review-first"
scheduler_priority = "low_retrievability"
short_term_source = "steps"
fuzz = false

[gpu_guard]
required = false
device = "cpu"
smoke = false

[performance]
device = "cpu"
timeout_seconds = 120.0
progress_interval_seconds = 10.0
write_performance_summary = true
diagnostic_csv_logs = false

[training]
lambda_grid = [0.0, 0.25, 0.5]

[training.sa]
chains = 4
iterations = 8
"""


class ExperimentConfigSchemaTests(unittest.TestCase):
    def test_loads_valid_toml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(VALID_CONFIG, encoding="utf-8")

            config = ExperimentConfig.from_toml(path)

        self.assertEqual(config.name, "rl-overfit-smoke")
        self.assertEqual(config.stages[1], StageName.TRAIN_OVERFIT)
        self.assertEqual(config.users.train, (1,))
        self.assertEqual(
            str(config.output_root), "artifacts/rl_scheduler/rl-overfit-smoke"
        )
        self.assertEqual(config.lambda_grid, (0.0, 0.25, 0.5))
        self.assertEqual(config.simulation.environment, "lstm")
        self.assertEqual(config.training_sa["chains"], 4)
        self.assertEqual(config.baseline.desired_retention_values, (0.9,))
        self.assertEqual(config.to_dict()["baseline"]["scheduler"], "fsrs6")
        self.assertEqual(config.performance.device, "cpu")
        self.assertEqual(config.performance.timeout_seconds, 120.0)
        self.assertTrue(config.performance.write_performance_summary)
        self.assertFalse(config.train_batch_baseline_desired_retention_values)
        self.assertFalse(config.training_batch.enabled)

    def test_rejects_overlapping_user_splits(self) -> None:
        raw = VALID_CONFIG.replace("validation = [2, 3]", "validation = [1, 3]")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "disjoint"):
                ExperimentConfig.from_toml(path)

    def test_rejects_invalid_performance_memory_budget(self) -> None:
        raw = VALID_CONFIG.replace(
            "write_performance_summary = true",
            "memory_budget_fraction = 1.5\nwrite_performance_summary = true",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "memory_budget_fraction"):
                ExperimentConfig.from_toml(path)

    def test_rejects_duplicate_lambda_grid(self) -> None:
        raw = VALID_CONFIG.replace(
            "lambda_grid = [0.0, 0.25, 0.5]",
            "lambda_grid = [0.0, 0.25, 0.25]",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                ExperimentConfig.from_toml(path)

    def test_loads_batched_baseline_retention_grid_flag(self) -> None:
        raw = VALID_CONFIG.replace(
            "lambda_grid = [0.0, 0.25, 0.5]",
            (
                "lambda_grid = [0.0, 0.25, 0.5]\n"
                "batch_baseline_desired_retention_values = true"
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            config = ExperimentConfig.from_toml(path)

        self.assertTrue(config.train_batch_baseline_desired_retention_values)
        self.assertTrue(
            config.to_dict()["training"]["batch_baseline_desired_retention_values"]
        )

    def test_loads_training_batch_config(self) -> None:
        raw = VALID_CONFIG.replace(
            "lambda_grid = [0.0, 0.25, 0.5]",
            (
                "lambda_grid = [0.0, 0.25, 0.5]\n\n"
                "[training.batch]\n"
                "enabled = true\n"
                'trainer = "fsrs6_adp_cmaes"\n'
                "batch_size = 8\n"
                "max_lanes_per_batch = 1024\n\n"
                "[training.adp]\n"
                "dr_batch_size = 3\n"
                "weight_delta_scale = 0.5"
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            config = ExperimentConfig.from_toml(path)

        self.assertTrue(config.training_batch.enabled)
        self.assertEqual(config.training_batch.trainer, "fsrs6_adp_cmaes")
        self.assertEqual(config.training_batch.batch_size, 8)
        self.assertEqual(config.training_batch.max_lanes_per_batch, 1024)
        self.assertEqual(config.training_adp["dr_batch_size"], 3)
        self.assertEqual(config.training_adp["weight_delta_scale"], 0.5)
        self.assertTrue(config.to_dict()["training"]["batch"]["enabled"])
        self.assertEqual(config.to_dict()["training"]["adp"]["dr_batch_size"], 3)

    def test_rejects_invalid_training_batch_trainer(self) -> None:
        raw = VALID_CONFIG.replace(
            "lambda_grid = [0.0, 0.25, 0.5]",
            (
                "lambda_grid = [0.0, 0.25, 0.5]\n\n"
                "[training.batch]\n"
                "enabled = true\n"
                'trainer = "external"'
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "training.batch.trainer"):
                ExperimentConfig.from_toml(path)

    def test_loads_batched_sweep_scheduler_artifacts_flag(self) -> None:
        raw = VALID_CONFIG + "\n[sweep]\nbatch_scheduler_artifacts = true\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.toml"
            path.write_text(raw, encoding="utf-8")

            config = ExperimentConfig.from_toml(path)

        self.assertTrue(config.sweep_batch_scheduler_artifacts)
        self.assertTrue(config.to_dict()["sweep"]["batch_scheduler_artifacts"])


if __name__ == "__main__":
    unittest.main()
