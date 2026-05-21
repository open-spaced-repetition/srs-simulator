from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from experiments.single_card_tradeoff.core.workflow_config import (
    SingleCardWorkflowStage,
    WorkflowTask,
    load_workflow_config,
)
from experiments.single_card_tradeoff.core.workflow_runner import (
    run_workflow,
    run_workflow_stage,
)
from experiments.single_card_tradeoff.core.workflow_tasks import (
    expected_artifacts,
    task_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "experiments" / "single_card_tradeoff" / "configs"


class SingleCardWorkflowTests(unittest.TestCase):
    def test_checked_in_configs_no_longer_use_legacy_commands(self) -> None:
        for path in sorted(CONFIG_ROOT.rglob("*.toml")):
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn(
                "commands",
                raw,
                msg=f"legacy commands remain in {path.relative_to(REPO_ROOT)}",
            )

    def test_distill_control_config_synthesizes_formal_stages(self) -> None:
        config = load_workflow_config(
            CONFIG_ROOT / "distill_train_weights_1_4_control_markov_off.toml"
        )

        self.assertEqual(config.name, "distill_train_weights_1_4_control_markov_off")
        self.assertEqual(len(config.tasks), 8)
        self.assertEqual(config.tasks[0].stage, SingleCardWorkflowStage.TRAIN)
        self.assertEqual(config.tasks[0].kind, "stationary_finite_distill_multiuser")
        self.assertEqual(config.tasks[-1].stage, SingleCardWorkflowStage.ANALYZE)
        self.assertEqual(config.tasks[-1].kind, "train_weight_control_analysis")

        train_task = next(
            task for task in config.tasks if task.name == "train_add_4_only"
        )
        train_command = task_command(train_task)
        self.assertIn("--cost-weights", train_command)
        self.assertEqual(
            train_command[train_command.index("--cost-weights") + 1],
            "0,4,16,64,256,1024",
        )

        eval_task = next(
            task for task in config.tasks if task.name == "evaluate_exact_value_formal"
        )
        eval_command = task_command(eval_task)
        self.assertIn("--skip-default-policies", eval_command)
        self.assertGreaterEqual(eval_command.count("--distill-policy"), 5)
        eval_artifacts = {path.name for path in expected_artifacts(eval_task)}
        self.assertIn("performance_summary.json", eval_artifacts)

    def test_tradeoff_config_uses_formal_task_contract(self) -> None:
        config = load_workflow_config(
            CONFIG_ROOT / "adr_vs_476_tradeoff_first8_users.toml"
        )

        self.assertEqual(len(config.tasks), 1)
        task = config.tasks[0]
        self.assertEqual(task.kind, "tradeoff_config")
        command = task_command(task)
        self.assertIn(
            "experiments.single_card_tradeoff.cli.run_tradeoff_config", command
        )
        artifacts = expected_artifacts(task)
        self.assertTrue(any(path.name == "combined_results.csv" for path in artifacts))
        self.assertTrue(any(path.name == "mean_summary.csv" for path in artifacts))

    def test_low_param_contract_includes_run_monitor_summary(self) -> None:
        config = load_workflow_config(
            CONFIG_ROOT / "reports" / "low_param_direct_search.toml"
        )
        task = next(
            task
            for task in config.tasks
            if task.kind == "low_param_direct_policy_search_multiuser"
        )

        artifact_names = {path.name for path in expected_artifacts(task)}

        self.assertIn("performance_summary.json", artifact_names)
        self.assertIn(
            "gpu_monitor", {path.parent.name for path in expected_artifacts(task)}
        )

    def test_native_adr_train_contract_includes_policy_manifest(self) -> None:
        task = WorkflowTask(
            name="train_native_adr",
            stage=SingleCardWorkflowStage.TRAIN,
            kind="fsrs6_adr_train_multiuser",
            description="Train native single-card ADR policies.",
            options={
                "env": "fsrs6_default",
                "user_ids": [1, 2],
                "cost_weights": [16.0, 32.0],
                "out_dir": (
                    "artifacts/single_card_tradeoff/"
                    "fsrs6_adr_single_card_direct_multiuser"
                ),
                "no_progress": True,
            },
            config_path=CONFIG_ROOT / "native_adr_train.toml",
        )

        command = task_command(task)
        artifacts = expected_artifacts(task)
        artifact_names = {path.name for path in artifacts}

        self.assertIn(
            "experiments.single_card_tradeoff.cli.fsrs6_adr_train_multiuser",
            command,
        )
        self.assertIn("--cost-weights", command)
        self.assertEqual(command[command.index("--cost-weights") + 1], "16,32")
        self.assertIn("summary.csv", artifact_names)
        self.assertIn("train_history.csv", artifact_names)
        self.assertIn("policy_manifest.toml", artifact_names)
        self.assertIn("performance_summary.json", artifact_names)
        self.assertTrue(
            any(
                path.as_posix().endswith("user_1/lambda_16/policy.json")
                for path in artifacts
            )
        )
        self.assertTrue(
            any(
                path.as_posix().endswith("user_2/lambda_32/metadata.json")
                for path in artifacts
            )
        )

    def test_policy_viz_contract_includes_distill_outputs(self) -> None:
        config = load_workflow_config(
            CONFIG_ROOT / "reports" / "stationary_finite_policy_viz.toml"
        )
        task = config.tasks[0]

        artifacts = expected_artifacts(task)
        artifact_names = {path.name for path in artifacts}

        self.assertIn("action_summary.csv", artifact_names)
        self.assertIn("distill_action_summary.csv", artifact_names)
        self.assertIn("distill_grid_actions.csv", artifact_names)
        self.assertIn("distill_binned_actions.csv", artifact_names)
        self.assertIn("distill_action_distribution.png", artifact_names)
        self.assertIn("distill_policy_heatmaps.png", artifact_names)
        self.assertIn("distill_exact_difference_heatmaps.png", artifact_names)
        self.assertTrue(
            all("stationary_finite_policy_viz" in path.as_posix() for path in artifacts)
        )

    def test_run_experiment_dry_run_prints_a_plan(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            results = run_workflow(
                config_path=CONFIG_ROOT / "adr_vs_476_tradeoff_first8_users.toml",
                stage="dry-run",
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        self.assertIn("tradeoff_config", buffer.getvalue())

    def test_preflight_writes_stage_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            config_path.write_text(
                f"""
schema_version = 1
name = "workflow_preflight"
family = "single_card_tradeoff"
seed = 42

[workflow]
run_root = "{root / "records"}"

[[tasks]]
name = "evaluate_smoke"
stage = "evaluate"
kind = "tradeoff"

[tasks.options]
out = "{root / "results.csv"}"
regret_auc_out = "{root / "regret_auc.csv"}"
no_progress = true
""".strip(),
                encoding="utf-8",
            )
            config = load_workflow_config(config_path)

            result = run_workflow_stage(
                config,
                stage=SingleCardWorkflowStage.PREFLIGHT,
                run_id="test-run",
            )

            self.assertTrue(result.passed)
            self.assertIsNotNone(result.summary_path)
            assert result.summary_path is not None
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage"], "preflight")
            self.assertEqual(summary["task_count"], 1)
            self.assertTrue((result.stage_root / "resolved_tasks.json").exists())


if __name__ == "__main__":
    unittest.main()
