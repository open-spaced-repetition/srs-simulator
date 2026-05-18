# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.run_portfolio_workflow import (
    _baseline_sweep_command,
    _default_baseline_run_id,
    _report_command,
    build_workflow_steps,
)
from simulator.experiment_infra.schemas import ExperimentConfig


class PortfolioWorkflowTests(unittest.TestCase):
    def test_default_baseline_run_id_is_manifest_derived(self) -> None:
        self.assertEqual(
            _default_baseline_run_id(
                Path(
                    "artifacts/rl_scheduler/baseline_dr_selection/"
                    "fsrs6_users_1_8_16dr_pop16_gen5.json"
                )
            ),
            "fsrs6_baseline_users_1_8_16dr_pop16_gen5",
        )

    def test_baseline_sweep_command_uses_config_and_manifest(self) -> None:
        config_path = (
            REPO_ROOT
            / "experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml"
        )
        config = ExperimentConfig.from_toml(config_path)
        manifest_config = config.baseline_dr_selection.manifest
        self.assertIsNotNone(manifest_config)
        assert manifest_config is not None
        manifest = REPO_ROOT / manifest_config

        command = _baseline_sweep_command(
            config_path=config_path,
            config=config,
            manifest_path=manifest,
            baseline_run_id="fsrs6_baseline_users_1_8_16dr_pop16_gen5",
            baseline_max_lanes_per_batch=None,
        )

        self.assertIn("experiments/retention_sweep/run_sweep_users_batched.py", command)
        self.assertEqual(command[command.index("--config") + 1], str(config_path))
        self.assertEqual(command[command.index("--env") + 1], "fsrs6,lstm")
        self.assertEqual(command[command.index("--sched") + 1], "fsrs6")
        self.assertEqual(
            command[command.index("--fsrs6-dr-manifest") + 1], str(manifest)
        )
        self.assertNotIn("--max-lanes-per-batch", command)

    def test_existing_manifest_skips_selector_without_force(self) -> None:
        config_path = (
            REPO_ROOT
            / "experiments/rl_scheduler/configs/fsrs6_adr_portfolio_users_1_8_v3.toml"
        )
        config = ExperimentConfig.from_toml(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")

            steps = build_workflow_steps(
                config_path=config_path,
                config=config,
                manifest_path=manifest,
                formal_run_id=config.name,
                baseline_run_id="baseline",
                selector_max_lanes_per_batch=8192,
                baseline_max_lanes_per_batch=None,
                formal_stage="all",
                force_manifest=False,
                skip_manifest=False,
                skip_baseline_sweep=True,
                skip_formal_stages=True,
            )

        self.assertTrue(steps[0].skipped)
        self.assertIn("already exists", steps[0].reason or "")
        self.assertTrue(steps[1].skipped)
        self.assertTrue(steps[2].skipped)

    def test_report_step_uses_config_and_can_be_skipped(self) -> None:
        config_path = (
            REPO_ROOT
            / "experiments/rl_scheduler/configs/anki_sm2_ap_portfolio_users_1_8_pop16_20_v1.toml"
        )
        config = ExperimentConfig.from_toml(config_path)
        manifest = config.baseline_dr_selection.manifest
        self.assertIsNotNone(manifest)
        assert manifest is not None
        command = _report_command(
            config=config,
            formal_run_id="anki_sm2_ap_portfolio_users_1_8_pop16_20_v1",
        )

        self.assertIn("experiments/rl_scheduler/generate_experiment_report.py", command)
        self.assertEqual(
            command[command.index("--candidate-label") + 1],
            "Anki SM2 AP",
        )
        self.assertIn("--comparison-run-root", command)

        steps = build_workflow_steps(
            config_path=config_path,
            config=config,
            manifest_path=REPO_ROOT / manifest,
            formal_run_id="anki_sm2_ap_portfolio_users_1_8_pop16_20_v1",
            baseline_run_id="baseline",
            selector_max_lanes_per_batch=8192,
            baseline_max_lanes_per_batch=None,
            formal_stage="all",
            force_manifest=False,
            skip_manifest=True,
            skip_baseline_sweep=True,
            skip_formal_stages=False,
        )
        self.assertEqual(steps[-1].name, "report")
        self.assertFalse(steps[-1].skipped)

        skipped_steps = build_workflow_steps(
            config_path=config_path,
            config=config,
            manifest_path=REPO_ROOT / manifest,
            formal_run_id="anki_sm2_ap_portfolio_users_1_8_pop16_20_v1",
            baseline_run_id="baseline",
            selector_max_lanes_per_batch=8192,
            baseline_max_lanes_per_batch=None,
            formal_stage="all",
            force_manifest=False,
            skip_manifest=True,
            skip_baseline_sweep=True,
            skip_formal_stages=False,
            skip_report=True,
        )
        self.assertTrue(skipped_steps[-1].skipped)
        self.assertIn("--skip-report", skipped_steps[-1].reason or "")

    def test_oracle_distill_portfolio_config_uses_matched_adr_report(self) -> None:
        config_path = (
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml"
        )
        config = ExperimentConfig.from_toml(config_path)

        self.assertEqual(
            config.training_batch.trainer,
            "auto",
        )
        self.assertTrue(
            any(
                Path(item).name
                == "train_fsrs6_oracle_stationary_finite_distill_portfolio.py"
                for item in config.train_command_template
            )
        )
        self.assertEqual(
            config.sweep_batched.schedulers,
            ("fsrs6_oracle_stationary_finite_distill",),
        )
        self.assertEqual(
            config.report.candidate_label, "Oracle stationary finite distill"
        )
        self.assertEqual(config.report.comparison_label, "ADR")
        self.assertIsNotNone(config.report.comparison_run_root)
        command = _report_command(
            config=config,
            formal_run_id=(
                "fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1"
            ),
        )
        self.assertIn("--comparison-run-root", command)
        self.assertEqual(
            command[command.index("--candidate-label") + 1],
            "Oracle stationary finite distill",
        )


if __name__ == "__main__":
    unittest.main()
