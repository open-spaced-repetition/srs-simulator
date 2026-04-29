from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import StageName
from simulator.experiment_infra.runner import run_stage


def _toml_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def _write_config(
    *,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    gpu_required: bool = False,
) -> Path:
    config_path = root / "experiment.toml"
    config_path.write_text(
        f"""
schema_version = 1
name = "runner-smoke"
family = "rl_scheduler"
seed = 42
output_root = "{_toml_path(output_root)}"
stages = ["dry-run", "preflight", "train-overfit"]

[users]
train = [1]
validation = [2]
reserved_test = [3]

[baseline]
scheduler = "fsrs6"
log_root = "{_toml_path(baseline_root)}"
expected_engine = "batched"

[simulation]
engine = "batched"
days = 30
deck = 100
learn_limit = 10
review_limit = 999
cost_limit_minutes = 60.0
priority = "review-first"
scheduler_priority = "low_retrievability"
short_term_source = "steps"
fuzz = false

[gpu_guard]
required = {str(gpu_required).lower()}
device = "cpu"
smoke = false

[training]
lambda_grid = [0.0, 0.5, 1.0]
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


class ExperimentInfraRunnerTests(unittest.TestCase):
    def test_dry_run_has_no_formal_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.DRY_RUN,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIsNone(result.stage_root)
            self.assertFalse(output_root.exists())
            self.assertEqual(result.summary["type"], "dry-run")
            self.assertEqual(result.summary["run_id"], "test-run")
            planned = result.summary["planned_stages"]
            self.assertFalse(planned[0]["writes_formal_outputs"])

    def test_preflight_writes_machine_readable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uv.lock").write_text("lock", encoding="utf-8")
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.PREFLIGHT,
                repo_root=root,
                run_id="test-run",
                command=["uv", "run", "runner"],
            )

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(result.stage_root)
            stage_root = output_root / "test-run" / "preflight"
            self.assertEqual(result.stage_root, stage_root)
            for name in (
                "config_snapshot.toml",
                "resolved_config.json",
                "gpu_summary.json",
                "gate_summary.json",
                "command_record.json",
                "run_record.json",
                "preflight_summary.json",
                "manifest.json",
            ):
                self.assertTrue((stage_root / name).exists(), name)
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertTrue(gate["passed"])
            command = json.loads((stage_root / "command_record.json").read_text())
            self.assertEqual(command["exit_code"], 0)

    def test_preflight_fails_on_missing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_baseline = root / "missing"
            output_root = root / "out"
            config_path = _write_config(
                root=root,
                baseline_root=missing_baseline,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.PREFLIGHT,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "preflight"
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-baseline", gate["failures"])

    def test_unsupported_stage_returns_nonzero_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.TRAIN_OVERFIT,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 2)
            self.assertEqual(result.summary["type"], "unsupported-stage")
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
