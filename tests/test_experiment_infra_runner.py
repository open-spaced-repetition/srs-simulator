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
from simulator.experiment_infra.runner import run_all, run_stage


def _toml_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def _write_config(
    *,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    gpu_required: bool = False,
    command_template: list[str] | None = None,
    stages: list[str] | None = None,
) -> Path:
    config_path = root / "experiment.toml"
    stage_values = stages or [
        "dry-run",
        "preflight",
        "stage-baseline",
        "train-overfit",
        "sweep",
    ]
    command_template_line = ""
    if command_template is not None:
        command_template_line = f"command_template = {json.dumps(command_template)}\n"
    config_path.write_text(
        f"""
schema_version = 1
name = "runner-smoke"
family = "rl_scheduler"
seed = 42
output_root = "{_toml_path(output_root)}"
stages = {json.dumps(stage_values)}

[users]
train = [1]
validation = [2]
reserved_test = [3]

[baseline]
scheduler = "fsrs6"
log_root = "{_toml_path(baseline_root)}"
expected_engine = "batched"
stage_mode = "copy"
desired_retention_values = [0.9]

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
{command_template_line}
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_baseline_log(
    path: Path,
    *,
    user_id: int,
    scheduler: str = "fsrs6",
    desired_retention: float = 0.9,
) -> None:
    meta = {
        "type": "meta",
        "data": {
            "engine": "batched",
            "days": 30,
            "deck_size": 100,
            "learn_limit": 10,
            "review_limit": 999,
            "cost_limit_minutes": 60.0,
            "priority": "review-first",
            "environment": "lstm",
            "scheduler": scheduler,
            "scheduler_spec": scheduler,
            "user_id": user_id,
            "desired_retention": desired_retention,
            "scheduler_priority": "low_retrievability",
            "seed": 42,
            "fuzz": False,
            "short_term": True,
            "short_term_source": "steps",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta) + "\n", encoding="utf-8")


def _write_artifact_writer(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
user_id = int(sys.argv[2])
lambda_value = float(sys.argv[3])
seed = int(sys.argv[4])
family = sys.argv[5]
engine = sys.argv[6]
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "policy.pt").write_bytes(b"policy")
(output_dir / "metadata.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "artifact_kind": "scheduler-policy",
            "artifact_id": f"user-{user_id}-lambda-{lambda_value}",
            "family": family,
            "scheduler_name": "fsrs6",
            "environment": "lstm",
            "engine": engine,
            "training_user_ids": [user_id],
            "validation_user_ids": [2],
            "seed": seed,
            "policy_path": "policy.pt",
            "feature_version": "v1",
            "action_space": "desired_retention_delta",
            "created_at": "2026-04-29T00:00:00Z",
            "code_commit": "test",
            "lambda_value": lambda_value,
            "capabilities": ["batched"],
        }
    ),
    encoding="utf-8",
)
print(f"wrote artifact for user={user_id} lambda={lambda_value}")
""".lstrip(),
        encoding="utf-8",
    )


def _artifact_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
        "{user_id}",
        "{lambda_value}",
        "{seed}",
        "{family}",
        "{engine}",
    ]


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

    def test_all_stops_at_train_overfit_without_command_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "train-overfit")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages, ["dry-run", "preflight", "stage-baseline", "train-overfit"]
            )
            self.assertTrue((output_root / "test-run" / "preflight").exists())
            self.assertTrue((output_root / "test-run" / "stage-baseline").exists())
            self.assertTrue((output_root / "test-run" / "train-overfit").exists())
            all_summary = output_root / "test-run" / "all" / "all_summary.json"
            self.assertTrue(all_summary.exists())

    def test_all_stops_at_sweep_after_valid_train_overfit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            script_path = root / "write_artifact.py"
            _write_artifact_writer(script_path)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(script_path),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 2)
            self.assertEqual(result.summary["stopped_at"], "sweep")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                ],
            )
            summary = json.loads(
                (
                    output_root / "test-run" / "train-overfit" / "training_summary.json"
                ).read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["artifact_paths"]), 3)

    def test_all_stops_before_baseline_stage_when_preflight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            config_path = _write_config(
                root=root,
                baseline_root=root / "missing",
                output_root=output_root,
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "preflight")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(stages, ["dry-run", "preflight"])
            self.assertFalse((output_root / "test-run" / "stage-baseline").exists())

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
                stage=StageName.SWEEP,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 2)
            self.assertEqual(result.summary["type"], "unsupported-stage")
            self.assertFalse(output_root.exists())

    def test_train_overfit_requires_command_template(self) -> None:
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

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "train-overfit"
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_train_overfit_runs_command_and_validates_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            script_path = root / "write_artifact.py"
            _write_artifact_writer(script_path)
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(script_path),
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.TRAIN_OVERFIT,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 0)
            stage_root = output_root / "test-run" / "train-overfit"
            summary = json.loads((stage_root / "training_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["command_results"]), 3)
            self.assertEqual(len(summary["artifact_paths"]), 3)
            self.assertEqual(
                len(list((stage_root / "train_outputs").rglob("metadata.json"))),
                3,
            )
            manifest = json.loads((stage_root / "manifest.json").read_text())
            self.assertIn("scheduler_artifact_metadata_0", manifest["artifacts"])

    def test_train_overfit_rejects_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)",
                    "{output_dir}",
                ],
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.TRAIN_OVERFIT,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "train-overfit"
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-artifact", gate["failures"])

    def test_stage_baseline_copies_exact_jsonl_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 0)
            stage_root = output_root / "test-run" / "stage-baseline"
            staged_logs = sorted((stage_root / "baseline_logs").rglob("*.jsonl"))
            self.assertEqual(len(staged_logs), 3)
            self.assertEqual(list((stage_root / "baseline_logs").rglob("*.csv")), [])
            summary = json.loads((stage_root / "baseline_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["matched_users"], [1, 2, 3])

    def test_stage_baseline_rejects_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                    scheduler="anki_sm2",
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "stage-baseline"
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-baseline", gate["failures"])
            self.assertEqual(list((stage_root / "baseline_logs").rglob("*.jsonl")), [])

    def test_stage_baseline_rejects_missing_retention_grid_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                    desired_retention=0.8,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "stage-baseline"
            summary = json.loads((stage_root / "baseline_summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertTrue(
                any("desired_retention" in note for note in summary["notes"])
            )


if __name__ == "__main__":
    unittest.main()
