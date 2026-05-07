from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import ExperimentConfig, StageName
from simulator.experiment_infra.runner import (
    TrainCommandJob,
    _build_train_user_batches,
    run_all,
    run_stage,
)


def _toml_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def _write_config(
    *,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    gpu_required: bool = False,
    command_template: list[str] | None = None,
    sweep_command_template: list[str] | None = None,
    pareto_command_template: list[str] | None = None,
    select_command_template: list[str] | None = None,
    aggregate_command_template: list[str] | None = None,
    reserved_test_command_template: list[str] | None = None,
    performance_timeout_seconds: float | None = None,
    stages: list[str] | None = None,
    training_extra: str = "",
    training_sa_extra: str = "",
    sweep_extra: str = "",
) -> Path:
    config_path = root / "experiment.toml"
    stage_values = stages or [
        "dry-run",
        "preflight",
        "stage-baseline",
        "train-overfit",
        "sweep",
        "pareto",
        "select",
        "aggregate",
        "reserved-test",
    ]
    command_template_line = ""
    if command_template is not None:
        command_template_line = f"command_template = {json.dumps(command_template)}\n"
    sweep_command_template_line = ""
    if sweep_command_template is not None:
        sweep_command_template_line = (
            f"command_template = {json.dumps(sweep_command_template)}\n"
        )
    pareto_command_template_line = ""
    if pareto_command_template is not None:
        pareto_command_template_line = (
            f"command_template = {json.dumps(pareto_command_template)}\n"
        )
    select_command_template_line = ""
    if select_command_template is not None:
        select_command_template_line = (
            f"command_template = {json.dumps(select_command_template)}\n"
        )
    aggregate_command_template_line = ""
    if aggregate_command_template is not None:
        aggregate_command_template_line = (
            f"command_template = {json.dumps(aggregate_command_template)}\n"
        )
    reserved_test_command_template_line = ""
    if reserved_test_command_template is not None:
        reserved_test_command_template_line = (
            f"command_template = {json.dumps(reserved_test_command_template)}\n"
        )
    performance_timeout_line = ""
    if performance_timeout_seconds is not None:
        performance_timeout_line = f"timeout_seconds = {performance_timeout_seconds}\n"
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

[performance]
device = "cpu"
{performance_timeout_line}
write_performance_summary = true
diagnostic_csv_logs = false

[training]
lambda_grid = [0.0, 0.5, 1.0]
{command_template_line}
{training_extra}
{training_sa_extra}
[sweep]
log_glob = "*.jsonl"
{sweep_extra}
{sweep_command_template_line}
[pareto]
result_glob = "*.json"
plot_glob = "*.png"
{pareto_command_template_line}
[select]
result_glob = "selection.json"
{select_command_template_line}
[aggregate]
result_glob = "aggregate.json"
{aggregate_command_template_line}
[reserved_test]
log_glob = "*.jsonl"
{reserved_test_command_template_line}
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
(output_dir / "training_progress.jsonl").write_text(
    json.dumps({"event": "artifacts_written"}) + "\\n",
    encoding="utf-8",
)
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


def _write_batch_artifact_writer(path: Path) -> None:
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
for desired_retention in (0.8, 0.9):
    dr_token = str(desired_retention).replace(".", "p")
    artifact_dir = output_dir / f"dr_{dr_token}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "policy.pt").write_bytes(b"policy")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "scheduler-policy",
                "artifact_id": f"user-{user_id}-dr-{desired_retention}-lambda-{lambda_value}",
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
                "baseline_desired_retention": desired_retention,
                "capabilities": ["batched"],
            }
        ),
        encoding="utf-8",
    )
(output_dir / "training_progress.jsonl").write_text(
    json.dumps({"event": "artifacts_written"}) + "\\n",
    encoding="utf-8",
)
print(f"wrote batched artifacts for user={user_id} lambda={lambda_value}")
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


def _train_job(root: Path, user_id: int, lambda_token: str = "0p5") -> TrainCommandJob:
    return TrainCommandJob(
        user_id=user_id,
        baseline_desired_retention=0.9,
        baseline_desired_retention_token="0p9",
        lambda_value=0.5,
        lambda_token=lambda_token,
        output_dir=root / f"user_{user_id}" / f"lambda_{lambda_token}",
        command_record_path=root / f"user_{user_id}_{lambda_token}.json",
        stdout_path=root / f"user_{user_id}_{lambda_token}.out",
        stderr_path=root / f"user_{user_id}_{lambda_token}.err",
        command=[],
    )


def _write_sweep_writer(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
user_id = int(sys.argv[2])
seed = int(sys.argv[3])
engine = sys.argv[4]
scheduler = sys.argv[5]
output_dir.mkdir(parents=True, exist_ok=True)
meta = {
    "type": "meta",
    "data": {
        "engine": engine,
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
        "desired_retention": 0.9,
        "scheduler_priority": "low_retrievability",
        "seed": seed,
        "fuzz": False,
        "short_term": True,
        "short_term_source": "steps",
    },
}
totals = {"type": "totals", "data": {"reviews": 1, "elapsed_minutes": 1.0}}
(output_dir / "sweep.jsonl").write_text(
    json.dumps(meta) + "\\n" + json.dumps(totals) + "\\n",
    encoding="utf-8",
)
print(f"wrote sweep log for user={user_id}")
""".lstrip(),
        encoding="utf-8",
    )


def _sweep_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
        "{user_id}",
        "{seed}",
        "{engine}",
        "{scheduler_name}",
    ]


def _write_pareto_writer(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
baseline_logs_dir = Path(sys.argv[2])
sweep_outputs_dir = Path(sys.argv[3])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "pareto.json").write_text(
    json.dumps(
        {
            "baseline_logs_dir": str(baseline_logs_dir),
            "sweep_outputs_dir": str(sweep_outputs_dir),
            "strict_dominance_points": 1,
        }
    ),
    encoding="utf-8",
)
(output_dir / "pareto.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
print("wrote pareto artifacts")
""".lstrip(),
        encoding="utf-8",
    )


def _pareto_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
        "{baseline_logs_dir}",
        "{sweep_outputs_dir}",
    ]


def _write_select_writer(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
train_summary_path = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
selected = summary["artifact_paths"][0]
(output_dir / "selection.json").write_text(
    json.dumps(
        {
            "selected_artifact_metadata_path": selected,
            "selection_reason": "best external Pareto point",
        }
    ),
    encoding="utf-8",
)
print("wrote selection")
""".lstrip(),
        encoding="utf-8",
    )


def _select_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
        "{train_summary_path}",
    ]


def _write_aggregate_writer(path: Path, *, passed: bool = True) -> None:
    path.write_text(
        f"""
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "aggregate.json").write_text(
    json.dumps(
        {{
            "passed": {str(passed)},
            "strict_dominance_points": 1,
            "high_memory_wins": 1,
            "dr95_wins": 1,
            "near_overlap_rate": 0.0,
            "feasible_time_worse_rate": 0.0,
        }}
    ),
    encoding="utf-8",
)
print("wrote aggregate")
""".lstrip(),
        encoding="utf-8",
    )


def _aggregate_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
    ]


def _write_reserved_test_writer(path: Path) -> None:
    path.write_text(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
user_ids = [int(item) for item in sys.argv[2].split(",") if item]
seed = int(sys.argv[3])
engine = sys.argv[4]
scheduler = sys.argv[5]
output_dir.mkdir(parents=True, exist_ok=True)
for user_id in user_ids:
    meta = {
        "type": "meta",
        "data": {
            "engine": engine,
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
            "desired_retention": 0.9,
            "scheduler_priority": "low_retrievability",
            "seed": seed,
            "fuzz": False,
            "short_term": True,
            "short_term_source": "steps",
        },
    }
    totals = {"type": "totals", "data": {"reviews": 1, "elapsed_minutes": 1.0}}
    (output_dir / f"reserved_user_{user_id}.jsonl").write_text(
        json.dumps(meta) + "\\n" + json.dumps(totals) + "\\n",
        encoding="utf-8",
    )
print("wrote reserved-test logs")
""".lstrip(),
        encoding="utf-8",
    )


def _reserved_test_writer_template(script_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "{output_dir}",
        "{reserved_user_ids}",
        "{seed}",
        "{engine}",
        "{scheduler_name}",
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

    def test_all_stops_at_sweep_without_command_template(self) -> None:
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

            self.assertEqual(result.exit_code, 1)
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
            gate = json.loads(
                (output_root / "test-run" / "sweep" / "gate_summary.json").read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_all_stops_at_pareto_without_command_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "pareto")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                    "pareto",
                ],
            )
            summary = json.loads(
                (output_root / "test-run" / "sweep" / "sweep_summary.json").read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["log_paths"]), 3)
            gate = json.loads(
                (output_root / "test-run" / "pareto" / "gate_summary.json").read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_all_stops_at_select_without_command_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            pareto_script = root / "write_pareto.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            _write_pareto_writer(pareto_script)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
                pareto_command_template=_pareto_writer_template(pareto_script),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "select")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                    "pareto",
                    "select",
                ],
            )
            summary = json.loads(
                (
                    output_root / "test-run" / "pareto" / "pareto_summary.json"
                ).read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["result_paths"]), 1)
            self.assertEqual(len(summary["plot_paths"]), 1)
            gate = json.loads(
                (output_root / "test-run" / "select" / "gate_summary.json").read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_all_stops_at_aggregate_without_command_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            pareto_script = root / "write_pareto.py"
            select_script = root / "write_select.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            _write_pareto_writer(pareto_script)
            _write_select_writer(select_script)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
                pareto_command_template=_pareto_writer_template(pareto_script),
                select_command_template=_select_writer_template(select_script),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "aggregate")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                    "pareto",
                    "select",
                    "aggregate",
                ],
            )
            summary = json.loads(
                (
                    output_root / "test-run" / "select" / "select_summary.json"
                ).read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["selection_paths"]), 1)
            self.assertEqual(len(summary["selected_artifact_paths"]), 1)
            gate = json.loads(
                (
                    output_root / "test-run" / "aggregate" / "gate_summary.json"
                ).read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_all_stops_at_reserved_test_without_command_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            pareto_script = root / "write_pareto.py"
            select_script = root / "write_select.py"
            aggregate_script = root / "write_aggregate.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            _write_pareto_writer(pareto_script)
            _write_select_writer(select_script)
            _write_aggregate_writer(aggregate_script)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
                pareto_command_template=_pareto_writer_template(pareto_script),
                select_command_template=_select_writer_template(select_script),
                aggregate_command_template=_aggregate_writer_template(aggregate_script),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "reserved-test")
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                    "pareto",
                    "select",
                    "aggregate",
                    "reserved-test",
                ],
            )
            summary = json.loads(
                (
                    output_root / "test-run" / "aggregate" / "aggregate_summary.json"
                ).read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["aggregate_gate_passed"])
            gate = json.loads(
                (
                    output_root / "test-run" / "reserved-test" / "gate_summary.json"
                ).read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("invalid-config", gate["failures"])

    def test_all_passes_after_valid_reserved_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            pareto_script = root / "write_pareto.py"
            select_script = root / "write_select.py"
            aggregate_script = root / "write_aggregate.py"
            reserved_script = root / "write_reserved.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            _write_pareto_writer(pareto_script)
            _write_select_writer(select_script)
            _write_aggregate_writer(aggregate_script)
            _write_reserved_test_writer(reserved_script)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
                pareto_command_template=_pareto_writer_template(pareto_script),
                select_command_template=_select_writer_template(select_script),
                aggregate_command_template=_aggregate_writer_template(aggregate_script),
                reserved_test_command_template=_reserved_test_writer_template(
                    reserved_script
                ),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 0)
            self.assertIsNone(result.summary["stopped_at"])
            stages = [item["stage"] for item in result.summary["stage_results"]]
            self.assertEqual(
                stages,
                [
                    "dry-run",
                    "preflight",
                    "stage-baseline",
                    "train-overfit",
                    "sweep",
                    "pareto",
                    "select",
                    "aggregate",
                    "reserved-test",
                ],
            )
            summary = json.loads(
                (
                    output_root
                    / "test-run"
                    / "reserved-test"
                    / "reserved_test_summary.json"
                ).read_text()
            )
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["log_paths"]), 1)

    def test_aggregate_result_false_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            pareto_script = root / "write_pareto.py"
            select_script = root / "write_select.py"
            aggregate_script = root / "write_aggregate.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            _write_pareto_writer(pareto_script)
            _write_select_writer(select_script)
            _write_aggregate_writer(aggregate_script, passed=False)
            for user_id in (1, 2, 3):
                _write_baseline_log(
                    baseline_root / f"user_{user_id}" / f"log_user_{user_id}.jsonl",
                    user_id=user_id,
                )
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
                pareto_command_template=_pareto_writer_template(pareto_script),
                select_command_template=_select_writer_template(select_script),
                aggregate_command_template=_aggregate_writer_template(aggregate_script),
            )

            result = run_all(
                config_path=config_path,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            self.assertEqual(result.summary["stopped_at"], "aggregate")
            gate = json.loads(
                (
                    output_root / "test-run" / "aggregate" / "gate_summary.json"
                ).read_text()
            )
            self.assertFalse(gate["passed"])
            self.assertIn("gate-failed", gate["failures"])

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

    def test_sweep_rejects_missing_train_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            sweep_script = root / "write_sweep.py"
            _write_sweep_writer(sweep_script)
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                sweep_command_template=_sweep_writer_template(sweep_script),
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.SWEEP,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            stage_root = output_root / "test-run" / "sweep"
            gate = json.loads((stage_root / "gate_summary.json").read_text())
            self.assertFalse(gate["passed"])
            self.assertIn("incomplete-output", gate["failures"])

    def test_sweep_runs_command_and_validates_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            artifact_script = root / "write_artifact.py"
            sweep_script = root / "write_sweep.py"
            _write_artifact_writer(artifact_script)
            _write_sweep_writer(sweep_script)
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(artifact_script),
                sweep_command_template=_sweep_writer_template(sweep_script),
            )

            train_result = run_stage(
                config_path=config_path,
                stage=StageName.TRAIN_OVERFIT,
                repo_root=root,
                run_id="test-run",
            )
            self.assertEqual(train_result.exit_code, 0)

            sweep_result = run_stage(
                config_path=config_path,
                stage=StageName.SWEEP,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(sweep_result.exit_code, 0)
            stage_root = output_root / "test-run" / "sweep"
            summary = json.loads((stage_root / "sweep_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(len(summary["command_results"]), 3)
            self.assertEqual(len(summary["log_paths"]), 3)
            performance = json.loads(
                (stage_root / "performance_summary.json").read_text()
            )
            self.assertTrue(performance["passed"])
            self.assertEqual(performance["stage"], "sweep")
            self.assertEqual(performance["disk_metrics"]["csv_count"], 0)
            manifest = json.loads((stage_root / "manifest.json").read_text())
            self.assertIn("performance_summary", manifest["artifacts"])

    def test_sweep_batches_scheduler_artifacts_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                training_extra=(
                    "batch_baseline_desired_retention_values = true\n"
                    'artifact_metadata_glob = "**/metadata.json"\n'
                ),
                training_sa_extra=(
                    "[training.sa]\n"
                    "baseline_desired_retention = 0.9\n"
                    "baseline_desired_retention_values = [0.8, 0.9]\n"
                ),
                sweep_extra="batch_scheduler_artifacts = true\n",
            )
            train_root = output_root / "test-run" / "train-overfit"
            artifact_paths = []
            for desired_retention in (0.8, 0.9):
                dr_token = str(desired_retention).replace(".", "p")
                artifact_dir = (
                    train_root / "train_outputs" / "user_1" / f"dr_{dr_token}"
                )
                artifact_dir.mkdir(parents=True)
                (artifact_dir / "policy.json").write_text("{}", encoding="utf-8")
                metadata_path = artifact_dir / "metadata.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "artifact_kind": "scheduler-policy",
                            "artifact_id": f"user-1-dr-{desired_retention}",
                            "family": "rl_scheduler",
                            "scheduler_name": "sa_fsrs6",
                            "environment": "lstm",
                            "engine": "batched",
                            "training_user_ids": [1],
                            "validation_user_ids": [2],
                            "seed": 42,
                            "policy_path": "policy.json",
                            "feature_version": "v1",
                            "action_space": "sd_retention_function",
                            "created_at": "2026-04-29T00:00:00Z",
                            "code_commit": "test",
                            "lambda_value": 0.5,
                            "baseline_desired_retention": desired_retention,
                            "capabilities": ["batched"],
                        }
                    ),
                    encoding="utf-8",
                )
                artifact_paths.append(str(metadata_path))
            train_root.mkdir(parents=True, exist_ok=True)
            (train_root / "training_summary.json").write_text(
                json.dumps({"passed": True, "artifact_paths": artifact_paths}),
                encoding="utf-8",
            )

            def fake_batched_sweep_jobs(*, jobs, record_path, **_kwargs):
                record_path.parent.mkdir(parents=True, exist_ok=True)
                record_path.write_text(
                    json.dumps({"batch_lane_count": len(jobs)}),
                    encoding="utf-8",
                )
                for job in jobs:
                    job.output_dir.mkdir(parents=True, exist_ok=True)
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
                            "scheduler": job.scheduler_name,
                            "scheduler_spec": job.scheduler_spec,
                            "user_id": job.user_id,
                            "desired_retention": job.desired_retention,
                            "scheduler_priority": "low_retrievability",
                            "seed": 42,
                            "fuzz": False,
                            "short_term": True,
                            "short_term_source": "steps",
                        },
                    }
                    totals = {"type": "totals", "data": {"reviews": 1}}
                    (job.output_dir / "sweep.jsonl").write_text(
                        json.dumps(meta) + "\n" + json.dumps(totals) + "\n",
                        encoding="utf-8",
                    )

            with patch(
                "simulator.experiment_infra.runner._run_batched_sweep_jobs",
                side_effect=fake_batched_sweep_jobs,
            ):
                result = run_stage(
                    config_path=config_path,
                    stage=StageName.SWEEP,
                    repo_root=root,
                    run_id="test-run",
                )

            self.assertEqual(result.exit_code, 0)
            stage_root = output_root / "test-run" / "sweep"
            summary = json.loads((stage_root / "sweep_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["batch_scheduler_artifacts"])
            self.assertEqual(summary["batch_runs_attempted"], 1)
            self.assertEqual(summary["batch_lanes"], 3)
            self.assertEqual(len(summary["command_results"]), 3)
            self.assertEqual(
                {item["source"] for item in summary["command_results"]},
                {"artifact", "baseline"},
            )
            self.assertEqual(len(summary["log_paths"]), 3)
            performance = json.loads(
                (stage_root / "performance_summary.json").read_text()
            )
            self.assertEqual(performance["execution_shape"]["subprocess_count"], 0)
            self.assertEqual(performance["execution_shape"]["batch_lane_count"], 3)
            manifest = json.loads((stage_root / "manifest.json").read_text())
            self.assertIn("batched_sweep_record", manifest["artifacts"])

    def test_configured_batched_sweep_filters_mixed_env_lane_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            log_root = root / "retention_logs"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                sweep_extra=(
                    'envs = ["fsrs6", "lstm"]\n'
                    'schedulers = ["fsrs6"]\n'
                    f'log_dir = "{_toml_path(log_root)}"\n'
                    'log_layout = "user"\n'
                    "batch_size = 1\n"
                    'torch_device = "cpu"\n'
                    "start_retention = 0.90\n"
                    "end_retention = 0.90\n"
                    "step = 0.02\n"
                    "no_progress = true\n"
                ),
            )
            train_root = output_root / "test-run" / "train-overfit"
            train_root.mkdir(parents=True)
            (train_root / "training_summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "artifact_paths": [str(root / "unused_metadata.json")],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_batches(*, args, ctx, batches, **_kwargs):
                from simulator.batched_sweep.runner import _build_sweep_lanes

                for batch in batches:
                    for environment in ctx.envs:
                        for lane in _build_sweep_lanes(
                            batch=batch,
                            ctx=ctx,
                            environment=environment,
                        ):
                            lane.final_log_dir.mkdir(parents=True, exist_ok=True)
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
                                    "environment": environment,
                                    "scheduler": lane.scheduler_name,
                                    "scheduler_spec": lane.scheduler_spec,
                                    "user_id": lane.user_id,
                                    "desired_retention": lane.desired_retention,
                                    "scheduler_priority": "low_retrievability",
                                    "seed": 42,
                                    "fuzz": False,
                                    "short_term": True,
                                    "short_term_source": "steps",
                                },
                            }
                            totals = {"type": "totals", "data": {"reviews": 1}}
                            log_path = lane.final_log_dir / (
                                f"log_env={environment}_engine=batched_"
                                f"sched={lane.scheduler_name}_st=steps_"
                                f"user={lane.user_id}_ret={lane.desired_retention:.2f}_"
                                "prio=review-first_seed=42.jsonl"
                            )
                            log_path.write_text(
                                json.dumps(meta) + "\n" + json.dumps(totals) + "\n",
                                encoding="utf-8",
                            )

            with patch(
                "simulator.batched_sweep.execution.run_batches",
                side_effect=fake_run_batches,
            ):
                result = run_stage(
                    config_path=config_path,
                    stage=StageName.SWEEP,
                    repo_root=root,
                    run_id="test-run",
                )

            self.assertEqual(result.exit_code, 0)
            stage_root = output_root / "test-run" / "sweep"
            summary = json.loads((stage_root / "sweep_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["batch_lanes"], 2)
            self.assertEqual(len(summary["log_paths"]), 2)
            self.assertEqual(
                {item["environment"] for item in summary["command_results"]},
                {"fsrs6", "lstm"},
            )

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
            self.assertEqual(len(summary["training_progress_paths"]), 3)
            self.assertEqual(
                len(list((stage_root / "train_outputs").rglob("metadata.json"))),
                3,
            )
            performance = json.loads(
                (stage_root / "performance_summary.json").read_text()
            )
            self.assertTrue(performance["passed"])
            self.assertEqual(performance["stage"], "train-overfit")
            self.assertEqual(performance["disk_metrics"]["csv_count"], 0)
            manifest = json.loads((stage_root / "manifest.json").read_text())
            self.assertIn("performance_summary", manifest["artifacts"])
            self.assertIn("training_progress_0", manifest["artifacts"])
            self.assertIn("scheduler_artifact_metadata_0", manifest["artifacts"])

    def test_train_overfit_batches_baseline_dr_grid_in_one_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            script_path = root / "write_batch_artifact.py"
            _write_batch_artifact_writer(script_path)
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=_artifact_writer_template(script_path),
                training_extra=(
                    'artifact_metadata_glob = "**/metadata.json"\n'
                    "batch_baseline_desired_retention_values = true\n"
                ),
                training_sa_extra=(
                    "[training.sa]\n"
                    "baseline_desired_retention = 0.9\n"
                    "baseline_desired_retention_values = [0.8, 0.9]\n"
                ),
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
            self.assertEqual(summary["baseline_desired_retention_values"], [0.8, 0.9])
            self.assertEqual(len(summary["command_results"]), 3)
            self.assertEqual(len(summary["artifact_paths"]), 6)
            self.assertEqual(len(summary["training_progress_paths"]), 3)
            self.assertEqual(
                len(list((stage_root / "train_outputs").rglob("metadata.json"))),
                6,
            )

    def test_train_overfit_in_process_batch_writes_per_user_artifacts(self) -> None:
        from simulator.experiment_infra.training_batch import InProcessTrainOutcome

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
                    "uv",
                    "run",
                    "python",
                    "experiments/rl_scheduler/train_sa_fsrs6.py",
                ],
                training_extra=(
                    'artifact_metadata_glob = "metadata.json"\n'
                    "[training.batch]\n"
                    "enabled = true\n"
                    'trainer = "auto"\n'
                    "batch_size = 1\n"
                ),
                training_sa_extra=(
                    "[training.sa]\n"
                    "chains = 4\n"
                    "iterations = 0\n"
                    "initial_temp = 0.05\n"
                    "final_temp = 0.005\n"
                    "proposal_scale = 0.35\n"
                    "coefficient_min = -8.0\n"
                    "coefficient_max = 8.0\n"
                    "retention_min = 0.5\n"
                    "retention_max = 0.98\n"
                    "baseline_desired_retention = 0.9\n"
                    'torch_device = "cpu"\n'
                    "short_term_threshold = 0.5\n"
                    "short_term_loops_limit = 10\n"
                ),
            )

            def fake_run_batch(*, jobs, **_kwargs):
                outcomes = []
                for job in jobs:
                    job.output_dir.mkdir(parents=True, exist_ok=True)
                    progress_path = job.output_dir / "training_progress.jsonl"
                    progress_path.write_text(
                        json.dumps({"event": "artifacts_written"}) + "\n",
                        encoding="utf-8",
                    )
                    (job.output_dir / "policy.pt").write_bytes(b"policy")
                    metadata_path = job.output_dir / "metadata.json"
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "artifact_kind": "scheduler-policy",
                                "artifact_id": (
                                    f"user-{job.user_id}-lambda-{job.lambda_value}"
                                ),
                                "family": "rl_scheduler",
                                "scheduler_name": "fsrs6",
                                "environment": "lstm",
                                "engine": "batched",
                                "training_user_ids": [job.user_id],
                                "validation_user_ids": [2],
                                "seed": 42,
                                "policy_path": "policy.pt",
                                "feature_version": "v1",
                                "action_space": "desired_retention_delta",
                                "created_at": "2026-04-29T00:00:00Z",
                                "code_commit": "test",
                                "lambda_value": job.lambda_value,
                                "baseline_desired_retention": (
                                    job.baseline_desired_retention
                                ),
                                "training_command_path": str(job.command_record_path),
                                "capabilities": ["batched"],
                            }
                        ),
                        encoding="utf-8",
                    )
                    outcomes.append(
                        InProcessTrainOutcome(
                            job=job,
                            passed=True,
                            artifact_paths=(metadata_path,),
                            progress_path=progress_path,
                        )
                    )
                return outcomes

            with (
                patch(
                    "simulator.experiment_infra.training_batch.run_in_process_train_batch",
                    side_effect=fake_run_batch,
                ) as run_batch,
                patch(
                    "simulator.experiment_infra.runner._run_train_command_job",
                    side_effect=AssertionError("subprocess path should not run"),
                ),
            ):
                result = run_stage(
                    config_path=config_path,
                    stage=StageName.TRAIN_OVERFIT,
                    repo_root=root,
                    run_id="test-run",
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(run_batch.call_count, 1)
            stage_root = output_root / "test-run" / "train-overfit"
            summary = json.loads((stage_root / "training_summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["training_batch"]["batch_size"], 1)
            self.assertEqual(len(summary["command_results"]), 3)
            self.assertEqual(
                {item["execution_mode"] for item in summary["command_results"]},
                {"in_process_batch"},
            )
            self.assertEqual(len(summary["artifact_paths"]), 3)
            performance = json.loads(
                (stage_root / "performance_summary.json").read_text()
            )
            self.assertEqual(performance["execution_shape"]["subprocess_count"], 0)
            self.assertEqual(performance["runtime_metrics"]["batch_runs_attempted"], 1)

    def test_train_overfit_in_process_batch_rejects_unknown_auto_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                command_template=["python", "custom_trainer.py"],
                training_extra=('[training.batch]\nenabled = true\ntrainer = "auto"\n'),
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.TRAIN_OVERFIT,
                repo_root=root,
                run_id="test-run",
            )

            self.assertEqual(result.exit_code, 1)
            summary = json.loads(
                (
                    output_root / "test-run" / "train-overfit" / "training_summary.json"
                ).read_text()
            )
            self.assertIn("invalid-config", summary["failures"])

    def test_training_batch_resolves_cmaes_fsrs6_and_estimates_lanes(self) -> None:
        from simulator.experiment_infra.training_batch import (
            estimate_lanes_per_job,
            resolve_in_process_trainer,
        )

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
                    "uv",
                    "run",
                    "python",
                    "experiments/rl_scheduler/train_cmaes_fsrs6.py",
                ],
                training_extra=(
                    "[training.batch]\n"
                    "enabled = true\n"
                    'trainer = "auto"\n'
                    "\n"
                    "[training.optimizer]\n"
                    'name = "cma_es"\n'
                    "population_size = 32\n"
                    "generations = 10\n"
                    "sigma0 = 0.8\n"
                ),
                training_sa_extra=(
                    "[training.sa]\n"
                    'feature_version = "sa_fsrs6_log_linear_v1"\n'
                    "retention_min = 0.5\n"
                    "retention_max = 0.98\n"
                    "baseline_desired_retention = 0.9\n"
                ),
            )
            config = ExperimentConfig.from_toml(config_path)

        trainer = resolve_in_process_trainer(
            configured_trainer=config.training_batch.trainer,
            command_template=config.train_command_template,
        )

        self.assertEqual(trainer, "cmaes_fsrs6")
        self.assertEqual(estimate_lanes_per_job(trainer=trainer, config=config), 32)

    def test_train_user_batches_keep_user_jobs_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [
                _train_job(root, 1, "0"),
                _train_job(root, 1, "0p5"),
                _train_job(root, 2, "0"),
                _train_job(root, 3, "0"),
            ]

            by_size = _build_train_user_batches(
                jobs=jobs,
                batch_size=2,
                max_lanes_per_batch=None,
                lanes_per_job=10,
            )
            by_lanes = _build_train_user_batches(
                jobs=jobs,
                batch_size=None,
                max_lanes_per_batch=20,
                lanes_per_job=10,
            )

        self.assertEqual(
            [[job.user_id for job in batch] for batch in by_size],
            [[1, 1, 2], [3]],
        )
        self.assertEqual(
            [[job.user_id for job in batch] for batch in by_lanes],
            [[1, 1], [2, 3]],
        )

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

    def test_train_overfit_classifies_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_root = root / "baseline"
            output_root = root / "out"
            baseline_root.mkdir()
            config_path = _write_config(
                root=root,
                baseline_root=baseline_root,
                output_root=output_root,
                performance_timeout_seconds=0.1,
                command_template=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
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
            self.assertIn("timeout", gate["failures"])
            summary = json.loads((stage_root / "training_summary.json").read_text())
            self.assertTrue(summary["command_results"][0]["timed_out"])
            performance = json.loads(
                (stage_root / "performance_summary.json").read_text()
            )
            self.assertEqual(performance["failure_class"], "timeout")

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
