from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.analyze_scheduler_comparison import (
    parse_args as parse_analyze_args,
    render_report,
)
from experiments.retention_sweep.build_pareto_users import (
    _build_command as build_pareto_user_command,
)
from experiments.retention_sweep.build_pareto_users import (
    parse_args as parse_build_args,
)
from simulator.experiment_infra import ExperimentConfig, StageName
from simulator.experiment_infra.runner import run_stage


def _write_workflow_config(
    path: Path,
    *,
    log_dir: Path,
    build_script: Path | None = None,
    analyze_script: Path | None = None,
) -> None:
    build_command = (
        f'command_template = ["{sys.executable}", "{build_script}", "{{output_dir}}", "{{log_dir}}"]'
        if build_script is not None
        else ""
    )
    analyze_command = (
        f'command_template = ["{sys.executable}", "{analyze_script}", "{{output_dir}}"]'
        if analyze_script is not None
        else ""
    )
    path.write_text(
        f"""
schema_version = 1
name = "workflow-config-test"
family = "rl_scheduler"
seed = 42
output_root = "{(path.parent / "out").as_posix()}"
stages = [
  "dry-run",
  "preflight",
  "stage-baseline",
  "train-overfit",
  "sweep",
  "build-pareto",
  "analyze-pareto",
]

[users]
train = [1, 2]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "{log_dir.as_posix()}"
expected_engine = "batched"
stage_mode = "copy"
environments = ["fsrs6", "lstm"]
desired_retention_values = [0.5, 0.52]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "new-first"
scheduler_priority = "low_retrievability"
fuzz = false

[gpu_guard]
required = false
device = "cpu"
smoke = false

[performance]
device = "cpu"
timeout_seconds = 60.0
write_performance_summary = true
diagnostic_csv_logs = false

[training]
lambda_grid = [0.5]

[training.sa]
chains = 1
iterations = 1
baseline_desired_retention_values = [0.5, 0.52]

[sweep]
log_glob = "**/*.jsonl"
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6_adr_direct"]
log_dir = "{log_dir.as_posix()}"
log_layout = "user"
batch_size = 2
torch_device = "cpu"
start_retention = 0.50
end_retention = 0.52
step = 0.02
no_progress = true
no_log = false

[build_pareto]
{build_command}
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6", "fsrs6_adr_direct"]
log_dir = "{log_dir.as_posix()}"
start_retention = 0.50
end_retention = 0.52
short_term = "off"
engine = "batched"
max_parallel = 2
hide_labels = true

[analyze_pareto]
{analyze_command}
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6", "fsrs6_adr_direct"]
comparisons = ["fsrs6_adr_direct:fsrs6"]
start_retention = 0.50
end_retention = 0.52
short_term = "off"
engine = "batched"
fuzz = "off"
""".lstrip(),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class UnifiedWorkflowConfigTests(unittest.TestCase):
    def test_checked_in_linear_dr_config_uses_unified_workflow(self) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_delta_linear_sa_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_delta_linear_sa_users_1_8")
        self.assertEqual(
            config.training_sa["feature_version"],
            "fsrs6_adr_delta_log_linear_v1",
        )
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr_delta",))
        self.assertEqual(
            config.stages,
            (
                StageName.DRY_RUN,
                StageName.PREFLIGHT,
                StageName.STAGE_BASELINE,
                StageName.TRAIN_OVERFIT,
                StageName.SWEEP,
                StageName.BUILD_PARETO,
                StageName.ANALYZE_PARETO,
            ),
        )

    def test_checked_in_cmaes_sa_linear_config_uses_unified_workflow(self) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_direct_linear_cmaes_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_direct_linear_cmaes_users_1_8")
        self.assertEqual(
            config.training_sa["feature_version"],
            "fsrs6_adr_direct_log_linear_v1",
        )
        self.assertEqual(
            len(config.training_sa["baseline_desired_retention_values"]),
            23,
        )
        self.assertEqual(config.training_optimizer["population_size"], 32)
        self.assertEqual(config.training_batch.max_lanes_per_batch, 6400)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr_direct",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr_direct"))
        self.assertEqual(
            config.stages,
            (
                StageName.DRY_RUN,
                StageName.PREFLIGHT,
                StageName.STAGE_BASELINE,
                StageName.TRAIN_OVERFIT,
                StageName.SWEEP,
                StageName.BUILD_PARETO,
                StageName.ANALYZE_PARETO,
            ),
        )

    def test_checked_in_cmaes_direct_poly_config_uses_six_parameter_policy(
        self,
    ) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_direct_cmaes_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_direct_cmaes_users_1_8")
        self.assertEqual(
            config.training_sa["feature_version"],
            "fsrs6_adr_direct_log_poly_v1",
        )
        self.assertEqual(
            len(config.training_sa["baseline_desired_retention_values"]),
            23,
        )
        self.assertEqual(config.training_optimizer["population_size"], 32)
        self.assertEqual(config.training_optimizer["generations"], 20)
        self.assertEqual(len(config.training_optimizer["bounds"][0]), 6)
        self.assertEqual(len(config.training_optimizer["bounds"][1]), 6)
        self.assertEqual(config.training_batch.max_lanes_per_batch, 6400)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr_direct",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr_direct"))

    def test_experiment_config_loads_new_workflow_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            _write_workflow_config(config_path, log_dir=root / "logs")

            config = ExperimentConfig.from_toml(config_path)

        self.assertEqual(config.stages[-2], StageName.BUILD_PARETO)
        self.assertEqual(config.stages[-1], StageName.ANALYZE_PARETO)
        self.assertEqual(config.baseline.environments, ("fsrs6", "lstm"))
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr_direct",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr_direct"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs6_adr_direct:fsrs6",))

    def test_build_pareto_users_config_expands_per_user_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            output_dir = root / "pareto"
            _write_workflow_config(config_path, log_dir=root / "logs")

            args, extra = parse_build_args(
                [
                    "--config",
                    str(config_path),
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            command = build_pareto_user_command(args, 1, extra)

        self.assertEqual(args.start_user, 1)
        self.assertEqual(args.end_user, 2)
        self.assertEqual(args.env, "fsrs6,lstm")
        self.assertEqual(args.sched, "fsrs6,fsrs6_adr_direct")
        self.assertTrue(
            any(
                "simulation_results_retention_sweep_user_1.json" in item
                for item in command
            )
        )
        self.assertIn("--hide-labels", command)

    def test_analyze_scheduler_comparison_reads_config_and_writes_report_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            log_dir = root / "pareto"
            _write_workflow_config(config_path, log_dir=root / "logs")
            log_dir.mkdir()
            (log_dir / "simulation_results_retention_sweep_user_1.json").write_text(
                json.dumps(
                    [
                        {
                            "environment": "fsrs6",
                            "scheduler": "fsrs6",
                            "user_id": 1,
                            "desired_retention": 0.5,
                            "memorized_average": 100.0,
                            "time_average": 1.0,
                            "reviews_average": 10.0,
                            "avg_accum_memorized_per_hour": 100.0,
                            "engine": "batched",
                            "short_term": False,
                            "fuzz": False,
                        },
                        {
                            "environment": "fsrs6",
                            "scheduler": "fsrs6_adr_direct",
                            "user_id": 1,
                            "desired_retention": 0.5,
                            "memorized_average": 110.0,
                            "time_average": 1.0,
                            "reviews_average": 10.0,
                            "avg_accum_memorized_per_hour": 110.0,
                            "engine": "batched",
                            "short_term": False,
                            "fuzz": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            args = parse_analyze_args(
                ["--config", str(config_path), "--log-dir", str(log_dir)]
            )
            report = render_report(args)

        self.assertIn("fsrs6_adr_direct - fsrs6", report)
        self.assertIn("### Same-user same-DR dominance", report)
        self.assertIn(
            "| fsrs6_adr_direct - fsrs6 | 1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 |", report
        )
        self.assertIn("Loaded 2 records", report)

    def test_runner_executes_build_and_analyze_pareto_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            build_script = root / "write_build.py"
            analyze_script = root / "write_analyze.py"
            build_script.write_text(
                """
from __future__ import annotations

import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "simulation_results_retention_sweep_user_1.json").write_text(
    json.dumps([]),
    encoding="utf-8",
)
(output_dir / "plot.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
""".lstrip(),
                encoding="utf-8",
            )
            analyze_script.write_text(
                """
from __future__ import annotations

import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "analysis.md").write_text("# Analysis\\n", encoding="utf-8")
""".lstrip(),
                encoding="utf-8",
            )
            _write_workflow_config(
                config_path,
                log_dir=root / "logs",
                build_script=build_script,
                analyze_script=analyze_script,
            )
            output_root = root / "out"
            run_id = "test-run"
            _write_json(
                output_root / run_id / "stage-baseline" / "baseline_summary.json",
                {"passed": True},
            )
            _write_json(
                output_root / run_id / "sweep" / "sweep_summary.json",
                {"passed": True},
            )

            build_result = run_stage(
                config_path=config_path,
                stage=StageName.BUILD_PARETO,
                repo_root=REPO_ROOT,
                run_id=run_id,
            )
            analyze_result = run_stage(
                config_path=config_path,
                stage=StageName.ANALYZE_PARETO,
                repo_root=REPO_ROOT,
                run_id=run_id,
            )

            self.assertEqual(build_result.exit_code, 0)
            self.assertEqual(analyze_result.exit_code, 0)
            self.assertEqual(build_result.summary["type"], "build-pareto")
            self.assertEqual(analyze_result.summary["type"], "analyze-pareto")
            build_command_record = json.loads(
                (
                    output_root
                    / run_id
                    / "build-pareto"
                    / "commands"
                    / "build_pareto_command.json"
                ).read_text()
            )
            self.assertIn(str(output_root / run_id), build_command_record["command"])
            self.assertNotIn(str(root / "logs"), build_command_record["command"])


if __name__ == "__main__":
    unittest.main()
