# ruff: noqa: E402
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

[training.policy_search]
baseline_desired_retention_values = [0.5, 0.52]

[sweep]
log_glob = "**/*.jsonl"
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6_adr"]
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
schedulers = ["fsrs6", "fsrs6_adr"]
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
schedulers = ["fsrs6", "fsrs6_adr"]
comparisons = ["fsrs6_adr:fsrs6"]
start_retention = 0.50
end_retention = 0.52
short_term = "off"
engine = "batched"
fuzz = "off"
""".lstrip(),
        encoding="utf-8",
    )


def _write_native_fsrs3_eval_config(path: Path) -> None:
    path.write_text(
        f"""
schema_version = 1
name = "native-fsrs3-eval-test"
family = "rl_scheduler"
seed = 42
output_root = "{(path.parent / "out").as_posix()}"
stages = [
  "dry-run",
  "preflight",
  "stage-baseline",
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
log_root = "{(path.parent / "logs").as_posix()}"
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
lambda_grid = [0.0]

[sweep]
log_glob = "**/*.jsonl"
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs3"]
log_dir = "{(path.parent / "logs").as_posix()}"
log_layout = "user"
max_lanes_per_batch = 1024
torch_device = "cpu"
start_retention = 0.50
end_retention = 0.52
step = 0.02
no_progress = true
no_log = false

[build_pareto]
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6", "fsrs3"]
start_retention = 0.50
end_retention = 0.52
short_term = "off"
engine = "batched"
max_parallel = 2
hide_labels = true

[analyze_pareto]
envs = ["fsrs6", "lstm"]
schedulers = ["fsrs6", "fsrs3"]
comparisons = ["fsrs3:fsrs6"]
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
    def test_checked_in_cmaes_linear_config_uses_unified_workflow(self) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_linear_cmaes_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_linear_cmaes_users_1_8")
        self.assertEqual(
            config.training_policy_search["feature_version"],
            "fsrs6_adr_log_linear_v1",
        )
        self.assertEqual(
            len(config.training_policy_search["baseline_desired_retention_values"]),
            23,
        )
        self.assertEqual(config.training_optimizer["population_size"], 32)
        self.assertEqual(config.training_batch.max_lanes_per_batch, 6400)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr"))
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

    def test_checked_in_adr_linear_portfolio_config_uses_portfolio_children(
        self,
    ) -> None:
        from simulator.experiment_infra.training_batch import (
            estimate_lanes_per_job,
            resolve_in_process_trainer,
        )

        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_linear_portfolio_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_linear_portfolio_users_1_8")
        self.assertEqual(config.lambda_grid, ())
        self.assertNotIn("{lambda_value}", config.train_command_template)
        self.assertNotIn("--lambda", config.train_command_template)
        self.assertEqual(config.train_artifact_glob, "policies/**/metadata.json")
        self.assertTrue(config.train_batch_baseline_desired_retention_values)
        self.assertTrue(config.training_batch.enabled)
        self.assertEqual(config.training_batch.trainer, "auto")
        self.assertEqual(
            config.training_policy_search["feature_version"],
            "fsrs6_adr_log_linear_v1",
        )
        self.assertEqual(
            len(config.training_policy_search["baseline_desired_retention_values"]),
            23,
        )
        self.assertEqual(config.training_portfolio["population_size"], 64)
        self.assertEqual(config.training_portfolio["offspring_size"], 64)
        self.assertEqual(config.training_portfolio["portfolio_size"], 23)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs6_adr:fsrs6",))

        trainer = resolve_in_process_trainer(
            configured_trainer=config.training_batch.trainer,
            command_template=config.train_command_template,
        )

        self.assertEqual(trainer, "fsrs6_adr_portfolio")
        self.assertEqual(estimate_lanes_per_job(trainer=trainer, config=config), 64)

    def test_checked_in_ap_cmaes_config_uses_dr_and_user_batching(self) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_ap_cmaes_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_ap_cmaes_users_1_8")
        self.assertTrue(config.train_batch_baseline_desired_retention_values)
        self.assertEqual(config.train_artifact_glob, "**/metadata.json")
        self.assertEqual(config.training_batch.batch_size, 8)
        self.assertEqual(config.training_ap["dr_batch_size"], 25)
        self.assertEqual(config.training_ap["weight_delta_scale"], 0.5)
        self.assertEqual(len(config.training_optimizer["bounds"][0]), 21)
        self.assertEqual(len(config.training_optimizer["bounds"][1]), 21)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_ap",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_ap"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs6_ap:fsrs6",))

    def test_checked_in_ap_portfolio_config_uses_portfolio_children(self) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_ap_portfolio_users_1_8_v2.toml"
        )

        self.assertEqual(config.name, "fsrs6_ap_portfolio_users_1_8_v2")
        self.assertEqual(config.lambda_grid, ())
        self.assertNotIn("{lambda_value}", config.train_command_template)
        self.assertNotIn("--lambda", config.train_command_template)
        self.assertEqual(config.train_artifact_glob, "policies/**/metadata.json")
        self.assertTrue(config.training_batch.enabled)
        self.assertEqual(config.training_batch.trainer, "auto")
        self.assertEqual(config.training_portfolio["population_size"], 64)
        self.assertEqual(config.training_portfolio["offspring_size"], 64)
        self.assertEqual(config.training_portfolio["portfolio_size"], 23)
        self.assertEqual(config.training_portfolio["retention_mutation_scale"], 0.02)
        self.assertEqual(config.training_ap["weight_delta_scale"], 0.5)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_ap",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_ap"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs6_ap:fsrs6",))

    def test_checked_in_cmaes_adr_poly_config_uses_six_parameter_policy(
        self,
    ) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs6_adr_cmaes_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs6_adr_cmaes_users_1_8")
        self.assertEqual(
            config.training_policy_search["feature_version"],
            "fsrs6_adr_log_poly_v1",
        )
        self.assertEqual(
            len(config.training_policy_search["baseline_desired_retention_values"]),
            23,
        )
        self.assertEqual(config.training_optimizer["population_size"], 32)
        self.assertEqual(config.training_optimizer["generations"], 20)
        self.assertEqual(len(config.training_optimizer["bounds"][0]), 6)
        self.assertEqual(len(config.training_optimizer["bounds"][1]), 6)
        self.assertEqual(config.training_batch.max_lanes_per_batch, 6400)
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr"))

    def test_checked_in_fsrs3_scheduler_eval_config_uses_native_sweep(
        self,
    ) -> None:
        config = ExperimentConfig.from_toml(
            REPO_ROOT
            / "experiments/rl_scheduler/configs/"
            / "fsrs3_scheduler_users_1_8.toml"
        )

        self.assertEqual(config.name, "fsrs3_scheduler_users_1_8")
        self.assertNotIn(StageName.TRAIN_OVERFIT, config.stages)
        self.assertEqual(config.baseline.scheduler, "fsrs6")
        self.assertEqual(config.baseline.environments, ("fsrs6", "lstm"))
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs3",))
        self.assertEqual(config.sweep_batched.max_lanes_per_batch, 8192)
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs3"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs3:fsrs6",))

    def test_experiment_config_loads_new_workflow_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "workflow.toml"
            _write_workflow_config(config_path, log_dir=root / "logs")

            config = ExperimentConfig.from_toml(config_path)

        self.assertEqual(config.stages[-2], StageName.BUILD_PARETO)
        self.assertEqual(config.stages[-1], StageName.ANALYZE_PARETO)
        self.assertEqual(config.baseline.environments, ("fsrs6", "lstm"))
        self.assertEqual(config.sweep_batched.schedulers, ("fsrs6_adr",))
        self.assertEqual(config.build_pareto.schedulers, ("fsrs6", "fsrs6_adr"))
        self.assertEqual(config.analyze_pareto.comparisons, ("fsrs6_adr:fsrs6",))

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
        self.assertEqual(args.sched, "fsrs6,fsrs6_adr")
        self.assertTrue(
            any(
                "simulation_results_retention_sweep_user_1.json" in item
                for item in command
            )
        )
        self.assertIn("--hide-labels", command)

    def test_native_scheduler_sweep_does_not_require_train_overfit_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "native_fsrs3.toml"
            _write_native_fsrs3_eval_config(config_path)

            with patch(
                "simulator.experiment_infra.runner._run_configured_batched_retention_sweep",
                return_value={
                    "batch_lane_count": 4,
                    "log_paths": [],
                    "lane_results": [
                        {
                            "source": "configured-batched-retention",
                            "scheduler": "fsrs3",
                            "exit_code": 0,
                        }
                    ],
                },
            ) as run_batched:
                result = run_stage(
                    config_path=config_path,
                    stage=StageName.SWEEP,
                    repo_root=REPO_ROOT,
                    run_id="native-fsrs3",
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.summary["batch_runs_succeeded"], 1)
        self.assertEqual(result.summary["input_artifact_paths"], [])
        run_batched.assert_called_once()

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
                            "scheduler": "fsrs6_adr",
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

        self.assertIn("fsrs6_adr - fsrs6", report)
        self.assertIn("### Same-user same-DR dominance", report)
        self.assertIn("| fsrs6_adr - fsrs6 | 1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 |", report)
        self.assertIn("Loaded 2 records", report)

    def test_analyze_scheduler_comparison_reports_no_dr_ap_hypervolume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "pareto"
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
                            "scheduler": "fsrs6_ap",
                            "user_id": 1,
                            "desired_retention": None,
                            "fsrs6_ap_baseline_desired_retention": None,
                            "fsrs6_ap_policy": "/tmp/policy_0/policy.json",
                            "title": "AP policy_0",
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
                [
                    "--log-dir",
                    str(log_dir),
                    "--env",
                    "fsrs6",
                    "--sched",
                    "fsrs6,fsrs6_ap",
                    "--comparisons",
                    "fsrs6_ap:fsrs6",
                    "--start-user",
                    "1",
                    "--end-user",
                    "1",
                    "--start-retention",
                    "0.50",
                    "--end-retention",
                    "0.52",
                ]
            )
            report = render_report(args)

        self.assertIn("### Hypervolume vs FSRS6 baseline", report)
        self.assertIn("| user | baseline HV | fsrs6_ap HV | HV delta |", report)
        self.assertIn("| 1 | 0.25 | 0.75 | 0.50 | 1 |", report)

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
