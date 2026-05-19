# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra.baseline_dr_selection import (
    load_baseline_dr_manifest,
)
from simulator.experiment_infra import StageName
from simulator.experiment_infra.runner import run_stage
from simulator.experiment_infra.schemas import ExperimentConfig
from experiments.rl_scheduler.policy_search_common import PolicySearchSettings
from experiments.rl_scheduler.portfolio_training_common import (
    _baseline_dr_values_by_job,
)


def _write_manifest(path: Path, values: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_count": len(values),
                "selection_environment": "fsrs6",
                "reference": "uniform_anchor",
                "users": [
                    {
                        "user_id": 1,
                        "desired_retention_values": values,
                        "objective": {"hypervolume": 1.0},
                        "reference_point": {
                            "memorized_average": 0.0,
                            "negative_time_average": -1.0,
                        },
                        "selected_metrics": [],
                        "optimizer": {"name": "cma_es"},
                        "config_snapshot": {"name": "test"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_workflow_config(path: Path, *, log_root: Path, manifest: Path) -> None:
    path.write_text(
        f"""
schema_version = 1
name = "baseline-manifest-stage-test"
family = "rl_scheduler"
seed = 42
output_root = "{(path.parent / "out").as_posix()}"
stages = ["stage-baseline"]

[users]
train = [1]
validation = []
reserved_test = []

[baseline]
scheduler = "fsrs6"
log_root = "{log_root.as_posix()}"
expected_engine = "batched"
stage_mode = "copy"
environments = ["fsrs6"]

[baseline_dr_selection]
manifest = "{manifest.as_posix()}"
selection_environment = "fsrs6"
target_count = 2
reference = "uniform_anchor"
population_size = 32
generations = 10

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
write_performance_summary = false

[training]
lambda_grid = [0.0]

[training.policy_search]
retention_min = 0.50
retention_max = 0.98
baseline_desired_retention = 0.90
torch_device = "cpu"
""".lstrip(),
        encoding="utf-8",
    )


def _write_baseline_log(path: Path, *, desired_retention: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "meta",
        "data": {
            "engine": "batched",
            "days": 2,
            "deck_size": 10,
            "learn_limit": 1,
            "review_limit": 10,
            "cost_limit_minutes": 60.0,
            "priority": "new-first",
            "environment": "fsrs6",
            "scheduler_priority": "low_retrievability",
            "seed": 42,
            "fuzz": False,
            "short_term": False,
            "short_term_source": None,
            "scheduler": "fsrs6",
            "user_id": 1,
            "desired_retention": desired_retention,
            "review_markov_transition": False,
        },
    }
    path.write_text(json.dumps(meta) + "\n", encoding="utf-8")


def _portfolio_config(output_root: Path, manifest_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "portfolio-manifest-test",
            "family": "rl_scheduler",
            "seed": 42,
            "output_root": str(output_root),
            "stages": ["train-overfit"],
            "users": {"train": [1, 2], "validation": [], "reserved_test": []},
            "baseline": {
                "scheduler": "fsrs6",
                "log_root": str(output_root / "logs"),
                "expected_engine": "batched",
                "stage_mode": "copy",
            },
            "baseline_dr_selection": {
                "manifest": str(manifest_path),
                "selection_environment": "fsrs6",
                "target_count": 2,
                "reference": "uniform_anchor",
                "population_size": 32,
                "generations": 10,
            },
            "simulation": {
                "engine": "batched",
                "environment": "fsrs6",
                "days": 2,
                "deck": 10,
                "learn_limit": 1,
                "review_limit": 10,
                "cost_limit_minutes": 60.0,
                "priority": "new-first",
                "scheduler_priority": "low_retrievability",
                "fuzz": False,
            },
            "gpu_guard": {"required": False, "device": "cpu", "smoke": False},
            "performance": {"device": "cpu", "write_performance_summary": False},
            "training": {
                "policy_search": {
                    "retention_min": 0.50,
                    "retention_max": 0.98,
                    "baseline_desired_retention": 0.90,
                    "torch_device": "cpu",
                },
                "portfolio": {"portfolio_size": 2},
            },
        },
        config_path=output_root / "config.toml",
    )


class BaselineDRSelectionManifestTests(unittest.TestCase):
    def test_exact_per_user_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.51, 0.63, 0.77])

            manifest = load_baseline_dr_manifest(path, target_count=3, user_ids=[1])

        self.assertEqual(manifest.values_for_user(1), (0.51, 0.63, 0.77))
        self.assertTrue(manifest.contains_value(1, 0.63))
        self.assertFalse(manifest.contains_value(1, 0.64))

    def test_missing_user_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.51, 0.63])

            with self.assertRaisesRegex(ValueError, "missing users: 2"):
                load_baseline_dr_manifest(path, target_count=2, user_ids=[1, 2])

    def test_wrong_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.51, 0.63])

            with self.assertRaisesRegex(ValueError, "target_count expected 3"):
                load_baseline_dr_manifest(path, target_count=3, user_ids=[1])

    def test_top_level_target_count_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "target_count": 3,
                        "users": [
                            {
                                "user_id": 1,
                                "desired_retention_values": [0.51, 0.63],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must contain 3 values"):
                load_baseline_dr_manifest(path, user_ids=[1])

    def test_duplicate_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.51, 0.51, 0.63])

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_baseline_dr_manifest(path, target_count=3, user_ids=[1])

    def test_out_of_bound_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.51, 1.0])

            with self.assertRaisesRegex(ValueError, "0.0 < value < 1.0"):
                load_baseline_dr_manifest(path, target_count=2, user_ids=[1])

    def test_unsorted_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            _write_manifest(path, [0.63, 0.51])

            with self.assertRaisesRegex(ValueError, "sorted"):
                load_baseline_dr_manifest(path, target_count=2, user_ids=[1])

    def test_stage_baseline_stages_only_manifest_selected_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, [0.51, 0.73])
            log_root = root / "logs"
            user_dir = log_root / "user_1"
            _write_baseline_log(
                user_dir
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.51.jsonl",
                desired_retention=0.51,
            )
            _write_baseline_log(
                user_dir
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.73.jsonl",
                desired_retention=0.73,
            )
            _write_baseline_log(
                user_dir
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.90.jsonl",
                desired_retention=0.90,
            )
            config_path = root / "config.toml"
            _write_workflow_config(
                config_path,
                log_root=log_root,
                manifest=manifest_path,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=REPO_ROOT,
                run_id="manifest-stage",
            )

        self.assertEqual(result.exit_code, 0, result.summary["notes"])
        self.assertEqual(len(result.summary["staged_logs"]), 2)
        self.assertEqual(
            result.summary["matched_retentions_by_user"]["1"],
            [0.51, 0.73],
        )
        self.assertTrue(
            all("0.90" not in path for path in result.summary["staged_logs"])
        )
        self.assertNotIn("0.9", "\n".join(result.summary["notes"]))

    def test_stage_baseline_matches_rounded_filename_boundary_drs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, [0.5002, 0.9796])
            log_root = root / "logs"
            user_dir = log_root / "user_1"
            _write_baseline_log(
                user_dir
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.50.jsonl",
                desired_retention=0.5002,
            )
            _write_baseline_log(
                user_dir
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.98.jsonl",
                desired_retention=0.9796,
            )
            config_path = root / "config.toml"
            _write_workflow_config(
                config_path,
                log_root=log_root,
                manifest=manifest_path,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=REPO_ROOT,
                run_id="manifest-rounded-boundary",
            )

        self.assertEqual(result.exit_code, 0, result.summary["notes"])
        self.assertEqual(
            result.summary["matched_retentions_by_user"]["1"],
            [0.5002, 0.9796],
        )

    def test_stage_baseline_keeps_same_rounded_filename_drs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            values = [0.8267075195512398, 0.8296779378941778]
            _write_manifest(manifest_path, values)
            log_root = root / "logs"
            user_dir = log_root / "user_1" / "sched_fsrs6"
            for value in values:
                _write_baseline_log(
                    user_dir
                    / f"dr_{str(value).replace('.', 'p')}"
                    / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.83.jsonl",
                    desired_retention=value,
                )
            config_path = root / "config.toml"
            _write_workflow_config(
                config_path,
                log_root=log_root,
                manifest=manifest_path,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=REPO_ROOT,
                run_id="manifest-same-rounded",
            )

        self.assertEqual(result.exit_code, 0, result.summary["notes"])
        self.assertEqual(len(result.summary["staged_logs"]), 2)
        self.assertEqual(
            result.summary["matched_retentions_by_user"]["1"],
            values,
        )
        self.assertEqual(
            len({Path(path).parent.name for path in result.summary["staged_logs"]}),
            2,
        )

    def test_stage_baseline_reports_missing_manifest_selected_dr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, [0.51, 0.73])
            log_root = root / "logs"
            _write_baseline_log(
                log_root
                / "user_1"
                / "env=fsrs6_sched=fsrs6_engine=batched_prio=new-first_seed=42_ret=0.51.jsonl",
                desired_retention=0.51,
            )
            config_path = root / "config.toml"
            _write_workflow_config(
                config_path,
                log_root=log_root,
                manifest=manifest_path,
            )

            result = run_stage(
                config_path=config_path,
                stage=StageName.STAGE_BASELINE,
                repo_root=REPO_ROOT,
                run_id="manifest-missing",
            )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("ret=0.73", "\n".join(result.summary["notes"]))

    def test_portfolio_common_reads_per_job_manifest_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "target_count": 2,
                        "users": [
                            {
                                "user_id": 1,
                                "desired_retention_values": [0.51, 0.73],
                            },
                            {
                                "user_id": 2,
                                "desired_retention_values": [0.52, 0.74],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = _portfolio_config(root, manifest_path)
            jobs = [
                type("Job", (), {"user_id": 1})(),
                type("Job", (), {"user_id": 2})(),
            ]

            values = _baseline_dr_values_by_job(
                jobs=jobs,
                config=config,
                repo_root=REPO_ROOT,
                fallback=(0.9,),
                settings=PolicySearchSettings.from_mapping(
                    config.training_policy_search
                ),
            )

        self.assertEqual(values, [(0.51, 0.73), (0.52, 0.74)])


if __name__ == "__main__":
    unittest.main()
