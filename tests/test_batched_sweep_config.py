# ruff: noqa: E402
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.run_sweep_users_batched import main as batched_main
from simulator.batched_sweep.config import load_batched_sweep_config
from simulator.batched_sweep.execution import run_batches
from simulator.batched_sweep.logging import BatchedSweepLogLane
from simulator.batched_sweep.plan import build_batched_sweep_plan
from simulator.batched_sweep.runner import (
    BatchedSweepContext,
    _build_mixed_scheduler_ops,
    _build_sweep_lanes,
    _split_lanes,
)
from simulator.batched_sweep.fsrs6_adr_policy import (
    format_float_token,
    resolve_fsrs6_adr_policy_specs,
)
from simulator.batched_sweep.fsrs6_cost_adr_policy import (
    DEFAULT_COST_WEIGHTS as DEFAULT_COST_ADR_COST_WEIGHTS,
    resolve_fsrs6_cost_adr_policy_specs,
)
from simulator.batched_sweep.fsrs6_ap_policy import (
    resolve_fsrs6_ap_policy_specs,
)
from simulator.batched_sweep.anki_sm2_ap_policy import (
    resolve_anki_sm2_ap_policy_specs,
)
from simulator.defaults import DEFAULT_MAX_LANES_PER_BATCH
from simulator.fsrs_defaults import DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS6_WEIGHTS
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.fsrs6_cost_conditioned_adr_policy import (
    ACTION_HEAD_INTERVAL,
    FEATURE_VERSION_INTERVAL_MONO as COST_ADR_FEATURE_VERSION_INTERVAL_MONO,
    FSRS6CostConditionedADRPolicy,
)
from simulator.fsrs6_ap_policy import FSRS6APPolicy
from simulator.anki_sm2_ap_policy import AnkiSM2APPolicy
from tests.lstm_batch_helpers import dummy_lstm_weights


def _valid_config(log_dir: Path) -> str:
    return f"""
schema_version = 1

[users]
start = 1
end = 2

[sweep]
envs = ["lstm"]
schedulers = ["fsrs6", "anki_sm2"]

[retention]
start = 0.50
end = 0.52
step = 0.02

[simulation]
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
seed = 42
priority = "review-first"
scheduler_priority = "low_retrievability"
fuzz = false

[paths]
log_dir = "{log_dir.as_posix()}"
benchmark_partition = "0"

[execution]
batch_size = 2

[logging]
no_log = true
no_progress = true
""".lstrip()


def _write_policy(path: Path, *, dr: float, offset: float = 0.0) -> None:
    base = FSRS6ADRPolicy.baseline(desired_retention=dr)
    policy = FSRS6ADRPolicy(
        coefficients=(base.coefficients[0] + offset, *base.coefficients[1:]),
        baseline_desired_retention=dr,
    )
    policy.write_json(path)


def _write_ap_policy(path: Path, *, dr: float, offset: float = 0.0) -> None:
    search_vector = [0.0] * 21
    search_vector[0] = offset
    policy = FSRS6APPolicy.from_search_vector(
        base_weights=DEFAULT_FSRS6_WEIGHTS,
        search_vector=search_vector,
        baseline_desired_retention=dr,
    )
    policy.write_json(path)


def _write_cost_adr_policy(path: Path, *, offset: float = 0.0) -> None:
    coefficients = [0.0] * 24
    coefficients[0] = offset
    policy = FSRS6CostConditionedADRPolicy(
        coefficients=tuple(coefficients),
        action_head=ACTION_HEAD_INTERVAL,
        feature_version=COST_ADR_FEATURE_VERSION_INTERVAL_MONO,
        max_interval_days=36500.0,
    )
    policy.write_json(path)


def _write_anki_sm2_ap_policy(path: Path, *, offset: float = 0.0) -> None:
    search_vector = [0.0] * 7
    search_vector[0] = offset
    policy = AnkiSM2APPolicy.from_search_vector(search_vector=search_vector)
    policy.write_json(path)


def _args(log_dir: Path, policy_path: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        user_ids=None,
        start_user=1,
        end_user=2,
        batch_size=2,
        max_lanes_per_batch=None,
        torch_device=None,
        cuda_devices=None,
        srs_benchmark_root=None,
        benchmark_result=None,
        benchmark_partition="0",
        log_dir=log_dir,
        log_layout="user",
        start_retention=0.50,
        end_retention=0.52,
        step=0.02,
        days=2,
        diagnostic_csv_logs=False,
        fsrs6_adr_policy=policy_path,
        fsrs6_adr_policy_root=None,
        fsrs6_adr_train_run_root=None,
        fsrs6_adr_policy_manifest=None,
        fsrs6_adr_lambda_values=None,
        fsrs6_ap_policy=None,
        fsrs6_ap_policy_root=None,
        fsrs6_ap_train_run_root=None,
        fsrs6_ap_policy_manifest=None,
        fsrs6_ap_lambda_values=None,
        anki_sm2_ap_policy=None,
        anki_sm2_ap_policy_root=None,
        anki_sm2_ap_train_run_root=None,
        anki_sm2_ap_policy_manifest=None,
    )


class BatchedSweepConfigTests(unittest.TestCase):
    def test_loads_valid_range_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            path.write_text(_valid_config(log_dir), encoding="utf-8")

            config = load_batched_sweep_config(path)

        self.assertIsNone(config.args.user_ids)
        self.assertEqual(config.args.start_user, 1)
        self.assertEqual(config.args.end_user, 2)
        self.assertEqual(config.envs, ("lstm",))
        self.assertEqual(config.schedulers, ("fsrs6", "anki_sm2"))
        self.assertEqual(config.args.log_layout, "user")
        self.assertFalse(config.args.diagnostic_csv_logs)
        self.assertFalse(config.args.review_markov_transition)

    def test_loads_review_markov_transition_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            path.write_text(
                _valid_config(log_dir).replace(
                    "fuzz = false",
                    "fuzz = false\nreview_markov_transition = true",
                ),
                encoding="utf-8",
            )

            config = load_batched_sweep_config(path)

        self.assertTrue(config.args.review_markov_transition)

    def test_loads_rl_scheduler_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_adr_linear_cmaes_users_1_8.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_adr",))
        self.assertEqual(config.args.log_dir, REPO_ROOT / "logs" / "retention_sweep")
        self.assertEqual(config.args.batch_size, 8)
        self.assertEqual(config.args.torch_device, "cuda")
        self.assertEqual(config.args.fsrs6_adr_lambda_values, (0.5,))
        self.assertFalse(config.args.no_progress)

    def test_loads_oracle_distill_portfolio_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_oracle_stationary_finite_distill",))
        self.assertEqual(
            config.args.fsrs6_oracle_stationary_finite_distill_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8"
            / "fsrs6_oracle_stationary_finite_distill_portfolio_users_1_8_pop16_v1_markov_off",
        )
        self.assertIsNone(config.args.fsrs6_oracle_stationary_finite_distill_policy)
        self.assertFalse(config.args.no_progress)

    def test_loads_cost_adr_cmaes_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_cmaes_users_1_8"
            / "fsrs6_cost_adr_cmaes_users_1_8_pop16_gen20_v1_markov_off",
        )

    def test_loads_cost_adr_meaninit16w_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_meaninit16w_users_1_8"
            / "fsrs6_cost_adr_meaninit16w_users_1_8_pop16_gen20_v1_markov_off",
        )

    def test_loads_cost_adr_schedhv_stdpre_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_8"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off",
        )

    def test_loads_cost_adr_retention_head_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8"
            / "fsrs6_cost_adr_rethead_schedhv_stdpre_users_1_8_pop16_gen20_v1_markov_off",
        )

    def test_loads_cost_adr_retention_head_interval_init_wide_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8"
            / "fsrs6_cost_adr_rethead_intervalinit_wide_users_1_8_pop16_gen20_v1_markov_off",
        )

    def test_loads_adr_pop16_users_1_128_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_adr_portfolio_users_1_128_pop16_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 129)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_adr",))
        self.assertEqual(
            config.args.fsrs6_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_adr_portfolio_users_1_128_pop16"
            / "fsrs6_adr_portfolio_users_1_128_pop16_v1_markov_off",
        )

    def test_loads_cost_adr_schedhv_stdpre_users_1_128_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 129)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("fsrs6_cost_adr",))
        self.assertEqual(
            config.args.fsrs6_cost_adr_cost_weights,
            DEFAULT_COST_ADR_COST_WEIGHTS,
        )
        self.assertEqual(
            config.args.fsrs6_cost_adr_train_run_root,
            REPO_ROOT
            / "artifacts"
            / "rl_scheduler"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_128"
            / "fsrs6_cost_adr_schedhv_stdpre_users_1_128_pop16_gen20_v1_markov_off",
        )

    def test_dry_run_accepts_rl_scheduler_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            output_root = root / "artifacts"
            run_root = output_root / "test-run"
            policy_root = run_root / "train-overfit" / "train_outputs"
            artifact_paths: list[str] = []
            for user_id in (1, 2):
                for desired_retention in (0.50, 0.52):
                    policy_path = (
                        policy_root
                        / f"user_{user_id}"
                        / f"dr_{format_float_token(desired_retention)}"
                        / "lambda_0p5"
                        / "policy.json"
                    )
                    policy_path.parent.mkdir(parents=True)
                    _write_policy(policy_path, dr=desired_retention)
                    metadata_path = policy_path.parent / "metadata.json"
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "scheduler_name": "fsrs6_adr",
                                "training_user_ids": [user_id],
                                "policy_path": "policy.json",
                                "baseline_desired_retention": desired_retention,
                                "lambda_value": 0.5,
                            }
                        ),
                        encoding="utf-8",
                    )
                    artifact_paths.append(str(metadata_path))
            summary_path = run_root / "train-overfit" / "training_summary.json"
            summary_path.write_text(
                json.dumps({"passed": True, "artifact_paths": artifact_paths}),
                encoding="utf-8",
            )
            path = root / "experiment.toml"
            path.write_text(
                f"""
schema_version = 1
name = "rl-experiment-sweep"
family = "rl_scheduler"
seed = 42
output_root = "{output_root.as_posix()}"
stages = ["sweep"]

[users]
train = [1, 2]

[simulation]
engine = "batched"
environment = "fsrs6"
days = 2
deck = 10
learn_limit = 1
review_limit = 10
cost_limit_minutes = 60.0
priority = "review-first"
scheduler_priority = "low_retrievability"
fuzz = false

[performance]
device = "cpu"
diagnostic_csv_logs = false

[training]
lambda_grid = [0.5]

[sweep]
envs = ["fsrs6"]
schedulers = ["fsrs6_adr"]
log_dir = "{log_dir.as_posix()}"
log_layout = "user"
batch_size = 2
torch_device = "cpu"
start_retention = 0.50
end_retention = 0.52
step = 0.02
no_progress = true
no_log = true
""".lstrip(),
                encoding="utf-8",
            )
            config = load_batched_sweep_config(path)
            self.assertEqual(config.args.run_id, "test-run")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = batched_main(["--config", str(path), "--dry-run"])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Batched sweep dry run", output)
        self.assertIn("expanded lanes: 4", output)

    def test_loads_explicit_sweep_log_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            raw = _valid_config(Path(tmp) / "logs").replace(
                "[logging]\n",
                '[logging]\nlog_layout = "sweep"\n',
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)

        self.assertEqual(config.args.log_layout, "sweep")

    def test_fsrs6_manifest_replaces_uniform_dr_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sweep.toml"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "target_count": 2,
                        "users": [
                            {
                                "user_id": 1,
                                "desired_retention_values": [0.51, 0.61],
                            },
                            {
                                "user_id": 2,
                                "desired_retention_values": [0.52, 0.62],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            raw = (
                _valid_config(root / "logs")
                .replace('schedulers = ["fsrs6", "anki_sm2"]', 'schedulers = ["fsrs6"]')
                .replace("end = 0.52", "end = 0.54")
                + f'\n[fsrs6]\ndr_manifest = "{manifest_path.as_posix()}"\n'
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=config.args,
                envs=list(config.envs),
                schedulers=list(config.schedulers),
            )
            lanes = _build_sweep_lanes(
                batch=plan.batches[0],
                ctx=plan.ctx,
                environment="lstm",
            )

        self.assertEqual(config.args.fsrs6_dr_manifest, manifest_path)
        self.assertEqual(plan.total_lanes, 4)
        self.assertEqual(
            [(lane.user_id, lane.desired_retention) for lane in lanes],
            [(1, 0.51), (1, 0.61), (2, 0.52), (2, 0.62)],
        )

    def test_defaults_to_single_user_batch_with_lane_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            raw = (
                _valid_config(log_dir)
                .replace("end = 2", "end = 3")
                .replace("batch_size = 2\n", "")
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=config.args,
                envs=list(config.envs),
                schedulers=list(config.schedulers),
            )

        self.assertIsNone(config.args.batch_size)
        self.assertEqual(config.args.max_lanes_per_batch, DEFAULT_MAX_LANES_PER_BATCH)
        self.assertEqual(plan.batches, [[1, 2, 3]])

    def test_auto_user_batches_respect_lane_cap_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            raw = (
                _valid_config(log_dir)
                .replace("end = 2", "end = 5")
                .replace("batch_size = 2\n", "max_lanes_per_batch = 6\n")
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=config.args,
                envs=list(config.envs),
                schedulers=list(config.schedulers),
            )

        self.assertIsNone(config.args.batch_size)
        self.assertEqual(plan.batches, [[1, 2], [3, 4], [5]])
        self.assertEqual(plan.total_lanes, 15)

    def test_environment_overrides_build_separate_user_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            raw = (
                _valid_config(log_dir)
                .replace("end = 2", "end = 5")
                .replace('envs = ["lstm"]', 'envs = ["fsrs6", "lstm"]')
                .replace(
                    "batch_size = 2\n",
                    (
                        "max_lanes_per_batch = 100\n\n"
                        "[execution.env_overrides.fsrs6]\n"
                        "max_lanes_per_batch = 12\n\n"
                        "[execution.env_overrides.lstm]\n"
                        "max_lanes_per_batch = 6\n\n"
                    ),
                )
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=config.args,
                envs=list(config.envs),
                schedulers=list(config.schedulers),
            )

        self.assertEqual(
            config.args.env_batch_overrides["fsrs6"]["max_lanes_per_batch"],
            12,
        )
        self.assertEqual(plan.batches_by_env["fsrs6"], [[1, 2, 3, 4], [5]])
        self.assertEqual(plan.batches_by_env["lstm"], [[1, 2], [3, 4], [5]])
        self.assertEqual(plan.max_lanes_per_batch_by_env["fsrs6"], 12)
        self.assertEqual(plan.max_lanes_per_batch_by_env["lstm"], 6)
        self.assertEqual(plan.total_lanes, 30)

    def test_run_batches_dispatches_environment_specific_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6", "lstm"],
                schedulers=["fsrs6"],
                dr_values=[0.50],
            )

            with patch(
                "simulator.batched_sweep.execution.run_batch_core"
            ) as run_batch_core:
                run_batches(
                    args=_args(Path(tmp) / "logs"),
                    ctx=ctx,
                    batches=[[1, 2]],
                    batches_by_env={
                        "fsrs6": [[1, 2, 3]],
                        "lstm": [[1], [2]],
                    },
                    devices=[],
                    device=torch.device("cpu"),
                    overall=None,
                )

        self.assertEqual(
            [
                (call.kwargs["batch"], call.kwargs["envs"])
                for call in run_batch_core.call_args_list
            ],
            [
                ([1, 2, 3], ["fsrs6"]),
                ([1], ["lstm"]),
                ([2], ["lstm"]),
            ],
        )

    def test_rejects_invalid_log_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            raw = _valid_config(Path(tmp) / "logs").replace(
                "[logging]\n",
                '[logging]\nlog_layout = "bad"\n',
            )
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "logging.log_layout"):
                load_batched_sweep_config(path)

    def test_loads_valid_ids_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            raw = _valid_config(Path(tmp) / "logs").replace(
                "start = 1\nend = 2", "ids = [1, 3]"
            )
            path.write_text(raw, encoding="utf-8")

            config = load_batched_sweep_config(path)

        self.assertEqual(config.args.user_ids, [1, 3])
        self.assertEqual(config.args.start_user, 1)
        self.assertEqual(config.args.end_user, 3)

    def test_rejects_mixed_user_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            raw = _valid_config(Path(tmp) / "logs").replace(
                "start = 1\nend = 2", "ids = [1]\nstart = 1\nend = 2"
            )
            path.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "users.ids"):
                load_batched_sweep_config(path)

    def test_rejects_invalid_env_scheduler_and_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            bad_retention = Path(tmp) / "bad_retention.toml"
            bad_retention.write_text(
                _valid_config(log_dir).replace("end = 0.52", "end = 1.00"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Retention grid"):
                load_batched_sweep_config(bad_retention)

            bad_env = Path(tmp) / "bad_env.toml"
            bad_env.write_text(
                _valid_config(log_dir).replace('envs = ["lstm"]', 'envs = ["dash"]'),
                encoding="utf-8",
            )
            config = load_batched_sweep_config(bad_env)
            with self.assertRaisesRegex(ValueError, "supports only"):
                build_batched_sweep_plan(
                    repo_root=REPO_ROOT,
                    args=config.args,
                    envs=list(config.envs),
                    schedulers=list(config.schedulers),
                )

            bad_sched = Path(tmp) / "bad_sched.toml"
            bad_sched.write_text(
                _valid_config(log_dir).replace(
                    'schedulers = ["fsrs6", "anki_sm2"]',
                    'schedulers = ["dash"]',
                ),
                encoding="utf-8",
            )
            config = load_batched_sweep_config(bad_sched)
            with self.assertRaisesRegex(ValueError, "Unsupported scheduler"):
                build_batched_sweep_plan(
                    repo_root=REPO_ROOT,
                    args=config.args,
                    envs=list(config.envs),
                    schedulers=list(config.schedulers),
                )

    def test_config_plan_matches_equivalent_cli_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            log_dir = Path(tmp) / "logs"
            path.write_text(_valid_config(log_dir), encoding="utf-8")
            config = load_batched_sweep_config(path)

            config_plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=config.args,
                envs=list(config.envs),
                schedulers=list(config.schedulers),
            )
            cli_plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_args(log_dir),
                envs=["lstm"],
                schedulers=["fsrs6", "anki_sm2"],
            )

        self.assertEqual(config_plan.batches, cli_plan.batches)
        self.assertEqual(config_plan.total_lanes, cli_plan.total_lanes)
        self.assertEqual(config_plan.total_user_days, cli_plan.total_user_days)

    def test_dry_run_emits_lane_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.toml"
            path.write_text(_valid_config(Path(tmp) / "logs"), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                code = batched_main(["--config", str(path), "--dry-run"])

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("Batched sweep dry run", output)
        self.assertIn("expanded lanes: 6", output)
        self.assertIn("log layout: user", output)
        self.assertIn("example log dir:", output)


class FSRS6ADRPolicyExpansionTests(unittest.TestCase):
    def test_single_policy_preserves_current_lane_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["lstm"],
                schedulers=["fsrs6_adr"],
                dr_values=[0.50, 0.52],
                fsrs6_adr_policy=policy_path,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(lanes), 2)
        self.assertTrue(all(lane.fsrs6_adr_policy == policy_path for lane in lanes))
        self.assertTrue(
            all(lane.fsrs6_adr_baseline_desired_retention is None for lane in lanes)
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_adr/policy_policy",
                "user_2/sched_fsrs6_adr/policy_policy",
            ],
        )

    def test_policy_root_expands_user_dr_lambda_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for user_id in (1, 2):
                for dr in (0.50, 0.52):
                    policy_path = (
                        root
                        / f"user_{user_id}"
                        / "lambda_0p5"
                        / f"dr_{format_float_token(dr)}"
                        / "policy.json"
                    )
                    _write_policy(policy_path, dr=dr)

            specs = resolve_fsrs6_adr_policy_specs(
                user_ids=[1, 2],
                dr_values=[0.50, 0.52],
                policy_root=root,
                lambda_values=[0.5],
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["lstm"],
                schedulers=["fsrs6_adr"],
                dr_values=[0.50, 0.52],
                fsrs6_adr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [
                (
                    lane.user_id,
                    lane.fsrs6_adr_baseline_desired_retention,
                    lane.fsrs6_adr_lambda_value,
                )
                for lane in lanes
            ],
            [
                (1, 0.50, 0.5),
                (1, 0.52, 0.5),
                (2, 0.50, 0.5),
                (2, 0.52, 0.5),
            ],
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_adr/dr_0p5/lambda_0p5",
                "user_1/sched_fsrs6_adr/dr_0p52/lambda_0p5",
                "user_2/sched_fsrs6_adr/dr_0p5/lambda_0p5",
                "user_2/sched_fsrs6_adr/dr_0p52/lambda_0p5",
            ],
        )

    def test_policy_root_expands_lambda_less_portfolio_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index in range(2):
                policy_dir = root / "user_1" / "policies" / f"policy_{index}"
                policy_dir.mkdir(parents=True)
                FSRS6ADRPolicy(
                    coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_desired_retention=None,
                ).write_json(policy_dir / "policy.json")
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_adr",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "portfolio_index": index,
                            "action_space": "sd_retention_function_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_adr_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_adr"],
                dr_values=[0.50, 0.52],
                fsrs6_adr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 2)
        self.assertEqual([spec.lambda_value for spec in specs], [None, None])
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_adr/policy_0",
                "user_1/sched_fsrs6_adr/policy_1",
            ],
        )

    def test_default_adr_uses_adr_policy_source_and_scheduler_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index in range(2):
                policy_dir = root / "user_1" / "policies" / f"policy_{index}"
                policy_dir.mkdir(parents=True)
                FSRS6ADRPolicy(
                    coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_desired_retention=None,
                ).write_json(policy_dir / "policy.json")
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_default_adr",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "portfolio_index": index,
                            "action_space": "sd_retention_function_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_adr_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_default_adr"],
                dr_values=[0.50, 0.52],
                fsrs6_adr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 2)
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_default_adr/policy_0",
                "user_1/sched_fsrs6_default_adr/policy_1",
            ],
        )
        self.assertTrue(
            all(lane.scheduler_name == "fsrs6_default_adr" for lane in lanes)
        )

    def test_time_adr_uses_adr_policy_source_and_scheduler_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index in range(2):
                policy_dir = root / "user_1" / "policies" / f"policy_{index}"
                policy_dir.mkdir(parents=True)
                FSRS6ADRPolicy(
                    coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_desired_retention=None,
                    feature_version="fsrs6_adr_log_poly_time_v1",
                ).write_json(policy_dir / "policy.json")
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_adr_time",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "portfolio_index": index,
                            "action_space": "sdt_retention_function_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_adr_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_adr_time"],
                dr_values=[0.50, 0.52],
                fsrs6_adr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 2)
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_adr_time/policy_0",
                "user_1/sched_fsrs6_adr_time/policy_1",
            ],
        )
        self.assertTrue(all(lane.scheduler_name == "fsrs6_adr_time" for lane in lanes))

    def test_policy_manifest_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "policies" / "u1_dr05" / "policy.json"
            _write_policy(policy_path, dr=0.50)
            manifest = root / "policies.toml"
            manifest.write_text(
                """
[[policies]]
user_id = 1
baseline_desired_retention = 0.50
lambda_value = 0.5
path = "policies/u1_dr05/policy.json"
""".lstrip(),
                encoding="utf-8",
            )

            specs = resolve_fsrs6_adr_policy_specs(
                user_ids=[1],
                dr_values=[0.50],
                policy_manifest=manifest,
            )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].path, policy_path.resolve())
        self.assertEqual(specs[0].lambda_value, 0.5)

    def test_policy_manifest_mismatch_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "policy.json"
            _write_policy(policy_path, dr=0.60)
            manifest = root / "policies.toml"
            manifest.write_text(
                """
[[policies]]
user_id = 1
baseline_desired_retention = 0.50
path = "policy.json"
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "baseline_desired_retention"):
                resolve_fsrs6_adr_policy_specs(
                    user_ids=[1],
                    dr_values=[0.50],
                    policy_manifest=manifest,
                )

    def test_policy_root_missing_expected_policy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            _write_policy(
                root / "user_1" / "lambda_0p5" / "dr_0p5" / "policy.json",
                dr=0.50,
            )

            with self.assertRaisesRegex(FileNotFoundError, "Missing FSRS6 ADR"):
                resolve_fsrs6_adr_policy_specs(
                    user_ids=[1],
                    dr_values=[0.50, 0.52],
                    policy_root=root,
                    lambda_values=[0.5],
                )

    def test_split_lanes_preserves_all_dr_lanes_for_each_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=root,
                overrides={},
                log_root=root / "logs",
                batch_log_root=root / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6"],
                dr_values=[0.50, 0.52, 0.54],
            )
            lanes = _build_sweep_lanes(
                batch=[1, 2, 3],
                ctx=ctx,
                environment="fsrs6",
            )

            chunks = _split_lanes(lanes, max_lanes_per_batch=4)

        self.assertEqual(
            [[lane.user_id for lane in chunk] for chunk in chunks],
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],
        )
        self.assertEqual(
            [[lane.desired_retention for lane in chunk] for chunk in chunks],
            [
                [0.50, 0.52, 0.54],
                [0.50, 0.52, 0.54],
                [0.50, 0.52, 0.54],
            ],
        )

    def test_lstm_multi_dr_lanes_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="fsrs6",
                    scheduler_name="lstm",
                    scheduler_spec="lstm",
                    desired_retention=0.50,
                    fixed_interval=None,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="fsrs6",
                    scheduler_name="lstm",
                    scheduler_spec="lstm",
                    desired_retention=0.52,
                    fixed_interval=None,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=None,
                fsrs_default_weights=None,
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=dummy_lstm_weights(2),
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 2)
        target = group.ops._target
        self.assertEqual(tuple(target.shape), (2,))
        self.assertEqual(
            [round(float(value), 2) for value in target.tolist()],
            [0.50, 0.52],
        )

    def test_fsrs3_multi_dr_lanes_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="fsrs6",
                    scheduler_name="fsrs3",
                    scheduler_spec="fsrs3",
                    desired_retention=0.50,
                    fixed_interval=None,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="fsrs6",
                    scheduler_name="fsrs3",
                    scheduler_spec="fsrs3",
                    desired_retention=0.52,
                    fixed_interval=None,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=None,
                fsrs_default_weights=None,
                fsrs3_weights=torch.tensor(
                    [DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS3_WEIGHTS],
                    dtype=torch.float32,
                ),
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 2)
        interval_factor = group.ops._interval_factor
        self.assertEqual(tuple(interval_factor.shape), (2,))
        self.assertGreater(float(interval_factor[0]), float(interval_factor[1]))

    def test_multiple_adr_policies_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.json"
            second = root / "p2.json"
            _write_policy(first, dr=0.50)
            _write_policy(second, dr=0.52, offset=-1.0)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="lstm",
                    scheduler_name="fsrs6_adr",
                    scheduler_spec="fsrs6_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_adr_policy=first,
                    fsrs6_adr_baseline_desired_retention=0.50,
                    fsrs6_adr_lambda_value=0.5,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="lstm",
                    scheduler_name="fsrs6_adr",
                    scheduler_spec="fsrs6_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_adr_policy=second,
                    fsrs6_adr_baseline_desired_retention=0.52,
                    fsrs6_adr_lambda_value=0.5,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS],
                    dtype=torch.float32,
                ),
                fsrs_default_weights=None,
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 2)

    def test_default_adr_scheduler_group_uses_default_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "policy.json"
            _write_policy(policy_path, dr=0.50)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "default",
                    environment="lstm",
                    scheduler_name="fsrs6_default_adr",
                    scheduler_spec="fsrs6_default_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_adr_policy=policy_path,
                )
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1],
                lanes=lanes,
                fsrs_weights=None,
                fsrs_default_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS],
                    dtype=torch.float32,
                ),
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 1)


class FSRS6CostADRPolicyExpansionTests(unittest.TestCase):
    def test_policy_root_expands_user_cost_weight_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for user_id in (1, 2):
                _write_cost_adr_policy(root / f"user_{user_id}" / "policy.json")

            specs = resolve_fsrs6_cost_adr_policy_specs(
                user_ids=[1, 2],
                cost_weights=[0.0, 4.0],
                policy_root=root,
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_cost_adr"],
                dr_values=[0.50, 0.52],
                fsrs6_cost_adr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [(spec.user_id, spec.cost_weight) for spec in specs],
            [(1, 0.0), (1, 4.0), (2, 0.0), (2, 4.0)],
        )
        self.assertEqual(
            [
                (
                    lane.user_id,
                    lane.fsrs6_cost_adr_policy,
                    lane.fsrs6_cost_adr_goal_cost_weight,
                )
                for lane in lanes
            ],
            [
                (1, specs[0].path, 0.0),
                (1, specs[1].path, 4.0),
                (2, specs[2].path, 0.0),
                (2, specs[3].path, 4.0),
            ],
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_cost_adr/policy_user_1/costw_0",
                "user_1/sched_fsrs6_cost_adr/policy_user_1/costw_4",
                "user_2/sched_fsrs6_cost_adr/policy_user_2/costw_0",
                "user_2/sched_fsrs6_cost_adr/policy_user_2/costw_4",
            ],
        )

    def test_multiple_cost_adr_lanes_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.json"
            second = root / "p2.json"
            _write_cost_adr_policy(first)
            _write_cost_adr_policy(second, offset=1.0)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="lstm",
                    scheduler_name="fsrs6_cost_adr",
                    scheduler_spec="fsrs6_cost_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_cost_adr_policy=first,
                    fsrs6_cost_adr_goal_cost_weight=0.0,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="lstm",
                    scheduler_name="fsrs6_cost_adr",
                    scheduler_spec="fsrs6_cost_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_cost_adr_policy=second,
                    fsrs6_cost_adr_goal_cost_weight=4.0,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=torch.tensor(
                    [DEFAULT_FSRS6_WEIGHTS, DEFAULT_FSRS6_WEIGHTS],
                    dtype=torch.float32,
                ),
                fsrs_default_weights=None,
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 2)
        self.assertEqual(tuple(group.ops._coefficients.shape), (2, 24))
        self.assertEqual(
            [round(float(value), 1) for value in group.ops._goal_cost_weight.tolist()],
            [0.0, 4.0],
        )


class FSRS6APPolicyExpansionTests(unittest.TestCase):
    def test_policy_root_expands_lambda_less_portfolio_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index, desired_retention in enumerate((0.83, 0.87)):
                policy_dir = root / "user_1" / "policies" / f"policy_{index}"
                policy_dir.mkdir(parents=True)
                _write_ap_policy(policy_dir / "policy.json", dr=desired_retention)
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_ap",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "scheduler_desired_retention": desired_retention,
                            "portfolio_index": index,
                            "action_space": "fsrs6_ap_weight_delta_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_ap_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_ap"],
                dr_values=[0.50, 0.52],
                fsrs6_ap_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 2)
        self.assertEqual([spec.lambda_value for spec in specs], [None, None])
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_ap/policy_0",
                "user_1/sched_fsrs6_ap/policy_1",
            ],
        )

    def test_policy_root_expands_portfolio_children_without_dr_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for index, desired_retention in enumerate((0.83, 0.87)):
                policy_dir = (
                    root / "user_1" / "lambda_0" / "policies" / f"policy_{index}"
                )
                policy_dir.mkdir(parents=True)
                _write_ap_policy(policy_dir / "policy.json", dr=desired_retention)
                (policy_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "scheduler_name": "fsrs6_ap",
                            "training_user_ids": [1],
                            "policy_path": "policy.json",
                            "baseline_desired_retention": None,
                            "scheduler_desired_retention": desired_retention,
                            "lambda_value": 0.0,
                            "portfolio_index": index,
                            "action_space": "fsrs6_ap_weight_delta_portfolio_child",
                        }
                    ),
                    encoding="utf-8",
                )

            specs = resolve_fsrs6_ap_policy_specs(
                user_ids=[1],
                dr_values=[0.50, 0.52],
                policy_root=root,
                lambda_values=[0.0],
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6_ap"],
                dr_values=[0.50, 0.52],
                fsrs6_ap_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 2)
        self.assertEqual(
            [spec.baseline_desired_retention for spec in specs], [None, None]
        )
        self.assertEqual([spec.policy_index for spec in specs], [0, 1])
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_ap/policy_0/lambda_0",
                "user_1/sched_fsrs6_ap/policy_1/lambda_0",
            ],
        )
        self.assertEqual(
            [lane.fsrs6_ap_baseline_desired_retention for lane in lanes],
            [None, None],
        )

    def test_policy_root_rejects_portfolio_metadata_desired_retention_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            policy_dir = root / "user_1" / "lambda_0" / "policies" / "policy_0"
            policy_dir.mkdir(parents=True)
            _write_ap_policy(policy_dir / "policy.json", dr=0.83)
            (policy_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "scheduler_name": "fsrs6_ap",
                        "training_user_ids": [1],
                        "policy_path": "policy.json",
                        "baseline_desired_retention": None,
                        "scheduler_desired_retention": 0.84,
                        "lambda_value": 0.0,
                        "portfolio_index": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scheduler_desired_retention"):
                resolve_fsrs6_ap_policy_specs(
                    user_ids=[1],
                    dr_values=[0.50, 0.52],
                    policy_root=root,
                    lambda_values=[0.0],
                )

    def test_policy_root_expands_user_dr_lambda_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for user_id in (1, 2):
                for dr in (0.50, 0.52):
                    policy_path = (
                        root
                        / f"user_{user_id}"
                        / "lambda_0p5"
                        / f"dr_{format_float_token(dr)}"
                        / "policy.json"
                    )
                    _write_ap_policy(policy_path, dr=dr)

            specs = resolve_fsrs6_ap_policy_specs(
                user_ids=[1, 2],
                dr_values=[0.50, 0.52],
                policy_root=root,
                lambda_values=[0.5],
            )
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=Path(tmp) / "logs",
                batch_log_root=Path(tmp) / "logs" / "batch_logs",
                envs=["lstm"],
                schedulers=["fsrs6_ap"],
                dr_values=[0.50, 0.52],
                fsrs6_ap_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [
                (
                    lane.user_id,
                    lane.fsrs6_ap_baseline_desired_retention,
                    lane.fsrs6_ap_lambda_value,
                )
                for lane in lanes
            ],
            [
                (1, 0.50, 0.5),
                (1, 0.52, 0.5),
                (2, 0.50, 0.5),
                (2, 0.52, 0.5),
            ],
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_fsrs6_ap/dr_0p5/lambda_0p5",
                "user_1/sched_fsrs6_ap/dr_0p52/lambda_0p5",
                "user_2/sched_fsrs6_ap/dr_0p5/lambda_0p5",
                "user_2/sched_fsrs6_ap/dr_0p52/lambda_0p5",
            ],
        )

    def test_multiple_ap_policies_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.json"
            second = root / "p2.json"
            _write_ap_policy(first, dr=0.50)
            _write_ap_policy(second, dr=0.52, offset=1.0)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="lstm",
                    scheduler_name="fsrs6_ap",
                    scheduler_spec="fsrs6_ap",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_ap_policy=first,
                    fsrs6_ap_baseline_desired_retention=0.50,
                    fsrs6_ap_lambda_value=0.5,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="lstm",
                    scheduler_name="fsrs6_ap",
                    scheduler_spec="fsrs6_ap",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_ap_policy=second,
                    fsrs6_ap_baseline_desired_retention=0.52,
                    fsrs6_ap_lambda_value=0.5,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=None,
                fsrs_default_weights=None,
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        group = ops._groups[0]
        self.assertEqual(int(group.lane_indices.numel()), 2)
        target = torch.pow(group.ops._retention_factor + 1.0, group.ops._decay)
        self.assertEqual(
            [round(float(value), 2) for value in target.tolist()],
            [0.50, 0.52],
        )

    def test_anki_sm2_ap_policy_root_expands_portfolio_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_root = root / "train_outputs"
            for user_id in (1, 2):
                for policy_index in (0, 1):
                    policy_dir = (
                        policy_root / f"user_{user_id}" / f"policy_{policy_index}"
                    )
                    policy_dir.mkdir(parents=True)
                    _write_anki_sm2_ap_policy(
                        policy_dir / "policy.json",
                        offset=float(policy_index),
                    )
                    (policy_dir / "metadata.json").write_text(
                        json.dumps(
                            {
                                "scheduler_name": "anki_sm2_ap",
                                "training_user_ids": [user_id],
                                "policy_path": "policy.json",
                                "baseline_desired_retention": None,
                                "portfolio_index": policy_index,
                                "action_space": ("anki_sm2_ap_params_portfolio_child"),
                            }
                        ),
                        encoding="utf-8",
                    )

            specs = resolve_anki_sm2_ap_policy_specs(
                user_ids=[1, 2],
                policy_root=policy_root,
            )
            ctx = BatchedSweepContext(
                repo_root=root,
                benchmark_root=root,
                overrides={},
                log_root=root / "logs",
                batch_log_root=root / "logs" / "batch_logs",
                envs=["fsrs6"],
                schedulers=["anki_sm2_ap"],
                dr_values=[0.50, 0.52],
                anki_sm2_ap_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="fsrs6")

        self.assertEqual(len(specs), 4)
        self.assertEqual(len(lanes), 4)
        self.assertTrue(all(lane.desired_retention is None for lane in lanes))
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(root / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_anki_sm2_ap/policy_0",
                "user_1/sched_anki_sm2_ap/policy_1",
                "user_2/sched_anki_sm2_ap/policy_0",
                "user_2/sched_anki_sm2_ap/policy_1",
            ],
        )

    def test_multiple_anki_sm2_ap_policies_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.json"
            second = root / "p2.json"
            _write_anki_sm2_ap_policy(first)
            _write_anki_sm2_ap_policy(second, offset=1.0)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="fsrs6",
                    scheduler_name="anki_sm2_ap",
                    scheduler_spec="anki_sm2_ap",
                    desired_retention=None,
                    fixed_interval=None,
                    anki_sm2_ap_policy=first,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="fsrs6",
                    scheduler_name="anki_sm2_ap",
                    scheduler_spec="anki_sm2_ap",
                    desired_retention=None,
                    fixed_interval=None,
                    anki_sm2_ap_policy=second,
                ),
            ]

            ops = _build_mixed_scheduler_ops(
                args=argparse.Namespace(scheduler_priority="low_retrievability"),
                active_batch=[1, 2],
                lanes=lanes,
                fsrs_weights=None,
                fsrs_default_weights=None,
                fsrs3_weights=None,
                fsrs3_default_weights=None,
                lstm_packed=None,
                short_term_source=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(ops._groups), 1)
        state = ops.init_state(user_count=2, deck_size=1)
        intervals = ops.update_learn(
            state,
            user_idx=torch.tensor([0, 1]),
            card_idx=torch.tensor([0, 0]),
            rating=torch.tensor([3, 3]),
        )
        self.assertGreater(float(intervals[1]), float(intervals[0]))


if __name__ == "__main__":
    unittest.main()
