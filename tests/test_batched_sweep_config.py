from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.run_sweep_users_batched import main as batched_main
from simulator.batched_sweep.config import load_batched_sweep_config
from simulator.batched_sweep.logging import BatchedSweepLogLane
from simulator.batched_sweep.plan import build_batched_sweep_plan
from simulator.batched_sweep.runner import (
    BatchedSweepContext,
    _build_mixed_scheduler_ops,
    _build_sweep_lanes,
    _split_lanes,
)
from simulator.batched_sweep.sa_policy import (
    format_float_token,
    resolve_sa_fsrs6_policy_specs,
)
from simulator.batched_sweep.sa_dr_policy import resolve_sa_fsrs6_dr_policy_specs
from simulator.defaults import DEFAULT_MAX_LANES_PER_BATCH
from simulator.fsrs_defaults import DEFAULT_FSRS3_WEIGHTS, DEFAULT_FSRS6_WEIGHTS
from simulator.sa_fsrs6_dr_policy import SAFSRS6DRPolicy
from simulator.sa_fsrs6_policy import SAFSRS6Policy
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
    base = SAFSRS6Policy.baseline(desired_retention=dr)
    policy = SAFSRS6Policy(
        coefficients=(base.coefficients[0] + offset, *base.coefficients[1:]),
        baseline_desired_retention=dr,
    )
    policy.write_json(path)


def _write_dr_policy(path: Path, *, offset: float = 0.0) -> None:
    base = SAFSRS6DRPolicy.baseline()
    coefficients = list(base.coefficients)
    coefficients[0] += offset
    policy = SAFSRS6DRPolicy(coefficients=tuple(coefficients))
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
        sa_fsrs6_policy=policy_path,
        sa_fsrs6_policy_root=None,
        sa_fsrs6_train_run_root=None,
        sa_fsrs6_policy_manifest=None,
        sa_fsrs6_lambda_values=None,
        sa_fsrs6_dr_policy=None,
        sa_fsrs6_dr_policy_root=None,
        sa_fsrs6_dr_train_run_root=None,
        sa_fsrs6_dr_policy_manifest=None,
        sa_fsrs6_dr_lambda_values=None,
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

    def test_loads_rl_scheduler_experiment_config(self) -> None:
        config = load_batched_sweep_config(
            REPO_ROOT
            / "experiments"
            / "rl_scheduler"
            / "configs"
            / "sa_fsrs6_dr_linear_batch_sweep_users_1_8.toml",
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config.args.user_ids, list(range(1, 9)))
        self.assertEqual(config.envs, ("fsrs6", "lstm"))
        self.assertEqual(config.schedulers, ("sa_fsrs6_dr",))
        self.assertEqual(config.args.log_dir, REPO_ROOT / "logs" / "retention_sweep")
        self.assertEqual(config.args.batch_size, 8)
        self.assertEqual(config.args.torch_device, "cuda")
        self.assertEqual(config.args.sa_fsrs6_dr_lambda_values, (0.5,))
        self.assertFalse(config.args.no_progress)

    def test_dry_run_accepts_rl_scheduler_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            output_root = root / "artifacts"
            run_root = output_root / "test-run"
            policy_root = run_root / "train-overfit" / "train_outputs"
            artifact_paths: list[str] = []
            for user_id in (1, 2):
                policy_path = (
                    policy_root / f"user_{user_id}" / "lambda_0p5" / "policy.json"
                )
                policy_path.parent.mkdir(parents=True)
                _write_dr_policy(policy_path)
                metadata_path = policy_path.parent / "metadata.json"
                metadata_path.write_text("{}", encoding="utf-8")
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
schedulers = ["sa_fsrs6_dr"]
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

    def test_loads_sa_fsrs6_dr_config_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sweep.toml"
            log_dir = root / "logs"
            policy_root = root / "policies"
            path.write_text(
                _valid_config(log_dir)
                + f"""
[sa_fsrs6_dr]
policy_root = "{policy_root.as_posix()}"
lambda_values = [0.5]
""",
                encoding="utf-8",
            )

            config = load_batched_sweep_config(path)

        self.assertEqual(config.args.sa_fsrs6_dr_policy_root, policy_root.resolve())
        self.assertEqual(config.args.sa_fsrs6_dr_lambda_values, (0.5,))

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


class SAFSRS6PolicyExpansionTests(unittest.TestCase):
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
                schedulers=["sa_fsrs6"],
                dr_values=[0.50, 0.52],
                sa_fsrs6_policy=policy_path,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(lanes), 2)
        self.assertTrue(all(lane.sa_fsrs6_policy == policy_path for lane in lanes))
        self.assertTrue(
            all(lane.sa_fsrs6_baseline_desired_retention is None for lane in lanes)
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_sa_fsrs6/policy_policy",
                "user_2/sched_sa_fsrs6/policy_policy",
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

            specs = resolve_sa_fsrs6_policy_specs(
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
                schedulers=["sa_fsrs6"],
                dr_values=[0.50, 0.52],
                sa_fsrs6_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [
                (
                    lane.user_id,
                    lane.sa_fsrs6_baseline_desired_retention,
                    lane.sa_fsrs6_lambda_value,
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
                "user_1/sched_sa_fsrs6/dr_0p5/lambda_0p5",
                "user_1/sched_sa_fsrs6/dr_0p52/lambda_0p5",
                "user_2/sched_sa_fsrs6/dr_0p5/lambda_0p5",
                "user_2/sched_sa_fsrs6/dr_0p52/lambda_0p5",
            ],
        )

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

            specs = resolve_sa_fsrs6_policy_specs(
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
                resolve_sa_fsrs6_policy_specs(
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

            with self.assertRaisesRegex(FileNotFoundError, "Missing SA FSRS-6"):
                resolve_sa_fsrs6_policy_specs(
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

    def test_multiple_sa_policies_share_one_scheduler_group(self) -> None:
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
                    scheduler_name="sa_fsrs6",
                    scheduler_spec="sa_fsrs6",
                    desired_retention=None,
                    fixed_interval=None,
                    sa_fsrs6_policy=first,
                    sa_fsrs6_baseline_desired_retention=0.50,
                    sa_fsrs6_lambda_value=0.5,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="lstm",
                    scheduler_name="sa_fsrs6",
                    scheduler_spec="sa_fsrs6",
                    desired_retention=None,
                    fixed_interval=None,
                    sa_fsrs6_policy=second,
                    sa_fsrs6_baseline_desired_retention=0.52,
                    sa_fsrs6_lambda_value=0.5,
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


class SAFSRS6DRPolicyExpansionTests(unittest.TestCase):
    def test_policy_root_expands_one_policy_per_user_across_dr_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "train_outputs"
            for user_id in (1, 2):
                policy_path = root / f"user_{user_id}" / "lambda_0p5" / "policy.json"
                _write_dr_policy(policy_path)

            specs = resolve_sa_fsrs6_dr_policy_specs(
                user_ids=[1, 2],
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
                schedulers=["sa_fsrs6_dr"],
                dr_values=[0.50, 0.52],
                sa_fsrs6_dr_policy_specs=specs,
            )

            lanes = _build_sweep_lanes(batch=[1, 2], ctx=ctx, environment="lstm")

        self.assertEqual(len(specs), 2)
        self.assertEqual(len(lanes), 4)
        self.assertEqual(
            [
                (
                    lane.user_id,
                    lane.desired_retention,
                    lane.sa_fsrs6_dr_lambda_value,
                    lane.sa_fsrs6_dr_policy.name if lane.sa_fsrs6_dr_policy else None,
                )
                for lane in lanes
            ],
            [
                (1, 0.50, 0.5, "policy.json"),
                (1, 0.52, 0.5, "policy.json"),
                (2, 0.50, 0.5, "policy.json"),
                (2, 0.52, 0.5, "policy.json"),
            ],
        )
        self.assertEqual(
            [
                lane.final_log_dir.relative_to(Path(tmp) / "logs").as_posix()
                for lane in lanes
            ],
            [
                "user_1/sched_sa_fsrs6_dr/dr_0p5/lambda_0p5",
                "user_1/sched_sa_fsrs6_dr/dr_0p52/lambda_0p5",
                "user_2/sched_sa_fsrs6_dr/dr_0p5/lambda_0p5",
                "user_2/sched_sa_fsrs6_dr/dr_0p52/lambda_0p5",
            ],
        )

    def test_multiple_sa_dr_policies_share_one_scheduler_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "p1.json"
            second = root / "p2.json"
            _write_dr_policy(first)
            _write_dr_policy(second, offset=1.0)
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "logs" / "a",
                    environment="lstm",
                    scheduler_name="sa_fsrs6_dr",
                    scheduler_spec="sa_fsrs6_dr",
                    desired_retention=0.50,
                    fixed_interval=None,
                    sa_fsrs6_dr_policy=first,
                    sa_fsrs6_dr_lambda_value=0.5,
                ),
                BatchedSweepLogLane(
                    user_id=2,
                    log_root=root / "logs" / "b",
                    environment="lstm",
                    scheduler_name="sa_fsrs6_dr",
                    scheduler_spec="sa_fsrs6_dr",
                    desired_retention=0.52,
                    fixed_interval=None,
                    sa_fsrs6_dr_policy=second,
                    sa_fsrs6_dr_lambda_value=0.5,
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
        self.assertEqual(
            [round(float(value), 2) for value in group.ops._desired_retention.tolist()],
            [0.50, 0.52],
        )


if __name__ == "__main__":
    unittest.main()
