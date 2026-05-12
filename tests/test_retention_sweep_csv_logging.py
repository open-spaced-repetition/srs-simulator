from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, cast
import unittest
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs") / "matplotlib"))

import simulate as simulate_cli
from simulator.batched_sweep.logging import (
    BatchedSweepLogLane,
    simulate_and_log,
    simulate_and_log_lanes,
)
from simulator.batched_sweep.plan import build_batched_sweep_plan
from simulator.batched_sweep.runner import (
    BatchedSweepContext,
    _build_dr_grid_lanes,
    _build_sweep_lanes,
    _group_lane_indices,
)
from simulator.core import SimulationStats
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS
from simulator.fsrs6_ap_policy import FSRS6APPolicy
from experiments.retention_sweep.build_pareto import (
    _build_results,
    _plot_ordered_entries,
    _split_results_by_series,
)
from experiments.retention_sweep.aggregate_users import (
    _iter_log_paths as _iter_aggregate_log_paths,
)


def _stats(days: int = 2, memorized: float = 10.0) -> SimulationStats:
    return SimulationStats(
        daily_reviews=[1 for _ in range(days)],
        daily_new=[1 for _ in range(days)],
        daily_retention=[1.0 for _ in range(days)],
        daily_cost=[60.0 for _ in range(days)],
        daily_memorized=[memorized for _ in range(days)],
        total_reviews=days,
        total_lapses=0,
        total_cost=60.0 * days,
        events=[],
        total_projected_retrievability=memorized,
        daily_phase_reviews=[1 for _ in range(days)],
        daily_phase_lapses=[0 for _ in range(days)],
        daily_short_loops=[0 for _ in range(days)],
    )


def _write_log_args(log_dir: Path, write_daily_csv: bool | None) -> argparse.Namespace:
    kwargs = {
        "engine": "batched",
        "days": 2,
        "deck": 10,
        "learn_limit": 1,
        "review_limit": 10,
        "cost_limit_minutes": 60.0,
        "priority": "review-first",
        "env": "fsrs6",
        "environment": "fsrs6",
        "scheduler": "anki_sm2",
        "scheduler_spec": "anki_sm2",
        "user_id": 1,
        "button_usage": None,
        "desired_retention": None,
        "scheduler_priority": "low_retrievability",
        "sspmmc_policy": None,
        "fixed_interval": None,
        "seed": 42,
        "fuzz": False,
        "short_term_source": None,
        "learning_steps": None,
        "relearning_steps": None,
        "short_term_threshold": 0.5,
        "short_term_loops_limit": None,
        "log_dir": log_dir,
        "log_reviews": False,
    }
    if write_daily_csv is not None:
        kwargs["write_daily_csv"] = write_daily_csv
    return argparse.Namespace(**kwargs)


def _batched_args(diagnostic_csv_logs: bool) -> argparse.Namespace:
    return argparse.Namespace(
        days=2,
        deck=10,
        seed=42,
        fuzz=False,
        priority="review-first",
        short_term_threshold=0.5,
        short_term_loops_limit=None,
        no_log=True,
        diagnostic_csv_logs=diagnostic_csv_logs,
        learn_limit=1,
        review_limit=10,
        cost_limit_minutes=60.0,
        button_usage=None,
        scheduler_priority="low_retrievability",
    )


def _plan_args(
    log_dir: Path,
    diagnostic_csv_logs: bool,
    *,
    fsrs6_adr_policy: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=2,
        torch_device=None,
        cuda_devices=None,
        start_user=1,
        end_user=2,
        srs_benchmark_root=None,
        benchmark_result=None,
        log_dir=log_dir,
        log_layout="user",
        start_retention=0.9,
        end_retention=0.9,
        step=0.01,
        days=2,
        diagnostic_csv_logs=diagnostic_csv_logs,
        fsrs6_adr_policy=fsrs6_adr_policy,
    )


class FakeEnvOps:
    device = torch.device("cpu")


class RetentionSweepCsvLoggingTests(unittest.TestCase):
    def test_write_log_can_skip_daily_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            simulate_cli._write_log(_write_log_args(log_dir, False), _stats())

            self.assertEqual(len(list(log_dir.glob("*.jsonl"))), 1)
            self.assertEqual(list(log_dir.glob("*.csv")), [])

    def test_write_log_includes_run_id_in_filename_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            args = _write_log_args(log_dir, False)
            args.run_id = "run/one"

            simulate_cli._write_log(args, _stats())

            json_logs = list(log_dir.glob("*.jsonl"))
            self.assertEqual(len(json_logs), 1)
            self.assertIn("run=run-one", json_logs[0].name)
            meta = json.loads(json_logs[0].read_text().splitlines()[0])
            self.assertEqual(meta["data"]["run_id"], "run/one")

    def test_write_log_shortens_redundant_adr_policy_filename_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            args = _write_log_args(log_dir, False)
            args.engine = "batched"
            args.days = 1825
            args.deck = 10000
            args.learn_limit = 10
            args.review_limit = 9999
            args.cost_limit_minutes = 720.0
            args.priority = "new-first"
            args.scheduler = "fsrs6_adr"
            args.scheduler_spec = "fsrs6_adr"
            args.run_id = "fsrs6_adr_linear_cmaes_users_1_8_v1"
            args.fsrs6_adr_policy = Path("policy.json")
            args.fsrs6_adr_baseline_desired_retention = 0.5
            args.fsrs6_adr_lambda_value = 0.5

            simulate_cli._write_log(args, _stats())

            json_logs = list(log_dir.glob("*.jsonl"))
            self.assertEqual(len(json_logs), 1)
            name = json_logs[0].name
            self.assertLessEqual(len(name), 240)
            self.assertIn("run=fsrs6_adr_linear_cmaes_users_1_8_v1", name)
            self.assertIn("policy=policy", name)
            self.assertNotIn("policy-dr=", name)
            self.assertNotIn("lambda=", name)
            meta = json.loads(json_logs[0].read_text().splitlines()[0])
            self.assertEqual(meta["data"]["fsrs6_adr_baseline_desired_retention"], 0.5)
            self.assertEqual(meta["data"]["fsrs6_adr_lambda_value"], 0.5)

    def test_write_log_preserves_default_daily_csv_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            simulate_cli._write_log(_write_log_args(log_dir, None), _stats())

            self.assertEqual(len(list(log_dir.glob("*.jsonl"))), 1)
            self.assertEqual(len(list(log_dir.glob("*.csv"))), 1)

    def test_batched_stats_csv_requires_diagnostic_flag(self) -> None:
        def fake_simulate_multiuser(**kwargs):
            batch_stats = kwargs["batch_stats"]
            if batch_stats is not None:
                batch_stats["gpu_peak_allocated_bytes"] = [1, 2]
                batch_stats["gpu_peak_reserved_bytes"] = [3, 4]
            return [_stats()]

        with tempfile.TemporaryDirectory() as tmp:
            batch_log_root = Path(tmp) / "batch_logs"
            with patch(
                "simulator.batched_sweep.logging.simulate_multiuser",
                side_effect=fake_simulate_multiuser,
            ):
                simulate_and_log(
                    write_log=lambda _args, _stats: None,
                    args=_batched_args(False),
                    batch=[1],
                    env_ops=cast(Any, FakeEnvOps()),
                    sched_ops=cast(Any, object()),
                    behavior=cast(Any, object()),
                    cost_model=cast(Any, object()),
                    progress=False,
                    progress_queue=None,
                    device_label="cpu",
                    run_label="test",
                    environment="fsrs6",
                    scheduler_name="anki_sm2",
                    scheduler_spec="anki_sm2",
                    desired_retention=None,
                    fixed_interval=None,
                    short_term_source=None,
                    learning_steps=[],
                    relearning_steps=[],
                    learning_steps_arg=None,
                    relearning_steps_arg=None,
                    log_root=Path(tmp),
                    batch_log_root=batch_log_root,
                )
            self.assertFalse(batch_log_root.exists())

        with tempfile.TemporaryDirectory() as tmp:
            batch_log_root = Path(tmp) / "batch_logs"
            with patch(
                "simulator.batched_sweep.logging.simulate_multiuser",
                side_effect=fake_simulate_multiuser,
            ):
                simulate_and_log(
                    write_log=lambda _args, _stats: None,
                    args=_batched_args(True),
                    batch=[1],
                    env_ops=cast(Any, FakeEnvOps()),
                    sched_ops=cast(Any, object()),
                    behavior=cast(Any, object()),
                    cost_model=cast(Any, object()),
                    progress=False,
                    progress_queue=None,
                    device_label="cpu",
                    run_label="test",
                    environment="fsrs6",
                    scheduler_name="anki_sm2",
                    scheduler_spec="anki_sm2",
                    desired_retention=None,
                    fixed_interval=None,
                    short_term_source=None,
                    learning_steps=[],
                    relearning_steps=[],
                    learning_steps_arg=None,
                    relearning_steps_arg=None,
                    log_root=Path(tmp),
                    batch_log_root=batch_log_root,
                )
            self.assertEqual(len(list(batch_log_root.glob("batch_*.csv"))), 1)

    def test_batched_lanes_write_distinct_logs_for_repeated_user(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_simulate_multiuser(**kwargs):
            calls.append(kwargs)
            return [_stats(), _stats()]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _batched_args(False)
            args.no_log = False
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_fsrs6" / "dr_0p5",
                    environment="fsrs6",
                    scheduler_name="fsrs6",
                    scheduler_spec="fsrs6",
                    desired_retention=0.50,
                    fixed_interval=None,
                ),
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_fsrs6" / "dr_0p6",
                    environment="fsrs6",
                    scheduler_name="fsrs6",
                    scheduler_spec="fsrs6",
                    desired_retention=0.60,
                    fixed_interval=None,
                ),
            ]
            with patch(
                "simulator.batched_sweep.logging.simulate_multiuser",
                side_effect=fake_simulate_multiuser,
            ):
                simulate_and_log_lanes(
                    write_log=simulate_cli._write_log,
                    args=args,
                    lanes=lanes,
                    env_ops=cast(Any, FakeEnvOps()),
                    sched_ops=cast(Any, object()),
                    behavior=cast(Any, object()),
                    cost_model=cast(Any, object()),
                    progress=False,
                    progress_queue=None,
                    device_label="cpu",
                    run_label="fsrs6 dr-grid",
                    short_term_source=None,
                    learning_steps=[],
                    relearning_steps=[],
                    learning_steps_arg=None,
                    relearning_steps_arg=None,
                    batch_log_root=root / "batch_logs",
                )

            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["batch_stats"])
            self.assertEqual(
                len(list((root / "sched_fsrs6" / "dr_0p5" / "user_1").glob("*.jsonl"))),
                1,
            )
            self.assertEqual(
                len(list((root / "sched_fsrs6" / "dr_0p6" / "user_1").glob("*.jsonl"))),
                1,
            )
            self.assertEqual(list(root.rglob("*.csv")), [])

    def test_batched_lanes_allow_mixed_scheduler_logs(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_simulate_multiuser(**kwargs):
            calls.append(kwargs)
            return [_stats(), _stats()]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = _batched_args(False)
            args.no_log = False
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_fsrs6" / "dr_0p5",
                    environment="fsrs6",
                    scheduler_name="fsrs6",
                    scheduler_spec="fsrs6",
                    desired_retention=0.50,
                    fixed_interval=None,
                ),
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_anki_sm2",
                    environment="fsrs6",
                    scheduler_name="anki_sm2",
                    scheduler_spec="anki_sm2",
                    desired_retention=None,
                    fixed_interval=None,
                ),
            ]
            with patch(
                "simulator.batched_sweep.logging.simulate_multiuser",
                side_effect=fake_simulate_multiuser,
            ):
                simulate_and_log_lanes(
                    write_log=simulate_cli._write_log,
                    args=args,
                    lanes=lanes,
                    env_ops=cast(Any, FakeEnvOps()),
                    sched_ops=cast(Any, object()),
                    behavior=cast(Any, object()),
                    cost_model=cast(Any, object()),
                    progress=False,
                    progress_queue=None,
                    device_label="cpu",
                    run_label="mixed schedulers",
                    short_term_source=None,
                    learning_steps=[],
                    relearning_steps=[],
                    learning_steps_arg=None,
                    relearning_steps_arg=None,
                    batch_log_root=root / "batch_logs",
                )

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                len(list((root / "sched_fsrs6" / "dr_0p5" / "user_1").glob("*.jsonl"))),
                1,
            )
            self.assertEqual(
                len(list((root / "sched_anki_sm2" / "user_1").glob("*.jsonl"))),
                1,
            )

    def test_dr_grid_lanes_expand_user_and_retention_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lanes = _build_dr_grid_lanes(
                batch=[1, 2],
                log_root=Path(tmp),
                environment="fsrs6",
                scheduler_name="fsrs6",
                scheduler_spec="fsrs6",
                dr_values=[0.50, 0.60],
                fixed_interval=None,
            )

            self.assertEqual(
                [(lane.user_id, lane.desired_retention) for lane in lanes],
                [(1, 0.50), (2, 0.50), (1, 0.60), (2, 0.60)],
            )
            self.assertEqual(
                [lane.log_root.relative_to(Path(tmp)).as_posix() for lane in lanes],
                [
                    "sched_fsrs6/dr_0p5",
                    "sched_fsrs6/dr_0p5",
                    "sched_fsrs6/dr_0p6",
                    "sched_fsrs6/dr_0p6",
                ],
            )

    def test_default_user_layout_places_scheduler_logs_under_each_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy_path = root / "policy.json"
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=root,
                batch_log_root=root / "batch_logs",
                envs=["lstm"],
                schedulers=["fsrs6", "fsrs6_adr", "fixed@3", "anki_sm2"],
                dr_values=[0.90],
                fsrs6_adr_policy=policy_path,
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="lstm")

        self.assertEqual(
            [lane.final_log_dir.relative_to(root).as_posix() for lane in lanes],
            [
                "user_1/sched_fsrs6/dr_0p9",
                "user_1/sched_fsrs6_adr/policy_policy",
                "user_1/sched_fixed/ivl_3",
                "user_1/sched_anki_sm2",
            ],
        )

    def test_sweep_layout_preserves_scheduler_first_lane_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=root,
                batch_log_root=root / "batch_logs",
                envs=["lstm"],
                schedulers=["fsrs6", "fixed@3", "anki_sm2"],
                dr_values=[0.90],
                log_layout="sweep",
            )

            lanes = _build_sweep_lanes(batch=[1], ctx=ctx, environment="lstm")

        self.assertEqual(
            [lane.final_log_dir.relative_to(root).as_posix() for lane in lanes],
            [
                "sched_fsrs6/dr_0p9/user_1",
                "sched_fixed/ivl_3/user_1",
                "sched_anki_sm2/user_1",
            ],
        )

    def test_sweep_lanes_expand_scheduler_and_parameter_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = BatchedSweepContext(
                repo_root=REPO_ROOT,
                benchmark_root=REPO_ROOT,
                overrides={},
                log_root=root,
                batch_log_root=root / "batch_logs",
                envs=["fsrs6"],
                schedulers=["fsrs6", "fsrs3", "anki_sm2", "fixed@3"],
                dr_values=[0.50, 0.60],
            )
            lanes = _build_sweep_lanes(
                batch=[1, 2],
                ctx=ctx,
                environment="fsrs6",
            )

            self.assertEqual(len(lanes), 12)
            self.assertEqual(
                [
                    (lane.scheduler_name, lane.user_id, lane.desired_retention)
                    for lane in lanes[:8]
                ],
                [
                    ("fsrs6", 1, 0.50),
                    ("fsrs6", 2, 0.50),
                    ("fsrs6", 1, 0.60),
                    ("fsrs6", 2, 0.60),
                    ("fsrs3", 1, 0.50),
                    ("fsrs3", 2, 0.50),
                    ("fsrs3", 1, 0.60),
                    ("fsrs3", 2, 0.60),
                ],
            )
            self.assertEqual(
                [
                    (lane.scheduler_name, lane.user_id, lane.fixed_interval)
                    for lane in lanes[8:]
                ],
                [
                    ("anki_sm2", 1, None),
                    ("anki_sm2", 2, None),
                    ("fixed", 1, 3.0),
                    ("fixed", 2, 3.0),
                ],
            )
            self.assertEqual(
                [len(group) for group in _group_lane_indices(lanes)], [4, 4, 2, 2]
            )

    def test_aggregate_log_iterator_recurses_nested_lane_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "sched_fsrs6" / "dr_0p5" / "user_1" / "a.jsonl"
            second = root / "sched_anki_sm2" / "user_1" / "b.jsonl"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")

            paths = list(_iter_aggregate_log_paths(root))

            self.assertEqual(paths, sorted([first, second]))

    def test_batched_plan_creates_batch_log_dir_only_for_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_plan_args(log_dir, False),
                envs=["lstm"],
                schedulers=["anki_sm2"],
            )

            self.assertEqual(plan.ctx.batch_log_root, log_dir / "batch_logs")
            self.assertFalse(plan.ctx.batch_log_root.exists())
            self.assertEqual(plan.ctx.log_layout, "user")

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_plan_args(log_dir, True),
                envs=["lstm"],
                schedulers=["anki_sm2"],
            )

            self.assertTrue(plan.ctx.batch_log_root.exists())

    def test_build_pareto_reads_default_user_layout_for_mixed_batched_sweep(
        self,
    ) -> None:
        def fake_simulate_multiuser(**_kwargs):
            return [_stats(), _stats()]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs" / "retention_sweep"
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(desired_retention=0.90).write_json(policy_path)
            args = _batched_args(False)
            args.no_log = False
            lanes = [
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_fsrs6" / "dr_0p9",
                    log_dir=root / "user_1" / "sched_fsrs6" / "dr_0p9",
                    environment="lstm",
                    scheduler_name="fsrs6",
                    scheduler_spec="fsrs6",
                    desired_retention=0.90,
                    fixed_interval=None,
                ),
                BatchedSweepLogLane(
                    user_id=1,
                    log_root=root / "sched_fsrs6_adr" / "policy_policy",
                    log_dir=root / "user_1" / "sched_fsrs6_adr" / "policy_policy",
                    environment="lstm",
                    scheduler_name="fsrs6_adr",
                    scheduler_spec="fsrs6_adr",
                    desired_retention=None,
                    fixed_interval=None,
                    fsrs6_adr_policy=policy_path,
                ),
            ]

            with patch(
                "simulator.batched_sweep.logging.simulate_multiuser",
                side_effect=fake_simulate_multiuser,
            ):
                simulate_and_log_lanes(
                    write_log=simulate_cli._write_log,
                    args=args,
                    lanes=lanes,
                    env_ops=cast(Any, FakeEnvOps()),
                    sched_ops=cast(Any, object()),
                    behavior=cast(Any, object()),
                    cost_model=cast(Any, object()),
                    progress=False,
                    progress_queue=None,
                    device_label="cpu",
                    run_label="mixed schedulers",
                    short_term_source=None,
                    learning_steps=[],
                    relearning_steps=[],
                    learning_steps_arg=None,
                    relearning_steps_arg=None,
                    batch_log_root=root / "batch_logs",
                )

            user_log_dir = root / "user_1"
            fsrs_results = _build_results(
                user_log_dir,
                "lstm",
                {"fsrs6"},
                0.50,
                0.98,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
            )
            adr_results = _build_results(
                user_log_dir,
                "lstm",
                {"fsrs6_adr"},
                0.50,
                0.98,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
            )

        self.assertEqual([entry["scheduler"] for entry in fsrs_results], ["fsrs6"])
        self.assertEqual([entry["scheduler"] for entry in adr_results], ["fsrs6_adr"])

    def test_build_pareto_filters_root_log_dir_to_requested_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs" / "retention_sweep"
            for user_id, memorized in ((1, 10.0), (2, 20.0)):
                user_log_dir = root / f"user_{user_id}" / "sched_fsrs6" / "dr_0p9"
                args = _write_log_args(user_log_dir, False)
                args.engine = "batched"
                args.scheduler = "fsrs6"
                args.scheduler_spec = "fsrs6"
                args.desired_retention = 0.90
                args.user_id = user_id
                simulate_cli._write_log(args, _stats(memorized=memorized))

            results = _build_results(
                root,
                "fsrs6",
                {"fsrs6"},
                0.50,
                0.98,
                [REPO_ROOT, root],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["user_id"], 1)
        self.assertEqual(results[0]["memorized_average"], 10.0)

    def test_build_pareto_preserves_exact_desired_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs" / "retention_sweep"
            user_log_dir = root / "user_1" / "sched_fsrs6" / "dr_exact"
            args = _write_log_args(user_log_dir, False)
            args.engine = "batched"
            args.scheduler = "fsrs6"
            args.scheduler_spec = "fsrs6"
            args.desired_retention = 0.5386559409988914
            simulate_cli._write_log(args, _stats())

            results = _build_results(
                root,
                "fsrs6",
                {"fsrs6"},
                0.50,
                0.98,
                [REPO_ROOT, root],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["desired_retention"],
            0.5386559409988914,
        )
        self.assertEqual(results[0]["title"], "DR=53.87%")

    def test_build_pareto_filters_fsrs6_adr_baseline_dr_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs" / "retention_sweep"
            for dr, memorized in ((0.50, 10.0), (0.52, 20.0), (0.98, 30.0)):
                dr_token = str(dr).replace(".", "p")
                user_log_dir = (
                    root
                    / "user_1"
                    / "sched_fsrs6_adr"
                    / f"dr_{dr_token}"
                    / "lambda_0p5"
                )
                policy_path = Path(tmp) / f"policy_dr_{dr_token}.json"
                FSRS6ADRPolicy.baseline(desired_retention=dr).write_json(policy_path)
                args = _write_log_args(user_log_dir, False)
                args.engine = "batched"
                args.env = "fsrs6"
                args.environment = "fsrs6"
                args.scheduler = "fsrs6_adr"
                args.scheduler_spec = "fsrs6_adr"
                args.desired_retention = None
                args.fsrs6_adr_policy = policy_path
                args.fsrs6_adr_baseline_desired_retention = dr
                args.fsrs6_adr_lambda_value = 0.5
                simulate_cli._write_log(args, _stats(memorized=memorized))

            results = _build_results(
                root,
                "fsrs6",
                {"fsrs6_adr"},
                0.52,
                0.96,
                [REPO_ROOT, root],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fsrs6_adr_baseline_desired_retention"], 0.52)
        self.assertEqual(results[0]["memorized_average"], 20.0)

    def test_build_pareto_keeps_no_dr_fsrs6_adr_portfolio_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_log_dir = (
                Path(tmp)
                / "logs"
                / "retention_sweep"
                / "user_1"
                / "sched_fsrs6_adr"
                / "policy_0"
            )
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy(
                coefficients=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                baseline_desired_retention=None,
                title="portfolio child",
            ).write_json(policy_path)
            args = _write_log_args(user_log_dir, False)
            args.engine = "batched"
            args.env = "fsrs6"
            args.environment = "fsrs6"
            args.scheduler = "fsrs6_adr"
            args.scheduler_spec = "fsrs6_adr"
            args.desired_retention = None
            args.fsrs6_adr_policy = policy_path
            args.fsrs6_adr_baseline_desired_retention = None
            args.fsrs6_adr_lambda_value = 0.0
            simulate_cli._write_log(args, _stats(memorized=42.0))

            results = _build_results(
                user_log_dir.parents[2],
                "fsrs6",
                {"fsrs6_adr"},
                0.52,
                0.96,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["fsrs6_adr_baseline_desired_retention"])
        self.assertEqual(results[0]["memorized_average"], 42.0)

    def test_build_pareto_keeps_no_dr_fsrs6_ap_portfolio_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_log_dir = (
                Path(tmp)
                / "logs"
                / "retention_sweep"
                / "user_1"
                / "sched_fsrs6_ap"
                / "policy_0"
            )
            policy_dir = Path(tmp) / "policies" / "policy_0"
            policy_dir.mkdir(parents=True)
            policy_path = policy_dir / "policy.json"
            FSRS6APPolicy.from_search_vector(
                base_weights=DEFAULT_FSRS6_WEIGHTS,
                search_vector=[0.0] * 21,
                baseline_desired_retention=0.83,
                title="portfolio child",
            ).write_json(policy_path)
            (policy_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "scheduler_name": "fsrs6_ap",
                        "training_user_ids": [1],
                        "policy_path": "policy.json",
                        "baseline_desired_retention": None,
                        "scheduler_desired_retention": 0.83,
                        "lambda_value": 0.0,
                        "portfolio_index": 0,
                        "action_space": "fsrs6_ap_weight_delta_portfolio_child",
                    }
                ),
                encoding="utf-8",
            )
            args = _write_log_args(user_log_dir, False)
            args.engine = "batched"
            args.env = "fsrs6"
            args.environment = "fsrs6"
            args.scheduler = "fsrs6_ap"
            args.scheduler_spec = "fsrs6_ap"
            args.desired_retention = None
            args.fsrs6_ap_policy = policy_path
            args.fsrs6_ap_baseline_desired_retention = None
            args.fsrs6_ap_lambda_value = 0.0
            simulate_cli._write_log(args, _stats(memorized=43.0))

            results = _build_results(
                user_log_dir.parents[2],
                "fsrs6",
                {"fsrs6_ap"},
                0.52,
                0.96,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["fsrs6_ap_baseline_desired_retention"])
        self.assertEqual(results[0]["title"], "AP policy_0")
        self.assertEqual(results[0]["memorized_average"], 43.0)

    def test_build_pareto_orders_no_dr_portfolio_plot_points_by_x_axis(self) -> None:
        entries = [
            {
                "memorized_average": 30.0,
                "time_average": 3.0,
                "title": "policy_0",
            },
            {
                "memorized_average": 10.0,
                "time_average": 1.0,
                "title": "policy_1",
            },
            {
                "memorized_average": 20.0,
                "time_average": 2.0,
                "title": "policy_2",
            },
        ]

        ordered = _plot_ordered_entries(entries)

        self.assertEqual(
            [entry["title"] for entry in ordered],
            ["policy_1", "policy_2", "policy_0"],
        )

    def test_build_pareto_keeps_fsrs6_adr_runs_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_log_dir = (
                Path(tmp) / "user_1" / "sched_fsrs6_adr" / "dr_0p9" / "lambda_0p5"
            )
            policy_path = Path(tmp) / "policy.json"
            FSRS6ADRPolicy.baseline(desired_retention=0.90).write_json(policy_path)
            for run_id, memorized in (("poly-run", 10.0), ("linear-run", 20.0)):
                args = _write_log_args(user_log_dir, False)
                args.engine = "batched"
                args.scheduler = "fsrs6_adr"
                args.scheduler_spec = "fsrs6_adr"
                args.desired_retention = None
                args.run_id = run_id
                args.fsrs6_adr_policy = policy_path
                args.fsrs6_adr_baseline_desired_retention = 0.90
                args.fsrs6_adr_lambda_value = 0.5
                simulate_cli._write_log(args, _stats(memorized=memorized))

            results = _build_results(
                user_log_dir.parents[2],
                "fsrs6",
                {"fsrs6_adr"},
                0.50,
                0.98,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
                user_id_filter=1,
            )
            series = _split_results_by_series(results)

        self.assertEqual(len(results), 2)
        self.assertCountEqual(
            [entry["series_key"] for entry in results],
            ["run=poly-run", "run=linear-run"],
        )
        self.assertEqual([len(entries) for _key, entries in series], [1, 1])

    def test_build_pareto_labels_fsrs6_adr_by_baseline_dr_without_deduping_policies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_log_dir = Path(tmp) / "user_1"
            for index, lambda_value in enumerate((0.5, 0.7), start=1):
                policy_path = Path(tmp) / f"fsrs6_adr_u1_dr_0.94_lambda_{index}.json"
                FSRS6ADRPolicy.baseline(desired_retention=0.94).write_json(policy_path)
                args = _write_log_args(user_log_dir, False)
                args.engine = "batched"
                args.env = "lstm"
                args.environment = "lstm"
                args.scheduler = "fsrs6_adr"
                args.scheduler_spec = "fsrs6_adr"
                args.fsrs6_adr_policy = policy_path
                args.fsrs6_adr_lambda_value = lambda_value
                simulate_cli._write_log(args, _stats(memorized=10.0 + index))

            results = _build_results(
                user_log_dir,
                "lstm",
                {"fsrs6_adr"},
                0.50,
                0.98,
                [REPO_ROOT, user_log_dir],
                None,
                None,
                None,
                "batched",
            )

        self.assertEqual(len(results), 2)
        self.assertEqual([entry["title"] for entry in results], ["DR=94%", "DR=94%"])
        self.assertEqual(
            [entry["fsrs6_adr_baseline_desired_retention"] for entry in results],
            [0.94, 0.94],
        )
        self.assertCountEqual(
            [entry["fsrs6_adr_lambda_value"] for entry in results],
            [0.5, 0.7],
        )

    def test_build_pareto_dedupes_no_desired_scheduler_by_key_with_engine_filter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user_log_dir = Path(tmp) / "user_1"
            flat_args = _write_log_args(user_log_dir, False)
            flat_args.engine = "batched"
            simulate_cli._write_log(flat_args, _stats(memorized=7.0))

            nested_args = _write_log_args(user_log_dir / "sched_anki_sm2", False)
            nested_args.engine = "batched"
            simulate_cli._write_log(nested_args, _stats(memorized=11.0))

            results = _build_results(
                user_log_dir,
                "fsrs6",
                {"anki_sm2"},
                0.50,
                0.98,
                [REPO_ROOT, user_log_dir],
                None,
                False,
                None,
                "batched",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["scheduler"], "anki_sm2")
        self.assertEqual(results[0]["memorized_average"], 11.0)

    def test_batched_plan_requires_fsrs6_adr_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            with self.assertRaisesRegex(ValueError, "--fsrs6-adr-policy"):
                build_batched_sweep_plan(
                    repo_root=REPO_ROOT,
                    args=_plan_args(log_dir, False),
                    envs=["lstm"],
                    schedulers=["fsrs6_adr"],
                )

            policy_path = Path(tmp) / "policy.json"
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_plan_args(log_dir, False, fsrs6_adr_policy=policy_path),
                envs=["lstm"],
                schedulers=["fsrs6_adr"],
            )

            self.assertEqual(plan.ctx.fsrs6_adr_policy, policy_path)


if __name__ == "__main__":
    unittest.main()
