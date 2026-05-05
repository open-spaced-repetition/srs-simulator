from __future__ import annotations

import argparse
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from experiments.retention_sweep.aggregate_users import (
    _iter_log_paths as _iter_aggregate_log_paths,
)


def _stats(days: int = 2) -> SimulationStats:
    return SimulationStats(
        daily_reviews=[1 for _ in range(days)],
        daily_new=[1 for _ in range(days)],
        daily_retention=[1.0 for _ in range(days)],
        daily_cost=[60.0 for _ in range(days)],
        daily_memorized=[10.0 for _ in range(days)],
        total_reviews=days,
        total_lapses=0,
        total_cost=60.0 * days,
        events=[],
        total_projected_retrievability=10.0,
        daily_phase_reviews=[1 for _ in range(days)],
        daily_phase_lapses=[0 for _ in range(days)],
        daily_short_loops=[0 for _ in range(days)],
    )


def _write_log_args(log_dir: Path, write_daily_csv: bool | None) -> argparse.Namespace:
    kwargs = {
        "engine": "vectorized",
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
    sa_fsrs6_policy: Path | None = None,
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
        start_retention=0.9,
        end_retention=0.9,
        step=0.01,
        days=2,
        diagnostic_csv_logs=diagnostic_csv_logs,
        sa_fsrs6_policy=sa_fsrs6_policy,
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
                [len(group) for group in _group_lane_indices(lanes)], [4, 2, 2, 2, 2]
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

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_plan_args(log_dir, True),
                envs=["lstm"],
                schedulers=["anki_sm2"],
            )

            self.assertTrue(plan.ctx.batch_log_root.exists())

    def test_batched_plan_requires_sa_fsrs6_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            with self.assertRaisesRegex(ValueError, "--sa-fsrs6-policy"):
                build_batched_sweep_plan(
                    repo_root=REPO_ROOT,
                    args=_plan_args(log_dir, False),
                    envs=["lstm"],
                    schedulers=["sa_fsrs6"],
                )

            policy_path = Path(tmp) / "policy.json"
            plan = build_batched_sweep_plan(
                repo_root=REPO_ROOT,
                args=_plan_args(log_dir, False, sa_fsrs6_policy=policy_path),
                envs=["lstm"],
                schedulers=["sa_fsrs6"],
            )

            self.assertEqual(plan.ctx.sa_fsrs6_policy, policy_path)


if __name__ == "__main__":
    unittest.main()
