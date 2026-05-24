from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import tempfile
import unittest

import torch

from experiments.single_card_tradeoff.core import tradeoff_runner
from experiments.single_card_tradeoff.core.config import SingleCardFSRS6Config
from experiments.single_card_tradeoff.core.defaults import (
    FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_SCHEDULER,
)
from experiments.single_card_tradeoff.core.types import UserContext
from experiments.single_card_tradeoff.models.policy_runtime import RetentionDistillNet
from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS
from simulator.fsrs_defaults import DEFAULT_FSRS6_WEIGHTS


def _base_args(policy_template: str) -> argparse.Namespace:
    return argparse.Namespace(
        engine="vectorized",
        fuzz=False,
        torch_device="cpu",
        days=3,
        particles=2,
        deck_scale=10_000,
        target_batch_size=1,
        no_progress=True,
        oracle_continuous_stationary_finite_distill_policy=Path("unused.pt"),
        oracle_continuous_stationary_finite_distill_policy_template=policy_template,
        oracle_continuous_stationary_finite_distill_cost_weights="0,4",
        user_id=1,
    )


def _fsrs_config(user_id: int) -> SingleCardFSRS6Config:
    return SingleCardFSRS6Config(
        environment="fsrs6",
        fsrs_weights=tuple(DEFAULT_FSRS6_WEIGHTS),
        first_rating_prob=tuple(DEFAULT_FIRST_RATING_PROB),
        review_rating_prob=tuple(DEFAULT_REVIEW_RATING_PROB),
        learning_costs=tuple(DEFAULT_STATE_RATING_COSTS.learning),
        review_costs=tuple(DEFAULT_STATE_RATING_COSTS.review),
        user_id=user_id,
        benchmark_result=None,
        benchmark_partition="0",
        srs_benchmark_root=None,
        button_usage=None,
    )


def _user_context(user_id: int, args: argparse.Namespace) -> UserContext:
    user_args = argparse.Namespace(**vars(args))
    user_args.user_id = user_id
    return cast(
        UserContext,
        SimpleNamespace(
            user_id=user_id,
            args=user_args,
            fsrs_config=_fsrs_config(user_id),
        ),
    )


def _write_policy_checkpoint(
    path: Path,
    *,
    retention_min: float = 0.5,
    retention_max: float = 0.98,
) -> None:
    model = RetentionDistillNet(
        obs_dim=3,
        hidden_size=4,
        action_count=2,
        architecture="residual",
        depth=1,
    )
    state_dict = {
        name: torch.zeros_like(value) for name, value in model.state_dict().items()
    }
    model.load_state_dict(state_dict)
    torch.save(
        {
            "policy_type": "fsrs6_oracle_continuous_stationary_finite_distill",
            "action_mode": "desired_retention",
            "cost_weights": [0.0, 4.0],
            "action_retentions": [0.5, 0.98],
            "obs_dim": 3,
            "obs_mode": "oracle_stationary",
            "hidden_size": 4,
            "network": "residual",
            "network_depth": 1,
            "retention_min": retention_min,
            "retention_max": retention_max,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


class ContinuousStationaryFiniteDistillTradeoffTests(unittest.TestCase):
    def test_multiuser_dispatch_uses_batched_evaluator(self) -> None:
        calls: list[tuple[str, int]] = []
        scheduler_name = FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_SCHEDULER
        original_batched = tradeoff_runner._run_fsrs6_batched_oracle_continuous_stationary_finite_distill
        original_single = tradeoff_runner._CUSTOM_SINGLE_USER_RUNNERS[scheduler_name]

        def fake_batched(
            args: argparse.Namespace,
            *,
            environment_name: str,
            scheduler_spec: str,
            seed: int,
            user_contexts: Sequence[UserContext],
        ) -> list[dict[str, Any]]:
            del args, environment_name, scheduler_spec, seed
            calls.append(("batched", len(user_contexts)))
            return [{"scheduler": scheduler_name}]

        def fake_single(
            args: argparse.Namespace,
            *,
            environment_name: str,
            scheduler_spec: str,
            seed: int,
        ) -> list[dict[str, Any]]:
            del args, environment_name, scheduler_spec, seed
            calls.append(("single", 1))
            return []

        tradeoff_runner._run_fsrs6_batched_oracle_continuous_stationary_finite_distill = fake_batched
        tradeoff_runner._CUSTOM_SINGLE_USER_RUNNERS[scheduler_name] = fake_single
        try:
            args = argparse.Namespace()
            contexts = [
                cast(
                    UserContext,
                    SimpleNamespace(user_id=1, args=argparse.Namespace(user_id=1)),
                ),
                cast(
                    UserContext,
                    SimpleNamespace(user_id=2, args=argparse.Namespace(user_id=2)),
                ),
            ]
            rows = tradeoff_runner._run_registered_custom_scheduler(
                args,
                environment_name="fsrs6",
                scheduler_name=scheduler_name,
                scheduler_spec=scheduler_name,
                seed=42,
                user_contexts=contexts,
            )
        finally:
            tradeoff_runner._run_fsrs6_batched_oracle_continuous_stationary_finite_distill = original_batched
            tradeoff_runner._CUSTOM_SINGLE_USER_RUNNERS[scheduler_name] = (
                original_single
            )

        self.assertEqual(calls, [("batched", 2)])
        self.assertEqual(rows, [{"scheduler": scheduler_name}])

    def test_multiuser_checkpoint_metadata_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_policy_checkpoint(root / "user_1_policy.pt", retention_min=0.5)
            _write_policy_checkpoint(root / "user_2_policy.pt", retention_min=0.55)
            args = _base_args(str(root / "user_{user_id}_policy.pt"))
            contexts = [_user_context(1, args), _user_context(2, args)]

            with self.assertRaises(SystemExit) as cm:
                tradeoff_runner._load_fsrs6_batched_oracle_continuous_stationary_finite_distill_policies(
                    args,
                    device=torch.device("cpu"),
                    user_contexts=contexts,
                )

        self.assertIn("retention_min", str(cm.exception))

    def test_batched_evaluator_outputs_each_user_weight_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_policy_checkpoint(root / "user_1_policy.pt")
            _write_policy_checkpoint(root / "user_2_policy.pt")
            args = _base_args(str(root / "user_{user_id}_policy.pt"))
            contexts = [_user_context(1, args), _user_context(2, args)]

            rows = tradeoff_runner._run_fsrs6_batched_oracle_continuous_stationary_finite_distill(
                args,
                environment_name="fsrs6",
                scheduler_spec=FSRS6_ORACLE_CONTINUOUS_STATIONARY_FINITE_DISTILL_SCHEDULER,
                seed=42,
                user_contexts=contexts,
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            sorted((row["user_id"], row["goal_cost_weight"]) for row in rows),
            [(1, 0.0), (1, 4.0), (2, 0.0), (2, 4.0)],
        )


if __name__ == "__main__":
    unittest.main()
