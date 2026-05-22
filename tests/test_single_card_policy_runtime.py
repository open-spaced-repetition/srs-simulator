from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.single_card_tradeoff.models.policy_runtime import (
    IntervalAwareRetentionLossConfig,
    continuous_intervals_for_retentions,
    interval_aware_retention_loss,
    interval_distill_loss,
    interval_underprediction_weights,
    predicted_retentions,
    retention_distill_loss,
    retention_logits_for_retentions,
    target_retentions_for_intervals,
    weighted_log_interval_smooth_l1,
)
from experiments.single_card_tradeoff.models.single_card_env import (
    FSRS6SingleCardBatch,
)


class SingleCardPolicyRuntimeLossTests(unittest.TestCase):
    def _env(self) -> FSRS6SingleCardBatch:
        return FSRS6SingleCardBatch(
            days=10,
            env_count=3,
            cost_weights=[0.0, 4.0, 16.0],
            action_retentions=[0.9],
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=123,
            exact_memory=False,
            goal_norm_max=16.0,
            obs_mode="basic",
        )

    def test_weighted_log_interval_loss_matches_wrappers(self) -> None:
        env = self._env()
        labels = torch.tensor([1, 3, 10], dtype=torch.int64)
        pred_log_interval = torch.log(torch.tensor([1.0, 2.0, 4.0]))
        target_log_interval = torch.log(labels.to(dtype=torch.float32))

        weights = interval_underprediction_weights(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            labels=labels,
            env=env,
            underprediction_loss_weight=2.0,
            terminal_underprediction_loss_weight=4.0,
        )
        shared = weighted_log_interval_smooth_l1(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            weights=weights,
        )
        legacy_interval = interval_distill_loss(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            labels=labels,
            env=env,
            underprediction_loss_weight=2.0,
            terminal_underprediction_loss_weight=4.0,
        )
        legacy_retention = interval_aware_retention_loss(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            labels=labels,
            env=env,
            underprediction_loss_weight=2.0,
            terminal_underprediction_loss_weight=4.0,
        )
        manual_elementwise = nn.functional.smooth_l1_loss(
            pred_log_interval,
            target_log_interval,
            reduction="none",
        )

        self.assertTrue(
            torch.allclose(shared, torch.mean(manual_elementwise * weights))
        )
        self.assertTrue(torch.allclose(legacy_interval, shared))
        self.assertTrue(torch.allclose(legacy_retention, shared))

    def test_retention_distill_loss_returns_composed_components(self) -> None:
        env = self._env()
        target_interval = torch.tensor([1, 3, 10], dtype=torch.int64)
        pred_logit = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        config = IntervalAwareRetentionLossConfig(
            interval_weight=1.7,
            retention_logit_weight=0.25,
            underprediction_weight=2.0,
            terminal_underprediction_weight=4.0,
        )

        result = retention_distill_loss(
            pred_logit=pred_logit,
            target_interval=target_interval,
            env=env,
            config=config,
            retention_min=0.5,
            retention_max=0.98,
        )
        pred_retention = predicted_retentions(
            pred_logit,
            retention_min=0.5,
            retention_max=0.98,
        )
        target_retention = target_retentions_for_intervals(
            env=env,
            intervals=target_interval,
            retention_min=0.5,
            retention_max=0.98,
        )
        pred_log_interval = torch.log(
            continuous_intervals_for_retentions(
                env=env,
                s=env.s,
                retention=pred_retention,
            )
        )
        target_log_interval = torch.log(target_interval.to(dtype=torch.float32))
        weights = interval_underprediction_weights(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            labels=target_interval,
            env=env,
            underprediction_loss_weight=config.underprediction_weight,
            terminal_underprediction_loss_weight=config.terminal_underprediction_weight,
        )
        expected_interval = weighted_log_interval_smooth_l1(
            pred_log_interval=pred_log_interval,
            target_log_interval=target_log_interval,
            weights=weights,
        )
        expected_retention = nn.functional.smooth_l1_loss(
            pred_logit,
            retention_logits_for_retentions(
                target_retention,
                retention_min=0.5,
                retention_max=0.98,
            ),
        )
        expected_total = (
            config.interval_weight * expected_interval
            + config.retention_logit_weight * expected_retention
        )

        self.assertTrue(torch.allclose(result.interval, expected_interval))
        self.assertTrue(torch.allclose(result.retention_logit, expected_retention))
        self.assertTrue(torch.allclose(result.total, expected_total))
        result.total.backward()
        self.assertIsNotNone(pred_logit.grad)


if __name__ == "__main__":
    unittest.main()
