from __future__ import annotations

import argparse
from typing import Any, cast
import unittest

import torch
from torch import nn

from experiments.single_card_tradeoff.cli.oracle_continuous_stationary_finite_distill_multiuser import (
    ContinuousDistillLossWeightingConfig,
    ContinuousTableBatch,
    continuous_table_retention_distill_loss,
    log_intervals_for_retentions,
    q_gap_loss_weights,
    resolve_loss_weighting,
)
from experiments.single_card_tradeoff.models.policy_runtime import (
    predicted_retentions,
    retention_logits_for_retentions,
)


class _ToyGuide:
    horizon = 9
    factor = torch.tensor([0.1], dtype=torch.float32)
    decay = torch.tensor([-0.5], dtype=torch.float32)


class ContinuousStationaryFiniteDistillLossTests(unittest.TestCase):
    def _guide(self) -> Any:
        return cast(Any, _ToyGuide())

    def test_baseline_loss_matches_unweighted_smooth_l1(self) -> None:
        guide = self._guide()
        target_retention = torch.tensor([[0.8, 0.7]], dtype=torch.float32)
        pred_retention = torch.tensor([[0.75, 0.72]], dtype=torch.float32)
        pred_logit = retention_logits_for_retentions(
            pred_retention,
            retention_min=0.5,
            retention_max=0.98,
        )
        batch = ContinuousTableBatch(
            obs=torch.zeros((1, 2, 3), dtype=torch.float32),
            target_retention=target_retention,
            s_values=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
            goal_norm=torch.ones((1, 2), dtype=torch.float32),
            q_gap=None,
            q_gap_normalizer=None,
        )
        config = ContinuousDistillLossWeightingConfig(
            mode="baseline",
            underprediction_loss_weight=0.0,
            terminal_underprediction_loss_weight=0.0,
            q_gap_loss_weight=0.0,
            q_gap_weight_cap=8.0,
        )

        actual = continuous_table_retention_distill_loss(
            pred_logit=pred_logit,
            batch=batch,
            guide=guide,
            config=config,
            retention_min=0.5,
            retention_max=0.98,
            interval_loss_weight=1.3,
            retention_logit_loss_weight=0.4,
        )
        pred_log_interval = log_intervals_for_retentions(
            guide=guide,
            s=batch.s_values,
            retention=predicted_retentions(
                pred_logit,
                retention_min=0.5,
                retention_max=0.98,
            ),
        )
        target_log_interval = log_intervals_for_retentions(
            guide=guide,
            s=batch.s_values,
            retention=target_retention,
        )
        expected_interval = nn.functional.smooth_l1_loss(
            pred_log_interval,
            target_log_interval,
            reduction="none",
        ).mean(dim=1)
        expected_retention = nn.functional.smooth_l1_loss(
            pred_logit,
            retention_logits_for_retentions(
                target_retention,
                retention_min=0.5,
                retention_max=0.98,
            ),
            reduction="none",
        ).mean(dim=1)
        expected_total = 1.3 * expected_interval + 0.4 * expected_retention

        self.assertTrue(torch.allclose(actual.interval_by_user, expected_interval))
        self.assertTrue(torch.allclose(actual.retention_by_user, expected_retention))
        self.assertTrue(torch.allclose(actual.total_by_user, expected_total))
        self.assertTrue(torch.allclose(actual.q_gap_weight_by_user, torch.ones(1)))

    def test_underprediction_weight_penalizes_short_intervals(self) -> None:
        guide = self._guide()
        target_retention = torch.tensor([[0.6, 0.9]], dtype=torch.float32)
        pred_retention = torch.tensor([[0.9, 0.6]], dtype=torch.float32)
        pred_logit = retention_logits_for_retentions(
            pred_retention,
            retention_min=0.5,
            retention_max=0.98,
        )
        batch = ContinuousTableBatch(
            obs=torch.zeros((1, 2, 3), dtype=torch.float32),
            target_retention=target_retention,
            s_values=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
            goal_norm=torch.ones((1, 2), dtype=torch.float32),
            q_gap=None,
            q_gap_normalizer=None,
        )
        config = ContinuousDistillLossWeightingConfig(
            mode="underpred",
            underprediction_loss_weight=3.0,
            terminal_underprediction_loss_weight=0.0,
            q_gap_loss_weight=0.0,
            q_gap_weight_cap=8.0,
        )

        actual = continuous_table_retention_distill_loss(
            pred_logit=pred_logit,
            batch=batch,
            guide=guide,
            config=config,
            retention_min=0.5,
            retention_max=0.98,
            interval_loss_weight=1.0,
            retention_logit_loss_weight=0.0,
        )
        baseline = continuous_table_retention_distill_loss(
            pred_logit=pred_logit,
            batch=batch,
            guide=guide,
            config=ContinuousDistillLossWeightingConfig(
                mode="baseline",
                underprediction_loss_weight=0.0,
                terminal_underprediction_loss_weight=0.0,
                q_gap_loss_weight=0.0,
                q_gap_weight_cap=8.0,
            ),
            retention_min=0.5,
            retention_max=0.98,
            interval_loss_weight=1.0,
            retention_logit_loss_weight=0.0,
        )

        self.assertAlmostEqual(float(actual.underprediction_rate_by_user.item()), 0.5)
        self.assertGreater(
            float(actual.interval_by_user.item()),
            float(baseline.interval_by_user.item()),
        )

    def test_terminal_underprediction_weight_applies_to_terminal_target(self) -> None:
        guide = self._guide()
        target_retention = torch.tensor([[0.5]], dtype=torch.float32)
        pred_retention = torch.tensor([[0.98]], dtype=torch.float32)
        pred_logit = retention_logits_for_retentions(
            pred_retention,
            retention_min=0.5,
            retention_max=0.98,
        )
        batch = ContinuousTableBatch(
            obs=torch.zeros((1, 1, 3), dtype=torch.float32),
            target_retention=target_retention,
            s_values=torch.tensor([[10.0]], dtype=torch.float32),
            goal_norm=torch.zeros((1, 1), dtype=torch.float32),
            q_gap=None,
            q_gap_normalizer=None,
        )
        weighted = continuous_table_retention_distill_loss(
            pred_logit=pred_logit,
            batch=batch,
            guide=guide,
            config=ContinuousDistillLossWeightingConfig(
                mode="underpred",
                underprediction_loss_weight=0.0,
                terminal_underprediction_loss_weight=5.0,
                q_gap_loss_weight=0.0,
                q_gap_weight_cap=8.0,
            ),
            retention_min=0.5,
            retention_max=0.98,
            interval_loss_weight=1.0,
            retention_logit_loss_weight=0.0,
        )
        baseline = continuous_table_retention_distill_loss(
            pred_logit=pred_logit,
            batch=batch,
            guide=guide,
            config=ContinuousDistillLossWeightingConfig(
                mode="baseline",
                underprediction_loss_weight=0.0,
                terminal_underprediction_loss_weight=0.0,
                q_gap_loss_weight=0.0,
                q_gap_weight_cap=8.0,
            ),
            retention_min=0.5,
            retention_max=0.98,
            interval_loss_weight=1.0,
            retention_logit_loss_weight=0.0,
        )

        self.assertTrue(
            torch.allclose(weighted.interval_by_user, baseline.interval_by_user * 6.0)
        )

    def test_q_gap_weights_normalize_and_clamp(self) -> None:
        weights = q_gap_loss_weights(
            q_gap=torch.tensor([[0.0, 2.0, 10.0]], dtype=torch.float32),
            q_gap_normalizer=torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32),
            config=ContinuousDistillLossWeightingConfig(
                mode="qgap",
                underprediction_loss_weight=0.0,
                terminal_underprediction_loss_weight=0.0,
                q_gap_loss_weight=1.0,
                q_gap_weight_cap=3.0,
            ),
            reference=torch.zeros((1, 3), dtype=torch.float32),
        )

        self.assertTrue(
            torch.allclose(
                weights,
                torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            )
        )

    def test_loss_weighting_modes_fill_recommended_defaults(self) -> None:
        config = resolve_loss_weighting(
            argparse.Namespace(
                loss_weighting="underpred_qgap",
                underprediction_loss_weight=0.0,
                terminal_underprediction_loss_weight=0.0,
                q_gap_loss_weight=0.0,
                q_gap_weight_cap=8.0,
            )
        )

        self.assertEqual(config.underprediction_loss_weight, 4.0)
        self.assertEqual(config.terminal_underprediction_loss_weight, 16.0)
        self.assertEqual(config.q_gap_loss_weight, 1.0)
        self.assertEqual(config.q_gap_weight_cap, 8.0)


if __name__ == "__main__":
    unittest.main()
