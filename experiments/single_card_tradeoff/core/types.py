from __future__ import annotations

import argparse
from dataclasses import dataclass

from experiments.single_card_tradeoff.core.config import (
    SingleCardFSRS6Config,
    SingleCardRuntimeContext,
)
from simulator.behavior import StochasticBehavior
from simulator.cost import StatefulCostModel


@dataclass(frozen=True)
class MemoryTargetRegretAucSummary:
    user_id: int
    environment: str
    review_markov_transition: bool | None
    baseline_scheduler: str
    scheduler: str
    baseline_point_count: int
    scheduler_point_count: int
    baseline_frontier_count: int
    scheduler_frontier_count: int
    target_count: int
    covered_target_count: int
    total_span: float
    covered_span: float
    same_target_time_saved_auc: float | None
    baseline_time_auc: float | None

    @property
    def time_regret_auc(self) -> float | None:
        if self.same_target_time_saved_auc is None:
            return None
        return -self.same_target_time_saved_auc


@dataclass
class SimMetrics:
    card_expected_retrievability: float
    card_minutes_per_day: float
    card_reviews_per_day: float
    card_total_reviews: float
    card_total_lapses: float
    card_total_cost_seconds: float
    observed_retention: float | None


@dataclass(frozen=True)
class UserContext:
    user_id: int
    args: argparse.Namespace
    runtime_context: SingleCardRuntimeContext
    fsrs_config: SingleCardFSRS6Config
    behavior: StochasticBehavior
    cost_model: StatefulCostModel


@dataclass(frozen=True)
class SchedulerPoint:
    scheduler_spec: str
    fixed_interval: float | None = None
    desired_retention: float | None = None


@dataclass(frozen=True)
class BatchRow:
    user_context: UserContext
    point: SchedulerPoint
