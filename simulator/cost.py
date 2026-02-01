from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from simulator.core import CardView, CostModel


@dataclass(frozen=True)
class StateRatingCosts:
    learning: Sequence[float]
    review: Sequence[float]
    relearning: Sequence[float]


DEFAULT_STATE_RATING_COSTS = StateRatingCosts(
    learning=[33.79, 24.3, 13.68, 6.5],
    review=[23.0, 11.68, 7.33, 5.6],
    relearning=[16.44, 15.25, 12.32, 8.03],
)


class StatefulCostModel(CostModel):
    """FSRS-style state cost table plus latency penalty for low retrievability."""

    def __init__(
        self,
        base_seconds: float = 0.0,
        penalty: float = 0.0,
        state_costs: Optional[StateRatingCosts] = None,
    ) -> None:
        self.base = base_seconds
        self.penalty = penalty
        self.state_costs = state_costs or DEFAULT_STATE_RATING_COSTS

    def learning_cost(
        self,
        rating: int,
        card_view: CardView,
        day: float,
        rng=None,
    ) -> float:
        return self._lookup(self.state_costs.learning, rating)

    def review_cost(
        self,
        retrievability: float,
        rating: int,
        card_view: CardView,
        day: float,
    ) -> float:
        latency = self.base * (1.0 + self.penalty * max(0.0, 1.0 - retrievability))
        phase = getattr(card_view.scheduler_state, "phase", None)
        if phase == "learning":
            review = self._lookup(self.state_costs.learning, rating)
        elif phase == "relearning":
            review = self._lookup(self.state_costs.relearning, rating)
        else:
            review = self._lookup(self.state_costs.review, rating)
        return latency + review

    @staticmethod
    def _lookup(table: Sequence[float], rating: int) -> float:
        idx = max(1, min(4, rating)) - 1
        return table[idx]
