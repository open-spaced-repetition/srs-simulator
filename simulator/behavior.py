from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .core import Action, BehaviorModel, CardView, review_first_priority


@dataclass
class RatingDistribution:
    success_weights: List[float] = field(default_factory=lambda: [0.2, 0.6, 0.2])

    def __post_init__(self) -> None:
        total = sum(self.success_weights)
        if not total:
            raise ValueError("success_weights must sum to > 0")
        self.success_weights = [w / total for w in self.success_weights]


class StochasticBehavior(BehaviorModel):
    """Simple behavior model with attendance, daily limits, and lazy grading."""

    def __init__(
        self,
        attendance_prob: float = 0.95,
        lazy_good_bias: float = 0.0,
        success_distribution: Optional[RatingDistribution] = None,
        max_new_per_day: Optional[int] = None,
        max_reviews_per_day: Optional[int] = None,
        max_cost_per_day: Optional[float] = None,
        priority_fn: Optional[Callable[[CardView], Sequence[float]]] = None,
    ) -> None:
        self.attendance_prob = attendance_prob
        self.lazy_good_bias = lazy_good_bias
        self.success_dist = success_distribution or RatingDistribution()
        self.max_new_per_day = max_new_per_day
        self.max_reviews_per_day = max_reviews_per_day
        self.max_cost_per_day = max_cost_per_day
        self._priority_fn = priority_fn or review_first_priority
        self._current_day: Optional[int] = None
        self._attending = True
        self._stop_for_day = False
        self._learned_today = 0
        self._reviews_today = 0
        self._cost_today = 0.0

    def start_day(self, day: int, rng: Callable[[], float]) -> None:
        if self._current_day == day:
            return
        self._current_day = day
        self._learned_today = 0
        self._reviews_today = 0
        self._cost_today = 0.0
        self._stop_for_day = False
        self._attending = rng() <= self.attendance_prob
        if not self._attending:
            self._stop_for_day = True

    def choose_action(
        self,
        day: int,
        next_review: Optional[CardView],
        next_new: Optional[CardView],
        rng: Callable[[], float],
    ) -> Optional[Action]:
        if self._current_day != day or self._stop_for_day:
            return None
        if (
            self.max_cost_per_day is not None
            and self._cost_today >= self.max_cost_per_day
        ):
            self._stop_for_day = True
            return None

        review_key: Optional[Tuple[float, ...]] = None
        if next_review is not None and (
            self.max_reviews_per_day is None
            or self._reviews_today < self.max_reviews_per_day
        ):
            review_key = tuple(self.priority_key(next_review))
        else:
            next_review = None

        new_key: Optional[Tuple[float, ...]] = None
        if next_new is not None and (
            self.max_new_per_day is None or self._learned_today < self.max_new_per_day
        ):
            new_key = tuple(self.priority_key(next_new))
        else:
            next_new = None

        if next_review is None or review_key is None:
            if next_new is None or new_key is None:
                self._stop_for_day = True
                return None
            return Action.LEARN
        if next_new is None or new_key is None:
            return Action.REVIEW
        return Action.LEARN if new_key < review_key else Action.REVIEW

    def priority_key(self, card_view: CardView) -> Sequence[float]:
        return self._priority_fn(card_view)

    def record_learning(self, cost: float) -> None:
        self._learned_today += 1
        self._cost_today += cost
        self._check_cost_limit()

    def initial_rating(self, rng: Callable[[], float]) -> int:
        return self._sample_success_rating(rng)

    def review_rating(
        self,
        retrievability: float,
        card_view: CardView,
        day: float,
        rng: Callable[[], float],
    ) -> Optional[int]:
        if self._stop_for_day or self._current_day != int(day):
            return None
        if (
            self.max_cost_per_day is not None
            and self._cost_today >= self.max_cost_per_day
        ):
            self._stop_for_day = True
            return None
        if rng() > retrievability:
            return 1
        if rng() < self.lazy_good_bias:
            return 3
        return self._sample_success_rating(rng)

    def record_review(self, cost: float) -> None:
        self._reviews_today += 1
        self._cost_today += cost
        self._check_cost_limit()

    def _check_cost_limit(self) -> None:
        if (
            self.max_cost_per_day is not None
            and self._cost_today >= self.max_cost_per_day
        ):
            self._stop_for_day = True

    def _sample_success_rating(self, rng: Callable[[], float]) -> int:
        p = rng()
        thresholds = self.success_dist.success_weights
        if p < thresholds[0]:
            return 2
        if p < thresholds[0] + thresholds[1]:
            return 3
        return 4
