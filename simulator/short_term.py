from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from simulator.core import CardView, IntervalSpec, Scheduler

SECONDS_PER_DAY = 86_400.0


@dataclass
class ShortTermState:
    phase: str
    remaining_steps: int
    base_state: Any


class LearningSteps:
    """Anki-style learning steps (minutes), ported from rslib."""

    def __init__(self, steps: Sequence[float]) -> None:
        self.steps = [float(step) for step in steps if step is not None]

    def _get_index(self, remaining: int) -> int:
        total = len(self.steps)
        if total <= 0:
            return 0
        return min(total - 1, max(0, total - int(remaining % 1000)))

    def _secs_at_index(self, index: int) -> Optional[int]:
        if index < 0 or index >= len(self.steps):
            return None
        return int(self.steps[index] * 60.0)

    def again_delay_secs_learn(self) -> Optional[int]:
        return self._secs_at_index(0)

    def hard_delay_secs(self, remaining: int) -> Optional[int]:
        idx = self._get_index(remaining)
        current = self._secs_at_index(idx)
        if current is None:
            current = self._secs_at_index(0)
        if current is None:
            return None
        if idx == 0:
            return _hard_delay_secs_for_first_step(current, self._secs_at_index(1))
        return current

    def good_delay_secs(self, remaining: int) -> Optional[int]:
        idx = self._get_index(remaining)
        return self._secs_at_index(idx + 1)

    def remaining_for_good(self, remaining: int) -> int:
        idx = self._get_index(remaining)
        return max(0, len(self.steps) - (idx + 1))

    def remaining_for_failed(self) -> int:
        return len(self.steps)

    def is_empty(self) -> bool:
        return not self.steps


def _hard_delay_secs_for_first_step(again_secs: int, next_secs: Optional[int]) -> int:
    day_secs = int(SECONDS_PER_DAY)
    if next_secs is not None:
        return _maybe_round_in_days((again_secs + next_secs) // 2)
    secs = min(again_secs * 3 // 2, again_secs + day_secs)
    return _maybe_round_in_days(secs)


def _maybe_round_in_days(secs: int) -> int:
    day_secs = int(SECONDS_PER_DAY)
    if secs > day_secs:
        return int(round(secs / day_secs)) * day_secs
    return secs


class ShortTermScheduler(Scheduler):
    def __init__(
        self,
        base: Scheduler,
        *,
        learning_steps: Sequence[float],
        relearning_steps: Sequence[float],
        threshold_days: float = 0.5,
        allow_short_term_interval: bool = True,
    ) -> None:
        self.base = base
        self.learning_steps = LearningSteps(learning_steps)
        self.relearning_steps = LearningSteps(relearning_steps)
        self.threshold_days = float(threshold_days)
        self.allow_short_term_interval = bool(allow_short_term_interval)

    def init_card(self, card_view: CardView, rating: int, day: float):
        if self.learning_steps.is_empty():
            interval, state = self.base.init_card(card_view, rating, day)
            return self._convert_interval(interval, state, phase="learning")
        interval, state = self._schedule_learning(
            rating,
            phase="learning",
            steps=self.learning_steps,
            remaining=self.learning_steps.remaining_for_failed(),
        )
        if interval is None:
            interval, state = self.base.init_card(card_view, rating, day)
            return self._convert_interval(interval, state, phase="learning")
        return interval, state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state
        if isinstance(state, ShortTermState):
            steps = (
                self.learning_steps
                if state.phase == "learning"
                else self.relearning_steps
            )
            interval, next_state = self._schedule_learning(
                rating,
                phase=state.phase,
                steps=steps,
                remaining=state.remaining_steps,
                base_state=state.base_state,
            )
            if interval is not None:
                return interval, next_state
            base_view = _with_scheduler_state(card_view, state.base_state)
            interval, base_state = self.base.schedule(base_view, rating, elapsed, day)
            phase = "relearning" if rating == 1 else "learning"
            return self._convert_interval(interval, base_state, phase=phase)

        if rating == 1 and not self.relearning_steps.is_empty():
            interval, next_state = self._schedule_learning(
                rating,
                phase="relearning",
                steps=self.relearning_steps,
                remaining=self.relearning_steps.remaining_for_failed(),
                base_state=card_view.scheduler_state,
            )
            if interval is not None:
                return interval, next_state
        interval, state = self.base.schedule(card_view, rating, elapsed, day)
        phase = "relearning" if rating == 1 else "learning"
        return self._convert_interval(interval, state, phase=phase)

    def review_priority(self, card_view: CardView, day: float):
        state = card_view.scheduler_state
        if isinstance(state, ShortTermState):
            if (
                state.phase == "learning"
                and not self.learning_steps.is_empty()
                or state.phase == "relearning"
                and not self.relearning_steps.is_empty()
            ):
                return (card_view.due, card_view.id)
            base_view = _with_scheduler_state(card_view, state.base_state)
            return self.base.review_priority(base_view, day)
        return self.base.review_priority(card_view, day)

    def _schedule_learning(
        self,
        rating: int,
        *,
        phase: str,
        steps: LearningSteps,
        remaining: int,
        base_state: Any = None,
    ) -> tuple[Optional[IntervalSpec], Any]:
        if steps.is_empty():
            return None, base_state
        if rating == 1:
            delay = steps.again_delay_secs_learn()
            if delay is None:
                return None, base_state
            remaining = steps.remaining_for_failed()
            return IntervalSpec.secs(delay), ShortTermState(
                phase, remaining, base_state
            )
        if rating == 2:
            delay = steps.hard_delay_secs(remaining)
            if delay is None:
                return None, base_state
            return IntervalSpec.secs(delay), ShortTermState(
                phase, remaining, base_state
            )
        if rating == 3:
            delay = steps.good_delay_secs(remaining)
            if delay is None:
                return None, base_state
            remaining = steps.remaining_for_good(remaining)
            return IntervalSpec.secs(delay), ShortTermState(
                phase, remaining, base_state
            )
        if rating == 4:
            return None, base_state
        return None, base_state

    def _convert_interval(
        self,
        interval: float,
        state: Any,
        *,
        phase: str | None = None,
    ) -> tuple[IntervalSpec, Any]:
        interval_days = max(0.0, float(interval))
        if self.allow_short_term_interval and interval_days < self.threshold_days:
            interval_spec = IntervalSpec.secs(interval_days * SECONDS_PER_DAY)
            if (
                phase is not None
                and self.learning_steps.is_empty()
                and self.relearning_steps.is_empty()
            ):
                state = ShortTermState(phase, 0, state)
            return interval_spec, state
        return IntervalSpec.days(interval_days), state


def _with_scheduler_state(card_view: CardView, state: Any) -> CardView:
    return CardView(
        id=card_view.id,
        due=card_view.due,
        last_review=card_view.last_review,
        interval=card_view.interval,
        reps=card_view.reps,
        lapses=card_view.lapses,
        history=card_view.history,
        scheduler_state=state,
        metadata=card_view.metadata,
        is_placeholder=card_view.is_placeholder,
    )
