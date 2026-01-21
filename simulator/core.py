from __future__ import annotations

import abc
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


@dataclass(slots=True)
class ReviewLog:
    """Single review interaction."""

    rating: int
    elapsed: float
    day: float


@dataclass(slots=True)
class Card:
    """Internal card representation holding hidden and public state."""

    id: int
    due: int = 0
    last_review: int = -1
    interval: int = 0
    reps: int = 0
    lapses: int = 0
    history: List[ReviewLog] = field(default_factory=list)
    memory_state: Any = None
    scheduler_state: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CardView:
    """Scheduler-visible projection of a card."""

    id: int
    due: int
    last_review: int
    interval: int
    reps: int
    lapses: int
    history: Sequence[ReviewLog]
    scheduler_state: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_placeholder: bool = False


@dataclass(slots=True)
class Event:
    day: int
    action: Action
    card_id: int
    rating: Optional[int] = None
    retrievability: Optional[float] = None
    cost: float = 0.0
    interval: Optional[int] = None
    due: Optional[int] = None
    last_review: Optional[int] = None
    elapsed: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "action": self.action.value,
            "card_id": self.card_id,
            "rating": self.rating,
            "retrievability": self.retrievability,
            "cost": self.cost,
            "interval": self.interval,
            "due": self.due,
            "last_review": self.last_review,
            "elapsed": self.elapsed,
        }


@dataclass(slots=True)
class SimulationStats:
    daily_reviews: List[int]
    daily_new: List[int]
    daily_retention: List[float]
    daily_cost: List[float]
    daily_memorized: List[float]
    total_reviews: int
    total_lapses: int
    total_cost: float
    events: List[Event]
    total_projected_retrievability: float


class Action(Enum):
    REVIEW = "review"
    LEARN = "learn"


class ReadyQueue:
    """Stable priority queue for ready reviews."""

    def __init__(self) -> None:
        self._heap: List[tuple[Sequence[float], int, int]] = []
        self._counter = 0

    def push(self, priority: Sequence[float], card_id: int) -> None:
        heapq.heappush(self._heap, (tuple(priority), self._counter, card_id))
        self._counter += 1

    def peek(self) -> Optional[int]:
        if not self._heap:
            return None
        return self._heap[0][2]

    def pop(self) -> Optional[int]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def drain(self) -> Iterable[int]:
        while self._heap:
            yield heapq.heappop(self._heap)[2]

    def __bool__(self) -> bool:
        return bool(self._heap)


class MemoryModel(abc.ABC):
    """Environment side: simulates the real memory dynamics."""

    @abc.abstractmethod
    def init_card(self, card: Card, rating: int) -> None:
        """Initialize memory_state after the first learning rating."""

    @abc.abstractmethod
    def predict_retention(self, card: Card, elapsed: float) -> float:
        """Return probability of successful recall after `elapsed` days."""

    @abc.abstractmethod
    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        """Update memory_state after observing `rating` at `elapsed` days."""


class BehaviorModel(abc.ABC):
    """User behavior: attendance, limits, grading."""

    @abc.abstractmethod
    def start_day(self, day: int, rng: Callable[[], float]) -> None:
        """Reset per-day state before acting on the given day."""

    @abc.abstractmethod
    def choose_action(
        self,
        day: int,
        next_review: Optional[CardView],
        next_new: Optional[CardView],
        rng: Callable[[], float],
    ) -> Optional[Action]:
        """Return an `Action` (review/learn) or None to indicate the user is done for the day."""

    @abc.abstractmethod
    def priority_key(self, card_view: CardView) -> Sequence[float]:
        """Return the ordering key for scheduling reviews (and synthetic new cards)."""

    @abc.abstractmethod
    def record_learning(self, cost: float) -> None:
        """Record the cost/time spent on a new card."""

    @abc.abstractmethod
    def initial_rating(self, rng: Callable[[], float]) -> int:
        """Return the rating given when a new card is first learned."""

    @abc.abstractmethod
    def review_rating(
        self,
        retrievability: float,
        card_view: CardView,
        day: float,
        rng: Callable[[], float],
    ) -> Optional[int]:
        """Return an optional rating; None indicates the user skipped the rest of the day."""

    @abc.abstractmethod
    def record_review(self, cost: float) -> None:
        """Record the cost/time spent on a review."""


class CostModel(abc.ABC):
    """Dynamic workload estimator."""

    @abc.abstractmethod
    def learning_cost(
        self,
        rating: int,
        card_view: CardView,
        day: float,
        rng: Optional[Callable[[], float]] = None,
    ) -> float:
        """Return the cost of introducing a new card."""

    @abc.abstractmethod
    def review_cost(
        self,
        retrievability: float,
        rating: int,
        card_view: CardView,
        day: float,
    ) -> float:
        """Return the cognitive/time cost of this review."""


class Scheduler(abc.ABC):
    """Agent under evaluation."""

    @abc.abstractmethod
    def init_card(
        self, card_view: CardView, rating: int, day: float
    ) -> Tuple[float, Any]:
        """Return (next interval, scheduler_state) for newly learned cards."""

    @abc.abstractmethod
    def schedule(
        self, card_view: CardView, rating: int, elapsed: float, day: float
    ) -> Tuple[float, Any]:
        """Return (next interval, scheduler_state) after a review."""

    def review_priority(self, card_view: CardView, day: float) -> Sequence[float]:
        """Priority hint for pending reviews; lower tuples are served first."""
        return (card_view.due, card_view.id)


class SimulationEngine:
    def __init__(
        self,
        *,
        days: int,
        deck_size: int,
        environment: MemoryModel,
        scheduler: Scheduler,
        behavior: BehaviorModel,
        cost_model: CostModel,
        seed_fn: Optional[Callable[[], float]],
        progress: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.days = days
        self.deck_size = deck_size
        self.environment = environment
        self.scheduler = scheduler
        self.behavior = behavior
        self.cost_model = cost_model
        self.rng = seed_fn or (lambda: __import__("random").random())
        self.progress = progress
        self.progress_callback = progress_callback
        self._progress_last = -1
        self._progress_bar: Optional[tqdm] = None

        self.cards = [Card(id=i) for i in range(deck_size)]
        self.future_queue: List[tuple[float, int, int]] = []
        self.ready_queue = ReadyQueue()
        self._future_counter = 0
        self._new_ptr = 0

        self.daily_reviews = [0 for _ in range(days)]
        self.daily_new = [0 for _ in range(days)]
        self.daily_retention = [0.0 for _ in range(days)]
        self.daily_cost = [0.0 for _ in range(days)]
        self.daily_memorized = [0.0 for _ in range(days)]
        self.daily_lapses = [0 for _ in range(days)]
        self.total_reviews = 0
        self.total_lapses = 0
        self.total_cost = 0.0
        self.events: List[Event] = []

    def run(self) -> SimulationStats:
        self._start_progress()
        try:
            self._update_progress(0)
            for day in range(self.days):
                self._start_day(day)
                self._compute_memorized(day)
                self._process_day(day)
                self._defer_remaining(day)
                self._update_progress(day + 1)
        finally:
            self._finish_progress()
        self._compute_retention()
        return SimulationStats(
            daily_reviews=self.daily_reviews,
            daily_new=self.daily_new,
            daily_retention=self.daily_retention,
            daily_cost=self.daily_cost,
            daily_memorized=self.daily_memorized,
            total_reviews=self.total_reviews,
            total_lapses=self.total_lapses,
            total_cost=self.total_cost,
            events=self.events,
            total_projected_retrievability=self._projected_retrievability_sum(),
        )

    def _start_progress(self) -> None:
        if not self.progress or self._progress_bar is not None:
            return
        self._progress_bar = tqdm(
            total=self.days, desc="Simulating", unit="day", leave=False
        )

    def _finish_progress(self) -> None:
        if self._progress_bar is None:
            return
        self._progress_bar.close()
        self._progress_bar = None

    def _update_progress(self, completed: int) -> None:
        if self.days <= 0:
            return
        if completed == self._progress_last:
            return
        self._progress_last = completed
        if self.progress_callback is not None:
            self.progress_callback(completed, self.days)
        if self._progress_bar is None:
            return
        delta = completed - self._progress_bar.n
        if delta > 0:
            self._progress_bar.update(delta)

    def _projected_retrievability_sum(self) -> float:
        total = 0.0
        for card in self.cards:
            if card.reps <= 0:
                continue
            elapsed = max(0.0, float(self.days - card.last_review))
            total += self.environment.predict_retention(card, elapsed)
        return total

    def _start_day(self, day: int) -> None:
        self.behavior.start_day(day, self.rng)
        while self.future_queue and self.future_queue[0][0] <= day:
            _, _, cid = heapq.heappop(self.future_queue)
            self._push_ready(cid, day)

    def _compute_memorized(self, day: int) -> None:
        memorized = 0.0
        for card in self.cards:
            if card.reps <= 0:
                continue
            elapsed = max(0.0, float(day - card.last_review))
            memorized += self.environment.predict_retention(card, elapsed)
        self.daily_memorized[day] = memorized

    def _process_day(self, day: int) -> None:
        while True:
            next_review_view = self._peek_ready_view()
            next_new_view = self._next_new_view(day)
            action = self.behavior.choose_action(
                day, next_review_view, next_new_view, self.rng
            )
            if action is None:
                break
            if action == Action.REVIEW:
                if not self._handle_review(day):
                    break
                continue
            if action == Action.LEARN:
                if not self._handle_learning(day):
                    break
                continue
            break

    def _handle_review(self, day: int) -> bool:
        cid = self.ready_queue.pop()
        if cid is None:
            return False
        card = self.cards[cid]
        elapsed = float(day) - float(card.last_review)
        retrievability = self.environment.predict_retention(card, elapsed)
        view = _card_view(card)
        rating = self.behavior.review_rating(retrievability, view, float(day), self.rng)
        if rating is None:
            self._schedule_card(card)
            return False
        self.environment.update_card(card, rating, elapsed)
        interval, sched_state = self.scheduler.schedule(
            view, rating, elapsed, float(day)
        )
        self._apply_schedule(card, interval, sched_state, day)
        card.reps += 1
        card.history.append(ReviewLog(rating, elapsed, float(day)))
        self._schedule_card(card)

        self.daily_reviews[day] += 1
        self.total_reviews += 1
        if rating == 1:
            card.lapses += 1
            self.daily_lapses[day] += 1
            self.total_lapses += 1

        cost = self.cost_model.review_cost(retrievability, rating, view, float(day))
        self.daily_cost[day] += cost
        self.total_cost += cost
        self.behavior.record_review(cost)
        self.events.append(
            Event(
                day=day,
                action=Action.REVIEW,
                card_id=card.id,
                rating=rating,
                retrievability=retrievability,
                cost=cost,
                interval=card.interval,
                due=card.due,
                last_review=card.last_review,
                elapsed=elapsed,
            )
        )
        return True

    def _handle_learning(self, day: int) -> bool:
        if self._new_ptr >= self.deck_size:
            return False
        card = self.cards[self._new_ptr]
        self._new_ptr += 1
        card.scheduler_state = None
        card.memory_state = None
        card.last_review = day
        card.interval = 0
        card.reps = 0
        card.lapses = 0
        card.history.clear()

        first_rating = self.behavior.initial_rating(self.rng)
        self.environment.init_card(card, first_rating)
        view = _card_view(card)
        learn_cost = self.cost_model.learning_cost(
            first_rating, view, float(day), self.rng
        )
        interval, sched_state = self.scheduler.init_card(view, first_rating, float(day))
        self._apply_schedule(card, interval, sched_state, day)
        card.reps = 1
        card.history.append(ReviewLog(first_rating, 0.0, float(day)))
        self.daily_new[day] += 1
        self.daily_cost[day] += learn_cost
        self.total_cost += learn_cost
        self.behavior.record_learning(learn_cost)
        self.events.append(
            Event(
                day=day,
                action=Action.LEARN,
                card_id=card.id,
                rating=first_rating,
                retrievability=None,
                cost=learn_cost,
                interval=card.interval,
                due=card.due,
                last_review=card.last_review,
                elapsed=float(day) - float(card.last_review),
            )
        )
        self._schedule_card(card)
        return True

    def _defer_remaining(self, day: int) -> None:
        for cid in self.ready_queue.drain():
            card = self.cards[cid]
            self._schedule_card(card)

    def _apply_schedule(
        self, card: Card, interval: float, sched_state: Any, day: int
    ) -> None:
        card.scheduler_state = sched_state
        interval_days = max(1, int(round(interval)))
        card.interval = interval_days
        card.last_review = day
        card.due = day + interval_days

    def _schedule_card(self, card: Card) -> None:
        heapq.heappush(self.future_queue, (card.due, self._future_counter, card.id))
        self._future_counter += 1

    def _push_ready(self, cid: int, day: int) -> None:
        card = self.cards[cid]
        view = _card_view(card)
        priority = tuple(self.scheduler.review_priority(view, float(day)))
        card.metadata["scheduler_priority"] = priority
        self.ready_queue.push(priority, cid)

    def _peek_ready_view(self) -> Optional[CardView]:
        cid = self.ready_queue.peek()
        if cid is None:
            return None
        return _card_view(self.cards[cid])

    def _next_new_view(self, day: int) -> Optional[CardView]:
        if self._new_ptr >= self.deck_size:
            return None
        return _new_card_placeholder(day, self._new_ptr)

    def _compute_retention(self) -> None:
        for i, r in enumerate(self.daily_reviews):
            self.daily_retention[i] = 0.0 if r == 0 else 1.0 - self.daily_lapses[i] / r


def simulate(
    *,
    days: int,
    deck_size: int,
    environment: MemoryModel,
    scheduler: Scheduler,
    behavior: BehaviorModel,
    cost_model: CostModel,
    seed_fn: Optional[Callable[[], float]] = None,
    progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SimulationStats:
    """Run an event-driven simulation with explicit environment/behavior separation."""

    engine = SimulationEngine(
        days=days,
        deck_size=deck_size,
        environment=environment,
        scheduler=scheduler,
        behavior=behavior,
        cost_model=cost_model,
        seed_fn=seed_fn,
        progress=progress,
        progress_callback=progress_callback,
    )
    return engine.run()


def _card_view(card: Card) -> CardView:
    return CardView(
        id=card.id,
        due=card.due,
        last_review=card.last_review,
        interval=card.interval,
        reps=card.reps,
        lapses=card.lapses,
        history=card.history,
        scheduler_state=card.scheduler_state,
        metadata=card.metadata,
        is_placeholder=False,
    )


def _new_card_placeholder(day: int, next_id: int) -> CardView:
    hint = (float(day), -float(next_id + 1))
    return CardView(
        id=-(next_id + 1),
        due=day,
        last_review=day,
        interval=0,
        reps=0,
        lapses=0,
        history=(),
        scheduler_state=None,
        metadata={"scheduler_priority": hint},
        is_placeholder=True,
    )


def _scheduler_priority_hint(view: CardView) -> Tuple[float, ...]:
    hint = view.metadata.get("scheduler_priority")
    if hint is None:
        return (view.due, view.id)
    return tuple(hint)


def review_first_priority(view: CardView) -> Sequence[float]:
    """Prioritize review cards (reps > 0) before new cards, then by due time."""
    is_new = 1 if view.reps == 0 else 0
    base = _scheduler_priority_hint(view)
    return (is_new,) + base


def new_first_priority(view: CardView) -> Sequence[float]:
    """Prioritize new cards (reps == 0) before review cards, then by due time."""
    is_review = 1 if view.reps > 0 else 0
    base = _scheduler_priority_hint(view)
    return (is_review,) + base
