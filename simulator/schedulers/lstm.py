import argparse
from typing import Any, List, Tuple

from simulator.core import CardView, Scheduler
from simulator.models.lstm import LSTMModel


class LSTMScheduler(Scheduler):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.model = LSTMModel(
            user_id=args.user_id,
            benchmark_root=args.srs_benchmark_root,
            device=args.torch_device,
        )
        self.desired_retention = args.desired_retention

    def init_card(
        self, card_view: CardView, rating: int, day: float
    ) -> Tuple[float, Any]:
        history: List[Tuple[float, int]] = [(0.0, rating)]
        interval = self.model.get_interval_from_history(
            history, self.desired_retention
        )
        return interval, history

    def schedule(
        self, card_view: CardView, rating: int, elapsed: float, day: float
    ) -> Tuple[float, Any]:
        history: List[Tuple[float, int]] = card_view.scheduler_state or []
        history.append((elapsed, rating))

        if len(history) > self.model.max_events:
            del history[: len(history) - self.model.max_events]

        interval = self.model.get_interval_from_history(
            history, self.desired_retention
        )
        return interval, history


__all__ = ["LSTMScheduler"]