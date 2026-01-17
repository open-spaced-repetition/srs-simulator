from __future__ import annotations

import math
from typing import Sequence

from simulator.core import Card, MemoryModel
from simulator.math.dash import dash_time_window_features
from simulator.config_loader import load_weights_config

_DEFAULT_WEIGHTS = load_weights_config("dash.json", expected_len=9)


class DASHModel(MemoryModel):
    """
    DASH-inspired stateless model.

    The real paper uses a richer feature set; this lightweight variant keeps the
    same interface and can be swapped out for a full implementation later.
    """

    def __init__(self, weights: Sequence[float] | None = None):
        coeffs = list(weights) if weights is not None else list(_DEFAULT_WEIGHTS)
        if len(coeffs) != 9:
            raise ValueError("DASHModel expects 9 weights.")
        self.w = tuple(float(x) for x in coeffs)

    def init_card(self, card: Card, rating: int) -> None:
        # stateless: no per-card memory_state needed
        card.memory_state = {}

    def predict_retention(self, card: Card, elapsed: float) -> float:
        feats = dash_time_window_features(card.history, elapsed)
        score = sum(math.log1p(f) * w for f, w in zip(feats, self.w[:8])) + self.w[8]
        return 1.0 / (1.0 + math.exp(-score))

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        # stateless, nothing to update beyond history (handled by simulator)
        return
