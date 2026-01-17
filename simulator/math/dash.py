from __future__ import annotations

import math
from typing import Sequence

from simulator.core import ReviewLog

TIME_WINDOWS = (1.0, 7.0, 30.0, float("inf"))
TAU = (0.2434, 1.9739, 16.0090, 129.8426)


def dash_time_window_features(
    history: Sequence[ReviewLog],
    future_elapsed: float,
    enable_decay: bool = False,
) -> list[float]:
    """
    Build the eight DASH time-window features as defined in the original repo.
    """
    n = len(history)
    if n == 0:
        return [0.0] * 8

    times_since = _times_since_reviews(history, future_elapsed)
    successes = [1.0 if log.rating > 1 else 0.0 for log in history]
    feats = [0.0] * 8

    for j, window in enumerate(TIME_WINDOWS):
        total = 0.0
        success = 0.0
        tau = TAU[j]
        for t, s in zip(times_since, successes):
            if t > window:
                continue
            decay = math.exp(-t / tau) if enable_decay else 1.0
            total += decay
            success += decay * s
        feats[j * 2] = total
        feats[j * 2 + 1] = success
    return feats


def _times_since_reviews(
    history: Sequence[ReviewLog], future_elapsed: float
) -> list[float]:
    """
    Return the time from each past review to the upcoming review that occurs
    `future_elapsed` days after the most recent review.
    """
    n = len(history)
    times = [0.0] * n
    running = max(0.0, float(future_elapsed))
    times[-1] = running
    for idx in range(n - 2, -1, -1):
        running += float(history[idx + 1].elapsed or 0.0)
        times[idx] = running
    return times


__all__ = ["dash_time_window_features"]
