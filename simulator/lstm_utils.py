from __future__ import annotations

import os


def resolve_lstm_max_batch_size(value: int | None) -> int | None:
    if value is not None:
        if value < 1:
            raise ValueError("max_batch_size must be >= 1 when set.")
        return value
    raw = os.getenv("SRS_LSTM_MAX_BATCH", "").strip().lower()
    if not raw or raw in {"0", "none", "off"}:
        return 65536
    parsed = int(raw)
    if parsed < 1:
        raise ValueError("SRS_LSTM_MAX_BATCH must be >= 1 when set.")
    return parsed
