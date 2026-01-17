from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_weights_config(
    filename: str,
    *,
    expected_len: int | None = None,
    key: str = "weights",
) -> Tuple[float, ...]:
    """
    Load a weight vector from `config/<filename>`.
    """
    path = _CONFIG_DIR / filename
    data = _read_json(path)
    weights = data.get(key)
    if not isinstance(weights, Sequence):
        raise ValueError(f"{path} missing '{key}' sequence.")
    vector = tuple(float(w) for w in weights)
    if expected_len is not None and len(vector) != expected_len:
        raise ValueError(f"{path} expected {expected_len} weights, got {len(vector)}.")
    return vector


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


__all__ = ["load_weights_config"]
