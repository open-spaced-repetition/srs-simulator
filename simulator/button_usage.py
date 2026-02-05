from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

from simulator.behavior import DEFAULT_FIRST_RATING_PROB, DEFAULT_REVIEW_RATING_PROB
from simulator.cost import DEFAULT_STATE_RATING_COSTS

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUTTON_USAGE_PATH = (
    _REPO_ROOT.parent / "Anki-button-usage" / "button_usage.jsonl"
)

DEFAULT_FIRST_RATING_OFFSETS = (0.0, 0.0, 0.0, 0.0)
DEFAULT_FIRST_SESSION_LENS = (0.0, 0.0, 0.0, 0.0)
DEFAULT_FORGET_RATING_OFFSET = 0.0
DEFAULT_FORGET_SESSION_LEN = 0.0
DEFAULT_RELEARNING_RATING_PROB_FALLBACK = (0.0, 1.0, 0.0)


def _coerce_user_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_list(
    entry: Mapping[str, Any], key: str, *, expected_len: int | None = None
) -> list[float]:
    value = entry.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Invalid {key} in button usage config.")
    if expected_len is not None and len(value) != expected_len:
        raise ValueError(
            f"Invalid {key} length in button usage config: "
            f"expected {expected_len}, got {len(value)}."
        )
    return [float(item) for item in value]


def _require_matrix(
    entry: Mapping[str, Any], key: str, *, expected_shape: tuple[int, int]
) -> list[list[float]]:
    value = entry.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Invalid {key} in button usage config.")
    if len(value) != expected_shape[0]:
        raise ValueError(
            f"Invalid {key} length in button usage config: "
            f"expected {expected_shape[0]} rows, got {len(value)}."
        )
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"Invalid {key} entry in button usage config.")
        if len(row) != expected_shape[1]:
            raise ValueError(
                f"Invalid {key} row length: expected {expected_shape[1]}, got {len(row)}."
            )
        matrix.append([float(item) for item in row])
    return matrix


class ButtonUsageConfig(TypedDict):
    learn_costs: list[float]
    review_costs: list[float]
    first_rating_prob: list[float]
    review_rating_prob: list[float]
    learning_rating_prob: list[float]
    relearning_rating_prob: list[float]
    state_rating_costs: list[list[float]]
    first_rating_offsets: list[float]
    first_session_lens: list[float]
    forget_rating_offset: float
    forget_session_len: float


def load_button_usage_config(
    button_usage_path: str | Path, user_id: int
) -> ButtonUsageConfig:
    path = Path(button_usage_path)
    target_user_id = _coerce_user_id(user_id)
    if target_user_id is None:
        raise ValueError(f"Invalid user id: {user_id}")
    if not path.exists():
        raise FileNotFoundError(f"Button usage data not found: {path}")

    config: ButtonUsageConfig | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            entry_user_id = _coerce_user_id(entry.get("user"))
            if entry_user_id != target_user_id:
                continue

            review_rating_prob = _require_list(
                entry, "review_rating_prob", expected_len=3
            )
            learning_rating_prob = entry.get("learning_rating_prob")
            if learning_rating_prob is None:
                learning_rating_prob = review_rating_prob
            else:
                learning_rating_prob = _require_list(
                    entry, "learning_rating_prob", expected_len=3
                )
            relearning_rating_prob = entry.get("relearning_rating_prob")
            if relearning_rating_prob is None:
                relearning_rating_prob = review_rating_prob
            else:
                relearning_rating_prob = _require_list(
                    entry, "relearning_rating_prob", expected_len=3
                )
            config = {
                "learn_costs": _require_list(entry, "learn_costs", expected_len=4),
                "review_costs": _require_list(entry, "review_costs", expected_len=4),
                "first_rating_prob": _require_list(
                    entry, "first_rating_prob", expected_len=4
                ),
                "review_rating_prob": review_rating_prob,
                "learning_rating_prob": learning_rating_prob,
                "relearning_rating_prob": relearning_rating_prob,
                "state_rating_costs": _require_matrix(
                    entry, "state_rating_costs", expected_shape=(3, 4)
                ),
                "first_rating_offsets": _require_list(
                    entry, "first_rating_offset", expected_len=4
                ),
                "first_session_lens": _require_list(
                    entry, "first_session_len", expected_len=4
                ),
                "forget_rating_offset": float(entry.get("forget_rating_offset")),
                "forget_session_len": float(entry.get("forget_session_len")),
            }
            break

    if config is None:
        raise ValueError(f"User {target_user_id} not found in {path}")
    return config


def _normalize_prob(
    values: Sequence[float], key: str, *, fallback: Sequence[float]
) -> list[float]:
    if any((not math.isfinite(value)) or value < 0 for value in values):
        logging.warning(
            "%s contains invalid probabilities %s; using fallback.", key, values
        )
        values = fallback
    total = float(sum(values))
    if not math.isfinite(total) or total <= 0:
        logging.warning("%s sums to invalid value %.6f; using fallback.", key, total)
        values = fallback
        total = float(sum(values))
    if abs(total - 1.0) > 0.01:
        logging.warning("%s does not sum to 1 (%.6f); normalizing.", key, total)
        return [float(value) / total for value in values]
    return [float(value) for value in values]


def _coerce_list(
    value: Sequence[float], *, expected_len: int, name: str
) -> list[float]:
    if len(value) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}")
    return [float(item) for item in value]


def _coerce_state_costs(
    value: Sequence[Sequence[float]] | None,
    *,
    learn_costs: Sequence[float],
    review_costs: Sequence[float],
) -> list[list[float]]:
    if value is None:
        return [
            list(learn_costs),
            list(review_costs),
            list(review_costs),
        ]
    if len(value) != 3:
        raise ValueError("state_rating_costs must have 3 rows.")
    rows: list[list[float]] = []
    for idx, row in enumerate(value):
        rows.append(
            _coerce_list(row, expected_len=4, name=f"state_rating_costs[{idx}]")
        )
    return rows


def normalize_button_usage(
    button_usage: Mapping[str, Any] | None,
) -> ButtonUsageConfig:
    source: Mapping[str, Any] = button_usage or {}

    learn_costs = _coerce_list(
        source.get("learn_costs", DEFAULT_STATE_RATING_COSTS.learning),
        expected_len=4,
        name="learn_costs",
    )
    review_costs = _coerce_list(
        source.get("review_costs", DEFAULT_STATE_RATING_COSTS.review),
        expected_len=4,
        name="review_costs",
    )
    state_rating_costs = _coerce_state_costs(
        source.get("state_rating_costs"),
        learn_costs=learn_costs,
        review_costs=review_costs,
    )
    first_rating_prob = _normalize_prob(
        _coerce_list(
            source.get("first_rating_prob", DEFAULT_FIRST_RATING_PROB),
            expected_len=4,
            name="first_rating_prob",
        ),
        "first_rating_prob",
        fallback=DEFAULT_FIRST_RATING_PROB,
    )
    review_rating_prob = _normalize_prob(
        _coerce_list(
            source.get("review_rating_prob", DEFAULT_REVIEW_RATING_PROB),
            expected_len=3,
            name="review_rating_prob",
        ),
        "review_rating_prob",
        fallback=DEFAULT_REVIEW_RATING_PROB,
    )
    learning_rating_prob = _normalize_prob(
        _coerce_list(
            source.get("learning_rating_prob", review_rating_prob),
            expected_len=3,
            name="learning_rating_prob",
        ),
        "learning_rating_prob",
        fallback=review_rating_prob,
    )
    relearning_rating_prob = _normalize_prob(
        _coerce_list(
            source.get("relearning_rating_prob", review_rating_prob),
            expected_len=3,
            name="relearning_rating_prob",
        ),
        "relearning_rating_prob",
        fallback=DEFAULT_RELEARNING_RATING_PROB_FALLBACK,
    )
    first_rating_offsets = _coerce_list(
        source.get("first_rating_offsets", DEFAULT_FIRST_RATING_OFFSETS),
        expected_len=4,
        name="first_rating_offsets",
    )
    first_session_lens = _coerce_list(
        source.get("first_session_lens", DEFAULT_FIRST_SESSION_LENS),
        expected_len=4,
        name="first_session_lens",
    )
    forget_rating_offset = float(
        source.get("forget_rating_offset", DEFAULT_FORGET_RATING_OFFSET)
    )
    forget_session_len = float(
        source.get("forget_session_len", DEFAULT_FORGET_SESSION_LEN)
    )

    return {
        "learn_costs": learn_costs,
        "review_costs": review_costs,
        "first_rating_prob": first_rating_prob,
        "review_rating_prob": review_rating_prob,
        "learning_rating_prob": learning_rating_prob,
        "relearning_rating_prob": relearning_rating_prob,
        "state_rating_costs": state_rating_costs,
        "first_rating_offsets": first_rating_offsets,
        "first_session_lens": first_session_lens,
        "forget_rating_offset": forget_rating_offset,
        "forget_session_len": forget_session_len,
    }
