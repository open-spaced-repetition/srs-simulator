from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProgressEvent:
    label: str
    completed: int
    total: int


def encode_progress_event(*, label: str, completed: int, total: int) -> str:
    # Keep schema stable for parent consumers (run_sweep_users.py).
    payload = {
        "type": "progress",
        "completed": int(completed),
        "total": int(total),
        "label": str(label),
    }
    return json.dumps(payload)


def try_parse_json(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def progress_event_from_payload(payload: Any) -> ProgressEvent | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "progress":
        return None
    label = payload.get("label", "")
    completed = payload.get("completed", 0)
    total = payload.get("total", 0)
    try:
        return ProgressEvent(
            label=str(label),
            completed=int(completed),
            total=int(total),
        )
    except (TypeError, ValueError):
        return None
