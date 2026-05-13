from __future__ import annotations

from pathlib import Path

from simulator.anki_sm2_ap_policy import AnkiSM2APPolicy
from simulator.schedulers.anki_sm2 import AnkiSM2Scheduler


class AnkiSM2APScheduler(AnkiSM2Scheduler):
    """Anki SM2 scheduler parameterized by an exported AP policy."""

    def __init__(self, *, policy_json: str | Path) -> None:
        self.policy = AnkiSM2APPolicy.from_json(policy_json)
        super().__init__(**self.policy.params_dict())


__all__ = ["AnkiSM2APScheduler"]
