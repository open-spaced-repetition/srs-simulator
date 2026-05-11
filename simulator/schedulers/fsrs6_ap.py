from __future__ import annotations

from pathlib import Path

from simulator.fsrs6_ap_policy import FSRS6APPolicy
from simulator.schedulers.fsrs import FSRS6Scheduler


class FSRS6APScheduler(FSRS6Scheduler):
    """
    FSRS6 scheduler using adaptive-parameter weights from an FSRS6 AP policy.
    """

    def __init__(
        self,
        policy_json: str | Path,
        *,
        priority_mode: str = "low_retrievability",
    ) -> None:
        self.policy = FSRS6APPolicy.from_json(policy_json)
        super().__init__(
            weights=self.policy.weights,
            desired_retention=self.policy.baseline_desired_retention,
            priority_mode=priority_mode,
        )


__all__ = ["FSRS6APScheduler"]
