from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(slots=True)
class LogFilenameFilter:
    envs: Sequence[str]
    scheds: Sequence[str]
    engine: str = "any"
    short_term: str = "any"
    short_term_source: str = "any"
    start_retention: float | None = None
    end_retention: float | None = None
    priority: str = "any"
    retention_values_by_scheduler: Mapping[str, float] | None = None

    def matches(self, name: str) -> bool:
        if self.envs and not any(f"env={env}" in name for env in self.envs):
            return False
        if self.scheds and not any(f"sched={sched}" in name for sched in self.scheds):
            return False
        if self.engine != "any" and f"engine={self.engine}" not in name:
            return False
        if self.priority != "any" and f"prio={self.priority}" not in name:
            return False

        if self.short_term == "on":
            if "_st=" not in name:
                return False
            if (
                self.short_term_source != "any"
                and f"st={self.short_term_source}" not in name
            ):
                return False
        elif self.short_term == "off":
            if self.short_term_source != "any":
                return False
            if "_st=" in name and "st=off" not in name:
                return False
        elif (
            self.short_term_source != "any"
            and f"st={self.short_term_source}" not in name
        ):
            return False

        if "ret=" not in name:
            return True

        match = re.search(r"ret=([0-9.]+)", name)
        if match is None:
            return True
        try:
            retention_value = float(match.group(1))
        except ValueError:
            return False

        if self.start_retention is not None and retention_value < self.start_retention:
            return False
        if self.end_retention is not None and retention_value > self.end_retention:
            return False

        if self.retention_values_by_scheduler is None:
            return True

        rounded_value = round(retention_value, 2)
        for scheduler, desired_retention in self.retention_values_by_scheduler.items():
            if f"sched={scheduler}" not in name:
                continue
            if rounded_value != desired_retention:
                return False
        return True


__all__ = ["LogFilenameFilter"]
