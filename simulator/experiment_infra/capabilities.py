from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from simulator.scheduler_spec import scheduler_uses_desired_retention


class EngineName(StrEnum):
    EVENT = "event"
    BATCHED = "batched"


EVENT_ENVS = ("lstm", "fsrs6", "fsrs6_default", "fsrs3", "fsrs3_default")
EVENT_SCHEDULERS = (
    "fsrs6",
    "fsrs6_default",
    "fsrs3",
    "fsrs3_default",
    "hlr",
    "dash",
    "lstm",
    "fixed",
    "anki_sm2",
    "memrise",
    "sspmmc",
    "fsrs6_adr",
    "fsrs6_adp",
)
BATCHED_ENVS = ("lstm", "fsrs6", "fsrs6_default")
BATCHED_SCHEDULERS = (
    "fsrs6",
    "fsrs6_default",
    "fsrs3",
    "fsrs3_default",
    "lstm",
    "fixed",
    "anki_sm2",
    "memrise",
    "fsrs6_adr",
    "fsrs6_adp",
)


@dataclass(frozen=True, slots=True)
class SchedulerCapability:
    scheduler: str
    envs_by_engine: dict[EngineName, tuple[str, ...]]
    supports_desired_retention: bool

    def supports(self, *, engine: str | EngineName, environment: str) -> bool:
        engine_name = EngineName(engine)
        return environment in self.envs_by_engine.get(engine_name, ())

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduler": self.scheduler,
            "supports_desired_retention": self.supports_desired_retention,
            "envs_by_engine": {
                engine.value: list(envs) for engine, envs in self.envs_by_engine.items()
            },
        }


def build_scheduler_capabilities() -> dict[str, SchedulerCapability]:
    schedulers = sorted(set(EVENT_SCHEDULERS) | set(BATCHED_SCHEDULERS))
    capabilities: dict[str, SchedulerCapability] = {}
    for scheduler in schedulers:
        envs_by_engine: dict[EngineName, tuple[str, ...]] = {}
        if scheduler in EVENT_SCHEDULERS:
            envs_by_engine[EngineName.EVENT] = EVENT_ENVS
        if scheduler in BATCHED_SCHEDULERS:
            envs_by_engine[EngineName.BATCHED] = BATCHED_ENVS
        capabilities[scheduler] = SchedulerCapability(
            scheduler=scheduler,
            envs_by_engine=envs_by_engine,
            supports_desired_retention=scheduler_uses_desired_retention(scheduler),
        )
    return capabilities


SCHEDULER_CAPABILITIES = build_scheduler_capabilities()


def get_scheduler_capability(scheduler: str) -> SchedulerCapability:
    try:
        return SCHEDULER_CAPABILITIES[scheduler]
    except KeyError as exc:
        raise ValueError(f"Unknown scheduler capability '{scheduler}'.") from exc


def supports_scheduler(*, scheduler: str, engine: str, environment: str) -> bool:
    return get_scheduler_capability(scheduler).supports(
        engine=engine,
        environment=environment,
    )
