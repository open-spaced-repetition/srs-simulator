from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from simulator.fsrs6_adr_policy import (
    FEATURE_VERSION_LOG_LINEAR,
    FEATURE_VERSION_LOG_POLY,
    FEATURE_VERSION_LOG_POLY_TIME,
)
from simulator.fsrs6_oracle_stationary_finite_distill_policy import (
    PORTFOLIO_CHILD_ACTION_SPACE as ORACLE_DISTILL_PORTFOLIO_CHILD_ACTION_SPACE,
)


class PolicySource(StrEnum):
    FSRS6_ADR = "fsrs6_adr"
    FSRS6_ORACLE_STATIONARY_FINITE_DISTILL = "fsrs6_oracle_stationary_finite_distill"
    FSRS6_AP = "fsrs6_ap"
    ANKI_SM2_AP = "anki_sm2_ap"
    SSPMMC = "sspmmc"


@dataclass(frozen=True, slots=True)
class SchedulerDescriptor:
    name: str
    supports_event: bool
    supports_batched: bool
    uses_desired_retention: bool
    policy_source: PolicySource | None
    run_id_scoped_sweep: bool


@dataclass(frozen=True, slots=True)
class FSRS6ADRVariant:
    scheduler_name: str
    action_space: str
    portfolio_child_action_space: str


SCHEDULER_DESCRIPTORS: dict[str, SchedulerDescriptor] = {
    # Desired-retention schedulers.
    "fsrs6": SchedulerDescriptor(
        name="fsrs6",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "fsrs6_default": SchedulerDescriptor(
        name="fsrs6_default",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "fsrs3": SchedulerDescriptor(
        name="fsrs3",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "fsrs3_default": SchedulerDescriptor(
        name="fsrs3_default",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "hlr": SchedulerDescriptor(
        name="hlr",
        supports_event=True,
        supports_batched=False,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "dash": SchedulerDescriptor(
        name="dash",
        supports_event=True,
        supports_batched=False,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "lstm": SchedulerDescriptor(
        name="lstm",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=True,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    # Non desired-retention schedulers.
    "fixed": SchedulerDescriptor(
        name="fixed",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "anki_sm2": SchedulerDescriptor(
        name="anki_sm2",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    "memrise": SchedulerDescriptor(
        name="memrise",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=None,
        run_id_scoped_sweep=False,
    ),
    # Policy schedulers.
    "sspmmc": SchedulerDescriptor(
        name="sspmmc",
        supports_event=True,
        supports_batched=False,
        uses_desired_retention=True,
        policy_source=PolicySource.SSPMMC,
        run_id_scoped_sweep=False,
    ),
    "anki_sm2_ap": SchedulerDescriptor(
        name="anki_sm2_ap",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.ANKI_SM2_AP,
        run_id_scoped_sweep=True,
    ),
    "fsrs6_ap": SchedulerDescriptor(
        name="fsrs6_ap",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.FSRS6_AP,
        run_id_scoped_sweep=True,
    ),
    "fsrs6_adr": SchedulerDescriptor(
        name="fsrs6_adr",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.FSRS6_ADR,
        run_id_scoped_sweep=True,
    ),
    "fsrs6_adr_time": SchedulerDescriptor(
        name="fsrs6_adr_time",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.FSRS6_ADR,
        run_id_scoped_sweep=True,
    ),
    "fsrs6_default_adr": SchedulerDescriptor(
        name="fsrs6_default_adr",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.FSRS6_ADR,
        run_id_scoped_sweep=True,
    ),
    "fsrs6_oracle_stationary_finite_distill": SchedulerDescriptor(
        name="fsrs6_oracle_stationary_finite_distill",
        supports_event=True,
        supports_batched=True,
        uses_desired_retention=False,
        policy_source=PolicySource.FSRS6_ORACLE_STATIONARY_FINITE_DISTILL,
        run_id_scoped_sweep=True,
    ),
}


PORTFOLIO_CHILD_ACTION_SPACES = frozenset(
    {
        "sd_retention_function_portfolio_child",
        "sdt_retention_function_portfolio_child",
        ORACLE_DISTILL_PORTFOLIO_CHILD_ACTION_SPACE,
        "fsrs6_ap_weight_delta_portfolio_child",
        "anki_sm2_ap_params_portfolio_child",
    }
)


_FSRS6_ADR_VARIANTS: dict[str, FSRS6ADRVariant] = {
    FEATURE_VERSION_LOG_POLY: FSRS6ADRVariant(
        scheduler_name="fsrs6_adr",
        action_space="sd_retention_function",
        portfolio_child_action_space="sd_retention_function_portfolio_child",
    ),
    FEATURE_VERSION_LOG_LINEAR: FSRS6ADRVariant(
        scheduler_name="fsrs6_adr",
        action_space="sd_retention_function",
        portfolio_child_action_space="sd_retention_function_portfolio_child",
    ),
    FEATURE_VERSION_LOG_POLY_TIME: FSRS6ADRVariant(
        scheduler_name="fsrs6_adr_time",
        action_space="sdt_retention_function",
        portfolio_child_action_space="sdt_retention_function_portfolio_child",
    ),
}


def get_scheduler_descriptor(name: str) -> SchedulerDescriptor:
    try:
        return SCHEDULER_DESCRIPTORS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown scheduler '{name}'.") from exc


def try_get_scheduler_descriptor(name: str) -> SchedulerDescriptor | None:
    return SCHEDULER_DESCRIPTORS.get(name)


def event_scheduler_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            desc.name for desc in SCHEDULER_DESCRIPTORS.values() if desc.supports_event
        )
    )


def batched_scheduler_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            desc.name
            for desc in SCHEDULER_DESCRIPTORS.values()
            if desc.supports_batched
        )
    )


def schedulers_for_policy_source(source: PolicySource) -> frozenset[str]:
    return frozenset(
        desc.name
        for desc in SCHEDULER_DESCRIPTORS.values()
        if desc.policy_source == source
    )


def run_id_scoped_sweep_schedulers() -> frozenset[str]:
    return frozenset(
        desc.name for desc in SCHEDULER_DESCRIPTORS.values() if desc.run_id_scoped_sweep
    )


def is_portfolio_child_action_space(action_space: str) -> bool:
    return action_space in PORTFOLIO_CHILD_ACTION_SPACES


def fsrs6_adr_variant_for_feature_version(feature_version: str) -> FSRS6ADRVariant:
    try:
        return _FSRS6_ADR_VARIANTS[feature_version]
    except KeyError as exc:
        supported = ", ".join(sorted(_FSRS6_ADR_VARIANTS))
        raise ValueError(
            f"Unsupported FSRS6 ADR feature_version {feature_version!r}; "
            f"expected one of: {supported}."
        ) from exc
