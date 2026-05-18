from __future__ import annotations

from typing import Any

import torch

from simulator.models import FSRS6Model, LSTMModel
from simulator.models.fsrs import FSRS6BatchedEnvOps
from simulator.models.lstm import LSTMBatchedEnvOps
from simulator.schedulers import (
    AnkiSM2Scheduler,
    DASHScheduler,
    LSTMScheduler,
    FixedIntervalScheduler,
    FSRS3Scheduler,
    FSRS6Scheduler,
    HLRScheduler,
    MemriseScheduler,
    FSRS6ADRScheduler,
    FSRS6OracleStationaryFiniteDistillScheduler,
    SSPMMCScheduler,
)
from simulator.schedulers.anki_sm2 import AnkiSM2BatchedSchedulerOps
from simulator.schedulers.fixed import FixedBatchedSchedulerOps
from simulator.schedulers.fsrs import (
    FSRS3BatchedSchedulerOps,
    FSRS6BatchedSchedulerOps,
)
from simulator.schedulers.hlr import HLRBatchedSchedulerOps
from simulator.schedulers.lstm import LSTMBatchedSchedulerOps
from simulator.schedulers.memrise import MemriseBatchedSchedulerOps
from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchedSchedulerOps
from simulator.schedulers.fsrs6_oracle_stationary_finite_distill import (
    FSRS6OracleStationaryFiniteDistillBatchedSchedulerOps,
)
from simulator.schedulers.sspmmc import SSPMMCBatchedSchedulerOps
from simulator.batched_engine.types import (
    BatchedEngineConfig,
    BatchedEnvOps,
    BatchedSchedulerOps,
)


def resolve_env_ops(environment: Any, config: BatchedEngineConfig) -> BatchedEnvOps:
    if isinstance(environment, LSTMModel):
        device = torch.device(config.device) if config.device is not None else None
        return LSTMBatchedEnvOps(environment, device=device)
    if isinstance(environment, FSRS6Model):
        device = (
            torch.device(config.device)
            if config.device is not None
            else torch.device("cpu")
        )
        dtype = config.dtype or torch.float64
        return FSRS6BatchedEnvOps(environment, device=device, dtype=dtype)
    raise ValueError(
        "Batched engine requires FSRS6Model or LSTMModel as the environment."
    )


def resolve_scheduler_ops(
    scheduler: Any,
    config: BatchedEngineConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> BatchedSchedulerOps:
    if isinstance(scheduler, FSRS6Scheduler):
        return FSRS6BatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FSRS3Scheduler):
        return FSRS3BatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, HLRScheduler):
        return HLRBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FixedIntervalScheduler):
        return FixedBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, MemriseScheduler):
        return MemriseBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, AnkiSM2Scheduler):
        return AnkiSM2BatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, SSPMMCScheduler):
        return SSPMMCBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FSRS6ADRScheduler):
        return FSRS6ADRBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FSRS6OracleStationaryFiniteDistillScheduler):
        return FSRS6OracleStationaryFiniteDistillBatchedSchedulerOps(
            scheduler, device=device, dtype=dtype
        )
    if isinstance(scheduler, DASHScheduler):
        raise ValueError(
            "Batched engine does not support DASHScheduler; "
            "use the event-driven engine instead."
        )
    if isinstance(scheduler, LSTMScheduler):
        return LSTMBatchedSchedulerOps(scheduler, device=device, dtype=dtype)
    raise ValueError(
        "Batched engine requires a supported scheduler "
        "(FSRS6, FSRS3, HLR, fixed, Memrise, Anki SM-2, SSPMMC, "
        "FSRS6 ADR, or FSRS6 oracle stationary finite distill)."
    )
