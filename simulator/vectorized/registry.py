from __future__ import annotations

from typing import Any

import torch

from simulator.models import FSRS6Model, LSTMModel
from simulator.models.fsrs import FSRS6VectorizedEnvOps
from simulator.models.lstm import LSTMVectorizedEnvOps
from simulator.schedulers import (
    AnkiSM2Scheduler,
    DASHScheduler,
    FixedIntervalScheduler,
    FSRS3Scheduler,
    FSRS6Scheduler,
    HLRScheduler,
    MemriseScheduler,
    SSPMMCScheduler,
)
from simulator.schedulers.anki_sm2 import AnkiSM2VectorizedSchedulerOps
from simulator.schedulers.fixed import FixedVectorizedSchedulerOps
from simulator.schedulers.fsrs import (
    FSRS3VectorizedSchedulerOps,
    FSRS6VectorizedSchedulerOps,
)
from simulator.schedulers.hlr import HLRVectorizedSchedulerOps
from simulator.schedulers.memrise import MemriseVectorizedSchedulerOps
from simulator.schedulers.sspmmc import SSPMMCVectorizedSchedulerOps
from simulator.vectorized.types import (
    VectorizedConfig,
    VectorizedEnvOps,
    VectorizedSchedulerOps,
)


def resolve_env_ops(environment: Any, config: VectorizedConfig) -> VectorizedEnvOps:
    if isinstance(environment, LSTMModel):
        device = torch.device(config.device) if config.device is not None else None
        return LSTMVectorizedEnvOps(
            environment, device=device, lstm_batch_size=config.lstm_batch_size
        )
    if isinstance(environment, FSRS6Model):
        device = (
            torch.device(config.device)
            if config.device is not None
            else torch.device("cpu")
        )
        dtype = config.dtype or torch.float64
        return FSRS6VectorizedEnvOps(environment, device=device, dtype=dtype)
    raise ValueError(
        "Vectorized engine requires FSRS6Model or LSTMModel as the environment."
    )


def resolve_scheduler_ops(
    scheduler: Any,
    config: VectorizedConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> VectorizedSchedulerOps:
    if isinstance(scheduler, FSRS6Scheduler):
        return FSRS6VectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FSRS3Scheduler):
        return FSRS3VectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, HLRScheduler):
        return HLRVectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, FixedIntervalScheduler):
        return FixedVectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, MemriseScheduler):
        return MemriseVectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, AnkiSM2Scheduler):
        return AnkiSM2VectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, SSPMMCScheduler):
        return SSPMMCVectorizedSchedulerOps(scheduler, device=device, dtype=dtype)
    if isinstance(scheduler, DASHScheduler):
        raise ValueError(
            "Vectorized engine does not support DASHScheduler; "
            "use the event-driven engine instead."
        )
    raise ValueError(
        "Vectorized engine requires a supported scheduler "
        "(FSRS6, FSRS3, HLR, fixed, Memrise, Anki SM-2, or SSPMMC)."
    )
