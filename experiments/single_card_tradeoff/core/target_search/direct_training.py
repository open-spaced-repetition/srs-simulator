from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from experiments.single_card_tradeoff.core.target_search.types import ConstrainedTarget

DirectTargetKind = Literal["memory", "time"]


@dataclass(frozen=True, slots=True)
class DirectTargetJob:
    user_idx: int
    user_id: int
    target: ConstrainedTarget

    @property
    def target_type(self) -> DirectTargetKind:
        return self.target.target_type

    @property
    def target_value(self) -> float:
        return self.target.value


@dataclass(frozen=True, slots=True)
class DirectRankResult:
    score: torch.Tensor
    feasible: torch.Tensor
    rank_metric: torch.Tensor


def direct_target_jobs(
    *,
    user_ids: Sequence[int],
    target_memories: Sequence[float],
    target_times: Sequence[float],
) -> list[DirectTargetJob]:
    jobs: list[DirectTargetJob] = []
    for user_idx, user_id in enumerate(user_ids):
        jobs.extend(
            DirectTargetJob(
                user_idx=user_idx,
                user_id=user_id,
                target=ConstrainedTarget("memory", value, user_id=user_id),
            )
            for value in target_memories
        )
        jobs.extend(
            DirectTargetJob(
                user_idx=user_idx,
                user_id=user_id,
                target=ConstrainedTarget("time", value, user_id=user_id),
            )
            for value in target_times
        )
    return jobs


def constrained_rank_candidates(
    *,
    memory: torch.Tensor,
    minutes: torch.Tensor,
    jobs: Sequence[DirectTargetJob],
) -> DirectRankResult:
    if memory.shape != minutes.shape:
        raise ValueError("memory and minutes must have the same shape.")
    if memory.ndim != 2:
        raise ValueError("memory and minutes must have shape [job, candidate].")
    if memory.shape[0] != len(jobs):
        raise ValueError("first dimension must match jobs.")

    feasible = torch.zeros_like(memory, dtype=torch.bool)
    rank_metric = torch.empty_like(memory)
    for job_idx, job in enumerate(jobs):
        if job.target_type == "memory":
            feasible[job_idx] = memory[job_idx] >= job.target_value
            rank_metric[job_idx] = torch.where(
                feasible[job_idx],
                -minutes[job_idx],
                memory[job_idx] - job.target_value,
            )
        else:
            feasible[job_idx] = minutes[job_idx] <= job.target_value
            rank_metric[job_idx] = torch.where(
                feasible[job_idx],
                memory[job_idx],
                job.target_value - minutes[job_idx],
            )

    score = feasible.to(dtype=memory.dtype) * 1e9 + rank_metric
    return DirectRankResult(score=score, feasible=feasible, rank_metric=rank_metric)
