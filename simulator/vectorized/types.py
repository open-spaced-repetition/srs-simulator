from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch


@dataclass(frozen=True)
class VectorizedConfig:
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None


class VectorizedEnvOps(Protocol):
    device: torch.device
    dtype: torch.dtype

    def init_state(self, deck_size: int) -> Any: ...

    def retrievability(
        self, state, idx: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor: ...

    def update_review(
        self,
        state,
        idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        retrievability: torch.Tensor,
    ) -> None: ...

    def update_learn(self, state, idx: torch.Tensor, rating: torch.Tensor) -> None: ...


class VectorizedSchedulerOps(Protocol):
    device: torch.device
    dtype: torch.dtype

    def init_state(self, deck_size: int) -> Any: ...

    def review_priority(
        self, state, idx: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor: ...

    def update_review(
        self,
        state,
        idx: torch.Tensor,
        elapsed: torch.Tensor,
        rating: torch.Tensor,
        prev_interval: torch.Tensor,
    ) -> torch.Tensor: ...

    def update_learn(
        self, state, idx: torch.Tensor, rating: torch.Tensor
    ) -> torch.Tensor: ...
