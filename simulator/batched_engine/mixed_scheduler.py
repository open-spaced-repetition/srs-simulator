from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from simulator.fuzz import resolve_max_interval


@dataclass(frozen=True, slots=True)
class MixedSchedulerGroup:
    lane_indices: Any
    ops: Any


@dataclass(slots=True)
class MixedBatchSchedulerState:
    states: tuple[Any, ...]


class MixedBatchSchedulerOps:
    def __init__(
        self,
        *,
        groups: Sequence[MixedSchedulerGroup],
        lane_count: int,
        device: Any,
        dtype: Any,
    ) -> None:
        if not groups:
            raise ValueError("Mixed scheduler ops require at least one group.")
        self._groups = tuple(groups)
        self.device = device
        self.dtype = dtype
        self._lane_to_local = torch.empty(
            lane_count,
            device=device,
            dtype=torch.int64,
        )
        self._lane_to_group = torch.empty(
            lane_count,
            device=device,
            dtype=torch.int64,
        )
        for group_index, group in enumerate(self._groups):
            local = torch.arange(
                int(group.lane_indices.numel()),
                device=device,
                dtype=torch.int64,
            )
            self._lane_to_local[group.lane_indices] = local
            self._lane_to_group[group.lane_indices] = group_index
        self.max_interval = max(
            resolve_max_interval(group.ops) for group in self._groups
        )
        interval_modes = {
            getattr(group.ops, "interval_mode", None) for group in self._groups
        }
        self.interval_mode = interval_modes.pop() if len(interval_modes) == 1 else None

    def init_state(self, user_count: int, deck_size: int) -> MixedBatchSchedulerState:
        states = tuple(
            group.ops.init_state(int(group.lane_indices.numel()), deck_size)
            for group in self._groups
        )
        return MixedBatchSchedulerState(states=states)

    def set_time_context(self, *, current_day: float, simulation_days: float) -> None:
        for group in self._groups:
            setter = getattr(group.ops, "set_time_context", None)
            if setter is not None:
                setter(
                    current_day=current_day,
                    simulation_days=simulation_days,
                )

    def review_priority(
        self,
        state: MixedBatchSchedulerState,
        elapsed: Any,
    ) -> Any:
        result = torch.empty(elapsed.shape, device=self.device, dtype=self.dtype)
        for group_index, group in enumerate(self._groups):
            group_elapsed = elapsed.index_select(0, group.lane_indices)
            result[group.lane_indices] = group.ops.review_priority(
                state.states[group_index],
                group_elapsed,
            )
        return result

    def update_review(
        self,
        state: MixedBatchSchedulerState,
        user_idx: Any,
        card_idx: Any,
        elapsed: Any,
        rating: Any,
        prev_interval: Any,
    ) -> Any:
        return self._dispatch_update(
            "update_review",
            state,
            user_idx,
            card_idx,
            elapsed,
            rating,
            prev_interval,
        )

    def update_learn(
        self,
        state: MixedBatchSchedulerState,
        user_idx: Any,
        card_idx: Any,
        rating: Any,
    ) -> Any:
        return self._dispatch_update(
            "update_learn",
            state,
            user_idx,
            card_idx,
            rating,
        )

    def _dispatch_update(
        self,
        method_name: str,
        state: MixedBatchSchedulerState,
        user_idx: Any,
        card_idx: Any,
        *args: Any,
    ) -> Any:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        result = torch.empty(
            (int(user_idx.numel()),),
            device=self.device,
            dtype=self.dtype,
        )
        selected_groups = torch.unique(
            self._lane_to_group.index_select(0, user_idx)
        ).tolist()
        for group_index in selected_groups:
            group_mask = self._lane_to_group.index_select(0, user_idx) == int(
                group_index
            )
            if not bool(group_mask.any().item()):
                continue
            local_user_idx = self._lane_to_local.index_select(
                0,
                user_idx[group_mask],
            )
            group_args = [
                arg[group_mask] if hasattr(arg, "__getitem__") else arg for arg in args
            ]
            update = getattr(self._groups[int(group_index)].ops, method_name)
            result[group_mask] = update(
                state.states[int(group_index)],
                local_user_idx,
                card_idx[group_mask],
                *group_args,
            )
        return result
