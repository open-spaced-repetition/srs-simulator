from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SSPMMCPPolicy:
    retention_matrix: np.ndarray
    cost_matrix: np.ndarray
    s_grid: np.ndarray
    d_grid: np.ndarray
    r_grid: np.ndarray
    metadata: dict[str, Any]
    state_space: dict[str, float]
    s_mid: float
    s_state_small_len: int

    @classmethod
    def from_json(cls, metadata_path: str | Path) -> "SSPMMCPPolicy":
        meta_path = Path(metadata_path)
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        state_space = meta.get("state_space")
        if not isinstance(state_space, dict):
            raise ValueError(f"SSP-MMC metadata {meta_path} missing 'state_space'.")
        npz_path = meta.get("policy_file")
        if npz_path is None:
            raise ValueError(f"SSP-MMC metadata {meta_path} missing 'policy_file'.")
        npz_path = Path(npz_path)
        if not npz_path.is_absolute():
            npz_path = meta_path.parent / npz_path
        arrays = np.load(npz_path, allow_pickle=False)
        required = {"retention_matrix", "cost_matrix", "s_state", "d_state", "r_state"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"Policy file {npz_path} missing arrays: {missing}")
        s_mid = min(
            float(state_space["long_step"])
            / (1.0 - math.exp(-float(state_space["short_step"]))),
            float(state_space["s_max"]),
        )
        s_state_small = np.exp(
            np.arange(
                math.log(float(state_space["s_min"])),
                math.log(s_mid),
                float(state_space["short_step"]),
            )
        )
        return cls(
            retention_matrix=arrays["retention_matrix"],
            cost_matrix=arrays["cost_matrix"],
            s_grid=arrays["s_state"],
            d_grid=arrays["d_state"],
            r_grid=arrays["r_state"],
            metadata=meta,
            state_space={k: float(v) for k, v in state_space.items()},
            s_mid=s_mid,
            s_state_small_len=len(s_state_small),
        )

    def lookup(self, stability: float, difficulty: float) -> tuple[float, bool]:
        s_idx = self.s2i(stability)
        d_idx = self.d2i(difficulty)
        retention = float(self.retention_matrix[d_idx, s_idx])
        graduated = (
            s_idx >= self.s_grid.size - 1 or stability >= self.state_space["s_max"]
        )
        return retention, graduated

    def s2i(self, s: float) -> int:
        s_min = self.state_space["s_min"]
        short_step = self.state_space["short_step"]
        long_step = self.state_space["long_step"]
        if s <= self.s_mid:
            idx = math.ceil((math.log(max(s, s_min)) - math.log(s_min)) / short_step)
            return int(_clamp(idx, 0, self.s_state_small_len - 1))
        if self.s_state_small_len <= 0:
            return int(_clamp(0, 0, self.s_grid.size - 1))
        offset = math.ceil(
            (s - self.s_grid[self.s_state_small_len - 1] - long_step) / long_step
        )
        large_len = self.s_grid.size - self.s_state_small_len
        if large_len <= 0:
            return int(_clamp(self.s_state_small_len - 1, 0, self.s_grid.size - 1))
        offset = _clamp(offset, 0, large_len - 1)
        return int(self.s_state_small_len + offset)

    def d2i(self, d: float) -> int:
        d_min = self.state_space["d_min"]
        d_max = self.state_space["d_max"]
        d_size = int(math.ceil((d_max - d_min) / self.state_space["d_eps"] + 1))
        d_size = min(d_size, self.d_grid.size)
        idx = math.floor((d - d_min) / (d_max - d_min) * d_size)
        return int(_clamp(idx, 0, d_size - 1))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["SSPMMCPPolicy"]
