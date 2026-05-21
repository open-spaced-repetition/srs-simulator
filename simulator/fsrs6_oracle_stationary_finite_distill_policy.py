from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from simulator.math.fsrs import Bounds


POLICY_TYPE = "fsrs6_oracle_stationary_finite_distill"
FEATURE_VERSION = "fsrs6_oracle_stationary_finite_distill_v1"
ACTION_SPACE = "goal_cost_weight"
PORTFOLIO_CHILD_ACTION_SPACE = (
    "fsrs6_oracle_stationary_finite_distill_goal_weight_portfolio_child"
)
DEFAULT_TITLE = "FSRS6 oracle stationary finite distill"


@dataclass(frozen=True, slots=True)
class FSRS6OracleStationaryFiniteDistillPolicy:
    checkpoint_path: Path
    goal_cost_weight: float
    goal_norm_max: float
    action_retentions: tuple[float, ...]
    obs_mode: str = "oracle_stationary"
    user_id: int | None = None
    portfolio_index: int | None = None
    title: str = DEFAULT_TITLE
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> FSRS6OracleStationaryFiniteDistillPolicy:
        policy_path = Path(path)
        with policy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Distill policy {policy_path} must be a JSON object.")
        policy_type = raw.get("policy_type", raw.get("policy_kind"))
        if policy_type not in {POLICY_TYPE, "fsrs6-oracle-stationary-finite-distill"}:
            raise ValueError(
                f"Distill policy {policy_path} has unsupported policy_type "
                f"{policy_type!r}."
            )
        checkpoint_path = _resolve_path(
            _require_str(raw.get("checkpoint_path"), "checkpoint_path"),
            base_path=policy_path.parent,
        )
        checkpoint = load_checkpoint_metadata(checkpoint_path)
        action_retentions = _optional_float_tuple(
            raw.get("action_retentions"), "action_retentions"
        )
        checkpoint_actions = tuple(float(value) for value in checkpoint["actions"])
        if action_retentions is None:
            action_retentions = checkpoint_actions
        elif action_retentions != checkpoint_actions:
            raise ValueError(
                f"Distill policy {policy_path} action_retentions do not match "
                f"checkpoint {checkpoint_path}."
            )
        obs_mode = _require_str(raw.get("obs_mode", checkpoint["obs_mode"]), "obs_mode")
        if obs_mode != "oracle_stationary":
            raise ValueError("Only obs_mode='oracle_stationary' is supported.")
        cost_weights = tuple(float(value) for value in checkpoint["cost_weights"])
        default_goal_norm_max = max(1.0, max(cost_weights))
        goal_norm_max = _float(
            raw.get("goal_norm_max", default_goal_norm_max), "goal_norm_max"
        )
        return cls(
            checkpoint_path=checkpoint_path,
            goal_cost_weight=_float(raw.get("goal_cost_weight"), "goal_cost_weight"),
            goal_norm_max=goal_norm_max,
            action_retentions=action_retentions,
            obs_mode=obs_mode,
            user_id=_optional_int(raw.get("user_id"), "user_id"),
            portfolio_index=_optional_int(
                raw.get("portfolio_index"), "portfolio_index"
            ),
            title=_require_str(raw.get("title", DEFAULT_TITLE), "title"),
            metadata=raw,
        )

    def __post_init__(self) -> None:
        if self.goal_cost_weight < 0.0 or not math.isfinite(self.goal_cost_weight):
            raise ValueError("goal_cost_weight must be finite and >= 0.")
        if self.goal_norm_max < 1.0 or not math.isfinite(self.goal_norm_max):
            raise ValueError("goal_norm_max must be finite and >= 1.")
        if self.obs_mode != "oracle_stationary":
            raise ValueError("Only obs_mode='oracle_stationary' is supported.")
        _validate_action_retentions(self.action_retentions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": POLICY_TYPE,
            "feature_version": FEATURE_VERSION,
            "title": self.title,
            "checkpoint_path": str(self.checkpoint_path),
            "goal_cost_weight": self.goal_cost_weight,
            "goal_norm_max": self.goal_norm_max,
            "action_retentions": list(self.action_retentions),
            "obs_mode": self.obs_mode,
            "user_id": self.user_id,
            "portfolio_index": self.portfolio_index,
        }

    def write_json(self, path: str | Path) -> None:
        policy_path = Path(path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate(self, stability: float, difficulty: float) -> float:
        import torch

        model, action_retentions = _cached_cpu_checkpoint_model(
            str(self.checkpoint_path.resolve())
        )
        obs = torch.tensor(
            [
                _normalized_stability(stability, Bounds()),
                _normalized_difficulty(difficulty, Bounds()),
                _normalized_goal(self.goal_cost_weight, self.goal_norm_max),
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = model(obs)
            action = int(torch.argmax(logits, dim=1).item())
        return float(action_retentions[action])


class FSRS6OracleStationaryFiniteDistillBatchPolicy:
    def __init__(
        self,
        policies: Sequence[FSRS6OracleStationaryFiniteDistillPolicy],
        *,
        device: Any,
        dtype: Any,
        bounds: Bounds = Bounds(),
    ) -> None:
        import torch

        if not policies:
            raise ValueError("At least one distill policy is required.")
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self.bounds = bounds
        action_count = len(policies[0].action_retentions)
        for policy in policies:
            if len(policy.action_retentions) != action_count:
                raise ValueError("All distill policies must share action count.")
            _validate_action_retentions(policy.action_retentions)
        self._action_retentions = torch.tensor(
            [policy.action_retentions for policy in policies],
            device=device,
            dtype=dtype,
        )
        self._goal_cost_weight = torch.tensor(
            [policy.goal_cost_weight for policy in policies],
            device=device,
            dtype=dtype,
        )
        self._goal_norm_max = torch.tensor(
            [policy.goal_norm_max for policy in policies],
            device=device,
            dtype=dtype,
        )
        self._log_s_min = float(torch.log(torch.tensor(bounds.s_min)).item())
        self._log_s_span = float(
            torch.log(torch.tensor(bounds.s_max / bounds.s_min)).item()
        )
        self._d_span = bounds.d_max - bounds.d_min
        self._groups: list[tuple[torch.Tensor, Any]] = []
        grouped: dict[Path, list[int]] = {}
        for lane_index, policy in enumerate(policies):
            grouped.setdefault(policy.checkpoint_path.resolve(), []).append(lane_index)
        for checkpoint_path, lane_indices in grouped.items():
            model, checkpoint_actions = load_checkpoint_model(
                checkpoint_path,
                device=device,
            )
            if len(checkpoint_actions) != action_count:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} action count does not match."
                )
            model.eval()
            lane_tensor = torch.tensor(lane_indices, device=device, dtype=torch.int64)
            self._groups.append((lane_tensor, model))

    def evaluate(
        self,
        stability: Any,
        difficulty: Any,
        *,
        lane_idx: Any | None = None,
    ) -> Any:
        torch = self._torch
        with torch.inference_mode():
            s = stability.to(device=self.device, dtype=self.dtype)
            d = difficulty.to(device=self.device, dtype=self.dtype)
            if lane_idx is None:
                lanes = torch.arange(s.numel(), device=self.device, dtype=torch.int64)
            else:
                lanes = lane_idx.to(device=self.device, dtype=torch.int64)
            if s.numel() != lanes.numel():
                raise ValueError("stability and lane_idx must have the same length.")
            obs = self._obs(s, d, lanes)
            out = torch.empty(s.shape, device=self.device, dtype=self.dtype)
            for group_lanes, model in self._groups:
                mask = _membership_mask(torch, lanes, group_lanes)
                if not bool(mask.any().item()):
                    continue
                positions = mask.nonzero(as_tuple=False).squeeze(1)
                logits, _ = model(
                    obs.index_select(0, positions).to(dtype=torch.float32)
                )
                actions = torch.argmax(logits, dim=1).to(torch.int64)
                lane_actions = self._action_retentions.index_select(
                    0, lanes.index_select(0, positions)
                )
                retention = lane_actions.gather(1, actions[:, None]).squeeze(1)
                out[positions] = retention.to(dtype=self.dtype)
            return out

    def _obs(self, s: Any, d: Any, lanes: Any) -> Any:
        torch = self._torch
        s_norm = (
            torch.log(torch.clamp(s, self.bounds.s_min, self.bounds.s_max))
            - self._log_s_min
        ) / self._log_s_span
        d_norm = (
            torch.clamp(d, self.bounds.d_min, self.bounds.d_max) - self.bounds.d_min
        ) / self._d_span
        goals = self._goal_cost_weight.index_select(0, lanes)
        goal_max = self._goal_norm_max.index_select(0, lanes)
        goal_norm = torch.log1p(goals) / torch.log1p(torch.clamp(goal_max, min=1.0))
        return torch.stack(
            [
                torch.clamp(s_norm, 0.0, 1.0),
                torch.clamp(d_norm, 0.0, 1.0),
                torch.clamp(goal_norm, 0.0, 1.0),
            ],
            dim=1,
        )


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Distill checkpoint must be a dict: {checkpoint_path}")
    if checkpoint.get("policy_type") != POLICY_TYPE:
        raise ValueError(f"Unexpected policy_type in {checkpoint_path}.")
    if checkpoint.get("obs_mode") != "oracle_stationary":
        raise ValueError(f"Unexpected obs_mode in {checkpoint_path}.")
    if int(checkpoint.get("obs_dim", 0)) != 3:
        raise ValueError(f"Distill checkpoint {checkpoint_path} must have obs_dim=3.")
    actions = _float_tuple(checkpoint.get("action_retentions"), "action_retentions")
    cost_weights = _float_tuple(checkpoint.get("cost_weights"), "cost_weights")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Distill checkpoint {checkpoint_path} has no model_state_dict."
        )
    return {
        "obs_dim": int(checkpoint["obs_dim"]),
        "obs_mode": str(checkpoint["obs_mode"]),
        "hidden_size": int(checkpoint["hidden_size"]),
        "network": str(checkpoint["network"]),
        "network_depth": int(checkpoint["network_depth"]),
        "actions": actions,
        "cost_weights": cost_weights,
    }


def load_checkpoint_model(
    path: str | Path, *, device: Any
) -> tuple[Any, tuple[float, ...]]:
    import torch
    from experiments.single_card_tradeoff.models.policy_net import PolicyValueNet

    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = load_checkpoint_metadata(checkpoint_path)
    model = PolicyValueNet(
        metadata["obs_dim"],
        len(metadata["actions"]),
        metadata["hidden_size"],
        architecture=metadata["network"],
        depth=metadata["network_depth"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device)
    model.eval()
    return model, tuple(float(value) for value in metadata["actions"])


@lru_cache(maxsize=32)
def _cached_cpu_checkpoint_model(path: str) -> tuple[Any, tuple[float, ...]]:
    import torch

    return load_checkpoint_model(Path(path), device=torch.device("cpu"))


def _membership_mask(torch: Any, lanes: Any, group_lanes: Any) -> Any:
    mask = torch.zeros(lanes.shape, device=lanes.device, dtype=torch.bool)
    for lane in group_lanes.tolist():
        mask |= lanes == int(lane)
    return mask


def _normalized_stability(value: float, bounds: Bounds) -> float:
    s = min(bounds.s_max, max(bounds.s_min, float(value)))
    return min(
        1.0,
        max(
            0.0,
            (math.log(s) - math.log(bounds.s_min))
            / (math.log(bounds.s_max) - math.log(bounds.s_min)),
        ),
    )


def _normalized_difficulty(value: float, bounds: Bounds) -> float:
    d = min(bounds.d_max, max(bounds.d_min, float(value)))
    return min(1.0, max(0.0, (d - bounds.d_min) / (bounds.d_max - bounds.d_min)))


def _normalized_goal(goal_cost_weight: float, goal_norm_max: float) -> float:
    return min(1.0, max(0.0, math.log1p(goal_cost_weight) / math.log1p(goal_norm_max)))


def _validate_action_retentions(values: Sequence[float]) -> None:
    if not values:
        raise ValueError("action_retentions must not be empty.")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("action_retentions must satisfy 0 < value < 1.")


def _resolve_path(value: str, *, base_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return int(value)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    values = tuple(
        _float(item, f"{field_name}[{idx}]") for idx, item in enumerate(value)
    )
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    return values


def _optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _float_tuple(value, field_name)


__all__ = [
    "ACTION_SPACE",
    "FEATURE_VERSION",
    "FSRS6OracleStationaryFiniteDistillBatchPolicy",
    "FSRS6OracleStationaryFiniteDistillPolicy",
    "POLICY_TYPE",
    "PORTFOLIO_CHILD_ACTION_SPACE",
    "load_checkpoint_metadata",
    "load_checkpoint_model",
]
