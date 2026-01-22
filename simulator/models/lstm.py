from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from simulator.core import Card, MemoryModel

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

EPS = 1e-7
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_benchmark_root() -> Path:
    return _REPO_ROOT.parent / "srs-benchmark"


def _resolve_benchmark_weights(user_id: int, benchmark_root: Path) -> Path | None:
    weights_dir = benchmark_root / "weights"
    if not weights_dir.exists():
        return None
    preferred_dir = weights_dir / "LSTM"
    candidate = preferred_dir / f"{user_id}.pth"
    if candidate.exists():
        return candidate
    for sub_dir in sorted(weights_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        if not sub_dir.name.startswith("LSTM"):
            continue
        fallback_candidate = sub_dir / f"{user_id}.pth"
        if fallback_candidate.exists():
            return fallback_candidate
    return None


def _resolve_weight_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No .pth files under {path}")
        return candidates[-1]
    raise FileNotFoundError(f"LSTM weights not found at {path}")


def _normalize_state_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        if obj and all(isinstance(key, str) for key in obj.keys()):
            return obj
        for key in ("state_dict", "model", "model_state_dict"):
            sub = obj.get(key)
            if isinstance(sub, Mapping):
                return sub
        if len(obj) == 1:
            value = next(iter(obj.values()))
            if isinstance(value, Mapping):
                return value
    raise ValueError("Unsupported weight checkpoint format.")


class _ResBlock(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: Tensor) -> Tensor:
        return self.module(inputs) + inputs


class _RNNWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, _ = self.module(inputs)
        return outputs


def _as_buffer(
    value: Tensor | Sequence[float] | float | None,
    length: int,
    default: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if value is None:
        tensor = torch.full((length,), float(default), dtype=dtype, device=device)
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim == 0:
            tensor = tensor.repeat(length)
        if tensor.shape[-1] != length:
            raise ValueError(
                f"Expected tensor with last dimension {length}, got {tensor.shape}."
            )
    return tensor


class _SequenceLSTM(nn.Module):
    """
    Torch module adapted from the `srs-benchmark` LSTM implementation.
    """

    def __init__(
        self,
        *,
        use_duration_feature: bool,
        input_mean: Tensor | Sequence[float] | float | None,
        input_std: Tensor | Sequence[float] | float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.use_duration_feature = use_duration_feature
        self.dtype = dtype
        self.device = device
        self.forward_calls = 0
        num_main_inputs = 1 + (1 if use_duration_feature else 0)
        self.n_main_inputs = num_main_inputs
        self.n_input = num_main_inputs + 4  # rating expands to 4 dims
        self.n_hidden = 20
        self.n_curves = 3

        self.register_buffer(
            "input_mean",
            _as_buffer(input_mean, num_main_inputs, 0.0, device, dtype),
        )
        self.register_buffer(
            "input_std",
            _as_buffer(input_std, num_main_inputs, 1.0, device, dtype),
        )

        self.process = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.SiLU(),
            nn.LayerNorm(self.n_hidden, bias=False),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SiLU(),
            _ResBlock(
                nn.Sequential(
                    nn.LayerNorm(self.n_hidden, bias=False),
                    _RNNWrapper(
                        nn.LSTM(
                            input_size=self.n_hidden,
                            hidden_size=self.n_hidden,
                            num_layers=1,
                        )
                    ),
                )
            ),
            _ResBlock(
                nn.Sequential(
                    nn.LayerNorm(self.n_hidden),
                    _RNNWrapper(
                        nn.LSTM(
                            input_size=self.n_hidden,
                            hidden_size=self.n_hidden,
                            num_layers=1,
                        )
                    ),
                )
            ),
            _ResBlock(
                nn.Sequential(
                    nn.LayerNorm(self.n_hidden, bias=False),
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.SiLU(),
                    nn.LayerNorm(self.n_hidden, bias=False),
                    nn.Linear(self.n_hidden, self.n_hidden),
                    nn.SiLU(),
                )
            ),
            nn.LayerNorm(self.n_hidden, bias=False),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.SiLU(),
        )

        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias_ih" in name:
                start_index = len(param.data) // 4
                end_index = len(param.data) // 2
                param.data[start_index:end_index].fill_(1.0)

        self.w_fc = nn.Linear(self.n_hidden, self.n_curves)
        self.s_fc = nn.Linear(self.n_hidden, self.n_curves)
        self.d_fc = nn.Linear(self.n_hidden, self.n_curves)

    def _build_features(self, x_lni: Tensor) -> Tensor:
        x_rating = x_lni[..., -1:]
        x_features = x_lni[..., :-1]

        x_delay = torch.log(EPS + x_features[..., :1])
        if self.use_duration_feature:
            duration = torch.clamp(x_features[..., 1:2], min=100.0, max=60000.0)
            x_duration = torch.log(duration)
            x_main = torch.cat([x_delay, x_duration], dim=-1)
        else:
            x_main = x_delay
        x_main = (x_main - self.input_mean) / self.input_std

        x_rating = torch.maximum(x_rating, torch.ones_like(x_rating))
        x_rating = torch.nn.functional.one_hot(
            x_rating.squeeze(-1).long() - 1, num_classes=4
        ).float()
        return torch.cat([x_main, x_rating], dim=-1)

    def _pack_like(self, packed: PackedSequence, data: Tensor) -> PackedSequence:
        return PackedSequence(
            data,
            packed.batch_sizes,
            packed.sorted_indices,
            packed.unsorted_indices,
        )

    def _apply_packed(
        self, module: nn.Module, packed: PackedSequence
    ) -> PackedSequence:
        if isinstance(module, nn.Sequential):
            out = packed
            for sub in module:
                out = self._apply_packed(sub, out)
            return out
        if isinstance(module, _ResBlock):
            inner = self._apply_packed(module.module, packed)
            return self._pack_like(packed, packed.data + inner.data)
        if isinstance(module, _RNNWrapper):
            return module(packed)
        return self._pack_like(packed, module(packed.data))

    def forward(self, x_lni: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.forward_calls += 1
        x = self._build_features(x_lni)
        x_lnh = self.process(x)

        w_lnh = torch.nn.functional.softmax(self.w_fc(x_lnh), dim=-1)
        s_lnh = torch.exp(torch.clamp(self.s_fc(x_lnh), min=-25, max=25))
        d_lnh = torch.exp(torch.clamp(self.d_fc(x_lnh), min=-25, max=25))
        return w_lnh, s_lnh, d_lnh

    def forward_packed_last(
        self, x_lni: PackedSequence, lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.forward_calls += 1
        x = self._build_features(x_lni.data)
        packed = self._pack_like(x_lni, x)
        packed_hidden = self._apply_packed(self.process, packed)
        hidden_padded, _ = pad_packed_sequence(packed_hidden)
        lengths_device = lengths.to(hidden_padded.device)
        idx = torch.clamp(lengths_device - 1, min=0)
        batch_idx = torch.arange(lengths_device.shape[0], device=hidden_padded.device)
        hidden_last = hidden_padded[idx, batch_idx]
        w_last = torch.nn.functional.softmax(self.w_fc(hidden_last), dim=-1)
        s_last = torch.exp(torch.clamp(self.s_fc(hidden_last), min=-25, max=25))
        d_last = torch.exp(torch.clamp(self.d_fc(hidden_last), min=-25, max=25))
        return w_last, s_last, d_last


class LSTMModel(MemoryModel):
    """
    Neural memory model that reuses the LSTM architecture from srs-benchmark.

    The model consumes review histories, predicts mixture-of-power-law
    forgetting curves (weights, stabilities, decays), and exposes them via
    the MemoryModel interface.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        user_id: int | None = None,
        benchmark_root: str | Path | None = None,
        use_duration_feature: bool = False,
        default_duration_ms: float = 2500.0,
        interval_scale: float = 1.0,
        max_events: int = 64,
        input_mean: Tensor | Sequence[float] | float | None = None,
        input_std: Tensor | Sequence[float] | float | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if device is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(default_device)
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.use_duration_feature = use_duration_feature
        self.default_duration_ms = float(default_duration_ms)
        self.interval_scale = float(interval_scale)
        self.max_events = max_events
        self.default_retention = 0.85

        self.network = _SequenceLSTM(
            use_duration_feature=use_duration_feature,
            input_mean=input_mean,
            input_std=input_std,
            device=self.device,
            dtype=self.dtype,
        ).to(self.device)
        self.network.eval()

        resolved_path: Path | None = None
        if weights_path is not None:
            resolved_path = Path(weights_path)
        elif user_id is not None:
            root = (
                Path(benchmark_root)
                if benchmark_root is not None
                else _default_benchmark_root()
            )
            resolved_path = _resolve_benchmark_weights(int(user_id), root)
            if resolved_path is None:
                weights_dir = root / "weights"
                raise FileNotFoundError(
                    f"LSTM weights for user {user_id} not found under {weights_dir}"
                )
        else:
            raise ValueError(
                "LSTM weights require --user-id or an explicit weights_path."
            )
        if resolved_path is not None:
            self._load_weights(resolved_path)

    def _load_weights(self, weights_path: str | Path) -> None:
        path = _resolve_weight_file(Path(weights_path))
        load_kwargs: dict[str, Any] = {"map_location": self.device}
        try:
            state_dict = torch.load(path, weights_only=False, **load_kwargs)
        except TypeError:
            state_dict = torch.load(path, **load_kwargs)
        cleaned = _normalize_state_dict(state_dict)
        for key in ("input_mean", "input_std"):
            tensor = cleaned.get(key)
            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 0:
                    cleaned[key] = tensor.repeat(self.network.n_main_inputs)
                elif tensor.shape[-1] != self.network.n_main_inputs:
                    raise ValueError(
                        f"{key} expects length {self.network.n_main_inputs}, "
                        f"got {tuple(tensor.shape)}"
                    )
        self.network.load_state_dict(cleaned)  # type: ignore[arg-type]

    def init_card(self, card: Card, rating: int) -> None:
        card.memory_state = {"events": []}
        self._append_event(card, float(0.0), rating)

    def reset_forward_calls(self) -> None:
        self.network.forward_calls = 0

    @property
    def forward_calls(self) -> int:
        return int(getattr(self.network, "forward_calls", 0))

    def predict_retention(self, card: Card, elapsed: float) -> float:
        state = self._ensure_state(card)
        curves = state.get("curves")
        if not curves:
            return self.default_retention
        elapsed_scaled = max(0.0, float(elapsed)) * self.interval_scale
        return self._forgetting_curve(elapsed_scaled, curves)

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        if card.memory_state is None:
            card.memory_state = {"events": []}
        self._append_event(card, float(max(elapsed, 0.0)), rating)

    def _append_event(self, card: Card, elapsed: float, rating: int) -> None:
        state = self._ensure_state(card)
        events: list[tuple[float, int]] = state.setdefault("events", [])
        events.append((elapsed, int(rating)))
        if len(events) > self.max_events:
            del events[: len(events) - self.max_events]
        state["curves"] = self._compute_curves(events)

    def _ensure_state(self, card: Card) -> dict[str, Any]:
        if not isinstance(card.memory_state, dict):
            card.memory_state = {"events": []}
        return card.memory_state

    def _build_sequence(self, events: Iterable[tuple[float, int]]) -> Tensor | None:
        rows: list[list[float]] = []
        for elapsed, rating in events:
            delay = max(elapsed, 0.0) * self.interval_scale
            features = [delay]
            if self.use_duration_feature:
                features.append(self.default_duration_ms)
            features.append(float(max(1, min(4, rating))))
            rows.append(features)
        if not rows:
            return None
        tensor = torch.as_tensor(rows, dtype=self.dtype, device=self.device)
        tensor = tensor.unsqueeze(1)  # sequence, batch, features
        return tensor

    def _compute_curves(
        self, events: Sequence[tuple[float, int]]
    ) -> dict[str, list[float]]:
        sequence = self._build_sequence(events)
        if sequence is None:
            return {}
        with torch.no_grad():
            w_lnh, s_lnh, d_lnh = self.network(sequence)
        idx = sequence.size(0) - 1
        w = w_lnh[idx, 0].detach().cpu().tolist()
        s = s_lnh[idx, 0].detach().cpu().tolist()
        d = d_lnh[idx, 0].detach().cpu().tolist()
        return {"w": w, "s": s, "d": d}

    def _forgetting_curve(
        self, elapsed_scaled: float, curves: dict[str, list[float]]
    ) -> float:
        weights = curves.get("w") or []
        stabilities = curves.get("s") or []
        decays = curves.get("d") or []
        if not (weights and stabilities and decays):
            return self.default_retention
        total = 0.0
        for w, s, d in zip(weights, stabilities, decays):
            denom = s + EPS
            total += w * (1.0 + elapsed_scaled / denom) ** (-d)
        return (1.0 - EPS) * max(0.0, min(1.0, total))


@dataclass
class LSTMVectorizedEnvState:
    event_delays: Tensor
    event_ratings: Tensor
    event_counts: Tensor
    mem_w: Tensor
    mem_s: Tensor
    mem_d: Tensor
    has_curves: Tensor


class LSTMVectorizedEnvOps:
    def __init__(
        self,
        environment: LSTMModel,
        *,
        device: torch.device | None,
        lstm_batch_size: int,
    ) -> None:
        if device is None:
            self.device = environment.device
        else:
            self.device = torch.device(device)
            if self.device != environment.device:
                environment.network.to(self.device)
                environment.device = self.device

        env_dtype = next(environment.network.parameters()).dtype
        environment.dtype = env_dtype
        self.dtype = env_dtype
        self.environment = environment
        self.batch_size = max(1, int(lstm_batch_size))
        self.max_events = int(environment.max_events)
        self.n_curves = int(environment.network.n_curves)
        self.interval_scale = float(environment.interval_scale)
        self.use_duration_feature = environment.use_duration_feature
        self.default_retention = float(environment.default_retention)
        self.duration_value = None
        if self.use_duration_feature:
            self.duration_value = torch.tensor(
                environment.default_duration_ms, device=self.device, dtype=self.dtype
            )

    def init_state(self, deck_size: int) -> LSTMVectorizedEnvState:
        event_delays = torch.zeros(
            (deck_size, self.max_events), dtype=self.dtype, device=self.device
        )
        event_ratings = torch.zeros(
            (deck_size, self.max_events), dtype=torch.int64, device=self.device
        )
        event_counts = torch.zeros(deck_size, dtype=torch.int64, device=self.device)
        mem_w = torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        mem_s = torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        mem_d = torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        has_curves = torch.zeros(deck_size, dtype=torch.bool, device=self.device)
        return LSTMVectorizedEnvState(
            event_delays=event_delays,
            event_ratings=event_ratings,
            event_counts=event_counts,
            mem_w=mem_w,
            mem_s=mem_s,
            mem_d=mem_d,
            has_curves=has_curves,
        )

    def retrievability(
        self, state: LSTMVectorizedEnvState, idx: Tensor, elapsed: Tensor
    ) -> Tensor:
        elapsed_scaled = torch.clamp(elapsed, min=0.0) * self.interval_scale
        memorized = torch.full(
            elapsed_scaled.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        curves_mask = state.has_curves[idx]
        if curves_mask.any():
            curves_idx = idx[curves_mask]
            memorized[curves_mask] = self._lstm_retention(
                elapsed_scaled[curves_mask],
                state.mem_w[curves_idx],
                state.mem_s[curves_idx],
                state.mem_d[curves_idx],
            )
        return memorized

    def update_review(
        self,
        state: LSTMVectorizedEnvState,
        idx: Tensor,
        elapsed: Tensor,
        rating: Tensor,
        retrievability: Tensor,
    ) -> None:
        if idx.numel() == 0:
            return
        self._update_curves(state, idx, elapsed, rating)

    def update_learn(
        self, state: LSTMVectorizedEnvState, idx: Tensor, rating: Tensor
    ) -> None:
        if idx.numel() == 0:
            return
        elapsed = torch.zeros(idx.shape[0], device=self.device, dtype=self.dtype)
        self._update_curves(state, idx, elapsed, rating)

    def _update_curves(
        self,
        state: LSTMVectorizedEnvState,
        idx: Tensor,
        delays: Tensor,
        ratings: Tensor,
    ) -> None:
        counts = state.event_counts[idx]
        full_mask = counts >= self.max_events
        if full_mask.any():
            full_idx = idx[full_mask]
            state.event_delays[full_idx, :-1] = state.event_delays[full_idx, 1:]
            state.event_ratings[full_idx, :-1] = state.event_ratings[full_idx, 1:]
            state.event_delays[full_idx, -1] = delays[full_mask]
            state.event_ratings[full_idx, -1] = ratings[full_mask]
            state.event_counts[full_idx] = self.max_events
        partial_mask = ~full_mask
        if partial_mask.any():
            part_idx = idx[partial_mask]
            positions = counts[partial_mask]
            state.event_delays[part_idx, positions] = delays[partial_mask]
            state.event_ratings[part_idx, positions] = ratings[partial_mask]
            state.event_counts[part_idx] = positions + 1

        lengths = state.event_counts[idx]
        if lengths.numel() == 0:
            return
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return
        order = torch.argsort(lengths, descending=True)
        idx_sorted = idx[order]
        lengths_sorted = lengths[order]

        for start in range(0, idx_sorted.numel(), self.batch_size):
            end = min(start + self.batch_size, idx_sorted.numel())
            batch_idx = idx_sorted[start:end]
            batch_lengths = lengths_sorted[start:end]
            if batch_lengths.numel() == 0:
                continue
            batch_max = int(batch_lengths.max().item())
            if batch_max <= 0:
                continue
            delays_group = state.event_delays[batch_idx, :batch_max]
            ratings_group = state.event_ratings[batch_idx, :batch_max]
            delay_scaled = torch.clamp(delays_group, min=0.0) * self.interval_scale
            seq_delay = delay_scaled.transpose(0, 1).unsqueeze(-1)
            rating_seq = ratings_group.transpose(0, 1).unsqueeze(-1).to(self.dtype)
            if self.use_duration_feature:
                duration_seq = self.duration_value.expand(seq_delay.shape)
                sequence = torch.cat([seq_delay, duration_seq, rating_seq], dim=-1)
            else:
                sequence = torch.cat([seq_delay, rating_seq], dim=-1)
            packed = pack_padded_sequence(
                sequence, batch_lengths.to("cpu"), enforce_sorted=True
            )
            w_last, s_last, d_last = self.environment.network.forward_packed_last(
                packed, batch_lengths
            )
            state.mem_w[batch_idx] = w_last
            state.mem_s[batch_idx] = s_last
            state.mem_d[batch_idx] = d_last
            state.has_curves[batch_idx] = True

    @staticmethod
    def _lstm_retention(
        elapsed_scaled: Tensor, weights: Tensor, stabilities: Tensor, decays: Tensor
    ) -> Tensor:
        denom = stabilities + EPS
        total = torch.sum(
            weights * torch.pow(1.0 + elapsed_scaled.unsqueeze(-1) / denom, -decays),
            dim=-1,
        )
        total = torch.clamp(total, min=0.0, max=1.0)
        return (1.0 - EPS) * total


__all__ = ["LSTMModel"]
