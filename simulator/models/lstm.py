from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from simulator.core import Card, MemoryModel

import torch
from torch import Tensor, nn

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
        self.n_rnns = sum(
            1 for module in self.process.modules() if isinstance(module, _RNNWrapper)
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

    def _apply_step(
        self,
        module: nn.Module,
        inputs: Tensor,
        states: list[tuple[Tensor, Tensor]],
        state_idx: int,
    ) -> tuple[Tensor, int]:
        if isinstance(module, nn.Sequential):
            out = inputs
            for sub in module:
                out, state_idx = self._apply_step(sub, out, states, state_idx)
            return out, state_idx
        if isinstance(module, _ResBlock):
            inner, state_idx = self._apply_step(
                module.module, inputs, states, state_idx
            )
            return inner + inputs, state_idx
        if isinstance(module, _RNNWrapper):
            h, c = states[state_idx]
            out_seq, (h_new, c_new) = module.module(
                inputs.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0))
            )
            states[state_idx] = (h_new.squeeze(0), c_new.squeeze(0))
            return out_seq.squeeze(0), state_idx + 1
        return module(inputs), state_idx

    def forward(self, x_lni: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.forward_calls += 1
        x = self._build_features(x_lni)
        x_lnh = self.process(x)

        w_lnh = torch.nn.functional.softmax(self.w_fc(x_lnh), dim=-1)
        s_lnh = torch.exp(torch.clamp(self.s_fc(x_lnh), min=-25, max=25))
        d_lnh = torch.exp(torch.clamp(self.d_fc(x_lnh), min=-25, max=25))
        return w_lnh, s_lnh, d_lnh

    def forward_step(
        self, x_lni: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        self.forward_calls += 1
        h, c = state
        if h.ndim == 2:
            h = h.unsqueeze(0)
        if c.ndim == 2:
            c = c.unsqueeze(0)
        if h.shape[0] != self.n_rnns:
            raise ValueError(f"Expected {self.n_rnns} LSTM states, got {h.shape[0]}.")
        states = [(h[i], c[i]) for i in range(self.n_rnns)]
        x = self._build_features(x_lni)
        x_lnh, final_idx = self._apply_step(self.process, x, states, 0)
        if final_idx != self.n_rnns:
            raise RuntimeError(
                f"Consumed {final_idx} LSTM states, expected {self.n_rnns}."
            )
        w_last = torch.nn.functional.softmax(self.w_fc(x_lnh), dim=-1)
        s_last = torch.exp(torch.clamp(self.s_fc(x_lnh), min=-25, max=25))
        d_last = torch.exp(torch.clamp(self.d_fc(x_lnh), min=-25, max=25))
        new_h = torch.stack([pair[0] for pair in states], dim=0)
        new_c = torch.stack([pair[1] for pair in states], dim=0)
        return w_last, s_last, d_last, (new_h, new_c)


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
        card.memory_state = {}
        self._update_state(card, float(0.0), rating)

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

    def predict_retention_from_curves(
        self, curves: dict[str, list[float]], elapsed: float
    ) -> float:
        if not curves:
            return self.default_retention
        elapsed_scaled = max(0.0, float(elapsed)) * self.interval_scale
        return self._forgetting_curve(elapsed_scaled, curves)

    def curves_from_events(
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

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        self._update_state(card, float(max(elapsed, 0.0)), rating)

    def _ensure_state(self, card: Card) -> dict[str, Any]:
        if not isinstance(card.memory_state, dict):
            card.memory_state = {}
        return card.memory_state

    def _init_lstm_state(self) -> tuple[Tensor, Tensor]:
        h = torch.zeros(
            (self.network.n_rnns, self.network.n_hidden),
            device=self.device,
            dtype=self.dtype,
        )
        c = torch.zeros_like(h)
        return h, c

    def _ensure_lstm_state(self, state: dict[str, Any]) -> tuple[Tensor, Tensor]:
        h = state.get("lstm_h")
        c = state.get("lstm_c")
        if not isinstance(h, torch.Tensor) or not isinstance(c, torch.Tensor):
            h, c = self._init_lstm_state()
        if h.shape != c.shape or h.ndim != 2:
            h, c = self._init_lstm_state()
        if h.shape[0] != self.network.n_rnns or h.shape[1] != self.network.n_hidden:
            h, c = self._init_lstm_state()
        if h.device != self.device or h.dtype != self.dtype:
            h = h.to(device=self.device, dtype=self.dtype)
            c = c.to(device=self.device, dtype=self.dtype)
        state["lstm_h"] = h
        state["lstm_c"] = c
        return h, c

    def _build_step_tensor(self, elapsed: float, rating: int) -> Tensor:
        delay = max(float(elapsed), 0.0) * self.interval_scale
        rating_clamped = max(1, min(4, int(rating)))
        features: list[float] = [delay]
        if self.use_duration_feature:
            features.append(self.default_duration_ms)
        features.append(float(rating_clamped))
        return torch.as_tensor([features], dtype=self.dtype, device=self.device)

    def _build_sequence(self, events: Sequence[tuple[float, int]]) -> Tensor | None:
        if not events:
            return None
        rows: list[list[float]] = []
        for elapsed, rating in events:
            delay = max(float(elapsed), 0.0) * self.interval_scale
            features = [delay]
            if self.use_duration_feature:
                features.append(self.default_duration_ms)
            features.append(float(max(1, min(4, int(rating)))))
            rows.append(features)
        tensor = torch.as_tensor(rows, dtype=self.dtype, device=self.device)
        return tensor.unsqueeze(1)  # sequence, batch, features

    def _update_state(self, card: Card, elapsed: float, rating: int) -> None:
        state = self._ensure_state(card)
        h, c = self._ensure_lstm_state(state)
        step = self._build_step_tensor(elapsed, rating)
        with torch.no_grad():
            w_last, s_last, d_last, (h_new, c_new) = self.network.forward_step(
                step, (h.unsqueeze(1), c.unsqueeze(1))
            )
        state["lstm_h"] = h_new.squeeze(1).detach()
        state["lstm_c"] = c_new.squeeze(1).detach()
        state["curves"] = {
            "w": w_last[0].detach().cpu().tolist(),
            "s": s_last[0].detach().cpu().tolist(),
            "d": d_last[0].detach().cpu().tolist(),
        }

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
    lstm_h: Tensor
    lstm_c: Tensor
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
        self.n_rnns = int(environment.network.n_rnns)
        self.n_hidden = int(environment.network.n_hidden)
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
        lstm_h = torch.zeros(
            (self.n_rnns, deck_size, self.n_hidden),
            dtype=self.dtype,
            device=self.device,
        )
        lstm_c = torch.zeros_like(lstm_h)
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
            lstm_h=lstm_h,
            lstm_c=lstm_c,
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
        if idx.numel() == 0:
            return
        delay_scaled = torch.clamp(delays, min=0.0) * self.interval_scale
        rating_clamped = torch.clamp(ratings, min=1, max=4).to(self.dtype)
        delay_feature = delay_scaled.unsqueeze(-1)
        rating_feature = rating_clamped.unsqueeze(-1)
        if self.use_duration_feature:
            duration_feature = self.duration_value.expand_as(delay_feature)
            step = torch.cat([delay_feature, duration_feature, rating_feature], dim=-1)
        else:
            step = torch.cat([delay_feature, rating_feature], dim=-1)

        h = state.lstm_h[:, idx, :]
        c = state.lstm_c[:, idx, :]
        w_last, s_last, d_last, (h_new, c_new) = self.environment.network.forward_step(
            step, (h, c)
        )
        state.lstm_h[:, idx, :] = h_new
        state.lstm_c[:, idx, :] = c_new
        state.mem_w[idx] = w_last
        state.mem_s[idx] = s_last
        state.mem_d[idx] = d_last
        state.has_curves[idx] = True

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
