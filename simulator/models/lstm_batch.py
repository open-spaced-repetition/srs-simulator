from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import os
import sys

import torch
from torch import Tensor

from simulator.models.lstm import EPS, _normalize_state_dict, _resolve_weight_file

_LN_EPS = 1e-5


def _env_bool(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_max_batch_size(value: int | None) -> int | None:
    if value is not None:
        if value < 1:
            raise ValueError("max_batch_size must be >= 1 when set.")
        return value
    raw = os.getenv("SRS_LSTM_MAX_BATCH", "").strip().lower()
    if not raw or raw in {"0", "none", "off"}:
        return None
    parsed = int(raw)
    if parsed < 1:
        raise ValueError("SRS_LSTM_MAX_BATCH must be >= 1 when set.")
    return parsed


@dataclass
class PackedLSTMWeights:
    use_duration_feature: bool
    n_users: int
    n_hidden: int
    n_curves: int
    n_main_inputs: int
    n_input: int
    n_rnns: int
    input_mean: Tensor
    input_std: Tensor
    process_0_weight: Tensor
    process_0_bias: Tensor
    process_2_weight: Tensor
    process_3_weight: Tensor
    process_3_bias: Tensor
    process_5_ln_weight: Tensor
    process_5_lstm_w_ih: Tensor
    process_5_lstm_w_hh: Tensor
    process_5_lstm_b_ih: Tensor
    process_5_lstm_b_hh: Tensor
    process_6_ln_weight: Tensor
    process_6_ln_bias: Tensor
    process_6_lstm_w_ih: Tensor
    process_6_lstm_w_hh: Tensor
    process_6_lstm_b_ih: Tensor
    process_6_lstm_b_hh: Tensor
    process_7_ln1_weight: Tensor
    process_7_fc1_weight: Tensor
    process_7_fc1_bias: Tensor
    process_7_ln2_weight: Tensor
    process_7_fc2_weight: Tensor
    process_7_fc2_bias: Tensor
    process_8_weight: Tensor
    process_9_weight: Tensor
    process_9_bias: Tensor
    w_fc_weight: Tensor
    w_fc_bias: Tensor
    s_fc_weight: Tensor
    s_fc_bias: Tensor
    d_fc_weight: Tensor
    d_fc_bias: Tensor

    @classmethod
    def from_paths(
        cls,
        weight_paths: Sequence[str | Path],
        *,
        use_duration_feature: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PackedLSTMWeights":
        state_dicts = [_load_state_dict(path, device=device) for path in weight_paths]
        if not state_dicts:
            raise ValueError("Expected at least one LSTM weight file.")

        sample = state_dicts[0]
        n_hidden = int(sample["process.0.weight"].shape[0])
        n_input = int(sample["process.0.weight"].shape[1])
        n_main_inputs = n_input - 4
        if n_main_inputs not in {1, 2}:
            raise ValueError(f"Unexpected LSTM input width {n_input}; expected 5 or 6.")
        if use_duration_feature and n_main_inputs != 2:
            raise ValueError("use_duration_feature=True expects 2 main inputs.")
        if not use_duration_feature and n_main_inputs != 1:
            raise ValueError("use_duration_feature=False expects 1 main input.")

        stacked = _stack_tensors(state_dicts, device=device, dtype=dtype)
        input_mean = _coerce_input_stat(
            stacked.pop("input_mean"), n_main_inputs, device=device, dtype=dtype
        )
        input_std = _coerce_input_stat(
            stacked.pop("input_std"), n_main_inputs, device=device, dtype=dtype
        )
        n_users = input_mean.shape[0]

        return cls(
            use_duration_feature=use_duration_feature,
            n_users=n_users,
            n_hidden=n_hidden,
            n_curves=3,
            n_main_inputs=n_main_inputs,
            n_input=n_input,
            n_rnns=2,
            input_mean=input_mean,
            input_std=input_std,
            process_0_weight=stacked["process.0.weight"],
            process_0_bias=stacked["process.0.bias"],
            process_2_weight=stacked["process.2.weight"],
            process_3_weight=stacked["process.3.weight"],
            process_3_bias=stacked["process.3.bias"],
            process_5_ln_weight=stacked["process.5.module.0.weight"],
            process_5_lstm_w_ih=stacked["process.5.module.1.module.weight_ih_l0"],
            process_5_lstm_w_hh=stacked["process.5.module.1.module.weight_hh_l0"],
            process_5_lstm_b_ih=stacked["process.5.module.1.module.bias_ih_l0"],
            process_5_lstm_b_hh=stacked["process.5.module.1.module.bias_hh_l0"],
            process_6_ln_weight=stacked["process.6.module.0.weight"],
            process_6_ln_bias=stacked["process.6.module.0.bias"],
            process_6_lstm_w_ih=stacked["process.6.module.1.module.weight_ih_l0"],
            process_6_lstm_w_hh=stacked["process.6.module.1.module.weight_hh_l0"],
            process_6_lstm_b_ih=stacked["process.6.module.1.module.bias_ih_l0"],
            process_6_lstm_b_hh=stacked["process.6.module.1.module.bias_hh_l0"],
            process_7_ln1_weight=stacked["process.7.module.0.weight"],
            process_7_fc1_weight=stacked["process.7.module.1.weight"],
            process_7_fc1_bias=stacked["process.7.module.1.bias"],
            process_7_ln2_weight=stacked["process.7.module.3.weight"],
            process_7_fc2_weight=stacked["process.7.module.4.weight"],
            process_7_fc2_bias=stacked["process.7.module.4.bias"],
            process_8_weight=stacked["process.8.weight"],
            process_9_weight=stacked["process.9.weight"],
            process_9_bias=stacked["process.9.bias"],
            w_fc_weight=stacked["w_fc.weight"],
            w_fc_bias=stacked["w_fc.bias"],
            s_fc_weight=stacked["s_fc.weight"],
            s_fc_bias=stacked["s_fc.bias"],
            d_fc_weight=stacked["d_fc.weight"],
            d_fc_bias=stacked["d_fc.bias"],
        )


def _load_state_dict(path: str | Path, *, device: torch.device) -> Mapping[str, Tensor]:
    path = _resolve_weight_file(Path(path))
    load_kwargs = {"map_location": device}
    try:
        raw = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        raw = torch.load(path, **load_kwargs)
    return _normalize_state_dict(raw)


def _stack_tensors(
    state_dicts: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    stacked: dict[str, Tensor] = {}
    for state in state_dicts:
        for key, value in state.items():
            if not isinstance(value, Tensor):
                continue
            stacked.setdefault(key, []).append(value.to(device=device, dtype=dtype))
    output: dict[str, Tensor] = {}
    for key, values in stacked.items():
        output[key] = torch.stack(values, dim=0)
    return output


def _coerce_input_stat(
    value: Tensor, length: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if value.ndim == 0:
        value = value.repeat(length)
    if value.ndim == 1:
        if value.shape[0] == length:
            value = value.unsqueeze(0)
        elif length == 1:
            value = value.unsqueeze(-1)
        else:
            raise ValueError(f"Expected input stat length {length}, got {value.shape}.")
    if value.shape[-1] != length:
        raise ValueError(f"Expected input stat length {length}, got {value.shape}.")
    return value.to(device=device, dtype=dtype)


def _select(param: Tensor, user_idx: Tensor) -> Tensor:
    return param.index_select(0, user_idx)


def _linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    out = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
    if bias is not None:
        out = out + bias
    return out


def _layer_norm(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + _LN_EPS)
    x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return x_hat


def _lstm_step(
    x: Tensor,
    h: Tensor,
    c: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
) -> tuple[Tensor, Tensor]:
    gates = _linear(x, w_ih, b_ih) + _linear(h, w_hh, b_hh)
    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
    i_gate = torch.sigmoid(i_gate)
    f_gate = torch.sigmoid(f_gate)
    g_gate = torch.tanh(g_gate)
    o_gate = torch.sigmoid(o_gate)
    c_new = f_gate * c + i_gate * g_gate
    h_new = o_gate * torch.tanh(c_new)
    return h_new, c_new


def forward_step(
    weights: PackedLSTMWeights,
    step: Tensor,
    state: tuple[Tensor, Tensor],
    user_idx: Tensor,
) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
    h, c = state
    if h.shape[0] != weights.n_rnns:
        raise ValueError(f"Expected {weights.n_rnns} LSTM states, got {h.shape[0]}.")
    input_mean = _select(weights.input_mean, user_idx)
    input_std = _select(weights.input_std, user_idx)

    x_rating = step[:, -1:]
    x_features = step[:, :-1]
    x_delay = torch.log(EPS + x_features[:, :1])
    if weights.use_duration_feature:
        duration = torch.clamp(x_features[:, 1:2], min=100.0, max=60000.0)
        x_duration = torch.log(duration)
        x_main = torch.cat([x_delay, x_duration], dim=-1)
    else:
        x_main = x_delay
    x_main = (x_main - input_mean) / input_std
    x_rating = torch.maximum(x_rating, torch.ones_like(x_rating))
    x_rating = torch.nn.functional.one_hot(
        x_rating.squeeze(-1).long() - 1, num_classes=4
    ).float()
    x = torch.cat([x_main, x_rating], dim=-1)

    x = _linear(
        x,
        _select(weights.process_0_weight, user_idx),
        _select(weights.process_0_bias, user_idx),
    )
    x = torch.nn.functional.silu(x)
    x = _layer_norm(x, _select(weights.process_2_weight, user_idx), None)
    x = _linear(
        x,
        _select(weights.process_3_weight, user_idx),
        _select(weights.process_3_bias, user_idx),
    )
    x = torch.nn.functional.silu(x)

    residual = x
    ln = _layer_norm(x, _select(weights.process_5_ln_weight, user_idx), None)
    h0, c0 = _lstm_step(
        ln,
        h[0],
        c[0],
        _select(weights.process_5_lstm_w_ih, user_idx),
        _select(weights.process_5_lstm_w_hh, user_idx),
        _select(weights.process_5_lstm_b_ih, user_idx),
        _select(weights.process_5_lstm_b_hh, user_idx),
    )
    x = h0 + residual

    residual = x
    ln = _layer_norm(
        x,
        _select(weights.process_6_ln_weight, user_idx),
        _select(weights.process_6_ln_bias, user_idx),
    )
    h1, c1 = _lstm_step(
        ln,
        h[1],
        c[1],
        _select(weights.process_6_lstm_w_ih, user_idx),
        _select(weights.process_6_lstm_w_hh, user_idx),
        _select(weights.process_6_lstm_b_ih, user_idx),
        _select(weights.process_6_lstm_b_hh, user_idx),
    )
    x = h1 + residual

    residual = x
    x = _layer_norm(x, _select(weights.process_7_ln1_weight, user_idx), None)
    x = _linear(
        x,
        _select(weights.process_7_fc1_weight, user_idx),
        _select(weights.process_7_fc1_bias, user_idx),
    )
    x = torch.nn.functional.silu(x)
    x = _layer_norm(x, _select(weights.process_7_ln2_weight, user_idx), None)
    x = _linear(
        x,
        _select(weights.process_7_fc2_weight, user_idx),
        _select(weights.process_7_fc2_bias, user_idx),
    )
    x = torch.nn.functional.silu(x)
    x = x + residual

    x = _layer_norm(x, _select(weights.process_8_weight, user_idx), None)
    x = _linear(
        x,
        _select(weights.process_9_weight, user_idx),
        _select(weights.process_9_bias, user_idx),
    )
    x = torch.nn.functional.silu(x)

    w_last = torch.nn.functional.softmax(
        _linear(
            x,
            _select(weights.w_fc_weight, user_idx),
            _select(weights.w_fc_bias, user_idx),
        ),
        dim=-1,
    )
    s_last = torch.exp(
        torch.clamp(
            _linear(
                x,
                _select(weights.s_fc_weight, user_idx),
                _select(weights.s_fc_bias, user_idx),
            ),
            min=-25,
            max=25,
        )
    )
    d_last = torch.exp(
        torch.clamp(
            _linear(
                x,
                _select(weights.d_fc_weight, user_idx),
                _select(weights.d_fc_bias, user_idx),
            ),
            min=-25,
            max=25,
        )
    )
    new_h = torch.stack([h0, h1], dim=0)
    new_c = torch.stack([c0, c1], dim=0)
    return w_last, s_last, d_last, (new_h, new_c)


@dataclass
class LSTMBatchedEnvState:
    lstm_h: Tensor
    lstm_c: Tensor
    mem_w: Tensor
    mem_s: Tensor
    mem_d: Tensor
    has_curves: Tensor


class LSTMBatchedEnvOps:
    def __init__(
        self,
        weights: PackedLSTMWeights,
        *,
        device: torch.device,
        dtype: torch.dtype,
        default_duration_ms: float = 2500.0,
        max_batch_size: int | None = None,
    ) -> None:
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.default_retention = 0.85
        self._mem_log = _env_bool("SRS_LSTM_MEM_LOG")
        self._mem_log_sync = _env_bool("SRS_LSTM_MEM_LOG_SYNC")
        self._mem_log_peak_alloc = 0
        self._mem_log_peak_reserved = 0
        self.max_batch_size = _resolve_max_batch_size(max_batch_size)
        self.use_duration_feature = weights.use_duration_feature
        self.n_hidden = weights.n_hidden
        self.n_curves = weights.n_curves
        self.n_rnns = weights.n_rnns
        self.duration_value = None
        if self.use_duration_feature:
            self.duration_value = torch.tensor(
                default_duration_ms, device=device, dtype=dtype
            )

    def _maybe_log_mem(self, batch_size: int, *, label: str) -> None:
        if not self._mem_log or self.device.type != "cuda":
            return
        if self._mem_log_sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        if self._mem_log_sync:
            torch.cuda.synchronize(self.device)

    def _maybe_log_mem_post(self, batch_size: int, *, label: str) -> None:
        if not self._mem_log or self.device.type != "cuda":
            return
        if self._mem_log_sync:
            torch.cuda.synchronize(self.device)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        if peak_alloc > self._mem_log_peak_alloc:
            self._mem_log_peak_alloc = peak_alloc
            self._mem_log_peak_reserved = max(
                self._mem_log_peak_reserved, peak_reserved
            )
            sys.stderr.write(
                "[lstm mem] "
                f"{label} batch={batch_size} "
                f"peak_alloc={peak_alloc / 1024**2:.1f}MiB "
                f"peak_reserved={peak_reserved / 1024**2:.1f}MiB\n"
            )

    def init_state(self, user_count: int, deck_size: int) -> LSTMBatchedEnvState:
        lstm_h = torch.zeros(
            (self.n_rnns, user_count, deck_size, self.n_hidden),
            dtype=self.dtype,
            device=self.device,
        )
        lstm_c = torch.zeros_like(lstm_h)
        mem_w = torch.zeros(
            (user_count, deck_size, self.n_curves),
            dtype=self.dtype,
            device=self.device,
        )
        mem_s = torch.zeros_like(mem_w)
        mem_d = torch.zeros_like(mem_w)
        has_curves = torch.zeros(
            (user_count, deck_size), dtype=torch.bool, device=self.device
        )
        return LSTMBatchedEnvState(
            lstm_h=lstm_h,
            lstm_c=lstm_c,
            mem_w=mem_w,
            mem_s=mem_s,
            mem_d=mem_d,
            has_curves=has_curves,
        )

    def retrievability(self, state: LSTMBatchedEnvState, elapsed: Tensor) -> Tensor:
        elapsed_clamped = torch.clamp(elapsed, min=0.0)
        memorized = torch.full(
            elapsed_clamped.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        curves_mask = state.has_curves
        if curves_mask.any():
            memorized[curves_mask] = self._lstm_retention(
                elapsed_clamped[curves_mask],
                state.mem_w[curves_mask],
                state.mem_s[curves_mask],
                state.mem_d[curves_mask],
            )
        return memorized

    def retrievability_entries(
        self,
        state: LSTMBatchedEnvState,
        user_idx: Tensor,
        card_idx: Tensor,
        elapsed: Tensor,
    ) -> Tensor:
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        elapsed_clamped = torch.clamp(elapsed, min=0.0)
        curves_mask = state.has_curves[user_idx, card_idx]
        memorized = torch.full(
            elapsed_clamped.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        if curves_mask.any():
            idx = curves_mask.nonzero(as_tuple=False).squeeze(1)
            u_sel = user_idx[idx]
            c_sel = card_idx[idx]
            memorized[idx] = self._lstm_retention(
                elapsed_clamped[idx],
                state.mem_w[u_sel, c_sel],
                state.mem_s[u_sel, c_sel],
                state.mem_d[u_sel, c_sel],
            )
        return memorized

    def update_review(
        self,
        state: LSTMBatchedEnvState,
        user_idx: Tensor,
        card_idx: Tensor,
        elapsed: Tensor,
        rating: Tensor,
    ) -> None:
        self._update_curves(state, user_idx, card_idx, elapsed, rating)

    def update_learn(
        self,
        state: LSTMBatchedEnvState,
        user_idx: Tensor,
        card_idx: Tensor,
        rating: Tensor,
    ) -> None:
        elapsed = torch.zeros(rating.shape[0], device=self.device, dtype=self.dtype)
        self._update_curves(state, user_idx, card_idx, elapsed, rating)

    def _update_curves(
        self,
        state: LSTMBatchedEnvState,
        user_idx: Tensor,
        card_idx: Tensor,
        delays: Tensor,
        ratings: Tensor,
    ) -> None:
        if user_idx.numel() == 0:
            return

        def _update_chunk(
            chunk_user: Tensor,
            chunk_card: Tensor,
            chunk_delays: Tensor,
            chunk_ratings: Tensor,
        ) -> None:
            delay_clamped = torch.clamp(chunk_delays, min=0.0)
            rating_clamped = torch.clamp(chunk_ratings, min=1, max=4).to(self.dtype)
            delay_feature = delay_clamped.unsqueeze(-1)
            rating_feature = rating_clamped.unsqueeze(-1)
            if self.use_duration_feature:
                duration_feature = self.duration_value.expand_as(delay_feature)
                step = torch.cat(
                    [delay_feature, duration_feature, rating_feature], dim=-1
                )
            else:
                step = torch.cat([delay_feature, rating_feature], dim=-1)

            h = state.lstm_h[:, chunk_user, chunk_card, :]
            c = state.lstm_c[:, chunk_user, chunk_card, :]
            self._maybe_log_mem(chunk_user.numel(), label="env-batched")
            w_last, s_last, d_last, (h_new, c_new) = forward_step(
                self.weights, step, (h, c), chunk_user
            )
            self._maybe_log_mem_post(chunk_user.numel(), label="env-batched")
            state.lstm_h[:, chunk_user, chunk_card, :] = h_new
            state.lstm_c[:, chunk_user, chunk_card, :] = c_new
            state.mem_w[chunk_user, chunk_card] = w_last
            state.mem_s[chunk_user, chunk_card] = s_last
            state.mem_d[chunk_user, chunk_card] = d_last
            state.has_curves[chunk_user, chunk_card] = True

        total = int(user_idx.numel())
        if self.max_batch_size is None or total <= self.max_batch_size:
            _update_chunk(user_idx, card_idx, delays, ratings)
            return
        for start in range(0, total, self.max_batch_size):
            end = start + self.max_batch_size
            _update_chunk(
                user_idx[start:end],
                card_idx[start:end],
                delays[start:end],
                ratings[start:end],
            )

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


__all__ = ["PackedLSTMWeights", "LSTMBatchedEnvOps", "LSTMBatchedEnvState"]
